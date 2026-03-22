"""
retriever.py
────────────
Finance RAG retriever for the Kenya Cabinet RAG pipeline.

Pipeline:
  1. Detect fiscal year intent from query (regex)
  2. Classify query domain via Groq LLM call → list of domains
  3. Embed query (Voyage dense + BM25 sparse)
  4. Hybrid search in Qdrant (dense + BM25 via RRF fusion)
     — Split search when fiscal year filter active:
         dated docs: (TOP_K - NA_CAP) candidates, filtered by domain
         na docs:    NA_CAP candidates, NO domain filter (Finance Acts
                     are legal_fiscal but relevant to all fiscal queries)
  5. Cross-encoder rerank with priority + rag_weight boosting
  6. Return top 3 chunks with full payload (text + metadata)

Domain classification:
  A lightweight Groq call classifies each query into 1-3 domains from the
  Finance agent's domain taxonomy before hitting Qdrant. This restricts the
  dated candidate pool to documents likely to contain the answer — e.g.
  "fiscal deficit" → fiscal_policy only, not MPC reports or KRA plans.
  On any Groq failure the classifier falls back to no domain filter (safe).

Fiscal year filter logic:
  - Explicit year in query  → expand to both FY ranges that year touches + na
  - "current/latest" query  → last 3 FY strings + na cap
  - Conceptual/historical   → no fiscal year filter, full corpus search
  - fiscal_year == "na"     → always included but capped at NA_CAP slots

Priority boosting (applied after cross-encoder):
  final_score = cross_encoder_score × priority_weight × rag_weight

  Constitutional domain guard: chunks with priority="constitutional" only
  receive their 2.0 multiplier when the query is about constitutional/legal
  topics. For all other queries they are capped at 0.8.

Dependencies:
    pip install voyageai qdrant-client fastembed sentence-transformers groq

Usage (as module):
    from retriever import retrieve
    chunks = retrieve("What was Kenya's GDP growth in 2019?")

Usage (CLI test):
    python retriever.py "What was Kenya's inflation rate in 2021?"
"""

import json
import os
import re
import sys

import voyageai
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Prefetch,
    FusionQuery,
    Fusion,
    SparseVector,
)
from fastembed import SparseTextEmbedding
from sentence_transformers import CrossEncoder

# ── Config ────────────────────────────────────────────────────────────────────

COLLECTION_NAME  = "kenya_executive_roundtable"
VOYAGE_MODEL     = "voyage-3"
RERANKER_MODEL   = "BAAI/bge-reranker-v2-m3"
GROQ_MODEL       = "llama-3.3-70b-versatile"
QDRANT_URL       = "http://localhost:6333"
AGENT            = "finance"

TOP_K_CANDIDATES = 20   # total candidates fetched before reranking
TOP_K_FINAL      = 3    # final chunks returned to LLM after reranking

# Max na-document chunks in candidate pool when a fiscal year filter is active.
# Remaining (TOP_K_CANDIDATES - NA_CAP) slots go to dated fiscal documents.
NA_CAP           = 4

# Most recent FY in corpus — update when new documents are added
LATEST_FY_START  = 2024
RECENT_FY_STARTS = list(range(LATEST_FY_START - 2, LATEST_FY_START + 1))

# Keywords that signal "current/latest" intent
RECENCY_KEYWORDS = {
    "current", "latest", "recent", "now", "today",
    "this year", "most recent", "new", "updated"
}

# Keywords that indicate a constitutional/legal query — used to decide
# whether constitutional chunks deserve their full 2.0 priority boost
CONSTITUTIONAL_KEYWORDS = {
    "constitution", "constitutional", "article", "bill of rights",
    "parliament", "legislature", "judiciary", "devolution", "county",
    "supremacy", "sovereignty", "statute", "act of parliament",
    "finance act", "finance bill", "legal", "mandate", "provision",
}

# Priority weights applied as multipliers on cross-encoder scores
PRIORITY_WEIGHTS = {
    "constitutional": 2.0,
    "high":           1.5,
    "medium":         1.0,
    "low":            0.5,
}

# Constitutional chunks on non-constitutional queries get this weight instead
CONSTITUTIONAL_NON_DOMAIN_WEIGHT = 0.8

# All valid domain values stored in Qdrant payloads for the Finance agent
VALID_DOMAINS = {
    "fiscal_policy",
    "monetary_policy",
    "audit_compliance",
    "revenue_tax",
    "tax_expenditure",
    "macroeconomic_data",
    "legal_fiscal",
    "constitutional",
    "external_assessment",
    "unknown",
}

# Domain classification prompt — given to Groq at query time
_CLASSIFY_PROMPT = """You are a query classifier for a Kenya government financial documents RAG system.

Given a query, identify which document domains are most likely to contain the answer.

Available domains and what they cover:
- fiscal_policy: budget policy statements, fiscal deficit, public debt, MTEF, government spending, budget implementation reports, budget summaries, debt management strategies
- monetary_policy: CBK reports, interest rates, inflation, exchange rates, MPC decisions, financial sector stability
- audit_compliance: auditor general reports, audit findings, procurement irregularities, pending bills
- revenue_tax: KRA revenue performance, tax collection, revenue targets, customs
- tax_expenditure: tax expenditures, tax reliefs, exemptions
- macroeconomic_data: economic surveys, GDP growth, employment, sectoral output, economic outlook
- legal_fiscal: finance acts, finance bills, tax legislation
- constitutional: constitution of Kenya, constitutional provisions
- external_assessment: IMF reports, World Bank reports, external debt assessments

Rules:
- Return 1-3 domains maximum — be specific, not broad
- fiscal_policy covers: deficit, debt, borrowing, budget, spending, revenue estimates, fiscal consolidation
- When a query could span two domains, return both e.g. GDP + fiscal = ["macroeconomic_data", "fiscal_policy"]
- - Respond ONLY with valid JSON: {{"domains": ["domain1"]}}
- No explanation, no markdown, no preamble — just the JSON object

Query: {query}"""


# ── Clients (lazy-loaded singletons) ─────────────────────────────────────────

_voyage   = None
_qdrant   = None
_bm25     = None
_reranker = None
_groq     = None


def get_voyage():
    global _voyage
    if _voyage is None:
        _voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    return _voyage


def get_qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL)
    return _qdrant


def get_bm25():
    global _bm25
    if _bm25 is None:
        _bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _bm25


def get_reranker():
    global _reranker
    if _reranker is None:
        print("  Loading reranker (first run only)...")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def get_groq():
    global _groq
    if _groq is None:
        _groq = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq


# ── Domain classification ─────────────────────────────────────────────────────

def classify_query(query: str) -> list[str]:
    """Use Groq to classify the query into relevant document domains.

    Returns a list of domain strings from VALID_DOMAINS.
    On any failure (API error, bad JSON, invalid domains) returns [] which
    means no domain filter is applied — safe fallback to full corpus search.

    Examples:
        "What was Kenya's fiscal deficit in 2022?"
            → ["fiscal_policy"]
        "What is the CBK's current interest rate policy?"
            → ["monetary_policy"]
        "How much revenue did KRA collect and what was the fiscal deficit?"
            → ["revenue_tax", "fiscal_policy"]
        "What does the IMF say about Kenya's debt sustainability?"
            → ["external_assessment", "fiscal_policy"]
    """
    try:
        response = get_groq().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": _CLASSIFY_PROMPT.format(query=query),
                }
            ],
            temperature=0.0,
            max_tokens=60,
        )
        raw = response.choices[0].message.content.strip()

        # Strip any accidental markdown fences
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()

        parsed  = json.loads(raw)
        domains = parsed.get("domains", [])

        # Validate — only accept known domain values
        valid = [d for d in domains if d in VALID_DOMAINS]
        return valid if valid else []

    except Exception as e:
        # Any failure → no domain filter (safe fallback)
        return []


# ── Fiscal year helpers ───────────────────────────────────────────────────────

def year_to_fy_strings(year: str) -> list[str]:
    """Convert a calendar year string to the FY strings it could belong to.

    A calendar year Y touches two fiscal years:
      - FY starting in Y:   e.g. 2022 → "2022_23"
      - FY starting in Y-1: e.g. 2022 → "2021_22"

    Examples:
        "2022" → ["2022_23", "2021_22"]
        "2019" → ["2019_20", "2018_19"]
    """
    y = int(year)
    return [
        f"{y}_{str(y + 1)[-2:]}",
        f"{y - 1}_{str(y)[-2:]}",
    ]


def fy_start_to_string(fy_start: int) -> str:
    """Convert an FY start year integer to stored string format.

    Example: 2022 → "2022_23"
    """
    return f"{fy_start}_{str(fy_start + 1)[-2:]}"


# ── Fiscal year intent detection ──────────────────────────────────────────────

def detect_fiscal_year_intent(query: str) -> dict:
    """Analyse query for fiscal year intent.

    Returns dict with:
      mode:      "explicit" | "recent" | "none"
      years:     list of fiscal_year strings in stored format
      raw_years: list of calendar year strings found in query
    """
    query_lower = query.lower()

    year_matches = re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)', query)
    if year_matches:
        fy_strings = []
        for y in sorted(set(year_matches)):
            fy_strings.extend(year_to_fy_strings(y))
        seen      = set()
        unique_fy = []
        for fy in fy_strings:
            if fy not in seen:
                seen.add(fy)
                unique_fy.append(fy)
        return {
            "mode":      "explicit",
            "years":     unique_fy,
            "raw_years": sorted(set(year_matches)),
        }

    if any(kw in query_lower for kw in RECENCY_KEYWORDS):
        fy_strings = [fy_start_to_string(y) for y in RECENT_FY_STARTS]
        return {"mode": "recent", "years": fy_strings, "raw_years": []}

    return {"mode": "none", "years": [], "raw_years": []}


def is_constitutional_query(query: str) -> bool:
    """Return True if the query is about constitutional or legal matters."""
    q = query.lower()
    return any(kw in q for kw in CONSTITUTIONAL_KEYWORDS)


# ── Qdrant filter builder ─────────────────────────────────────────────────────

def build_filter(year_intent: dict, agent: str = AGENT) -> Filter:
    """Build base Qdrant filter from year intent and agent.

    Domain filtering is applied separately inside hybrid_search to allow
    different domain rules for dated vs na documents.
    """
    agent_condition = FieldCondition(
        key="agent",
        match=MatchValue(value=agent)
    )

    if not year_intent["years"]:
        return Filter(must=[agent_condition])

    na_condition   = FieldCondition(key="fiscal_year", match=MatchValue(value="na"))
    year_condition = FieldCondition(key="fiscal_year", match=MatchAny(any=year_intent["years"]))
    fiscal_filter  = Filter(should=[na_condition, year_condition])

    return Filter(must=[agent_condition, fiscal_filter])


# ── Filter introspection helpers ──────────────────────────────────────────────

def _filter_has_year(f: Filter) -> bool:
    """Return True if the filter contains a fiscal_year condition."""
    for c in (f.must or []):
        if isinstance(c, Filter):
            for sc in (c.should or []):
                if isinstance(sc, FieldCondition) and sc.key == "fiscal_year":
                    return True
    return False


def _extract_agent(f: Filter) -> str:
    """Extract agent value from filter must conditions."""
    for c in (f.must or []):
        if isinstance(c, FieldCondition) and c.key == "agent":
            return c.match.value
    return AGENT


def _extract_year_values(f: Filter) -> list[str]:
    """Extract non-na fiscal_year values from filter (the MatchAny list)."""
    for c in (f.must or []):
        if isinstance(c, Filter):
            for sc in (c.should or []):
                if isinstance(sc, FieldCondition) and sc.key == "fiscal_year":
                    if hasattr(sc.match, "any"):
                        return sc.match.any
    return []


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_query_dense(query: str) -> list[float]:
    result = get_voyage().embed([query], model=VOYAGE_MODEL, input_type="query")
    return result.embeddings[0]


def embed_query_sparse(query: str) -> SparseVector:
    embeddings = list(get_bm25().embed([query]))
    e = embeddings[0]
    return SparseVector(
        indices=e.indices.tolist(),
        values=e.values.tolist(),
    )


# ── Hybrid search with domain filter + na-cap split ──────────────────────────

def hybrid_search(
    query:         str,
    qdrant_filter: Filter,
    domains:       list[str],
    top_k:         int = TOP_K_CANDIDATES,
) -> list[dict]:
    """Hybrid Qdrant search: dense + BM25 sparse fused via RRF.

    When a fiscal year filter is active, runs two separate searches:

      dated search  — domain-filtered, (top_k - NA_CAP) candidates
                      Only pulls from document families relevant to the query.
                      e.g. "fiscal deficit" → fiscal_policy docs only.

      na search     — NO domain filter, NA_CAP candidates
                      Timeless documents (Finance Acts, Constitution) are not
                      domain-restricted because a Finance Act tagged legal_fiscal
                      is still directly relevant to fiscal policy queries.

    When no fiscal year filter is active, runs a single unified search with
    domain filter applied if domains were classified.

    Returns combined list of payload dicts.
    """
    dense_vec  = embed_query_dense(query)
    sparse_vec = embed_query_sparse(query)

    has_year_filter = _filter_has_year(qdrant_filter)

    # Build optional domain condition for dated documents
    domain_condition = None
    if domains:
        domain_condition = FieldCondition(
            key="domain",
            match=MatchAny(any=domains),
        )

    if not has_year_filter:
        # No year filter — single unified search
        # Apply domain filter if classified, otherwise full corpus
        if domain_condition:
            agent_val       = _extract_agent(qdrant_filter)
            unified_filter  = Filter(must=[
                FieldCondition(key="agent", match=MatchValue(value=agent_val)),
                domain_condition,
            ])
        else:
            unified_filter = qdrant_filter

        results = get_qdrant().query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(query=dense_vec,  using="dense", limit=top_k, filter=unified_filter),
                Prefetch(query=sparse_vec, using="bm25",  limit=top_k, filter=unified_filter),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return [point.payload for point in results.points]

    # ── Split search ──────────────────────────────────────────────────────────
    fiscal_limit = top_k - NA_CAP
    agent_val    = _extract_agent(qdrant_filter)
    year_vals    = _extract_year_values(qdrant_filter)

    # Dated filter: agent + fiscal years + domain (if classified)
    dated_must = [
        FieldCondition(key="agent",       match=MatchValue(value=agent_val)),
        FieldCondition(key="fiscal_year", match=MatchAny(any=year_vals)),
    ]
    if domain_condition:
        dated_must.append(domain_condition)
    dated_filter = Filter(must=dated_must)

    # na filter: agent + fiscal_year=na, NO domain restriction
    na_filter = Filter(must=[
        FieldCondition(key="agent",       match=MatchValue(value=agent_val)),
        FieldCondition(key="fiscal_year", match=MatchValue(value="na")),
    ])

    dated_results = get_qdrant().query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vec,  using="dense", limit=fiscal_limit, filter=dated_filter),
            Prefetch(query=sparse_vec, using="bm25",  limit=fiscal_limit, filter=dated_filter),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=fiscal_limit,
        with_payload=True,
    )

    na_results = get_qdrant().query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vec,  using="dense", limit=NA_CAP, filter=na_filter),
            Prefetch(query=sparse_vec, using="bm25",  limit=NA_CAP, filter=na_filter),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=NA_CAP,
        with_payload=True,
    )

    dated = [p.payload for p in dated_results.points]
    na    = [p.payload for p in na_results.points]
    return dated + na


# ── Reranking with priority + rag_weight boosting ─────────────────────────────

def rerank_with_boost(
    query:                str,
    candidates:           list[dict],
    top_k:                int  = TOP_K_FINAL,
    constitutional_query: bool = False,
) -> list[dict]:
    """Rerank candidates using cross-encoder then boost by priority and rag_weight.

    Scoring:
        final_score = cross_encoder_score × priority_weight × rag_weight

    Constitutional domain guard:
        constitutional chunks get CONSTITUTIONAL_NON_DOMAIN_WEIGHT (0.8)
        on non-constitutional queries, preventing them from outranking
        high-priority fiscal documents (which score 1.5 × rag_weight).
    """
    if not candidates:
        return []

    reranker = get_reranker()
    pairs    = [[query, c["text"]] for c in candidates]
    scores   = reranker.predict(pairs)

    boosted = []
    for score, chunk in zip(scores, candidates):
        priority = chunk.get("priority", "medium")
        rag_w    = float(chunk.get("rag_weight", 1.0))

        if priority == "constitutional" and not constitutional_query:
            pw = CONSTITUTIONAL_NON_DOMAIN_WEIGHT
        else:
            pw = PRIORITY_WEIGHTS.get(priority, 1.0)

        final_score = float(score) * pw * rag_w
        boosted.append((final_score, chunk))

    boosted.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in boosted[:top_k]]


# ── Context formatter ─────────────────────────────────────────────────────────

def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for LLM injection."""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}: {c['source_file']}, FY:{c['fiscal_year']}, p.{c['page_number']}]\n"
            f"{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


# ── Main retrieve function ────────────────────────────────────────────────────

def retrieve(query: str, agent: str = "finance", verbose: bool = False) -> list[dict]:
    """Full retrieval pipeline.

    Steps:
        1. Fiscal year intent detection (regex)
        2. Domain classification (Groq LLM call)
        3. Hybrid search with domain-scoped split (Qdrant)
        4. Priority + rag_weight boosted rerank (cross-encoder)

    Returns up to TOP_K_FINAL chunk dicts.
    """
    year_intent   = detect_fiscal_year_intent(query)
    qdrant_filter = build_filter(year_intent, agent=agent)
    const_query   = is_constitutional_query(query)
    domains       = classify_query(query)

    if verbose:
        mode  = year_intent["mode"]
        years = year_intent["years"]
        raw   = year_intent.get("raw_years", [])
        if mode == "explicit":
            print(f"  [retriever] Fiscal filter : {raw} → FY strings {years} + 'na'")
        elif mode == "recent":
            print(f"  [retriever] Fiscal filter : recent FYs {years} + 'na'")
        else:
            print(f"  [retriever] Fiscal filter : none (full corpus)")
        print(f"  [retriever] Domains       : {domains if domains else '(none — full corpus)'}")
        print(f"  [retriever] Constitutional : {const_query}")
        if year_intent["years"]:
            print(f"  [retriever] Split search  : {TOP_K_CANDIDATES - NA_CAP} dated + {NA_CAP} na slots")

    candidates = hybrid_search(query, qdrant_filter, domains)

    if verbose:
        print(f"  [retriever] Candidates    : {len(candidates)}")

    chunks = rerank_with_boost(query, candidates, TOP_K_FINAL, const_query)

    if verbose:
        print(f"  [retriever] Final top {len(chunks)}  :")
        for i, c in enumerate(chunks):
            pri = c.get("priority", "?")
            rw  = c.get("rag_weight",  "?")
            fy  = c.get("fiscal_year", "?")
            dom = c.get("domain",      "?")
            print(f"    {i+1}. [{fy}] [{pri}|rw={rw}|{dom}] "
                  f"{c.get('heading_path')} — {c.get('source_file')}")

    return chunks


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
            "What is Kenya's fiscal deficit target?"
    print(f"\nQuery: {query}\n")
    results = retrieve(query, agent="finance", verbose=True)
    print(f"\n── Top {len(results)} chunks ──\n")
    for i, chunk in enumerate(results, 1):
        print(
            f"[{i}] {chunk.get('source_file')} | p.{chunk.get('page_number')} | "
            f"FY:{chunk.get('fiscal_year')} | priority:{chunk.get('priority')} | "
            f"domain:{chunk.get('domain')}"
        )
        print(f"     {chunk.get('heading_path')}")
        print(f"     {chunk.get('text', '')[:200]}...")
        print()