"""
retriever.py
────────────
Multi-agent RAG retriever for the Kenya AI Executive Roundtable pipeline.

Pipeline:
  1. Detect fiscal year intent from query (regex)
  2. Classify query domain via Groq LLM call → list of domains
  3. Embed query (Voyage dense + BM25 sparse)
  4. Hybrid search in Qdrant (dense + BM25 via RRF fusion)
     — Split search when fiscal year filter active:
         dated docs: (TOP_K - NA_CAP) candidates, filtered by agent_access + domain
         na docs:    NA_CAP candidates, filtered by agent_access only
  5. Cross-encoder rerank with priority + rag_weight boosting
  6. Return top 3 chunks with full payload (text + metadata)

Agent filtering:
  Each chunk has an agent_access[] array in its payload. Filtering uses
  MatchAny on agent_access rather than exact MatchValue on a single agent
  field. This allows one chunk to serve multiple agents without duplication.
  e.g. a BPS chunk with agent_access=["finance","education","president"]
  is returned for any of those three agents' queries.

Domain classification:
  A lightweight Groq call classifies each query into 1-3 domains from the
  full 7-agent domain taxonomy. This restricts the dated candidate pool to
  documents likely to contain the answer.
  On any Groq failure the classifier falls back to no domain filter (safe).

Fiscal year filter logic:
  - Explicit year in query  → expand to both FY ranges that year touches + na
  - "current/latest" query  → last 3 FY strings + na cap
  - Conceptual/historical   → no fiscal year filter, full corpus search
  - fiscal_year == "na"     → always included but capped at NA_CAP slots

Priority boosting (applied after cross-encoder):
  final_score = cross_encoder_score × priority_weight × rag_weight

  Constitutional domain guard: constitutional chunks are capped at 0.8×
  on non-constitutional queries to prevent them dominating fiscal results.

Dependencies:
    pip install voyageai qdrant-client fastembed sentence-transformers groq python-dotenv

Usage (as module):
    from retriever import retrieve
    chunks = retrieve("What was Kenya's GDP growth in 2019?", agent="finance")

Usage (CLI test):
    python retriever.py "What was Kenya's inflation rate in 2021?"
    python retriever.py --agent education "What is Kenya's primary school enrollment rate?"
"""

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from same directory as this script
load_dotenv(Path(__file__).parent / ".env")

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
QDRANT_URL       = os.environ.get("QDRANT_URL", "http://localhost:6333")
DEFAULT_AGENT    = "finance"

TOP_K_CANDIDATES = 20   # total candidates fetched before reranking
TOP_K_FINAL      = 3    # final chunks returned to LLM after reranking

# Max na-document chunks in candidate pool when a fiscal year filter is active.
# Remaining (TOP_K_CANDIDATES - NA_CAP) slots go to dated documents.
NA_CAP           = 4

# Most recent FY in corpus — update when new documents are added
LATEST_FY_START  = 2025
RECENT_FY_STARTS = list(range(LATEST_FY_START - 2, LATEST_FY_START + 1))

# Keywords that signal "current/latest" intent
RECENCY_KEYWORDS = {
    "current", "latest", "recent", "now", "today",
    "this year", "most recent", "new", "updated"
}

# Keywords that indicate a constitutional/legal query
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

# Constitutional chunks on non-constitutional queries get this cap
CONSTITUTIONAL_NON_DOMAIN_WEIGHT = 0.8

# All valid domain values across all 7 agents
VALID_DOMAINS = {
    # Finance
    "fiscal_policy",
    "monetary_policy",
    "audit_compliance",
    "revenue_tax",
    "tax_expenditure",
    "macroeconomic_data",
    "legal_fiscal",
    "constitutional",
    "external_assessment",
    # AntiCorruption / cross-cutting
    "governance",
    "procurement",
    "legal_compliance",
    # ICT
    "internet_governance",
    # All agents
    "sector_policy",
    "sector_data",
    "sector_research",
    "institutional",
    "unknown",
}

# Domain classification prompt — full 7-agent taxonomy
_CLASSIFY_PROMPT = """You are a query classifier for a Kenya government documents RAG system covering 7 Cabinet agents.

Given a query, identify which document domains are most likely to contain the answer.

Available domains and what they cover:
- fiscal_policy: budget policy statements, fiscal deficit, public debt, MTEF, government spending, budget implementation, debt management strategies, controller of budget reports
- monetary_policy: CBK reports, interest rates, inflation, exchange rates, MPC decisions, financial sector stability
- audit_compliance: auditor general reports, audit findings, procurement irregularities, pending bills, financial statements
- revenue_tax: KRA revenue performance, tax collection, revenue targets, customs
- tax_expenditure: tax expenditures, tax reliefs, exemptions
- macroeconomic_data: economic surveys, GDP growth, employment, sectoral output, economic outlook
- legal_fiscal: finance acts, finance bills, tax legislation, financial regulations
- constitutional: constitution of Kenya, constitutional provisions, bills of rights
- external_assessment: IMF reports, World Bank reports, external debt assessments
- governance: EACC reports, anti-corruption, ethics, public officer conduct, prosecution, FATF/MER
- procurement: PPRA/PPOA reports, public procurement, debarment, asset disposal
- internet_governance: IGF reports, Kenya IGF, internet policy forums
- sector_policy: strategic plans, masterplans, policies, frameworks, guidelines for specific sectors
- sector_data: statistics reports, energy statistics, education statistics, agriculture production data
- sector_research: research reports, assessments, surveys, conference reports
- institutional: annual reports, corporate plans, financial statements of specific institutions

Rules:
- Return 1-3 domains maximum — be specific, not broad
- For cross-cutting queries return multiple: e.g. GDP + fiscal = ["macroeconomic_data", "fiscal_policy"]
- For education queries: sector_policy (strategic plans, CBC) or sector_data (KCSE stats)
- For infrastructure queries: sector_policy (road design, energy strategy) or sector_data (energy stats)
- For anti-corruption queries: governance (EACC) or procurement (PPRA) or audit_compliance
- Respond ONLY with valid JSON: {{"domains": ["domain1"]}}
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
        api_key = os.environ.get("VOYAGE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "VOYAGE_API_KEY not set. "
                "Add it to ~/PRES/Executive/.env or export it."
            )
        _voyage = voyageai.Client(api_key=api_key)
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
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. "
                "Add it to ~/PRES/Executive/.env or export it."
            )
        _groq = Groq(api_key=api_key)
    return _groq


# ── Domain classification ─────────────────────────────────────────────────────

def classify_query(query: str) -> list[str]:
    """Use Groq to classify the query into relevant document domains.

    Returns a list of domain strings from VALID_DOMAINS.
    On any failure returns [] — safe fallback to full corpus search.
    """
    try:
        response = get_groq().chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": _CLASSIFY_PROMPT.format(query=query)}],
            temperature=0.0,
            max_tokens=60,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()

        parsed  = json.loads(raw)
        domains = parsed.get("domains", [])
        valid   = [d for d in domains if d in VALID_DOMAINS]
        return valid if valid else []

    except Exception:
        return []


# ── Fiscal year helpers ───────────────────────────────────────────────────────

def year_to_fy_strings(year: str) -> list[str]:
    """Convert a calendar year to the two FY strings it touches.

    "2022" → ["2022_23", "2021_22"]
    """
    y = int(year)
    return [
        f"{y}_{str(y + 1)[-2:]}",
        f"{y - 1}_{str(y)[-2:]}",
    ]


def fy_start_to_string(fy_start: int) -> str:
    """2022 → "2022_23" """
    return f"{fy_start}_{str(fy_start + 1)[-2:]}"


def detect_fiscal_year_intent(query: str) -> dict:
    """Detect fiscal year intent from query text.

    Returns:
        mode:      "explicit" | "recent" | "none"
        years:     FY strings in stored format e.g. ["2022_23", "2021_22"]
        raw_years: calendar years found in query e.g. ["2022"]
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
        return {"mode": "explicit", "years": unique_fy, "raw_years": sorted(set(year_matches))}

    if any(kw in query_lower for kw in RECENCY_KEYWORDS):
        return {
            "mode":      "recent",
            "years":     [fy_start_to_string(y) for y in RECENT_FY_STARTS],
            "raw_years": [],
        }

    return {"mode": "none", "years": [], "raw_years": []}


def is_constitutional_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in CONSTITUTIONAL_KEYWORDS)


# ── Qdrant filter builders ────────────────────────────────────────────────────

def _agent_condition(agent: str) -> FieldCondition:
    """Filter: agent must appear in the chunk's agent_access array."""
    return FieldCondition(
        key="agent_access",
        match=MatchAny(any=[agent]),
    )


def build_filter(year_intent: dict, agent: str = DEFAULT_AGENT) -> Filter:
    """Build base Qdrant filter from year intent and agent.

    Agent filtering uses MatchAny on agent_access[] — one chunk can serve
    multiple agents without duplication.
    Domain filtering is applied separately inside hybrid_search.
    """
    ac = _agent_condition(agent)

    if not year_intent["years"]:
        return Filter(must=[ac])

    na_condition   = FieldCondition(key="fiscal_year", match=MatchValue(value="na"))
    year_condition = FieldCondition(key="fiscal_year", match=MatchAny(any=year_intent["years"]))
    fiscal_filter  = Filter(should=[na_condition, year_condition])

    return Filter(must=[ac, fiscal_filter])


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
    """Extract the agent value from the filter's agent_access condition."""
    for c in (f.must or []):
        if isinstance(c, FieldCondition) and c.key == "agent_access":
            if hasattr(c.match, "any") and c.match.any:
                return c.match.any[0]
    return DEFAULT_AGENT


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
    return SparseVector(indices=e.indices.tolist(), values=e.values.tolist())


# ── Hybrid search with domain filter + na-cap split ──────────────────────────

def hybrid_search(
    query:         str,
    qdrant_filter: Filter,
    domains:       list[str],
    agent:         str,
    top_k:         int = TOP_K_CANDIDATES,
) -> list[dict]:
    """Hybrid Qdrant search: dense + BM25 sparse fused via RRF.

    Split search when fiscal year filter is active:
      dated search — (top_k - NA_CAP) slots, agent_access + fiscal_year + domain
      na search    — NA_CAP slots, agent_access only (no domain restriction)

    Unified search when no fiscal year filter:
      agent_access + domain (if classified)
    """
    dense_vec  = embed_query_dense(query)
    sparse_vec = embed_query_sparse(query)

    has_year_filter  = _filter_has_year(qdrant_filter)
    domain_condition = (
        FieldCondition(key="domain", match=MatchAny(any=domains))
        if domains else None
    )

    if not has_year_filter:
        # Single unified search
        if domain_condition:
            unified_filter = Filter(must=[_agent_condition(agent), domain_condition])
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
    year_vals    = _extract_year_values(qdrant_filter)

    # Dated: agent_access + fiscal_year + domain
    dated_must = [
        _agent_condition(agent),
        FieldCondition(key="fiscal_year", match=MatchAny(any=year_vals)),
    ]
    if domain_condition:
        dated_must.append(domain_condition)
    dated_filter = Filter(must=dated_must)

    # na: agent_access + fiscal_year=na (no domain restriction)
    na_filter = Filter(must=[
        _agent_condition(agent),
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

    return [p.payload for p in dated_results.points] + [p.payload for p in na_results.points]


# ── Reranking with priority + rag_weight boosting ─────────────────────────────

def rerank_with_boost(
    query:                str,
    candidates:           list[dict],
    top_k:                int  = TOP_K_FINAL,
    constitutional_query: bool = False,
) -> list[dict]:
    """Cross-encoder rerank then boost by priority × rag_weight.

    final_score = cross_encoder_score × priority_weight × rag_weight
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

        boosted.append((float(score) * pw * rag_w, chunk))

    boosted.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in boosted[:top_k]]


# ── Context formatter ─────────────────────────────────────────────────────────

def format_context(chunks: list[dict]) -> str:
    """Format chunks into a context string for LLM prompt injection."""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}: {c['source_file']}, FY:{c['fiscal_year']}, p.{c['page_number']}]\n"
            f"{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


# ── Main retrieve function ────────────────────────────────────────────────────

def retrieve(
    query:   str,
    agent:   str  = DEFAULT_AGENT,
    verbose: bool = False,
) -> list[dict]:
    """Full retrieval pipeline for any of the 7 Cabinet agents.

    Args:
        query:   natural language question
        agent:   one of: finance | education | agriculture | ict |
                         infrastructure | anticorruption | president
        verbose: print debug info

    Returns:
        list of up to TOP_K_FINAL chunk dicts, each with full payload
        including text, source_file, fiscal_year, domain, priority,
        rag_weight, agent_access, topics, and all other metadata.
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
        print(f"  [retriever] Agent         : {agent}")
        print(f"  [retriever] Domains       : {domains if domains else '(none — full corpus)'}")
        print(f"  [retriever] Constitutional : {const_query}")
        if year_intent["years"]:
            print(f"  [retriever] Split search  : {TOP_K_CANDIDATES - NA_CAP} dated + {NA_CAP} na slots")

    candidates = hybrid_search(query, qdrant_filter, domains, agent)

    if verbose:
        print(f"  [retriever] Candidates    : {len(candidates)}")

    chunks = rerank_with_boost(query, candidates, TOP_K_FINAL, const_query)

    if verbose:
        print(f"  [retriever] Final top {len(chunks)}  :")
        for i, c in enumerate(chunks):
            print(
                f"    {i+1}. [{c.get('fiscal_year')}] "
                f"[{c.get('priority')}|rw={c.get('rag_weight')}|{c.get('domain')}] "
                f"{c.get('heading_path')} — {c.get('source_file')}"
            )

    return chunks


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the RAG retriever")
    parser.add_argument("query", nargs="*", help="Query string")
    parser.add_argument(
        "--agent", "-a",
        default=DEFAULT_AGENT,
        choices=["finance", "education", "agriculture", "ict",
                 "infrastructure", "anticorruption", "president"],
        help=f"Agent to retrieve for (default: {DEFAULT_AGENT})",
    )
    args = parser.parse_args()

    query = " ".join(args.query) if args.query else "What is Kenya's fiscal deficit target?"

    print(f"\nQuery : {query}")
    print(f"Agent : {args.agent}\n")

    results = retrieve(query, agent=args.agent, verbose=True)

    print(f"\n── Top {len(results)} chunks ──\n")
    for i, chunk in enumerate(results, 1):
        print(
            f"[{i}] {chunk.get('source_file')} | "
            f"p.{chunk.get('page_number')} | "
            f"FY:{chunk.get('fiscal_year')} | "
            f"priority:{chunk.get('priority')} | "
            f"domain:{chunk.get('domain')}"
        )
        print(f"     {chunk.get('heading_path')}")
        print(f"     {chunk.get('text', '')[:200]}...")
        print()