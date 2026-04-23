"""
chunk_documents.py — Universal chunker for Kenya AI Executive Roundtable RAG pipeline.

Reads the universal config.yaml (generated from central_inventory.csv) and applies
per-document chunking strategy, chunk_size, chunk_overlap, skip_sections, and rich
metadata — including agent_access array for Qdrant RBAC payload filtering.

Architecture:
  - One Qdrant collection: kenya_executive_roundtable
  - RBAC via agent_access[] array in chunk payload
  - All 438 docs share one flat cache dir and one config.yaml
  - config.yaml is generated from central_inventory.csv — do not edit manually

Cache format (per doc JSON in processed/):
  blocks[]: block_type, heading_path, text, block_index, page_number
  tables[]: table_id, table_index, caption, heading_path, page_number,
            markdown, columns, records

Chunking strategies:
  narrative       — text (overflow split + heading cut) + tables
  legal           — text (heading section = one chunk, no overflow split) + tables
  audit_findings  — same as legal (finding boundaries via headings)
  tables_only     — tables only
  hybrid          — text (narrative) + tables

Usage:
    python chunk_documents.py \\
        --config  /home/darwin/PRES/Executive/config.yaml \\
        --input   /home/darwin/PRES/Executive/data/processed/ \\
        --output  /home/darwin/PRES/Executive/data/chunks/

    # Re-chunk even if output JSONL already exists
    python chunk_documents.py \\
        --config  /home/darwin/PRES/Executive/config.yaml \\
        --input   /home/darwin/PRES/Executive/data/processed/ \\
        --output  /home/darwin/PRES/Executive/data/chunks/ \\
        --force

    # Chunk a single document by doc_slug
    python chunk_documents.py \\
        --config  /home/darwin/PRES/Executive/config.yaml \\
        --input   /home/darwin/PRES/Executive/data/processed/ \\
        --output  /home/darwin/PRES/Executive/data/chunks/ \\
        --slug    finance_2023.budget.policy.statement

    # Chunk only a specific agent's documents
    python chunk_documents.py \\
        --config  /home/darwin/PRES/Executive/config.yaml \\
        --input   /home/darwin/PRES/Executive/data/processed/ \\
        --output  /home/darwin/PRES/Executive/data/chunks/ \\
        --agent   finance
"""

import re
import json
import uuid
import argparse
from collections import Counter
from pathlib import Path

import tiktoken
import yaml


# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZER
# ══════════════════════════════════════════════════════════════════════════════

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = _TOKENIZER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _TOKENIZER.decode(tokens[:max_tokens])


# ══════════════════════════════════════════════════════════════════════════════
# SKIP-SECTION CHECKER
# ══════════════════════════════════════════════════════════════════════════════

# Default skip sections per document_type — covers the most common boilerplate.
# Config-level skip_sections (if present) override these entirely.
DEFAULT_SKIP_SECTIONS = {
    "budget_policy_statement":  ["foreword", "acknowledgement", "table of contents"],
    "budget_review_outlook":    ["foreword", "acknowledgement", "table of contents"],
    "debt_management_strategy": ["foreword", "acknowledgement"],
    "public_debt_report":       ["foreword", "acknowledgement"],
    "controller_of_budget":     ["county breakdown", "appendix", "foreword", "acknowledgement"],
    "auditor_general_report":   ["foreword", "table of contents"],
    "audit_report":             ["foreword", "table of contents"],
    "forensic_audit":           ["foreword", "table of contents"],
    "financial_statements":     ["foreword", "table of contents", "notes to financial statements"],
    "cbk_annual_report": [
        "directors report", "directors' report", "financial statements",
        "statement of financial position", "income statement",
        "cash flow statement", "notes to financial statements",
        "staff costs", "human resources", "corporate governance", "board committees",
    ],
    "imf_report":               ["foreword", "acknowledgement"],
    "world_bank_report":        ["foreword", "acknowledgement"],
    "economic_survey":          ["foreword", "acknowledgement"],
    "kra_revenue_performance":  ["foreword", "acknowledgement"],
    "kra_corporate_plan":       ["foreword", "acknowledgement"],
    "tax_expenditure_report":   ["foreword", "acknowledgement"],
    "eacc_report":              ["foreword", "acknowledgement", "table of contents"],
    "ppra_report":              ["foreword", "acknowledgement", "table of contents"],
    "odpp_report":              ["foreword", "acknowledgement", "table of contents"],
    "mer_report":               ["foreword", "acknowledgement", "table of contents"],
    "strategic_plan":           ["foreword", "acknowledgement", "table of contents"],
    "masterplan":               ["foreword", "acknowledgement", "table of contents"],
    "annual_report":            ["foreword", "acknowledgement", "table of contents"],
    "policy":                   ["foreword", "acknowledgement"],
    "sector_report":            ["foreword", "acknowledgement", "table of contents"],
    "igf_report":               ["foreword", "acknowledgement", "table of contents"],
    "statistics_report":        ["foreword", "table of contents"],
    "research_report":          ["foreword", "acknowledgement"],
    "survey":                   ["foreword", "acknowledgement"],
    "guidelines":               ["foreword", "acknowledgement"],
    "manual":                   ["foreword", "acknowledgement", "table of contents"],
    "framework":                ["foreword", "acknowledgement", "table of contents"],
    "assessment":               ["foreword", "acknowledgement"],
    "conference_report":        ["foreword", "acknowledgement", "table of contents"],
    "magazine": [
        "table of contents", "advertisement", "editor's note",
        "from the desk", "letters to the editor",
    ],
}


def get_skip_sections(doc_config: dict) -> list:
    """Return skip_sections list. Config value takes precedence over defaults."""
    if "skip_sections" in doc_config:
        return doc_config["skip_sections"] or []
    doc_type = doc_config.get("document_type", "unknown")
    return DEFAULT_SKIP_SECTIONS.get(doc_type, [])


def make_skip_checker(skip_sections: list):
    """Return callable: True if heading should be skipped (case-insensitive substring)."""
    skip_lower = [s.lower() for s in skip_sections]

    def should_skip(heading: str) -> bool:
        if not skip_lower:
            return False
        h = heading.lower()
        return any(s in h for s in skip_lower)

    return should_skip


# ══════════════════════════════════════════════════════════════════════════════
# CACHE FILE LOOKUP
# ══════════════════════════════════════════════════════════════════════════════

def normalize_filename_to_stem(name: str) -> str:
    """
    Normalise a PDF filename to match how Docling writes cache stems.

    Docling replaces spaces with underscores but preserves hyphens and dots
    in most positions. We lowercase and replace spaces only — this mirrors
    what we see in the processed/ directory.

    Examples:
        2023-Budget-Policy-Statement.pdf  →  2023-Budget-Policy-Statement
        CBK_2017 Annual Report.pdf        →  CBK_2017_Annual_Report
        Finance Act 2016.pdf              →  Finance_Act_2016
    """
    stem = Path(name).stem if name.lower().endswith(".pdf") else name
    # Replace spaces with underscores, keep hyphens and dots as-is
    return re.sub(r"\s+", "_", stem)


def normalize_cache_stem(stem: str) -> str:
    """Lowercase a cache stem for case-insensitive comparison."""
    return stem.lower()


def build_cache_index(cache_dir: Path) -> dict:
    """
    Return two indexes:
        exact_index  — {lowercase_stem: Path}   for O(1) exact lookup
        stems        — [(lowercase_stem, Path)]  for prefix scanning
    """
    exact = {}
    for f in cache_dir.glob("*.json"):
        exact[f.stem.lower()] = f
    return exact


_MIN_PREFIX_LENGTH = 20


def find_cache_file(cache_index: dict, source_file: str, doc_slug: str) -> Path | None:
    """
    Find the cache JSON for a document.

    Primary key  : source_file stem (normalised), e.g. "2023-Budget-Policy-Statement"
    Fallback key : doc_slug dots→underscores, e.g. "finance_2023_budget_policy_statement"

    Resolution order:
        1. Exact match on normalised source_file stem (case-insensitive)
        2. Exact match on normalised doc_slug
        3. Cache stem is a prefix of the normalised source_file stem (truncation)
        4. Normalised source_file stem is a prefix of cache stem (rare)
    """
    # Build normalised versions of both keys
    sf_stem  = normalize_filename_to_stem(source_file).lower()   # e.g. "2023-budget-policy-statement"
    slug_key = doc_slug.replace(".", "_").lower()                  # e.g. "finance_2023_budget_policy_statement"

    # 1. Exact match on source_file stem
    if sf_stem in cache_index:
        return cache_index[sf_stem]

    # 2. Exact match on slug
    if slug_key in cache_index:
        return cache_index[slug_key]

    # 3. Cache stem is a truncated prefix of source_file stem
    for stem, path in cache_index.items():
        if len(stem) >= _MIN_PREFIX_LENGTH and sf_stem.startswith(stem):
            return path

    # 4. Source_file stem is a prefix of a longer cache stem
    if len(sf_stem) >= _MIN_PREFIX_LENGTH:
        for stem, path in cache_index.items():
            if stem.startswith(sf_stem):
                return path

    return None


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: Path) -> tuple[dict, list]:
    """
    Load universal config.yaml.

    Returns:
        pipeline   — pipeline: section dict
        documents  — list of doc config dicts (in config order)
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pipeline  = config["pipeline"]
    documents = config.get("documents", [])
    return pipeline, documents


# ══════════════════════════════════════════════════════════════════════════════
# BASE METADATA  (embedded in every chunk payload)
# ══════════════════════════════════════════════════════════════════════════════

def base_metadata(doc_config: dict) -> dict:
    """
    Build the rich metadata payload for every chunk from the config entry.

    agent_access is kept as a list for Qdrant array filtering:
        MUST filter: {"key": "agent_access", "match": {"any": ["finance"]}}
    """
    topics      = doc_config.get("topics", []) or []
    agent_access = doc_config.get("agent_access", []) or []

    return {
        # Identity
        "source_file":    doc_config["source_file"],
        "doc_id":         doc_config.get("doc_id", ""),
        "doc_slug":       doc_config.get("doc_slug", ""),
        # Agent routing (RBAC)
        "agent_access":   sorted(agent_access),          # list — Qdrant array filter
        "primary_agents": doc_config.get("primary_agents", []),
        "issuing_agent":  doc_config.get("issuing_agent", "unknown"),
        # Classification
        "document_type":  doc_config.get("document_type",  "unknown"),
        "domain":         doc_config.get("domain",         "unknown"),
        "topics":         sorted(topics),
        # Temporal
        "fiscal_year":    str(doc_config.get("fiscal_year", "na")),
        "doc_year":       doc_config.get("doc_year",       None),
        "report_period":  doc_config.get("report_period",  "annual"),
        # Retrieval weights
        "priority":       doc_config.get("priority",       "medium"),
        "rag_weight":     float(doc_config.get("rag_weight", 1.0)),
        "category":       doc_config.get("category",       0),
        # Document properties
        "is_scanned":     doc_config.get("is_scanned",     False),
        "language":       doc_config.get("language",       "english"),
        "geographic_scope": doc_config.get("geographic_scope", "national"),
        "superseded":     doc_config.get("superseded",     False),
        # Chunking info (useful for debugging)
        "chunk_strategy": doc_config.get("chunking_strategy", "narrative"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CAPTION LOOKUP  (for table caption backfill)
# ══════════════════════════════════════════════════════════════════════════════

def build_caption_lookup(blocks: list) -> dict:
    """Return {page_number: caption_text} from caption-type blocks."""
    lookup = {}
    for block in blocks:
        if block.get("block_type") == "caption":
            page = block["page_number"]
            if page not in lookup:
                lookup[page] = block["text"].strip()
    return lookup


# ══════════════════════════════════════════════════════════════════════════════
# TEXT FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

def format_chunk_text(texts: list, heading: str) -> str:
    """Join block texts, prefixed with [heading] if present."""
    body = "\n\n".join(t for t in texts if t)
    if heading:
        return f"[{heading}]\n\n{body}"
    return body


# ══════════════════════════════════════════════════════════════════════════════
# TEXT BLOCK CHUNKER
# ══════════════════════════════════════════════════════════════════════════════

CONTENT_BLOCK_TYPES = {"paragraph", "list_item"}


def chunk_text_blocks(
    doc:               dict,
    doc_config:        dict,
    meta:              dict,
    chunk_size:        int,
    chunk_overlap:     int,
    split_on_overflow: bool,
) -> list:
    """
    Core text chunker used by all text-bearing strategies.

    Args:
        doc               — parsed cache JSON
        doc_config        — this document's config entry
        meta              — pre-built base_metadata dict
        chunk_size        — max tokens per chunk
        chunk_overlap     — overlap tokens (used to carry last block on overflow)
        split_on_overflow — True  → narrative/hybrid: split on token overflow
                            False → legal/audit_findings: one heading = one chunk

    Flush rules:
        Hard cut (no overlap): heading change
        Soft cut (carry last block as overlap): token overflow (narrative only)
    """
    blocks      = [b for b in doc.get("blocks", []) if b.get("block_type") in CONTENT_BLOCK_TYPES]
    skip        = make_skip_checker(get_skip_sections(doc_config))
    chunks      = []

    cur_texts      = []
    cur_block_objs = []
    cur_tokens     = 0
    cur_heading    = ""
    cur_page       = 1
    cur_idx_start  = 0
    cur_idx_end    = 0

    def emit_chunk():
        if not cur_texts:
            return
        chunks.append({
            "chunk_id":          str(uuid.uuid4()),
            "chunk_type":        "text",
            "text":              format_chunk_text(cur_texts, cur_heading),
            "heading_path":      cur_heading,
            "page_number":       cur_page,
            "block_index_start": cur_idx_start,
            "block_index_end":   cur_idx_end,
            **meta,
        })

    for block in blocks:
        heading = block["heading_path"][0] if block.get("heading_path") else ""

        if skip(heading):
            continue

        btext = block["text"].strip()
        if not btext:
            continue

        btokens = count_tokens(btext)
        bidx    = block["block_index"]
        bpage   = block["page_number"]

        heading_changed = bool(cur_texts) and (heading != cur_heading)
        token_overflow  = (
            split_on_overflow
            and bool(cur_texts)
            and (cur_tokens + btokens > chunk_size)
        )

        # Hard cut: heading change
        if heading_changed:
            emit_chunk()
            cur_texts      = []
            cur_block_objs = []
            cur_tokens     = 0

        # Soft cut: token overflow — carry last block as overlap
        elif token_overflow:
            overlap_block = cur_block_objs[-1] if cur_block_objs else None

            emit_chunk()
            cur_texts      = []
            cur_block_objs = []
            cur_tokens     = 0

            if overlap_block is not None:
                ob_text    = overlap_block["text"].strip()
                ob_heading = (
                    overlap_block["heading_path"][0]
                    if overlap_block.get("heading_path")
                    else ""
                )
                cur_texts      = [ob_text]
                cur_block_objs = [overlap_block]
                cur_tokens     = count_tokens(ob_text)
                cur_heading    = ob_heading
                cur_page       = overlap_block["page_number"]
                cur_idx_start  = overlap_block["block_index"]
                cur_idx_end    = overlap_block["block_index"]

        # Init accumulator on first block or after flush
        if not cur_texts:
            cur_heading   = heading
            cur_page      = bpage
            cur_idx_start = bidx
            cur_idx_end   = bidx

        cur_texts.append(btext)
        cur_block_objs.append(block)
        cur_tokens  += btokens
        cur_idx_end  = bidx

    emit_chunk()
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# TABLE CHUNKER
# ══════════════════════════════════════════════════════════════════════════════

def chunk_tables(
    doc:        dict,
    doc_config: dict,
    meta:       dict,
    chunk_size: int,
) -> list:
    """
    One chunk per table.

    Caption resolution order:
        1. table.caption  (if non-empty)
        2. caption-type block on the same page
        3. table.table_id  (final fallback)

    Markdown truncated to chunk_size tokens for embedding.
    Full records + columns preserved in payload.
    """
    tables = doc.get("tables", [])
    if not tables:
        return []

    caption_lookup = build_caption_lookup(doc.get("blocks", []))
    chunks         = []

    for table in tables:
        caption = table.get("caption", "").strip()
        if not caption:
            caption = caption_lookup.get(table["page_number"], "")
        if not caption:
            caption = table.get("table_id", "")

        heading  = table["heading_path"][0] if table.get("heading_path") else ""
        markdown = truncate_to_tokens(table.get("markdown", ""), chunk_size)

        parts = []
        if heading:
            parts.append(f"[{heading}]")
        if caption:
            parts.append(caption)
        parts.append(markdown)
        text = "\n\n".join(p for p in parts if p)

        chunks.append({
            "chunk_id":    str(uuid.uuid4()),
            "chunk_type":  "table",
            "text":        text,
            "heading_path": heading,
            "page_number": table["page_number"],
            "table_id":    table.get("table_id", ""),
            "table_index": table.get("table_index", 0),
            "caption":     caption,
            "columns":     table.get("columns", []),
            "records":     table.get("records", []),
            **meta,
        })

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def chunk_document(
    doc:        dict,
    doc_config: dict,
    chunk_min:  int,
    chunk_max:  int,
) -> list:
    """
    Route document to the correct chunking strategy and return all chunks.

    chunk_size is clamped to [chunk_min, chunk_max] from pipeline config.
    chunk_overlap is taken directly from config (no clamping — already computed).
    """
    strategy      = doc_config.get("chunking_strategy", "narrative")
    chunk_size    = doc_config.get("chunk_size", 350)
    chunk_size    = max(chunk_min, min(chunk_max, chunk_size))
    chunk_overlap = doc_config.get("chunk_overlap", 88)

    meta   = base_metadata(doc_config)
    chunks = []

    if strategy == "narrative":
        chunks += chunk_text_blocks(doc, doc_config, meta, chunk_size, chunk_overlap, split_on_overflow=True)
        chunks += chunk_tables(doc, doc_config, meta, chunk_size)

    elif strategy == "legal":
        chunks += chunk_text_blocks(doc, doc_config, meta, chunk_size, chunk_overlap, split_on_overflow=False)
        chunks += chunk_tables(doc, doc_config, meta, chunk_size)

    elif strategy == "audit_findings":
        chunks += chunk_text_blocks(doc, doc_config, meta, chunk_size, chunk_overlap, split_on_overflow=False)
        chunks += chunk_tables(doc, doc_config, meta, chunk_size)

    elif strategy == "tables_only":
        chunks += chunk_tables(doc, doc_config, meta, chunk_size)

    elif strategy == "hybrid":
        chunks += chunk_text_blocks(doc, doc_config, meta, chunk_size, chunk_overlap, split_on_overflow=True)
        chunks += chunk_tables(doc, doc_config, meta, chunk_size)

    else:
        print(f"  WARNING: unknown strategy '{strategy}' — falling back to narrative")
        chunks += chunk_text_blocks(doc, doc_config, meta, chunk_size, chunk_overlap, split_on_overflow=True)
        chunks += chunk_tables(doc, doc_config, meta, chunk_size)

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Universal document chunker — Kenya AI Executive Roundtable"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to universal config.yaml",
    )
    parser.add_argument(
        "--input", required=True,
        help="Directory of parsed cache JSONs (data/processed/)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Directory for output JSONL files (data/chunks/)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-chunk even if output JSONL already exists",
    )
    parser.add_argument(
        "--slug", default=None,
        help="Process only one document by doc_slug",
    )
    parser.add_argument(
        "--agent", default=None,
        help="Process only documents where agent is in primary_agents",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    cache_dir   = Path(args.input)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        return
    if not cache_dir.exists():
        print(f"ERROR: cache directory not found: {cache_dir}")
        return

    # ── Load config ───────────────────────────────────────────────────────────
    pipeline, documents = load_config(config_path)
    chunk_min = pipeline.get("chunk_min_tokens", 100)
    chunk_max = pipeline.get("chunk_max_tokens", 500)

    print(f"Collection      : {pipeline.get('qdrant_collection', 'kenya_executive_roundtable')}")
    print(f"Config entries  : {len(documents)}")
    print(f"Cache dir       : {cache_dir}")
    print(f"Output dir      : {output_dir}")
    print(f"Token range     : {chunk_min}–{chunk_max}")
    print(f"Force re-chunk  : {args.force}")
    if args.slug:
        print(f"Single slug     : {args.slug}")
    if args.agent:
        print(f"Agent filter    : {args.agent}")
    print()

    # ── Build cache index ─────────────────────────────────────────────────────
    cache_index = build_cache_index(cache_dir)
    print(f"Cache files found: {len(cache_index)}\n")

    # ── Apply filters ─────────────────────────────────────────────────────────
    if args.slug:
        documents = [d for d in documents if d.get("doc_slug") == args.slug]
        if not documents:
            print(f"ERROR: slug '{args.slug}' not found in config")
            return

    if args.agent:
        documents = [
            d for d in documents
            if args.agent in (d.get("primary_agents") or [])
        ]
        print(f"Docs for agent '{args.agent}': {len(documents)}\n")

    # ── Process documents ─────────────────────────────────────────────────────
    total_chunks      = 0
    processed         = []
    skipped_no_cache  = []
    skipped_exists    = []
    skipped_error     = []
    skipped_collision = []
    used_cache_files  = {}   # {str(cache_path): first_claiming_slug}

    for doc_config in documents:
        slug     = doc_config.get("doc_slug", "")
        fname    = doc_config.get("source_file", slug)
        strategy = doc_config.get("chunking_strategy", "narrative")

        # ── No cache file → warn and skip ────────────────────────────────
        cache_file = find_cache_file(cache_index, fname, slug)
        if cache_file is None:
            print(f"  SKIP (no cache)  : {fname}")
            skipped_no_cache.append(fname)
            continue

        # ── Collision detection ───────────────────────────────────────────
        cache_key = str(cache_file)
        if cache_key in used_cache_files:
            first = used_cache_files[cache_key]
            print(f"  SKIP (collision) : {fname}  →  cache claimed by {first}")
            skipped_collision.append(fname)
            continue
        used_cache_files[cache_key] = fname

        # ── Already chunked and not forcing ──────────────────────────────
        out_file = output_dir / f"{slug}.jsonl"
        if out_file.exists() and not args.force:
            skipped_exists.append(slug)
            continue

        # ── Load parsed doc ───────────────────────────────────────────────
        try:
            with open(cache_file, encoding="utf-8") as f:
                doc = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  SKIP (read err)  : {fname}  →  {e}")
            skipped_error.append(fname)
            continue

        if doc.get("error"):
            print(f"  SKIP (parse err) : {fname}  →  {doc['error']}")
            skipped_error.append(fname)
            continue

        # ── Chunk ─────────────────────────────────────────────────────────
        try:
            chunks = chunk_document(doc, doc_config, chunk_min, chunk_max)
        except Exception as e:
            print(f"  SKIP (chunk err) : {fname}  →  {e}")
            skipped_error.append(fname)
            continue

        n = len(chunks)

        # ── Write JSONL ───────────────────────────────────────────────────
        with open(out_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        access_str = "|".join(doc_config.get("agent_access", []))
        print(f"  ✓ {n:>4} chunks  [{strategy:<15}]  {slug}.jsonl  [{access_str}]")
        total_chunks += n
        processed.append((slug, n, strategy))

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"Processed        : {len(processed):>4} documents → {total_chunks:,} total chunks")
    print(f"Skipped (exists) : {len(skipped_exists):>4}  (use --force to re-chunk)")
    print(f"Skipped (errors) : {len(skipped_error):>4}  (parse/chunk errors)")
    print(f"Skipped (no cach): {len(skipped_no_cache):>4}  (not yet parsed)")
    print(f"Skipped (collide): {len(skipped_collision):>4}  (parser truncation — needs separate parse)")

    if skipped_no_cache:
        print()
        print("Not yet parsed:")
        for fname in sorted(skipped_no_cache):
            print(f"  - {fname}")

    if skipped_collision:
        print()
        print("Cache collisions — parse these separately:")
        for fname in sorted(skipped_collision):
            print(f"  - {fname}")

    print("=" * 70)

    if processed:
        print()
        strat_counts   = Counter(s for _, _, s in processed)
        chunk_by_strat = {}
        for _, n, s in processed:
            chunk_by_strat[s] = chunk_by_strat.get(s, 0) + n
        print("Strategy breakdown:")
        for strat, doc_count in sorted(strat_counts.items()):
            print(f"  {strat:<18} : {doc_count:>3} docs  →  {chunk_by_strat[strat]:>6,} chunks")


if __name__ == "__main__":
    main()