"""
chunk_documents.py — Config-aware chunker for Kenya AI Executive Roundtable RAG pipeline.

Reads per-agent config.yaml to apply:
  - Per-document chunking strategy (narrative | legal | audit_findings | tables_only | hybrid)
  - Per-document chunk_size and chunk_overlap from config
  - skip_sections filtering via heading_path
  - Rich metadata on every chunk (domain, priority, rag_weight, document_type, etc.)
  - Agent ID from config (not hardcoded)

Fixes vs original:
  - normalize_fiscal_year(): YAML parses 2016_17 as int 201617 — converts back to "2016_17"
  - build_cache_index() + find_cache_file(): prefix-match for parser-truncated filenames
  - Collision detection: one cache file → one config entry, prevents duplicate chunks

Usage:
    python chunk_documents.py --config Finance/config.yaml \\
                              --input  Finance/data/cache/ \\
                              --output Finance/data/chunks/

    # Re-chunk even if output JSONL already exists
    python chunk_documents.py --config Finance/config.yaml \\
                              --input  Finance/data/cache/ \\
                              --output Finance/data/chunks/ \\
                              --force
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
    """Truncate text to at most max_tokens tokens, decoding cleanly."""
    tokens = _TOKENIZER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _TOKENIZER.decode(tokens[:max_tokens])


# ══════════════════════════════════════════════════════════════════════════════
# SLUG NORMALISATION  (filename ↔ cache file matching)
# ══════════════════════════════════════════════════════════════════════════════

def normalize_to_slug(name: str) -> str:
    """Convert a filename or stem to doc_slug format.

    Rules: strip .pdf, replace hyphens / spaces / dots with underscores, lowercase.

    Examples:
        2019-Budget-Policy-Statement.pdf  → 2019_budget_policy_statement
        CBK_2022 Annual Report.pdf        → cbk_2022_annual_report
    """
    stem = Path(name).stem if name.lower().endswith(".pdf") else name
    return re.sub(r"[\s\-\.]", "_", stem).lower()


def build_cache_index(cache_dir: Path) -> dict:
    """Scan cache directory once and return {stem: Path} for all .json files.

    Used for both exact and prefix-match lookups — the parser truncates long
    filenames when writing cache files, so a direct slug lookup often fails
    even though the file exists under a shorter name.
    """
    return {f.stem: f for f in cache_dir.glob("*.json")}


# Minimum stem length for a prefix match to be trusted (avoids false positives
# like "imf" matching "imf_2018" and "imf_2021" simultaneously).
_MIN_PREFIX_LENGTH = 35


def find_cache_file(cache_index: dict, slug: str) -> Path | None:
    """Locate the cache JSON for a given doc slug.

    Resolution order:
        1. Exact match               slug == cache stem
        2. Slug is a prefix          cache_stem is truncated version of slug
        3. Cache stem is a prefix    slug is truncated (rare edge case)

    The _MIN_PREFIX_LENGTH guard prevents short stems like "imf" from
    matching multiple unrelated documents.
    """
    # 1. Exact
    if slug in cache_index:
        return cache_index[slug]

    # 2. Cache file is a truncated prefix of the full config slug
    for stem, path in cache_index.items():
        if len(stem) >= _MIN_PREFIX_LENGTH and slug.startswith(stem):
            return path

    # 3. Config slug is a prefix of a longer cache stem (rare)
    if len(slug) >= _MIN_PREFIX_LENGTH:
        for stem, path in cache_index.items():
            if stem.startswith(slug):
                return path

    return None


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: Path) -> tuple[dict, dict]:
    """Load config.yaml.

    Returns:
        pipeline   — the pipeline: section dict
        doc_lookup — {doc_slug: doc_config_dict} for all documents
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pipeline   = config["pipeline"]
    doc_lookup = {}
    for doc in config.get("documents", []):
        slug             = normalize_to_slug(doc["filename"])
        doc_lookup[slug] = doc

    return pipeline, doc_lookup


# ══════════════════════════════════════════════════════════════════════════════
# SKIP-SECTION CHECKER
# ══════════════════════════════════════════════════════════════════════════════

def make_skip_checker(skip_sections: list):
    """Return a callable that checks whether a heading should be skipped.

    Matching is case-insensitive substring — e.g. "foreword" matches
    "FOREWORD", "The Foreword", "CS Foreword and Acknowledgements".
    """
    skip_lower = [s.lower() for s in (skip_sections or [])]

    def should_skip(heading: str) -> bool:
        if not skip_lower:
            return False
        h = heading.lower()
        return any(s in h for s in skip_lower)

    return should_skip


# ══════════════════════════════════════════════════════════════════════════════
# CAPTION LOOKUP  (for table caption backfill)
# ══════════════════════════════════════════════════════════════════════════════

def build_caption_lookup(blocks: list) -> dict:
    """Build {page_number: caption_text} from caption-type blocks.

    Keeps the first caption encountered per page (matches table ordering).
    Used to backfill empty table captions from adjacent caption blocks.
    """
    lookup = {}
    for block in blocks:
        if block.get("block_type") == "caption":
            page = block["page_number"]
            if page not in lookup:
                lookup[page] = block["text"].strip()
    return lookup


# ══════════════════════════════════════════════════════════════════════════════
# FISCAL YEAR NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def normalize_fiscal_year(fy) -> str:
    """Normalize fiscal_year from config to the stored string format "YYYY_YY".

    YAML parses bare values like 2016_17 as the integer 201617 because
    underscores are valid numeric separators in YAML 1.1 (used by PyYAML).
    This converts that integer back to the correct string "2016_17".

    Handles all cases:
        201617    (int)  → "2016_17"
        "2016_17" (str)  → "2016_17"   (already correct)
        "na"      (str)  → "na"        (timeless document)
        None             → "na"
    """
    if fy is None:
        return "na"
    s = str(fy)
    if s in ("na", "None"):
        return "na"
    # Integer form e.g. "201617" → "2016_17"
    if "_" not in s and len(s) == 6 and s.isdigit():
        return f"{s[:4]}_{s[4:]}"
    return s


# ══════════════════════════════════════════════════════════════════════════════
# BASE METADATA  (embedded in every chunk)
# ══════════════════════════════════════════════════════════════════════════════

def base_metadata(doc: dict, doc_config: dict, agent: str) -> dict:
    """All rich metadata fields added to every chunk payload."""
    return {
        "source_file":    doc["source_file"],
        "doc_slug":       doc["doc_slug"],
        "agent":          agent,
        "fiscal_year":    normalize_fiscal_year(doc_config.get("fiscal_year", "na")),
        "document_type":  doc_config.get("document_type",     "unknown"),
        "domain":         doc_config.get("domain",            "unknown"),
        "priority":       doc_config.get("priority",          "medium"),
        "rag_weight":     doc_config.get("rag_weight",        1.0),
        "report_period":  doc_config.get("report_period",     "annual"),
        "category":       doc_config.get("category",          0),
        "is_scanned":     doc.get("is_scanned",               False),
        "chunk_strategy": doc_config.get("chunking_strategy", "narrative"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TEXT FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

def format_chunk_text(texts: list, heading: str) -> str:
    """Join block texts, prefixed with [heading] if present.

    Mirrors the format already used in production chunks:
        [1.5 Economic Outlook]\\n\\n62. In addition, ...
    """
    body = "\n\n".join(t for t in texts if t)
    if heading:
        return f"[{heading}]\n\n{body}"
    return body


# ══════════════════════════════════════════════════════════════════════════════
# TEXT BLOCK CHUNKER  (narrative / legal / audit_findings / hybrid)
# ══════════════════════════════════════════════════════════════════════════════

# Block types to embed — heading and caption are structural markers, not content
CONTENT_BLOCK_TYPES = {"paragraph", "list_item"}


def chunk_text_blocks(
    doc:               dict,
    doc_config:        dict,
    agent:             str,
    chunk_size:        int,
    split_on_overflow: bool,
) -> list:
    """Core text chunker used by all text-bearing strategies.

    Args:
        doc               — parsed cache JSON
        doc_config        — this document's config.yaml entry
        agent             — agent ID from pipeline config
        chunk_size        — target max tokens per chunk (already clamped)
        split_on_overflow — True  → narrative/hybrid: split large sections with overlap
                            False → legal/audit_findings: one heading section = one chunk

    Flush rules:
        Always: heading change (hard cut, no overlap across section boundaries)
        Narrative only: token overflow within same heading (overlap = re-include last block)
    """
    blocks = [b for b in doc.get("blocks", []) if b.get("block_type") in CONTENT_BLOCK_TYPES]
    skip   = make_skip_checker(doc_config.get("skip_sections", []))
    meta   = base_metadata(doc, doc_config, agent)
    chunks = []

    # ── Accumulator state ─────────────────────────────────────────────────────
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

    # ── Main loop ──────────────────────────────────────────────────────────────
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

        # ── Flush: heading change (hard cut, no overlap) ──────────────────
        if heading_changed:
            emit_chunk()
            cur_texts      = []
            cur_block_objs = []
            cur_tokens     = 0

        # ── Flush: token overflow (soft cut, carry overlap block) ─────────
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

        # ── Initialise accumulator on first block or after flush ──────────
        if not cur_texts:
            cur_heading   = heading
            cur_page      = bpage
            cur_idx_start = bidx
            cur_idx_end   = bidx

        # ── Append block ──────────────────────────────────────────────────
        cur_texts.append(btext)
        cur_block_objs.append(block)
        cur_tokens  += btokens
        cur_idx_end  = bidx

    # ── Final flush ───────────────────────────────────────────────────────────
    emit_chunk()

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# TABLE CHUNKER  (all strategies that include tables)
# ══════════════════════════════════════════════════════════════════════════════

def chunk_tables(
    doc:        dict,
    doc_config: dict,
    agent:      str,
    chunk_size: int,
) -> list:
    """One chunk per table.

    Caption resolution order:
        1. table.caption (if non-empty)
        2. caption-type block on the same page
        3. table.table_id (final fallback)

    Markdown is pre-rendered — truncated to chunk_size tokens for embedding.
    Full records + columns preserved in payload.
    """
    tables = doc.get("tables", [])
    if not tables:
        return []

    caption_lookup = build_caption_lookup(doc.get("blocks", []))
    meta           = base_metadata(doc, doc_config, agent)
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
            "table_id":    table["table_id"],
            "table_index": table["table_index"],
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
    agent:      str,
    chunk_min:  int,
    chunk_max:  int,
) -> list:
    """Route a document to the correct chunking strategy and return all chunks.

    Strategy mapping:
        narrative       — text (split on overflow + heading change) + tables
        legal           — text (one heading section = one chunk, no overflow split) + tables
        audit_findings  — same as legal (finding boundaries via heading blocks)
        tables_only     — tables only, all text skipped
        hybrid          — text (narrative) + tables

    chunk_size is clamped between chunk_min and chunk_max from pipeline config.
    """
    strategy   = doc_config.get("chunking_strategy", "narrative")
    chunk_size = doc_config.get("chunk_size", 350)
    chunk_size = max(chunk_min, min(chunk_max, chunk_size))

    chunks = []

    if strategy == "narrative":
        chunks += chunk_text_blocks(doc, doc_config, agent, chunk_size, split_on_overflow=True)
        chunks += chunk_tables(doc, doc_config, agent, chunk_size)

    elif strategy == "legal":
        chunks += chunk_text_blocks(doc, doc_config, agent, chunk_size, split_on_overflow=False)
        chunks += chunk_tables(doc, doc_config, agent, chunk_size)

    elif strategy == "audit_findings":
        chunks += chunk_text_blocks(doc, doc_config, agent, chunk_size, split_on_overflow=False)
        chunks += chunk_tables(doc, doc_config, agent, chunk_size)

    elif strategy == "tables_only":
        chunks += chunk_tables(doc, doc_config, agent, chunk_size)

    elif strategy == "hybrid":
        chunks += chunk_text_blocks(doc, doc_config, agent, chunk_size, split_on_overflow=True)
        chunks += chunk_tables(doc, doc_config, agent, chunk_size)

    else:
        print(f"  WARNING: unknown strategy '{strategy}' — falling back to narrative")
        chunks += chunk_text_blocks(doc, doc_config, agent, chunk_size, split_on_overflow=True)
        chunks += chunk_tables(doc, doc_config, agent, chunk_size)

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Config-aware document chunker — Kenya AI Executive Roundtable"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to agent config.yaml  (e.g. Finance/config.yaml)",
    )
    parser.add_argument(
        "--input", required=True,
        help="Directory of parsed cache JSONs  (e.g. Finance/data/cache/)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Directory for output JSONL files  (e.g. Finance/data/chunks/)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-chunk even if output JSONL already exists",
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
    pipeline, doc_lookup = load_config(config_path)
    agent     = pipeline["agent"]
    chunk_min = pipeline.get("chunk_min_tokens", 100)
    chunk_max = pipeline.get("chunk_max_tokens", 500)

    print(f"Agent          : {agent}")
    print(f"Config entries : {len(doc_lookup)}")
    print(f"Cache dir      : {cache_dir}")
    print(f"Output dir     : {output_dir}")
    print(f"Token range    : {chunk_min}–{chunk_max}")
    print(f"Force re-chunk : {args.force}")
    print()

    # ── Build cache index ─────────────────────────────────────────────────────
    cache_index = build_cache_index(cache_dir)
    print(f"Cache files found: {len(cache_index)}\n")

    # ── Process documents ─────────────────────────────────────────────────────
    total_chunks      = 0
    processed         = []
    skipped_no_index  = []
    skipped_no_cache  = []
    skipped_exists    = []
    skipped_error     = []
    skipped_collision = []
    used_cache_files  = {}   # {str(cache_path): first_claiming_fname}

    for slug, doc_config in doc_lookup.items():
        fname    = doc_config["filename"]
        strategy = doc_config.get("chunking_strategy", "narrative")

        # ── index: false → skip entirely ─────────────────────────────────
        if not doc_config.get("index", True):
            skipped_no_index.append(fname)
            continue

        # ── No cache file → warn and skip ────────────────────────────────
        cache_file = find_cache_file(cache_index, slug)
        if cache_file is None:
            print(f"  SKIP (no cache)  : {fname}")
            skipped_no_cache.append(fname)
            continue

        # ── Collision detection ───────────────────────────────────────────
        # One physical cache file can only serve one config entry.
        # If two slugs resolve to the same file, the parser truncated the
        # filename and only one PDF was actually parsed. Skip the extras —
        # they need to be parsed separately before they can be chunked.
        cache_key = str(cache_file)
        if cache_key in used_cache_files:
            first = used_cache_files[cache_key]
            print(f"  SKIP (collision) : {fname}  →  cache claimed by {Path(first).name}")
            skipped_collision.append(fname)
            continue
        used_cache_files[cache_key] = fname

        # ── Already chunked and not forcing ──────────────────────────────
        out_file = output_dir / f"{slug}.jsonl"
        if out_file.exists() and not args.force:
            skipped_exists.append(slug)
            continue

        # ── Load parsed doc ───────────────────────────────────────────────
        with open(cache_file, encoding="utf-8") as f:
            doc = json.load(f)

        if doc.get("error"):
            print(f"  SKIP (parse err) : {fname}  →  {doc['error']}")
            skipped_error.append(fname)
            continue

        # ── Chunk ─────────────────────────────────────────────────────────
        chunks = chunk_document(doc, doc_config, agent, chunk_min, chunk_max)
        n      = len(chunks)

        # ── Write JSONL ───────────────────────────────────────────────────
        with open(out_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        print(f"  ✓ {n:>4} chunks  [{strategy:<15}]  {slug}.jsonl")
        total_chunks += n
        processed.append((slug, n, strategy))

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"Processed        : {len(processed):>4} documents → {total_chunks} total chunks")
    print(f"Skipped (exists) : {len(skipped_exists):>4}  (use --force to re-chunk)")
    print(f"Skipped (no idx) : {len(skipped_no_index):>4}  (index: false in config)")
    print(f"Skipped (errors) : {len(skipped_error):>4}  (parser errors)")
    print(f"Skipped (no cach): {len(skipped_no_cache):>4}  (not yet parsed)")
    print(f"Skipped (collide): {len(skipped_collision):>4}  (parser truncation — need separate parse)")

    if skipped_no_cache:
        print()
        print("Not yet parsed:")
        for fname in sorted(skipped_no_cache):
            print(f"  - {fname}")

    if skipped_collision:
        print()
        print("Cache collisions — these PDFs need to be parsed separately:")
        for fname in sorted(skipped_collision):
            print(f"  - {fname}")

    print("=" * 65)

    if processed:
        print()
        strat_counts   = Counter(s for _, _, s in processed)
        chunk_by_strat = {}
        for _, n, s in processed:
            chunk_by_strat[s] = chunk_by_strat.get(s, 0) + n
        print("Strategy breakdown:")
        for strat, doc_count in sorted(strat_counts.items()):
            print(f"  {strat:<18} : {doc_count:>3} docs  →  {chunk_by_strat[strat]:>5} chunks")


if __name__ == "__main__":
    main()