"""
chunk_documents.py
──────────────────
Universal chunker for the Kenya AI Executive Roundtable RAG pipeline.

Reads the universal config.yaml (generated from central_inventory.csv via
generate_config.py) and applies per-document chunking strategy, chunk_size,
chunk_overlap, skip_sections filtering, and rich metadata — including the
agent_access array for Qdrant payload-based RBAC filtering.

Architecture
────────────
  - One Qdrant collection  : kenya_executive_roundtable
  - RBAC                   : agent_access[] array in every chunk payload
  - 438 docs               : one flat cache dir, one config.yaml
  - Output                 : one JSONL per doc in data/chunks/
  - config.yaml            : generated from central_inventory.csv — never edit manually

Cache format  (data/processed/<stem>.json — native DoclingDocument v1.10+)
──────────────────────────────────────────────────────────────────────────
  schema_name  : "DoclingDocument"
  version      : "1.10.0"

  texts[]:                         ← all text content lives here
    self_ref       — "#/texts/N"
    label          — "text" | "list_item" | "section_header" |
                     "caption" | "page_header" | "page_footer" | ...
    text           — str            ← the actual text
    orig           — str            ← pre-normalisation original
    prov[]         — [{"page_no": int, "bbox": {...}, ...}]
    level          — int            ← heading level (section_header only)

  tables[]:
    label          — str  e.g. "document_index" (table of contents → skip)
    prov[]         — [{"page_no": int, ...}]
    captions[]     — list of $ref objects (resolved via texts[])
    data:
      grid[][]     — row-major list of cell dicts
                     {"text": str, "column_header": bool, ...}
      table_cells[]— flat fallback list
      num_rows     — int
      num_cols     — int

Heading inference
─────────────────
  Native DoclingDocument does not pre-compute heading_path on text items.
  We infer it by tracking the last seen section_header as we iterate texts[]
  in document order.  Section headers are NOT added to chunk text — they
  serve only as section boundary markers and chunk label prefixes.

Labels used
───────────
  CONTENT_LABELS   — text items included in prose chunks
  HEADING_LABEL    — triggers a heading-context update
  SKIP_LABELS      — discarded entirely (running headers/footers, pictures)

Chunking strategies
───────────────────
  narrative       text (overflow split + heading cut) + tables
  legal           text (one heading section = one chunk, no overflow) + tables
  audit_findings  same as legal  (finding boundaries via headings)
  tables_only     tables only, all text blocks skipped
  hybrid          text (narrative rules) + tables

Usage
─────
  # Full corpus
  python chunk_documents.py \\
      --config  /home/darwin/PRES/Executive/config.yaml \\
      --input   /home/darwin/PRES/Executive/data/processed/ \\
      --output  /home/darwin/PRES/Executive/data/chunks/

  # Re-chunk even if output JSONL already exists
  python chunk_documents.py ... --force

  # Single document by doc_slug
  python chunk_documents.py ... --slug finance_2023.budget.policy.statement

  # All documents whose primary_agents contains a given agent
  python chunk_documents.py ... --agent finance
"""

import argparse
import json
import re
import sys
import uuid
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
# CANONICAL STEM  (cache file ↔ config entry matching)
# ══════════════════════════════════════════════════════════════════════════════

def canonical_stem(name: str) -> str:
    stem       = Path(name).stem if name.lower().endswith(".pdf") else name
    normalized = re.sub(r"[^a-z0-9]+", "_", stem.lower())
    return normalized.strip("_")


def build_cache_index(cache_dir: Path) -> dict:
    """Return {canonical_stem: Path} for every .json file in cache_dir."""
    index = {}
    for json_file in cache_dir.glob("*.json"):
        key = canonical_stem(json_file.stem)
        if key in index:
            if len(json_file.stem) > len(index[key].stem):
                index[key] = json_file
        else:
            index[key] = json_file
    return index


def find_cache_file(cache_index: dict, source_file: str, doc_slug: str) -> Path | None:
    sf_key   = canonical_stem(source_file)
    slug_key = canonical_stem(doc_slug)
    if sf_key in cache_index:
        return cache_index[sf_key]
    if slug_key in cache_index:
        return cache_index[slug_key]
    return None


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: Path) -> tuple[dict, list]:
    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    pipeline  = config.get("pipeline", {})
    documents = config.get("documents", [])
    return pipeline, documents


# ══════════════════════════════════════════════════════════════════════════════
# SKIP-SECTION CHECKER
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_SKIP_SECTIONS: dict[str, list] = {
    "budget_policy_statement":  ["foreword", "acknowledgement", "table of contents"],
    "budget_review_outlook":    ["foreword", "acknowledgement", "table of contents"],
    "budget_summary":           [],
    "budget_speech":            [],
    "post_election_report":     ["foreword", "acknowledgement"],
    "debt_management_strategy": ["foreword", "acknowledgement"],
    "public_debt_report":       ["foreword", "acknowledgement"],
    "controller_of_budget": [
        "county breakdown", "appendix", "foreword", "acknowledgement",
    ],
    "auditor_general_report":   ["foreword", "table of contents"],
    "audit_report":             ["foreword", "table of contents"],
    "forensic_audit":           ["foreword", "table of contents"],
    "financial_statements": [
        "foreword", "table of contents", "notes to financial statements",
    ],
    "cbk_annual_report": [
        "directors report", "directors' report", "financial statements",
        "statement of financial position", "income statement",
        "cash flow statement", "notes to financial statements",
        "staff costs", "human resources", "corporate governance", "board committees",
    ],
    "cbk_mpc_report":          ["foreword", "table of contents"],
    "cbk_fsr_report":          ["foreword", "table of contents"],
    "imf_report":              ["foreword", "acknowledgement"],
    "world_bank_report":       ["foreword", "acknowledgement"],
    "economic_survey":         ["foreword", "acknowledgement"],
    "kra_revenue_performance": ["foreword", "acknowledgement"],
    "kra_corporate_plan":      ["foreword", "acknowledgement"],
    "tax_expenditure_report":  ["foreword", "acknowledgement"],
    "eacc_report":             ["foreword", "acknowledgement", "table of contents"],
    "ppra_report":             ["foreword", "acknowledgement", "table of contents"],
    "odpp_report":             ["foreword", "acknowledgement", "table of contents"],
    "mer_report":              ["foreword", "acknowledgement", "table of contents"],
    "strategic_plan":          ["foreword", "acknowledgement", "table of contents"],
    "masterplan":              ["foreword", "acknowledgement", "table of contents"],
    "annual_report":           ["foreword", "acknowledgement", "table of contents"],
    "policy":                  ["foreword", "acknowledgement"],
    "sector_report":           ["foreword", "acknowledgement", "table of contents"],
    "igf_report":              ["foreword", "acknowledgement", "table of contents"],
    "statistics_report":       ["foreword", "table of contents"],
    "research_report":         ["foreword", "acknowledgement"],
    "survey":                  ["foreword", "acknowledgement"],
    "guidelines":              ["foreword", "acknowledgement"],
    "manual":                  ["foreword", "acknowledgement", "table of contents"],
    "framework":               ["foreword", "acknowledgement", "table of contents"],
    "assessment":              ["foreword", "acknowledgement"],
    "conference_report":       ["foreword", "acknowledgement", "table of contents"],
    "magazine": [
        "table of contents", "advertisement", "editor's note",
        "from the desk", "letters to the editor",
    ],
}


def get_skip_sections(doc_config: dict) -> list:
    if "skip_sections" in doc_config:
        return doc_config["skip_sections"] or []
    doc_type = doc_config.get("document_type", "unknown")
    return DEFAULT_SKIP_SECTIONS.get(doc_type, [])


def make_skip_checker(skip_sections: list):
    skip_lower = [s.lower() for s in skip_sections]

    def should_skip(heading: str) -> bool:
        if not skip_lower:
            return False
        h = heading.lower()
        return any(s in h for s in skip_lower)

    return should_skip


# ══════════════════════════════════════════════════════════════════════════════
# BASE METADATA
# ══════════════════════════════════════════════════════════════════════════════

def base_metadata(doc_config: dict) -> dict:
    topics       = doc_config.get("topics")       or []
    agent_access = doc_config.get("agent_access") or []
    primary      = doc_config.get("primary_agents") or []

    return {
        "source_file":      doc_config.get("source_file",   ""),
        "doc_id":           doc_config.get("doc_id",         ""),
        "doc_slug":         doc_config.get("doc_slug",       ""),
        "agent_access":     sorted(agent_access),
        "primary_agents":   sorted(primary),
        "issuing_agent":    doc_config.get("issuing_agent",  "unknown"),
        "document_type":    doc_config.get("document_type",  "unknown"),
        "domain":           doc_config.get("domain",         "unknown"),
        "topics":           sorted(topics),
        "fiscal_year":      str(doc_config.get("fiscal_year", "na")),
        "doc_year":         doc_config.get("doc_year",       None),
        "report_period":    doc_config.get("report_period",  "annual"),
        "priority":         doc_config.get("priority",       "medium"),
        "rag_weight":       float(doc_config.get("rag_weight", 1.0)),
        "category":         doc_config.get("category",       0),
        "is_scanned":       bool(doc_config.get("is_scanned",     False)),
        "language":         doc_config.get("language",        "english"),
        "geographic_scope": doc_config.get("geographic_scope", "national"),
        "superseded":       bool(doc_config.get("superseded",     False)),
        "chunk_strategy":   doc_config.get("chunking_strategy", "narrative"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# NATIVE DOCLING DOCUMENT — TEXT ITEM HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# Labels treated as prose content (added to chunk text)
CONTENT_LABELS = {"text", "list_item", "caption"}

# Label that updates the current heading context (not added to chunk text)
HEADING_LABEL  = "section_header"

# Labels discarded entirely — running headers/footers add noise to embeddings
SKIP_LABELS    = {"page_header", "page_footer", "picture", "formula"}


def _text_page(item: dict) -> int:
    """Extract page number from a texts[] item's prov list."""
    prov = item.get("prov") or []
    if prov:
        return prov[0].get("page_no", 1)
    return 1


def build_caption_lookup(doc: dict) -> dict:
    """
    Build {page_number: caption_text} from texts[] items with label 'caption'.

    Keeps the first caption per page (Docling top-to-bottom ordering).
    Used in table chunker when a table's captions[] list is empty.
    """
    lookup: dict[int, str] = {}
    for item in doc.get("texts", []):
        if item.get("label") == "caption":
            page = _text_page(item)
            text = item.get("text", "").strip()
            if text and page not in lookup:
                lookup[page] = text
    return lookup


# ══════════════════════════════════════════════════════════════════════════════
# TEXT FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

def format_chunk_text(texts: list, heading: str) -> str:
    """
    Join block texts with blank-line separators, prefixed with [heading].

        [1.2 Fiscal Policy Outlook]

        In FY 2023/24, the fiscal deficit stood at 5.4% of GDP ...
    """
    body = "\n\n".join(text for text in texts if text)
    return f"[{heading}]\n\n{body}" if heading else body


# ══════════════════════════════════════════════════════════════════════════════
# TEXT BLOCK CHUNKER  (native DoclingDocument)
# ══════════════════════════════════════════════════════════════════════════════

def chunk_text_blocks(
    doc:               dict,
    doc_config:        dict,
    meta:              dict,
    chunk_size:        int,
    chunk_overlap:     int,
    split_on_overflow: bool,
) -> list:
    """
    Chunk prose text from a native DoclingDocument (texts[] array).

    Heading inference
    ─────────────────
    Native DoclingDocument does not pre-compute heading_path.  We track the
    last seen section_header label as we iterate texts[] in document order.
    A new section_header always flushes the current accumulator and updates
    the heading context — identical semantics to the old heading_changed cut.

    Flush rules (unchanged from original)
    ──────────────────────────────────────
    Hard cut  — section_header encountered while accumulator is non-empty.
    Soft cut  — token overflow (narrative/hybrid only); carry last item as
                single-block overlap.
    Legal     — split_on_overflow=False; one section = one chunk regardless
                of length.
    """
    texts_list = doc.get("texts", [])
    skip        = make_skip_checker(get_skip_sections(doc_config))
    chunks      = []

    # ── Accumulator state ─────────────────────────────────────────────────────
    cur_texts:      list[str]  = []
    cur_item_objs:  list[dict] = []
    cur_tokens:     int        = 0
    cur_heading:    str        = ""
    cur_page:       int        = 1
    cur_idx_start:  int        = 0
    cur_idx_end:    int        = 0

    def emit_chunk() -> None:
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
    for idx, item in enumerate(texts_list):
        label = item.get("label", "")

        # ── Update heading context on section_header ──────────────────────────
        if label == HEADING_LABEL:
            heading_text = item.get("text", "").strip()
            if not heading_text:
                continue
            # Flush accumulator before switching section
            if cur_texts:
                emit_chunk()
                cur_texts     = []
                cur_item_objs = []
                cur_tokens    = 0
            # Only update heading if this section is not in the skip list
            if not skip(heading_text):
                cur_heading   = heading_text
                cur_page      = _text_page(item)
                cur_idx_start = idx
                cur_idx_end   = idx
            else:
                # Entering a skipped section — blank the heading so content
                # from the next section doesn't inherit the skipped label
                cur_heading = ""
            continue

        # ── Discard noise labels ──────────────────────────────────────────────
        if label in SKIP_LABELS:
            continue

        # ── Only process content labels ───────────────────────────────────────
        if label not in CONTENT_LABELS:
            continue

        # ── Skip if current heading is in the skip list ───────────────────────
        if skip(cur_heading):
            continue

        itext = item.get("text", "").strip()
        if not itext:
            continue

        itokens = count_tokens(itext)
        ipage   = _text_page(item)

        # ── Soft cut: token overflow ──────────────────────────────────────────
        token_overflow = (
            split_on_overflow
            and bool(cur_texts)
            and (cur_tokens + itokens > chunk_size)
        )

        if token_overflow:
            overlap_item = cur_item_objs[-1] if cur_item_objs else None

            emit_chunk()
            cur_texts     = []
            cur_item_objs = []
            cur_tokens    = 0

            if overlap_item is not None:
                ob_text   = overlap_item.get("text", "").strip()
                cur_texts     = [ob_text]
                cur_item_objs = [overlap_item]
                cur_tokens    = count_tokens(ob_text)
                cur_page      = _text_page(overlap_item)
                cur_idx_start = texts_list.index(overlap_item) if overlap_item in texts_list else idx
                cur_idx_end   = cur_idx_start

        # ── Initialise accumulator on first item or after any flush ───────────
        if not cur_texts:
            cur_page      = ipage
            cur_idx_start = idx
            cur_idx_end   = idx

        cur_texts.append(itext)
        cur_item_objs.append(item)
        cur_tokens  += itokens
        cur_idx_end  = idx

    # ── Final flush ───────────────────────────────────────────────────────────
    emit_chunk()
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# TABLE RENDERING  (Docling grid → markdown)
# ══════════════════════════════════════════════════════════════════════════════

def render_table_markdown(table: dict) -> str:
    """
    Render a Docling table object to a markdown string.

    Docling stores tables in data['grid'] (row-major list of cell dicts).
    Each cell dict has at minimum: {"text": str, "column_header": bool}.
    Falls back to data['table_cells'] (flat list) if grid is absent.
    """
    data = table.get("data", {})
    grid = data.get("grid")

    if grid:
        rows: list[list[str]] = []
        for row_cells in grid:
            seen:  set       = set()
            cells: list[str] = []
            for cell in row_cells:
                txt = cell.get("text", "").strip()
                if txt not in seen:
                    seen.add(txt)
                    cells.append(txt)
            rows.append(cells)

        if not rows:
            return ""

        num_cols  = max(len(row) for row in rows)
        first_row = rows[0]
        is_header = any(
            cell.get("column_header", False)
            for cell in (grid[0] if grid else [])
        )

        lines: list[str] = []
        if is_header:
            lines.append("| " + " | ".join(first_row) + " |")
            lines.append("| " + " | ".join(["---"] * len(first_row)) + " |")
            data_rows = rows[1:]
        else:
            lines.append("| " + " | ".join([""] * num_cols) + " |")
            lines.append("| " + " | ".join(["---"] * num_cols) + " |")
            data_rows = rows

        for row in data_rows:
            padded = row + [""] * (num_cols - len(row))
            lines.append("| " + " | ".join(padded) + " |")

        return "\n".join(lines)

    # Fallback: flat table_cells
    cells = data.get("table_cells", [])
    texts = [cell.get("text", "").strip() for cell in cells if cell.get("text", "").strip()]
    return "\n".join(texts)


# ══════════════════════════════════════════════════════════════════════════════
# TABLE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def extract_table_page(table: dict) -> int:
    prov = table.get("prov", [])
    if prov:
        return prov[0].get("page_no", 1)
    return 1


def resolve_caption_from_refs(table: dict, doc: dict) -> str:
    """
    Resolve caption text from a table's captions[] $ref list.

    DoclingDocument captions[] contains objects like {"$ref": "#/texts/42"}.
    We resolve each ref into the texts[] array and join any text found.
    Falls back to empty string if refs are absent or unresolvable.
    """
    captions = table.get("captions", [])
    if not captions:
        return ""

    texts_list = doc.get("texts", [])
    parts = []
    for cap_ref in captions:
        ref = cap_ref.get("$ref", "")
        # "#/texts/42" → index 42
        match = re.match(r"#/texts/(\d+)$", ref)
        if match:
            text_idx = int(match.group(1))
            if 0 <= text_idx < len(texts_list):
                text = texts_list[text_idx].get("text", "").strip()
                if text:
                    parts.append(text)
    return " ".join(parts)


def extract_table_caption(
    table:          dict,
    doc:            dict,
    caption_lookup: dict,
    table_idx:      int,
) -> str:
    """
    Resolve table caption with three-level fallback:
        1. $ref resolution from table's captions[] list  (most accurate)
        2. Page-level caption_lookup from texts[]         (proximity match)
        3. Generated fallback: "Table N"
    """
    ref_caption = resolve_caption_from_refs(table, doc)
    if ref_caption:
        return ref_caption

    page = extract_table_page(table)
    if page in caption_lookup:
        return caption_lookup[page]

    return f"Table {table_idx + 1}"


def extract_table_columns(table: dict) -> list:
    grid = table.get("data", {}).get("grid", [])
    if not grid:
        return []
    first_row = grid[0]
    if any(cell.get("column_header", False) for cell in first_row):
        return [cell.get("text", "").strip() for cell in first_row]
    return []


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
    Produce one chunk per table.

    Skips tables labelled "document_index" (table of contents).
    Skips tables whose rendered markdown is empty after truncation.
    Caption is prepended for embedding context.
    """
    tables = doc.get("tables", [])
    if not tables:
        return []

    caption_lookup = build_caption_lookup(doc)
    doc_slug       = meta.get("doc_slug", "unknown")
    chunks         = []

    for idx, table in enumerate(tables):
        if table.get("label") == "document_index":
            continue

        page     = extract_table_page(table)
        caption  = extract_table_caption(table, doc, caption_lookup, idx)
        table_id = f"{doc_slug}_table_{idx:03d}"

        markdown = render_table_markdown(table)
        if not markdown.strip():
            continue

        markdown = truncate_to_tokens(markdown, chunk_size)

        parts = [part for part in [caption, markdown] if part]
        text  = "\n\n".join(parts)

        chunks.append({
            "chunk_id":    str(uuid.uuid4()),
            "chunk_type":  "table",
            "text":        text,
            "heading_path": "",
            "page_number": page,
            "table_id":    table_id,
            "table_index": idx,
            "caption":     caption,
            "num_rows":    table.get("data", {}).get("num_rows", 0),
            "num_cols":    table.get("data", {}).get("num_cols", 0),
            "columns":     extract_table_columns(table),
            **meta,
        })

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY ROUTER
# ══════════════════════════════════════════════════════════════════════════════

VALID_STRATEGIES = {"narrative", "legal", "audit_findings", "tables_only", "hybrid"}


def chunk_document(
    doc:        dict,
    doc_config: dict,
    chunk_min:  int,
    chunk_max:  int,
) -> list:
    strategy      = doc_config.get("chunking_strategy", "narrative")
    chunk_size    = int(doc_config.get("chunk_size",    350))
    chunk_size    = max(chunk_min, min(chunk_max, chunk_size))
    chunk_overlap = int(doc_config.get("chunk_overlap", 88))

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
        print(f"    WARNING: unknown strategy '{strategy}' — falling back to narrative")
        chunks += chunk_text_blocks(doc, doc_config, meta, chunk_size, chunk_overlap, split_on_overflow=True)
        chunks += chunk_tables(doc, doc_config, meta, chunk_size)

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Universal document chunker — Kenya AI Executive Roundtable"
    )
    parser.add_argument("--config",  required=True, help="Path to universal config.yaml")
    parser.add_argument("--input",   required=True, help="Directory of DoclingDocument JSONs (data/processed/)")
    parser.add_argument("--output",  required=True, help="Directory for output JSONL files (data/chunks/)")
    parser.add_argument("--force",   action="store_true", help="Re-chunk even if output JSONL already exists")
    parser.add_argument("--slug",    default=None,  help="Process only the document matching this doc_slug")
    parser.add_argument("--agent",   default=None,  help="Process only documents where AGENT is in primary_agents")
    args = parser.parse_args()

    config_path = Path(args.config)
    cache_dir   = Path(args.input)
    output_dir  = Path(args.output)

    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        sys.exit(1)
    if not cache_dir.is_dir():
        print(f"ERROR: cache directory not found: {cache_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline, documents = load_config(config_path)
    chunk_min  = int(pipeline.get("chunk_min_tokens", 100))
    chunk_max  = int(pipeline.get("chunk_max_tokens", 500))
    collection = pipeline.get("qdrant_collection", "kenya_executive_roundtable")

    print()
    print(f"Collection      : {collection}")
    print(f"Config entries  : {len(documents)}")
    print(f"Cache dir       : {cache_dir}")
    print(f"Output dir      : {output_dir}")
    print(f"Token range     : {chunk_min}–{chunk_max}")
    print(f"Force re-chunk  : {args.force}")

    cache_index = build_cache_index(cache_dir)
    print(f"Cache files     : {len(cache_index)}")

    if args.slug:
        documents = [d for d in documents if d.get("doc_slug") == args.slug]
        if not documents:
            print(f"\nERROR: slug '{args.slug}' not found in config")
            sys.exit(1)
        print(f"Single slug     : {args.slug}")

    if args.agent:
        documents = [
            d for d in documents
            if args.agent in (d.get("primary_agents") or [])
        ]
        print(f"Agent filter    : {args.agent} → {len(documents)} documents")

    print()

    total_chunks      = 0
    processed:        list[tuple[str, int, str]] = []
    skipped_hard:     list[str] = []
    skipped_no_cache: list[str] = []
    skipped_exists:   list[str] = []
    skipped_error:    list[str] = []
    skipped_collide:  list[str] = []

    used_cache_files: dict[str, str] = {}

    for doc_config in documents:
        slug     = doc_config.get("doc_slug", "")
        fname    = doc_config.get("source_file", slug)
        strategy = doc_config.get("chunking_strategy", "narrative")

        if doc_config.get("HARD"):
            reasons    = doc_config.get("hard_reasons", ["see config HARD flag"])
            reason_str = " | ".join(reasons)
            print(f"  SKIP (HARD)      : {fname}")
            print(f"                     {reason_str}")
            skipped_hard.append(fname)
            continue

        cache_file = find_cache_file(cache_index, fname, slug)
        if cache_file is None:
            print(f"  SKIP (no cache)  : {fname}")
            skipped_no_cache.append(fname)
            continue

        cache_key = str(cache_file)
        if cache_key in used_cache_files:
            first = used_cache_files[cache_key]
            print(f"  SKIP (collision) : {fname}  →  cache claimed by '{first}'")
            skipped_collide.append(fname)
            continue
        used_cache_files[cache_key] = fname

        out_file = output_dir / f"{slug}.jsonl"
        if out_file.exists() and not args.force:
            skipped_exists.append(slug)
            continue

        try:
            with open(cache_file, encoding="utf-8") as fh:
                doc = json.load(fh)
        except json.JSONDecodeError as err:
            print(f"  SKIP (bad JSON)  : {fname}  →  {err}")
            skipped_error.append(fname)
            continue
        except OSError as err:
            print(f"  SKIP (read err)  : {fname}  →  {err}")
            skipped_error.append(fname)
            continue

        if doc.get("error"):
            print(f"  SKIP (parse err) : {fname}  →  {doc['error']}")
            skipped_error.append(fname)
            continue

        # Sanity check: warn if this is not a native DoclingDocument
        if doc.get("schema_name") != "DoclingDocument":
            print(f"  WARN (schema)    : {fname}  →  schema_name='{doc.get('schema_name')}' "
                  f"(expected 'DoclingDocument') — attempting anyway")

        try:
            chunks = chunk_document(doc, doc_config, chunk_min, chunk_max)
        except Exception as err:
            print(f"  SKIP (chunk err) : {fname}  →  {err}")
            skipped_error.append(fname)
            continue

        n = len(chunks)

        # Warn loudly if a non-tables_only strategy produced zero text chunks
        text_chunks  = sum(1 for chunk in chunks if chunk.get("chunk_type") == "text")
        table_chunks = sum(1 for chunk in chunks if chunk.get("chunk_type") == "table")
        if strategy != "tables_only" and text_chunks == 0 and n > 0:
            print(f"  WARN (no prose)  : {fname}  →  {table_chunks} table chunks only, "
                  f"0 text chunks — check texts[] labels in cache file")

        try:
            with open(out_file, "w", encoding="utf-8") as fh:
                for chunk in chunks:
                    fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        except OSError as err:
            print(f"  SKIP (write err) : {fname}  →  {err}")
            skipped_error.append(fname)
            continue

        access_str = "|".join(sorted(doc_config.get("agent_access") or []))
        print(f"  ✓ {n:>4} chunks  [{strategy:<15}]  "
              f"[txt={text_chunks} tbl={table_chunks}]  {slug}  [{access_str}]")
        total_chunks += n
        processed.append((slug, n, strategy))

    print()
    print("=" * 72)
    print(f"Processed         : {len(processed):>4} documents → {total_chunks:,} total chunks")
    print(f"Skipped (exists)  : {len(skipped_exists):>4}  (use --force to re-chunk)")
    print(f"Skipped (HARD)    : {len(skipped_hard):>4}  (unresolved in config — fix in inventory)")
    print(f"Skipped (no cache): {len(skipped_no_cache):>4}  (not yet parsed by docling)")
    print(f"Skipped (errors)  : {len(skipped_error):>4}  (malformed JSON / parse / write errors)")
    print(f"Skipped (collide) : {len(skipped_collide):>4}  (two slugs mapped to same cache file)")

    if skipped_no_cache:
        print()
        print("Not yet parsed (run docling on these PDFs):")
        for fname in sorted(skipped_no_cache):
            print(f"  - {fname}")

    if skipped_hard:
        print()
        print("HARD-flagged (fix central_inventory.csv → regenerate config):")
        for fname in sorted(skipped_hard):
            print(f"  - {fname}")

    if skipped_collide:
        print()
        print("Cache collisions (two config entries resolved to the same file):")
        for fname in sorted(skipped_collide):
            print(f"  - {fname}")

    print("=" * 72)

    if processed:
        print()
        strat_counts:    Counter = Counter(strategy for _, _, strategy in processed)
        chunks_by_strat: dict   = {}
        for _, n, strategy in processed:
            chunks_by_strat[strategy] = chunks_by_strat.get(strategy, 0) + n

        print("Strategy breakdown:")
        for strategy, doc_count in sorted(strat_counts.items()):
            print(
                f"  {strategy:<18} : {doc_count:>3} docs  →  "
                f"{chunks_by_strat[strategy]:>6,} chunks"
            )
        print()

    if skipped_hard or skipped_error:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()