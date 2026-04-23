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

Cache format  (data/processed/<stem>.json written by Docling)
─────────────────────────────────────────────────────────────
  blocks[]:
    block_type   — "paragraph" | "heading" | "list_item" | "caption" | "table"
    heading_path — list[str]   e.g. ["1.2 Fiscal Policy Outlook"]
    text         — str
    block_index  — int
    page_number  — int

  tables[]:
    prov[]           — [{"page_no": int, ...}]
    captions[]       — list of caption-ref objects (text resolved via blocks)
    label            — str  e.g. "document_index" (table of contents → skip)
    data:
      grid[][]       — row-major list of cell dicts  {"text": str, "column_header": bool}
      table_cells[]  — flat fallback list
      num_rows       — int
      num_cols       — int

Chunking strategies
───────────────────
  narrative       text (overflow split + heading cut) + tables
  legal           text (one heading section = one chunk, no overflow) + tables
  audit_findings  same as legal  (finding boundaries via headings)
  tables_only     tables only, all text blocks skipped
  hybrid          text (narrative rules) + tables

Cache file lookup
─────────────────
  Docling normalizes PDF filenames when writing cache stems:
  all non-alphanumeric characters → underscores, runs collapsed, lowercased.
  We apply the same canonical_stem() reduction to the config source_file
  before comparing, so dots, commas, parens, em-dashes, and semicolons in
  original filenames never cause a lookup miss.

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
    """
    Reduce a filename stem or Docling cache stem to a canonical form.

    Docling normalizes PDF filenames when writing cache JSON stems:
    all non-alphanumeric characters are replaced with underscores and
    consecutive underscores are collapsed.  We apply the same reduction to
    config source_file values before comparing so that dots, commas,
    parentheses, em-dashes, and semicolons in original filenames never cause
    a lookup miss.

    Examples
    ────────
        "RDM-1.2-Traffic-Surveys"                              → "rdm_1_2_traffic_surveys"
        "CBK_16th Monetary Policy Committee Report, April 2016"→ "cbk_16th_monetary_policy_committee_report_april_2016"
        "CBK_Annual Report 2015 16 (book)"                     → "cbk_annual_report_2015_16_book"
        "2025-STRENGTHENING-OF-KANDWIA-–-KYUSO"               → "2025_strengthening_of_kandwia_kyuso"
        "First Half NGBIRR FY 23-24 - COB final 13.3.14"      → "first_half_ngbirr_fy_23_24_cob_final_13_3_14"
        "Office of the Controller of Budget;.pdf"              → "office_of_the_controller_of_budget"
    """
    stem       = Path(name).stem if name.lower().endswith(".pdf") else name
    normalized = re.sub(r"[^a-z0-9]+", "_", stem.lower())
    return normalized.strip("_")


def build_cache_index(cache_dir: Path) -> dict:
    """Return {canonical_stem: Path} for every .json file in cache_dir."""
    index = {}
    for json_file in cache_dir.glob("*.json"):
        key = canonical_stem(json_file.stem)
        if key in index:
            # Two different cache files reduce to the same canonical stem —
            # keep the one whose original stem is longer (more specific).
            if len(json_file.stem) > len(index[key].stem):
                index[key] = json_file
        else:
            index[key] = json_file
    return index


def find_cache_file(cache_index: dict, source_file: str, doc_slug: str) -> Path | None:
    """
    Resolve a config entry → cache file using canonical stem matching.

    Resolution order:
        1. Canonical source_file stem  (primary — closest to original PDF name)
        2. Canonical doc_slug          (fallback — dot-notation config slug)

    The prefix-scan fallback present in earlier versions is intentionally
    removed; canonical matching handles all real-world cases without the
    false-positive collision risk of prefix matching.
    """
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
    """
    Load universal config.yaml.

    Returns:
        pipeline  — the pipeline: section dict
        documents — list of per-document config dicts in config order
    """
    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    pipeline  = config.get("pipeline", {})
    documents = config.get("documents", [])
    return pipeline, documents


# ══════════════════════════════════════════════════════════════════════════════
# SKIP-SECTION CHECKER
# ══════════════════════════════════════════════════════════════════════════════

# Boilerplate section names per document_type.
# A config-level skip_sections list (if present) overrides these entirely.
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
    """
    Return the skip_sections list for a document.
    Config-level value (if present) takes precedence over type-level defaults.
    """
    if "skip_sections" in doc_config:
        return doc_config["skip_sections"] or []
    doc_type = doc_config.get("document_type", "unknown")
    return DEFAULT_SKIP_SECTIONS.get(doc_type, [])


def make_skip_checker(skip_sections: list):
    """
    Return a callable: True if a heading should be skipped.
    Matching is case-insensitive substring.
    """
    skip_lower = [s.lower() for s in skip_sections]

    def should_skip(heading: str) -> bool:
        if not skip_lower:
            return False
        h = heading.lower()
        return any(s in h for s in skip_lower)

    return should_skip


# ══════════════════════════════════════════════════════════════════════════════
# BASE METADATA  (embedded in every chunk payload)
# ══════════════════════════════════════════════════════════════════════════════

def base_metadata(doc_config: dict) -> dict:
    """
    Build the rich metadata dict attached to every chunk.

    agent_access is kept as a sorted list for Qdrant array filtering:
        {"key": "agent_access", "match": {"any": ["finance"]}}

    fiscal_year is always cast to str to avoid YAML int-parsing artefacts
    (PyYAML reads  2016_17  as the integer 201617).
    """
    topics       = doc_config.get("topics")       or []
    agent_access = doc_config.get("agent_access") or []
    primary      = doc_config.get("primary_agents") or []

    return {
        # Identity
        "source_file":      doc_config.get("source_file",   ""),
        "doc_id":           doc_config.get("doc_id",         ""),
        "doc_slug":         doc_config.get("doc_slug",       ""),
        # Agent routing / RBAC
        "agent_access":     sorted(agent_access),
        "primary_agents":   sorted(primary),
        "issuing_agent":    doc_config.get("issuing_agent",  "unknown"),
        # Classification
        "document_type":    doc_config.get("document_type",  "unknown"),
        "domain":           doc_config.get("domain",         "unknown"),
        "topics":           sorted(topics),
        # Temporal
        "fiscal_year":      str(doc_config.get("fiscal_year", "na")),
        "doc_year":         doc_config.get("doc_year",       None),
        "report_period":    doc_config.get("report_period",  "annual"),
        # Retrieval weights
        "priority":         doc_config.get("priority",       "medium"),
        "rag_weight":       float(doc_config.get("rag_weight", 1.0)),
        "category":         doc_config.get("category",       0),
        # Document properties
        "is_scanned":       bool(doc_config.get("is_scanned",     False)),
        "language":         doc_config.get("language",        "english"),
        "geographic_scope": doc_config.get("geographic_scope", "national"),
        "superseded":       bool(doc_config.get("superseded",     False)),
        # Chunking strategy (useful for debugging / re-processing)
        "chunk_strategy":   doc_config.get("chunking_strategy", "narrative"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CAPTION LOOKUP  (used in table chunker for caption backfill)
# ══════════════════════════════════════════════════════════════════════════════

def build_caption_lookup(blocks: list) -> dict:
    """
    Build {page_number: caption_text} from caption-type blocks.

    Keeps the first caption per page (matches Docling's top-to-bottom ordering).
    Used when a table's captions[] list is empty or unresolvable.
    """
    lookup: dict[int, str] = {}
    for block in blocks:
        if block.get("block_type") == "caption":
            page = block.get("page_number", 1)
            text = block.get("text", "").strip()
            if text and page not in lookup:
                lookup[page] = text
    return lookup


# ══════════════════════════════════════════════════════════════════════════════
# TEXT FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

def format_chunk_text(texts: list, heading: str) -> str:
    """
    Join block texts with blank-line separators, prefixed with [heading].

    Output format (mirrors Finance prototype):
        [1.2 Fiscal Policy Outlook]

        In FY 2023/24, the fiscal deficit stood at 5.4% of GDP ...

        The consolidation path targets a deficit of 4.8% ...
    """
    body = "\n\n".join(text for text in texts if text)
    return f"[{heading}]\n\n{body}" if heading else body


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
    Core text chunker shared by all text-bearing strategies.

    Flush rules
    ───────────
    Hard cut (heading_changed):
        Always flush on section boundary. No overlap carried across headings —
        different sections have independent context.

    Soft cut (token_overflow, narrative/hybrid only):
        Flush when adding the next block would exceed chunk_size.
        Carry the last block of the flushed chunk as the first block of the
        next chunk (single-block overlap rather than a byte-count window).
        This preserves inter-sentence coherence at chunk boundaries.

    Legal / audit_findings:
        split_on_overflow=False — one heading section = one chunk regardless
        of length.  Legal clauses and audit findings must not be split mid-
        section because the Qdrant reranker needs the full finding for scoring.

    Args:
        doc               — parsed Docling cache JSON
        doc_config        — this document's config entry
        meta              — pre-built base_metadata dict (shared across chunks)
        chunk_size        — max tokens per chunk (already clamped to pipeline range)
        chunk_overlap     — carried overlap token budget (informational; enforced
                            via single-block carry, not a byte window)
        split_on_overflow — True for narrative/hybrid, False for legal/audit_findings
    """
    blocks = [
        block for block in doc.get("blocks", [])
        if block.get("block_type") in CONTENT_BLOCK_TYPES
    ]

    skip   = make_skip_checker(get_skip_sections(doc_config))
    chunks = []

    # ── Accumulator state ─────────────────────────────────────────────────────
    cur_texts:      list[str]  = []
    cur_block_objs: list[dict] = []
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
    for block in blocks:
        heading = (
            block["heading_path"][0]
            if block.get("heading_path") and isinstance(block["heading_path"], list)
            else ""
        )

        if skip(heading):
            continue

        btext = block.get("text", "").strip()
        if not btext:
            continue

        btokens = count_tokens(btext)
        bidx    = block.get("block_index", 0)
        bpage   = block.get("page_number", 1)

        heading_changed = bool(cur_texts) and (heading != cur_heading)
        token_overflow  = (
            split_on_overflow
            and bool(cur_texts)
            and (cur_tokens + btokens > chunk_size)
        )

        # ── Hard cut: heading change — no overlap across section boundaries ───
        if heading_changed:
            emit_chunk()
            cur_texts      = []
            cur_block_objs = []
            cur_tokens     = 0

        # ── Soft cut: token overflow — carry last block as overlap ────────────
        elif token_overflow:
            overlap_block = cur_block_objs[-1] if cur_block_objs else None

            emit_chunk()
            cur_texts      = []
            cur_block_objs = []
            cur_tokens     = 0

            if overlap_block is not None:
                ob_text    = overlap_block.get("text", "").strip()
                ob_heading = (
                    overlap_block["heading_path"][0]
                    if overlap_block.get("heading_path")
                    and isinstance(overlap_block["heading_path"], list)
                    else ""
                )
                cur_texts      = [ob_text]
                cur_block_objs = [overlap_block]
                cur_tokens     = count_tokens(ob_text)
                cur_heading    = ob_heading
                cur_page       = overlap_block.get("page_number", 1)
                cur_idx_start  = overlap_block.get("block_index", 0)
                cur_idx_end    = overlap_block.get("block_index", 0)

        # ── Initialise accumulator on first block or after any flush ──────────
        if not cur_texts:
            cur_heading   = heading
            cur_page      = bpage
            cur_idx_start = bidx
            cur_idx_end   = bidx

        cur_texts.append(btext)
        cur_block_objs.append(block)
        cur_tokens  += btokens
        cur_idx_end  = bidx

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

    Header detection: if any cell in the first row has column_header=True,
    the first row becomes the markdown table header row.
    """
    data = table.get("data", {})
    grid = data.get("grid")

    if grid:
        # ── Build clean row list ──────────────────────────────────────────────
        rows: list[list[str]] = []
        for row_cells in grid:
            # Docling repeats spanning cells across grid positions — deduplicate
            # within each row while preserving order.
            seen:  set         = set()
            cells: list[str]   = []
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
            # First row is the header
            lines.append("| " + " | ".join(first_row) + " |")
            lines.append("| " + " | ".join(["---"] * len(first_row)) + " |")
            data_rows = rows[1:]
        else:
            # No header — emit a blank header so the markdown is valid
            lines.append("| " + " | ".join([""] * num_cols) + " |")
            lines.append("| " + " | ".join(["---"] * num_cols) + " |")
            data_rows = rows

        for row in data_rows:
            padded = row + [""] * (num_cols - len(row))
            lines.append("| " + " | ".join(padded) + " |")

        return "\n".join(lines)

    # ── Fallback: flat table_cells ────────────────────────────────────────────
    cells = data.get("table_cells", [])
    texts = [cell.get("text", "").strip() for cell in cells if cell.get("text", "").strip()]
    return "\n".join(texts)


# ══════════════════════════════════════════════════════════════════════════════
# TABLE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def extract_table_page(table: dict) -> int:
    """Extract page number from Docling prov list."""
    prov = table.get("prov", [])
    if prov:
        return prov[0].get("page_no", 1)
    return 1


def extract_table_caption(
    table:          dict,
    caption_lookup: dict,
    table_idx:      int,
) -> str:
    """
    Resolve table caption.

    Resolution order:
        1. caption-type block on same page (from blocks[] via caption_lookup)
        2. Generated fallback: "Table N"

    Note: Docling's captions[] list contains $ref objects pointing into the
    document body, not embedded text. The caption text is already captured in
    the blocks[] array as a "caption" block_type, so we use the page-level
    lookup which is simpler and equivalent.
    """
    page = extract_table_page(table)
    if page in caption_lookup:
        return caption_lookup[page]
    return f"Table {table_idx + 1}"


def extract_table_columns(table: dict) -> list:
    """
    Extract column names from the first row of the grid if it is a header row.
    Returns an empty list if no header row is detected.
    """
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

    Skips tables that Docling labelled "document_index" (table of contents).
    Skips tables whose rendered markdown is empty after truncation.

    The caption is prepended to the markdown text so the embedding model
    has semantic context for the numbers in the table.
    """
    tables = doc.get("tables", [])
    if not tables:
        return []

    caption_lookup = build_caption_lookup(doc.get("blocks", []))
    doc_slug       = meta.get("doc_slug", "unknown")
    chunks         = []

    for idx, table in enumerate(tables):
        # Skip Docling table-of-contents tables
        if table.get("label") == "document_index":
            continue

        page     = extract_table_page(table)
        caption  = extract_table_caption(table, caption_lookup, idx)
        table_id = f"{doc_slug}_table_{idx:03d}"

        markdown = render_table_markdown(table)
        if not markdown.strip():
            continue

        markdown = truncate_to_tokens(markdown, chunk_size)

        # Build chunk text: caption then markdown
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
    """
    Route a document to the correct chunking strategy and return all chunks.

    chunk_size is clamped to [chunk_min, chunk_max] from the pipeline section.
    chunk_overlap is taken directly from config (pre-computed by generate_config.py).
    """
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
    parser.add_argument(
        "--config", required=True,
        help="Path to universal config.yaml",
    )
    parser.add_argument(
        "--input", required=True,
        help="Directory of parsed Docling cache JSONs (data/processed/)",
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
        help="Process only the document matching this doc_slug",
    )
    parser.add_argument(
        "--agent", default=None,
        help="Process only documents where AGENT is in primary_agents",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    cache_dir   = Path(args.input)
    output_dir  = Path(args.output)

    # ── Existence checks ──────────────────────────────────────────────────────
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        sys.exit(1)
    if not cache_dir.is_dir():
        print(f"ERROR: cache directory not found: {cache_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config ───────────────────────────────────────────────────────────
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

    # ── Build cache index ─────────────────────────────────────────────────────
    cache_index = build_cache_index(cache_dir)
    print(f"Cache files     : {len(cache_index)}")

    # ── Apply filters ─────────────────────────────────────────────────────────
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

    # ── Process documents ─────────────────────────────────────────────────────
    total_chunks     = 0
    processed:       list[tuple[str, int, str]] = []
    skipped_hard:    list[str] = []
    skipped_no_cache:list[str] = []
    skipped_exists:  list[str] = []
    skipped_error:   list[str] = []
    skipped_collide: list[str] = []

    # Collision guard: one physical cache file → one config entry
    used_cache_files: dict[str, str] = {}   # {str(cache_path): first_claiming_slug}

    for doc_config in documents:
        slug     = doc_config.get("doc_slug", "")
        fname    = doc_config.get("source_file", slug)
        strategy = doc_config.get("chunking_strategy", "narrative")

        # ── Skip HARD-flagged documents ───────────────────────────────────────
        # HARD flag is set by generate_config.py for unresolved classification,
        # invalid doc_type, empty agent_access, doc_year anomalies, etc.
        if doc_config.get("HARD"):
            reasons = doc_config.get("hard_reasons", ["see config HARD flag"])
            reason_str = " | ".join(reasons)
            print(f"  SKIP (HARD)      : {fname}")
            print(f"                     {reason_str}")
            skipped_hard.append(fname)
            continue

        # ── Locate cache file ─────────────────────────────────────────────────
        cache_file = find_cache_file(cache_index, fname, slug)

        if cache_file is None:
            print(f"  SKIP (no cache)  : {fname}")
            skipped_no_cache.append(fname)
            continue

        # ── Collision detection ───────────────────────────────────────────────
        cache_key = str(cache_file)
        if cache_key in used_cache_files:
            first = used_cache_files[cache_key]
            print(f"  SKIP (collision) : {fname}  →  cache claimed by '{first}'")
            skipped_collide.append(fname)
            continue
        used_cache_files[cache_key] = fname

        # ── Skip if already chunked and not forcing ───────────────────────────
        out_file = output_dir / f"{slug}.jsonl"
        if out_file.exists() and not args.force:
            skipped_exists.append(slug)
            continue

        # ── Load cache JSON ───────────────────────────────────────────────────
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

        # ── Chunk ─────────────────────────────────────────────────────────────
        try:
            chunks = chunk_document(doc, doc_config, chunk_min, chunk_max)
        except Exception as err:
            print(f"  SKIP (chunk err) : {fname}  →  {err}")
            skipped_error.append(fname)
            continue

        n = len(chunks)

        # ── Write JSONL ───────────────────────────────────────────────────────
        try:
            with open(out_file, "w", encoding="utf-8") as fh:
                for chunk in chunks:
                    fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        except OSError as err:
            print(f"  SKIP (write err) : {fname}  →  {err}")
            skipped_error.append(fname)
            continue

        access_str = "|".join(sorted(doc_config.get("agent_access") or []))
        print(f"  ✓ {n:>4} chunks  [{strategy:<15}]  {slug}  [{access_str}]")
        total_chunks += n
        processed.append((slug, n, strategy))

    # ── Summary ───────────────────────────────────────────────────────────────
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
        strat_counts:  Counter = Counter(strategy for _, _, strategy in processed)
        chunks_by_strat: dict  = {}
        for _, n, strategy in processed:
            chunks_by_strat[strategy] = chunks_by_strat.get(strategy, 0) + n

        print("Strategy breakdown:")
        for strategy, doc_count in sorted(strat_counts.items()):
            print(
                f"  {strategy:<18} : {doc_count:>3} docs  →  "
                f"{chunks_by_strat[strategy]:>6,} chunks"
            )
        print()

    # Exit non-zero if there were hard failures so CI pipelines can detect them
    if skipped_hard or skipped_error:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()