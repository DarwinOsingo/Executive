"""
pipeline/extractor.py — Docling wrapper for Kenya AI Executive Roundtable

Converts a PDF into structured extraction output consumed by chunker.py.

Returns an ExtractedDocument containing:
  - blocks : list of TextBlock  (narrative text, headings, list items)
  - tables : list of ExtractedTable (raw DataFrame + position metadata)

Usage:
    from pipeline.extractor import Extractor, ExtractedDocument
    extractor = Extractor()
    doc = extractor.extract(pdf_path, doc_config)

Install:
    pip install docling pandas
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ── Docling imports ────────────────────────────────────────────────────────────
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        EasyOcrOptions,
        TableFormerMode,
    )
    from docling.datamodel.document import DoclingDocument
    from docling_core.types.doc import (
        DocItemLabel,
        TableItem,
        TextItem,
        SectionHeaderItem,
        ListItem,
    )
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    log.warning("docling not installed — run: pip install docling")


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TextBlock:
    """A single narrative unit from the document."""
    text:         str
    block_type:   str          # "heading" | "paragraph" | "list_item" | "caption"
    heading_path: list[str]    # e.g. ["Chapter 3", "3.2 Revenue Performance"]
    page_number:  int
    block_index:  int          # position in document reading order
    table_ref:    Optional[str] = None  # set by table_processor, not here


@dataclass
class ExtractedTable:
    """A single table extracted from the document."""
    table_id:     str          # e.g. "2026_bps_table_001"
    df:           pd.DataFrame
    caption:      str          # text immediately before the table, if any
    heading_path: list[str]    # heading context at point of extraction
    page_number:  int
    table_index:  int          # 0-based index within this document
    markdown:     str          # df rendered as markdown (for prose conversion)


@dataclass
class ExtractedDocument:
    """Full extraction output for one PDF."""
    source_file:  str
    doc_slug:     str          # filename stem, sanitised, used in table IDs
    is_scanned:   bool
    total_pages:  int
    blocks:       list[TextBlock]  = field(default_factory=list)
    tables:       list[ExtractedTable] = field(default_factory=list)
    error:        Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

class Extractor:
    """
    Wraps docling's DocumentConverter with settings tuned for Kenyan
    government PDFs — mixed narrative/table content, some scanned pages.

    One Extractor instance should be reused across all documents in a
    pipeline run — docling caches its pipeline initialisation internally,
    so creating a new instance per document is expensive.
    """

    def __init__(self):
        if not DOCLING_AVAILABLE:
            raise ImportError("docling is required. Run: pip install docling")

        # Standard converter — no OCR, accurate table extraction
        # Used for machine-readable PDFs
        standard_options = PdfPipelineOptions(
            do_table_structure=True,
        )
        standard_options.table_structure_options.mode = TableFormerMode.ACCURATE
        standard_options.table_structure_options.do_cell_matching = True

        self._converter_standard = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=standard_options
                )
            }
        )

        # OCR converter — EasyOCR for scanned PDFs
        ocr_options = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
            ocr_options=EasyOcrOptions(lang=["en"]),
        )
        ocr_options.table_structure_options.mode = TableFormerMode.ACCURATE

        self._converter_ocr = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=ocr_options
                )
            }
        )

        log.info("Extractor initialised (standard + OCR converters ready)")

    # ── Public entry point ─────────────────────────────────────────────────

    def extract(
        self,
        pdf_path: Path,
        doc_config: dict,
        cache_dir: Optional[Path] = None,
    ) -> ExtractedDocument:
        """
        Extract text blocks and tables from a PDF.

        Args:
            pdf_path   : Path to the PDF file
            doc_config : Single document entry from config.yaml
            cache_dir  : Optional directory to cache extraction results.
                         If provided and a cache file exists, the cached
                         result is returned without re-running docling.
                         Cache file: <cache_dir>/<doc_slug>.json

        Returns:
            ExtractedDocument with blocks and tables populated
        """
        pdf_path  = Path(pdf_path)
        fname     = pdf_path.name
        doc_slug  = _make_slug(pdf_path.stem)
        is_scanned = doc_config.get("is_scanned", False)
        skip_sections = [s.lower() for s in doc_config.get("skip_sections", [])]

        # Check cache first
        if cache_dir is not None:
            cache_path = Path(cache_dir) / f"{doc_slug}.json"
            if cache_path.exists():
                log.info(f"Cache hit: {fname} → loading from {cache_path.name}")
                return _load_from_cache(cache_path, fname, doc_slug, is_scanned)

        log.info(f"Extracting: {fname} (scanned={is_scanned})")

        result = ExtractedDocument(
            source_file = fname,
            doc_slug    = doc_slug,
            is_scanned  = is_scanned,
            total_pages = 0,
        )

        try:
            converter = self._converter_ocr if is_scanned else self._converter_standard
            conv      = converter.convert(str(pdf_path))
            doc       = conv.document

            result.total_pages = _count_pages(doc)
            result.blocks, result.tables = self._parse_document(
                doc, doc_slug, skip_sections
            )

            log.info(
                f"  → {len(result.blocks)} blocks, "
                f"{len(result.tables)} tables, "
                f"{result.total_pages} pages"
            )

            # Save to cache
            if cache_dir is not None and not result.error:
                cache_path = Path(cache_dir) / f"{doc_slug}.json"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                _save_to_cache(result, cache_path)
                log.info(f"  Cached → {cache_path.name}")

        except Exception as e:
            log.error(f"Extraction failed for {fname}: {e}")
            result.error = str(e)

        return result

    # ── Core parsing ───────────────────────────────────────────────────────

    def _parse_document(
        self,
        doc: "DoclingDocument",
        doc_slug: str,
        skip_sections: list[str],
    ) -> tuple[list[TextBlock], list[ExtractedTable]]:
        """
        Walk docling's item list in reading order.
        Build heading path as we go.
        Skip sections that match skip_sections config.
        """
        blocks:         list[TextBlock]     = []
        tables:         list[ExtractedTable] = []
        heading_path:   list[str]           = []
        block_index:    int                 = 0
        table_index:    int                 = 0
        skip_active:    bool                = False
        last_text_block: Optional[TextBlock] = None  # for table caption lookup

        for item, _ in doc.iterate_items():
            label = _get_label(item)
            page  = _get_page(item)

            # ── Heading — update path and check skip ──────────────────────
            if label == "section_header":
                text          = _get_text(item).strip()
                level         = _get_heading_level(item)
                heading_path  = _update_heading_path(heading_path, text, level)
                skip_active   = _should_skip(heading_path, skip_sections)

                if not skip_active:
                    block = TextBlock(
                        text         = text,
                        block_type   = "heading",
                        heading_path = list(heading_path),
                        page_number  = page,
                        block_index  = block_index,
                    )
                    blocks.append(block)
                    last_text_block = block
                    block_index += 1
                continue

            if skip_active:
                continue

            # ── Paragraph ─────────────────────────────────────────────────
            if label == "text":
                text = _get_text(item).strip()
                if not text:
                    continue
                block = TextBlock(
                    text         = text,
                    block_type   = "paragraph",
                    heading_path = list(heading_path),
                    page_number  = page,
                    block_index  = block_index,
                )
                blocks.append(block)
                last_text_block = block
                block_index += 1

            # ── List item ─────────────────────────────────────────────────
            elif label == "list_item":
                text = _get_text(item).strip()
                if not text:
                    continue
                block = TextBlock(
                    text         = text,
                    block_type   = "list_item",
                    heading_path = list(heading_path),
                    page_number  = page,
                    block_index  = block_index,
                )
                blocks.append(block)
                last_text_block = block
                block_index += 1

            # ── Caption ───────────────────────────────────────────────────
            elif label == "caption":
                text = _get_text(item).strip()
                if not text:
                    continue
                block = TextBlock(
                    text         = text,
                    block_type   = "caption",
                    heading_path = list(heading_path),
                    page_number  = page,
                    block_index  = block_index,
                )
                blocks.append(block)
                last_text_block = block
                block_index += 1

            # ── Table ─────────────────────────────────────────────────────
            elif label == "table":
                table_id = f"{doc_slug}_table_{table_index:03d}"

                try:
                    df = item.export_to_dataframe(doc=doc)
                except Exception as e:
                    log.warning(f"  Table {table_id}: export_to_dataframe failed — {e}")
                    table_index += 1
                    continue

                if df.empty:
                    log.debug(f"  Table {table_id}: empty, skipping")
                    table_index += 1
                    continue

                # Use the last text block's text as caption if it's short enough
                caption = ""
                if last_text_block and len(last_text_block.text) < 200:
                    caption = last_text_block.text

                markdown = _df_to_markdown(df)

                table = ExtractedTable(
                    table_id     = table_id,
                    df           = df,
                    caption      = caption,
                    heading_path = list(heading_path),
                    page_number  = page,
                    table_index  = table_index,
                    markdown     = markdown,
                )
                tables.append(table)
                table_index += 1

        return blocks, tables


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _make_slug(stem: str) -> str:
    """Convert filename stem to a safe slug for use in table IDs."""
    slug = stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    # Truncate to 40 chars to keep table IDs manageable
    return slug[:40]


def _get_label(item) -> str:
    """Normalise item label to a simple string."""
    try:
        label = item.label
        if hasattr(label, "value"):
            return label.value.lower()
        return str(label).lower()
    except Exception:
        return "unknown"


def _get_text(item) -> str:
    try:
        if hasattr(item, "text"):
            return item.text or ""
        return ""
    except Exception:
        return ""


def _get_page(item) -> int:
    """Return 1-based page number, 0 if unavailable."""
    try:
        provs = getattr(item, "prov", [])
        if provs:
            return provs[0].page_no
        return 0
    except Exception:
        return 0


def _get_heading_level(item) -> int:
    """Return heading level 1-6. Falls back to 1 if not determinable."""
    try:
        if hasattr(item, "level"):
            return max(1, min(6, int(item.level)))
        # Infer from label value if present (e.g. "section_header_1")
        label = str(getattr(item, "label", ""))
        match = re.search(r"(\d+)$", label)
        if match:
            return max(1, min(6, int(match.group(1))))
        return 1
    except Exception:
        return 1


def _update_heading_path(path: list[str], text: str, level: int) -> list[str]:
    """
    Maintain a heading breadcrumb trail.
    Level 1 resets everything. Level 2 keeps level 1. Etc.
    """
    new_path = path[: level - 1]
    new_path.append(text)
    return new_path


def _should_skip(heading_path: list[str], skip_sections: list[str]) -> bool:
    """
    Return True if any heading in the current path matches a skip section.
    Matching is case-insensitive substring match.
    """
    if not skip_sections:
        return False
    path_lower = " ".join(heading_path).lower()
    return any(s in path_lower for s in skip_sections)


def _count_pages(doc) -> int:
    try:
        return len(doc.pages) if hasattr(doc, "pages") else 0
    except Exception:
        return 0


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table string. Truncates large tables."""
    try:
        # Cap at 50 rows for the markdown repr — full data stays in df
        display_df = df.head(50) if len(df) > 50 else df
        return display_df.to_markdown(index=False)
    except Exception:
        # tabulate not installed or other issue — fall back to csv-style
        try:
            return df.head(50).to_csv(index=False)
        except Exception:
            return "[table — could not render]"


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: extract a single PDF without config (for testing)
# ══════════════════════════════════════════════════════════════════════════════

def extract_single(pdf_path: str | Path, is_scanned: bool = False) -> ExtractedDocument:
    """
    Quick extraction for testing without a full config.yaml.

    Example:
        from pipeline.extractor import extract_single
        doc = extract_single("data/raw/2026 Budget Policy Statement.pdf")
        for block in doc.blocks[:5]:
            print(block.block_type, block.heading_path, block.text[:80])
        for table in doc.tables[:2]:
            print(table.table_id, table.df.shape)
    """
    extractor = Extractor()
    config    = {"is_scanned": is_scanned, "skip_sections": []}
    return extractor.extract(Path(pdf_path), config)




# ══════════════════════════════════════════════════════════════════════════════
# CACHE SERIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _save_to_cache(result: ExtractedDocument, cache_path: Path):
    """Serialise ExtractedDocument to JSON. DataFrames stored as records."""
    data = {
        "source_file":  result.source_file,
        "doc_slug":     result.doc_slug,
        "is_scanned":   result.is_scanned,
        "total_pages":  result.total_pages,
        "error":        result.error,
        "blocks": [
            {
                "text":         b.text,
                "block_type":   b.block_type,
                "heading_path": b.heading_path,
                "page_number":  b.page_number,
                "block_index":  b.block_index,
                "table_ref":    b.table_ref,
            }
            for b in result.blocks
        ],
        "tables": [
            {
                "table_id":     t.table_id,
                "caption":      t.caption,
                "heading_path": t.heading_path,
                "page_number":  t.page_number,
                "table_index":  t.table_index,
                "markdown":     t.markdown,
                "records":      t.df.to_dict(orient="records"),
                "columns":      list(t.df.columns),
            }
            for t in result.tables
        ],
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_from_cache(
    cache_path: Path,
    fname: str,
    doc_slug: str,
    is_scanned: bool,
) -> ExtractedDocument:
    """Deserialise ExtractedDocument from JSON cache."""
    with open(cache_path, encoding="utf-8") as f:
        data = json.load(f)

    blocks = [
        TextBlock(
            text         = b["text"],
            block_type   = b["block_type"],
            heading_path = b["heading_path"],
            page_number  = b["page_number"],
            block_index  = b["block_index"],
            table_ref    = b.get("table_ref"),
        )
        for b in data["blocks"]
    ]

    tables = [
        ExtractedTable(
            table_id     = t["table_id"],
            df           = pd.DataFrame.from_records(t["records"], columns=t["columns"]),
            caption      = t["caption"],
            heading_path = t["heading_path"],
            page_number  = t["page_number"],
            table_index  = t["table_index"],
            markdown     = t["markdown"],
        )
        for t in data["tables"]
    ]

    return ExtractedDocument(
        source_file = data.get("source_file", fname),
        doc_slug    = data.get("doc_slug", doc_slug),
        is_scanned  = data.get("is_scanned", is_scanned),
        total_pages = data.get("total_pages", 0),
        blocks      = blocks,
        tables      = tables,
        error       = data.get("error"),
    )
# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python extractor.py <path/to/file.pdf> [--scanned]")
        sys.exit(1)

    path      = Path(sys.argv[1])
    is_scanned = "--scanned" in sys.argv

    print(f"\nExtracting: {path.name}")
    print(f"Scanned   : {is_scanned}\n")

    doc = extract_single(path, is_scanned)

    if doc.error:
        print(f"ERROR: {doc.error}")
        sys.exit(1)

    print(f"Pages  : {doc.total_pages}")
    print(f"Blocks : {len(doc.blocks)}")
    print(f"Tables : {len(doc.tables)}")
    print(f"Slug   : {doc.doc_slug}")

    print("\n── First 5 blocks ──────────────────────────────────────")
    for b in doc.blocks[:5]:
        path_str = " > ".join(b.heading_path) or "(no heading)"
        print(f"  [{b.block_type:10}] p{b.page_number:3}  {path_str}")
        print(f"              {b.text[:100]!r}")

    if doc.tables:
        print("\n── First 2 tables ──────────────────────────────────────")
        for t in doc.tables[:2]:
            print(f"  {t.table_id}  shape={t.df.shape}  p{t.page_number}")
            print(f"  caption: {t.caption[:80]!r}")
            print(f"  heading: {' > '.join(t.heading_path)}")
            print()