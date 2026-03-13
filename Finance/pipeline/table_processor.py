"""
pipeline/table_processor.py — Table processing for Kenya AI Executive Roundtable

Takes the raw ExtractedDocument from extractor.py and:
  1. Exports each table to markdown using docling's built-in exporter
  2. Detects data_type (actual/projection/target/mixed) from column headers
  3. Inserts [TABLE_REF: table_id] markers into the narrative block stream
  4. Filters trivial/empty tables

Returns a ProcessedDocument ready for chunker.py.

Usage:
    from pipeline.table_processor import TableProcessor

    processor = TableProcessor()
    processed = processor.process(raw_doc, doc_config)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from pipeline.extractor import ExtractedDocument, ExtractedTable, TextBlock

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TableChunk:
    """
    A single table exported to markdown, ready to be treated as a chunk.
    Primary output unit for category 2 (pure tables) and the table
    component of category 8 (hybrid).
    """
    table_id:        str
    markdown:        str             # docling markdown export
    caption:         str
    heading_path:    list[str]
    page_number:     int
    table_index:     int
    source_file:     str
    shape:           tuple[int, int] # (rows, cols) of original DataFrame
    data_type_hint:  str             # actual | projection | target | mixed


@dataclass
class ProcessedDocument:
    """
    Full output of table_processor — ready for chunker.py.

    blocks       : narrative TextBlocks with TABLE_REF markers inserted
    table_chunks : markdown tables ordered by table_index
    """
    source_file:  str
    doc_slug:     str
    is_scanned:   bool
    total_pages:  int
    blocks:       list[TextBlock]  = field(default_factory=list)
    table_chunks: list[TableChunk] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# DATA TYPE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

PROJECTION_RE = re.compile(
    r"proj|est\.|estimate|forecast|target|budget|plan|revised\s+est",
    re.IGNORECASE
)
ACTUAL_RE = re.compile(
    r"actual|outturn|audited|preliminary|prel\.|prov\.",
    re.IGNORECASE
)


def _detect_data_type(df: pd.DataFrame) -> str:
    """
    Infer whether a table contains actual, projection, target, or mixed data
    by scanning column headers.
    """
    headers    = " ".join(str(c) for c in df.columns)
    has_actual = bool(ACTUAL_RE.search(headers))
    has_proj   = bool(PROJECTION_RE.search(headers))

    if has_actual and has_proj:
        return "mixed"
    if has_actual:
        return "actual"
    if has_proj:
        return "projection"
    return "actual"   # default — most tables in government reports are actuals


# ══════════════════════════════════════════════════════════════════════════════
# TABLE REF INSERTION
# ══════════════════════════════════════════════════════════════════════════════

def _insert_table_refs(
    blocks: list[TextBlock],
    tables: list[ExtractedTable],
) -> list[TextBlock]:
    """
    Insert [TABLE_REF: table_id] marker blocks into the narrative block list
    immediately after the last block on or before each table's page.

    Also sets block.table_ref on the preceding narrative block so the
    chunker can link narrative chunks to their associated tables.

    Returns a new block list — original is not mutated.
    """
    if not tables:
        return list(blocks)

    result = list(blocks)

    for table in tables:
        marker_text = f"[TABLE_REF: {table.table_id}]"
        target_page = table.page_number

        # Find the last block on or before the table's page
        insert_after = -1
        for i, block in enumerate(result):
            if block.page_number <= target_page:
                insert_after = i

        if insert_after == -1:
            insert_pos = 0
        else:
            insert_pos = insert_after + 1
            # Tag the preceding block so chunker can build table_refs list
            result[insert_after].table_ref = table.table_id

        marker = TextBlock(
            text         = marker_text,
            block_type   = "table_ref",
            heading_path = list(table.heading_path),
            page_number  = table.page_number,
            block_index  = -1,   # synthetic
            table_ref    = table.table_id,
        )
        result.insert(insert_pos, marker)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# TABLE PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class TableProcessor:

    def process(
        self,
        raw_doc: ExtractedDocument,
        doc_config: dict,
    ) -> ProcessedDocument:
        """
        Args:
            raw_doc    : Output of Extractor.extract()
            doc_config : Single document entry from config.yaml

        Returns:
            ProcessedDocument with markdown table_chunks and annotated blocks
        """
        log.info(
            f"Processing tables: {raw_doc.source_file} "
            f"({len(raw_doc.tables)} tables)"
        )

        table_chunks = []
        valid_tables = []   # tables that passed filtering, for ref insertion

        for table in raw_doc.tables:
            chunk = self._convert_table(table, raw_doc.source_file)
            if chunk:
                table_chunks.append(chunk)
                valid_tables.append(table)

        annotated_blocks = _insert_table_refs(raw_doc.blocks, valid_tables)

        log.info(
            f"  -> {len(table_chunks)} table chunks, "
            f"{len(annotated_blocks)} blocks "
            f"(incl. {len(valid_tables)} TABLE_REF markers)"
        )

        return ProcessedDocument(
            source_file  = raw_doc.source_file,
            doc_slug     = raw_doc.doc_slug,
            is_scanned   = raw_doc.is_scanned,
            total_pages  = raw_doc.total_pages,
            blocks       = annotated_blocks,
            table_chunks = table_chunks,
        )

    def _convert_table(
        self,
        table: ExtractedTable,
        source_file: str,
    ) -> Optional[TableChunk]:
        """
        Convert a single ExtractedTable to a TableChunk using the markdown
        already produced by extractor.py (table.markdown).
        """
        df = table.df

        # Filter trivial tables
        if df.empty:
            log.debug(f"  Skip empty: {table.table_id}")
            return None
        if df.shape[0] < 2 or df.shape[1] < 2:
            log.debug(f"  Skip trivial {table.table_id} shape={df.shape}")
            return None

        markdown = table.markdown
        if not markdown or not markdown.strip():
            log.debug(f"  Skip no-markdown: {table.table_id}")
            return None

        # Prepend heading context and caption so the chunk is self-contained
        # when retrieved in isolation by the RAG system
        header_lines = []
        if table.heading_path:
            header_lines.append(f"Section: {' > '.join(table.heading_path)}")
        if table.caption:
            header_lines.append(f"Table: {table.caption}")
        header_lines.append(f"Reference: {table.table_id}")

        full_markdown = "\n".join(header_lines) + "\n\n" + markdown

        return TableChunk(
            table_id       = table.table_id,
            markdown       = full_markdown,
            caption        = table.caption,
            heading_path   = table.heading_path,
            page_number    = table.page_number,
            table_index    = table.table_index,
            source_file    = source_file,
            shape          = df.shape,
            data_type_hint = _detect_data_type(df),
        )


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python pipeline/table_processor.py --from-cache <cache_file.json>")
        print("       python pipeline/table_processor.py <path/to/file.pdf>")
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).parent.parent))

    if sys.argv[1] == "--from-cache":
        if len(sys.argv) < 3:
            print("Provide cache file path.")
            sys.exit(1)
        from pipeline.extractor import _load_from_cache
        cache_path = Path(sys.argv[2])
        print(f"Loading from cache: {cache_path.name}\n")
        raw_doc = _load_from_cache(cache_path, cache_path.stem, cache_path.stem, False)
    else:
        from pipeline.extractor import extract_single
        print(f"Extracting: {sys.argv[1]}\n")
        raw_doc = extract_single(sys.argv[1])

    processor = TableProcessor()
    processed = processor.process(raw_doc, {})

    print(f"\nSource      : {processed.source_file}")
    print(f"Blocks      : {len(processed.blocks)} (incl. TABLE_REF markers)")
    print(f"Table chunks: {len(processed.table_chunks)}")

    print("\n-- First 3 table chunks ----------------------------------------")
    for chunk in processed.table_chunks[:3]:
        print(f"\n  {chunk.table_id}  shape={chunk.shape}  p{chunk.page_number}")
        print(f"  heading   : {' > '.join(chunk.heading_path)}")
        print(f"  data_type : {chunk.data_type_hint}")
        print(f"  markdown preview:")
        for line in chunk.markdown.splitlines()[:8]:
            print(f"    {line}")
        total_lines = len(chunk.markdown.splitlines())
        if total_lines > 8:
            print(f"    ... ({total_lines} lines total)")

    print("\n-- TABLE_REF markers (first 10) --------------------------------")
    ref_blocks = [b for b in processed.blocks if b.block_type == "table_ref"]
    for b in ref_blocks[:10]:
        print(f"  p{b.page_number:3}  {b.text}")