"""
kaggle_advanced_pipeline.py
═════════════════════════════════════════════════════════════════════════════
Advanced Kaggle script: Full pipeline with chunking & taxonomy tagging

Integrates:
  • Docling extraction (batches)
  • Table processing (markdown export + data type detection)
  • Document taxonomy (doc_type, domain, agent classification)
  • Chunking (narrative / tables-only / hybrid)
  • Metadata enrichment from central_inventory.csv

Perfect for: Building production RAG datasets on Kaggle GPU

Usage:
  python kaggle_advanced_pipeline.py \\
    --batch-dir /kaggle/input/batches \\
    --output-dir /kaggle/working/output \\
    --config-csv /kaggle/input/config/central_inventory.csv \\
    --taxonomy-py /kaggle/input/config/doc_type_taxonomy.py

Outputs:
  • cache/              — Raw docling extractions (JSON)
  • chunks/             — Chunked documents (JSONL)
  • metadata.csv        — Flat metadata for all processed docs
  • progress.json       — Resume checkpoint
"""

import argparse
import gc
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import importlib.util

import psutil

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        EasyOcrOptions,
        TableFormerMode,
    )
    from docling_core.types.doc import (
        TableItem,
        TextItem,
        SectionHeaderItem,
        ListItem,
    )
    DOCLING_OK = True
except ImportError:
    DOCLING_OK = False

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TextBlock:
    text: str
    block_type: str
    heading_path: List[str]
    page_number: int
    block_index: int


@dataclass
class TableChunk:
    table_id: str
    markdown: str
    caption: str
    heading_path: List[str]
    page_number: int
    rows: int
    cols: int
    data_type: str


@dataclass
class ProcessedDocument:
    source_file: str
    doc_slug: str
    total_pages: int
    is_scanned: bool
    blocks: List[TextBlock]
    tables: List[TableChunk]
    extraction_time: float


@dataclass
class Chunk:
    """A single chunk ready for embedding."""
    chunk_id: str
    source_doc: str
    content: str
    chunk_type: str  # narrative | table | heading
    heading_path: List[str]
    page_number: int
    token_estimate: int
    metadata: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "pipeline.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)s]  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    
    return logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE & PATHS
# ══════════════════════════════════════════════════════════════════════════════

def get_device() -> str:
    if not TORCH_OK:
        return "cpu"
    return f"cuda ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "cpu"


def setup_paths(args) -> Dict[str, Path]:
    is_kaggle = Path("/kaggle/input").exists()
    
    batch_dir = Path(args.batch_dir) if args.batch_dir else (
        Path("/kaggle/input/batches") if is_kaggle else Path("./batches")
    )
    
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("/kaggle/working/output") if is_kaggle else Path("./output")
    )
    
    return {
        "batch_dir": batch_dir,
        "output_dir": output_dir,
        "cache_dir": output_dir / "cache",
        "chunks_dir": output_dir / "chunks",
        "progress_file": output_dir / "progress.json",
        "metadata_file": output_dir / "metadata.csv",
    }


# ══════════════════════════════════════════════════════════════════════════════
# TAXONOMY LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_taxonomy(py_path: Optional[str], log) -> dict:
    """Dynamically load doc_type_taxonomy.py if provided."""
    if not py_path:
        log.warning("No taxonomy file provided — using basic classification")
        return {}
    
    try:
        spec = importlib.util.spec_from_file_location("taxonomy", py_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        taxonomy = {
            "valid_doc_types": getattr(module, "VALID_DOC_TYPES", set()),
            "all_agents": getattr(module, "ALL_AGENTS", []),
            "agent_patterns": getattr(module, "AGENT_PATTERNS", {}),
        }
        
        log.info(f"✓ Taxonomy loaded: {len(taxonomy['valid_doc_types'])} doc types")
        return taxonomy
    
    except Exception as e:
        log.error(f"Failed to load taxonomy: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# METADATA LOOKUP
# ══════════════════════════════════════════════════════════════════════════════

def load_inventory_csv(csv_path: Optional[str], log) -> pd.DataFrame:
    """Load central_inventory.csv for metadata lookup."""
    if not csv_path or not Path(csv_path).exists():
        log.warning("No inventory CSV — metadata lookups will use filename fallback")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path)
        log.info(f"✓ Inventory loaded: {len(df)} rows")
        return df
    except Exception as e:
        log.error(f"Failed to load inventory: {e}")
        return pd.DataFrame()


def lookup_metadata(filename: str, inventory: pd.DataFrame) -> dict:
    """Look up metadata from inventory by filename."""
    if inventory.empty:
        return {}
    
    matches = inventory[inventory["filename"] == filename]
    if matches.empty:
        return {}
    
    row = matches.iloc[0]
    
    return {
        "agent": row.get("agent"),
        "document_type": row.get("document_type"),
        "domain": row.get("domain"),
        "fiscal_year": row.get("fiscal_year"),
        "doc_year": row.get("doc_year"),
        "topics": row.get("topics", "").split("|") if isinstance(row.get("topics"), str) else [],
        "priority": row.get("priority"),
        "is_scanned": bool(row.get("is_scanned")),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DOCLING CONVERTERS
# ══════════════════════════════════════════════════════════════════════════════

def build_converters(log) -> Tuple:
    """Create standard + OCR converters."""
    if not DOCLING_OK:
        log.error("docling not installed")
        sys.exit(1)
    
    standard_opts = PdfPipelineOptions(do_table_structure=True)
    standard_opts.table_structure_options.mode = TableFormerMode.ACCURATE
    standard_opts.table_structure_options.do_cell_matching = True
    
    converter_std = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=standard_opts)
        }
    )
    
    ocr_opts = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        ocr_options=EasyOcrOptions(lang=["en"]),
    )
    ocr_opts.table_structure_options.mode = TableFormerMode.ACCURATE
    
    converter_ocr = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=ocr_opts)
        }
    )
    
    log.info("✓ Docling converters ready")
    
    return converter_std, converter_ocr


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTION + TABLE PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def extract_and_process(
    pdf_path: Path,
    converter_std,
    converter_ocr,
    log,
) -> Optional[ProcessedDocument]:
    """Extract text/tables and process into narrative + table chunks."""
    
    start = time.time()
    doc_slug = pdf_path.stem.lower().replace(" ", "_").replace("-", "_")[:50]
    
    try:
        doc = converter_std.convert(str(pdf_path))
        
        blocks = []
        heading_path = []
        block_idx = 0
        
        # Extract textual content
        for item in doc.document.body:
            if isinstance(item, SectionHeaderItem):
                heading_path = [item.text]
                blocks.append(TextBlock(
                    text=item.text,
                    block_type="heading",
                    heading_path=heading_path.copy(),
                    page_number=item.prov[0].page_num if item.prov else 0,
                    block_index=block_idx,
                ))
                block_idx += 1
            
            elif isinstance(item, TextItem):
                blocks.append(TextBlock(
                    text=item.text,
                    block_type="paragraph",
                    heading_path=heading_path.copy(),
                    page_number=item.prov[0].page_num if item.prov else 0,
                    block_index=block_idx,
                ))
                block_idx += 1
            
            elif isinstance(item, ListItem):
                blocks.append(TextBlock(
                    text=item.text,
                    block_type="list_item",
                    heading_path=heading_path.copy(),
                    page_number=item.prov[0].page_num if item.prov else 0,
                    block_index=block_idx,
                ))
                block_idx += 1
        
        # Extract tables
        tables = []
        table_idx = 0
        
        for item in doc.document.body:
            if isinstance(item, TableItem):
                try:
                    df = item.data.to_pandas()
                    
                    # Detect data type
                    headers = " ".join(str(c) for c in df.columns)
                    proj = bool(re.search(
                        r"proj|est\.|estimate|forecast|target|budget",
                        headers, re.IGNORECASE
                    ))
                    actual = bool(re.search(
                        r"actual|outturn|audited|preliminary",
                        headers, re.IGNORECASE
                    ))
                    
                    data_type = "mixed" if (proj and actual) else ("actual" if actual else ("projection" if proj else "mixed"))
                    
                    markdown = df.to_markdown(index=False)
                    
                    tables.append(TableChunk(
                        table_id=f"{doc_slug}_table_{table_idx:03d}",
                        markdown=markdown,
                        caption=item.caption or "",
                        heading_path=heading_path.copy(),
                        page_number=item.prov[0].page_num if item.prov else 0,
                        rows=df.shape[0],
                        cols=df.shape[1],
                        data_type=data_type,
                    ))
                    
                    table_idx += 1
                
                except Exception as e:
                    log.warning(f"    Table extraction failed: {e}")
                    continue
        
        elapsed = time.time() - start
        
        log.info(f"  ✓ {len(blocks)} blocks, {len(tables)} tables ({elapsed:.1f}s)")
        
        return ProcessedDocument(
            source_file=str(pdf_path),
            doc_slug=doc_slug,
            total_pages=len(doc.pages),
            is_scanned=False,
            blocks=blocks,
            tables=tables,
            extraction_time=elapsed,
        )
    
    except Exception as e:
        log.error(f"  ✗ Extraction failed: {str(e)[:150]}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    """Rough token estimate (1 token ~ 4 chars)."""
    return len(text) // 4


def create_chunks(doc: ProcessedDocument, metadata: dict) -> List[Chunk]:
    """Convert extracted document into chunks for embedding."""
    
    chunks = []
    chunk_counter = 0
    
    # Strategy 1: Narrative chunking (group adjacent blocks)
    current_chunk = []
    current_heading = []
    chunk_start_page = 0
    
    for block in doc.blocks:
        if block.block_type == "heading" and current_chunk:
            # Save current chunk before heading
            text = "\n".join([b.text for b in current_chunk])
            tokens = estimate_tokens(text)
            
            if tokens > 0:
                chunk_id = f"{doc.doc_slug}_chunk_{chunk_counter:04d}"
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    source_doc=doc.doc_slug,
                    content=text,
                    chunk_type="narrative",
                    heading_path=current_heading,
                    page_number=chunk_start_page,
                    token_estimate=tokens,
                    metadata=metadata,
                ))
                chunk_counter += 1
            
            current_chunk = []
        
        current_chunk.append(block)
        if not current_heading:
            current_heading = block.heading_path
        if not chunk_start_page:
            chunk_start_page = block.page_number
    
    # Save final chunk
    if current_chunk:
        text = "\n".join([b.text for b in current_chunk])
        tokens = estimate_tokens(text)
        if tokens > 0:
            chunk_id = f"{doc.doc_slug}_chunk_{chunk_counter:04d}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                source_doc=doc.doc_slug,
                content=text,
                chunk_type="narrative",
                heading_path=current_heading,
                page_number=chunk_start_page,
                token_estimate=tokens,
                metadata=metadata,
            ))
            chunk_counter += 1
    
    # Strategy 2: Table chunks
    for table in doc.tables:
        chunk_id = f"{doc.doc_slug}_{table.table_id}"
        chunks.append(Chunk(
            chunk_id=chunk_id,
            source_doc=doc.doc_slug,
            content=f"# {table.caption or 'Table'}\n\n{table.markdown}",
            chunk_type="table",
            heading_path=table.heading_path,
            page_number=table.page_number,
            token_estimate=estimate_tokens(table.markdown),
            metadata={**metadata, "table_id": table.table_id, "data_type": table.data_type},
        ))
        chunk_counter += 1
    
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# PROGRESS TRACKING
# ══════════════════════════════════════════════════════════════════════════════

class ProgressTracker:
    def __init__(self, path: Path):
        self.path = path
        self.data = self._load()
    
    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        
        return {
            "completed": [],
            "failed": [],
            "stats": {
                "docs": 0,
                "blocks": 0,
                "tables": 0,
                "chunks": 0,
                "time": 0,
            },
        }
    
    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)
    
    def is_done(self, fname: str) -> bool:
        return fname in self.data["completed"]
    
    def mark_complete(self, fname: str, blocks: int, tables: int, chunks: int, time_taken: float):
        if fname not in self.data["completed"]:
            self.data["completed"].append(fname)
        
        self.data["stats"]["docs"] += 1
        self.data["stats"]["blocks"] += blocks
        self.data["stats"]["tables"] += tables
        self.data["stats"]["chunks"] += chunks
        self.data["stats"]["time"] += time_taken
        
        self._save()
    
    def summary(self) -> str:
        s = self.data["stats"]
        return (
            f"Docs: {s['docs']} | Blocks: {s['blocks']} | "
            f"Tables: {s['tables']} | Chunks: {s['chunks']} | "
            f"Time: {s['time']:.1f}s"
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Advanced Kaggle pipeline with chunking")
    parser.add_argument("--batch-dir", help="Batches folder")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--config-csv", help="Path to central_inventory.csv")
    parser.add_argument("--taxonomy-py", help="Path to doc_type_taxonomy.py")
    parser.add_argument("--force", action="store_true", help="Re-process all")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    
    args = parser.parse_args()
    
    paths = setup_paths(args)
    for p in paths.values():
        if isinstance(p, Path) and p.name not in ["progress.json", "metadata.csv"]:
            p.mkdir(parents=True, exist_ok=True)
    
    log = setup_logging(paths["output_dir"])
    
    log.info("╔════════════════════════════════════════════════════════════════╗")
    log.info("║  Kaggle Advanced Pipeline — Extraction + Chunking + Taxonomy  ║")
    log.info("╚════════════════════════════════════════════════════════════════╝")
    
    log.info(f"Device: {get_device()}")
    log.info(f"Batch dir: {paths['batch_dir']}")
    log.info(f"Output dir: {paths['output_dir']}")
    
    if not DOCLING_OK:
        log.error("docling not installed")
        sys.exit(1)
    
    if not paths["batch_dir"].exists():
        log.error(f"Batch directory not found")
        sys.exit(1)
    
    # Load taxonomy and inventory
    taxonomy = load_taxonomy(args.taxonomy_py, log)
    inventory = load_inventory_csv(args.config_csv, log)
    
    # Find PDFs
    batch_dirs = sorted([d for d in paths["batch_dir"].iterdir() if d.is_dir()])
    pdf_files = []
    
    for batch_dir in batch_dirs:
        pdfs = sorted(batch_dir.glob("*.pdf"))
        log.info(f"  {batch_dir.name}: {len(pdfs)} PDFs")
        pdf_files.extend([(batch_dir.name, pdf) for pdf in pdfs])
    
    log.info(f"Total: {len(pdf_files)} PDFs")
    
    if args.dry_run:
        log.info("DRY RUN — no processing")
        return
    
    # Setup
    converter_std, converter_ocr = build_converters(log)
    tracker = ProgressTracker(paths["progress_file"])
    
    # Process
    all_chunks = []
    docs_metadata = []
    
    for batch_name, pdf_path in pdf_files:
        log.info(f"\n📄 {batch_name} / {pdf_path.name}")
        
        if tracker.is_done(pdf_path.name) and not args.force:
            log.info("  (cached)")
            continue
        
        # Extract
        doc = extract_and_process(pdf_path, converter_std, converter_ocr, log)
        if not doc:
            tracker.mark_complete(pdf_path.name, 0, 0, 0, 0)
            continue
        
        # Metadata
        metadata = lookup_metadata(pdf_path.name, inventory)
        
        # Create chunks
        chunks = create_chunks(doc, metadata)
        
        # Save cache
        cache_file = paths["cache_dir"] / f"{doc.doc_slug}.json"
        with open(cache_file, "w") as f:
            json.dump({
                "source_file": doc.source_file,
                "doc_slug": doc.doc_slug,
                "blocks": len(doc.blocks),
                "tables": len(doc.tables),
            }, f, indent=2)
        
        # Save chunks
        chunks_file = paths["chunks_dir"] / f"{doc.doc_slug}.jsonl"
        with open(chunks_file, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        
        all_chunks.extend(chunks)
        docs_metadata.append({
            "doc_slug": doc.doc_slug,
            "pdf_file": pdf_path.name,
            "batch": batch_name,
            "pages": doc.total_pages,
            "blocks": len(doc.blocks),
            "tables": len(doc.tables),
            "chunks": len(chunks),
            **metadata,
        })
        
        tracker.mark_complete(pdf_path.name, len(doc.blocks), len(doc.tables), len(chunks), doc.extraction_time)
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save metadata
    if docs_metadata:
        df_meta = pd.DataFrame(docs_metadata)
        df_meta.to_csv(paths["metadata_file"], index=False)
        log.info(f"\n✓ Metadata saved: {len(df_meta)} documents")
    
    log.info("\n" + "="*70)
    log.info(f"COMPLETE: {tracker.summary()}")
    log.info(f"Chunks: {len(all_chunks)}")
    log.info("="*70)


if __name__ == "__main__":
    main()
