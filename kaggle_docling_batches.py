"""
kaggle_docling_batches.py
═════════════════════════════════════════════════════════════════════════════
Kaggle-ready script: Extract and process batches using Docling

Designed for:
  • Kaggle GPU (T4/P100) with automatic resource detection
  • Batch folders (batch_1, batch_2, ...) uploaded as dataset
  • Full resume support (progress tracking via JSON)
  • Memory optimization (clear cache between documents)

Usage:
  1. Upload your batches folder to Kaggle as a dataset
  2. Add this script + doc_type_taxonomy.py to your notebook
  3. Run with: python kaggle_docling_batches.py
  
Optional args:
  --batch-dir    Path to batches folder (default: /kaggle/input/batches)
  --output-dir   Output directory (default: /kaggle/working/processing)
  --config-csv   Path to central_inventory.csv for metadata lookup
  --force        Re-process even if cached
  --dry-run      Show what would run without extracting
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List
import hashlib

import psutil

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS — Docling + pandas
# ══════════════════════════════════════════════════════════════════════════════

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
        DocItemLabel,
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
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to both stdout and file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "extraction.log"
    
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
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TextBlock:
    """A single narrative unit."""
    text: str
    block_type: str  # heading | paragraph | list_item | caption
    heading_path: List[str]
    page_number: int
    block_index: int


@dataclass
class ExtractedTable:
    """A single table."""
    table_id: str
    caption: str
    heading_path: List[str]
    page_number: int
    table_index: int
    rows: int
    cols: int
    markdown: str
    data_type: str = "mixed"  # actual | projection | target | mixed


@dataclass
class ExtractedDocument:
    """Full extraction output for one PDF."""
    source_file: str
    doc_slug: str
    total_pages: int
    is_scanned: bool
    blocks: List[TextBlock] = field(default_factory=list)
    tables: List[ExtractedTable] = field(default_factory=list)
    error: Optional[str] = None
    extraction_time: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE & ENVIRONMENT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def get_device() -> str:
    """Detect GPU (CUDA) or fallback to CPU."""
    if not TORCH_OK:
        return "cpu"
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return f"cuda ({device_name})"
    
    return "cpu"


def detect_kaggle() -> bool:
    """Check if running on Kaggle."""
    return Path("/kaggle/input").exists()


def setup_paths(args) -> Dict[str, Path]:
    """Setup input/output paths based on environment."""
    is_kaggle = detect_kaggle()
    
    if args.batch_dir:
        batch_dir = Path(args.batch_dir)
    elif is_kaggle:
        batch_dir = Path("/kaggle/input/batches")
    else:
        batch_dir = Path("./batches")
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif is_kaggle:
        output_dir = Path("/kaggle/working/processing")
    else:
        output_dir = Path("./processing_output")
    
    cache_dir = output_dir / "cache"
    progress_file = output_dir / "progress.json"
    
    return {
        "batch_dir": batch_dir,
        "output_dir": output_dir,
        "cache_dir": cache_dir,
        "progress_file": progress_file,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MEMORY MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def release_memory(log):
    """Free memory without touching model weights."""
    gc.collect()
    if TORCH_OK and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def log_memory(log, label: str = ""):
    """Log current memory usage."""
    mem = psutil.virtual_memory()
    msg = f"RAM: {mem.used/1e9:.1f}/{mem.total/1e9:.1f} GB ({mem.percent}%)"
    
    if TORCH_OK and torch.cuda.is_available():
        try:
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            msg += f" | VRAM: {used:.1f}/{total:.1f} GB"
        except:
            pass
    
    if label:
        msg = f"{label} — {msg}"
    
    log.info(msg)


# ══════════════════════════════════════════════════════════════════════════════
# DOCLING EXTRACTOR SETUP
# ══════════════════════════════════════════════════════════════════════════════

def build_docling_extractors(log) -> tuple:
    """Create standard + OCR converters."""
    if not DOCLING_OK:
        log.error("docling not installed. Run: pip install docling")
        sys.exit(1)
    
    # Standard converter (for machine-readable PDFs)
    standard_opts = PdfPipelineOptions(
        do_table_structure=True,
    )
    standard_opts.table_structure_options.mode = TableFormerMode.ACCURATE
    standard_opts.table_structure_options.do_cell_matching = True
    
    converter_standard = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=standard_opts)
        }
    )
    
    # OCR converter (for scanned PDFs)
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
    
    log.info("✓ Docling converters ready (standard + OCR)")
    
    return converter_standard, converter_ocr


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_document(
    pdf_path: Path,
    converter_standard,
    converter_ocr,
    log,
    try_ocr: bool = False,
) -> Optional[ExtractedDocument]:
    """Extract text and tables from a PDF using docling."""
    
    start_time = time.time()
    doc_slug = pdf_path.stem.lower().replace(" ", "_").replace("-", "_")[:50]
    
    try:
        converter = converter_ocr if try_ocr else converter_standard
        
        doc = converter.convert(str(pdf_path))
        is_scanned = try_ocr
        
        blocks = []
        heading_path = []
        block_index = 0
        
        # Extract text blocks
        for item in doc.document.body:
            if isinstance(item, SectionHeaderItem):
                heading_path = [item.text]
                blocks.append(TextBlock(
                    text=item.text,
                    block_type="heading",
                    heading_path=heading_path.copy(),
                    page_number=item.prov[0].page_num if item.prov else 0,
                    block_index=block_index,
                ))
                block_index += 1
            
            elif isinstance(item, TextItem):
                blocks.append(TextBlock(
                    text=item.text,
                    block_type="paragraph",
                    heading_path=heading_path.copy(),
                    page_number=item.prov[0].page_num if item.prov else 0,
                    block_index=block_index,
                ))
                block_index += 1
            
            elif isinstance(item, ListItem):
                blocks.append(TextBlock(
                    text=item.text,
                    block_type="list_item",
                    heading_path=heading_path.copy(),
                    page_number=item.prov[0].page_num if item.prov else 0,
                    block_index=block_index,
                ))
                block_index += 1
        
        # Extract tables
        tables = []
        table_index = 0
        
        for item in doc.document.body:
            if isinstance(item, TableItem):
                try:
                    df = item.data.to_pandas()
                    
                    # Detect data type from headers
                    headers = " ".join(str(c) for c in df.columns)
                    has_proj = bool(__import__('re').search(
                        r"proj|est\.|estimate|forecast|target|budget",
                        headers,
                        __import__('re').IGNORECASE
                    ))
                    has_actual = bool(__import__('re').search(
                        r"actual|outturn|audited|preliminary",
                        headers,
                        __import__('re').IGNORECASE
                    ))
                    
                    if has_actual and has_proj:
                        data_type = "mixed"
                    elif has_actual:
                        data_type = "actual"
                    elif has_proj:
                        data_type = "projection"
                    else:
                        data_type = "mixed"
                    
                    table_id = f"{doc_slug}_table_{table_index:03d}"
                    
                    # Convert to markdown
                    markdown = df.to_markdown(index=False)
                    
                    tables.append(ExtractedTable(
                        table_id=table_id,
                        caption=item.caption or "",
                        heading_path=heading_path.copy(),
                        page_number=item.prov[0].page_num if item.prov else 0,
                        table_index=table_index,
                        rows=df.shape[0],
                        cols=df.shape[1],
                        markdown=markdown,
                        data_type=data_type,
                    ))
                    
                    table_index += 1
                
                except Exception as e:
                    log.warning(f"  ⚠ Table extraction failed: {e}")
                    continue
        
        elapsed = time.time() - start_time
        
        result = ExtractedDocument(
            source_file=str(pdf_path),
            doc_slug=doc_slug,
            total_pages=len(doc.pages),
            is_scanned=is_scanned,
            blocks=blocks,
            tables=tables,
            extraction_time=elapsed,
        )
        
        log.info(f"  ✓ {len(blocks)} blocks, {len(tables)} tables ({elapsed:.1f}s)")
        
        return result
    
    except Exception as e:
        error_msg = str(e)[:200]
        log.error(f"  ✗ Extraction failed: {error_msg}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PROGRESS TRACKING
# ══════════════════════════════════════════════════════════════════════════════

class ProgressTracker:
    """Track which documents have been processed."""
    
    def __init__(self, progress_file: Path):
        self.path = progress_file
        self.data = self._load()
    
    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        
        return {
            "completed": [],
            "failed": [],
            "skipped": [],
            "doc_count": 0,
            "table_count": 0,
            "total_time": 0.0,
        }
    
    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
    
    def is_done(self, fname: str) -> bool:
        return fname in self.data["completed"]
    
    def mark_complete(self, fname: str, elapsed: float, tables: int):
        if fname not in self.data["completed"]:
            self.data["completed"].append(fname)
        self.data["total_time"] += elapsed
        self.data["doc_count"] += 1
        self.data["table_count"] += tables
        self._save()
    
    def mark_failed(self, fname: str, error: str):
        self.data["failed"] = [e for e in self.data["failed"] if e["file"] != fname]
        self.data["failed"].append({"file": fname, "error": error[:100]})
        self._save()
    
    def summary(self) -> str:
        completed = len(self.data["completed"])
        failed = len(self.data["failed"])
        
        return (
            f"✓ {completed} docs | "
            f"✗ {failed} failed | "
            f"📊 {self.data['table_count']} tables | "
            f"⏱ {self.data['total_time']:.1f}s"
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Extract and process batches using Docling on Kaggle"
    )
    parser.add_argument("--batch-dir", type=str, help="Path to batches folder")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--config-csv", type=str, help="Path to central_inventory.csv")
    parser.add_argument("--force", action="store_true", help="Re-process cached docs")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    
    args = parser.parse_args()
    
    # Setup
    paths = setup_paths(args)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["cache_dir"].mkdir(parents=True, exist_ok=True)
    
    log = setup_logging(paths["output_dir"])
    
    log.info("╔═══════════════════════════════════════════════════════════════╗")
    log.info("║  Kaggle Docling Batch Processor                              ║")
    log.info("╚═══════════════════════════════════════════════════════════════╝")
    
    # Preflight checks
    log.info(f"Device: {get_device()}")
    log.info(f"Kaggle: {detect_kaggle()}")
    log.info(f"Batch dir: {paths['batch_dir']}")
    log.info(f"Output dir: {paths['output_dir']}")
    
    if not DOCLING_OK:
        log.error("docling not installed")
        sys.exit(1)
    
    if not paths["batch_dir"].exists():
        log.error(f"Batch directory not found: {paths['batch_dir']}")
        sys.exit(1)
    
    # Find all batches
    batch_dirs = sorted([d for d in paths["batch_dir"].iterdir() if d.is_dir()])
    
    if not batch_dirs:
        log.error("No batch folders found")
        sys.exit(1)
    
    log.info(f"Found {len(batch_dirs)} batches")
    
    # Find all PDFs
    pdf_files = []
    for batch_dir in batch_dirs:
        batch_name = batch_dir.name
        pdfs = sorted(batch_dir.glob("*.pdf"))
        log.info(f"  {batch_name}: {len(pdfs)} PDFs")
        pdf_files.extend([(batch_name, pdf) for pdf in pdfs])
    
    log.info(f"Total: {len(pdf_files)} PDFs across batches")
    
    if args.dry_run:
        log.info("DRY RUN — no extraction will occur")
        return
    
    log_memory(log, "Initial")
    
    # Setup docling
    converter_std, converter_ocr = build_docling_extractors(log)
    
    # Progress tracking
    tracker = ProgressTracker(paths["progress_file"])
    
    # Process each PDF
    for batch_name, pdf_path in pdf_files:
        log.info(f"\n📄 {batch_name} / {pdf_path.name}")
        
        # Skip if already done
        if tracker.is_done(pdf_path.name) and not args.force:
            log.info("  (already processed, skipping)")
            continue
        
        # Check cache first
        cache_file = paths["cache_dir"] / f"{pdf_path.stem}.json"
        if cache_file.exists() and not args.force:
            log.info("  (cached)")
            # Load from cache to update tracker
            with open(cache_file) as f:
                cached = json.load(f)
                tracker.mark_complete(pdf_path.name, 0, len(cached.get("tables", [])))
            continue
        
        # Extract
        result = extract_document(
            pdf_path,
            converter_std,
            converter_ocr,
            log,
            try_ocr=False,
        )
        
        if not result:
            tracker.mark_failed(pdf_path.name, "extraction failed")
            continue
        
        # Save result
        output_data = {
            "source_file": result.source_file,
            "doc_slug": result.doc_slug,
            "total_pages": result.total_pages,
            "is_scanned": result.is_scanned,
            "extraction_time": result.extraction_time,
            "blocks": [asdict(b) for b in result.blocks],
            "tables": [asdict(t) for t in result.tables],
        }
        
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Update tracker
        tracker.mark_complete(pdf_path.name, result.extraction_time, len(result.tables))
        
        # Memory cleanup
        release_memory(log)
        if len(pdf_files) % 10 == 0:
            log_memory(log, "Checkpoint")
    
    # Summary
    log.info("\n" + "="*70)
    log.info(f"COMPLETE: {tracker.summary()}")
    log.info("="*70)
    
    log_memory(log, "Final")


if __name__ == "__main__":
    main()
