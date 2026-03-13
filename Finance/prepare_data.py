"""
prepare_data.py — Extraction orchestrator for Kenya AI Executive Roundtable

Runs all 170 Finance CS documents through:
    extractor.py → table_processor.py → data/cache/*.json

Designed to run on Kaggle GPU (T4/P100) with full resume support.
If the session times out, re-run and it skips already-cached documents.
here we are

Usage:
    python prepare_data.py                        # auto-detects Kaggle or local
    python prepare_data.py --dry-run              # show what would run, no extraction
    python prepare_data.py --force                # re-extract even if cached
    python prepare_data.py --raw-dir /path/to/pdfs --cache-dir /path/to/cache
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.extractor import Extractor
from pipeline.table_processor import TableProcessor

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "prepare_data.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def find_pdf_dir(base: Path) -> Path:
    """Walk base directory and return the first folder containing PDFs."""
    for root, dirs, files in os.walk(base):
        if any(f.endswith(".pdf") for f in files):
            return Path(root)
    return base


def detect_environment(args) -> dict:
    kaggle_input   = Path("/kaggle/input")
    kaggle_working = Path("/kaggle/working")

    # Command-line overrides take highest priority
    if args.raw_dir:
        raw_dir = Path(args.raw_dir)
    elif kaggle_input.exists():
        raw_dir = find_pdf_dir(kaggle_input)
        log.info(f"Auto-detected PDF directory: {raw_dir}")
    else:
        raw_dir = ROOT / "data" / "raw"

    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    elif kaggle_input.exists():
        cache_dir = kaggle_working / "cache"
    else:
        cache_dir = ROOT / "data" / "cache"

    if kaggle_input.exists():
        log_dir  = kaggle_working
        device   = "cuda"
        env_name = "kaggle"
    else:
        log_dir  = ROOT
        device   = "cpu"
        env_name = "local"

    return {
        "env_name":  env_name,
        "raw_dir":   raw_dir,
        "cache_dir": cache_dir,
        "log_dir":   log_dir,
        "device":    device,
        "config":    ROOT / "config.yaml",
    }


# ══════════════════════════════════════════════════════════════════════════════
# PROGRESS TRACKING
# ══════════════════════════════════════════════════════════════════════════════

class ProgressTracker:
    def __init__(self, log_dir: Path):
        self.path = log_dir / "extraction_progress.json"
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        return {"completed": [], "failed": [], "skipped": [], "timing": {}}

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def is_done(self, fname: str) -> bool:
        return fname in self.data["completed"]

    def mark_complete(self, fname: str, elapsed: float):
        if fname not in self.data["completed"]:
            self.data["completed"].append(fname)
        self.data["timing"][fname] = round(elapsed, 1)
        self._save()

    def mark_failed(self, fname: str, error: str):
        # Remove from failed if retrying
        self.data["failed"] = [e for e in self.data["failed"] if e["file"] != fname]
        self.data["failed"].append({"file": fname, "error": str(error)[:200]})
        self._save()

    def mark_skipped(self, fname: str, reason: str):
        if fname not in self.data["skipped"]:
            self.data["skipped"].append(fname)
        self._save()

    def summary(self) -> str:
        return (
            f"completed={len(self.data['completed'])}  "
            f"failed={len(self.data['failed'])}  "
            f"skipped={len(self.data['skipped'])}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTOR SETUP
# ══════════════════════════════════════════════════════════════════════════════

def build_extractor(device: str) -> Extractor:
    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                log.info(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                log.warning("CUDA not available — falling back to CPU")
                device = "cpu"
        except ImportError:
            log.warning("torch not importable — falling back to CPU")
            device = "cpu"

    if device == "cuda":
        try:
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                AcceleratorOptions,
                TableFormerMode,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat

            opts = PdfPipelineOptions(
                do_table_structure=True,
                accelerator_options=AcceleratorOptions(
                    num_threads=4,
                    device="cuda",
                ),
            )
            opts.table_structure_options.mode = TableFormerMode.ACCURATE
            opts.table_structure_options.do_cell_matching = True

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=opts)
                }
            )
            extractor = Extractor.__new__(Extractor)
            extractor._converter_standard = converter
            extractor._converter_ocr = converter
            log.info("Extractor ready — GPU ACCURATE mode")
            return extractor

        except Exception as e:
            log.warning(f"GPU extractor setup failed ({e}) — falling back to CPU")

    extractor = Extractor()
    log.info("Extractor ready — CPU ACCURATE mode")
    return extractor


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def load_documents(config_path: Path) -> list[dict]:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    docs = config.get("documents", [])
    log.info(f"Loaded {len(docs)} documents from config.yaml")
    return docs


CATEGORY_ORDER = [2, 1, 3, 4, 5, 6, 7, 8, 0]


def sort_documents(documents: list[dict]) -> list[dict]:
    rank = {cat: i for i, cat in enumerate(CATEGORY_ORDER)}
    return sorted(
        documents,
        key=lambda d: (rank.get(d.get("category", 0), 99), d["filename"]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir",   default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--force",     action="store_true")
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args()

    env = detect_environment(args)

    raw_dir   = env["raw_dir"]
    cache_dir = env["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Environment : {env['env_name']}")
    log.info(f"Raw PDFs    : {raw_dir}")
    log.info(f"Cache dir   : {cache_dir}")
    log.info(f"Device      : {env['device']}")

    if not env["config"].exists():
        log.error(f"config.yaml not found at {env['config']}")
        sys.exit(1)

    documents = sort_documents(load_documents(env["config"]))
    tracker   = ProgressTracker(env["log_dir"])
    log.info(f"Resume state: {tracker.summary()}")

    # Dry run
    if args.dry_run:
        log.info("\n── DRY RUN ──────────────────────────────────────────")
        for i, doc in enumerate(documents, 1):
            fname  = doc["filename"]
            cached = (cache_dir / f"{Path(fname).stem.lower().replace(' ', '_').replace('-', '_')}.json").exists()
            done   = tracker.is_done(fname)
            status = "DONE  " if done else ("CACHED" if cached else "PENDING")
            log.info(f"  {i:3}. [{status}] {fname[:65]}")
        log.info(f"\nTotal: {len(documents)} documents")
        return

    extractor = build_extractor(env["device"])
    processor = TableProcessor()

    total       = len(documents)
    done_count  = 0
    fail_count  = 0
    skip_count  = 0
    session_t0  = time.time()

    log.info(f"\nStarting — {total} documents\n")

    for i, doc_config in enumerate(documents, 1):
        fname    = doc_config["filename"]
        pdf_path = raw_dir / fname

        log.info(f"[{i:3}/{total}] {fname[:70]}")

        # Already done
        if not args.force and tracker.is_done(fname):
            log.info("  → already cached, skipping")
            skip_count += 1
            continue

        # PDF missing
        if not pdf_path.exists():
            log.warning(f"  → PDF not found at {pdf_path}")
            tracker.mark_skipped(fname, "pdf_not_found")
            skip_count += 1
            continue

        t0 = time.time()
        try:
            raw_doc = extractor.extract(pdf_path, doc_config, cache_dir=cache_dir)

            if raw_doc.error:
                raise RuntimeError(raw_doc.error)

            processed = processor.process(raw_doc, doc_config)
            elapsed   = time.time() - t0

            log.info(
                f"  → {len(processed.blocks)} blocks, "
                f"{len(processed.table_chunks)} table chunks  "
                f"[{elapsed:.0f}s]"
            )
            tracker.mark_complete(fname, elapsed)
            done_count += 1

        except Exception as e:
            elapsed = time.time() - t0
            log.error(f"  → FAILED [{elapsed:.0f}s]: {e}")
            tracker.mark_failed(fname, str(e))
            fail_count += 1

    session_elapsed = time.time() - session_t0
    log.info(f"\n{'='*60}")
    log.info(f"Session complete — {session_elapsed/60:.1f} min")
    log.info(f"  Processed : {done_count}")
    log.info(f"  Skipped   : {skip_count}")
    log.info(f"  Failed    : {fail_count}")
    log.info(f"  Overall   : {tracker.summary()}")
    log.info(f"{'='*60}")

    if fail_count > 0:
        log.info("\nFailed files:")
        for entry in tracker.data["failed"]:
            log.info(f"  {entry['file']}: {entry['error'][:100]}")


if __name__ == "__main__":
    main()