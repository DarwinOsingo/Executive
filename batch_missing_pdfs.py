#!/usr/bin/env python3
"""
create_batch_folders.py

Reads missing_pdfs.txt (list of absolute PDF paths) and copies the files
into batch folders: batch_1/, batch_2/, batch_3/.
"""

import shutil
from pathlib import Path

INPUT_FILE = Path("missing_pdfs.txt")
BATCH_COUNT = 3
OUTPUT_BASE = Path("batches")  # will create batch_1, batch_2, batch_3 inside

if not INPUT_FILE.exists():
    print(f"ERROR: {INPUT_FILE} not found.")
    exit(1)

with open(INPUT_FILE, "r") as f:
    paths = [Path(line.strip()) for line in f if line.strip()]

total = len(paths)
batch_size = total // BATCH_COUNT
remainder = total % BATCH_COUNT

print(f"Total missing PDFs: {total}")
print(f"Creating {BATCH_COUNT} batch folders in '{OUTPUT_BASE}/'...")

start = 0
for i in range(BATCH_COUNT):
    extra = 1 if i < remainder else 0
    end = start + batch_size + extra
    batch_paths = paths[start:end]
    
    batch_dir = OUTPUT_BASE / f"batch_{i+1}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    missing = 0
    for pdf_path in batch_paths:
        if pdf_path.exists():
            shutil.copy2(pdf_path, batch_dir / pdf_path.name)
            copied += 1
        else:
            print(f"  WARNING: Missing file: {pdf_path}")
            missing += 1
    
    print(f"  batch_{i+1}: {copied} PDFs copied" + (f" ({missing} missing)" if missing else ""))
    start = end

print(f"\nDone. Folders are in {OUTPUT_BASE.resolve()}/")