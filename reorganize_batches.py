#!/usr/bin/env python3
"""
Reorganize batches into chunks of ~70 documents each.
"""

import os
import shutil
from pathlib import Path
import math

# Configuration
BATCHES_DIR = Path("/home/darwin/PRES/Executive/batches")
DOCS_PER_BATCH = 70
OUTPUT_DIR = Path("/home/darwin/PRES/Executive/batches_reorganized")

def main():
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Collect all documents from existing batches
    all_documents = []
    for batch_dir in sorted(BATCHES_DIR.glob("batch_*")):
        if batch_dir.is_dir():
            documents = sorted([f for f in batch_dir.glob("*") if f.is_file()])
            all_documents.extend(documents)
    
    print(f"Total documents found: {len(all_documents)}")
    
    # Calculate number of new batches needed
    num_batches = math.ceil(len(all_documents) / DOCS_PER_BATCH)
    print(f"Reorganizing into {num_batches} batches of ~{DOCS_PER_BATCH} documents")
    
    # Create new batches
    for batch_num in range(1, num_batches + 1):
        batch_name = f"batch_{batch_num}"
        batch_dir = OUTPUT_DIR / batch_name
        batch_dir.mkdir(exist_ok=True)
        
        # Calculate start and end indices for this batch
        start_idx = (batch_num - 1) * DOCS_PER_BATCH
        end_idx = min(batch_num * DOCS_PER_BATCH, len(all_documents))
        
        batch_docs = all_documents[start_idx:end_idx]
        
        print(f"\n{batch_name}: {len(batch_docs)} documents")
        
        # Copy documents to new batch directory
        for doc in batch_docs:
            dest = batch_dir / doc.name
            shutil.copy2(doc, dest)
            print(f"  → {doc.name}")
    
    print(f"\n✓ Reorganization complete!")
    print(f"New batches saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
