#!/usr/bin/env python3
"""
Copy missing files from raw to batch_6.
"""

from pathlib import Path
import shutil

raw_dir = Path("/home/darwin/PRES/Executive/data/raw")
batch_6_dir = Path("/home/darwin/PRES/Executive/batches/batch_6")

# Get all filenames in raw folder
raw_files = {f.name for f in raw_dir.glob("*") if f.is_file()}

# Get all filenames in batches folder
batches_dir = Path("/home/darwin/PRES/Executive/batches")
batch_files = {f.name for f in batches_dir.rglob("*") if f.is_file()}

# Find missing files
missing_files = raw_files - batch_files

# Create batch_6 directory
batch_6_dir.mkdir(exist_ok=True)

print(f"Creating batch_6 with {len(missing_files)} documents...\n")

# Copy missing files to batch_6
for filename in sorted(missing_files):
    src = raw_dir / filename
    dest = batch_6_dir / filename
    shutil.copy2(src, dest)
    print(f"✓ {filename}")

print(f"\n✓ Successfully created batch_6 with {len(missing_files)} documents")

# Verify
batch_6_count = len(list(batch_6_dir.glob("*")))
print(f"Files in batch_6: {batch_6_count}")
