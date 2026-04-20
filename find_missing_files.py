#!/usr/bin/env python3
"""
Find files in raw that are not in batches.
"""

from pathlib import Path

# Get all filenames in raw folder
raw_dir = Path("/home/darwin/PRES/Executive/data/raw")
raw_files = {f.name for f in raw_dir.glob("*") if f.is_file()}

# Get all filenames in batches folder
batches_dir = Path("/home/darwin/PRES/Executive/batches")
batch_files = {f.name for f in batches_dir.rglob("*") if f.is_file()}

# Find missing files
missing_files = raw_files - batch_files

print(f"Total files in raw: {len(raw_files)}")
print(f"Total files in batches: {len(batch_files)}")
print(f"Missing files: {len(missing_files)}\n")

# Show missing files sorted
for filename in sorted(missing_files):
    print(filename)
