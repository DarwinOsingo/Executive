#!/bin/bash
# copy_batches_to_downloads.sh
# Copies the batches/ folders to /mnt/c/Users/User/Downloads/

SOURCE_DIR="./batches"
DEST_BASE="/mnt/c/Users/User/Downloads"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: $SOURCE_DIR not found. Run create_batch_folders.py first."
    exit 1
fi

echo "Copying batch folders to $DEST_BASE ..."
cp -r "$SOURCE_DIR" "$DEST_BASE/"

echo "Done. Folders are in $DEST_BASE/batches/"
