#!/bin/bash
# copy_batches_to_downloads.sh
# Copies the batches/ folders to Windows Downloads

SOURCE_DIR="./batches"
DEST_BASE="/mnt/c/Users/User/Downloads"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: $SOURCE_DIR not found."
    exit 1
fi

if [ ! -d "$DEST_BASE" ]; then
    echo "Error: Destination path $DEST_BASE does not exist."
    echo "Please ensure your Windows Downloads folder is accessible."
    exit 1
fi

DEST_DIR="$DEST_BASE/batches"

# Remove existing if present
if [ -d "$DEST_DIR" ]; then
    echo "Removing existing batches folder..."
    rm -rf "$DEST_DIR"
fi

echo "Creating batches folder in Downloads..."
mkdir -p "$DEST_DIR"

echo "Copying batch folders (this may take a few minutes)..."
echo "Using rsync for reliable transfer..."

rsync -av --progress "$SOURCE_DIR"/ "$DEST_DIR/"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Success! Batches copied to: $DEST_BASE/batches/"
    echo "You can now access:"
    ls -d "$DEST_DIR"/batch_* 2>/dev/null | while read batch; do
        echo "  - $(basename $batch)"
    done
else
    echo "Error during copy operation."
    exit 1
fi
