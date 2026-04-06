#!/usr/bin/env python3
"""
move_to_central.py — Consolidates all sector PDFs and existing chunks into one store.

Moves:
  Finance/data/raw/*.pdf        -> Executive/data/raw/
  Finance/data/chunks/*.jsonl   -> Executive/data/chunks/
  {Sector}/data/raw/*.pdf       -> Executive/data/raw/   (other sectors)

Writes:
  central_inventory.csv  — master record: agent, filename, central_path, chunked (yes/no)

Usage:
    python move_to_central.py --dry-run
    python move_to_central.py --apply
"""

import csv
import shutil
import argparse
from pathlib import Path

CENTRAL_RAW    = Path("data/raw")
CENTRAL_CHUNKS = Path("data/chunks")
CENTRAL_CSV    = "central_inventory.csv"

SECTORS = [
    "Finance",
    "Agriculture",
    "Education",
    "ICT",
    "Infastructure",
    "AntiCorruption",
    "President",
]

FIELDNAMES = [
    "agent", "filename", "central_path", "chunked", "chunk_path",
]


def normalise(stem: str) -> str:
    return stem.lower().replace(" ", "_").replace("-", "_").replace(".", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--apply",   action="store_true")
    args = parser.parse_args()
    if args.apply:
        args.dry_run = False

    # Build existing chunk index
    existing_chunks = {}
    finance_chunk_dir = Path("Finance/data/chunks")
    if finance_chunk_dir.exists():
        for jf in finance_chunk_dir.glob("*.jsonl"):
            existing_chunks[normalise(jf.stem)] = jf
    if CENTRAL_CHUNKS.exists():
        for jf in CENTRAL_CHUNKS.glob("*.jsonl"):
            existing_chunks[normalise(jf.stem)] = jf

    # Collect all PDFs across sectors
    pdf_moves   = []
    chunk_moves = []

    for sector in SECTORS:
        raw_dir = Path(sector) / "data" / "raw"
        if not raw_dir.exists():
            continue
        for pdf in sorted(raw_dir.glob("*.pdf")):
            dst = CENTRAL_RAW / pdf.name
            pdf_moves.append((pdf, dst, sector))

    if finance_chunk_dir.exists():
        for jf in sorted(finance_chunk_dir.glob("*.jsonl")):
            dst = CENTRAL_CHUNKS / jf.name
            chunk_moves.append((jf, dst))

    print("PLAN")
    print("  PDFs to move   : " + str(len(pdf_moves)))
    print("  Chunks to move : " + str(len(chunk_moves)))
    print("  Dest raw       : " + str(CENTRAL_RAW))
    print("  Dest chunks    : " + str(CENTRAL_CHUNKS))
    print("")

    conflicts = [dst.name for src, dst, ag in pdf_moves if dst.exists() and dst != src]
    if conflicts:
        print("WARNING: " + str(len(conflicts)) + " filename conflicts:")
        for c in conflicts:
            print("  " + c)
        print("")

    if args.dry_run:
        print("Sample (first 10 PDFs):")
        for src, dst, agent in pdf_moves[:10]:
            print("  [" + agent + "] " + src.name)
        print("  ...")
        print("\nDry run. Add --apply to execute.")
        return

    CENTRAL_RAW.mkdir(parents=True, exist_ok=True)
    CENTRAL_CHUNKS.mkdir(parents=True, exist_ok=True)

    moved_pdfs   = 0
    moved_chunks = 0
    errors       = 0

    for src, dst, agent in pdf_moves:
        if dst.exists() and dst != src:
            continue
        try:
            shutil.move(str(src), str(dst))
            moved_pdfs += 1
        except Exception as e:
            print("  [ERROR] " + src.name + ": " + str(e))
            errors += 1

    for src, dst in chunk_moves:
        if dst.exists():
            continue
        try:
            shutil.move(str(src), str(dst))
            moved_chunks += 1
        except Exception as e:
            print("  [ERROR] " + src.name + ": " + str(e))
            errors += 1

    # Rebuild chunk index after moves
    final_chunks = {}
    for jf in CENTRAL_CHUNKS.glob("*.jsonl"):
        final_chunks[normalise(jf.stem)] = jf

    # Write central inventory
    inventory = []
    agent_map = {dst.name: ag for src, dst, ag in pdf_moves}

    for pdf in sorted(CENTRAL_RAW.glob("*.pdf")):
        norm       = normalise(pdf.stem)
        chunk_path = final_chunks.get(norm, "")
        chunked    = "yes" if chunk_path else "no"
        agent      = agent_map.get(pdf.name, "unknown")

        inventory.append({
            "agent":        agent,
            "filename":     pdf.name,
            "central_path": str(pdf),
            "chunked":      chunked,
            "chunk_path":   str(chunk_path) if chunk_path else "",
        })

    with open(CENTRAL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(inventory)

    chunked_count   = sum(1 for r in inventory if r["chunked"] == "yes")
    unchunked_count = sum(1 for r in inventory if r["chunked"] == "no")

    print("Done.")
    print("  PDFs moved     : " + str(moved_pdfs))
    print("  Chunks moved   : " + str(moved_chunks))
    print("  Errors         : " + str(errors))
    print("")
    print("Inventory: " + CENTRAL_CSV)
    print("  Total PDFs   : " + str(len(inventory)))
    print("  Chunked      : " + str(chunked_count) + "  (will skip)")
    print("  Need chunking: " + str(unchunked_count))
    print("")
    print("Next: python tag_relationships.py")


if __name__ == "__main__":
    main()