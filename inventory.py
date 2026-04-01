#!/usr/bin/env python3
"""
inventory.py — Step 1 of 4 in the multi-agent RAG pipeline.

Scans all agent data/raw/ folders, records every PDF, flags obvious
naming problems, and outputs a CSV for human review before anything
is renamed or configured.

Usage:
    python inventory.py
    python inventory.py --base ~/PRES/Executive
    python inventory.py --base ~/PRES/Executive --output my_inventory.csv

Output:
    inventory.csv with columns:
        agent | original_filename | filepath | size_kb | flags
"""

import os
import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict


# ── Configuration ────────────────────────────────────────────────────────────

AGENTS = [
    "Education",
    "Agriculture",
    "ICT",
    "Infastructure",
    "AntiCorruption",
    # "President",   # no PDFs yet — skip
    # "Finance",     # locked — skip
]

DEFAULT_BASE = os.path.expanduser("~/PRES/Executive")
DEFAULT_OUTPUT = "inventory.csv"


# ── Flag detectors ────────────────────────────────────────────────────────────

def detect_flags(filename: str, all_filenames_in_agent: list[str]) -> list[str]:
    """
    Returns a list of flag strings describing problems with this filename.
    Empty list = no problems detected.
    """
    flags = []
    stem = Path(filename).stem
    name_lower = filename.lower()

    # Duplicate suffix: file (1).pdf, file (2).pdf, etc.
    if re.search(r'\s*\(\d+\)\s*\.pdf$', filename, re.IGNORECASE):
        flags.append("DUPLICATE_SUFFIX")

    # Browser download garbage extensions
    if ".coredownload" in name_lower:
        flags.append("COREDOWNLOAD_GARBAGE")
    if name_lower.endswith(".pdf.pdf"):
        flags.append("DOUBLE_PDF_EXTENSION")
    if ".inline" in name_lower:
        flags.append("INLINE_GARBAGE")

    # Suspiciously short stem (likely auto-named: "document", "file", "report")
    if len(stem.strip()) <= 8:
        flags.append("SHORT_NAME")

    # Generic names that tell us nothing
    generic_patterns = [
        r'^document\d*$',
        r'^file\d*$',
        r'^report\d*$',
        r'^download\d*$',
        r'^untitled\d*$',
        r'^new\s',
    ]
    for pattern in generic_patterns:
        if re.match(pattern, stem.strip(), re.IGNORECASE):
            flags.append("GENERIC_NAME")
            break

    # All caps — usually okay but worth flagging for review
    if stem == stem.upper() and len(stem) > 10:
        flags.append("ALL_CAPS")

    # Very long filename — may cause OS/filesystem issues
    if len(filename) > 180:
        flags.append("VERY_LONG_NAME")

    # Contains special characters that break slugs
    if re.search(r'[&@#$%^*+=\[\]{}|\\<>?]', stem):
        flags.append("SPECIAL_CHARS")

    # Spaces in filename (will be slugified — just informational)
    if ' ' in filename:
        flags.append("HAS_SPACES")

    # Check for duplicate filenames within the same agent folder
    # (same name appearing more than once — shouldn't happen with PDFs but worth catching)
    if all_filenames_in_agent.count(filename) > 1:
        flags.append("DUPLICATE_FILENAME")

    return flags


# ── Main ──────────────────────────────────────────────────────────────────────

def scan_agent(agent: str, base_path: Path) -> list[dict]:
    """Scan one agent's data/raw/ folder and return list of record dicts."""
    raw_dir = base_path / agent / "data" / "raw"

    if not raw_dir.exists():
        print(f"  [WARN] {agent}: data/raw/ not found at {raw_dir} — skipping")
        return []

    pdfs = sorted([f for f in raw_dir.iterdir() if f.suffix.lower() == ".pdf"])

    if not pdfs:
        print(f"  [WARN] {agent}: no PDFs found in {raw_dir}")
        return []

    all_filenames = [f.name for f in pdfs]
    records = []

    for pdf in pdfs:
        size_kb = round(pdf.stat().st_size / 1024, 1)
        flags = detect_flags(pdf.name, all_filenames)
        flag_str = "|".join(flags) if flags else ""

        records.append({
            "agent":             agent,
            "original_filename": pdf.name,
            "filepath":          str(pdf.resolve()),
            "size_kb":           size_kb,
            "flags":             flag_str,
        })

    return records


def main():
    parser = argparse.ArgumentParser(description="Inventory all agent PDF corpora.")
    parser.add_argument("--base",   default=DEFAULT_BASE,   help="Base directory containing agent folders")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV filename")
    args = parser.parse_args()

    base_path = Path(args.base).expanduser().resolve()
    print(f"\nBase directory : {base_path}")
    print(f"Agents to scan : {', '.join(AGENTS)}")
    print(f"Output file    : {args.output}\n")

    if not base_path.exists():
        print(f"[ERROR] Base directory not found: {base_path}")
        return

    all_records = []
    agent_stats = {}

    for agent in AGENTS:
        print(f"Scanning {agent}...")
        records = scan_agent(agent, base_path)
        all_records.extend(records)

        flagged = [r for r in records if r["flags"]]
        agent_stats[agent] = {
            "total":   len(records),
            "flagged": len(flagged),
        }
        print(f"  {len(records)} PDFs found, {len(flagged)} flagged\n")

    # Write CSV
    fieldnames = ["agent", "original_filename", "filepath", "size_kb", "flags"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    # Summary
    print("─" * 60)
    print("SUMMARY")
    print("─" * 60)
    total = 0
    total_flagged = 0
    for agent, stats in agent_stats.items():
        total += stats["total"]
        total_flagged += stats["flagged"]
        flag_note = f"  ← {stats['flagged']} need review" if stats["flagged"] else ""
        print(f"  {agent:<20} {stats['total']:>4} PDFs{flag_note}")

    print("─" * 60)
    print(f"  {'TOTAL':<20} {total:>4} PDFs,  {total_flagged} flagged")
    print(f"\nInventory written to: {args.output}")

    # Flag type breakdown
    flag_counts = defaultdict(int)
    for r in all_records:
        for flag in r["flags"].split("|"):
            if flag:
                flag_counts[flag] += 1

    if flag_counts:
        print("\nFlag breakdown:")
        for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            print(f"  {flag:<30} {count}")
    else:
        print("\nNo flags detected — filenames look clean.")

    print()


if __name__ == "__main__":
    main()
