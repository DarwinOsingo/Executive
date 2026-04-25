#!/usr/bin/env python3
"""
audit_processed.py
──────────────────
Scans data/processed/ for docling JSONL output and diagnoses quality issues.

Usage:
    python audit_processed.py --input /home/darwin/PRES/Executive/data/processed/
    python audit_processed.py --input ./data/processed/ --output audit_report.csv
    python audit_processed.py --input ./data/processed/ --show-bad-only

Diagnoses:
    OK           — healthy, prose-rich content
    EMPTY        — zero text extracted (docling returned nothing)
    MALFORMED    — file is not valid JSON / JSONL at all
    TABLE_ONLY   — content exists but is almost entirely table cells, no prose
    SHORT        — extracted text is suspiciously short (< MIN_CHARS threshold)
    SPARSE       — very few chunks relative to file size on disk
"""

import argparse
import json
import os
import re
import sys
import csv
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ── Sitrep helpers ────────────────────────────────────────────────────────────
_sitrep_start = time.monotonic()

def _elapsed() -> str:
    secs = int(time.monotonic() - _sitrep_start)
    return f"{secs // 60:02d}:{secs % 60:02d}"

def sitrep(msg: str, end: str = "\n") -> None:
    """Print a timestamped status line, overwriting the current line if end=''."""
    print(f"\r  [{_elapsed()}] {msg}", end=end, flush=True)

def sitrep_inline(msg: str) -> None:
    """Overwrite the current terminal line (for per-file spinners)."""
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 100
    line = f"  [{_elapsed()}] {msg}"
    print(f"\r{line:<{cols}}", end="", flush=True)
# ─────────────────────────────────────────────────────────────────────────────

# ── Thresholds (tune these to your corpus) ────────────────────────────────────
MIN_CHARS          = 500      # below this → SHORT
TABLE_RATIO_CUTOFF = 0.80     # if >80% of text looks like table cells → TABLE_ONLY
SHORT_CHUNK_CHARS  = 80       # a chunk with fewer chars than this is considered "thin"
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that suggest table-cell content rather than prose
TABLE_CELL_PATTERN = re.compile(
    r"^\s*(\|.*\|[\s\|]*$"          # markdown pipe tables
    r"|[-─═]+\s*$"                   # separator lines
    r"|\S+(\s+\S+){0,4}\s*$"        # very short lines (≤5 tokens) typical of table cells
    r")",
    re.MULTILINE,
)


@dataclass
class FileAudit:
    filename:        str
    status:          str          # OK | EMPTY | MALFORMED | TABLE_ONLY | SHORT | SPARSE
    issues:          list = field(default_factory=list)
    num_records:     int  = 0
    total_chars:     int  = 0
    thin_chunks:     int  = 0     # chunks below SHORT_CHUNK_CHARS
    table_ratio:     float= 0.0
    file_bytes:      int  = 0
    first_text_preview: str = ""


PROSE_BLOCK_TYPES = {"paragraph", "list_item", "heading", "caption"}


def _extract_text_from_record(doc: dict) -> tuple:
    """
    Extract prose from a docling cache JSON object (real schema).

    Schema:
        {"blocks": [{"block_type": str, "text": str, ...}], "tables": [...]}

    Returns (prose_text, prose_block_count, table_count).
    """
    blocks = doc.get("blocks") or []
    tables = doc.get("tables") or []

    parts = []
    prose_count = 0
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("block_type") in PROSE_BLOCK_TYPES:
            text = block.get("text", "").strip()
            if text:
                parts.append(text)
                prose_count += 1

    return "\n\n".join(parts), prose_count, len(tables)



    # Fallback — stringify every string value found at depth ≤ 2
    parts = []
    for value in record.values():
        if isinstance(value, str) and len(value) > 10:
            parts.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    for sub_val in item.values():
                        if isinstance(sub_val, str) and len(sub_val) > 10:
                            parts.append(sub_val)
    return "\n".join(parts)


def _table_ratio(text: str) -> float:
    """Estimate fraction of lines that look like table cells rather than prose."""
    if not text.strip():
        return 0.0
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    table_lines = sum(1 for line in lines if TABLE_CELL_PATTERN.match(line))
    return table_lines / len(lines)


def audit_file(filepath: Path) -> FileAudit:
    filename   = filepath.name
    file_bytes = filepath.stat().st_size
    audit      = FileAudit(filename=filename, status="OK", file_bytes=file_bytes)

    # ── Read raw bytes ────────────────────────────────────────────────────────
    try:
        raw = filepath.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as read_err:
        audit.status = "MALFORMED"
        audit.issues.append(f"Cannot read file: {read_err}")
        return audit

    if not raw:
        audit.status = "EMPTY"
        audit.issues.append("File is completely empty (0 bytes of text)")
        return audit

    # ── Parse — docling cache files are single JSON objects ─────────────────
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as json_err:
        audit.status = "MALFORMED"
        audit.issues.append(f"Not valid JSON: {json_err}")
        return audit

    if not isinstance(doc, dict):
        audit.status = "MALFORMED"
        audit.issues.append(f"Expected JSON object, got {type(doc).__name__}")
        return audit

    # ── Extract prose text ───────────────────────────────────────────────────
    full_text, prose_count, num_tables = _extract_text_from_record(doc)
    audit.num_records = prose_count
    audit.total_chars = len(full_text)
    audit.thin_chunks = num_tables   # repurposed: table count

    if num_tables > 0 and prose_count == 0:
        audit.issues.append(
            f"Has {num_tables} table(s) but zero prose blocks — "
            f"possible tables-only doc or missed body text"
        )

    if full_text:
        # Preview: first 120 printable chars
        preview = " ".join(full_text.split())[:120]
        audit.first_text_preview = preview

    # ── Diagnose ──────────────────────────────────────────────────────────────
    if audit.total_chars == 0:
        audit.status = "EMPTY"
        audit.issues.append("All records extracted to empty text")
        return audit

    audit.table_ratio = _table_ratio(full_text)

    if audit.table_ratio > TABLE_RATIO_CUTOFF:
        audit.status = "TABLE_ONLY"
        audit.issues.append(
            f"~{audit.table_ratio:.0%} of lines look like table cells — "
            f"docling may have extracted only table markup, no prose"
        )

    elif audit.total_chars < MIN_CHARS:
        audit.status = "SHORT"
        audit.issues.append(
            f"Only {audit.total_chars} chars extracted "
            f"(threshold: {MIN_CHARS}). Possible extraction failure."
        )

    # Extra warnings (don't override status, just annotate)
    if thin_count > 0 and audit.num_records > 0:
        thin_pct = thin_count / audit.num_records
        if thin_pct > 0.5:
            audit.issues.append(
                f"{thin_count}/{audit.num_records} chunks are thin "
                f"(< {SHORT_CHUNK_CHARS} chars) — may be header/table fragments"
            )

    if parse_errors and audit.status == "OK":
        audit.status = "MALFORMED"   # partial parse — flag it

    return audit


def _collect_files(input_dir: Path) -> list[Path]:
    """
    Fast flat + one-level-deep scan using os.scandir instead of rglob.
    rglob walks the entire tree silently before yielding anything — on
    a large corpus that stalls for seconds before a single file is found.
    """
    found: list[Path] = []
    extensions = {".jsonl", ".json"}

    def _scan(directory: Path) -> None:
        try:
            with os.scandir(directory) as scanner:
                for entry in scanner:
                    if entry.is_file(follow_symlinks=False):
                        if Path(entry.name).suffix.lower() in extensions:
                            found.append(Path(entry.path))
                    elif entry.is_dir(follow_symlinks=False):
                        _scan(Path(entry.path))   # recurse one level
        except PermissionError:
            pass

    sitrep(f"Scanning {input_dir} …", end="")
    _scan(input_dir)
    found.sort()
    print(f"\r  [{_elapsed()}] Found {len(found)} file(s) to audit.{' ' * 20}")
    return found


def run_audit(input_dir: Path, show_bad_only: bool, output_csv: Optional[Path]):
    jsonl_files = _collect_files(input_dir)

    if not jsonl_files:
        print(f"[!] No .jsonl / .json files found in {input_dir}")
        sys.exit(1)

    results: list[FileAudit] = []
    total = len(jsonl_files)

    for index, filepath in enumerate(jsonl_files, start=1):
        kb = filepath.stat().st_size / 1024
        sitrep_inline(
            f"Auditing [{index}/{total}]  {filepath.name[:50]:<50}  ({kb:.1f} KB)"
        )
        result = audit_file(filepath)
        results.append(result)
        # Print a one-liner immediately for bad files so you see problems as they appear
        if result.status != "OK":
            print()   # newline after the inline overwrite
            sitrep(f"  ⚠  {result.status:<12} {result.filename}")

    print()  # final newline after spinner

    # ── Summary counts ────────────────────────────────────────────────────────
    status_counts: dict[str, int] = {}
    for audit in results:
        status_counts[audit.status] = status_counts.get(audit.status, 0) + 1

    total = len(results)
    print("\n" + "═" * 70)
    print(f"  PRES/Executive — data/processed/ Audit Report")
    print(f"  Input : {input_dir}")
    print(f"  Files : {total}")
    print("═" * 70)

    status_order = ["MALFORMED", "EMPTY", "TABLE_ONLY", "SHORT", "SPARSE", "OK"]
    for status in status_order:
        count = status_counts.get(status, 0)
        if count:
            bar = "█" * count
            print(f"  {status:<12} {count:>4}  {bar}")
    print("─" * 70)

    # ── Per-file detail ───────────────────────────────────────────────────────
    for audit in results:
        if show_bad_only and audit.status == "OK":
            continue

        status_label = {
            "OK":         "✓  OK        ",
            "EMPTY":      "✗  EMPTY     ",
            "MALFORMED":  "✗  MALFORMED ",
            "TABLE_ONLY": "⚠  TABLE_ONLY",
            "SHORT":      "⚠  SHORT     ",
            "SPARSE":     "⚠  SPARSE    ",
        }.get(audit.status, f"?  {audit.status:<11}")

        print(f"\n  {status_label}  {audit.filename}")
        print(f"             records={audit.num_records}  chars={audit.total_chars:,}  "
              f"file={audit.file_bytes:,}b  table_ratio={audit.table_ratio:.0%}")

        for issue in audit.issues:
            print(f"             → {issue}")

        if audit.first_text_preview and audit.status != "EMPTY":
            print(f"             preview: \"{audit.first_text_preview}\"")

    print("\n" + "═" * 70)

    # ── CSV export ────────────────────────────────────────────────────────────
    if output_csv:
        sitrep(f"Writing CSV → {output_csv} …", end="")
        fieldnames = [
            "filename", "status", "num_records", "total_chars",
            "thin_chunks", "table_ratio", "file_bytes", "issues", "preview"
        ]
        with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for audit in results:
                writer.writerow({
                    "filename":    audit.filename,
                    "status":      audit.status,
                    "num_records": audit.num_records,
                    "total_chars": audit.total_chars,
                    "thin_chunks": audit.thin_chunks,
                    "table_ratio": f"{audit.table_ratio:.2f}",
                    "file_bytes":  audit.file_bytes,
                    "issues":      " | ".join(audit.issues),
                    "preview":     audit.first_text_preview[:100],
                })
        print(f"\r  [{_elapsed()}] CSV written → {output_csv}{' ' * 20}")

    sitrep(f"Done. Total time {_elapsed()}.  {status_counts.get('OK', 0)}/{total} OK, "
           f"{total - status_counts.get('OK', 0)} need attention.\n")

    # ── Actionable advice ─────────────────────────────────────────────────────
    bad_statuses = {"EMPTY", "MALFORMED", "TABLE_ONLY", "SHORT"}
    bad_files    = [result for result in results if result.status in bad_statuses]

    if bad_files:
        print("  ACTION ITEMS")
        print("  ─────────────────────────────────────────────────────────────")
        if any(result.status == "EMPTY" for result in bad_files):
            print("  EMPTY files  → Re-run docling with --force-ocr if the source PDF")
            print("                 is scanned/image-based. Check docling logs for")
            print("                 per-file errors.")
        if any(result.status == "TABLE_ONLY" for result in bad_files):
            print("  TABLE_ONLY   → Docling extracted tables but skipped body text.")
            print("                 Try --table-mode=skip or extract with pdfplumber")
            print("                 as a fallback for prose-heavy PDFs.")
        if any(result.status == "SHORT" for result in bad_files):
            print("  SHORT        → Very little text. Source may be a cover page,")
            print("                 a purely graphical document, or a failed OCR pass.")
        if any(result.status == "MALFORMED" for result in bad_files):
            print("  MALFORMED    → Check if docling crashed mid-write (truncated JSON).")
            print("                 Delete and re-extract the affected file.")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Audit docling JSONL output in data/processed/"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing docling JSONL files (default: ./data/processed)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Optional path to write a CSV report"
    )
    parser.add_argument(
        "--show-bad-only",
        action="store_true",
        help="Only print files that have issues (suppress OK entries)"
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[!] Input directory not found: {args.input}")
        sys.exit(1)

    run_audit(
        input_dir    = args.input,
        show_bad_only= args.show_bad_only,
        output_csv   = args.output,
    )


if __name__ == "__main__":
    main()