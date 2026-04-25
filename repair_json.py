#!/usr/bin/env python3
"""
repair_json.py
──────────────
Converts docling output files written in Python repr format (single-quote dicts)
to valid JSON. Repairs in-place or writes to a separate output directory.

The root cause: docling result was serialised with str() / repr() instead of
json.dumps(), producing single-quoted Python dicts that are invalid JSON but
perfectly readable by ast.literal_eval().

Usage:
    # Dry-run — show what would change, write nothing
    python repair_json.py --input data/processed/

    # Repair in-place (overwrites originals)
    python repair_json.py --input data/processed/ --inplace

    # Repair into a new directory (safe — originals untouched)
    python repair_json.py --input data/processed/ --output data/processed_fixed/

    # Repair in-place but keep originals as .bak files
    python repair_json.py --input data/processed/ --inplace --backup
"""

import ast
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

_start = time.monotonic()

def _t() -> str:
    secs = int(time.monotonic() - _start)
    return f"{secs // 60:02d}:{secs % 60:02d}"

def log(msg: str, end: str = "\n") -> None:
    print(f"  [{_t()}] {msg}", end=end, flush=True)

def log_inline(msg: str) -> None:
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 100
    print(f"\r  [{_t()}] {msg:<{cols}}", end="", flush=True)


def repair_file(
    src_path: Path,
    dst_path: Path,
    dry_run: bool,
    backup: bool,
) -> tuple[str, str]:
    """
    Returns (status, detail) where status is one of:
        FIXED     — successfully converted and written
        ALREADY   — file is already valid JSON, skipped
        EMPTY     — file has no content
        FAILED    — ast.literal_eval could not parse it either
        DRY_RUN   — would have fixed (dry-run mode)
    """
    try:
        raw = src_path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as read_err:
        return "FAILED", f"Cannot read: {read_err}"

    if not raw:
        return "EMPTY", "File is empty"

    # ── Check if it's already valid JSON ─────────────────────────────────────
    try:
        json.loads(raw)
        return "ALREADY", "Already valid JSON"
    except json.JSONDecodeError:
        pass

    # ── Parse as Python literal (handles single-quoted dicts/lists) ──────────
    # The file may be:
    #   A) A single Python dict/list  → wrap in a list for uniform handling
    #   B) One Python object per line (Python-repr JSONL equivalent)

    repaired_lines: list[str] = []
    parse_failures  = 0

    lines = [line.strip() for line in raw.splitlines() if line.strip()]

    for line in lines:
        try:
            python_obj = ast.literal_eval(line)
            repaired_lines.append(json.dumps(python_obj, ensure_ascii=False))
        except (ValueError, SyntaxError):
            parse_failures += 1
            # Keep the line as-is so we don't silently drop content
            repaired_lines.append(line)

    if parse_failures == len(lines):
        # Nothing could be parsed — not Python repr either
        return "FAILED", (
            f"Neither JSON nor Python repr. "
            f"First 80 chars: {raw[:80]!r}"
        )

    new_content = "\n".join(repaired_lines) + "\n"

    if dry_run:
        fixed_count = len(lines) - parse_failures
        return "DRY_RUN", f"Would fix {fixed_count}/{len(lines)} lines"

    # ── Write output ──────────────────────────────────────────────────────────
    if backup and src_path == dst_path:
        shutil.copy2(src_path, src_path.with_suffix(src_path.suffix + ".bak"))

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(new_content, encoding="utf-8")

    fixed_count = len(lines) - parse_failures
    detail = f"Fixed {fixed_count}/{len(lines)} lines"
    if parse_failures:
        detail += f" ({parse_failures} lines could not be parsed, kept as-is)"
    return "FIXED", detail


def collect_files(input_dir: Path) -> list[Path]:
    found: list[Path] = []
    extensions = {".json", ".jsonl"}

    def _scan(directory: Path) -> None:
        try:
            with os.scandir(directory) as scanner:
                for entry in scanner:
                    if entry.is_file(follow_symlinks=False):
                        if Path(entry.name).suffix.lower() in extensions:
                            found.append(Path(entry.path))
                    elif entry.is_dir(follow_symlinks=False):
                        _scan(Path(entry.path))
        except PermissionError:
            pass

    log(f"Scanning {input_dir} …", end="")
    _scan(input_dir)
    found.sort()
    print(f"\r  [{_t()}] Found {len(found)} file(s).{' ' * 30}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair Python-repr docling output → valid JSON/JSONL"
    )
    parser.add_argument("--input",  "-i", type=Path, required=True,
                        help="Directory containing broken JSON files")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--inplace", action="store_true",
                        help="Overwrite files in-place (use with --backup for safety)")
    parser.add_argument("--backup", action="store_true",
                        help="Keep .bak copies of originals when repairing in-place")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan and report without writing anything")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[!] Input directory not found: {args.input}")
        sys.exit(1)

    # Resolve output path
    if args.dry_run:
        output_dir = None
        log("DRY RUN — nothing will be written")
    elif args.inplace:
        output_dir = args.input
    elif args.output:
        output_dir = args.output
        if output_dir == args.input:
            pass  # same dir, fine
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            log(f"Output → {output_dir}")
    else:
        print("[!] Specify --inplace or --output <dir>. Use --dry-run to preview.")
        sys.exit(1)

    files = collect_files(args.input)
    if not files:
        print(f"[!] No .json/.jsonl files found in {args.input}")
        sys.exit(1)

    # ── Process ───────────────────────────────────────────────────────────────
    counts: dict[str, int] = {}
    failures: list[tuple[str, str]] = []
    total = len(files)

    for index, src_path in enumerate(files, start=1):
        kb = src_path.stat().st_size / 1024
        log_inline(
            f"[{index}/{total}]  {src_path.name[:55]:<55}  {kb:>8.1f} KB"
        )

        if args.dry_run:
            dst_path = src_path  # unused in dry-run
        elif args.inplace:
            dst_path = src_path
        else:
            # Preserve subdirectory structure under output_dir
            rel = src_path.relative_to(args.input)
            dst_path = output_dir / rel

        status, detail = repair_file(
            src_path  = src_path,
            dst_path  = dst_path,
            dry_run   = args.dry_run,
            backup    = args.backup,
        )

        counts[status] = counts.get(status, 0) + 1

        if status == "FAILED":
            print()  # end spinner line
            log(f"  ✗ FAILED   {src_path.name}")
            log(f"             {detail}")
            failures.append((src_path.name, detail))

    print(f"\n\n{'═' * 66}")
    print(f"  Repair complete — {_t()} elapsed")
    print(f"{'─' * 66}")

    labels = {
        "FIXED":   "✓  FIXED   (repaired and written)",
        "DRY_RUN": "~  DRY_RUN (would be repaired)",
        "ALREADY": "·  ALREADY (valid JSON, skipped)",
        "EMPTY":   "·  EMPTY   (skipped)",
        "FAILED":  "✗  FAILED  (neither JSON nor Python repr)",
    }
    for status, label in labels.items():
        count = counts.get(status, 0)
        if count:
            bar = "█" * min(count, 40)
            print(f"  {label:<38}  {count:>5}  {bar}")

    if failures:
        print(f"\n  Failed files ({len(failures)}):")
        for filename, detail in failures:
            print(f"    {filename}")
            print(f"      {detail}")

    if not args.dry_run:
        fixed = counts.get("FIXED", 0)
        if fixed:
            dest = "in-place" if args.inplace else str(output_dir)
            print(f"\n  {fixed} files repaired → {dest}")
            if args.backup:
                print(f"  Originals backed up as .bak alongside each file")
    print()


if __name__ == "__main__":
    main()