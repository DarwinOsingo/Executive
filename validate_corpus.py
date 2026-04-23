"""
validate_corpus.py
──────────────────
Pre-chunk validator for the Kenya AI Executive Roundtable RAG pipeline.

Runs BEFORE chunk_documents.py to guarantee every config.yaml entry has a
matching cache JSON, every cache JSON has a config entry, and the content
of each cache file is structurally sound for chunking.

Produces:
  - Console summary with pass/fail counts
  - validation_report.csv   — one row per issue, sortable in Excel/Sheets
  - orphan_cache_files.txt  — cache files with no config entry

Usage:
    python validate_corpus.py \\
        --config  /home/darwin/PRES/Executive/config.yaml \\
        --cache   /home/darwin/PRES/Executive/data/processed/ \\
        --output  /home/darwin/PRES/Executive/validation_report.csv

    # Fail immediately on any HARD issue (useful in CI)
    python validate_corpus.py \\
        --config  /home/darwin/PRES/Executive/config.yaml \\
        --cache   /home/darwin/PRES/Executive/data/processed/ \\
        --strict

Exit codes:
    0 — all clear (or only soft warnings)
    1 — one or more HARD issues found
    2 — config or cache directory not found
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import yaml


# ══════════════════════════════════════════════════════════════════════════════
# SEVERITY LEVELS
# ══════════════════════════════════════════════════════════════════════════════

HARD = "HARD"   # chunk_documents.py will skip this doc; needs fix before chunking
SOFT = "SOFT"   # warning; chunking proceeds but output may be degraded
INFO = "INFO"   # diagnostic; no action required


# ══════════════════════════════════════════════════════════════════════════════
# CACHE FILE LOOKUP
# ══════════════════════════════════════════════════════════════════════════════

def canonical_stem(name: str) -> str:
    """
    Reduce a filename stem or cache JSON stem to a canonical form for matching.

    Docling normalizes PDF filenames when writing cache files, converting
    spaces, dots, commas, parentheses, em-dashes, and all other non-alphanumeric
    characters to underscores and collapsing runs. We apply the same reduction
    to both the config source_file and the cache stem so lookups are
    character-class-agnostic.

    Examples:
        "RDM-1.2-Traffic-Surveys"                          → "rdm_1_2_traffic_surveys"
        "CBK_16th Monetary Policy Committee Report, April" → "cbk_16th_monetary_policy_committee_report_april"
        "CBK_Annual Report 2015 16 (book)"                 → "cbk_annual_report_2015_16_book"
        "2025-STRENGTHENING-OF-KANDWIA-–-KYUSO"           → "2025_strengthening_of_kandwia_kyuso"
        "First Half NGBIRR FY 23-24 - COB final 13.3.14"  → "first_half_ngbirr_fy_23_24_cob_final_13_3_14"
    """
    stem = Path(name).stem if name.lower().endswith(".pdf") else name
    normalized = re.sub(r"[^a-z0-9]+", "_", stem.lower())
    return normalized.strip("_")


def build_cache_index(cache_dir: Path) -> dict:
    """Return {canonical_stem: Path} for every .json file in cache_dir."""
    return {canonical_stem(f.stem): f for f in cache_dir.glob("*.json")}


def find_cache_file(cache_index: dict, source_file: str, doc_slug: str) -> Path | None:
    """
    Resolve a config entry → cache file using canonical stem matching.

    Both the config source_file stem and the cache stem are reduced to the
    same canonical form before comparison, so Docling's filename normalization
    (dots/spaces/punctuation → underscores) never causes a mismatch.

    Resolution order:
        1. Canonical source_file stem (primary key — closest to original filename)
        2. Canonical doc_slug          (fallback — dot-notation slug)
    """
    sf_key   = canonical_stem(source_file)
    slug_key = canonical_stem(doc_slug)

    if sf_key in cache_index:
        return cache_index[sf_key]
    if slug_key in cache_index:
        return cache_index[slug_key]

    return None

# ══════════════════════════════════════════════════════════════════════════════
# CACHE CONTENT CHECKS
# ══════════════════════════════════════════════════════════════════════════════

CONTENT_BLOCK_TYPES = {"paragraph", "list_item"}
TABLE_ONLY_STRATEGY  = "tables_only"

REQUIRED_BLOCK_KEYS = {"block_type", "heading_path", "text", "block_index", "page_number"}
REQUIRED_TABLE_KEYS = {"prov", "data"}


def check_cache_content(
    cache_path: Path,
    doc_config: dict,
) -> list[tuple[str, str]]:
    """
    Load and structurally validate a cache JSON.

    Returns list of (severity, message) tuples.
    Empty list means the file is clean.
    """
    issues: list[tuple[str, str]] = []
    slug     = doc_config.get("doc_slug", "?")
    strategy = doc_config.get("chunking_strategy", "narrative")

    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        with open(cache_path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except json.JSONDecodeError as err:
        issues.append((HARD, f"cache JSON is malformed: {err}"))
        return issues
    except OSError as err:
        issues.append((HARD, f"cannot read cache file: {err}"))
        return issues

    # ── Top-level parse error flag ────────────────────────────────────────────
    if doc.get("error"):
        issues.append((HARD, f"docling parse error recorded in cache: {doc['error']}"))
        return issues

    # ── Blocks ────────────────────────────────────────────────────────────────
    blocks = doc.get("blocks", [])

    if strategy != TABLE_ONLY_STRATEGY:
        content_blocks = [b for b in blocks if b.get("block_type") in CONTENT_BLOCK_TYPES]

        if not content_blocks:
            if doc.get("is_scanned"):
                issues.append((SOFT, "no content blocks — scanned doc (OCR may have failed)"))
            else:
                issues.append((SOFT, "no content blocks (paragraph/list_item) — doc may be image-only or empty"))

        # Field-level check on first few blocks
        for idx, block in enumerate(blocks[:10]):
            missing = REQUIRED_BLOCK_KEYS - set(block.keys())
            if missing:
                issues.append((SOFT, f"blocks[{idx}] missing keys: {sorted(missing)}"))
                break  # one warning per doc is enough

        # heading_path type check
        bad_hp = [
            idx for idx, b in enumerate(blocks[:20])
            if "heading_path" in b and not isinstance(b["heading_path"], list)
        ]
        if bad_hp:
            issues.append((SOFT, f"blocks[{bad_hp[0]}].heading_path is not a list — "
                                  "chunker will treat as empty heading"))

    # ── Tables ────────────────────────────────────────────────────────────────
    tables = doc.get("tables", [])

    if strategy == TABLE_ONLY_STRATEGY and not tables:
        issues.append((HARD, "chunking_strategy=tables_only but cache has zero tables"))

    for idx, table in enumerate(tables[:5]):
        missing = REQUIRED_TABLE_KEYS - set(table.keys())
        if missing:
            issues.append((SOFT, f"tables[{idx}] missing keys: {sorted(missing)}"))
            break

        # Check grid is accessible
        data = table.get("data", {})
        if not data.get("grid") and not data.get("table_cells"):
            issues.append((SOFT, f"tables[{idx}].data has neither 'grid' nor 'table_cells' — "
                                  "will render as empty markdown"))

    # ── Scanned flag cross-check ──────────────────────────────────────────────
    config_scanned = doc_config.get("is_scanned", False)
    cache_scanned  = doc.get("is_scanned", False)
    if bool(config_scanned) != bool(cache_scanned):
        issues.append((SOFT, f"is_scanned mismatch — config={config_scanned}, cache={cache_scanned}"))

    return issues


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG-LEVEL CHECKS
# ══════════════════════════════════════════════════════════════════════════════

VALID_STRATEGIES = {"narrative", "legal", "audit_findings", "tables_only", "hybrid"}
VALID_PRIORITIES = {"constitutional", "high", "medium", "low"}
ALL_AGENTS       = {"finance", "education", "agriculture", "ict",
                    "infrastructure", "anticorruption", "president"}


def check_config_entry(doc_config: dict) -> list[tuple[str, str]]:
    """
    Validate the config entry itself — before even touching the cache.

    Returns list of (severity, message) tuples.
    """
    issues: list[tuple[str, str]] = []

    strategy     = doc_config.get("chunking_strategy", "")
    priority     = doc_config.get("priority", "")
    agent_access = doc_config.get("agent_access") or []
    primary      = doc_config.get("primary_agents") or []
    doc_type     = doc_config.get("document_type", "")
    domain       = doc_config.get("domain", "")
    rag_weight   = doc_config.get("rag_weight")
    chunk_size   = doc_config.get("chunk_size")
    doc_year     = doc_config.get("doc_year")

    # HARD: config was generated with a HARD flag
    if doc_config.get("HARD"):
        reasons = doc_config.get("hard_reasons", ["no reason recorded"])
        issues.append((HARD, "generate_config.py HARD flag: " + " | ".join(reasons)))

    # HARD: missing critical fields
    if not strategy or strategy not in VALID_STRATEGIES:
        issues.append((HARD, f"invalid chunking_strategy='{strategy}' — "
                              f"must be one of {sorted(VALID_STRATEGIES)}"))

    if not doc_type or doc_type == "unknown":
        issues.append((HARD, f"document_type='{doc_type}' — classification unresolved"))

    if not domain or domain == "unknown":
        issues.append((HARD, f"domain='{domain}' — not set"))

    if not agent_access:
        issues.append((HARD, "agent_access is empty — doc unreachable by any agent"))

    if not primary:
        issues.append((HARD, "primary_agents is empty — ownership unresolved"))

    # HARD: doc_year out of range (artefact from filename parsing)
    if doc_year is not None:
        try:
            y = int(doc_year)
            if y > 2026:
                issues.append((HARD, f"doc_year={y} > 2026 — "
                                     "likely filename parse artefact (e.g. '2024-28.2')"))
            elif y < 1990:
                issues.append((HARD, f"doc_year={y} < 1990 — implausible year"))
        except (TypeError, ValueError):
            issues.append((SOFT, f"doc_year='{doc_year}' is not an integer"))

    # SOFT: unknown agents in agent_access
    bad_agents = set(agent_access) - ALL_AGENTS
    if bad_agents:
        issues.append((SOFT, f"unrecognised agents in agent_access: {sorted(bad_agents)}"))

    # SOFT: primary_agents not a subset of agent_access
    unreachable = set(primary) - set(agent_access)
    if unreachable:
        issues.append((SOFT, f"primary_agents {sorted(unreachable)} not in agent_access — "
                              "owner cannot retrieve own document"))

    # SOFT: invalid priority
    if priority not in VALID_PRIORITIES:
        issues.append((SOFT, f"priority='{priority}' is not in {sorted(VALID_PRIORITIES)}"))

    # SOFT: rag_weight/priority consistency
    weight_map = {"constitutional": 2.0, "high": 1.5, "medium": 1.0, "low": 0.5}
    expected_weight = weight_map.get(priority)
    if expected_weight is not None and rag_weight is not None:
        try:
            if abs(float(rag_weight) - expected_weight) > 0.01:
                issues.append((SOFT, f"rag_weight={rag_weight} inconsistent with "
                                     f"priority='{priority}' (expected {expected_weight})"))
        except (TypeError, ValueError):
            pass

    # SOFT: chunk_size sanity
    if chunk_size is not None:
        try:
            cs = int(chunk_size)
            if cs < 50:
                issues.append((SOFT, f"chunk_size={cs} is very small — likely misconfigured"))
            elif cs > 4096:
                issues.append((SOFT, f"chunk_size={cs} is very large — may exceed embedding limits"))
        except (TypeError, ValueError):
            issues.append((SOFT, f"chunk_size='{chunk_size}' is not an integer"))

    # INFO: soft_reasons from generate_config.py
    for reason in doc_config.get("soft_reasons", []):
        issues.append((INFO, f"generate_config.py soft warning: {reason}"))

    return issues


# ══════════════════════════════════════════════════════════════════════════════
# ORPHAN DETECTOR  (cache files with no config entry)
# ══════════════════════════════════════════════════════════════════════════════

def find_orphan_cache_files(
    cache_index:    dict,
    claimed_paths:  set,
) -> list[Path]:
    """
    Return cache files that were not matched by any config entry.

    These are parsed PDFs that never made it into central_inventory.csv,
    or whose filenames changed after parsing.
    """
    all_paths = set(cache_index.values())
    return sorted(all_paths - claimed_paths)


# ══════════════════════════════════════════════════════════════════════════════
# REPORT WRITERS
# ══════════════════════════════════════════════════════════════════════════════

def write_csv_report(rows: list[dict], output_path: Path) -> None:
    if not rows:
        print("  (no issues — CSV not written)")
        return
    fields = ["severity", "doc_slug", "source_file", "agent_access",
              "chunking_strategy", "document_type", "issue"]
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Report written → {output_path}  ({len(rows)} issues)")


def write_orphan_report(orphans: list[Path], output_path: Path) -> None:
    if not orphans:
        print("  No orphan cache files.")
        return
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Orphan cache files ({len(orphans)})\n")
        fh.write("# These JSONs have no matching config entry.\n\n")
        for path in orphans:
            fh.write(str(path) + "\n")
    print(f"  Orphan list written → {output_path}  ({len(orphans)} files)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-chunk validator — Kenya AI Executive Roundtable"
    )
    parser.add_argument("--config",  required=True, help="Path to config.yaml")
    parser.add_argument("--cache",   required=True, help="Path to data/processed/ directory")
    parser.add_argument(
        "--output", default="validation_report.csv",
        help="Path for the output CSV report (default: validation_report.csv)",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit with code 1 if any HARD issues are found",
    )
    parser.add_argument(
        "--agent", default=None,
        help="Validate only documents where AGENT is in primary_agents",
    )
    parser.add_argument(
        "--slug", default=None,
        help="Validate only one document by doc_slug",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    cache_dir   = Path(args.cache)
    output_path = Path(args.output)

    # ── Existence checks ──────────────────────────────────────────────────────
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        sys.exit(2)
    if not cache_dir.is_dir():
        print(f"ERROR: cache directory not found: {cache_dir}")
        sys.exit(2)

    # ── Load config ───────────────────────────────────────────────────────────
    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    pipeline  = config.get("pipeline", {})
    documents = config.get("documents", [])

    print(f"\nCollection      : {pipeline.get('qdrant_collection', '?')}")
    print(f"Config entries  : {len(documents)}")
    print(f"Cache dir       : {cache_dir}")

    # ── Build cache index ─────────────────────────────────────────────────────
    cache_index = build_cache_index(cache_dir)
    print(f"Cache files     : {len(cache_index)}\n")

    # ── Apply filters ─────────────────────────────────────────────────────────
    if args.slug:
        documents = [d for d in documents if d.get("doc_slug") == args.slug]
        if not documents:
            print(f"ERROR: slug '{args.slug}' not found in config")
            sys.exit(2)

    if args.agent:
        documents = [
            d for d in documents
            if args.agent in (d.get("primary_agents") or [])
        ]
        print(f"Agent filter '{args.agent}': {len(documents)} documents\n")

    # ── Validate ──────────────────────────────────────────────────────────────
    report_rows:    list[dict] = []
    claimed_paths:  set        = set()

    counts = {HARD: 0, SOFT: 0, INFO: 0}
    doc_hard = 0
    doc_soft = 0
    no_cache = 0
    collision_map: dict[str, str] = {}  # cache_path_str → first claiming slug

    for doc_config in documents:
        slug     = doc_config.get("doc_slug", "?")
        fname    = doc_config.get("source_file", slug)
        strategy = doc_config.get("chunking_strategy", "narrative")
        access   = "|".join(sorted(doc_config.get("agent_access") or []))

        doc_issues: list[tuple[str, str]] = []

        # 1. Config-level checks
        doc_issues += check_config_entry(doc_config)

        # 2. Cache matching
        cache_file = find_cache_file(cache_index, fname, slug)

        if cache_file is None:
            doc_issues.append((HARD, "no matching cache file in data/processed/ — not yet parsed"))
            no_cache += 1
        else:
            cache_key = str(cache_file)

            # Collision detection
            if cache_key in collision_map:
                first = collision_map[cache_key]
                doc_issues.append((HARD, f"cache collision — same file claimed by '{first}'"))
            else:
                collision_map[cache_key] = slug
                claimed_paths.add(cache_file)

                # 3. Cache content checks
                doc_issues += check_cache_content(cache_file, doc_config)

        # Accumulate
        severities = {sev for sev, _ in doc_issues}
        if HARD in severities:
            doc_hard += 1
        elif SOFT in severities:
            doc_soft += 1

        for sev, msg in doc_issues:
            counts[sev] += 1
            report_rows.append({
                "severity":         sev,
                "doc_slug":         slug,
                "source_file":      fname,
                "agent_access":     access,
                "chunking_strategy": strategy,
                "document_type":    doc_config.get("document_type", "?"),
                "issue":            msg,
            })

    # ── Orphan detection ──────────────────────────────────────────────────────
    orphans = find_orphan_cache_files(cache_index, claimed_paths)

    # ── Print summary ─────────────────────────────────────────────────────────
    total_docs   = len(documents)
    clean_docs   = total_docs - doc_hard - doc_soft
    ready_docs   = total_docs - doc_hard    # SOFT docs still chunk

    print("─" * 65)
    print(f"TOTAL DOCUMENTS   : {total_docs:>4}")
    print(f"  ✓ Clean         : {clean_docs:>4}  (no issues)")
    print(f"  ⚠  Soft only    : {doc_soft:>4}  (will chunk with warnings)")
    print(f"  ✗ Hard flagged  : {doc_hard:>4}  (will be SKIPPED by chunker)")
    print(f"  ✗ No cache      : {no_cache:>4}  (not yet parsed by docling)")
    print(f"  ↔ Orphan cache  : {len(orphans):>4}  (parsed but not in config)")
    print()
    print(f"ISSUE COUNTS")
    print(f"  HARD   : {counts[HARD]:>4}")
    print(f"  SOFT   : {counts[SOFT]:>4}")
    print(f"  INFO   : {counts[INFO]:>4}")
    print()
    print(f"READY TO CHUNK    : {ready_docs} / {total_docs} documents")
    print("─" * 65)

    # ── Write reports ─────────────────────────────────────────────────────────
    print("\nWriting reports:")

    # Sort: HARD first, then SOFT, then INFO; within each group alphabetically by slug
    sev_order = {HARD: 0, SOFT: 1, INFO: 2}
    report_rows.sort(key=lambda row: (sev_order.get(row["severity"], 9), row["doc_slug"]))

    write_csv_report(report_rows, output_path)

    orphan_path = output_path.parent / "orphan_cache_files.txt"
    write_orphan_report(orphans, orphan_path)

    # ── Final advisory ────────────────────────────────────────────────────────
    if doc_hard > 0:
        print(f"\n⚠  {doc_hard} documents have HARD issues and will be skipped by chunk_documents.py.")
        print("   Fix them in central_inventory.csv → re-run generate_config.py → re-validate.\n")
    else:
        print("\n✓  No HARD issues. Safe to run chunk_documents.py.\n")

    if args.strict and counts[HARD] > 0:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()