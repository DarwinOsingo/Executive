"""
generate_config.py
──────────────────
Reads central_inventory.csv and emits a unified config.yaml for consumption
by chunk_documents.py.

This script contains ZERO classification logic — no regex, no LLM calls.
It is a deterministic, auditable projection of the metadata registry.

Usage
─────
    # Full corpus config
    python generate_config.py --input central_inventory.csv --output config.yaml

    # Single-agent view (includes all docs where agent appears in agent_access)
    python generate_config.py --input central_inventory.csv \
                               --output finance_config.yaml \
                               --agent finance

    # Dry-run: validate only, no output file written
    python generate_config.py --input central_inventory.csv --validate-only

Outputs
───────
    config.yaml          — main chunking config consumed by chunk_documents.py
    config_hard.csv      — rows flagged HARD (require human review before chunking)
    config_soft.csv      — rows flagged SOFT (warnings, chunking proceeds)
"""

import argparse
import csv
import logging
import sys
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

CURRENT_YEAR   = date.today().year          # 2026
MIN_VALID_YEAR = 1990
MAX_VALID_YEAR = CURRENT_YEAR               # flag anything beyond today's year

# Document types for which empty topics is a HARD flag (not just a soft warning)
TOPIC_REQUIRED_TYPES = {
    "budget_policy_statement", "budget_review_outlook", "budget_summary",
    "debt_management_strategy", "public_debt_report", "cbk_annual_report",
    "cbk_mpc_report", "cbk_fsr_report", "kra_revenue_performance",
    "economic_survey", "finance_act", "finance_bill", "auditor_general_report",
    "eacc_report", "ppra_report", "mer_report", "strategic_plan", "policy",
    "sector_report", "masterplan",
}

# For acts/bills we expect finance to be in agent_access when the slug contains
# finance-related keywords.
FINANCE_ACT_KEYWORDS = {"finance", "budget", "revenue", "tax", "fiscal"}

ALL_AGENTS = [
    "finance", "education", "agriculture", "ict",
    "infrastructure", "anticorruption", "president",
]

# ── YAML helpers ──────────────────────────────────────────────────────────────

class _QuotedStr(str):
    """Marker class — YAML dumper will always quote these."""


def _quoted_str_representer(dumper: yaml.Dumper, data: _QuotedStr) -> yaml.Node:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


def _build_yaml_dumper() -> type:
    dumper = yaml.Dumper
    dumper.add_representer(_QuotedStr, _quoted_str_representer)
    return dumper


# ── CSV parsing helpers ───────────────────────────────────────────────────────

def _parse_list_col(value: Any) -> list[str]:
    """
    Convert a pipe-delimited CSV cell to a sorted list of stripped strings.
    Handles NaN, empty string, and optional spaces around the delimiter.

        "agriculture|finance|president" → ["agriculture", "finance", "president"]
        ""  → []
        NaN → []
    """
    if pd.isna(value) or str(value).strip() == "":
        return []
    parts = [part.strip() for part in str(value).split("|") if part.strip()]
    return sorted(parts)


def _parse_bool_col(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in ("true", "1", "yes")


def _parse_float_col(value: Any, default: float = 0.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int_col(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


# ── Validation ────────────────────────────────────────────────────────────────

def validate_row(row: pd.Series) -> tuple[list[str], list[str]]:
    """
    Return (hard_reasons, soft_reasons).

    HARD  → chunk_documents.py will SKIP this document and log it.
    SOFT  → warning logged, document still processed.
    """
    hard: list[str] = []
    soft: list[str] = []

    filename    = str(row.get("filename", "")).strip()
    doc_type    = str(row.get("document_type", "")).strip()
    domain      = str(row.get("domain", "")).strip()
    topics      = _parse_list_col(row.get("topics", ""))
    doc_year    = _parse_int_col(row.get("doc_year", 0), default=0)
    agent_access = _parse_list_col(row.get("agent_access", ""))
    primary_agents = _parse_list_col(row.get("primary_agents", ""))
    issuing_agent = str(row.get("issuing_agent", "")).strip()
    confidence  = _parse_float_col(row.get("classification_confidence", 0.85))

    # ── HARD checks ───────────────────────────────────────────────────────────

    # 1. Missing or unknown doc_type
    if not doc_type or doc_type == "unknown":
        hard.append(f"document_type is '{doc_type}' — classification required")

    # 2. Missing or unknown domain
    if not domain or domain == "unknown":
        hard.append(f"domain is '{domain}' — re-derive from DOMAIN_MAP or classify manually")

    # 3. doc_year anomaly — beyond reasonable bounds
    if doc_year != 0:
        if doc_year > MAX_VALID_YEAR:
            hard.append(
                f"doc_year={doc_year} > {MAX_VALID_YEAR} — "
                "likely a slug parse artefact (e.g. '2024-28.2' parsed as 2028)"
            )
        elif doc_year < MIN_VALID_YEAR:
            hard.append(f"doc_year={doc_year} < {MIN_VALID_YEAR} — implausible year")

    # 4. Topics required but empty for specific document types
    if doc_type in TOPIC_REQUIRED_TYPES and not topics:
        hard.append(
            f"topics is empty for document_type='{doc_type}' — "
            "add to FILENAME_TOPIC_OVERRIDES or MANUAL_OVERRIDES"
        )

    # 5. Finance Act / Finance Bill without finance in agent_access
    if doc_type in ("finance_act", "finance_bill", "act", "bill"):
        slug_lower = filename.lower()
        is_finance_related = any(kw in slug_lower for kw in FINANCE_ACT_KEYWORDS)
        if is_finance_related and "finance" not in agent_access:
            hard.append(
                f"finance-related {doc_type} but 'finance' not in agent_access={agent_access} — "
                "add MANUAL_OVERRIDE or update SHARED_ACCESS_MAP"
            )

    # 6. No agents at all
    if not agent_access:
        hard.append("agent_access is empty — document unreachable by any agent")

    if not primary_agents:
        hard.append("primary_agents is empty — ownership unresolved")

    # ── SOFT checks ───────────────────────────────────────────────────────────

    # 7. Low classification confidence
    if confidence < 0.7:
        soft.append(f"classification_confidence={confidence:.2f} below 0.70 — verify classification")

    # 8. issuing_agent is unknown for legal documents
    if doc_type in ("act", "bill", "regulations", "finance_act", "finance_bill"):
        if issuing_agent == "unknown":
            soft.append(
                f"issuing_agent='unknown' for legal doc_type='{doc_type}' — "
                "expected 'president', 'finance', or an agency"
            )

    # 9. Empty topics for generic types (soft only — these often legitimately have no topics)
    if not topics and doc_type not in TOPIC_REQUIRED_TYPES:
        soft.append(f"topics is empty for document_type='{doc_type}'")

    # 10. primary_agents not a subset of agent_access
    unreachable = set(primary_agents) - set(agent_access)
    if unreachable:
        soft.append(
            f"primary_agents {sorted(unreachable)} not present in agent_access — "
            "primary owner cannot retrieve own document"
        )

    return hard, soft


# ── Document entry builder ────────────────────────────────────────────────────

def build_document_entry(row: pd.Series, hard_reasons: list[str], soft_reasons: list[str]) -> dict:
    """
    Convert one CSV row into the YAML document entry dict.

    fiscal_year is always emitted as a quoted string to prevent PyYAML from
    interpreting values like "na" as null or "2022_23" as an identifier.
    """
    # Normalise fiscal_year: ensure it is always a quoted string in YAML
    raw_fy = str(row.get("fiscal_year", "na")).strip()
    fiscal_year_val = _QuotedStr(raw_fy if raw_fy else "na")

    entry: dict[str, Any] = {
        "source_file":              str(row.get("filename", "")).strip(),
        "filepath":                 str(row.get("filepath", "")).strip(),
        "doc_id":                   str(row.get("doc_id", "")).strip(),
        "doc_slug":                 str(row.get("doc_id", "")).strip(),   # same as doc_id
        "document_type":            str(row.get("document_type", "unknown")).strip(),
        "domain":                   str(row.get("domain", "unknown")).strip(),
        "primary_agents":           _parse_list_col(row.get("primary_agents", "")),
        "issuing_agent":            str(row.get("issuing_agent", "unknown")).strip(),
        "agent_access":             _parse_list_col(row.get("agent_access", "")),
        "topics":                   _parse_list_col(row.get("topics", "")),
        "classification_confidence":_parse_float_col(row.get("classification_confidence", 0.85)),
        "classification_method":    str(row.get("classification_method", "rule")).strip(),
        "fiscal_year":              fiscal_year_val,
        "doc_year":                 _parse_int_col(row.get("doc_year", 0), default=0) or None,
        "report_period":            str(row.get("report_period", "annual")).strip(),
        "superseded":               _parse_bool_col(row.get("superseded", False)),
        "priority":                 str(row.get("priority", "low")).strip(),
        "rag_weight":               _parse_float_col(row.get("rag_weight", 0.5)),
        "category":                 _parse_int_col(row.get("category", 0)),
        "chunk_strategy":           str(row.get("chunk_strategy", "narrative")).strip(),
        "is_scanned":               _parse_bool_col(row.get("is_scanned", False)),
        "language":                 str(row.get("language", "english")).strip(),
        "geographic_scope":         str(row.get("geographic_scope", "national")).strip(),
    }

    # Audit flags — only written when non-empty so clean entries stay tidy
    if hard_reasons:
        entry["HARD"] = True
        entry["hard_reasons"] = hard_reasons

    if soft_reasons:
        entry["soft_reasons"] = soft_reasons

    return entry


# ── Config writer ─────────────────────────────────────────────────────────────

def write_config(documents: list[dict], output_path: Path, agent_filter: str | None) -> None:
    """Write the final YAML config file."""
    config = {
        "# Auto-generated by generate_config.py — DO NOT EDIT MANUALLY": None,
        "meta": {
            "source":    "central_inventory.csv",
            "generated": str(date.today()),
            "agent_filter": agent_filter or "all",
            "total_documents": len(documents),
            "hard_flagged": sum(1 for doc in documents if doc.get("HARD")),
        },
        "documents": documents,
    }

    # Use a clean stream — strip the spurious None-value comment key
    stream = StringIO()
    dumper = _build_yaml_dumper()
    yaml.dump(
        {"meta": config["meta"], "documents": config["documents"]},
        stream,
        Dumper=dumper,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=120,
    )

    header = (
        "# Auto-generated by generate_config.py — DO NOT EDIT MANUALLY\n"
        f"# Source: central_inventory.csv\n"
        f"# Generated: {date.today()}\n"
        f"# Agent filter: {agent_filter or 'all'}\n\n"
    )

    output_path.write_text(header + stream.getvalue(), encoding="utf-8")
    logger.info(f"Config written → {output_path}  ({len(documents)} documents)")


# ── Audit report writers ──────────────────────────────────────────────────────

def write_audit_csv(flagged_rows: list[dict], output_path: Path, flag_type: str) -> None:
    """Write HARD or SOFT flagged rows to a CSV for human review."""
    if not flagged_rows:
        logger.info(f"No {flag_type} flags — audit CSV not written.")
        return

    df = pd.DataFrame(flagged_rows)
    df.to_csv(output_path, index=False)
    logger.info(f"{flag_type.upper()} audit CSV written → {output_path}  ({len(df)} rows)")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def read_central_inventory(csv_path: Path) -> pd.DataFrame:
    """
    Read central_inventory.csv. Coerce types where needed.
    The 'agent' column (first column) is the routing key from the old
    per-agent naming convention — we keep it but don't use it for filtering.
    """
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df.columns = [col.strip() for col in df.columns]
    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project central_inventory.csv → config.yaml for chunk_documents.py"
    )
    parser.add_argument(
        "--input", "-i",
        default="central_inventory.csv",
        help="Path to central_inventory.csv (default: central_inventory.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        default="config.yaml",
        help="Path for the output config YAML (default: config.yaml)",
    )
    parser.add_argument(
        "--agent", "-a",
        default=None,
        choices=ALL_AGENTS + [None],
        help=(
            "Emit only documents where AGENT appears in agent_access. "
            "Omit to emit the full corpus config."
        ),
    )
    parser.add_argument(
        "--validate-only", "-v",
        action="store_true",
        help="Run validation and print summary; do not write any output files.",
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input CSV not found: {input_path}")
        sys.exit(1)

    df = read_central_inventory(input_path)

    # ── Optional agent filter ─────────────────────────────────────────────────
    if args.agent:
        # Keep rows where the requested agent appears anywhere in agent_access
        mask = df["agent_access"].apply(
            lambda cell: args.agent in _parse_list_col(cell)
        )
        df = df[mask].reset_index(drop=True)
        logger.info(
            f"Agent filter '{args.agent}': {len(df)} documents retained "
            f"(those with '{args.agent}' in agent_access)"
        )

    # ── Build entries and collect audit rows ──────────────────────────────────
    documents:   list[dict] = []
    hard_audit:  list[dict] = []
    soft_audit:  list[dict] = []

    hard_count = 0
    soft_count = 0

    for _, row in df.iterrows():
        hard_reasons, soft_reasons = validate_row(row)

        if hard_reasons:
            hard_count += 1
        if soft_reasons:
            soft_count += 1

        entry = build_document_entry(row, hard_reasons, soft_reasons)
        documents.append(entry)

        # Audit record: key metadata + flag reasons
        base_audit = {
            "filename":      row.get("filename", ""),
            "doc_id":        row.get("doc_id", ""),
            "document_type": row.get("document_type", ""),
            "agent_access":  row.get("agent_access", ""),
            "doc_year":      row.get("doc_year", ""),
            "topics":        row.get("topics", ""),
        }
        if hard_reasons:
            hard_audit.append({**base_audit, "hard_reasons": " | ".join(hard_reasons)})
        if soft_reasons:
            soft_audit.append({**base_audit, "soft_reasons": " | ".join(soft_reasons)})

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(
        f"Validation complete — "
        f"total={len(documents)}  HARD={hard_count}  SOFT={soft_count}"
    )

    if args.validate_only:
        logger.info("--validate-only: no files written.")
        if hard_audit:
            logger.warning(f"{hard_count} HARD rows require review before chunking:")
            for audit_row in hard_audit:
                logger.warning(f"  [{audit_row['filename']}]  {audit_row['hard_reasons']}")
        sys.exit(0)

    # ── Write outputs ─────────────────────────────────────────────────────────
    write_config(documents, output_path, args.agent)

    stem = output_path.stem
    parent = output_path.parent
    write_audit_csv(hard_audit, parent / f"{stem}_hard.csv",  "hard")
    write_audit_csv(soft_audit, parent / f"{stem}_soft.csv", "soft")

    # ── Final advisory ────────────────────────────────────────────────────────
    if hard_count:
        logger.warning(
            f"{hard_count} documents flagged HARD — chunk_documents.py will SKIP them. "
            f"Review {stem}_hard.csv and add MANUAL_OVERRIDES or fix the CSV."
        )
    if soft_count:
        logger.info(
            f"{soft_count} documents have soft warnings — "
            f"see {stem}_soft.csv for details."
        )


if __name__ == "__main__":
    main()