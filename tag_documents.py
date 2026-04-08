#!/usr/bin/env python3
"""
tag_documents.py — Universal Document Tagger (Centralised raw/ version)
Scans the single data/raw/ folder and infers agent using taxonomy rules.

FIXES (v5):
  FIX 04 — import re moved to top of module (was defined mid-function, broken).
  FIX 07 — domain re-derived from DOMAIN_MAP after resolve_overrides() so that
            MANUAL_OVERRIDE doc_type changes propagate to domain correctly.
            e.g. controller_of_budget → fiscal_policy (not "unknown"),
                 kra_corporate_plan  → revenue_tax   (not "institutional").

  FIX E — Guard priority re-derivation so manual overrides are respected.
"""

import csv
import re
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "Common"))

from doc_type_taxonomy import (
    normalize_slug, match_doc_type, match_primary_agents,
    match_issuing_agent, build_agent_access, build_topics,
    extract_doc_year, resolve_overrides, ALL_AGENTS, DOMAIN_MAP,
    PRIORITY_MAP, RAG_WEIGHT_MAP, CATEGORY_MAP, CHUNKING_STRATEGY_MAP,
    MANUAL_OVERRIDES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("tagging.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

COLUMNS = [
    "agent", "filename", "filepath", "doc_id", "document_type", "domain",
    "primary_agents", "issuing_agent", "agent_access", "topics",
    "classification_confidence", "classification_method",
    "fiscal_year", "doc_year", "report_period", "superseded",
    "priority", "rag_weight", "category", "chunk_strategy",
    "is_scanned", "language", "geographic_scope",
]


def infer_agent_from_filename(slug: str) -> str:
    from doc_type_taxonomy import AGENT_PATTERNS
    best_agent, max_matches = "president", 0
    for agent, patterns in AGENT_PATTERNS.items():
        matches = sum(1 for p in patterns if re.search(p, slug))
        if matches > max_matches:
            max_matches, best_agent = matches, agent
    return best_agent


def main():
    root    = Path.cwd()
    raw_dir = root / "data" / "raw"
    output_csv = root / "central_inventory.csv"

    if not raw_dir.exists():
        logger.error(f"Central raw directory not found: {raw_dir}")
        return

    pdfs = list(raw_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdfs)} PDFs in central {raw_dir}")

    rows   = []
    total  = 0
    skipped = 0

    for pdf_path in sorted(pdfs):
        try:
            filename = pdf_path.name
            slug     = normalize_slug(filename)

            agent  = infer_agent_from_filename(slug)
            dt     = match_doc_type(slug)
            pa     = match_primary_agents(slug, dt)
            ia     = match_issuing_agent(slug, dt)
            access = build_agent_access(slug, dt, pa, ia)
            topics = build_topics(slug, dt)
            year   = extract_doc_year(filename)

            category = CATEGORY_MAP.get(dt, 0)
            priority = PRIORITY_MAP.get(dt, "low")

            meta = {
                "agent":                    agent,
                "filename":                 filename,
                "filepath":                 str(pdf_path),
                "doc_id":                   f"{agent}_{normalize_slug(filename)}",
                "document_type":            dt,
                "domain":                   DOMAIN_MAP.get(dt, "unknown"),
                "primary_agents":           pa,
                "issuing_agent":            ia,
                "agent_access":             access,
                "topics":                   topics,
                "classification_confidence": 0.85 if dt != "unknown" else 0.60,
                "classification_method":    "rule",
                "fiscal_year":              "na",
                "doc_year":                 year or "na",
                "report_period":            "annual",
                "superseded":               False,
                "priority":                 priority,
                "rag_weight":               RAG_WEIGHT_MAP.get(priority, 1.0),
                "category":                 category,
                "chunk_strategy":           CHUNKING_STRATEGY_MAP.get(category, "narrative"),
                "is_scanned":               False,
                "language":                 "english",
                "geographic_scope":         "national",
            }

            # Apply manual overrides
            meta = resolve_overrides(filename, meta)

            # Get override once (reuse)
            _file_override = MANUAL_OVERRIDES.get(filename, {})

            # FIX 07: re-derive domain unless override explicitly set it
            if "domain" not in _file_override:
                meta["domain"] = DOMAIN_MAP.get(
                    meta["document_type"],
                    meta.get("domain", "unknown")
                )

            # FIX E: Guard priority re-derivation
            if "priority" not in _file_override:
                meta["priority"]   = PRIORITY_MAP.get(meta["document_type"], "low")
                meta["rag_weight"] = RAG_WEIGHT_MAP[meta["priority"]]
            else:
                # Override set priority — ensure rag_weight matches it
                meta["rag_weight"] = RAG_WEIGHT_MAP.get(meta["priority"], 0.5)

            rows.append(meta)
            total += 1

            if total % 50 == 0:
                logger.info(f"Processed {total} documents...")

        except Exception as e:
            logger.error(f"Failed on {filename}: {e}")
            skipped += 1

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            row_copy = dict(row)
            for key in ["primary_agents", "agent_access", "topics"]:
                if isinstance(row_copy.get(key), (list, tuple)):
                    row_copy[key] = "|".join(str(x) for x in row_copy[key])
            writer.writerow(row_copy)

    logger.info("✅ Tagging completed!")
    logger.info(f"   Documents processed : {total}")
    logger.info(f"   Skipped             : {skipped}")
    logger.info(f"   Output → {output_csv.resolve()}")
    logger.info("   Next: Proceed to chunk_documents.py")


if __name__ == "__main__":
    main()