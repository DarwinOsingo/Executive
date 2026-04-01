#!/usr/bin/env python3
"""
rename_with_llm.py — Step 2 of 4 in the multi-agent RAG pipeline.

Reads inventory.csv, extracts text from the first 2 pages of each PDF,
sends it to Groq to suggest a standardized filename, and outputs
rename_manifest.csv for human review.

NOTHING IS RENAMED HERE. This script only produces suggestions.
Script 3 (apply_renames.py) does the actual renaming after you approve.

Usage:
    python rename_with_llm.py
    python rename_with_llm.py --inventory inventory.csv --output rename_manifest.csv
    python rename_with_llm.py --agent ICT          # process one agent only
    python rename_with_llm.py --resume             # skip already-processed rows

Output:
    rename_manifest.csv with columns:
        agent | original_filename | filepath | extracted_title |
        suggested_name | confidence | flags | approved

Workflow after this script:
    1. Open rename_manifest.csv
    2. Review every row — edit suggested_name if wrong
    3. Set approved=yes for rows you accept
    4. Leave approved blank for rows you want to skip
    5. Run apply_renames.py
"""

import os
import re
import csv
import json
import time
import argparse
import pdfplumber
from pathlib import Path
from groq import Groq


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_INVENTORY = "inventory.csv"
DEFAULT_OUTPUT    = "rename_manifest.csv"

GROQ_MODEL        = "llama-3.3-70b-versatile"
MAX_TEXT_CHARS    = 2000   # chars from first 2 pages sent to Groq
RETRY_ATTEMPTS    = 3
RETRY_DELAY       = 5      # seconds between retries
INTER_FILE_DELAY  = 0.3    # seconds between API calls (rate limit buffer)

# Agent → domain description for Groq context
AGENT_DOMAINS = {
    "Education":      "Kenya education sector — Ministry of Education, KNEC, TSC, universities, TVET",
    "Agriculture":    "Kenya agriculture sector — KALRO, AFA, crop research institutes, food security",
    "ICT":            "Kenya ICT and digital economy — CA, KICTANET, digital policy, cybersecurity, fintech",
    "Infastructure":  "Kenya infrastructure — roads, energy, ports, SGR, KETRACO, EPRA, KURA, KeNHA",
    "AntiCorruption": "Kenya governance and anti-corruption — EACC, PPRA, PPOA, procurement, audit",
    "President":      "Kenya cross-cutting policy — presidency, cross-ministry, national planning",
    "Finance":        "Kenya public finance — National Treasury, CBK, KRA, debt management, fiscal policy",
}

# Document type vocabulary for Groq to choose from
DOC_TYPES = [
    "Annual-Report",
    "Strategic-Plan",
    "Policy",
    "Act",
    "Bill",
    "Regulations",
    "Budget",
    "Audit-Report",
    "Survey",
    "Statistics-Report",
    "Research-Report",
    "Framework",
    "Manual",
    "Guidelines",
    "Assessment",
    "Conference-Report",
    "IGF-Report",
    "Forensic-Audit",
    "Financial-Statements",
    "MER-Report",
    "Case-Report",
    "Masterplan",
    "Bulletin",
    "Catalogue",
]

SYSTEM_PROMPT = """You are a document naming expert for a Kenyan government AI research system.
Your job is to produce clean, standardized filenames for government PDF documents.

NAMING CONVENTION:
- Format: YYYY-DocType-Issuer-ShortDescription.pdf
- Year: use the document's publication year or fiscal year start (e.g. 2023 for FY2023-24)
- DocType: pick the closest match from this list: {doc_types}
- Issuer: the organization that produced the document (e.g. KNEC, PPRA, KALRO, KETRACO, EACC)
- ShortDescription: 2-5 words max, hyphenated, Title-Case
- If fiscal year is present use format FY-YYYY-YY (e.g. FY-2023-24)
- No spaces anywhere — hyphens only
- No special characters: no &, /, (), [], @
- No leading underscores
- Max 80 characters total including .pdf
- If document has version info (v1, v2) preserve it at the end before .pdf

EXAMPLES:
  "PPRA ANNUAL REPORT 2023-2024.pdf" → "2024-Annual-Report-PPRA-FY-2023-24.pdf"
  "KNEC-Audited-Annual-Report-FY-2019_2020.pdf" → "2020-Annual-Report-KNEC-FY-2019-20.pdf"
  "Kenya AI Strategy 2025 - 2030.pdf" → "2025-Strategic-Plan-MoE-AI-Strategy-2025-30.pdf"
  "Agricultural-Finance-Corporationauditor general.pdf" → "2023-Audit-Report-AFC-Financial-Statements.pdf"
  "Mobile-Payments-v1.pdf" → "2023-Research-Report-CA-Mobile-Payments-v1.pdf"

CONFIDENCE LEVELS:
  HIGH   — title, issuer, and year all clearly visible in the text
  MEDIUM — at least two of the three are clear
  LOW    — text is unclear, scanned, or missing key info

Respond ONLY with a JSON object — no preamble, no explanation, no markdown:
{{"suggested_name": "YYYY-DocType-Issuer-Description.pdf", "confidence": "HIGH|MEDIUM|LOW", "extracted_title": "exact title as it appears in the document"}}"""

USER_PROMPT_TEMPLATE = """Agent domain: {agent_domain}
Original filename: {original_filename}

First 2 pages of document text:
---
{text}
---

Suggest a standardized filename following the naming convention."""


# ── PDF text extraction ────────────────────────────────────────────────────────

def extract_pdf_text(filepath: str, max_chars: int = MAX_TEXT_CHARS) -> tuple[str, str]:
    """
    Extract text from first 2 pages of a PDF.
    Returns (text, extraction_status) where status is 'ok', 'partial', or 'failed'.
    """
    try:
        with pdfplumber.open(filepath) as pdf:
            pages = pdf.pages[:2]
            text_parts = []
            for page in pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text.strip())

            full_text = "\n\n".join(text_parts).strip()

            if not full_text:
                return "", "failed"

            truncated = full_text[:max_chars]
            status = "partial" if len(full_text) > max_chars else "ok"
            return truncated, status

    except Exception as e:
        return "", f"error: {str(e)[:80]}"


# ── Groq call ─────────────────────────────────────────────────────────────────

def call_groq(client: Groq, agent: str, original_filename: str, text: str) -> dict:
    """
    Call Groq to suggest a standardized filename.
    Returns dict with suggested_name, confidence, extracted_title.
    Retries on failure.
    """
    agent_domain = AGENT_DOMAINS.get(agent, f"{agent} sector Kenya government")
    doc_types_str = ", ".join(DOC_TYPES)

    system = SYSTEM_PROMPT.format(doc_types=doc_types_str)
    user   = USER_PROMPT_TEMPLATE.format(
        agent_domain=agent_domain,
        original_filename=original_filename,
        text=text if text else "[No text extracted — scanned or image-only PDF]",
    )

    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0,
                max_tokens=200,
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            raw = re.sub(r'^```json\s*', '', raw)
            raw = re.sub(r'\s*```$',     '', raw)
            raw = raw.strip()

            result = json.loads(raw)

            # Validate required fields
            if "suggested_name" not in result:
                raise ValueError("Missing suggested_name in response")

            # Sanitize suggested name
            name = result["suggested_name"]
            name = re.sub(r'\s+', '-', name)           # spaces → hyphens
            name = re.sub(r'^_+', '', name)             # strip leading underscores
            name = re.sub(r'[&@#$%^*+=\[\]{}|\\<>?]', '', name)  # strip special chars
            if not name.lower().endswith('.pdf'):
                name = name + '.pdf'
            result["suggested_name"] = name

            result.setdefault("confidence",      "LOW")
            result.setdefault("extracted_title", "")

            return result

        except json.JSONDecodeError as e:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
                continue
            return {
                "suggested_name":  _fallback_name(original_filename),
                "confidence":      "LOW",
                "extracted_title": f"JSON parse error: {str(e)[:60]}",
            }

        except Exception as e:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
                continue
            return {
                "suggested_name":  _fallback_name(original_filename),
                "confidence":      "LOW",
                "extracted_title": f"API error: {str(e)[:60]}",
            }


def _fallback_name(original: str) -> str:
    """Produce a minimal cleaned name when Groq fails."""
    name = Path(original).stem
    name = re.sub(r'\s+', '-', name)
    name = re.sub(r'^_+', '', name)
    name = re.sub(r'[&@#$%^*+=\[\]{}|\\<>?]', '', name)
    return name + ".pdf"


# ── Load existing manifest for resume ─────────────────────────────────────────

def load_existing_manifest(output_path: str) -> set[str]:
    """Return set of filepaths already processed in an existing manifest."""
    processed = set()
    if Path(output_path).exists():
        with open(output_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("filepath"):
                    processed.add(row["filepath"])
    return processed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM-powered PDF renamer — produces manifest for human review.")
    parser.add_argument("--inventory", default=DEFAULT_INVENTORY, help="Input inventory CSV")
    parser.add_argument("--output",    default=DEFAULT_OUTPUT,    help="Output manifest CSV")
    parser.add_argument("--agent",     default=None,              help="Process one agent only")
    parser.add_argument("--resume",    action="store_true",       help="Skip already-processed rows")
    args = parser.parse_args()

    # API key
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] GROQ_API_KEY not set. Run: export GROQ_API_KEY=gsk_...")
        return

    client = Groq(api_key=api_key)

    # Load inventory
    if not Path(args.inventory).exists():
        print(f"[ERROR] Inventory file not found: {args.inventory}")
        return

    with open(args.inventory, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Filter by agent if specified
    if args.agent:
        rows = [r for r in rows if r["agent"] == args.agent]
        if not rows:
            print(f"[ERROR] No rows found for agent: {args.agent}")
            return
        print(f"Processing agent: {args.agent} ({len(rows)} files)")

    # Resume — skip already processed
    already_done = set()
    manifest_exists = Path(args.output).exists()
    if args.resume and manifest_exists:
        already_done = load_existing_manifest(args.output)
        skipped = sum(1 for r in rows if r["filepath"] in already_done)
        print(f"Resume mode: skipping {skipped} already-processed files")
        rows = [r for r in rows if r["filepath"] not in already_done]

    if not rows:
        print("Nothing to process.")
        return

    print(f"\nProcessing {len(rows)} PDFs...")
    print(f"Model: {GROQ_MODEL}")
    print(f"Output: {args.output}\n")

    # Open output CSV
    fieldnames = [
        "agent", "original_filename", "filepath",
        "extracted_title", "suggested_name",
        "confidence", "extraction_status", "approved",
    ]

    open_mode = "a" if (args.resume and manifest_exists) else "w"
    out_file = open(args.output, open_mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
    if open_mode == "w":
        writer.writeheader()

    # Stats
    stats = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "extract_failed": 0}

    try:
        for i, row in enumerate(rows, 1):
            agent    = row["agent"]
            filename = row["original_filename"]
            filepath = row["filepath"]

            print(f"[{i:>3}/{len(rows)}] {agent:<20} {filename[:60]}")

            # Extract PDF text
            text, extraction_status = extract_pdf_text(filepath)

            if extraction_status.startswith("error") or extraction_status == "failed":
                stats["extract_failed"] += 1
                print(f"         ⚠ Extraction: {extraction_status}")

            # Call Groq
            result = call_groq(client, agent, filename, text)

            confidence = result.get("confidence", "LOW")
            stats[confidence] = stats.get(confidence, 0) + 1

            confidence_icon = {"HIGH": "✓", "MEDIUM": "~", "LOW": "⚠"}.get(confidence, "?")
            print(f"         {confidence_icon} {confidence:<6} → {result['suggested_name']}")

            writer.writerow({
                "agent":             agent,
                "original_filename": filename,
                "filepath":          filepath,
                "extracted_title":   result.get("extracted_title", ""),
                "suggested_name":    result["suggested_name"],
                "confidence":        confidence,
                "extraction_status": extraction_status,
                "approved":          "",   # human fills this in
            })
            out_file.flush()

            time.sleep(INTER_FILE_DELAY)

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Progress saved. Re-run with --resume to continue.")

    finally:
        out_file.close()

    # Summary
    total = sum(stats[k] for k in ["HIGH", "MEDIUM", "LOW"])
    print("\n" + "─" * 60)
    print("SUMMARY")
    print("─" * 60)
    print(f"  Total processed : {total}")
    print(f"  HIGH confidence : {stats['HIGH']}  ✓ — likely correct")
    print(f"  MEDIUM confidence: {stats['MEDIUM']}  ~ — review recommended")
    print(f"  LOW confidence  : {stats['LOW']}  ⚠ — MUST review")
    print(f"  Extract failures: {stats['extract_failed']}  (scanned/image PDFs)")
    print(f"\nManifest written to: {args.output}")
    print("\nNEXT STEPS:")
    print("  1. Open rename_manifest.csv")
    print("  2. Review suggested_name for each row")
    print("  3. Edit any suggested_name values that are wrong")
    print("  4. Set approved=yes for rows you accept")
    print("  5. Run: python apply_renames.py")
    print()


if __name__ == "__main__":
    main()
