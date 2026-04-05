#!/usr/bin/env python3
"""
rename_with_llm.py — Step 2 of 4 in the multi-agent RAG pipeline.

Reads inventory.csv, extracts text from the first 3 pages of each PDF,
sends it to an LLM to suggest a standardized filename, and outputs
rename_manifest.csv for human review.

NOTHING IS RENAMED HERE. This script only produces suggestions.
Script 3 (apply_renames.py) does the actual renaming after you approve.

Provider fallback chain (in order):
    1. Groq       — llama-3.3-70b-versatile         (primary, fastest)
    2. Gemini     — gemini-2.0-flash                 (free tier, generous limits)
    3. OpenRouter — mistral-small-3.2-24b (free)     (last resort)

On a 429 from any provider, the script immediately rotates to the next one.
No waiting — just seamless fallback. All providers produce identical CSV output.

Usage:
    python rename_with_llm.py
    python rename_with_llm.py --resume          # continue after interruption
    python rename_with_llm.py --agent ICT       # one agent only
    python rename_with_llm.py --inventory inventory.csv --output rename_manifest.csv

Output columns:
    agent | original_filename | filepath | reasoning | extracted_title |
    suggested_name | confidence | extraction_status | approved

    Corrupt files: confidence=SKIP, suggested_name=original filename (no rename)
"""

import os
import re
import csv
import json
import time
import argparse
import pdfplumber
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the same directory as this script
load_dotenv(Path(__file__).parent / ".env")


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_INVENTORY = "inventory.csv"
DEFAULT_OUTPUT    = "rename_manifest.csv"

MAX_TEXT_CHARS   = 3500
RETRY_ATTEMPTS   = 2
RETRY_DELAY      = 3
INTER_FILE_DELAY = 2.0

GROQ_MODEL       = "llama-3.3-70b-versatile"
GEMINI_MODEL     = "gemini-2.0-flash"
OPENROUTER_MODEL = "mistralai/mistral-small-3.2-24b-instruct:free"  # fixed — old model removed

AGENT_DOMAINS = {
    "Education":      "Kenya education sector — Ministry of Education, KNEC, TSC, universities, TVET",
    "Agriculture":    "Kenya agriculture sector — KALRO, AFA, crop research institutes, food security",
    "ICT":            "Kenya ICT and digital economy — CA, KICTANET, digital policy, cybersecurity, fintech",
    "Infastructure":  "Kenya infrastructure — roads, energy, ports, SGR, KETRACO, EPRA, KURA, KeNHA",
    "AntiCorruption": "Kenya governance and anti-corruption — EACC, PPRA, PPOA, procurement, audit",
    "President":      "Kenya cross-cutting policy — presidency, cross-ministry, national planning",
    "Finance":        "Kenya public finance — National Treasury, CBK, KRA, debt management, fiscal policy",
}

DOC_TYPES = [
    "Annual-Report", "Strategic-Plan", "Policy", "Act", "Bill", "Regulations",
    "Budget", "Audit-Report", "Survey", "Statistics-Report", "Research-Report",
    "Framework", "Manual", "Guidelines", "Assessment", "Conference-Report",
    "IGF-Report", "Forensic-Audit", "Financial-Statements", "MER-Report",
    "Case-Report", "Masterplan", "Bulletin", "Catalogue",
]


# ── Prompts ───────────────────────────────────────────────────────────────────

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

CONFIDENCE LEVELS — assign the highest level that honestly applies:
  HIGH   — title, issuer, AND year are all explicitly stated in the extracted text; no inference needed
  MEDIUM — a clear, usable filename can be produced; one element (usually year or issuer) requires
           reasonable inference from context, the filename, or domain knowledge.
           THIS IS THE NORMAL, EXPECTED CASE FOR MOST GOVERNMENT DOCUMENTS.
  LOW    — the document title itself is genuinely indeterminate; text is empty, OCR garbage,
           or so fragmentary that any name would be a fabrication. Use LOW sparingly.

Most documents should receive MEDIUM. Only fall back to LOW if you truly cannot determine
what this document is about even with reasonable inference from its filename and domain.

Respond ONLY with a JSON object — no preamble, no explanation, no markdown fences:
{{"reasoning": "1-2 sentences: what title/issuer/year you found and where, or why you cannot determine them", "suggested_name": "YYYY-DocType-Issuer-Description.pdf", "confidence": "HIGH|MEDIUM|LOW", "extracted_title": "exact title as it appears in the document"}}"""

USER_PROMPT_TEMPLATE = """Agent domain: {agent_domain}
Original filename: {original_filename}

First 3 pages of document text:
---
{text}
---

Suggest a standardized filename following the naming convention."""


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _sanitize_name(name: str) -> str:
    name = re.sub(r'\s+', '-', name)
    name = re.sub(r'^_+', '', name)
    name = re.sub(r'[&@#$%^*+=\[\]{}|\\<>?]', '', name)
    if not name.lower().endswith('.pdf'):
        name += '.pdf'
    return name

def _fallback_name(original: str) -> str:
    return _sanitize_name(Path(original).stem)

def _is_rate_limit(err: str) -> bool:
    # FIX: was too broad — "quota" matched unrelated errors in PDF content
    # Now strictly matches actual rate limit HTTP responses only
    return "429" in err or "rate_limit_exceeded" in err.lower() or "rateLimitExceeded" in err

def _parse_result(raw: str, original_filename: str) -> dict:
    raw = re.sub(r'^```json\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw).strip()
    result = json.loads(raw)
    if "suggested_name" not in result:
        raise ValueError("Missing suggested_name")
    result["suggested_name"] = _sanitize_name(result["suggested_name"])
    result.setdefault("confidence",      "LOW")
    result.setdefault("extracted_title", "")
    result.setdefault("reasoning",       "")
    return result

RATE_LIMITED = {"confidence": "PROVIDER_FAILED", "reason": "rate_limited"}
PROVIDER_ERR = {"confidence": "PROVIDER_FAILED", "reason": "error"}


# ── Provider 1: Groq ──────────────────────────────────────────────────────────

def call_groq(system: str, user: str, original_filename: str) -> dict:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return PROVIDER_ERR

    try:
        from groq import Groq
        client   = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=300,
        )
        return _parse_result(response.choices[0].message.content, original_filename)

    except Exception as e:
        if _is_rate_limit(str(e)):
            return RATE_LIMITED
        return PROVIDER_ERR


# ── Provider 2: Gemini ────────────────────────────────────────────────────────

def call_gemini(system: str, user: str, original_filename: str) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return PROVIDER_ERR

    try:
        from google import genai
        from google.genai import types

        client   = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0,
                max_output_tokens=300,
            ),
        )
        return _parse_result(response.text, original_filename)

    except Exception as e:
        if _is_rate_limit(str(e)):
            return RATE_LIMITED
        return PROVIDER_ERR


# ── Provider 3: OpenRouter ────────────────────────────────────────────────────

def call_openrouter(system: str, user: str, original_filename: str) -> dict:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return PROVIDER_ERR

    try:
        import requests
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
                "HTTP-Referer":  "https://github.com/PRES-Executive",
                "X-Title":       "PRES-Rename-Pipeline",
            },
            json={
                "model":           OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "temperature":     0,
                "max_tokens":      300,
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )
        if response.status_code == 429:
            return RATE_LIMITED
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return _parse_result(content, original_filename)

    except Exception as e:
        if _is_rate_limit(str(e)):
            return RATE_LIMITED
        return PROVIDER_ERR


# ── Fallback chain ────────────────────────────────────────────────────────────

PROVIDER_CHAIN = [
    ("Groq",       call_groq),
    ("Gemini",     call_gemini),
    ("OpenRouter", call_openrouter),
]

def call_llm(agent: str, original_filename: str, text: str) -> dict:
    system = SYSTEM_PROMPT.format(doc_types=", ".join(DOC_TYPES))
    user   = USER_PROMPT_TEMPLATE.format(
        agent_domain=AGENT_DOMAINS.get(agent, f"{agent} sector Kenya government"),
        original_filename=original_filename,
        text=text if text else "[No text extracted — scanned or image-only PDF]",
    )

    for provider_name, provider_fn in PROVIDER_CHAIN:
        for attempt in range(RETRY_ATTEMPTS):
            result = provider_fn(system, user, original_filename)

            # Success
            if result.get("confidence") != "PROVIDER_FAILED":
                if provider_name != PROVIDER_CHAIN[0][0]:
                    print(f"         ✦ Used: {provider_name}")
                return result

            # Rate limited — rotate immediately
            if result.get("reason") == "rate_limited":
                print(f"         ↷ {provider_name} rate limited — rotating")
                break

            # Other error — retry within this provider
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
                continue

            # Retries exhausted — rotate
            print(f"         ↷ {provider_name} failed — rotating")
            break

    return {
        "reasoning":       "All providers failed or rate limited",
        "suggested_name":  _fallback_name(original_filename),
        "confidence":      "LOW",
        "extracted_title": "",
    }


# ── Corrupt PDF detection ──────────────────────────────────────────────────────

CORRUPT_PDF_ERRORS = [
    "'L' format requires 0 <= number <= 4294967295",
    "invalid literal for int()",
    "PdfReadError",
    "struct.error",
]

def _is_corrupt_error(err: str) -> bool:
    return any(sig in err for sig in CORRUPT_PDF_ERRORS)


# ── PDF text extraction ────────────────────────────────────────────────────────

def extract_pdf_text(filepath: str, max_chars: int = MAX_TEXT_CHARS) -> tuple[str, str]:
    try:
        with pdfplumber.open(filepath) as pdf:
            parts = []
            for page in pdf.pages[:3]:
                try:
                    t = page.extract_text()
                    if t:
                        parts.append(t.strip())
                except Exception as e:
                    if _is_corrupt_error(str(e)):
                        return "", "corrupt_pdf"
                    continue

            full = "\n\n".join(parts).strip()
            if not full:
                return "", "failed"
            truncated = full[:max_chars]
            return truncated, "partial" if len(full) > max_chars else "ok"

    except Exception as e:
        err = str(e)
        if _is_corrupt_error(err):
            return "", "corrupt_pdf"
        return "", f"error: {err[:80]}"


def extract_pdf_text_with_ocr(filepath: str, max_chars: int = MAX_TEXT_CHARS) -> tuple[str, str]:
    text, status = extract_pdf_text(filepath, max_chars)

    if status == "corrupt_pdf":
        return "", "corrupt_pdf"
    if text.strip():
        return text, status

    try:
        import fitz
        import pytesseract
        from PIL import Image
        import io

        doc   = fitz.open(filepath)
        parts = []
        for i in range(min(3, len(doc))):
            pix = doc[i].get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            t   = pytesseract.image_to_string(img).strip()
            if t:
                parts.append(t)
        doc.close()

        combined = "\n\n".join(parts)[:max_chars]
        return (combined, "ocr") if combined.strip() else ("", "ocr_failed")

    except ImportError:
        return "", "failed"
    except Exception as e:
        return "", f"ocr_error: {str(e)[:60]}"


# ── Record helpers ─────────────────────────────────────────────────────────────

FIELDNAMES = [
    "agent", "original_filename", "filepath",
    "reasoning", "extracted_title", "suggested_name",
    "confidence", "extraction_status", "approved",
]

def make_corrupt_record(row: dict) -> dict:
    return {
        "agent":             row["agent"],
        "original_filename": row["original_filename"],
        "filepath":          row["filepath"],
        "reasoning":         "CORRUPT — binary error, unreadable by pdfplumber and OCR",
        "extracted_title":   "",
        "suggested_name":    row["original_filename"],
        "confidence":        "SKIP",
        "extraction_status": "corrupt_pdf",
        "approved":          "",
    }

def load_existing_manifest(output_path: str) -> set[str]:
    processed = set()
    if Path(output_path).exists():
        with open(output_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("filepath"):
                    processed.add(row["filepath"])
    return processed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM-powered PDF renamer with multi-provider fallback."
    )
    parser.add_argument("--inventory", default=DEFAULT_INVENTORY)
    parser.add_argument("--output",    default=DEFAULT_OUTPUT)
    parser.add_argument("--agent",     default=None)
    parser.add_argument("--resume",    action="store_true")
    args = parser.parse_args()

    providers_armed = []
    if os.environ.get("GROQ_API_KEY"):       providers_armed.append("Groq")
    if os.environ.get("GEMINI_API_KEY"):     providers_armed.append("Gemini")
    if os.environ.get("OPENROUTER_API_KEY"): providers_armed.append("OpenRouter")

    if not providers_armed:
        print("[ERROR] No API keys found. Check your .env file.")
        print("        Expected: GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY")
        return

    if not Path(args.inventory).exists():
        print(f"[ERROR] Inventory file not found: {args.inventory}")
        return

    with open(args.inventory, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.agent:
        rows = [r for r in rows if r["agent"] == args.agent]
        if not rows:
            print(f"[ERROR] No rows for agent: {args.agent}")
            return

    already_done    = set()
    manifest_exists = Path(args.output).exists()
    if args.resume and manifest_exists:
        already_done = load_existing_manifest(args.output)
        skipped      = sum(1 for r in rows if r["filepath"] in already_done)
        print(f"Resume mode: skipping {skipped} already-processed files")
        rows = [r for r in rows if r["filepath"] not in already_done]

    if not rows:
        print("Nothing to process.")
        return

    print(f"\nProviders armed  : {' → '.join(providers_armed)}")
    print(f"Processing       : {len(rows)} PDFs")
    print(f"Output           : {args.output}")
    print(f"Inter-file delay : {INTER_FILE_DELAY}s\n")

    open_mode = "a" if (args.resume and manifest_exists) else "w"
    out_file  = open(args.output, open_mode, newline="", encoding="utf-8")
    writer    = csv.DictWriter(out_file, fieldnames=FIELDNAMES)
    if open_mode == "w":
        writer.writeheader()

    stats = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "SKIP": 0, "ocr": 0, "extract_failed": 0}

    try:
        for i, row in enumerate(rows, 1):
            agent    = row["agent"]
            filename = row["original_filename"]
            filepath = row["filepath"]

            print(f"[{i:>3}/{len(rows)}] {agent:<20} {filename[:60]}")

            text, extraction_status = extract_pdf_text_with_ocr(filepath)

            if extraction_status == "corrupt_pdf":
                stats["SKIP"] += 1
                print(f"         ✗ CORRUPT — SKIP record written")
                writer.writerow(make_corrupt_record(row))
                out_file.flush()
                continue

            if extraction_status in ("failed", "ocr_failed") or \
               extraction_status.startswith(("ocr_error", "error")):
                stats["extract_failed"] += 1
                print(f"         ⚠ Extraction: {extraction_status}")
            elif extraction_status == "ocr":
                stats["ocr"] += 1
                print(f"         ○ OCR used")

            result     = call_llm(agent, filename, text)
            confidence = result.get("confidence", "LOW")
            stats[confidence] = stats.get(confidence, 0) + 1

            icon = {"HIGH": "✓", "MEDIUM": "~", "LOW": "⚠"}.get(confidence, "?")
            print(f"         {icon} {confidence:<6} → {result['suggested_name']}")
            if result.get("reasoning"):
                print(f"         ↳ {result['reasoning'][:80]}")

            writer.writerow({
                "agent":             agent,
                "original_filename": filename,
                "filepath":          filepath,
                "reasoning":         result.get("reasoning", ""),
                "extracted_title":   result.get("extracted_title", ""),
                "suggested_name":    result["suggested_name"],
                "confidence":        confidence,
                "extraction_status": extraction_status,
                "approved":          "",
            })
            out_file.flush()
            time.sleep(INTER_FILE_DELAY)

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Progress saved. Re-run with --resume to continue.")
    finally:
        out_file.close()

    total = sum(stats[k] for k in ["HIGH", "MEDIUM", "LOW", "SKIP"])
    print("\n" + "─" * 60)
    print("SUMMARY")
    print("─" * 60)
    print(f"  Total processed  : {total}")
    print(f"  HIGH confidence  : {stats['HIGH']}   ✓ — correct")
    print(f"  MEDIUM confidence: {stats['MEDIUM']}   ~ — review recommended")
    print(f"  LOW confidence   : {stats['LOW']}   ⚠ — MUST review")
    print(f"  SKIP (corrupt)   : {stats['SKIP']}   ✗ — no rename suggested")
    print(f"  OCR used         : {stats['ocr']}   ○ — scanned PDFs")
    print(f"  Extract failures : {stats['extract_failed']}")
    print(f"\nManifest: {args.output}")
    print("\nNEXT STEPS:")
    print("  1. Open rename_manifest.csv and review suggested_name + reasoning")
    print("  2. Edit any suggested_name values that are wrong")
    print("  3. Set approved=yes for rows you accept")
    print("  4. SKIP rows: delete the PDF or replace with a clean copy")
    print("  5. Run: python apply_renames.py")
    print()


if __name__ == "__main__":
    main()