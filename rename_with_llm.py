#!/usr/bin/env python3
"""
rename_with_llm.py — Step 2 of 4 in the multi-agent RAG pipeline.

Provider chain:
    1. Ollama (local)  — qwen2.5:1.5b, filename-only, zero rate limits
    2. Groq            — cloud fallback, only when Ollama returns LOW or placeholder
    3. Gemini          — fallback if Groq rate-limits
    4. OpenRouter      — last resort (qwen3-next-80b free tier)

Strategy:
    - Ollama gets filename only (~50 tokens, instant, offline)
    - HIGH or MEDIUM with no placeholder text → done, no cloud call needed
    - LOW or placeholder detected → extract PDF text → try cloud providers
"""

import os
import re
import csv
import json
import time
import argparse
import pdfplumber
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_INVENTORY  = "inventory.csv"
DEFAULT_OUTPUT     = "rename_manifest.csv"

MAX_TEXT_CHARS     = 1500
RETRY_ATTEMPTS     = 2
RETRY_DELAY        = 3
INTER_FILE_DELAY   = 1.0
ROTATION_PAUSE     = 8

OLLAMA_HOST        = "http://localhost:11434"
OLLAMA_MODEL       = "qwen2.5:1.5b"
OLLAMA_TIMEOUT     = 30

GROQ_MODEL         = "llama-3.3-70b-versatile"
GEMINI_MODEL       = "gemini-2.0-flash"
OPENROUTER_MODEL   = "qwen/qwen3-next-80b-a3b-instruct:free"  # fixed — previous model removed

PLACEHOLDER_WORDS  = ["DocType", "Issuer", "Description", "ShortDescription"]
RETRY_CONFIDENCES  = {"LOW"}

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

OLLAMA_SYSTEM = """You are a document renaming expert for Kenyan government PDFs.
Produce a standardized filename in the format: YYYY-DocType-Issuer-ShortDescription.pdf

Rules:
- DocType from: {doc_types}
- Year: publication year or FY start year
- Issuer: real organization abbreviation (e.g. KNEC, PPRA, CBK, KETRACO, MoE, KALRO)
- ShortDescription: 2-4 real descriptive words, Title-Case, hyphens only
- No spaces, no special characters except hyphens
- Max 80 characters total
- NEVER use the placeholder words "DocType", "Issuer", "Description" literally in the output

Confidence:
  HIGH   — year, issuer, AND doc type are all obvious from the filename
  MEDIUM — one element needs a small inference but a clean name is clearly achievable
  LOW    — filename is too vague or ambiguous to produce a reliable name

Respond ONLY with valid JSON, nothing else:
{{"suggested_name": "YYYY-DocType-Issuer-Description.pdf", "confidence": "HIGH|MEDIUM|LOW", "reasoning": "one sentence"}}"""

CLOUD_SYSTEM = """You are a document naming expert for a Kenyan government AI research system.
Produce a clean standardized filename for a government PDF.

NAMING CONVENTION:
- Format: YYYY-DocType-Issuer-ShortDescription.pdf
- DocType from: {doc_types}
- Year: publication year or fiscal year start
- FY format: FY-YYYY-YY (e.g. FY-2023-24)
- No spaces — hyphens only. No special characters. Max 80 chars.

CONFIDENCE:
  HIGH   — title, issuer, AND year all explicit in text
  MEDIUM — clean name achievable with one reasonable inference (NORMAL CASE)
  LOW    — text empty, OCR garbage, or completely indeterminate

Respond ONLY with JSON:
{{"reasoning": "1-2 sentences", "suggested_name": "YYYY-DocType-Issuer-Description.pdf", "confidence": "HIGH|MEDIUM|LOW", "extracted_title": "exact title from document"}}"""

CLOUD_USER = """Agent domain: {agent_domain}
Original filename: {original_filename}

First pages of document text:
---
{text}
---

Suggest a standardized filename."""


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

def _has_placeholders(name: str) -> bool:
    return any(p in name for p in PLACEHOLDER_WORDS)

def _is_rate_limit(err: str) -> bool:
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


# ── Provider 0: Ollama (local) ────────────────────────────────────────────────

def check_ollama(model: str) -> bool:
    try:
        resp  = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        names = [m["name"] for m in resp.json().get("models", [])]
        return any(m.startswith(model.split(":")[0]) for m in names)
    except Exception:
        return False

def call_ollama(filename: str, model: str) -> dict:
    system = OLLAMA_SYSTEM.format(doc_types=", ".join(DOC_TYPES))
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model":   model,
                "stream":  False,
                "format":  "json",
                "options": {"temperature": 0},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": f"Filename: {filename}"},
                ],
            },
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        return _parse_result(resp.json()["message"]["content"], filename)
    except Exception as e:
        print(f"         [Ollama error] {str(e)[:80]}")
        return PROVIDER_ERR


# ── Cloud providers ───────────────────────────────────────────────────────────

def call_groq(system: str, user: str, filename: str) -> dict:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return PROVIDER_ERR
    try:
        from groq import Groq
        resp = Groq(api_key=api_key).chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0, max_tokens=300,
        )
        return _parse_result(resp.choices[0].message.content, filename)
    except Exception as e:
        err = str(e)
        print(f"         [Groq error] {err[:120]}")
        return RATE_LIMITED if _is_rate_limit(err) else PROVIDER_ERR


def call_gemini(system: str, user: str, filename: str) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return PROVIDER_ERR
    try:
        from google import genai
        from google.genai import types
        # Use context manager to properly manage connection lifecycle
        with genai.Client(api_key=api_key) as client:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0,
                    max_output_tokens=300,
                ),
            )
        return _parse_result(resp.text, filename)
    except Exception as e:
        err = str(e)
        print(f"         [Gemini error] {err[:120]}")
        return RATE_LIMITED if _is_rate_limit(err) else PROVIDER_ERR


def call_openrouter(system: str, user: str, filename: str) -> dict:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return PROVIDER_ERR
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
                "HTTP-Referer":  "https://github.com/PRES-Executive",
                "X-Title":       "PRES-Rename-Pipeline",
            },
            json={
                "model":           OPENROUTER_MODEL,
                "messages":        [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature":     0,
                "max_tokens":      300,
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )
        if resp.status_code == 429:
            return RATE_LIMITED
        resp.raise_for_status()
        return _parse_result(resp.json()["choices"][0]["message"]["content"], filename)
    except Exception as e:
        err = str(e)
        print(f"         [OpenRouter error] {err[:120]}")
        return RATE_LIMITED if _is_rate_limit(err) else PROVIDER_ERR


CLOUD_CHAIN = [("Groq", call_groq), ("Gemini", call_gemini), ("OpenRouter", call_openrouter)]

def call_cloud(agent: str, filename: str, text: str) -> dict:
    system = CLOUD_SYSTEM.format(doc_types=", ".join(DOC_TYPES))
    user   = CLOUD_USER.format(
        agent_domain=AGENT_DOMAINS.get(agent, f"{agent} sector Kenya government"),
        original_filename=filename,
        text=text or "[No text extracted]",
    )
    for provider_name, provider_fn in CLOUD_CHAIN:
        for attempt in range(RETRY_ATTEMPTS):
            result = provider_fn(system, user, filename)
            if result.get("confidence") != "PROVIDER_FAILED":
                print(f"         ✦ Cloud: {provider_name}")
                return result
            if result.get("reason") == "rate_limited":
                print(f"         ↷ {provider_name} rate limited — rotating")
                time.sleep(ROTATION_PAUSE)
                break
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
                continue
            print(f"         ↷ {provider_name} failed — rotating")
            break
    return {
        "reasoning":       "All providers failed",
        "suggested_name":  _fallback_name(filename),
        "confidence":      "LOW",
        "extracted_title": "",
    }


# ── PDF extraction ─────────────────────────────────────────────────────────────

CORRUPT_PDF_ERRORS = [
    "'L' format requires 0 <= number <= 4294967295",
    "invalid literal for int()", "PdfReadError", "struct.error",
]

def _is_corrupt_error(err: str) -> bool:
    return any(sig in err for sig in CORRUPT_PDF_ERRORS)

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
            return full[:max_chars], "partial" if len(full) > max_chars else "ok"
    except Exception as e:
        err = str(e)
        return ("", "corrupt_pdf") if _is_corrupt_error(err) else ("", f"error: {err[:80]}")

def extract_pdf_text_with_ocr(filepath: str, max_chars: int = MAX_TEXT_CHARS) -> tuple[str, str]:
    text, status = extract_pdf_text(filepath, max_chars)
    if status == "corrupt_pdf" or text.strip():
        return text, status
    try:
        import fitz, pytesseract, io
        from PIL import Image
        doc   = fitz.open(filepath)
        parts = [
            pytesseract.image_to_string(
                Image.open(io.BytesIO(doc[i].get_pixmap(dpi=200).tobytes("png")))
            ).strip()
            for i in range(min(3, len(doc)))
        ]
        doc.close()
        combined = "\n\n".join(p for p in parts if p)[:max_chars]
        return (combined, "ocr") if combined.strip() else ("", "ocr_failed")
    except ImportError:
        return "", "failed"
    except Exception as e:
        return "", f"ocr_error: {str(e)[:60]}"


# ── Record helpers ─────────────────────────────────────────────────────────────

FIELDNAMES = [
    "agent", "original_filename", "filepath", "reasoning",
    "extracted_title", "suggested_name", "confidence", "extraction_status", "approved",
]

def make_corrupt_record(row: dict) -> dict:
    return {
        "agent":             row["agent"],
        "original_filename": row["original_filename"],
        "filepath":          row["filepath"],
        "reasoning":         "CORRUPT — unreadable by pdfplumber and OCR",
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
    parser = argparse.ArgumentParser(description="LLM-powered PDF renamer — local-first via Ollama.")
    parser.add_argument("--inventory",    default=DEFAULT_INVENTORY)
    parser.add_argument("--output",       default=DEFAULT_OUTPUT)
    parser.add_argument("--agent",        default=None)
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--retry-low",    action="store_true",
                        help="Reprocess LOW confidence rows with cloud")
    parser.add_argument("--ollama-model", default=OLLAMA_MODEL)
    args = parser.parse_args()

    ollama_model = args.ollama_model
    ollama_ready = check_ollama(ollama_model)

    if not ollama_ready:
        print(f"[WARNING] Ollama not running or model '{ollama_model}' not found.")
        print(f"  Fix:  ollama serve  (in separate terminal)")
        print(f"        ollama pull {ollama_model}")
        print(f"  Continuing with cloud-only fallback...\n")

    providers_armed = []
    if ollama_ready:                         providers_armed.append(f"Ollama({ollama_model})")
    if os.environ.get("GROQ_API_KEY"):       providers_armed.append("Groq")
    if os.environ.get("GEMINI_API_KEY"):     providers_armed.append("Gemini")
    if os.environ.get("OPENROUTER_API_KEY"): providers_armed.append("OpenRouter")

    if not providers_armed:
        print("[ERROR] No providers available.")
        return

    if not Path(args.inventory).exists():
        print(f"[ERROR] Inventory not found: {args.inventory}")
        return

    with open(args.inventory, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.agent:
        rows = [r for r in rows if r["agent"] == args.agent]
        if not rows:
            print(f"[ERROR] No rows for agent: {args.agent}")
            return

    manifest_exists = Path(args.output).exists()

    if args.resume and manifest_exists:
        already_done = load_existing_manifest(args.output)
        skipped      = sum(1 for r in rows if r["filepath"] in already_done)
        print(f"Resume mode: skipping {skipped} already-processed files")
        rows = [r for r in rows if r["filepath"] not in already_done]

    if args.retry_low and manifest_exists:
        with open(args.output, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        keep_rows  = [r for r in existing if r["confidence"] not in RETRY_CONFIDENCES]
        retry_rows = [r for r in existing if r["confidence"] in RETRY_CONFIDENCES]
        if not retry_rows:
            print("No LOW confidence rows found.")
            return
        print(f"Retry mode: {len(retry_rows)} LOW rows → reprocessing, {len(keep_rows)} kept\n")
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writeheader()
            w.writerows(keep_rows)
        retry_paths = {r["filepath"] for r in retry_rows}
        rows = [r for r in rows if r["filepath"] in retry_paths]

    if not rows:
        print("Nothing to process.")
        return

    print(f"Providers  : {' → '.join(providers_armed)}")
    print(f"Processing : {len(rows)} PDFs")
    print(f"Output     : {args.output}")
    print(f"Strategy   : Ollama(filename only) → cloud(+PDF text) if LOW or placeholder\n")

    if args.retry_low:
        open_mode = "a"
    else:
        open_mode = "a" if (args.resume and manifest_exists) else "w"

    out_file = open(args.output, open_mode, newline="", encoding="utf-8")
    writer   = csv.DictWriter(out_file, fieldnames=FIELDNAMES)
    if open_mode == "w":
        writer.writeheader()

    stats = {
        "HIGH": 0, "MEDIUM": 0, "LOW": 0, "SKIP": 0,
        "ocr": 0, "extract_failed": 0, "ollama_resolved": 0, "cloud_calls": 0,
    }

    try:
        for i, row in enumerate(rows, 1):
            agent    = row["agent"]
            filename = row["original_filename"]
            filepath = row["filepath"]

            print(f"[{i:>3}/{len(rows)}] {agent:<16} {filename[:65]}")

            result            = None
            extraction_status = "skipped"

            # Step 1 — Ollama: filename only, instant, no API cost
            if ollama_ready:
                result = call_ollama(filename, ollama_model)
                name   = result.get("suggested_name", "")
                conf   = result.get("confidence", "LOW")

                if conf not in ("PROVIDER_FAILED", "LOW") and not _has_placeholders(name):
                    stats["ollama_resolved"] += 1
                else:
                    if _has_placeholders(name):
                        print(f"         ⚠ Ollama placeholder detected — escalating to cloud")
                    result = None

            # Step 2 — Cloud: extract PDF text and call cloud provider
            if result is None:
                text, extraction_status = extract_pdf_text_with_ocr(filepath)

                if extraction_status == "corrupt_pdf":
                    stats["SKIP"] += 1
                    print(f"         ✗ CORRUPT")
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

                stats["cloud_calls"] += 1
                result = call_cloud(agent, filename, text)

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

    total     = sum(stats[k] for k in ["HIGH", "MEDIUM", "LOW", "SKIP"])
    cloud_pct = round(100 * stats["cloud_calls"] / total) if total else 0

    print("\n" + "─" * 60)
    print("SUMMARY")
    print("─" * 60)
    print(f"  Total processed  : {total}")
    print(f"  HIGH confidence  : {stats['HIGH']}   ✓")
    print(f"  MEDIUM confidence: {stats['MEDIUM']}   ~")
    print(f"  LOW confidence   : {stats['LOW']}   ⚠  review these")
    print(f"  SKIP (corrupt)   : {stats['SKIP']}   ✗")
    print(f"  OCR used         : {stats['ocr']}")
    print(f"  Ollama resolved  : {stats['ollama_resolved']}  (no cloud call)")
    print(f"  Cloud API calls  : {stats['cloud_calls']}  ({cloud_pct}% of files)")
    print(f"\nManifest: {args.output}")
    print("\nNEXT STEPS:")
    print("  1. Review rename_manifest.csv — LOW confidence rows first")
    print("  2. Edit suggested_name values that are wrong")
    print("  3. Set approved=yes for accepted rows")
    print("  4. Run: python apply_renames.py")
    print()


if __name__ == "__main__":
    main()