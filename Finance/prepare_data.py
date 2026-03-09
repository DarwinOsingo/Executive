"""
prepare_data.py — Finance Cabinet Agent Data Pipeline
Kenya AI Executive Roundtable
Stage 1: Extract → Stage 2: Clean → Stage 3: Chunk → Stage 4: Convert to JSONL
"""

import os
import re
import json
import argparse
from pathlib import Path

# ── Dependencies ───────────────────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("Run: pip install pymupdf")

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text): return len(enc.encode(text))
except ImportError:
    # Fallback: rough word-based estimate
    def count_tokens(text): return int(len(text.split()) * 1.3)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
RAW_DIR    = BASE_DIR / "data" / "raw"
PROC_DIR   = BASE_DIR / "data" / "processed"
RAG_DIR    = BASE_DIR / "data" / "rag"

PROC_DIR.mkdir(parents=True, exist_ok=True)
RAG_DIR.mkdir(parents=True, exist_ok=True)

# ── Finance CS System Prompt ───────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are Kenya's Cabinet Secretary for the National Treasury and Economic Planning. "
    "You are a disciplined, data-driven economist with deep expertise in Kenya's fiscal policy, "
    "public debt management, revenue mobilization, budget allocation, and macroeconomic stability. "
    "You speak with authority grounded in Kenya's own government data. You cite specific figures, "
    "budget lines, and economic indicators. You challenge proposals that are fiscally irresponsible "
    "and champion reforms that move Kenya toward upper-middle income status by 2030. "
    "You are not a politician — you are a technocrat who follows the numbers."
)

# ── STAGE 1: EXTRACT ──────────────────────────────────────────────────────────

def extract_pdf(pdf_path: Path) -> str:
    """Extract raw text from a PDF using PyMuPDF. Falls back to OCR hint if blank."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append(f"[PAGE {i+1}]\n{text}")
        else:
            print(f"  ⚠️  Page {i+1} appears scanned/blank — consider OCR with pytesseract")
    raw = "\n".join(pages)
    print(f"  ✓ Extracted {len(doc)} pages, {len(raw):,} characters")
    return raw


# ── STAGE 2: CLEAN ────────────────────────────────────────────────────────────

# Patterns that are noise (headers, footers, page stamps, etc.)
_NOISE_PATTERNS = [
    r"\[PAGE \d+\]",                       # page markers we added
    r"^\s*\d+\s*$",                        # lone page numbers
    r"(?i)confidential|draft|for official use only",
    r"(?i)republic of kenya",              # repeated masthead
    r"(?i)national treasury",              # repeated masthead
    r"www\.\S+",                           # URLs
    r"\b\d{1,2}/\d{1,2}/\d{4}\b",         # dates as artifacts
    r"_{3,}",                              # rule lines
    r"-{3,}",
    r"={3,}",
    r"\f",                                 # form feed
]

def clean_text(raw: str) -> str:
    """Remove noise and normalize whitespace."""
    text = raw

    # Remove noise patterns (line-by-line for line-based patterns)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        skip = False
        for pat in _NOISE_PATTERNS:
            if re.fullmatch(pat.strip(), line.strip()):
                skip = True
                break
        if not skip:
            # Inline pattern cleanup
            for pat in [r"www\.\S+", r"_{3,}", r"-{3,}", r"={3,}", r"\f"]:
                line = re.sub(pat, " ", line)
            cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)          # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)        # max 2 consecutive newlines
    text = text.strip()

    print(f"  ✓ Cleaned: {len(text):,} characters remain")
    return text


# ── STAGE 3: CHUNK ────────────────────────────────────────────────────────────

def smart_chunk(text: str, min_tokens: int = 200, max_tokens: int = 800) -> list[str]:
    """
    Split into topic-coherent chunks.
    Strategy: split on section headers first, then re-split oversized blocks by paragraph.
    """
    # Detect section boundaries (numbered sections, ALL CAPS headings, etc.)
    section_pattern = re.compile(
        r"(?=\n(?:"
        r"\d+[\.\d]*\s+[A-Z]"            # e.g. "3.1 Revenue"
        r"|[A-Z][A-Z\s]{8,}"             # ALL CAPS headers ≥ 8 chars
        r"|Chapter\s+\d+"               # Chapter headings
        r"|PART\s+[IVXLC]+"             # Roman numeral parts
        r"))",
        re.MULTILINE
    )

    raw_sections = section_pattern.split(text)
    chunks = []

    for section in raw_sections:
        section = section.strip()
        if not section:
            continue
        token_count = count_tokens(section)

        if token_count <= max_tokens:
            if token_count >= min_tokens:
                chunks.append(section)
            # else: too short, skip (likely a heading orphan)
        else:
            # Re-split oversized section by paragraph
            paras = re.split(r"\n\n+", section)
            buffer = ""
            for para in paras:
                candidate = (buffer + "\n\n" + para).strip() if buffer else para.strip()
                if count_tokens(candidate) <= max_tokens:
                    buffer = candidate
                else:
                    if buffer and count_tokens(buffer) >= min_tokens:
                        chunks.append(buffer)
                    buffer = para.strip()
            if buffer and count_tokens(buffer) >= min_tokens:
                chunks.append(buffer)

    print(f"  ✓ Produced {len(chunks)} chunks ({min_tokens}–{max_tokens} tokens each)")
    return chunks


# ── STAGE 4: CONVERT TO JSONL ─────────────────────────────────────────────────

# Finance-domain question templates — inject chunk content as context
_QUESTION_TEMPLATES = [
    "As Kenya's Cabinet Secretary for the National Treasury, what does this fiscal data tell us about Kenya's economic trajectory?\n\nContext:\n{chunk}",
    "Analyze the following budget policy information and identify its implications for Kenya's public debt sustainability:\n\n{chunk}",
    "A fellow cabinet member challenges this expenditure plan as wasteful. Defend or critique it using the following data:\n\n{chunk}",
    "How does the following budget statement align with Kenya's Vision 2030 upper-middle income goal?\n\n{chunk}",
    "What revenue mobilization opportunities or fiscal risks are embedded in the following policy section?\n\n{chunk}",
    "From your perspective as National Treasury CS, summarize the key fiscal positions in the following excerpt and their second-order consequences:\n\n{chunk}",
]

def chunk_to_training_record(chunk: str, template_idx: int) -> dict:
    """Wrap a chunk into a JSONL training record."""
    template = _QUESTION_TEMPLATES[template_idx % len(_QUESTION_TEMPLATES)]
    user_msg = template.format(chunk=chunk)
    # The assistant response is intentionally left as a placeholder —
    # for fine-tuning you need real completions. This pipeline produces
    # the INPUT side; completions should be generated via Groq/GPT4 or
    # written by the researcher for a gold-standard dataset.
    return {
        "messages": [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": user_msg},
            {"role": "assistant","content": "[[COMPLETION_NEEDED]]"},
        ]
    }

def chunks_to_jsonl(chunks: list[str], output_path: Path):
    """Write chunks to JSONL training format."""
    records = []
    for i, chunk in enumerate(chunks):
        record = chunk_to_training_record(chunk, template_idx=i)
        records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  ✓ Wrote {len(records)} JSONL records → {output_path}")
    return records


# ── RAG CHUNK EXPORT (ChromaDB-ready) ─────────────────────────────────────────

def export_rag_chunks(chunks: list[str], source_name: str, output_path: Path):
    """
    Export chunks in a format ready for ChromaDB ingestion.
    Each line: {"id": "...", "document": "...", "metadata": {...}}
    """
    rag_records = []
    for i, chunk in enumerate(chunks):
        rag_records.append({
            "id":       f"{source_name}_chunk_{i:04d}",
            "document": chunk,
            "metadata": {
                "source":    source_name,
                "chunk_idx": i,
                "agent":     "finance_cs",
                "tokens":    count_tokens(chunk),
            }
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for record in rag_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  ✓ RAG export: {len(rag_records)} chunks → {output_path}")


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def process_pdf(pdf_path: Path):
    source_name = pdf_path.stem.lower().replace(" ", "_")
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path.name}")
    print(f"{'='*60}")

    # Stage 1
    print("\n[Stage 1] Extracting text...")
    raw = extract_pdf(pdf_path)

    raw_txt_path = PROC_DIR / f"{source_name}_raw.txt"
    raw_txt_path.write_text(raw, encoding="utf-8")
    print(f"  ✓ Raw text saved → {raw_txt_path}")

    # Stage 2
    print("\n[Stage 2] Cleaning...")
    cleaned = clean_text(raw)

    clean_txt_path = PROC_DIR / f"{source_name}_clean.txt"
    clean_txt_path.write_text(cleaned, encoding="utf-8")
    print(f"  ✓ Clean text saved → {clean_txt_path}")

    # Stage 3
    print("\n[Stage 3] Chunking...")
    chunks = smart_chunk(cleaned, min_tokens=200, max_tokens=700)

    # Stage 4 — Training JSONL
    print("\n[Stage 4] Converting to JSONL...")
    jsonl_path = PROC_DIR / f"{source_name}_training.jsonl"
    chunks_to_jsonl(chunks, jsonl_path)

    # RAG export
    rag_path = RAG_DIR / f"{source_name}_rag.jsonl"
    export_rag_chunks(chunks, source_name, rag_path)

    # Summary
    print(f"\n{'─'*60}")
    print(f"✅ Done: {pdf_path.name}")
    print(f"   Training records : {len(chunks)}")
    print(f"   Avg tokens/chunk : {sum(count_tokens(c) for c in chunks) // max(len(chunks),1)}")
    print(f"   JSONL            : {jsonl_path}")
    print(f"   RAG chunks       : {rag_path}")
    print(f"{'─'*60}")

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Finance CS Data Pipeline")
    parser.add_argument("--file", type=str, help="Process a single PDF (relative to data/raw/)")
    parser.add_argument("--all",  action="store_true", help="Process all PDFs in data/raw/")
    args = parser.parse_args()

    if args.file:
        pdf_path = RAW_DIR / args.file
        if not pdf_path.exists():
            print(f"❌ File not found: {pdf_path}")
            return
        process_pdf(pdf_path)

    elif args.all:
        pdfs = list(RAW_DIR.glob("*.pdf"))
        if not pdfs:
            print(f"❌ No PDFs found in {RAW_DIR}")
            return
        print(f"Found {len(pdfs)} PDF(s) to process")
        for pdf in pdfs:
            process_pdf(pdf)

    else:
        # Default: process all PDFs found
        pdfs = list(RAW_DIR.glob("*.pdf"))
        if not pdfs:
            print(f"❌ No PDFs in {RAW_DIR}. Usage: python prepare_data.py --all")
            return
        print(f"Found {len(pdfs)} PDF(s) — processing all...")
        for pdf in pdfs:
            process_pdf(pdf)


if __name__ == "__main__":
    main()