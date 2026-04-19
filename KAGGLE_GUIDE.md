# Kaggle Docling Pipeline Guide

## Overview

Two ready-to-run scripts for processing your batches with Docling on Kaggle GPU:

1. **`kaggle_docling_batches.py`** — Basic extraction
   - Fast, lightweight docling wrapper
   - Auto-detects GPU (CUDA)
   - Progress resumption built-in
   - Best for: Quick extraction & caching

2. **`kaggle_advanced_pipeline.py`** — Full pipeline
   - Extraction + table processing + chunking
   - Integrates taxonomy & inventory metadata
   - Produces production-ready chunks (JSONL)
   - Best for: Building RAG datasets

---

## Quick Start

### Option 1: Basic Extraction (Recommended for first run)

#### Setup on Kaggle:
1. Create a new Notebook
2. In **+ Add data**, upload your `batches/` folder
3. In **+ Add data**, create a folder with `kaggle_docling_batches.py`
4. In the first notebook cell:

```python
!pip install docling -q
!pip install psutil -q
```

5. Second cell:

```python
%cd /kaggle/working
!python /path/to/kaggle_docling_batches.py \
    --batch-dir /kaggle/input/batches \
    --output-dir /kaggle/working/extraction
```

#### What's created:
- `extraction/cache/` — JSON files (one per PDF)
- `extraction/progress.json` — Resume checkpoint
- `extraction/extraction.log` — Full run log

---

### Option 2: Advanced Pipeline (Production RAG)

#### Setup:
1. Upload: `batches/` folder
2. Upload: `central_inventory.csv`
3. Upload: `Common/doc_type_taxonomy.py`
4. Upload: `kaggle_advanced_pipeline.py`

#### First cell:
```python
!pip install docling pandas -q
!pip install psutil -q
```

#### Second cell:
```python
%cd /kaggle/working

!python /path/to/kaggle_advanced_pipeline.py \
    --batch-dir /kaggle/input/batches \
    --output-dir /kaggle/working/output \
    --config-csv /kaggle/input/central_inventory.csv \
    --taxonomy-py /kaggle/input/doc_type_taxonomy.py
```

#### What's created:
- `output/cache/` — Raw docling extractions
- `output/chunks/` — JSONL chunks (one per document)
- `output/metadata.csv` — Flattened metadata
- `output/progress.json` — Resume checkpoint

---

## Resuming Interrupted Runs

Both scripts support resumption:

```python
# On timeout, just re-run the same command
# It will skip already-processed PDFs automatically
!python kaggle_advanced_pipeline.py ...
```

Progress is tracked in `progress.json` — delete it to restart from scratch.

---

## GPU Notes

### Device Detection
- Both scripts auto-detect CUDA
- Falls back to CPU if unavailable

### What's optimized for GPU:
- Docling table extraction (TableFormer)
- Memory cleanup (`torch.cuda.empty_cache()`) after each document

### Typical performance (Kaggle T4):
- 50-100 PDFs per hour (extraction only)
- ~2-5 tables per document (large variance by doc)

---

## Docling Output Structure

### Extracted Blocks (TextBlock):
```json
{
  "text": "..."            // actual paragraph text
  "block_type": "paragraph",  // heading | paragraph | list_item
  "heading_path": [...],   // hierarchical context
  "page_number": 3,
  "block_index": 42
}
```

### Extracted Tables (TableChunk):
```json
{
  "table_id": "bps_2022_table_001",
  "caption": "Revenue Performance",
  "markdown": "| ... |",   // full markdown table
  "rows": 15,
  "cols": 6,
  "data_type": "actual",   // actual | projection | target | mixed
  "heading_path": [...]
}
```

### Chunks (for embedding):
```json
{
  "chunk_id": "bps_2022_chunk_0001",
  "source_doc": "bps_2022",
  "content": "...",        // full narrative or table markdown
  "chunk_type": "narrative",  // narrative | table | heading
  "token_estimate": 250,
  "metadata": {
    "agent": "finance",
    "document_type": "budget_policy_statement",
    "fiscal_year": "2021_22"
  }
}
```

---

## Table Data Type Detection

Auto-detects from column headers:

| Headers contain | Data Type |
|---|---|
| "Actual" / "Outturn" | `actual` |
| "Projected" / "Estimate" | `projection` |
| Both | `mixed` |
| Neither | `mixed` |

---

## Memory & Performance Tips

### For large batches (500+ PDFs):

1. **Monitor memory:**
```python
import psutil
print(psutil.virtual_memory())
```

2. **If running out of memory:**
   - Add `--force` flag to skip re-extraction
   - Restart notebook between batches

3. **Check remaining quota:**
```bash
!echo "RAM available:" && free -h
```

---

## Troubleshooting

### "docling not installed"
```python
!pip install docling --upgrade
```

### "Batch directory not found"
- Verify dataset is added to notebook
- Check exact path: `/kaggle/input/your-dataset-name/`

### "Memory error during extraction"
- Restart notebook kernel
- Process one batch at a time
- Increase GPU memory threshold

### Incomplete PDF extraction
- Some PDFs might fail silently (logged in progress.json)
- Check `extraction.log` for details
- Scanned PDFs fallback to CPU OCR (slower)

---

## Comparing with Your Local Finance Pipeline

| Feature | kaggle_docling_batches.py | kaggle_advanced_pipeline.py | Finance/ (local) |
|---|---|---|---|
| Extraction | ✓ | ✓ | ✓ |
| Table detection | ✓ | ✓ | ✓ |
| Chunking | ✗ | ✓ | ✓ |
| Taxonomy tagging | ✗ | ✓ (partial) | ✓ |
| Config.yaml support | ✗ | ✓ | ✓ |
| Metadata enrichment | ✗ | ✓ | ✓ |
| GPU-optimized | ✓ | ✓ | ✓ |

---

## Custom Processing

### Save chunks for embedding:
```python
import json

chunk_count = 0
with open('/kaggle/working/output/chunks/all_chunks.jsonl', 'w') as out:
    for chunk_file in Path('/kaggle/working/output/chunks').glob('*.jsonl'):
        with open(chunk_file) as f:
            for line in f:
                out.write(line)
                chunk_count += 1

print(f"Exported {chunk_count} chunks")
```

### Filter by metadata:
```python
metadata = pd.read_csv('/kaggle/working/output/metadata.csv')

# Finance-only documents
finance_docs = metadata[metadata['agent'] == 'finance']

# High-priority documents
high_priority = metadata[metadata['priority'] == 'high']
```

### Check extraction stats:
```python
metadata = pd.read_csv('/kaggle/working/output/metadata.csv')
print(f"Total documents: {len(metadata)}")
print(f"Total blocks: {metadata['blocks'].sum()}")
print(f"Total tables: {metadata['tables'].sum()}")
print(f"Total chunks: {metadata['chunks'].sum()}")
print(f"\nAverage blocks per doc: {metadata['blocks'].mean():.1f}")
print(f"Average tables per doc: {metadata['tables'].mean():.1f}")
```

---

## Next Steps

After extracting on Kaggle, you can:

1. **Embed using OpenAI/Cohere API:**
   - Read chunks from JSONL
   - Send to embedding API
   - Save embeddings as vectors

2. **Build vector store (Chroma/Pinecone):**
   - Use chunks + embeddings
   - Create searchable index

3. **Evaluate quality:**
   - Check token distribution
   - Verify chunk completeness
   - Test retrieval

4. **Download results:**
   - Export from `/kaggle/working/output/`
   - Use Kaggle API: `kaggle kernels pull -p /local/path`

---

## Questions?

- Check logs in `extraction.log`
- Review `progress.json` for per-document status
- Re-run with `--dry-run` to test without extraction

Good luck! 🚀
