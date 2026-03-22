"""
upsert.py
─────────
Embeds chunked Finance CS documents and upserts into Qdrant.

Each Qdrant point contains:
  - Dense vector  : Voyage AI voyage-3 (1024 dims)
  - Sparse vector : BM25 via fastembed (for hybrid search)
  - Payload       : all chunk metadata fields

Features:
  - Checkpointing: saves progress so interrupted runs resume where they left off
  - Idempotent: skipping already-upserted chunk_ids
  - Batched: 32 chunks per Voyage API call, 64 points per Qdrant upsert

Dependencies:
    pip install voyageai qdrant-client fastembed

Usage:
    python upsert.py --chunks data/chunks/ --collection kenya_cabinet
"""

import os
import json
import time
import argparse
from pathlib import Path

import voyageai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
)
from fastembed import SparseTextEmbedding

# ── Config ────────────────────────────────────────────────────────────────────

COLLECTION_NAME = "kenya_executive_roundtable"  
VOYAGE_MODEL     = "voyage-3"
DENSE_DIMS       = 1024
VOYAGE_BATCH     = 32    # chunks per Voyage API call
UPSERT_BATCH     = 64    # points per Qdrant upsert call
VOYAGE_SLEEP     = 0.5   # small buffer between calls
CHECKPOINT_FILE  = "upsert_checkpoint.json"
QDRANT_URL       = "http://localhost:6333"

# ── Clients ───────────────────────────────────────────────────────────────────

voyage  = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
qdrant  = QdrantClient(url=QDRANT_URL)
bm25    = SparseTextEmbedding(model_name="Qdrant/bm25")

# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint() -> set:
    """Load set of already-upserted chunk_ids."""
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(done: set):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(list(done), f)

# ── Collection setup ──────────────────────────────────────────────────────────

def ensure_collection():
    """Create collection if it doesn't exist. Safe to call multiple times."""
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"✓ Collection '{COLLECTION_NAME}' already exists")
        return

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=DENSE_DIMS,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "bm25": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )
    print(f"✓ Created collection '{COLLECTION_NAME}'")

# ── Payload builder ───────────────────────────────────────────────────────────

def build_payload(chunk: dict) -> dict:
    return dict(chunk)  # keep everything including text — retriever needs it for context injection

# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_dense(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via Voyage AI. Retries on rate limit."""
    for attempt in range(3):
        try:
            result = voyage.embed(texts, model=VOYAGE_MODEL, input_type="document")
            return result.embeddings
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  Voyage error: {e} — retrying in 10s")
            time.sleep(10)


def embed_sparse(texts: list[str]) -> list[SparseVector]:
    """Generate BM25 sparse vectors via fastembed."""
    results = []
    for embedding in bm25.embed(texts):
        indices = embedding.indices.tolist()
        values  = embedding.values.tolist()
        results.append(SparseVector(indices=indices, values=values))
    return results

# ── Main pipeline ─────────────────────────────────────────────────────────────

def load_all_chunks(chunks_dir: Path) -> list[dict]:
    """Load all chunks from all .jsonl files in directory."""
    chunks = []
    for jsonl_file in sorted(chunks_dir.glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    return chunks


def upsert_batch(points: list[PointStruct]):
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def run(chunks_dir: Path):
    ensure_collection()

    print(f"\nLoading chunks from {chunks_dir}...")
    all_chunks = load_all_chunks(chunks_dir)
    print(f"  Total chunks: {len(all_chunks)}")

    done       = load_checkpoint()
    remaining  = [c for c in all_chunks if c["chunk_id"] not in done]
    print(f"  Already upserted: {len(done)}")
    print(f"  Remaining: {len(remaining)}\n")

    if not remaining:
        print("Nothing to do — all chunks already upserted.")
        return

    points_buffer = []
    total_upserted = 0

    for i in range(0, len(remaining), VOYAGE_BATCH):
        batch = remaining[i : i + VOYAGE_BATCH]
        texts = [c["text"] for c in batch]

        # Embed
        dense_vecs  = embed_dense(texts)
        time.sleep(VOYAGE_SLEEP)  # respect 3 RPM free tier limit
        sparse_vecs = embed_sparse(texts)

        # Build Qdrant points
        for chunk, dense, sparse in zip(batch, dense_vecs, sparse_vecs):
            points_buffer.append(
                PointStruct(
                    id=chunk["chunk_id"],
                    vector={
                        "dense": dense,
                        "bm25":  sparse,
                    },
                    payload=build_payload(chunk),
                )
            )

        # Upsert when buffer is full
        if len(points_buffer) >= UPSERT_BATCH:
            upsert_batch(points_buffer)
            for p in points_buffer:
                done.add(p.id)
            total_upserted += len(points_buffer)
            save_checkpoint(done)
            points_buffer = []
            print(f"  ✓ {total_upserted}/{len(remaining)} upserted")

    # Flush remaining buffer
    if points_buffer:
        upsert_batch(points_buffer)
        for p in points_buffer:
            done.add(p.id)
        total_upserted += len(points_buffer)
        save_checkpoint(done)
        print(f"  ✓ {total_upserted}/{len(remaining)} upserted")

    print(f"\n✓ Done. {total_upserted} new points in '{COLLECTION_NAME}'.")


def main():
    global COLLECTION_NAME
    parser = argparse.ArgumentParser(description="Embed and upsert Finance chunks into Qdrant.")
    parser.add_argument("--chunks",     default="data/chunks/", help="Directory of .jsonl chunk files")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Qdrant collection name")
    args = parser.parse_args()

    COLLECTION_NAME = args.collection
    run(Path(args.chunks))


if __name__ == "__main__":
    main()