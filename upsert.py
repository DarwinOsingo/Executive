"""
upsert.py
─────────
Embeds all chunked documents and upserts into Qdrant.
Handles the full 7-agent Kenya AI Executive Roundtable corpus.

Each Qdrant point contains:
  - Dense vector  : Voyage AI voyage-3 (1024 dims)
  - Sparse vector : BM25 via fastembed (for hybrid search)
  - Payload       : all chunk metadata fields (including agent_access array,
                    topics, primary_agents, fiscal_year, domain, priority,
                    rag_weight, and all other fields from chunk_documents.py)

Features:
  - Checkpointing : saves progress after every Qdrant batch — safe to Ctrl+C
                    and resume. Checkpoint lives next to the chunks directory.
  - Idempotent    : skips chunk_ids already in checkpoint
  - Batched       : configurable Voyage batch size + Qdrant upsert size
  - Retry         : exponential backoff on Voyage API errors

Dependencies:
    pip install voyageai qdrant-client fastembed

Usage:
    # Full corpus
    python upsert.py --chunks data/chunks/ --collection kenya_executive_roundtable

    # Re-upsert everything (ignore checkpoint)
    python upsert.py --chunks data/chunks/ --collection kenya_executive_roundtable --force

    # Dry-run: count chunks, validate payloads, do NOT embed or upsert
    python upsert.py --chunks data/chunks/ --dry-run

Environment:
    VOYAGE_API_KEY  — required
    QDRANT_URL      — optional, defaults to http://localhost:6333
"""

import argparse
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env from same directory as this script
load_dotenv(Path(__file__).parent / ".env")

import voyageai
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

# ── Config ────────────────────────────────────────────────────────────────────

COLLECTION_NAME  = "kenya_executive_roundtable"
VOYAGE_MODEL     = "voyage-3"
DENSE_DIMS       = 1024

# Batch sizes — tune based on Voyage tier
# Paid tier: VOYAGE_BATCH=32, VOYAGE_SLEEP=0.1 is safe
# Free tier: VOYAGE_BATCH=8,  VOYAGE_SLEEP=20 (3 RPM limit)
VOYAGE_BATCH     = 32
UPSERT_BATCH     = 64    # points per Qdrant upsert call
VOYAGE_SLEEP     = 0.1   # seconds between Voyage API calls (paid tier)

QDRANT_URL       = os.environ.get("QDRANT_URL", "http://localhost:6333")


# ── Clients (initialised in main after arg parsing) ───────────────────────────

_voyage  = None
_qdrant  = None
_bm25    = None


def get_voyage() -> voyageai.Client:
    global _voyage
    if _voyage is None:
        api_key = os.environ.get("VOYAGE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "VOYAGE_API_KEY environment variable not set.\n"
                "Run: export VOYAGE_API_KEY='pa-...'"
            )
        _voyage = voyageai.Client(api_key=api_key)
    return _voyage


def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL)
    return _qdrant


def get_bm25() -> SparseTextEmbedding:
    global _bm25
    if _bm25 is None:
        print("  Loading BM25 model (first run only)...")
        _bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _bm25


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def checkpoint_path(chunks_dir: Path) -> Path:
    """Checkpoint lives next to the chunks directory, not in CWD."""
    return chunks_dir.parent / "upsert_checkpoint.json"


def load_checkpoint(chunks_dir: Path) -> set:
    path = checkpoint_path(chunks_dir)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_checkpoint(chunks_dir: Path, done: set):
    path = checkpoint_path(chunks_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(done), f)


# ── Collection setup ──────────────────────────────────────────────────────────

def ensure_collection(collection_name: str):
    """Create collection if it does not exist. Safe to call multiple times."""
    existing = [c.name for c in get_qdrant().get_collections().collections]
    if collection_name in existing:
        info = get_qdrant().get_collection(collection_name)
        print(
            f"  Collection '{collection_name}' exists — "
            f"{info.points_count:,} points"
        )
        return

    get_qdrant().create_collection(
        collection_name=collection_name,
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
    print(f"  Created collection '{collection_name}'")


# ── Chunk loading ─────────────────────────────────────────────────────────────

def load_all_chunks(chunks_dir: Path) -> list[dict]:
    """Load all chunks from all .jsonl files in the directory."""
    chunks = []
    files  = sorted(chunks_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {chunks_dir}")

    for jsonl_file in files:
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        chunks.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"  WARNING: bad JSON in {jsonl_file.name}: {e}")
    return chunks


def validate_chunks(chunks: list[dict]) -> list[str]:
    """Return list of validation warnings. Does not raise."""
    warnings = []
    seen_ids: set = set()

    for i, chunk in enumerate(chunks):
        cid = chunk.get("chunk_id")
        if not cid:
            warnings.append(f"Chunk #{i}: missing chunk_id")
            continue
        if cid in seen_ids:
            warnings.append(f"Chunk #{i}: duplicate chunk_id {cid}")
        seen_ids.add(cid)

        if not chunk.get("text", "").strip():
            warnings.append(f"Chunk {cid}: empty text — will embed empty string")

        if not chunk.get("agent_access"):
            warnings.append(f"Chunk {cid}: agent_access is empty")

    return warnings


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_dense(texts: list[str]) -> list[list[float]]:
    """Embed a batch via Voyage AI with exponential backoff retry."""
    for attempt in range(4):
        try:
            result = get_voyage().embed(
                texts, model=VOYAGE_MODEL, input_type="document"
            )
            return result.embeddings
        except Exception as e:
            if attempt == 3:
                raise
            wait = 10 * (2 ** attempt)   # 10s, 20s, 40s
            print(f"  Voyage error (attempt {attempt + 1}/4): {e}")
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)


def embed_sparse(texts: list[str]) -> list[SparseVector]:
    """Generate BM25 sparse vectors via fastembed (local, no API call)."""
    results = []
    for embedding in get_bm25().embed(texts):
        results.append(
            SparseVector(
                indices=embedding.indices.tolist(),
                values=embedding.values.tolist(),
            )
        )
    return results


# ── Upsert ────────────────────────────────────────────────────────────────────

def upsert_points(points: list[PointStruct], collection_name: str):
    get_qdrant().upsert(collection_name=collection_name, points=points)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(
    chunks_dir:      Path,
    collection_name: str,
    force:           bool = False,
    dry_run:         bool = False,
):
    print(f"\nChunks dir  : {chunks_dir}")
    print(f"Collection  : {collection_name}")
    print(f"Qdrant URL  : {QDRANT_URL}")
    print(f"Voyage model: {VOYAGE_MODEL}  batch={VOYAGE_BATCH}  sleep={VOYAGE_SLEEP}s")
    print()

    # ── Load chunks ───────────────────────────────────────────────────────────
    print("Loading chunks...")
    all_chunks = load_all_chunks(chunks_dir)
    print(f"  Total chunks: {len(all_chunks):,}")

    # ── Validate ──────────────────────────────────────────────────────────────
    warnings = validate_chunks(all_chunks)
    if warnings:
        print(f"\n  {len(warnings)} validation warnings:")
        for w in warnings[:20]:
            print(f"    {w}")
        if len(warnings) > 20:
            print(f"    ... and {len(warnings) - 20} more")
        print()

    # ── Dry run ───────────────────────────────────────────────────────────────
    if dry_run:
        # Agent breakdown
        from collections import Counter
        agent_counts: Counter = Counter()
        for c in all_chunks:
            for agent in (c.get("agent_access") or []):
                agent_counts[agent] += 1
        print("Agent access breakdown (chunks per agent):")
        for agent, count in sorted(agent_counts.items()):
            print(f"  {agent:<20}: {count:>7,} chunks")
        print(f"\nDry run complete — no data written.")
        return

    # ── Collection ────────────────────────────────────────────────────────────
    ensure_collection(collection_name)

    # ── Checkpoint ───────────────────────────────────────────────────────────
    if force:
        done = set()
        print("  --force: ignoring checkpoint, upserting everything")
    else:
        done = load_checkpoint(chunks_dir)
        print(f"  Already upserted: {len(done):,}")

    remaining = [c for c in all_chunks if c["chunk_id"] not in done]
    print(f"  Remaining       : {len(remaining):,}")

    if not remaining:
        print("\nNothing to do — all chunks already upserted.")
        return

    # ── Estimate time ─────────────────────────────────────────────────────────
    n_batches   = (len(remaining) + VOYAGE_BATCH - 1) // VOYAGE_BATCH
    est_seconds = n_batches * (VOYAGE_SLEEP + 0.5)   # 0.5s estimated Voyage latency
    est_minutes = est_seconds / 60
    print(f"\n  Batches  : {n_batches:,} × {VOYAGE_BATCH} chunks")
    print(f"  Estimated: {est_minutes:.0f} min (may vary with Voyage latency)")
    print()

    # ── Embed + upsert ────────────────────────────────────────────────────────
    points_buffer:  list[PointStruct] = []
    total_upserted: int = 0
    start_time = time.time()

    for i in range(0, len(remaining), VOYAGE_BATCH):
        batch = remaining[i : i + VOYAGE_BATCH]
        texts = [c["text"] for c in batch]

        # Dense embedding via Voyage AI
        dense_vecs = embed_dense(texts)
        time.sleep(VOYAGE_SLEEP)

        # Sparse embedding via local BM25
        sparse_vecs = embed_sparse(texts)

        # Build Qdrant points
        for chunk, dense, sparse in zip(batch, dense_vecs, sparse_vecs):
            points_buffer.append(
                PointStruct(
                    id=chunk["chunk_id"],
                    vector={"dense": dense, "bm25": sparse},
                    payload=dict(chunk),   # full chunk as payload including text
                )
            )

        # Flush when buffer reaches UPSERT_BATCH
        if len(points_buffer) >= UPSERT_BATCH:
            upsert_points(points_buffer, collection_name)
            for p in points_buffer:
                done.add(p.id)
            total_upserted += len(points_buffer)
            save_checkpoint(chunks_dir, done)
            points_buffer = []

            # Progress
            pct      = total_upserted / len(remaining) * 100
            elapsed  = time.time() - start_time
            rate     = total_upserted / elapsed if elapsed > 0 else 0
            eta_secs = (len(remaining) - total_upserted) / rate if rate > 0 else 0
            print(
                f"  {total_upserted:>7,}/{len(remaining):,}  "
                f"({pct:4.1f}%)  "
                f"{rate:4.0f} chunks/min  "
                f"ETA {eta_secs / 60:.0f}min"
            )

    # ── Final flush ───────────────────────────────────────────────────────────
    if points_buffer:
        upsert_points(points_buffer, collection_name)
        for p in points_buffer:
            done.add(p.id)
        total_upserted += len(points_buffer)
        save_checkpoint(chunks_dir, done)

    elapsed = time.time() - start_time
    print(
        f"\n✓ Done. {total_upserted:,} new points upserted in "
        f"{elapsed / 60:.1f} min."
    )

    # ── Final collection stats ────────────────────────────────────────────────
    info = get_qdrant().get_collection(collection_name)
    print(f"  Collection '{collection_name}': {info.points_count:,} total points")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global COLLECTION_NAME

    parser = argparse.ArgumentParser(
        description="Embed and upsert all agent chunks into Qdrant."
    )
    parser.add_argument(
        "--chunks",
        default="data/chunks/",
        help="Directory of .jsonl chunk files (default: data/chunks/)",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help=f"Qdrant collection name (default: {COLLECTION_NAME})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore checkpoint and re-upsert everything",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report only — no embedding or upserting",
    )
    args = parser.parse_args()

    COLLECTION_NAME = args.collection

    run(
        chunks_dir      = Path(args.chunks),
        collection_name = args.collection,
        force           = args.force,
        dry_run         = args.dry_run,
    )


if __name__ == "__main__":
    main()