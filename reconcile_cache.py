import pandas as pd
from pathlib import Path

CSV_PATH = Path("~/PRES/Executive/central_inventory.csv").expanduser()
CACHE_DIR = Path("~/PRES/Executive/data/cache").expanduser()

df = pd.read_csv(CSV_PATH)
cache_files = list(CACHE_DIR.glob("*.json"))
cache_stems = {p.stem for p in cache_files}

def doc_id_to_possible_stems(doc_id: str) -> list[str]:
    """Generate all plausible cache stems for a doc_id."""
    # Remove agent prefix
    if "_" in doc_id:
        base = doc_id.split("_", 1)[1]
    else:
        base = doc_id
    base = base.replace(".", "_")
    stems = [base]
    # Also try with agent prefix (some older caches kept it)
    stems.append(doc_id.replace(".", "_"))
    return stems

matched_docs = 0
unmatched_cache = []
for cache_file in cache_files:
    stem = cache_file.stem
    found = False
    for _, row in df.iterrows():
        if stem in doc_id_to_possible_stems(row["doc_id"]):
            found = True
            break
    if not found:
        unmatched_cache.append(cache_file.name)

print(f"Total cache files: {len(cache_files)}")
print(f"Matched to CSV: {len(cache_files) - len(unmatched_cache)}")
print(f"Unmatched cache files: {len(unmatched_cache)}")
if unmatched_cache:
    print("\nUnmatched files (first 20):")
    for f in unmatched_cache[:20]:
        print(f"  {f}")

# Missing PDFs
missing = []
for _, row in df.iterrows():
    stems = doc_id_to_possible_stems(row["doc_id"])
    if not any(s in cache_stems for s in stems):
        missing.append(row["filepath"])

print(f"\nMissing PDFs still to extract: {len(missing)}")
with open("missing_pdfs.txt", "w") as f:
    f.write("\n".join(missing))