import pandas as pd
from pathlib import Path
import re

# Load full inventory
df = pd.read_csv("central_inventory.csv")

# Normalize a doc_id to match cache filename stem format
def normalize_for_cache(doc_id: str) -> str:
    # Remove leading agent prefix (e.g., "finance_")
    if "_" in doc_id:
        base = doc_id.split("_", 1)[1]   # everything after first underscore
    else:
        base = doc_id
    # Replace dots with underscores
    return base.replace(".", "_")

# Build set of processed doc_ids from cache JSON files
cache_dir = Path("Finance/data/cache")
processed_stems = {p.stem for p in cache_dir.glob("*.json")}

# Filter: keep rows where normalized doc_id is NOT in processed_stems
df["cache_stem"] = df["doc_id"].apply(normalize_for_cache)
df_remaining = df[~df["cache_stem"].isin(processed_stems)]

# Report by agent
print("=== REMAINING DOCUMENTS BY AGENT ===")
for agent in sorted(df_remaining["agent"].unique()):
    count = len(df_remaining[df_remaining["agent"] == agent])
    print(f"{agent:<15} {count:>3} documents")

print(f"\nTotal remaining: {len(df_remaining)}")

# Save list of remaining PDF paths
remaining_paths = df_remaining["filepath"].tolist()
with open("remaining_pdfs.txt", "w") as f:
    f.write("\n".join(remaining_paths))
print("\nSaved remaining file paths to 'remaining_pdfs.txt'")