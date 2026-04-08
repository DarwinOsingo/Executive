import pandas as pd
from pathlib import Path

# Paths – adjust if needed
CSV_PATH = Path("~/PRES/Executive/central_inventory.csv").expanduser()
CACHE_DIR = Path("~/PRES/Executive/Finance/data/cache").expanduser()   # or wherever your cache JSONs live

# Load CSV
df = pd.read_csv(CSV_PATH)

# Get set of existing cache stems (without .json extension)
existing_stems = {p.stem for p in CACHE_DIR.glob("*.json")} if CACHE_DIR.exists() else set()

# Normalize doc_id to match cache stem format (replace dots with underscores, remove agent prefix)
def doc_id_to_cache_stem(doc_id: str) -> str:
    # Remove agent prefix (e.g., "finance_")
    if "_" in doc_id:
        base = doc_id.split("_", 1)[1]
    else:
        base = doc_id
    return base.replace(".", "_")

df["cache_stem"] = df["doc_id"].apply(doc_id_to_cache_stem)
df["cached"] = df["cache_stem"].isin(existing_stems)

# Report by agent
print("=== CACHE INVENTORY BY AGENT ===")
for agent in sorted(df["agent"].unique()):
    agent_df = df[df["agent"] == agent]
    cached = agent_df["cached"].sum()
    total = len(agent_df)
    print(f"{agent:<15} {cached:>3} / {total:>3} cached")

total_cached = df["cached"].sum()
total_docs = len(df)
print(f"\nTotal: {total_cached} / {total_docs} cache files present")

# Write missing PDFs to file
missing = df[~df["cached"]]
missing_paths = missing["filepath"].tolist()
with open("missing_pdfs.txt", "w") as f:
    f.write("\n".join(missing_paths))
print(f"\nMissing PDFs written to missing_pdfs.txt ({len(missing)} files)")