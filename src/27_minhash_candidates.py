
from pathlib import Path
import re
import json
import time

import numpy as np
import pandas as pd

from datasketch import MinHash, MinHashLSH

ROOT = Path(__file__).resolve().parents[1]

BASE = ROOT / "outputs" / "anxiety_lineages"

SOURCE = BASE / "unique_text_families.parquet"

OUT = BASE / "minhash"
OUT.mkdir(parents=True, exist_ok=True)

NUM_PERM = 64
LSH_THRESHOLD = 0.45
SHINGLE_SIZE = 5
FLUSH_EVERY = 50000

# Load unique normalized texts
df = pd.read_parquet(SOURCE)

assert "exact_family_id" in df.columns
assert "normalized_text" in df.columns
assert df["exact_family_id"].is_unique

texts = df["normalized_text"].fillna("").astype(str).tolist()

print("Unique texts:", len(texts), flush=True)

# Slightly more aggressive lexical representation
def normalize_lexical(text):
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = text.replace("ـ", "")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())

def shingles(text, n=5):
    text = normalize_lexical(text)

    if not text:
        return {b"<EMPTY>"}

    if len(text) < n:
        return {text.encode("utf-8")}

    return {
        text[i:i+n].encode("utf-8")
        for i in range(len(text) - n + 1)
    }

def build_minhash(text):
    mh = MinHash(num_perm=NUM_PERM, seed=42)

    for gram in shingles(text, SHINGLE_SIZE):
        mh.update(gram)

    return mh

# Index contains earlier texts only.
# Query before insertion ensures each discovered pair
# is emitted once with earlier_id < later_id.
lsh = MinHashLSH(
    threshold=LSH_THRESHOLD,
    num_perm=NUM_PERM
)

BUFFER = []
PART = 0
PAIR_COUNT = 0

def flush_pairs():
    global BUFFER, PART

    if not BUFFER:
        return

    output = OUT / f"candidate_pairs_{PART:04d}.parquet"

    pd.DataFrame(
        BUFFER,
        columns=["text_id_1", "text_id_2"]
    ).to_parquet(output, index=False)

    PART += 1
    BUFFER = []

start_time = time.time()

for i, text in enumerate(texts):

    mh = build_minhash(text)

    neighbors = lsh.query(mh)

    for neighbor in neighbors:
        j = int(neighbor)

        if j == i:
            continue

        BUFFER.append((j, i))
        PAIR_COUNT += 1

    lsh.insert(str(i), mh)

    if len(BUFFER) >= FLUSH_EVERY:
        flush_pairs()

    if (i + 1) % 10000 == 0:
        elapsed = (time.time() - start_time) / 60

        print(
            f"Processed {i+1:,}/{len(texts):,} | "
            f"Candidate pairs: {PAIR_COUNT:,} | "
            f"Elapsed: {elapsed:.1f} min",
            flush=True
        )

flush_pairs()

# Preserve the row-to-text ID mapping
df[[
    "exact_family_id",
    "normalized_text",
    "post_count"
]].reset_index(names="text_id").to_parquet(
    OUT / "text_id_mapping.parquet",
    index=False
)

summary = {
    "unique_texts": len(texts),
    "minhash_permutations": NUM_PERM,
    "lsh_threshold": LSH_THRESHOLD,
    "shingle_size": SHINGLE_SIZE,
    "lexical_candidate_pairs": PAIR_COUNT,
    "candidate_files": PART,
    "elapsed_minutes": round(
        (time.time() - start_time) / 60, 2
    )
}

with open(
    OUT / "minhash_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("MINHASH CANDIDATE RETRIEVAL COMPLETE")
print("=" * 60)

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nOutputs:", OUT)
