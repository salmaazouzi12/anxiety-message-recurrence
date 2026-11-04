
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from scipy.sparse import vstack
from sklearn.feature_extraction.text import HashingVectorizer

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "anxiety_lineages"

SOURCE = BASE / "minhash"
OUT = BASE / "lexical_scoring"
OUT.mkdir(parents=True, exist_ok=True)

MAPPING = SOURCE / "text_id_mapping.parquet"

CORE_THRESHOLD = 0.90
FLOOR_THRESHOLD = 0.86

VECTOR_BATCH = 2000
PAIR_BATCH = 10000

# ----------------------------------------------------
# 1. Load unique text mapping
# ----------------------------------------------------

mapping = pd.read_parquet(MAPPING)

assert mapping["text_id"].is_unique

mapping = mapping.sort_values("text_id").reset_index(drop=True)

assert np.array_equal(
    mapping["text_id"].to_numpy(),
    np.arange(len(mapping))
)

texts = mapping["normalized_text"].fillna("").astype(str).tolist()

print(f"Loaded {len(texts):,} unique texts", flush=True)

# ----------------------------------------------------
# 2. Build character 3-5 gram representations
# ----------------------------------------------------

# Same type of hashing representation as the paper.
# L2 normalization permits cosine similarity via
# elementwise multiplication and summation.

vectorizer = HashingVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    n_features=2**18,
    alternate_sign=False,
    norm="l2",
    dtype=np.float32
)

print("\nBuilding text vectors...", flush=True)

blocks = []

for start in range(0, len(texts), VECTOR_BATCH):

    end = min(start + VECTOR_BATCH, len(texts))

    block = vectorizer.transform(texts[start:end])

    blocks.append(block)

    if end % 20000 < VECTOR_BATCH:
        print(
            f"Vectorized {end:,}/{len(texts):,}",
            flush=True
        )

X = vstack(blocks, format="csr")
del blocks

print(
    f"Vector matrix: {X.shape}, "
    f"nonzero values: {X.nnz:,}",
    flush=True
)

# ----------------------------------------------------
# 3. Score candidate pairs
# ----------------------------------------------------

files = sorted(
    SOURCE.glob("candidate_pairs_*.parquet")
)

assert files, "No candidate pair files found."

print(f"\nCandidate files: {len(files)}", flush=True)

total_pairs = 0
core_pairs = 0
weak_pairs = 0
below_floor = 0

start_time = time.time()

for file_idx, path in enumerate(files, start=1):

    pairs = pd.read_parquet(path)

    required = {"text_id_1", "text_id_2"}
    assert required.issubset(pairs.columns)

    a = pairs["text_id_1"].to_numpy(dtype=np.int64)
    b = pairs["text_id_2"].to_numpy(dtype=np.int64)

    assert (a >= 0).all() and (a < len(texts)).all()
    assert (b >= 0).all() and (b < len(texts)).all()

    accepted_parts = []

    for start in range(0, len(pairs), PAIR_BATCH):

        end = min(start + PAIR_BATCH, len(pairs))

        aa = a[start:end]
        bb = b[start:end]

        # Both matrices are L2-normalized.
        # Elementwise row multiplication sums to
        # cosine similarity for each corresponding pair.

        similarity = np.asarray(
            X[aa].multiply(X[bb]).sum(axis=1)
        ).ravel()

        strong = similarity >= CORE_THRESHOLD

        weak = (
            (similarity >= FLOOR_THRESHOLD)
            & ~strong
        )

        accepted = strong | weak

        total_pairs += len(similarity)

        core_pairs += int(strong.sum())
        weak_pairs += int(weak.sum())

        below_floor += int((~accepted).sum())

        if accepted.any():
            accepted_parts.append(
                pd.DataFrame({
                    "text_id_1": aa[accepted],
                    "text_id_2": bb[accepted],
                    "lexical_cosine": similarity[
                        accepted
                    ].astype(np.float32),
                    "edge_type": np.where(
                        strong[accepted],
                        "core",
                        "weak"
                    )
                })
            )

    if accepted_parts:

        accepted_df = pd.concat(
            accepted_parts,
            ignore_index=True
        )

        accepted_df.to_parquet(
            OUT / f"scored_pairs_{file_idx:04d}.parquet",
            index=False
        )

    elapsed = (time.time() - start_time) / 60

    print(
        f"File {file_idx}/{len(files)} | "
        f"Pairs scored: {total_pairs:,} | "
        f"Core: {core_pairs:,} | "
        f"Weak: {weak_pairs:,} | "
        f"Elapsed: {elapsed:.1f} min",
        flush=True
    )

# ----------------------------------------------------
# 4. Save summary
# ----------------------------------------------------

summary = {
    "unique_texts": len(texts),
    "input_candidate_files": len(files),
    "candidate_pairs_scored": total_pairs,
    "core_threshold": CORE_THRESHOLD,
    "lexical_floor": FLOOR_THRESHOLD,
    "core_pairs": core_pairs,
    "weak_pairs": weak_pairs,
    "below_floor_pairs": below_floor,
    "retained_pairs": core_pairs + weak_pairs,
    "retention_rate": (
        (core_pairs + weak_pairs) / total_pairs
        if total_pairs else 0
    ),
    "elapsed_minutes": round(
        (time.time() - start_time) / 60, 2
    )
}

with open(
    OUT / "lexical_scoring_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 65)
print("LEXICAL SCORING COMPLETE")
print("=" * 65)

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nResults:", OUT)
