
# ============================================================
# 33_merge_score_candidates.py
#
# Merge E5 and MinHash candidate retrieval channels.
#
# 1. Globally deduplicate E5 pairs.
# 2. Exclude pairs already retrieved through MinHash.
# 3. Score ONLY novel E5 pairs with character 3-5 gram cosine.
# 4. Retain pairs meeting the paper's lexical floor of 0.86.
# 5. Produce combined retrieval statistics.
#
# CPU only. No embedding or model recomputation.
# Resumable during lexical scoring.
# ============================================================

from pathlib import Path
import json
import time

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sklearn.feature_extraction.text import HashingVectorizer

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "anxiety_lineages"

MINHASH = BASE / "minhash"
E5 = BASE / "e5_retrieval"
LEXICAL = BASE / "lexical_scoring"

OUT = BASE / "combined_retrieval"
OUT.mkdir(parents=True, exist_ok=True)

TEMP = OUT / "duckdb_temp"
TEMP.mkdir(exist_ok=True)

NOVEL_FILE = OUT / "e5_novel_pairs.parquet"
SCORED_DIR = OUT / "novel_e5_scored"
SCORED_DIR.mkdir(exist_ok=True)

MAPPING = MINHASH / "text_id_mapping.parquet"

CORE = 0.90
FLOOR = 0.86

# Number of input pairs per resumable output file.
OUTPUT_BATCH = 20000

# Number of pairs vectorized simultaneously.
SCORING_BATCH = 2000

# ============================================================
# 1. CHECK INPUTS
# ============================================================

mapping = pd.read_parquet(MAPPING)

mapping = (
    mapping.sort_values("text_id")
    .reset_index(drop=True)
)

assert np.array_equal(
    mapping["text_id"].to_numpy(),
    np.arange(len(mapping))
)

texts = (
    mapping["normalized_text"]
    .fillna("")
    .astype(str)
    .tolist()
)

n_texts = len(texts)

assert n_texts == 253983, (
    f"Unexpected text count: {n_texts}"
)

assert list(MINHASH.glob("candidate_pairs_*.parquet"))
assert list(E5.glob("semantic_pairs_*.parquet"))
assert list(LEXICAL.glob("scored_pairs_*.parquet"))

print(f"Unique texts: {n_texts:,}", flush=True)

# ============================================================
# 2. SET UP DISK-BACKED DUCKDB
# ============================================================

db = duckdb.connect(
    str(OUT / "retrieval_merge.duckdb")
)

db.execute("SET memory_limit = '3GB'")
db.execute("SET threads = 4")

temp_sql = TEMP.as_posix().replace("'", "''")
db.execute(f"SET temp_directory = '{temp_sql}'")

minhash_glob = (
    MINHASH / "candidate_pairs_*.parquet"
).as_posix()

e5_glob = (
    E5 / "semantic_pairs_*.parquet"
).as_posix()

lexical_glob = (
    LEXICAL / "scored_pairs_*.parquet"
).as_posix()

# ============================================================
# 3. DEDUPLICATE AND ANTI-JOIN
# ============================================================

if not NOVEL_FILE.exists():

    print(
        "\nDeduplicating E5 pairs and removing "
        "MinHash overlaps...",
        flush=True
    )

    # Canonical unordered pair: smaller ID first.
    # The MinHash files already use j < i, but we
    # canonicalize both inputs to be safe.

    query = f"""
    COPY (
        WITH semantic_unique AS (
            SELECT
                LEAST(text_id_1, text_id_2)::BIGINT AS a,
                GREATEST(text_id_1, text_id_2)::BIGINT AS b
            FROM read_parquet('{e5_glob}')
            WHERE text_id_1 <> text_id_2
            GROUP BY 1, 2
        ),
        lexical_unique AS (
            SELECT
                LEAST(text_id_1, text_id_2)::BIGINT AS a,
                GREATEST(text_id_1, text_id_2)::BIGINT AS b
            FROM read_parquet('{minhash_glob}')
            WHERE text_id_1 <> text_id_2
            GROUP BY 1, 2
        )
        SELECT
            s.a AS text_id_1,
            s.b AS text_id_2
        FROM semantic_unique s
        ANTI JOIN lexical_unique m
            ON s.a = m.a AND s.b = m.b
    )
    TO '{NOVEL_FILE.as_posix()}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    db.execute(query)

    print("Novel E5 pairs saved.", flush=True)

else:
    print("\nReusing saved novel E5 pairs.", flush=True)

novel_count = db.execute(
    f"SELECT COUNT(*) FROM read_parquet('{NOVEL_FILE.as_posix()}')"
).fetchone()[0]

print(f"Novel E5 candidate pairs: {novel_count:,}", flush=True)

# ============================================================
# 4. LEXICAL SCORING OF NOVEL E5 PAIRS
# ============================================================

# Match Script 28's hashing representation exactly.
vectorizer = HashingVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    n_features=2**18,
    alternate_sign=False,
    norm="l2",
    dtype=np.float32
)

schema = pa.schema([
    ("text_id_1", pa.int64()),
    ("text_id_2", pa.int64()),
    ("lexical_cosine", pa.float32()),
    ("edge_type", pa.string())
])

reader = pq.ParquetFile(NOVEL_FILE)

print("\nScoring novel semantic candidates...", flush=True)

start_time = time.time()

for batch_idx, batch in enumerate(
    reader.iter_batches(batch_size=OUTPUT_BATCH)
):

    output = (
        SCORED_DIR
        / f"scored_semantic_{batch_idx:05d}.parquet"
    )

    # Already completed output batches are reused.
    if output.exists():
        continue

    pairs = batch.to_pandas()

    a = pairs["text_id_1"].to_numpy(dtype=np.int64)
    b = pairs["text_id_2"].to_numpy(dtype=np.int64)

    assert (a >= 0).all() and (a < n_texts).all()
    assert (b >= 0).all() and (b < n_texts).all()

    accepted = []

    for start in range(0, len(pairs), SCORING_BATCH):

        end = min(start + SCORING_BATCH, len(pairs))

        aa = a[start:end]
        bb = b[start:end]

        # Only vectorize texts required by this batch.
        # Reuse local vectors when a text occurs
        # in multiple pairs within the same batch.

        unique_ids, inverse = np.unique(
            np.concatenate([aa, bb]),
            return_inverse=True
        )

        X = vectorizer.transform(
            [texts[int(i)] for i in unique_ids]
        )

        k = len(aa)

        ia = inverse[:k]
        ib = inverse[k:]

        sim = np.asarray(
            X[ia].multiply(X[ib]).sum(axis=1)
        ).ravel()

        keep = sim >= FLOOR

        if keep.any():
            accepted.append(pd.DataFrame({
                "text_id_1": aa[keep],
                "text_id_2": bb[keep],
                "lexical_cosine": sim[keep].astype(np.float32),
                "edge_type": np.where(
                    sim[keep] >= CORE,
                    "core",
                    "weak"
                )
            }))

    if accepted:
        scored = pd.concat(accepted, ignore_index=True)
        table = pa.Table.from_pandas(
            scored, schema=schema, preserve_index=False
        )
    else:
        table = pa.Table.from_batches(
            [],
            schema=schema
        )

    # Atomic output: interrupted batches won't be
    # mistaken for successfully completed files.
    temp_file = output.with_suffix(".tmp.parquet")

    pq.write_table(
        table,
        temp_file,
        compression="zstd"
    )

    temp_file.replace(output)

    if (batch_idx + 1) % 10 == 0:
        elapsed = (time.time() - start_time) / 60

        print(
            f"Processed {(batch_idx + 1) * OUTPUT_BATCH:,} "
            f"novel pairs | "
            f"Session elapsed: {elapsed:.1f} min",
            flush=True
        )

# ============================================================
# 5. SUMMARIZE RESULTS
# ============================================================

scored_glob = (
    SCORED_DIR / "scored_semantic_*.parquet"
).as_posix()

# Use the saved lexical scoring statistics rather than
# recomputing any existing MinHash similarities.
original_summary = json.loads(
    (LEXICAL / "lexical_scoring_summary.json")
    .read_text(encoding="utf-8")
)

new_counts = db.execute(f"""
    SELECT
        COUNT(*) AS total,
        COUNT(*) FILTER (
            WHERE edge_type = 'core'
        ) AS core,
        COUNT(*) FILTER (
            WHERE edge_type = 'weak'
        ) AS weak
    FROM read_parquet('{scored_glob}')
""").fetchone()

new_total = int(new_counts[0])
new_core = int(new_counts[1])
new_weak = int(new_counts[2])

old_core = int(original_summary["core_pairs"])
old_weak = int(original_summary["weak_pairs"])

# Calculate globally unique E5 pairs for reporting.
semantic_unique = db.execute(f"""
    SELECT COUNT(*) FROM (
        SELECT
            LEAST(text_id_1, text_id_2) AS a,
            GREATEST(text_id_1, text_id_2) AS b
        FROM read_parquet('{e5_glob}')
        WHERE text_id_1 <> text_id_2
        GROUP BY 1, 2
    )
""").fetchone()[0]

semantic_unique = int(semantic_unique)

minhash_total = int(
    original_summary["candidate_pairs_scored"]
)

summary = {
    "unique_texts": n_texts,
    "minhash_candidate_pairs": minhash_total,
    "e5_unique_candidate_pairs": semantic_unique,
    "e5_overlap_with_minhash": (
        semantic_unique - novel_count
    ),
    "e5_novel_candidate_pairs": novel_count,
    "combined_unique_candidate_pairs": (
        minhash_total + novel_count
    ),
    "minhash_core_pairs": old_core,
    "minhash_weak_pairs": old_weak,
    "new_e5_core_pairs": new_core,
    "new_e5_weak_pairs": new_weak,
    "new_e5_retained_pairs": new_total,
    "combined_core_pairs": old_core + new_core,
    "combined_weak_pairs": old_weak + new_weak,
    "combined_retained_pairs": (
        old_core + old_weak + new_total
    ),
    "new_e5_lexical_retention_rate": (
        new_total / novel_count if novel_count else 0
    ),
    "status": "combined_candidates_scored",
    "thresholds": {
        "core": CORE,
        "floor": FLOOR
    }
}

summary_path = OUT / "combined_retrieval_summary.json"

summary_path.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 65)
print("COMBINED RETRIEVAL COMPLETE")
print("=" * 65)

for key, value in summary.items():
    if isinstance(value, dict):
        print(f"{key}: {value}")
    elif isinstance(value, int):
        print(f"{key}: {value:,}")
    else:
        print(f"{key}: {value}")

print("\nSaved to:", OUT)

db.close()
