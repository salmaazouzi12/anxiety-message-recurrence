
# ============================================================
# 32_e5_faiss_candidates.py
#
# Semantic candidate retrieval:
# multilingual-e5-small + FAISS IVF-Flat
#
# CPU compatible
# Resumable embedding generation and neighbor retrieval
# No annotation or classifier training
# Existing lineage outputs remain untouched
# ============================================================

from pathlib import Path
import json
import hashlib
import time

import numpy as np
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "anxiety_lineages"

SOURCE = BASE / "minhash" / "text_id_mapping.parquet"
OUT = BASE / "e5_retrieval"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "intfloat/multilingual-e5-small"

DIM = 384
EMBED_BATCH = 64
CHECKPOINT_EVERY = 512

QUERY_BATCH = 2000
NEIGHBORS = 15

NLIST = 512
NPROBE = 32
TRAIN_SIZE = 50000
SEED = 42

EMBED_FILE = OUT / "e5_embeddings.npy"
PROGRESS_FILE = OUT / "embedding_progress.json"
INDEX_FILE = OUT / "e5_faiss.index"
SUMMARY_FILE = OUT / "retrieval_summary.json"

# ============================================================
# 1. LOAD TEXT MAPPING
# ============================================================

mapping = pd.read_parquet(SOURCE)
mapping = mapping.sort_values("text_id").reset_index(drop=True)

assert np.array_equal(
    mapping["text_id"].to_numpy(),
    np.arange(len(mapping))
)

assert mapping["exact_family_id"].is_unique

texts = (
    mapping["normalized_text"]
    .fillna("")
    .astype(str)
    .tolist()
)

N = len(texts)

fingerprint = hashlib.sha256(
    "\n".join(
        mapping["exact_family_id"].astype(str)
    ).encode("utf-8")
).hexdigest()

print(f"Unique texts: {N:,}", flush=True)
print(f"Model: {MODEL_NAME}", flush=True)

# ============================================================
# 2. LOAD OR CREATE EMBEDDING CACHE
# ============================================================

if PROGRESS_FILE.exists():
    with open(PROGRESS_FILE, encoding="utf-8") as f:
        state = json.load(f)

    assert state["fingerprint"] == fingerprint, (
        "Text mapping changed. Existing E5 cache "
        "cannot safely be reused."
    )
    assert state["model"] == MODEL_NAME

    completed = int(state["completed"])

    embeddings = np.lib.format.open_memmap(
        EMBED_FILE,
        mode="r+"
    )

    assert embeddings.shape == (N, DIM)

else:
    embeddings = np.lib.format.open_memmap(
        EMBED_FILE,
        mode="w+",
        dtype=np.float32,
        shape=(N, DIM)
    )

    completed = 0

    state = {
        "fingerprint": fingerprint,
        "model": MODEL_NAME,
        "completed": 0
    }

    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

print(f"Embeddings already completed: {completed:,}")

# ============================================================
# 3. GENERATE EMBEDDINGS
# ============================================================

if completed < N:

    print("\nLoading multilingual E5...", flush=True)

    encoder = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    # E5 uses text prefixes. For symmetric text-to-text
    # retrieval we use the same passage representation
    # for both indexed and queried texts.
    #
    # Long posts are truncated at the model limit;
    # this must be recorded as a methodological limitation.

    start_time = time.time()

    for start in range(completed, N, CHECKPOINT_EVERY):

        end = min(start + CHECKPOINT_EVERY, N)

        current = [
            "passage: " + t
            for t in texts[start:end]
        ]

        vectors = encoder.encode(
            current,
            batch_size=EMBED_BATCH,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype(np.float32)

        assert vectors.shape == (end - start, DIM)

        embeddings[start:end] = vectors

        embeddings.flush()

        state["completed"] = end

        temp = PROGRESS_FILE.with_suffix(".tmp")
        temp.write_text(
            json.dumps(state, indent=2),
            encoding="utf-8"
        )
        temp.replace(PROGRESS_FILE)

        if end % 10000 < CHECKPOINT_EVERY or end == N:

            elapsed = (time.time() - start_time) / 60

            print(
                f"Encoded {end:,}/{N:,} | "
                f"Session elapsed: {elapsed:.1f} min",
                flush=True
            )

    del encoder

print("\nEmbeddings ready.", flush=True)

# ============================================================
# 4. BUILD OR LOAD FAISS INDEX
# ============================================================

if INDEX_FILE.exists():

    print("Loading saved FAISS index...", flush=True)
    index = faiss.read_index(str(INDEX_FILE))

    assert index.ntotal == N
    assert index.d == DIM

else:

    print("\nBuilding FAISS IVF-Flat index...", flush=True)

    faiss.omp_set_num_threads(4)

    quantizer = faiss.IndexFlatIP(DIM)

    index = faiss.IndexIVFFlat(
        quantizer,
        DIM,
        NLIST,
        faiss.METRIC_INNER_PRODUCT
    )

    rng = np.random.default_rng(SEED)

    sample_ids = rng.choice(
        N,
        size=min(TRAIN_SIZE, N),
        replace=False
    )

    training_vectors = np.ascontiguousarray(
        embeddings[sample_ids],
        dtype=np.float32
    )

    print("Training FAISS coarse quantizer...", flush=True)
    index.train(training_vectors)

    print("Adding embeddings to index...", flush=True)

    for start in range(0, N, 10000):

        end = min(start + 10000, N)

        batch = np.ascontiguousarray(
            embeddings[start:end],
            dtype=np.float32
        )

        index.add(batch)

        print(f"Indexed {end:,}/{N:,}", flush=True)

    faiss.write_index(index, str(INDEX_FILE))

index.nprobe = NPROBE

print(f"Indexed vectors: {index.ntotal:,}")

# ============================================================
# 5. RETRIEVE SEMANTIC NEIGHBORS
# ============================================================

print("\nRetrieving nearest neighbors...", flush=True)

part = 0
raw_pairs = 0
start_time = time.time()

for start in range(0, N, QUERY_BATCH):

    end = min(start + QUERY_BATCH, N)

    output_file = (
        OUT / f"semantic_pairs_{part:04d}.parquet"
    )

    # Resume retrieval without rewriting completed files.
    if output_file.exists():
        existing = pd.read_parquet(
            output_file,
            columns=["text_id_1", "text_id_2"]
        )
        raw_pairs += len(existing)
        part += 1
        continue

    query_vectors = np.ascontiguousarray(
        embeddings[start:end],
        dtype=np.float32
    )

    # Request extra neighbors to allow removal of self.
    similarities, neighbors = index.search(
        query_vectors,
        NEIGHBORS + 10
    )

    records = []

    for local_id in range(end - start):

        source_id = start + local_id
        kept = 0

        for score, neighbor in zip(
            similarities[local_id],
            neighbors[local_id]
        ):

            neighbor = int(neighbor)

            if neighbor < 0 or neighbor == source_id:
                continue

            a = min(source_id, neighbor)
            b = max(source_id, neighbor)

            records.append((
                a,
                b,
                float(score)
            ))

            kept += 1

            if kept >= NEIGHBORS:
                break

    pairs = pd.DataFrame(
        records,
        columns=[
            "text_id_1",
            "text_id_2",
            "semantic_cosine"
        ]
    )

    # Deduplicate within this batch.
    # A later union step will deduplicate across batches.
    pairs = (
        pairs.groupby(
            ["text_id_1", "text_id_2"],
            as_index=False
        )["semantic_cosine"]
        .max()
    )

    pairs.to_parquet(output_file, index=False)

    raw_pairs += len(pairs)

    part += 1

    if part % 10 == 0 or end == N:

        elapsed = (time.time() - start_time) / 60

        print(
            f"Retrieved {end:,}/{N:,} texts | "
            f"Pairs saved: {raw_pairs:,} | "
            f"Session elapsed: {elapsed:.1f} min",
            flush=True
        )

# ============================================================
# 6. SAVE SUMMARY
# ============================================================

summary = {
    "unique_texts": N,
    "model": MODEL_NAME,
    "embedding_dimensions": DIM,
    "faiss_index": "IVF-Flat",
    "nlist": NLIST,
    "nprobe": NPROBE,
    "neighbors_per_text": NEIGHBORS,
    "saved_pair_rows_before_global_dedup": raw_pairs,
    "candidate_files": part,
    "status": "semantic_candidate_retrieval_complete",
    "limitations": [
        "Approximate nearest neighbor retrieval",
        "Long texts may be truncated by the encoder",
        "Semantic similarity does not establish textual reuse",
        "Global pair deduplication still required"
    ]
}

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\n" + "=" * 65)
print("E5 + FAISS RETRIEVAL COMPLETE")
print("=" * 65)

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nOutputs:", OUT)
