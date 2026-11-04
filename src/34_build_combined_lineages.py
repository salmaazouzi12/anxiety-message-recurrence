
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import HashingVectorizer

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "anxiety_lineages"

MINHASH = BASE / "minhash"
LEXICAL = BASE / "lexical_scoring"
SEMANTIC = BASE / "combined_retrieval" / "novel_e5_scored"

OUT = BASE / "combined_lineages"
OUT.mkdir(parents=True, exist_ok=True)

CORE = 0.90
FLOOR = 0.86
BATCH_SIZE = 5000

mapping = pd.read_parquet(
    MINHASH / "text_id_mapping.parquet"
).sort_values("text_id").reset_index(drop=True)

n = len(mapping)
assert np.array_equal(
    mapping["text_id"].to_numpy(),
    np.arange(n)
)

texts = mapping["normalized_text"].fillna("").astype(str).tolist()
post_counts = mapping["post_count"].to_numpy()

files = (
    sorted(LEXICAL.glob("scored_pairs_*.parquet"))
    + sorted(SEMANTIC.glob("scored_semantic_*.parquet"))
)

assert files, "No scored edge files found."

parent = np.arange(n, dtype=np.int32)
size = np.ones(n, dtype=np.int32)

def find(x):
    x = int(x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = int(parent[x])
    return x

def union(a, b):
    a, b = find(a), find(b)
    if a == b:
        return
    if size[a] < size[b]:
        a, b = b, a
    parent[b] = a
    size[a] += size[b]

# ============================================================
# 1. STRONG CORE GRAPH
# ============================================================

print("Building combined strong-core graph...", flush=True)

for k, path in enumerate(files, 1):
    edges = pd.read_parquet(
        path,
        columns=["text_id_1", "text_id_2", "lexical_cosine"]
    )

    core = edges[edges["lexical_cosine"] >= CORE]

    for a, b in zip(
        core["text_id_1"].to_numpy(),
        core["text_id_2"].to_numpy()
    ):
        union(a, b)

    if k % 25 == 0:
        print(f"Core files processed: {k}/{len(files)}", flush=True)

roots = np.array([find(i) for i in range(n)], dtype=np.int32)
core_sizes = np.bincount(roots, minlength=n)

is_core = core_sizes[roots] >= 2

# Representative is the text appearing most frequently
# as an exact version in the original post corpus.
representative = {}

for i in range(n):
    root = int(roots[i])
    previous = representative.get(root)

    if previous is None or (
        post_counts[i] > post_counts[previous]
        or (
            post_counts[i] == post_counts[previous]
            and i < previous
        )
    ):
        representative[root] = i

print(
    "Core families:",
    int((core_sizes >= 2).sum()),
    flush=True
)

# ============================================================
# 2. COLLECT WEAK ATTACHMENT CANDIDATES
# ============================================================

# A singleton can attach to a strong core.
# Two existing strong cores are NEVER merged by weak edges.

print("\nCollecting weak attachments...", flush=True)

candidates = []

for path in files:
    edges = pd.read_parquet(
        path,
        columns=["text_id_1", "text_id_2", "lexical_cosine"]
    )

    weak = edges[
        (edges["lexical_cosine"] >= FLOOR)
        & (edges["lexical_cosine"] < CORE)
    ]

    for a, b, s in weak.itertuples(index=False, name=None):
        a, b = int(a), int(b)
        ra, rb = int(roots[a]), int(roots[b])

        if not is_core[a] and is_core[b]:
            candidates.append((a, rb, float(s)))

        elif not is_core[b] and is_core[a]:
            candidates.append((b, ra, float(s)))

candidates = pd.DataFrame(
    candidates,
    columns=["node", "target_root", "edge_score"]
)

# Multiple candidate edges may propose the same
# node -> core relationship.
candidates = (
    candidates.sort_values(
        ["node", "target_root", "edge_score"],
        ascending=[True, True, False]
    )
    .drop_duplicates(["node", "target_root"])
    .reset_index(drop=True)
)

print(
    "Candidate singleton-to-core relationships:",
    len(candidates),
    flush=True
)

# ============================================================
# 3. REPRESENTATIVE SIMILARITY
# ============================================================

vectorizer = HashingVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    n_features=2**18,
    alternate_sign=False,
    norm="l2",
    dtype=np.float32
)

candidates["representative_id"] = (
    candidates["target_root"].map(representative)
)

accepted = []

print("Checking representative similarity...", flush=True)

for start in range(0, len(candidates), BATCH_SIZE):
    batch = candidates.iloc[start:start+BATCH_SIZE].copy()

    nodes = batch["node"].to_numpy(dtype=np.int64)
    reps = batch["representative_id"].to_numpy(dtype=np.int64)

    unique_ids, inverse = np.unique(
        np.concatenate([nodes, reps]),
        return_inverse=True
    )

    X = vectorizer.transform(
        [texts[int(i)] for i in unique_ids]
    )

    k = len(nodes)

    similarity = np.asarray(
        X[inverse[:k]]
        .multiply(X[inverse[k:]])
        .sum(axis=1)
    ).ravel()

    batch["representative_cosine"] = similarity

    accepted.append(
        batch[batch["representative_cosine"] >= FLOOR]
    )

    if (start // BATCH_SIZE + 1) % 10 == 0:
        print(
            f"Checked {min(start+BATCH_SIZE, len(candidates)):,}"
            f"/{len(candidates):,}",
            flush=True
        )

if accepted:
    accepted = pd.concat(accepted, ignore_index=True)
else:
    accepted = candidates.iloc[:0].copy()

# One assignment per singleton:
# strongest direct edge, then representative similarity.
accepted = (
    accepted.sort_values(
        ["node", "edge_score", "representative_cosine", "target_root"],
        ascending=[True, False, False, True]
    )
    .drop_duplicates("node")
)

assignments = roots.copy()
attached_nodes = set()

for row in accepted.itertuples(index=False):
    node = int(row.node)
    assignments[node] = int(row.target_root)
    attached_nodes.add(node)

print("Singletons attached to cores:", len(attached_nodes))

# ============================================================
# 4. WEAK-ONLY DISJOINT PAIRS
# ============================================================

# Gather weak edges between initial singletons.
# Greedy matching prevents nodes from being assigned twice
# and prevents weak-only transitive chains.

print("\nConstructing weak-only disjoint pairs...", flush=True)

weak_only = []

for path in files:
    edges = pd.read_parquet(
        path,
        columns=["text_id_1", "text_id_2", "lexical_cosine"]
    )

    weak = edges[
        (edges["lexical_cosine"] >= FLOOR)
        & (edges["lexical_cosine"] < CORE)
    ]

    for a, b, s in weak.itertuples(index=False, name=None):
        a, b = int(a), int(b)

        if is_core[a] or is_core[b]:
            continue

        if a in attached_nodes or b in attached_nodes:
            continue

        weak_only.append(
            (min(a, b), max(a, b), float(s))
        )

weak_only = pd.DataFrame(
    weak_only,
    columns=["a", "b", "score"]
)

weak_only = (
    weak_only.sort_values(
        ["score", "a", "b"],
        ascending=[False, True, True]
    )
    .drop_duplicates(["a", "b"])
)

used = set(attached_nodes)
weak_pairs_created = 0

for a, b, score in weak_only.itertuples(index=False, name=None):
    a, b = int(a), int(b)

    if a in used or b in used:
        continue

    # Both were singleton components in the strong graph.
    # Use the smaller text ID as their shared root.
    assignments[a] = a
    assignments[b] = a

    used.add(a)
    used.add(b)
    weak_pairs_created += 1

print("Weak-only pairs created:", weak_pairs_created)

# ============================================================
# 5. INVENTORY AND POST COUNTS
# ============================================================

mapping["combined_lineage_root"] = assignments

inventory = (
    mapping.groupby("combined_lineage_root")
    .agg(
        unique_versions=("text_id", "size"),
        total_posts=("post_count", "sum")
    )
    .reset_index()
)

inventory = inventory[
    inventory["unique_versions"] >= 2
].copy()

valid_roots = set(inventory["combined_lineage_root"])

members = mapping[
    mapping["combined_lineage_root"].isin(valid_roots)
].copy()

assert members["text_id"].is_unique

assert (
    int(inventory["unique_versions"].sum()) == len(members)
)

# Keep format compatible with the previous repair script.
members["lexical_lineage_root"] = (
    members["combined_lineage_root"]
)

members.to_parquet(
    OUT / "combined_lineage_members.parquet",
    index=False
)

inventory.to_parquet(
    OUT / "combined_lineage_inventory.parquet",
    index=False
)

accepted.to_parquet(
    OUT / "accepted_weak_core_attachments.parquet",
    index=False
)

summary = {
    "unique_texts": n,
    "strong_core_families": int((core_sizes >= 2).sum()),
    "singleton_core_attachments": len(attached_nodes),
    "weak_only_pairs": weak_pairs_created,
    "combined_preliminary_lineages": int(len(inventory)),
    "unique_versions_in_lineages": int(len(members)),
    "posts_in_lineages": int(inventory["total_posts"].sum()),
    "largest_lineage_versions": int(
        inventory["unique_versions"].max()
    ),
    "median_lineage_versions": float(
        inventory["unique_versions"].median()
    ),
    "status": "combined_preliminary_before_cohesion_repair"
}

(OUT / "combined_lineage_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 65)
print("COMBINED PRELIMINARY LINEAGES")
print("=" * 65)

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nSaved to:", OUT)
