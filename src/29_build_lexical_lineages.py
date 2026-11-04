
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "anxiety_lineages"
SCORED = BASE / "lexical_scoring"
OUT = BASE / "lexical_lineages"
OUT.mkdir(parents=True, exist_ok=True)

mapping = pd.read_parquet(
    BASE / "minhash" / "text_id_mapping.parquet"
).sort_values("text_id").reset_index(drop=True)

n = len(mapping)
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

files = sorted(SCORED.glob("scored_pairs_*.parquet"))
assert files, "No scored pair files found."

print("Building strong-core components...", flush=True)

# Stage 1: Strong edges only
for k, path in enumerate(files, 1):
    df = pd.read_parquet(path)
    core = df[df["lexical_cosine"] >= 0.90]

    for a, b in zip(
        core["text_id_1"].to_numpy(),
        core["text_id_2"].to_numpy()
    ):
        union(a, b)

    if k % 10 == 0:
        print(f"Core files {k}/{len(files)}", flush=True)

roots = np.array([find(i) for i in range(n)], dtype=np.int32)
counts = pd.Series(roots).value_counts()

# Representative: most frequently occurring exact text
# in each core component, with deterministic tie-breaking.
post_counts = mapping["post_count"].to_numpy()
representatives = {}

for i, root in enumerate(roots):
    root = int(root)
    if root not in representatives:
        representatives[root] = i
    else:
        old = representatives[root]
        if post_counts[i] > post_counts[old]:
            representatives[root] = i

print("Strong-core components:", len(counts))
print("Largest core component:", int(counts.max()))

# Stage 2: Weak-edge controlled attachment
# A weak node may attach directly to an established core
# only when similarity to its representative is also >= .86.
#
# We only have scored candidate edges. Build a lookup of
# observed similarities for the representative check.
# Missing edges are NOT assumed similar.

similarities = {}

print("Indexing retained lexical similarities...", flush=True)

for path in files:
    df = pd.read_parquet(path)

    for a, b, s in zip(
        df["text_id_1"].to_numpy(),
        df["text_id_2"].to_numpy(),
        df["lexical_cosine"].to_numpy()
    ):
        a, b = int(a), int(b)
        if a > b:
            a, b = b, a
        similarities[(a, b)] = float(s)

def pair_similarity(a, b):
    if a == b:
        return 1.0
    if a > b:
        a, b = b, a
    return similarities.get((a, b), 0.0)

# Freeze initial core memberships so weak edges cannot
# recursively chain additional weak nodes into a family.
core_members = {
    int(root) for root, count in counts.items()
    if count >= 2
}

assignments = roots.copy()
best_attachment = {}

for path in files:
    df = pd.read_parquet(path)
    weak = df[
        (df["lexical_cosine"] >= 0.86)
        & (df["lexical_cosine"] < 0.90)
    ]

    for a, b, score in zip(
        weak["text_id_1"].to_numpy(),
        weak["text_id_2"].to_numpy(),
        weak["lexical_cosine"].to_numpy()
    ):
        a, b = int(a), int(b)

        ra, rb = int(roots[a]), int(roots[b])

        for node, target_root in [(a, rb), (b, ra)]:
            if target_root not in core_members:
                continue
            if int(roots[node]) in core_members:
                continue

            representative = representatives[target_root]
            rep_score = pair_similarity(node, representative)

            if rep_score < 0.86:
                continue

            candidate = (float(score), float(rep_score), target_root)

            if (
                node not in best_attachment
                or candidate > best_attachment[node]
            ):
                best_attachment[node] = candidate

for node, (_, _, target_root) in best_attachment.items():
    assignments[node] = target_root

# Assign weak-only edges conservatively as two-node
# families only when neither node has an existing family.
for path in files:
    df = pd.read_parquet(path)
    weak = df[
        (df["lexical_cosine"] >= 0.86)
        & (df["lexical_cosine"] < 0.90)
    ]

    for a, b in zip(
        weak["text_id_1"].to_numpy(),
        weak["text_id_2"].to_numpy()
    ):
        a, b = int(a), int(b)
        if (
            counts.get(int(roots[a]), 0) == 1
            and counts.get(int(roots[b]), 0) == 1
            and a not in best_attachment
            and b not in best_attachment
            and assignments[a] == roots[a]
            and assignments[b] == roots[b]
        ):
            assignments[b] = assignments[a]

mapping["lexical_lineage_root"] = assignments

family_sizes = (
    mapping.groupby("lexical_lineage_root")
    .agg(
        unique_versions=("text_id", "size"),
        posts=("post_count", "sum")
    )
    .reset_index()
)

lineages = family_sizes[
    family_sizes["unique_versions"] >= 2
].copy()

valid_roots = set(lineages["lexical_lineage_root"])

members = mapping[
    mapping["lexical_lineage_root"].isin(valid_roots)
].copy()

members.to_parquet(
    OUT / "lexical_lineage_members.parquet",
    index=False
)

lineages.to_parquet(
    OUT / "lexical_lineage_inventory.parquet",
    index=False
)

summary = {
    "unique_texts": int(n),
    "preliminary_lineages": int(len(lineages)),
    "unique_versions_in_lineages": int(len(members)),
    "posts_in_lineages": int(lineages["posts"].sum()),
    "largest_lineage_versions": (
        int(lineages["unique_versions"].max())
        if len(lineages) else 0
    ),
    "status": "preliminary_lexical_only",
}

with open(
    OUT / "preliminary_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

print("\nPRELIMINARY LEXICAL LINEAGES")
for key, value in summary.items():
    print(f"{key}: {value}")

print("\nSaved to:", OUT)
