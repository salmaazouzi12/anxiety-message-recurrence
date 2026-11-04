
from pathlib import Path
import json
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import HashingVectorizer

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "anxiety_lineages"

SOURCE = (
    BASE / "combined_lineages"
    / "combined_lineage_members.parquet"
)

OUT = BASE / "combined_cohesion_repair"
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.86
MAX_LOW_SIM_FRACTION = 0.10

df = pd.read_parquet(SOURCE)

required = [
    "text_id",
    "normalized_text",
    "post_count",
    "lexical_lineage_root"
]
assert all(c in df.columns for c in required)
assert df["text_id"].is_unique

vectorizer = HashingVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    n_features=2**18,
    alternate_sign=False,
    norm="l2",
    dtype=np.float32
)

def cluster_cohesion(sim):
    # Select medoid: member with highest mean
    # lexical similarity to the other members.
    rep = int(np.argmax(sim.mean(axis=1)))
    scores = sim[:, rep]
    low_fraction = float(
        np.mean(scores < THRESHOLD - 1e-6)
    )
    return rep, low_fraction, float(np.median(scores))

def hierarchical_repair(sim, method):
    distance = np.clip(1.0 - sim, 0, 1)
    np.fill_diagonal(distance, 0)

    condensed = squareform(
        distance, checks=False
    )

    tree = linkage(condensed, method=method)

    return fcluster(
        tree,
        t=1.0 - THRESHOLD,
        criterion="distance"
    )

audit_rows = []
member_rows = []
next_lineage_id = 0

groups = df.groupby(
    "lexical_lineage_root",
    sort=True
)

print("Auditing", groups.ngroups, "lineages", flush=True)

for number, (old_root, group) in enumerate(groups, 1):

    group = group.reset_index(drop=True)
    texts = group["normalized_text"].fillna("").tolist()

    # Only vectorize the current family, keeping RAM low.
    X = vectorizer.transform(texts)

    sim = (X @ X.T).toarray()
    sim = np.clip(sim, 0, 1)
    np.fill_diagonal(sim, 1)

    rep, initial_low, initial_median = cluster_cohesion(sim)

    initial_size = len(group)
    repaired = False
    method_used = "none"

    if initial_low > MAX_LOW_SIM_FRACTION:
        labels = hierarchical_repair(sim, "average")
        repaired = True
        method_used = "average"

        # Inspect the new components; apply stricter
        # complete linkage only to components still
        # showing insufficient cohesion.
        next_label = int(labels.max()) + 1

        for cid in np.unique(labels):
            ids = np.where(labels == cid)[0]

            if len(ids) < 2:
                continue

            sub_sim = sim[np.ix_(ids, ids)]
            _, low, _ = cluster_cohesion(sub_sim)

            if low > MAX_LOW_SIM_FRACTION:
                sublabels = hierarchical_repair(
                    sub_sim, "complete"
                )

                for subid in np.unique(sublabels):
                    subids = ids[sublabels == subid]
                    labels[subids] = next_label
                    next_label += 1

                method_used = "average+complete"

    else:
        labels = np.ones(len(group), dtype=int)

    n_components = 0
    excluded_singletons = 0

    for cid in np.unique(labels):
        ids = np.where(labels == cid)[0]

        if len(ids) < 2:
            excluded_singletons += len(ids)
            continue

        n_components += 1
        next_lineage_id += 1

        sub_sim = sim[np.ix_(ids, ids)]
        sub_rep, sub_low, sub_median = cluster_cohesion(
            sub_sim
        )

        for pos in ids:
            row = group.iloc[pos]

            member_rows.append({
                "text_id": int(row["text_id"]),
                "exact_family_id": row["exact_family_id"],
                "post_count": int(row["post_count"]),
                "original_lineage_root": int(old_root),
                "repaired_lineage_id": next_lineage_id,
                "normalized_text": row["normalized_text"]
            })

    audit_rows.append({
        "original_lineage_root": int(old_root),
        "initial_versions": initial_size,
        "initial_posts": int(group["post_count"].sum()),
        "initial_median_similarity": initial_median,
        "initial_low_similarity_fraction": initial_low,
        "repair_applied": repaired,
        "repair_method": method_used,
        "resulting_lineages": n_components,
        "excluded_singletons": excluded_singletons
    })

    if number % 1000 == 0:
        print(
            f"Audited {number:,}/{groups.ngroups:,}",
            flush=True
        )

audit = pd.DataFrame(audit_rows)
members = pd.DataFrame(member_rows)

inventory = (
    members.groupby("repaired_lineage_id")
    .agg(
        unique_versions=("text_id", "size"),
        total_posts=("post_count", "sum"),
        original_lineage_root=("original_lineage_root", "first")
    )
    .reset_index()
)

audit.to_parquet(
    OUT / "cohesion_audit.parquet",
    index=False
)

members.to_parquet(
    OUT / "repaired_lineage_members.parquet",
    index=False
)

inventory.to_parquet(
    OUT / "repaired_lineage_inventory.parquet",
    index=False
)

audit[audit["repair_applied"]].to_csv(
    OUT / "repaired_components.csv",
    index=False,
    encoding="utf-8-sig"
)

summary = {
    "preliminary_lineages": int(groups.ngroups),
    "flagged_for_repair": int(audit["repair_applied"].sum()),
    "repaired_lineages": int(len(inventory)),
    "unique_versions_in_lineages": int(len(members)),
    "posts_in_lineages": int(inventory["total_posts"].sum()),
    "excluded_singletons": int(
        audit["excluded_singletons"].sum()
    ),
    "largest_lineage_versions": int(
        inventory["unique_versions"].max()
    ),
    "median_lineage_versions": float(
        inventory["unique_versions"].median()
    ),
    "status": "lexical_only_cohesion_repaired"
}

with open(
    OUT / "cohesion_repair_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("COHESION REPAIR RESULTS")
print("=" * 60)

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nResults saved to:", OUT)
