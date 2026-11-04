
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "anxiety_lineages"

OUT = BASE / "semantic_transformation"
OUT.mkdir(parents=True, exist_ok=True)

POSTS_FILE = (
    BASE / "propagation_analysis"
    / "lineage_posts.parquet"
)

MEMBERS_FILE = (
    BASE / "combined_cohesion_repair"
    / "repaired_lineage_members.parquet"
)

MAPPING_FILE = (
    BASE / "minhash"
    / "text_id_mapping.parquet"
)

EMBEDDINGS_FILE = (
    BASE / "e5_retrieval"
    / "e5_embeddings.npy"
)

# ============================================================
# 1. LOAD EXISTING DATA
# ============================================================

print("=" * 65)
print("SEMANTIC TRANSFORMATION ANALYSIS")
print("=" * 65)

posts = pd.read_parquet(POSTS_FILE)
members = pd.read_parquet(MEMBERS_FILE)
mapping = pd.read_parquet(MAPPING_FILE)

embeddings = np.load(
    EMBEDDINGS_FILE,
    mmap_mode="r"
)

print(f"Posts loaded: {len(posts):,}")
print(f"Members loaded: {len(members):,}")
print(f"Unique texts: {len(mapping):,}")
print(f"Embedding matrix: {embeddings.shape}")

assert embeddings.shape == (len(mapping), 384)

assert mapping["text_id"].is_unique
assert members["exact_family_id"].is_unique

# ============================================================
# 2. BUILD EXACT-FAMILY -> EMBEDDING-ID LOOKUP
# ============================================================

id_lookup = mapping[
    ["exact_family_id", "text_id"]
].copy()

multi_posts = posts[
    posts["lineage_type"] == "multi_version"
].copy()

# Already contains the corrected full timestamps
# from Script 36.
multi_posts["created_at"] = pd.to_datetime(
    multi_posts["created_at"],
    errors="coerce"
)

if multi_posts["created_at"].isna().any():
    raise ValueError(
        "Missing timestamps in multi-version posts."
    )

# Each distinct textual version is represented by
# its earliest observed appearance in the corpus.
versions = (
    multi_posts.groupby(
        ["lineage_id", "exact_family_id"],
        as_index=False
    )
    .agg(
        first_seen=("created_at", "min"),
        appearances=("exact_family_id", "size"),
        pages=("page_key", "nunique")
    )
)

versions = versions.merge(
    id_lookup,
    on="exact_family_id",
    how="left",
    validate="many_to_one"
)

if versions["text_id"].isna().any():
    raise ValueError(
        "Some textual versions have no E5 embedding ID."
    )

versions["text_id"] = (
    versions["text_id"].astype(np.int64)
)

# ============================================================
# 3. SORT VERSIONS CHRONOLOGICALLY
# ============================================================

versions = (
    versions.sort_values(
        ["lineage_id", "first_seen", "text_id"]
    )
    .reset_index(drop=True)
)

versions["version_order"] = (
    versions.groupby("lineage_id").cumcount()
)

first = (
    versions.groupby("lineage_id", as_index=False)
    .first()[
        ["lineage_id", "first_seen", "text_id"]
    ]
    .rename(
        columns={
            "first_seen": "lineage_first_seen",
            "text_id": "first_text_id"
        }
    )
)

versions = versions.merge(
    first,
    on="lineage_id",
    validate="many_to_one"
)

versions["hours_since_first"] = (
    (
        versions["first_seen"]
        - versions["lineage_first_seen"]
    ).dt.total_seconds()
    / 3600
)

versions["days_since_first"] = (
    versions["hours_since_first"] / 24
)

# ============================================================
# 4. COMPUTE SEMANTIC DISTANCE
# ============================================================

print("\nCalculating semantic distances...", flush=True)

ids = versions["text_id"].to_numpy(dtype=np.int64)
first_ids = versions["first_text_id"].to_numpy(
    dtype=np.int64
)

# E5 embeddings were normalized when generated,
# but normalize again to ensure safe cosine scoring.
def cosine_for_pairs(a_ids, b_ids, batch_size=30000):
    result = np.empty(len(a_ids), dtype=np.float32)

    for start in range(0, len(a_ids), batch_size):
        end = min(start + batch_size, len(a_ids))

        a = np.asarray(
            embeddings[a_ids[start:end]],
            dtype=np.float32
        )
        b = np.asarray(
            embeddings[b_ids[start:end]],
            dtype=np.float32
        )

        dot = np.einsum("ij,ij->i", a, b)

        norm_a = np.linalg.norm(a, axis=1)
        norm_b = np.linalg.norm(b, axis=1)

        similarity = dot / np.maximum(
            norm_a * norm_b,
            1e-12
        )

        result[start:end] = np.clip(
            similarity, -1.0, 1.0
        )

    return result

versions["cosine_to_first"] = cosine_for_pairs(
    ids, first_ids
)

versions["semantic_distance_to_first"] = (
    1.0 - versions["cosine_to_first"]
).clip(lower=0)

# ============================================================
# 5. COMPARE CONSECUTIVE OBSERVED VERSIONS
# ============================================================

versions["previous_text_id"] = (
    versions.groupby("lineage_id")["text_id"]
    .shift(1)
)

versions["previous_seen"] = (
    versions.groupby("lineage_id")["first_seen"]
    .shift(1)
)

has_previous = versions["previous_text_id"].notna()

versions["semantic_distance_to_previous"] = np.nan

if has_previous.any():
    previous_ids = (
        versions.loc[has_previous, "previous_text_id"]
        .astype(np.int64)
        .to_numpy()
    )

    current_ids = (
        versions.loc[has_previous, "text_id"]
        .to_numpy(dtype=np.int64)
    )

    previous_cosine = cosine_for_pairs(
        current_ids, previous_ids
    )

    versions.loc[
        has_previous,
        "semantic_distance_to_previous"
    ] = np.maximum(0, 1.0 - previous_cosine)

versions["hours_since_previous"] = (
    (
        versions["first_seen"]
        - versions["previous_seen"]
    ).dt.total_seconds()
    / 3600
)

# ============================================================
# 6. LINEAGE-LEVEL DESCRIPTIVE STATISTICS
# ============================================================

print("Building lineage-level statistics...", flush=True)

lineage_stats = (
    versions.groupby("lineage_id")
    .agg(
        unique_versions=("text_id", "size"),
        first_seen=("first_seen", "min"),
        last_version_seen=("first_seen", "max"),
        max_semantic_distance=(
            "semantic_distance_to_first", "max"
        ),
        mean_semantic_distance=(
            "semantic_distance_to_first", "mean"
        ),
        mean_step_distance=(
            "semantic_distance_to_previous", "mean"
        ),
        max_days_since_first=(
            "days_since_first", "max"
        ),
    )
    .reset_index()
)

# Compare earliest vs latest observed version.
last_versions = (
    versions.groupby("lineage_id", as_index=False)
    .last()[
        ["lineage_id", "semantic_distance_to_first"]
    ]
    .rename(
        columns={
            "semantic_distance_to_first":
            "latest_version_semantic_distance"
        }
    )
)

lineage_stats = lineage_stats.merge(
    last_versions,
    on="lineage_id",
    validate="one_to_one"
)

# ============================================================
# 7. TEMPORAL DISTANCE BINS
# ============================================================

# Descriptive analysis: these bins can have different
# lineage compositions and are not causal estimates.

later = versions[
    versions["version_order"] > 0
].copy()

bins = [
    -0.001,
    1,
    7,
    30,
    90,
    180,
    365,
    np.inf
]

labels = [
    "0-1 day",
    "1-7 days",
    "7-30 days",
    "30-90 days",
    "90-180 days",
    "180-365 days",
    "365+ days"
]

later["time_bin"] = pd.cut(
    later["days_since_first"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

binned = (
    later.groupby(
        "time_bin",
        observed=True
    )
    .agg(
        versions=("text_id", "size"),
        lineages=("lineage_id", "nunique"),
        mean_semantic_distance=(
            "semantic_distance_to_first", "mean"
        ),
        median_semantic_distance=(
            "semantic_distance_to_first", "median"
        ),
        median_step_distance=(
            "semantic_distance_to_previous", "median"
        )
    )
    .reset_index()
)

# ============================================================
# 8. WITHIN-LINEAGE ASSOCIATION
# ============================================================

# Hypothesis:
# Later appearances of distinct textual versions
# are associated with greater distance from the
# earliest observed version.
#
# Include only lineages with at least two later versions
# and at least two different elapsed-time values.
#
# Use log1p(days) to reduce extreme leverage from
# a small number of very long time spans.

analysis = later.copy()

analysis["log_elapsed_days"] = np.log1p(
    analysis["days_since_first"].clip(lower=0)
)

eligible_sizes = (
    analysis.groupby("lineage_id")["text_id"].transform("size")
)

time_variation = (
    analysis.groupby("lineage_id")["log_elapsed_days"]
    .transform("nunique")
)

analysis = analysis[
    (eligible_sizes >= 2)
    & (time_variation >= 2)
].copy()

# Demeaning is equivalent to a lineage fixed-effects
# regression with one explanatory variable.
if len(analysis):

    analysis["x_centered"] = (
        analysis["log_elapsed_days"]
        - analysis.groupby("lineage_id")[
            "log_elapsed_days"
        ].transform("mean")
    )

    analysis["y_centered"] = (
        analysis["semantic_distance_to_first"]
        - analysis.groupby("lineage_id")[
            "semantic_distance_to_first"
        ].transform("mean")
    )

    x = analysis["x_centered"].to_numpy()
    y = analysis["y_centered"].to_numpy()

    denominator = float(np.dot(x, x))

    beta = (
        float(np.dot(x, y) / denominator)
        if denominator > 0
        else np.nan
    )

    # Cluster-robust standard error by lineage.
    # This accounts for multiple observations
    # originating from the same reconstructed family.
    residual = y - beta * x

    analysis["cluster_score"] = (
        analysis["x_centered"]
        * residual
    )

    cluster_sums = (
        analysis.groupby("lineage_id")["cluster_score"]
        .sum()
        .to_numpy()
    )

    G = analysis["lineage_id"].nunique()
    N = len(analysis)
    K = 1

    correction = (
        (G / (G - 1)) * ((N - 1) / (N - K))
        if G > 1 and N > K
        else 1.0
    )

    variance = (
        correction
        * float(np.dot(cluster_sums, cluster_sums))
        / (denominator ** 2)
    )

    se = float(np.sqrt(max(variance, 0)))

    ci_lower = beta - 1.96 * se
    ci_upper = beta + 1.96 * se

else:
    beta = np.nan
    se = np.nan
    ci_lower = np.nan
    ci_upper = np.nan
    G = 0
    N = 0

# ============================================================
# 9. SAVE OUTPUTS
# ============================================================

versions.to_parquet(
    OUT / "semantic_version_trajectories.parquet",
    index=False
)

lineage_stats.to_parquet(
    OUT / "semantic_lineage_statistics.parquet",
    index=False
)

lineage_stats.to_csv(
    OUT / "semantic_lineage_statistics.csv",
    index=False,
    encoding="utf-8-sig"
)

binned.to_csv(
    OUT / "semantic_distance_by_time.csv",
    index=False,
    encoding="utf-8-sig"
)

summary = {
    "model": "intfloat/multilingual-e5-small",
    "total_multi_version_lineages": int(
        len(lineage_stats)
    ),
    "total_unique_versions": int(len(versions)),
    "later_version_observations": int(len(later)),
    "median_max_semantic_distance": float(
        lineage_stats["max_semantic_distance"].median()
    ),
    "median_latest_version_distance": float(
        lineage_stats[
            "latest_version_semantic_distance"
        ].median()
    ),
    "median_consecutive_distance": float(
        later["semantic_distance_to_previous"].median()
    ),
    "median_time_span_days": float(
        lineage_stats["max_days_since_first"].median()
    ),
    "fixed_effects": {
        "dependent_variable": "semantic_distance_to_first",
        "independent_variable": "log1p(days_since_first)",
        "coefficient": (
            float(beta) if np.isfinite(beta) else None
        ),
        "cluster_robust_se": (
            float(se) if np.isfinite(se) else None
        ),
        "ci_95_lower": (
            float(ci_lower)
            if np.isfinite(ci_lower) else None
        ),
        "ci_95_upper": (
            float(ci_upper)
            if np.isfinite(ci_upper) else None
        ),
        "eligible_lineages": int(G),
        "observations": int(N)
    },
    "limitations": [
        "Lineages were constructed using lexical similarity thresholds",
        "Semantic distance is measured using E5 representations",
        "Earliest observed version is not necessarily the original source",
        "Chronological order does not establish a transmission path",
        "E5 inputs may have been truncated during embedding",
        "Source AST timestamps are treated as a consistent local clock"
    ],
    "status": "semantic_transformation_analysis_complete"
}

(OUT / "semantic_transformation_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

# ============================================================
# 10. REPORT
# ============================================================

print("\n" + "=" * 65)
print("SEMANTIC TRANSFORMATION RESULTS")
print("=" * 65)

for key, value in summary.items():
    if key != "fixed_effects" and key != "limitations":
        print(f"{key}: {value}")

print("\nWITHIN-LINEAGE FIXED EFFECTS")

for key, value in summary["fixed_effects"].items():
    print(f"{key}: {value}")

print("\nSEMANTIC DISTANCE BY TIME BIN")
print(binned.to_string(index=False))

print("\nResults saved to:")
print(OUT)

print("\nAnalysis complete.")
