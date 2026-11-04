
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "anxiety_lineages"

SOURCE = (
    BASE / "semantic_transformation"
    / "semantic_version_trajectories.parquet"
)

OUT = BASE / "semantic_robustness"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("SCRIPT 39 — SEMANTIC TRANSFORMATION ROBUSTNESS")
print("=" * 70)

df = pd.read_parquet(SOURCE)

required = [
    "lineage_id",
    "version_order",
    "days_since_first",
    "semantic_distance_to_first",
]

for col in required:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")

print(f"\nLoaded {len(df):,} textual versions")
print(f"Lineages: {df['lineage_id'].nunique():,}")

# ============================================================
# 1. PREPARE ANALYSIS VARIABLES
# ============================================================

df["days_since_first"] = pd.to_numeric(
    df["days_since_first"],
    errors="coerce"
)

df["semantic_distance_to_first"] = pd.to_numeric(
    df["semantic_distance_to_first"],
    errors="coerce"
)

df["version_order"] = pd.to_numeric(
    df["version_order"],
    errors="coerce"
)

df = df.dropna(subset=required).copy()

df["log_elapsed_days"] = np.log1p(
    df["days_since_first"].clip(lower=0)
)

# Total distinct versions in each family.
lineage_size = (
    df.groupby("lineage_id")["version_order"]
    .transform("size")
)

df["lineage_size"] = lineage_size

# Number of distinct versions before the current one.
df["log_version_order"] = np.log1p(
    df["version_order"]
)

# Exclude the reference version.
# It has zero distance from itself by construction.
later = df[df["version_order"] > 0].copy()

# ============================================================
# 2. LINEAGE FIXED-EFFECTS ESTIMATOR
# ============================================================

def estimate_fe(
    data,
    predictors,
    equal_lineage_weight=False
):
    """
    Lineage fixed effects through within-group demeaning.

    Uses a cluster-robust covariance estimator,
    clustered by reconstructed lineage.

    The estimator excludes groups with fewer than
    two usable later-version observations.

    Models with multiple predictors additionally
    require a full-rank design matrix.
    """

    needed = (
        ["lineage_id", "semantic_distance_to_first"]
        + predictors
    )

    d = data[needed].dropna().copy()

    counts = d.groupby("lineage_id")[
        "lineage_id"
    ].transform("size")

    d = d[counts >= 2].copy()

    if d.empty:
        return None

    group = d.groupby("lineage_id", sort=False)

    y = (
        d["semantic_distance_to_first"]
        - group["semantic_distance_to_first"]
        .transform("mean")
    ).to_numpy(dtype=float)

    X = np.column_stack([
        (
            d[p]
            - group[p].transform("mean")
        ).to_numpy(dtype=float)
        for p in predictors
    ])

    # Remove lineages where none of the explanatory
    # variables vary within the family.
    varying = np.any(np.abs(X) > 1e-12, axis=1)

    d = d.loc[varying].copy()
    X = X[varying]
    y = y[varying]

    if len(d) == 0:
        return None

    group_codes, group_labels = pd.factorize(
        d["lineage_id"],
        sort=False
    )

    G = len(group_labels)
    N = len(d)
    K = len(predictors)

    if G <= 1 or N <= K:
        return None

    if equal_lineage_weight:
        group_n = np.bincount(group_codes)
        weights = 1.0 / group_n[group_codes]
    else:
        weights = np.ones(N, dtype=float)

    sqrt_w = np.sqrt(weights)

    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w

    xtx = Xw.T @ Xw

    if np.linalg.matrix_rank(xtx) < K:
        return None

    beta = np.linalg.solve(
        xtx,
        Xw.T @ yw
    )

    residual = y - X @ beta

    # Cluster score for each lineage.
    scores = X * (weights * residual)[:, None]

    cluster_scores = np.zeros((G, K), dtype=float)

    np.add.at(
        cluster_scores,
        group_codes,
        scores
    )

    meat = cluster_scores.T @ cluster_scores
    bread = np.linalg.inv(xtx)

    # Standard finite-sample cluster correction.
    correction = (
        (G / (G - 1))
        * ((N - 1) / (N - K))
    )

    covariance = correction * bread @ meat @ bread
    ses = np.sqrt(
        np.maximum(np.diag(covariance), 0)
    )

    result = {
        "observations": int(N),
        "lineages": int(G),
        "equal_lineage_weight": equal_lineage_weight,
    }

    for i, predictor in enumerate(predictors):
        coefficient = float(beta[i])
        se = float(ses[i])

        z = coefficient / se if se > 0 else np.nan

        p_value = (
            float(2 * norm.sf(abs(z)))
            if np.isfinite(z)
            else None
        )

        result[predictor] = {
            "coefficient": coefficient,
            "cluster_robust_se": se,
            "ci_95_lower": coefficient - 1.96 * se,
            "ci_95_upper": coefficient + 1.96 * se,
            "p_value": p_value,
        }

    return result

# ============================================================
# 3. RUN MODEL SPECIFICATIONS
# ============================================================

results = []

def run_model(
    name,
    data,
    predictors,
    equal_weight=False
):
    print(f"\nRunning: {name}", flush=True)

    result = estimate_fe(
        data,
        predictors,
        equal_lineage_weight=equal_weight
    )

    if result is None:
        print("Insufficient identifying variation.")
        return

    time_result = result.get("log_elapsed_days", {})

    row = {
        "model": name,
        "observations": result["observations"],
        "lineages": result["lineages"],
        "beta_time": time_result.get("coefficient"),
        "se_time": time_result.get("cluster_robust_se"),
        "ci_lower": time_result.get("ci_95_lower"),
        "ci_upper": time_result.get("ci_95_upper"),
        "p_value_time": time_result.get("p_value"),
        "beta_order": result.get(
            "log_version_order", {}
        ).get("coefficient"),
        "se_order": result.get(
            "log_version_order", {}
        ).get("cluster_robust_se"),
    }

    results.append(row)

    print(
        f"Lineages: {row['lineages']:,} | "
        f"Observations: {row['observations']:,}"
    )
    print(
        f"Time coefficient: {row['beta_time']:.6f}"
    )
    print(
        f"95% CI: [{row['ci_lower']:.6f}, "
        f"{row['ci_upper']:.6f}]"
    )

    if row["beta_order"] is not None:
        print(
            f"Version-order coefficient: "
            f"{row['beta_order']:.6f}"
        )

# Model A: Original specification.
run_model(
    "A_original",
    later,
    ["log_elapsed_days"]
)

# Model B: Account for version order.
run_model(
    "B_adjusted_version_order",
    later,
    ["log_elapsed_days", "log_version_order"]
)

# Model C: Equal weight per lineage.
run_model(
    "C_equal_lineage_weight",
    later,
    ["log_elapsed_days"],
    equal_weight=True
)

# ============================================================
# 4. LINEAGE SIZE SUBGROUPS
# ============================================================

# Two-version families only have one later version,
# so a within-lineage time slope is not identifiable.
# Three-version families have only two later versions.
#
# These categories expose whether the overall result
# depends on families with many distinct versions.

size_groups = [
    ("D_3_to_4_versions", 3, 4),
    ("E_5_to_9_versions", 5, 9),
    ("F_10_plus_versions", 10, np.inf),
]

for name, low, high in size_groups:
    subset = later[
        (later["lineage_size"] >= low)
        & (later["lineage_size"] <= high)
    ]

    run_model(
        name,
        subset,
        ["log_elapsed_days"]
    )

# ============================================================
# 5. EXCLUDE EXTREME LINEAGES
# ============================================================

lineage_properties = (
    df.groupby("lineage_id")
    .agg(
        size=("version_order", "size"),
        max_elapsed_days=("days_since_first", "max")
    )
)

size_cutoff = float(
    lineage_properties["size"].quantile(0.99)
)

time_cutoff = float(
    lineage_properties["max_elapsed_days"].quantile(0.99)
)

nonextreme_ids = lineage_properties.index[
    (lineage_properties["size"] <= size_cutoff)
    & (
        lineage_properties["max_elapsed_days"]
        <= time_cutoff
    )
]

subset = later[
    later["lineage_id"].isin(nonextreme_ids)
]

run_model(
    "G_exclude_top_1pct_size_and_duration",
    subset,
    ["log_elapsed_days"]
)

# ============================================================
# 6. RESTRICT TO FIRST YEAR
# ============================================================

subset = later[
    later["days_since_first"] <= 365
]

run_model(
    "H_first_365_days",
    subset,
    ["log_elapsed_days"]
)

# ============================================================
# 7. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUT / "robustness_models.csv",
    index=False,
    encoding="utf-8-sig"
)

summary = {
    "source": str(SOURCE),
    "models": results,
    "lineage_size_99th_percentile": size_cutoff,
    "lineage_duration_99th_percentile_days": time_cutoff,
    "notes": [
        "All models include lineage fixed effects.",
        "The reference version is excluded.",
        "Standard errors are clustered by lineage.",
        "Two-version families cannot identify a within-lineage time slope.",
        "Version order and elapsed time may be highly correlated.",
        "Subgroup coefficients may differ in precision.",
        "Lexical thresholds restrict semantic variation.",
        "Results are observational, not causal."
    ],
    "status": "semantic_robustness_complete"
}

(OUT / "robustness_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("SEMANTIC ROBUSTNESS RESULTS")
print("=" * 70)

display_columns = [
    "model",
    "lineages",
    "observations",
    "beta_time",
    "ci_lower",
    "ci_upper",
    "beta_order",
]

print(
    results_df[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nSaved to:", OUT)
print("Analysis complete.")
