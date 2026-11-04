
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "anxiety_lineages"

OUT = BASE / "institutional_circulation"
OUT.mkdir(parents=True, exist_ok=True)

POSTS_FILE = BASE / "propagation_analysis" / "lineage_posts.parquet"
ALL_POSTS_FILE = BASE / "posts_with_exact_families.parquet"

print("=" * 70)
print("SCRIPT 40 — INSTITUTIONAL CIRCULATION AND ADAPTATION")
print("=" * 70)

posts = pd.read_parquet(POSTS_FILE)
all_posts = pd.read_parquet(
    ALL_POSTS_FILE,
    columns=["Facebook Id", "Page Category"]
)

required = [
    "lineage_id", "lineage_type", "exact_family_id",
    "page_key", "Page Category", "created_at"
]
for col in required:
    if col not in posts.columns:
        raise KeyError(f"Missing column: {col}")

# ============================================================
# 1. NORMALIZE PAGE METADATA
# ============================================================

def clean_category(s):
    x = s.astype("string").str.strip().str.upper()
    x = x.str.replace(r"\s+", "_", regex=True)
    return x.mask(x.isin(["", "NAN", "NONE", "UNKNOWN", "<NA>"]))

posts["category"] = clean_category(posts["Page Category"])

posts["created_at"] = pd.to_datetime(
    posts["created_at"], errors="coerce"
)

posts = posts.dropna(
    subset=["lineage_id", "page_key", "created_at"]
).copy()

# Total observed posts per page in the entire query corpus.
all_posts["page_key"] = (
    all_posts["Facebook Id"].astype("string").str.strip()
)

page_activity = (
    all_posts.groupby("page_key")
    .size()
    .rename("page_activity")
)

posts = posts.join(page_activity, on="page_key")

posts["page_activity"] = (
    posts["page_activity"].fillna(1).clip(lower=1)
)

# ============================================================
# 2. RAW CATEGORY PARTICIPATION
# ============================================================

print("\nCalculating category participation...", flush=True)

baseline = all_posts.copy()
baseline["category"] = clean_category(baseline["Page Category"])

baseline_counts = (
    baseline.groupby("category", dropna=False)
    .size()
    .rename("all_corpus_posts")
)

recurrent_counts = (
    posts.groupby("category", dropna=False)
    .size()
    .rename("recurrent_posts")
)

category_overview = pd.concat(
    [baseline_counts, recurrent_counts], axis=1
).fillna(0)

category_overview["recurrence_rate"] = (
    category_overview["recurrent_posts"]
    / category_overview["all_corpus_posts"].replace(0, np.nan)
)

category_overview["share_of_recurrent_posts"] = (
    category_overview["recurrent_posts"]
    / len(posts)
)

category_overview = (
    category_overview.reset_index()
    .sort_values("recurrent_posts", ascending=False)
)

category_overview.to_csv(
    OUT / "category_participation.csv",
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 3. ASSIGN EARLIEST OBSERVED TEXT PER LINEAGE
# ============================================================

multi = posts[
    posts["lineage_type"] == "multi_version"
].copy()

# Earliest observation of every distinct textual version.
version_first = (
    multi.groupby(
        ["lineage_id", "exact_family_id"], as_index=False
    )["created_at"]
    .min()
)

version_first = version_first.sort_values(
    ["lineage_id", "created_at", "exact_family_id"]
)

# Tie-breaking is deterministic, but simultaneous earliest
# observations are not evidence of a true original.
first_version = (
    version_first.drop_duplicates("lineage_id")
    [["lineage_id", "exact_family_id", "created_at"]]
    .rename(columns={
        "exact_family_id": "reference_family_id",
        "created_at": "lineage_start"
    })
)

multi = multi.merge(
    first_version,
    on="lineage_id",
    how="left",
    validate="many_to_one"
)

multi["modified_from_reference"] = (
    multi["exact_family_id"].astype("string")
    != multi["reference_family_id"].astype("string")
).astype(int)

multi["elapsed_days"] = (
    (multi["created_at"] - multi["lineage_start"])
    .dt.total_seconds() / 86400
).clip(lower=0)

multi["log_elapsed"] = np.log1p(multi["elapsed_days"])
multi["log_page_activity"] = np.log1p(
    multi["page_activity"]
)

# Count distinct versions in each lineage.
sizes = (
    multi.groupby("lineage_id")["exact_family_id"]
    .nunique()
    .rename("lineage_versions")
)

multi = multi.join(sizes, on="lineage_id")

# ============================================================
# 4. PAGE-LINEAGE FIRST APPEARANCES
# ============================================================

# Use the first observed post from each page within
# each lineage. This avoids treating a single page
# posting the same content repeatedly as independent
# institutional decisions.
events = (
    multi.sort_values(
        ["lineage_id", "page_key", "created_at", "exact_family_id"]
    )
    .drop_duplicates(["lineage_id", "page_key"])
    .copy()
)

events["category"] = events["category"].fillna("UNCLASSIFIED")

# Categories must be sufficiently common to compare.
category_counts = events["category"].value_counts()

MIN_EVENTS = 100

eligible_categories = category_counts[
    category_counts >= MIN_EVENTS
].index

# Keep top eight eligible categories. Remaining ones are
# grouped, rather than mixing hundreds of sparse dummies.
top_categories = category_counts[
    category_counts.index.isin(eligible_categories)
].head(8).index.tolist()

events["category_group"] = np.where(
    events["category"].isin(top_categories),
    events["category"],
    "OTHER_CATEGORIES"
)

# Category of the earliest observed post in each lineage.
earliest_events = (
    events.sort_values(
        ["lineage_id", "created_at", "page_key"]
    )
    .drop_duplicates("lineage_id")
)

first_page_summary = (
    earliest_events["category_group"]
    .value_counts()
    .rename_axis("category")
    .reset_index(name="earliest_observed_lineages")
)

first_page_summary.to_csv(
    OUT / "earliest_observed_page_categories.csv",
    index=False,
    encoding="utf-8-sig"
)

institutional = (
    events.groupby("category_group")
    .agg(
        page_lineage_events=("lineage_id", "size"),
        distinct_lineages=("lineage_id", "nunique"),
        distinct_pages=("page_key", "nunique"),
        modified_events=("modified_from_reference", "sum"),
        modification_rate=("modified_from_reference", "mean"),
        median_elapsed_days=("elapsed_days", "median"),
        median_page_activity=("page_activity", "median")
    )
    .reset_index()
    .sort_values(
        "page_lineage_events", ascending=False
    )
)

institutional.to_csv(
    OUT / "institutional_adaptation.csv",
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 5. CROSS-CATEGORY CIRCULATION
# ============================================================

category_reach = (
    events.groupby("lineage_id")
    .agg(
        category_count=("category_group", "nunique"),
        pages=("page_key", "nunique"),
        versions=("lineage_versions", "first")
    )
    .reset_index()
)

category_reach.to_csv(
    OUT / "lineage_institutional_reach.csv",
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 6. WITHIN-LINEAGE CATEGORY COMPARISON
# ============================================================

# Linear probability model with:
# - lineage fixed effects
# - category indicators
# - log elapsed time
# - log page activity
#
# Dependent variable: whether the page's FIRST observed
# post in the lineage differs from the lineage's earliest
# observed textual version.
#
# Coefficients are associations (percentage-point changes),
# not causal estimates.

print("\nEstimating within-lineage model...", flush=True)

# Use the most frequent category as the reference.
reference = institutional.iloc[0]["category_group"]

categories = [
    c for c in institutional["category_group"].tolist()
    if c != reference
]

model = events[
    ["lineage_id", "modified_from_reference",
     "category_group", "log_elapsed",
     "log_page_activity"]
].copy()

model = model.dropna()

predictors = []

for i, category in enumerate(categories):
    col = f"cat_{i}"
    model[col] = (
        model["category_group"] == category
    ).astype(float)
    predictors.append(col)

predictors += ["log_elapsed", "log_page_activity"]

# Only lineages with at least two page observations
# can contribute to within-lineage estimation.
counts = model.groupby("lineage_id")[
    "lineage_id"
].transform("size")

model = model[counts >= 2].copy()

group = model.groupby("lineage_id", sort=False)

y = (
    model["modified_from_reference"]
    - group["modified_from_reference"].transform("mean")
).to_numpy(dtype=float)

X = np.column_stack([
    (
        model[p] - group[p].transform("mean")
    ).to_numpy(dtype=float)
    for p in predictors
])

# Drop predictors with no usable within-lineage variation.
active = (np.abs(X).sum(axis=0) > 1e-10)
active_predictors = [
    p for p, a in zip(predictors, active) if a
]
X = X[:, active]

# Pseudoinverse tolerates perfect collinearity, but
# coefficients are not uniquely identified in that case.
rank = int(np.linalg.matrix_rank(X))
full_rank = rank == X.shape[1]

if len(model) and full_rank and X.shape[1] > 0:
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ beta

    codes, labels = pd.factorize(
        model["lineage_id"], sort=False
    )

    G = len(labels)
    N, K = X.shape

    cluster_scores = np.zeros(
        (G, K), dtype=float
    )
    np.add.at(
        cluster_scores,
        codes,
        X * residual[:, None]
    )

    bread = np.linalg.inv(X.T @ X)
    meat = cluster_scores.T @ cluster_scores

    correction = (
        (G / (G - 1))
        * ((N - 1) / (N - K))
        if G > 1 and N > K else 1.0
    )

    covariance = (
        correction * bread @ meat @ bread
    )

    se = np.sqrt(
        np.maximum(np.diag(covariance), 0)
    )

    coefficient_rows = []

    for j, predictor in enumerate(active_predictors):
        b = float(beta[j])
        s = float(se[j])

        original_category = predictor

        if predictor.startswith("cat_"):
            original_category = categories[
                int(predictor.split("_")[1])
            ]

        coefficient_rows.append({
            "predictor": original_category,
            "coefficient": b,
            "standard_error": s,
            "ci_lower": b - 1.96 * s,
            "ci_upper": b + 1.96 * s,
            "p_value": (
                float(2 * norm.sf(abs(b / s)))
                if s > 0 else np.nan
            )
        })

    coefficients = pd.DataFrame(coefficient_rows)

    coefficients.to_csv(
        OUT / "within_lineage_category_model.csv",
        index=False,
        encoding="utf-8-sig"
    )

else:
    G = model["lineage_id"].nunique()
    N = len(model)
    coefficients = pd.DataFrame()
    print(
        "Model not estimable with full-rank predictors. "
        "Descriptive results were saved."
    )

# ============================================================
# 7. SUMMARY
# ============================================================

summary = {
    "recurrent_posts": int(len(posts)),
    "multi_version_posts": int(len(multi)),
    "multi_version_lineages": int(multi["lineage_id"].nunique()),
    "unique_page_lineage_events": int(len(events)),
    "categories_in_model": top_categories,
    "reference_category": reference,
    "cross_category_lineages": int(
        (category_reach["category_count"] >= 2).sum()
    ),
    "cross_category_share": float(
        (category_reach["category_count"] >= 2).mean()
    ),
    "regression_observations": int(N),
    "regression_lineages": int(G),
    "regression_full_rank": bool(full_rank),
    "methodology": (
        "First observed post per page-lineage, "
        "lineage fixed-effects linear probability model, "
        "clustered standard errors by lineage."
    ),
    "limitations": [
        "Earliest observed page is not necessarily the origin.",
        "Modification is relative to the earliest observed text.",
        "Page Category is an administrative classification.",
        "Only multi-version lineages enter modification analysis.",
        "Page activity is measured within the collected corpus.",
        "Category comparisons are observational, not causal.",
        "OTHER_CATEGORIES is heterogeneous.",
        "Results depend on lexical lineage construction."
    ],
    "status": "institutional_analysis_complete"
}

(OUT / "institutional_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("INSTITUTIONAL CIRCULATION RESULTS")
print("=" * 70)

for key, value in summary.items():
    if key != "limitations":
        print(f"{key}: {value}")

print("\nINSTITUTIONAL ADAPTATION")
print(institutional.to_string(index=False))

print("\nWITHIN-LINEAGE CATEGORY MODEL")
if not coefficients.empty:
    print(coefficients.to_string(index=False))
else:
    print("Model not estimated; see descriptive outputs.")

print("\nResults saved to:", OUT)
print("Analysis complete.")
