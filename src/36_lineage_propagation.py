
from pathlib import Path
import json

import numpy as np
import pandas as pd

# ============================================================
# SCRIPT 36 — ANXIETY MESSAGE PROPAGATION ANALYSIS
# Corrected complete version
#
# Uses:
#   - Repaired combined lineages
#   - Original Facebook post metadata
#   - Exact-only recurring message families
#
# Does NOT rerun embeddings, MinHash, FAISS, or clustering.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "anxiety_lineages"
OUT = BASE / "propagation_analysis"
OUT.mkdir(parents=True, exist_ok=True)

POSTS_PATH = BASE / "posts_with_exact_families.parquet"

MEMBERS_PATH = (
    BASE
    / "combined_cohesion_repair"
    / "repaired_lineage_members.parquet"
)

FAMILIES_PATH = BASE / "unique_text_families.parquet"

print("=" * 65)
print("ANXIETY MESSAGE PROPAGATION ANALYSIS")
print("=" * 65)

# ============================================================
# 1. LOAD SAVED DATA
# ============================================================

posts = pd.read_parquet(POSTS_PATH)
members = pd.read_parquet(MEMBERS_PATH)
families = pd.read_parquet(FAMILIES_PATH)

print(f"\nOriginal nonempty posts: {len(posts):,}")
print(f"Repaired lineage members: {len(members):,}")
print(f"Unique exact-text families: {len(families):,}")

required_posts = [
    "exact_family_id",
    "Facebook Id",
    "Page Admin Top Country",
    "Post Created",
    "Post Created Date",
]

required_members = [
    "exact_family_id",
    "repaired_lineage_id",
]

for col in required_posts:
    if col not in posts.columns:
        raise KeyError(f"Missing post column: {col}")

for col in required_members:
    if col not in members.columns:
        raise KeyError(f"Missing member column: {col}")

assert posts["exact_family_id"].notna().all()
assert members["exact_family_id"].is_unique

# ============================================================
# 2. CONNECT REPAIRED LINEAGES TO ORIGINAL POSTS
# ============================================================

lookup = members[
    ["exact_family_id", "repaired_lineage_id"]
].copy()

posts = posts.merge(
    lookup,
    on="exact_family_id",
    how="left",
    validate="many_to_one"
)

posts["lineage_type"] = np.where(
    posts["repaired_lineage_id"].notna(),
    "multi_version",
    "not_multi_version"
)

# ============================================================
# 3. INCLUDE EXACT-ONLY REPEATED FAMILIES
# ============================================================

# Calculate recurrence directly from the original posts
# to avoid assumptions about the stored family counts.

exact_counts = posts["exact_family_id"].value_counts()

posts["exact_family_posts"] = (
    posts["exact_family_id"].map(exact_counts)
)

exact_only = (
    posts["repaired_lineage_id"].isna()
    & posts["exact_family_posts"].ge(2)
)

posts.loc[exact_only, "lineage_type"] = "exact_only"

# Unique lineage IDs:
# M_ = reconstructed multi-version lineage
# E_ = exact-only recurring text family

posts["lineage_id"] = pd.Series(
    pd.NA,
    index=posts.index,
    dtype="string"
)

multi = posts["repaired_lineage_id"].notna()

posts.loc[multi, "lineage_id"] = (
    "M_"
    + posts.loc[multi, "repaired_lineage_id"]
    .astype("int64")
    .astype("string")
)

posts.loc[exact_only, "lineage_id"] = (
    "E_"
    + posts.loc[exact_only, "exact_family_id"]
    .astype("string")
)

# ============================================================
# 4. FIX TIMESTAMPS
# ============================================================

print("\n" + "=" * 65)
print("TIMESTAMP VALIDATION")
print("=" * 65)

raw_timestamp = (
    posts["Post Created"]
    .astype("string")
    .str.strip()
)

# Preserve original timezone label for auditing.
posts["source_timezone_label"] = (
    raw_timestamp.str.extract(
        r"\s+([A-Za-z]{2,5})$",
        expand=False
    )
)

# Strip the timezone abbreviation.
# AST is ambiguous globally, so we do not invent a UTC
# offset. The source clock time is retained.
timestamp_without_zone = (
    raw_timestamp.str.replace(
        r"\s+[A-Za-z]{2,5}$",
        "",
        regex=True
    )
)

full_timestamps = pd.to_datetime(
    timestamp_without_zone,
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce"
)

fallback_dates = pd.to_datetime(
    posts["Post Created Date"].astype("string"),
    format="%Y-%m-%d",
    errors="coerce"
)

posts["created_at"] = (
    full_timestamps.fillna(fallback_dates)
)

print(
    "Full timestamps parsed:",
    f"{full_timestamps.notna().sum():,}"
)

print(
    "Fallback dates used:",
    f"{(full_timestamps.isna() & fallback_dates.notna()).sum():,}"
)

print(
    "Missing timestamps:",
    f"{posts['created_at'].isna().sum():,}"
)

print("\nTimezone labels:")
print(
    posts["source_timezone_label"]
    .value_counts(dropna=False)
    .to_string()
)

print("Earliest:", posts["created_at"].min())
print("Latest:", posts["created_at"].max())

# ============================================================
# 5. RESTORE PAGE AND COUNTRY IDENTIFIERS
# ============================================================

print("\n" + "=" * 65)
print("PAGE AND COUNTRY METADATA")
print("=" * 65)

# Script 37 verified that Facebook Id behaves as a
# stable page identifier, not as a unique post ID.

posts["page_key"] = (
    posts["Facebook Id"]
    .astype("string")
    .str.strip()
)

posts["page_key"] = posts["page_key"].mask(
    posts["page_key"].isin(
        ["", "nan", "None", "<NA>"]
    )
)

# Country is the page-admin country, not the
# location of the person reading the post.

posts["country_key"] = (
    posts["Page Admin Top Country"]
    .astype("string")
    .str.strip()
    .str.upper()
)

posts["country_key"] = posts["country_key"].mask(
    posts["country_key"].isin(
        ["", "NAN", "NONE", "UNKNOWN", "<NA>"]
    )
)

print(
    "Posts with page identifiers:",
    f"{posts['page_key'].notna().sum():,}"
)

print(
    "Distinct pages:",
    f"{posts['page_key'].nunique():,}"
)

print(
    "Posts with country metadata:",
    f"{posts['country_key'].notna().sum():,}"
)

print(
    "Distinct recorded countries:",
    f"{posts['country_key'].nunique():,}"
)

# ============================================================
# 6. CLEAN ENGAGEMENT VARIABLES
# ============================================================

for col in ["Total Interactions", "Shares"]:
    if col not in posts.columns:
        posts[col] = np.nan

    posts[col] = pd.to_numeric(
        posts[col],
        errors="coerce"
    )

# ============================================================
# 7. KEEP POSTS PARTICIPATING IN RECURRENCE
# ============================================================

recurrent = posts[
    posts["lineage_id"].notna()
].copy()

print("\n" + "=" * 65)
print("RECURRENCE COVERAGE")
print("=" * 65)

print(f"Total nonempty posts: {len(posts):,}")
print(f"Recurrent posts: {len(recurrent):,}")

print(
    "Recurrence coverage:",
    f"{len(recurrent) / len(posts):.2%}"
)

# ============================================================
# 8. BUILD LINEAGE INVENTORY
# ============================================================

print("\nBuilding lineage inventory...", flush=True)

inventory = (
    recurrent.groupby("lineage_id", sort=False)
    .agg(
        lineage_type=("lineage_type", "first"),
        posts=("lineage_id", "size"),
        unique_versions=("exact_family_id", "nunique"),
        pages=("page_key", "nunique"),
        countries=("country_key", "nunique"),
        first_observed=("created_at", "min"),
        last_observed=("created_at", "max"),
        total_interactions=("Total Interactions", "sum"),
        total_shares=("Shares", "sum")
    )
    .reset_index()
)

inventory["lifespan_days"] = (
    (
        inventory["last_observed"]
        - inventory["first_observed"]
    ).dt.total_seconds()
    / 86400
)

# ============================================================
# 9. TEMPORAL MILESTONES
# ============================================================

# Measure time between first observed appearances of
# distinct pages or distinct textual versions.
#
# These are observational intervals, NOT verified
# Facebook sharing or influence paths.

print("\nCalculating temporal milestones...", flush=True)

def milestone_table(df, identity_col, prefix):
    """
    Return hours from the earliest observed event in a
    lineage to the 2nd, 5th, and 10th distinct identity.

    For pages:
        identity_col = page_key

    For text versions:
        identity_col = exact_family_id
    """

    valid = df.dropna(
        subset=["lineage_id", "created_at", identity_col]
    )

    # Earliest recorded appearance of each distinct
    # page or text version within each lineage.
    first_seen = (
        valid.groupby(
            ["lineage_id", identity_col],
            sort=False
        )["created_at"]
        .min()
        .reset_index()
    )

    first_seen = first_seen.sort_values(
        ["lineage_id", "created_at", identity_col]
    )

    first_seen["rank"] = (
        first_seen.groupby("lineage_id").cumcount() + 1
    )

    # The baseline is the earliest observed appearance
    # within the eligible set of this particular measure.
    first_times = (
        first_seen.groupby("lineage_id")["created_at"]
        .min()
        .rename("_first")
    )

    selected = first_seen[
        first_seen["rank"].isin([2, 5, 10])
    ].copy()

    selected = selected.join(
        first_times,
        on="lineage_id"
    )

    selected["hours"] = (
        (
            selected["created_at"]
            - selected["_first"]
        ).dt.total_seconds()
        / 3600
    )

    wide = selected.pivot(
        index="lineage_id",
        columns="rank",
        values="hours"
    )

    wide = wide.rename(
        columns={
            2: f"hours_to_2_{prefix}",
            5: f"hours_to_5_{prefix}",
            10: f"hours_to_10_{prefix}",
        }
    )

    expected_columns = [
        f"hours_to_2_{prefix}",
        f"hours_to_5_{prefix}",
        f"hours_to_10_{prefix}",
    ]

    for col in expected_columns:
        if col not in wide.columns:
            wide[col] = np.nan

    return wide[
        expected_columns
    ].reset_index()

page_milestones = milestone_table(
    recurrent,
    "page_key",
    "pages"
)

version_milestones = milestone_table(
    recurrent,
    "exact_family_id",
    "versions"
)

inventory = inventory.merge(
    page_milestones,
    on="lineage_id",
    how="left",
    validate="one_to_one"
)

inventory = inventory.merge(
    version_milestones,
    on="lineage_id",
    how="left",
    validate="one_to_one"
)

# ============================================================
# 10. SUMMARY STATISTICS
# ============================================================

def median_nonnull(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.median()) if len(s) else None

multi_inventory = inventory[
    inventory["lineage_type"] == "multi_version"
]

exact_inventory = inventory[
    inventory["lineage_type"] == "exact_only"
]

summary = {
    "total_posts": int(len(posts)),
    "recurrent_posts": int(len(recurrent)),
    "recurrence_coverage": float(
        len(recurrent) / len(posts)
    ),

    "total_lineages": int(len(inventory)),
    "multi_version_lineages": int(len(multi_inventory)),
    "exact_only_lineages": int(len(exact_inventory)),

    "multi_version_posts": int(
        recurrent["lineage_type"].eq("multi_version").sum()
    ),
    "exact_only_posts": int(
        recurrent["lineage_type"].eq("exact_only").sum()
    ),

    "cross_page_lineages": int(
        inventory["pages"].ge(2).sum()
    ),
    "cross_page_share": float(
        inventory["pages"].ge(2).mean()
    ),

    "cross_country_lineages": int(
        inventory["countries"].ge(2).sum()
    ),
    "cross_country_share": float(
        inventory["countries"].ge(2).mean()
    ),

    "median_posts_per_lineage": median_nonnull(
        inventory["posts"]
    ),
    "median_versions_per_lineage": median_nonnull(
        inventory["unique_versions"]
    ),
    "median_lifespan_days": median_nonnull(
        inventory["lifespan_days"]
    ),

    "median_hours_to_2_pages": median_nonnull(
        inventory["hours_to_2_pages"]
    ),
    "median_hours_to_2_versions": median_nonnull(
        inventory["hours_to_2_versions"]
    ),

    "lineages_reaching_2_pages": int(
        inventory["hours_to_2_pages"].notna().sum()
    ),
    "lineages_reaching_2_versions": int(
        inventory["hours_to_2_versions"].notna().sum()
    ),

    "full_timestamps_parsed": int(
        full_timestamps.notna().sum()
    ),
    "missing_timestamps": int(
        posts["created_at"].isna().sum()
    ),

    "status": "propagation_metadata_corrected",
    "timestamp_note": (
        "Source clock times parsed; trailing AST label "
        "preserved but not converted to UTC."
    )
}

# ============================================================
# 11. SAVE RESULTS
# ============================================================

recurrent.to_parquet(
    OUT / "lineage_posts.parquet",
    index=False
)

inventory.to_parquet(
    OUT / "lineage_inventory.parquet",
    index=False
)

inventory.to_csv(
    OUT / "lineage_inventory.csv",
    index=False,
    encoding="utf-8-sig"
)

with open(
    OUT / "propagation_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

# ============================================================
# 12. REPORT RESULTS
# ============================================================

print("\n" + "=" * 65)
print("ANXIETY MESSAGE PROPAGATION RESULTS — CORRECTED")
print("=" * 65)

for key, value in summary.items():
    if isinstance(value, float):
        print(f"{key}: {value:.6f}")
    else:
        print(f"{key}: {value}")

print("\nResults saved to:")
print(OUT)

print("\nAnalysis complete.")
