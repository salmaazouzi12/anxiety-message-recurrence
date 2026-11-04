
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "anxiety_lineages"
OUT = BASE / "quality_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

POSTS_FILE = (
    BASE / "propagation_analysis"
    / "lineage_posts.parquet"
)
VERSIONS_FILE = (
    BASE / "semantic_transformation"
    / "semantic_version_trajectories.parquet"
)
MAPPING_FILE = (
    BASE / "minhash"
    / "text_id_mapping.parquet"
)

SEED = 42
PREVIEW_CHARS = 700

print("=" * 70)
print("SCRIPT 42 — LINEAGE QUALITY & TOPICAL RELEVANCE")
print("=" * 70)

# ============================================================
# 1. LOAD EXISTING RESULTS
# ============================================================

posts = pd.read_parquet(POSTS_FILE)
versions = pd.read_parquet(VERSIONS_FILE)
mapping = pd.read_parquet(
    MAPPING_FILE,
    columns=["text_id", "exact_family_id", "normalized_text"]
)

required = [
    "lineage_id",
    "text_id",
    "version_order",
    "semantic_distance_to_first",
    "days_since_first"
]

for col in required:
    if col not in versions.columns:
        raise KeyError(f"Missing version column: {col}")

if not {"lineage_id", "Message", "exact_family_id"}.issubset(
    posts.columns
):
    raise KeyError("Missing required post columns.")

print(f"Recurrent posts: {len(posts):,}")
print(f"Multi-version texts: {len(versions):,}")
print(f"Unique texts in mapping: {len(mapping):,}")

assert mapping["text_id"].is_unique
assert mapping["exact_family_id"].is_unique

# ============================================================
# 2. EXPLICIT ANXIETY-TOPIC SCREENING
# ============================================================

# This is deliberately NOT a relevance classifier.
# It detects explicit words/phrases in selected languages.
#
# A positive match may be incidental, promotional,
# metaphorical, or refer to animals.
# A negative match does not imply irrelevance.

CUE_PATTERNS = {
    "english": r"\b(?:anxiety|anxious|panic attacks?|"
               r"anxiety disorders?|social anxiety)\b",

    "french": r"\b(?:anxi[eé]t[eé]|anxieux|anxieuse|"
              r"angoisse|angoiss[eé]|crise[s]? d.angoisse)\b",

    "arabic": r"(?:القلق|قلق|اضطراب القلق|"
              r"نوبة هلع|نوبات الهلع|توتر نفسي)",

    "spanish": r"\b(?:ansiedad|ansioso|ansiosa|"
               r"ataques? de p[aá]nico)\b",

    "portuguese": r"\b(?:ansiedade|ansioso|ansiosa|"
                  r"ataques? de p[aâ]nico)\b",
}

message = posts["Message"].fillna("").astype("string")

cue_columns = []

print("\nScreening explicit anxiety terminology...", flush=True)

for language, pattern in CUE_PATTERNS.items():
    col = f"cue_{language}"

    posts[col] = message.str.contains(
        pattern,
        case=False,
        regex=True,
        na=False
    ).astype(bool)

    cue_columns.append(col)

posts["has_explicit_anxiety_cue"] = (
    posts[cue_columns].any(axis=1)
)

# Potentially incidental animal-related terminology.
# This is a review flag, NOT an exclusion criterion.
animal_pattern = (
    r"\b(?:dog|dogs|puppy|puppies|cat|cats|"
    r"pet|pets|animal|animals|shelter)\b"
)

posts["animal_context_flag"] = (
    message.str.contains(
        animal_pattern,
        case=False,
        regex=True,
        na=False
    )
)

# Promotional language may also appear in relevant posts.
commercial_pattern = (
    r"\b(?:buy now|limited offer|free trial|"
    r"discount|promo code|sale ends)\b"
)

posts["commercial_context_flag"] = (
    message.str.contains(
        commercial_pattern,
        case=False,
        regex=True,
        na=False
    )
)

# ============================================================
# 3. POST-LEVEL SCREENING STATISTICS
# ============================================================

screening_summary = {
    "recurrent_posts": int(len(posts)),
    "explicit_cue_posts": int(
        posts["has_explicit_anxiety_cue"].sum()
    ),
    "explicit_cue_share": float(
        posts["has_explicit_anxiety_cue"].mean()
    ),
    "no_explicit_cue_posts": int(
        (~posts["has_explicit_anxiety_cue"]).sum()
    ),
    "animal_context_posts": int(
        posts["animal_context_flag"].sum()
    ),
    "commercial_context_posts": int(
        posts["commercial_context_flag"].sum()
    )
}

for language in CUE_PATTERNS:
    screening_summary[f"cue_{language}_posts"] = int(
        posts[f"cue_{language}"].sum()
    )

# These are indicators, not disjoint language categories.

# ============================================================
# 4. MAP FULL TEXTS TO VERSION TRAJECTORIES
# ============================================================

print("Preparing lineage cohesion diagnostics...", flush=True)

versions = versions.merge(
    mapping[["text_id", "normalized_text"]],
    on="text_id",
    how="left",
    validate="many_to_one"
)

if versions["normalized_text"].isna().any():
    raise ValueError(
        "Some versions could not be linked to text."
    )

versions = versions.sort_values(
    ["lineage_id", "version_order", "text_id"]
).reset_index(drop=True)

# Word-set overlap is a fast, supplementary measure.
# It is NOT the char 3–5 gram cosine used to construct
# the original lexical graph.

def word_jaccard(a, b):
    aa = set(str(a).casefold().split())
    bb = set(str(b).casefold().split())

    if not aa and not bb:
        return 1.0

    return len(aa & bb) / max(len(aa | bb), 1)

# Earliest observed text in each lineage.
reference = (
    versions.drop_duplicates("lineage_id")
    [["lineage_id", "normalized_text"]]
    .rename(
        columns={"normalized_text": "reference_text"}
    )
)

versions = versions.merge(
    reference,
    on="lineage_id",
    validate="many_to_one"
)

# Calculate overlap only for the most semantically
# distant version of each lineage, rather than
# unnecessarily comparing every pair.

farthest = (
    versions.sort_values(
        [
            "lineage_id",
            "semantic_distance_to_first",
            "version_order",
            "text_id"
        ],
        ascending=[True, False, False, True]
    )
    .drop_duplicates("lineage_id")
    .copy()
)

farthest["word_overlap_to_reference"] = [
    word_jaccard(a, b)
    for a, b in zip(
        farthest["reference_text"],
        farthest["normalized_text"]
    )
]

# ============================================================
# 5. LINEAGE-LEVEL QUALITY INDICATORS
# ============================================================

lineage = (
    versions.groupby("lineage_id")
    .agg(
        unique_versions=("text_id", "size"),
        max_semantic_distance=(
            "semantic_distance_to_first", "max"
        ),
        median_semantic_distance=(
            "semantic_distance_to_first", "median"
        ),
        duration_days=("days_since_first", "max")
    )
    .reset_index()
)

lineage = lineage.merge(
    farthest[
        [
            "lineage_id",
            "text_id",
            "normalized_text",
            "reference_text",
            "word_overlap_to_reference"
        ]
    ].rename(
        columns={
            "text_id": "farthest_text_id",
            "normalized_text": "farthest_text"
        }
    ),
    on="lineage_id",
    validate="one_to_one"
)

# Diagnostic cutoffs chosen for prioritizing inspection.
# They are NOT validated ground-truth boundaries.

lineage["high_semantic_distance_flag"] = (
    lineage["max_semantic_distance"] >= 0.08
)

lineage["low_word_overlap_flag"] = (
    lineage["word_overlap_to_reference"] < 0.45
)

lineage["cohesion_review_flag"] = (
    lineage["high_semantic_distance_flag"]
    | lineage["low_word_overlap_flag"]
)

# Mark lineages with evidence of both high distance
# and low lexical overlap as highest priority.

lineage["high_priority_review"] = (
    lineage["high_semantic_distance_flag"]
    & lineage["low_word_overlap_flag"]
)

# ============================================================
# 6. ADD TOPICAL SCREENING AT LINEAGE LEVEL
# ============================================================

topic_by_lineage = (
    posts.groupby("lineage_id")
    .agg(
        posts=("lineage_id", "size"),
        pages=("page_key", "nunique"),
        explicit_cue_posts=(
            "has_explicit_anxiety_cue", "sum"
        ),
        animal_context_posts=(
            "animal_context_flag", "sum"
        ),
        commercial_context_posts=(
            "commercial_context_flag", "sum"
        )
    )
    .reset_index()
)

topic_by_lineage["explicit_cue_share"] = (
    topic_by_lineage["explicit_cue_posts"]
    / topic_by_lineage["posts"]
)

lineage = lineage.merge(
    topic_by_lineage,
    on="lineage_id",
    validate="one_to_one"
)

lineage["no_explicit_cue_flag"] = (
    lineage["explicit_cue_posts"] == 0
)

# ============================================================
# 7. SELECT EXAMPLES FOR INSPECTION
# ============================================================

# Prioritize doubtful lineages, while also sampling
# ordinary ones to avoid focusing only on extremes.

flagged = lineage[
    lineage["cohesion_review_flag"]
].copy()

flagged = flagged.sort_values(
    [
        "high_priority_review",
        "max_semantic_distance",
        "unique_versions"
    ],
    ascending=[False, False, False]
)

selected_flagged = flagged.head(80).copy()
selected_flagged["selection_reason"] = "cohesion_review"

no_cue = lineage[
    lineage["no_explicit_cue_flag"]
].copy()

selected_no_cue = (
    no_cue.sample(
        n=min(40, len(no_cue)),
        random_state=SEED
    )
    if len(no_cue)
    else no_cue
)

selected_no_cue["selection_reason"] = (
    "no_explicit_anxiety_cue"
)

ordinary = lineage[
    ~lineage["cohesion_review_flag"]
    & ~lineage["no_explicit_cue_flag"]
].copy()

selected_ordinary = (
    ordinary.sample(
        n=min(40, len(ordinary)),
        random_state=SEED
    )
    if len(ordinary)
    else ordinary
)

selected_ordinary["selection_reason"] = (
    "ordinary_comparison"
)

selected = pd.concat(
    [
        selected_flagged,
        selected_no_cue,
        selected_ordinary
    ],
    ignore_index=True
)

# Retain every reason for duplicate selections.
reasons = (
    selected.groupby("lineage_id")[
        "selection_reason"
    ]
    .agg(lambda s: "; ".join(sorted(set(s))))
    .reset_index()
)

selected = (
    selected.drop(columns=["selection_reason"])
    .drop_duplicates("lineage_id")
    .merge(reasons, on="lineage_id")
)

def preview(value):
    return " ".join(
        str(value).split()
    )[:PREVIEW_CHARS]

# Full text is retained separately in the Parquet.
selected.to_parquet(
    OUT / "diagnostic_examples_full.parquet",
    index=False
)

readable = selected.copy()

for col in ["reference_text", "farthest_text"]:
    readable[col] = readable[col].map(preview)

readable.to_csv(
    OUT / "diagnostic_examples.csv",
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 8. SAVE COMPLETE DIAGNOSTIC INVENTORY
# ============================================================

# Avoid duplicating full text in the complete inventory.
lineage_metadata = lineage.drop(
    columns=["reference_text", "farthest_text"]
)

lineage_metadata.to_parquet(
    OUT / "lineage_quality_inventory.parquet",
    index=False
)

lineage_metadata.to_csv(
    OUT / "lineage_quality_inventory.csv",
    index=False,
    encoding="utf-8-sig"
)

summary = {
    "recurrent_post_screening": screening_summary,
    "multi_version_lineages": int(len(lineage)),
    "high_semantic_distance_lineages": int(
        lineage["high_semantic_distance_flag"].sum()
    ),
    "low_word_overlap_lineages": int(
        lineage["low_word_overlap_flag"].sum()
    ),
    "cohesion_review_lineages": int(
        lineage["cohesion_review_flag"].sum()
    ),
    "high_priority_review_lineages": int(
        lineage["high_priority_review"].sum()
    ),
    "lineages_without_explicit_anxiety_cue": int(
        lineage["no_explicit_cue_flag"].sum()
    ),
    "sampled_diagnostic_lineages": int(len(selected)),
    "cautions": [
        "Explicit keyword screening is not a relevance classifier.",
        "Keyword absence is not proof of topical irrelevance.",
        "Keyword presence is not proof of substantive relevance.",
        "Only selected languages and phrases are screened.",
        "Semantic distance and Jaccard cutoffs are exploratory.",
        "High distance is not necessarily a lineage error.",
        "Topical statistics cover recurrent posts only.",
        "No reconstructed lineages are changed by this script."
    ],
    "status": "quality_diagnostics_complete"
}

(OUT / "quality_diagnostic_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

# ============================================================
# 9. REPORT
# ============================================================

print("\n" + "=" * 70)
print("QUALITY DIAGNOSTICS COMPLETE")
print("=" * 70)

for key, value in summary.items():
    if key == "recurrent_post_screening":
        print("\nPOST-LEVEL SCREENING")
        for k, v in value.items():
            print(f"{k}: {v}")
    elif key != "cautions":
        print(f"{key}: {value}")

print("\nResults saved to:", OUT)
