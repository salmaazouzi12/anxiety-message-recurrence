
from pathlib import Path
import hashlib
import re
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "anxiety_lineages"
OUT.mkdir(parents=True, exist_ok=True)

# Optional: set this to the original full-corpus CSV or
# Parquet path if automatic discovery picks the wrong file.
INPUT = None

def read_dataset(path):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(
            path, encoding="utf-8-sig",
            low_memory=False
        )
    except UnicodeDecodeError:
        raise ValueError(
            f"Encoding problem in {path}. "
            "Use a correctly encoded original source."
        )

def find_source():
    if INPUT:
        return Path(INPUT)

    candidates = []

    for path in DATA.rglob("*"):
        if path.suffix.lower() not in (".csv", ".parquet"):
            continue

        try:
            if path.suffix.lower() == ".parquet":
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(path)
                columns = pf.schema.names
                n = pf.metadata.num_rows
            else:
                header = pd.read_csv(
                    path, nrows=0, encoding="utf-8-sig"
                )
                columns = header.columns.tolist()
                # Count physical records after loading below
                n = -1

            if "Message" not in columns:
                continue

            if "URL" not in columns:
                continue

            candidates.append((path, n))

        except Exception:
            continue

    if not candidates:
        raise FileNotFoundError(
            "No CSV/Parquet with Message and URL found in data/"
        )

    # Prefer a post-level Parquet with the most records.
    parquet = [
        c for c in candidates if c[1] >= 0
    ]

    if parquet:
        return max(parquet, key=lambda x: x[1])[0]

    # With CSV-only sources, report candidates rather than
    # guessing which file represents the original corpus.
    raise ValueError(
        "Set INPUT to the original post-level CSV:\n"
        + "\n".join(str(p) for p, _ in candidates)
    )

def normalize(text):
    if pd.isna(text):
        return ""

    text = unicodedata.normalize("NFKC", str(text))

    text = re.sub(
        r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]",
        "",
        text
    )

    return " ".join(text.split()).strip()

def text_id(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

source = find_source()

print("\nSelected source:", source)

df = read_dataset(source)

print("Original rows:", len(df))
print("Available columns:", df.columns.tolist())

# Do not substitute an already deduplicated file for
# post-level circulation analysis.
if "unique_texts" in source.stem.lower():
    raise ValueError(
        "Selected unique_texts, which is deduplicated. "
        "Set INPUT to the full original post-level corpus."
    )

assert "Message" in df.columns
assert "URL" in df.columns

df = df.dropna(subset=["Message"]).copy()

df["normalized_text"] = (
    df["Message"].map(normalize)
)

df = df[
    df["normalized_text"].str.len() > 0
].copy()

df["exact_family_id"] = (
    df["normalized_text"].map(text_id)
)

# Keep every original post.
df.to_parquet(
    OUT / "posts_with_exact_families.parquet",
    index=False
)

# Unique text versions and their repetition counts.
families = (
    df.groupby("exact_family_id")
    .agg(
        normalized_text=("normalized_text", "first"),
        post_count=("exact_family_id", "size"),
        unique_urls=("URL", "nunique")
    )
    .reset_index()
)

families.to_parquet(
    OUT / "unique_text_families.parquet",
    index=False
)

repeated = families[
    families["post_count"] >= 2
]

repeated_ids = set(repeated["exact_family_id"])

posts_in_repeated = int(
    df["exact_family_id"].isin(repeated_ids).sum()
)

print("\n" + "=" * 60)
print("EXACT TEXT RECURRENCE — ANXIETY CORPUS")
print("=" * 60)

print("Posts analyzed:", len(df))
print("Unique text versions:", len(families))
print("Repeated exact-text families:", len(repeated))
print("Posts in repeated families:", posts_in_repeated)

print(
    "Exact repetition coverage:",
    f"{posts_in_repeated / len(df):.2%}"
)

print(
    "Largest exact family:",
    int(families["post_count"].max())
)

print("\nOutputs saved to:", OUT)
