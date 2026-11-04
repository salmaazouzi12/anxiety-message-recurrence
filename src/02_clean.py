
from pathlib import Path
import re
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

csv_files = list(RAW_DIR.glob("*.csv"))

if len(csv_files) != 1:
    raise ValueError("Expected exactly one CSV in data/raw/")

print("Loading dataset...")
df = pd.read_csv(csv_files[0], low_memory=False)

# --------------------------------------------------
# 1. Standardize dates
# --------------------------------------------------

df["date"] = pd.to_datetime(
    df["Post Created Date"],
    errors="coerce"
)

df["month"] = df["date"].dt.to_period("M").astype("string")

# --------------------------------------------------
# 2. Clean textual content
# --------------------------------------------------

def clean_text(text):
    if pd.isna(text):
        return ""

    text = unicodedata.normalize("NFKC", str(text))

    # Remove URLs (including CrowdTangle link syntax)
    text = re.sub(
        r"https?://\S+",
        " ",
        text
    )

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["text_clean"] = df["Message"].apply(clean_text)

df["has_text"] = df["text_clean"].ne("")

# --------------------------------------------------
# 3. Standardize numeric engagement fields
# --------------------------------------------------

engagement_columns = [
    "Likes", "Love", "Wow", "Haha",
    "Sad", "Angry", "Care",
    "Comments", "Shares",
    "Total Interactions"
]

for col in engagement_columns:
    if col in df.columns:
        cleaned = (
            df[col]
            .astype("string")
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        df[col] = pd.to_numeric(
            cleaned,
            errors="coerce"
        )

# --------------------------------------------------
# 4. Save all publication records
# --------------------------------------------------

df.to_parquet(
    PROCESSED_DIR / "posts_clean.parquet",
    index=False
)

print("Saved posts_clean.parquet")

# --------------------------------------------------
# 5. Create unique text corpus
# --------------------------------------------------

unique_texts = (
    df[df["has_text"]]
    .drop_duplicates(subset=["text_clean"])
    .copy()
)

unique_texts.to_parquet(
    PROCESSED_DIR / "unique_texts.parquet",
    index=False
)

print("Saved unique_texts.parquet")

# --------------------------------------------------
# 6. Summary
# --------------------------------------------------

print("\n" + "=" * 55)
print("CLEANING SUMMARY")
print("=" * 55)

print(f"Original posts: {len(df):,}")
print(f"Posts with cleaned text: {df['has_text'].sum():,}")
print(f"Unique cleaned texts: {len(unique_texts):,}")
print(f"Missing dates: {df['date'].isna().sum():,}")

print("\nCleaning completed.")

