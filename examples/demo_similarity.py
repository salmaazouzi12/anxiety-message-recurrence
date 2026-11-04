import csv
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path(__file__).resolve().parent / "synthetic_posts.csv"


def clean_text(text, remove_hashtags=False):
    text = text.lower()
    if remove_hashtags:
        text = re.sub(r"(?<!\w)#[\w]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


with DATA.open(encoding="utf-8-sig", newline="") as file:
    posts = list(csv.DictReader(file))


def score_pairs(remove_hashtags):
    texts = [
        clean_text(post["text"], remove_hashtags)
        for post in posts
    ]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5)
    )
    vectors = vectorizer.fit_transform(texts)
    return cosine_similarity(vectors)


raw = score_pairs(False)
cleaned = score_pairs(True)

examples = [
    (1, 2, "Near-duplicate messages"),
    (6, 7, "Same topic, distinct messages"),
    (8, 9, "Different messages, shared hashtag"),
]

print("Illustrative character n-gram similarity")
print("-" * 65)

for first, second, description in examples:
    i, j = first - 1, second - 1
    print(f"\n{description}: posts {first} and {second}")
    print(f"  Before hashtag removal: {raw[i, j]:.3f}")
    print(f"  After hashtag removal:  {cleaned[i, j]:.3f}")

print("\nThese scores are illustrative, not paper evaluation results.")
