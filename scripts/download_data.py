"""Download the IMDB dataset (50K movie reviews) into data/raw/.

Usage:
    python scripts/download_data.py

Other datasets (manual download from Kaggle, needs a free account):
  - Flipkart product reviews   -> search Kaggle for "Flipkart Product Reviews"
  - Amazon Fine Food Reviews   -> https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
    (has a human-written 'Summary' column -> reference summaries for ROUGE)
Put the CSV files in data/raw/. Use the star rating as ground truth, NOT any
'sentiment' column that was generated with VADER (that would be circular).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from src.data import IMDB_TEST, IMDB_TRAIN

LABELS = {0: "negative", 1: "positive"}


def main():
    print("Downloading stanfordnlp/imdb from Hugging Face (~80 MB)...")
    ds = load_dataset("stanfordnlp/imdb")
    for split, path in (("train", IMDB_TRAIN), ("test", IMDB_TEST)):
        df = ds[split].to_pandas()
        df["label"] = df["label"].map(LABELS)
        df.to_csv(path, index=False)
        print(f"  {split}: {len(df):,} reviews -> {path}")
        print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
