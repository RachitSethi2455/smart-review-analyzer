"""Loading datasets (Lab Exp 1: reading CSV files with pandas)."""
import pandas as pd

from src.config import DATA_RAW, RANDOM_STATE

IMDB_TRAIN = DATA_RAW / "imdb_train.csv"
IMDB_TEST = DATA_RAW / "imdb_test.csv"

# Likely column names in Kaggle review CSVs (Flipkart, Amazon Fine Food, ...)
TEXT_COLUMN_GUESSES = ["review", "text", "Review", "Text", "review_text", "Summary", "content", "comment"]
RATING_COLUMN_GUESSES = ["rating", "Rating", "Rate", "Score", "stars", "score"]


def load_imdb(split="train", sample=None):
    path = IMDB_TRAIN if split == "train" else IMDB_TEST
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/download_data.py")
    df = pd.read_csv(path)
    if sample and sample < len(df):
        # stratified sample keeps the 50/50 class balance
        df = df.groupby("label").sample(n=sample // 2, random_state=RANDOM_STATE)
    return df.reset_index(drop=True)


def star_to_label(rating):
    """1-2 stars = negative, 3 = neutral, 4-5 = positive."""
    try:
        r = float(rating)
    except (TypeError, ValueError):
        return None
    if r <= 2:
        return "negative"
    if r < 4:
        return "neutral"
    return "positive"


def guess_column(df, guesses):
    for g in guesses:
        if g in df.columns:
            return g
    return None


def load_reviews_csv(path_or_buffer, text_col=None, rating_col=None, max_rows=None):
    """Load any review CSV and standardise it to columns: text, rating, label."""
    df = pd.read_csv(path_or_buffer, encoding_errors="replace")
    text_col = text_col or guess_column(df, TEXT_COLUMN_GUESSES)
    if text_col is None:
        raise ValueError(f"Could not find a text column. Columns are: {list(df.columns)}")
    rating_col = rating_col or guess_column(df, RATING_COLUMN_GUESSES)
    out = pd.DataFrame({"text": df[text_col].astype(str)})
    if rating_col is not None:
        out["rating"] = pd.to_numeric(df[rating_col], errors="coerce")
        out["label"] = out["rating"].map(star_to_label)
    out = out[out["text"].str.strip().str.len() > 0]
    if max_rows:
        out = out.head(max_rows)
    return out.reset_index(drop=True)
