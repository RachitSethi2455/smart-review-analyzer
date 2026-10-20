"""All experiments for the report. Each command writes a CSV to results/ and figures to results/figures/.

Usage (run from the project folder):
    python scripts/experiments.py eda                  # dataset plots (Exp 1, 4)
    python scripts/experiments.py ml                   # TF-IDF + NB / LR / SVM, ablations, saves best model (Exp 7)
    python scripts/experiments.py lexicon              # VADER and TextBlob baselines (Exp 7)
    python scripts/experiments.py ngram                # n-gram LM perplexity classifier (Exp 6)
    python scripts/experiments.py transformer          # DistilBERT on a test subset (optional)
    python scripts/experiments.py summaries --csv data/raw/Reviews.csv   # ROUGE (Exp 8)
    python scripts/experiments.py table                # merge everything into one comparison table

Add --sample 5000 to any IMDB command for a quick run while developing.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd

from src.config import BEST_MODEL_PATH, DATA_PROCESSED, FIGURES_DIR, RESULTS_DIR
from src.data import load_imdb
from src.evaluation import report, save_results
from src.preprocessing import clean_text, preprocess

PREPROCESS_VARIANTS = {
    "default": dict(remove_stopwords=True, use_lemmatize=True, keep_negations=True),
    "no_stopword_removal": dict(remove_stopwords=False, use_lemmatize=True),
    "negations_removed": dict(remove_stopwords=True, use_lemmatize=True, keep_negations=False),
    "no_lemmatization": dict(remove_stopwords=True, use_lemmatize=False, keep_negations=True),
    "lm": dict(remove_stopwords=False, use_lemmatize=False),  # for language models
}


def get_tokens(split, sample, variant="default"):
    """Preprocess IMDB once and cache the tokens to data/processed/."""
    cache = DATA_PROCESSED / f"imdb_{split}_{sample or 'all'}_{variant}.joblib"
    if cache.exists():
        return joblib.load(cache)
    df = load_imdb(split, sample)
    print(f"Preprocessing {len(df):,} {split} reviews ({variant})...", flush=True)
    start = time.time()
    tokens = [preprocess(t, **PREPROCESS_VARIANTS[variant]) for t in df["text"]]
    print(f"  done in {time.time() - start:.0f}s")
    joblib.dump((tokens, df["label"].tolist()), cache)
    return tokens, df["label"].tolist()


# ---------------------------------------------------------------- EDA
def cmd_eda(args):
    from src.text_stats import bar_figure, label_distribution_figure, top_words, wordcloud_figure, word_frequencies
    import matplotlib.pyplot as plt

    df = load_imdb("train", args.sample)
    label_distribution_figure(df["label"], "IMDB train: class distribution").savefig(FIGURES_DIR / "eda_labels.png", dpi=150)

    lengths = df["text"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(lengths.clip(upper=1000), bins=50, color="#1565c0")
    ax.set_title("Review length (words, clipped at 1000)")
    ax.set_xlabel("words")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_lengths.png", dpi=150)
    print(lengths.describe().round(1).to_string())

    for label, cmap in (("positive", "Greens"), ("negative", "Reds")):
        texts = df.loc[df["label"] == label, "text"].head(3000)
        bar_figure(top_words(texts, 10), "word", "count", f"Top 10 words: {label}").savefig(
            FIGURES_DIR / f"eda_top10_{label}.png", dpi=150)
        wordcloud_figure(word_frequencies(texts), f"Word cloud: {label} reviews", cmap).savefig(
            FIGURES_DIR / f"eda_wordcloud_{label}.png", dpi=150)
    print(f"Figures saved to {FIGURES_DIR}")


# ---------------------------------------------------------------- Classical ML
def cmd_ml(args):
    from src.sentiment import CLASSIFIER_NAMES, build_pipeline, save_model

    X_train, y_train = get_tokens("train", args.sample)
    X_test, y_test = get_tokens("test", args.sample)
    X_train, X_test = [" ".join(t) for t in X_train], [" ".join(t) for t in X_test]

    rows, best = [], (None, -1.0, None)
    # E2: classifiers x n-gram range
    for clf in ("nb", "lr", "svm"):
        for ngrams, ng_name in (((1, 1), "unigram"), ((1, 2), "uni+bigram")):
            name = f"TF-IDF {ng_name} + {CLASSIFIER_NAMES[clf]}"
            start = time.time()
            pipe = build_pipeline(clf, ngrams).fit(X_train, y_train)
            train_s = time.time() - start
            row = report(name, y_test, pipe.predict(X_test),
                         extra={"family": "classical ML", "train_seconds": round(train_s, 1)})
            rows.append(row)
            if row["f1"] > best[1]:
                best = (pipe, row["f1"], name)

    # E3: preprocessing ablation with the LR uni+bigram model
    for variant in ("no_stopword_removal", "negations_removed", "no_lemmatization"):
        Xtr, ytr = get_tokens("train", args.sample, variant)
        Xte, yte = get_tokens("test", args.sample, variant)
        pipe = build_pipeline("lr", (1, 2)).fit([" ".join(t) for t in Xtr], ytr)
        rows.append(report(f"Ablation: LR uni+bigram, {variant}", yte, pipe.predict([" ".join(t) for t in Xte]),
                           extra={"family": "ablation"}, save_cm=False))

    save_results(rows, "ml_results.csv")
    # LinearSVC has no predict_proba; the app prefers LR, so save LR if it is within 0.5% of the best
    lr_row = next(r for r in rows if r["model"] == "TF-IDF uni+bigram + Logistic Regression")
    if best[2] != lr_row["model"] and lr_row["f1"] >= best[1] - 0.005:
        best = (build_pipeline("lr", (1, 2)).fit(X_train, y_train), lr_row["f1"], lr_row["model"])
    save_model(best[0])
    print(f"\nSaved best model ({best[2]}, F1={best[1]}) to {BEST_MODEL_PATH}")


# ---------------------------------------------------------------- Lexicon baselines
def cmd_lexicon(args):
    from src.sentiment import textblob, vader

    df = load_imdb("test", args.sample)
    rows = []
    for name, fn in (("VADER", vader), ("TextBlob", textblob)):
        start = time.time()
        preds = [fn(t, binary=True)["label"] for t in df["text"]]
        rows.append(report(name, df["label"], preds,
                           extra={"family": "lexicon", "predict_seconds": round(time.time() - start, 1)}))
    save_results(rows, "lexicon_results.csv")


# ---------------------------------------------------------------- N-gram LM (Exp 6)
def cmd_ngram(args):
    from src.ngram_lm import PerplexityClassifier

    X_train, y_train = get_tokens("train", args.sample, "lm")
    X_test, y_test = get_tokens("test", args.sample, "lm")
    rows = []
    for n in (1, 2, 3):
        for k in (1.0, 0.1, 0.01):
            name = f"{n}-gram LM perplexity classifier (add-k, k={k})"
            start = time.time()
            clf = PerplexityClassifier(n=n, k=k).fit(X_train, y_train)
            preds = clf.predict(X_test)
            rows.append(report(name, y_test, preds, save_cm=(k == 0.1),
                               extra={"family": "n-gram LM", "n": n, "k": k,
                                      "seconds": round(time.time() - start, 1)}))
    save_results(rows, "ngram_results.csv")


# ---------------------------------------------------------------- Transformer (optional)
def cmd_transformer(args):
    from src.sentiment import transformer_predict

    n = args.sample or 1000
    df = load_imdb("test", n)
    print(f"Running DistilBERT on {len(df)} test reviews (CPU, a few minutes)...")
    start = time.time()
    preds = transformer_predict(df["text"].tolist())
    if preds is None:
        sys.exit("transformers/torch are not installed. See requirements-optional.txt")
    rows = [report(f"DistilBERT-SST2 (off the shelf, {len(df)}-review subset)", df["label"],
                   [p["label"] for p in preds],
                   extra={"family": "transformer", "predict_seconds": round(time.time() - start, 1)})]
    save_results(rows, "transformer_results.csv")


# ---------------------------------------------------------------- Summaries (Exp 8)
def cmd_summaries(args):
    from src.summarize import rouge, summarize

    if not args.csv:
        sys.exit("Pass --csv path/to/Reviews.csv (Amazon Fine Food Reviews from Kaggle)")
    df = pd.read_csv(args.csv, usecols=["Text", "Summary"]).dropna()
    df["Text"] = df["Text"].map(clean_text)
    # only reviews long enough to need summarising
    df = df[df["Text"].str.split().str.len() > 80].head(args.sample or 500)
    print(f"Evaluating on {len(df)} reviews; reference = human-written 'Summary' field")
    rows = []
    for method in ("lead", "frequency", "lexrank", "textrank", "lsa", "luhn"):
        scores = pd.DataFrame([rouge(summarize(t, 1, method), ref) for t, ref in zip(df["Text"], df["Summary"])])
        rows.append({"method": method, **scores.mean().round(4).to_dict()})
    save_results(rows, "summary_results.csv")


# ---------------------------------------------------------------- Final table
def cmd_table(args):
    import matplotlib.pyplot as plt

    frames = [pd.read_csv(RESULTS_DIR / f) for f in
              ("lexicon_results.csv", "ml_results.csv", "ngram_results.csv", "transformer_results.csv")
              if (RESULTS_DIR / f).exists()]
    if not frames:
        sys.exit("No results yet - run the other commands first.")
    df = pd.concat(frames, ignore_index=True)
    cols = ["family", "model", "accuracy", "precision", "recall", "f1"]
    table = df[df["family"] != "ablation"][cols].sort_values("f1", ascending=False)
    table.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)
    print(table.to_string(index=False))

    top = table.drop_duplicates("family", keep="first").sort_values("f1")
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(top) + 1.5))
    ax.barh(top["model"], top["f1"], color="#1565c0")
    for i, v in enumerate(top["f1"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center")
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Macro F1 on IMDB test")
    ax.set_title("Best model per family")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "comparison_best_per_family.png", dpi=150)
    print(f"\nSaved {RESULTS_DIR / 'comparison_table.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["eda", "ml", "lexicon", "ngram", "transformer", "summaries", "table"])
    parser.add_argument("--sample", type=int, default=None, help="use only N reviews per split (quick runs)")
    parser.add_argument("--csv", help="CSV path for the summaries command")
    args = parser.parse_args()
    globals()[f"cmd_{args.command}"](args)


if __name__ == "__main__":
    main()
