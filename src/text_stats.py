"""Word counts, top words, word clouds and plots (Lab Exp 1 & 4)."""
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # no GUI needed (works in scripts and Streamlit)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from wordcloud import WordCloud

from src.preprocessing import preprocess

COLORS = {"positive": "#2e7d32", "neutral": "#9e9e9e", "negative": "#c62828"}


def word_frequencies(texts, **preprocess_kwargs):
    if isinstance(texts, str):
        texts = [texts]
    counts = Counter()
    for t in texts:
        counts.update(preprocess(t, **preprocess_kwargs))
    return counts


def top_words(texts, n=10):
    return pd.DataFrame(word_frequencies(texts).most_common(n), columns=["word", "count"])


def basic_counts(texts):
    if isinstance(texts, str):
        texts = [texts]
    words = [len(str(t).split()) for t in texts]
    return {"documents": len(texts), "total_words": sum(words),
            "avg_words_per_doc": round(sum(words) / max(len(words), 1), 1)}


def wordcloud_figure(frequencies, title=None, colormap="viridis"):
    fig, ax = plt.subplots(figsize=(8, 4))
    if frequencies:
        wc = WordCloud(width=800, height=400, background_color="white", colormap=colormap,
                       max_words=100).generate_from_frequencies(frequencies)
        ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def bar_figure(df, x, y, title, color="#1565c0", horizontal=True):
    fig, ax = plt.subplots(figsize=(7, 4))
    if horizontal:
        ax.barh(df[x][::-1], df[y][::-1], color=color)
        ax.set_xlabel(y)
    else:
        ax.bar(df[x], df[y], color=color)
        ax.set_ylabel(y)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def label_distribution_figure(labels, title="Sentiment distribution"):
    counts = pd.Series(labels).value_counts()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(counts.index, counts.values, color=[COLORS.get(l, "#1565c0") for l in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def confusion_matrix_figure(y_true, y_pred, title, labels=None):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    return fig
