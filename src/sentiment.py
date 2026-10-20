"""Sentiment and subjectivity (Lab Exp 7).

Three families of models, so we can compare them on the same test split:
  1. Lexicon:      VADER, TextBlob (no training)
  2. Classical ML: TF-IDF + Naive Bayes / Logistic Regression / Linear SVM
  3. Transformer:  DistilBERT fine-tuned on SST-2 (optional, inference only)
"""
from functools import lru_cache

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import BEST_MODEL_PATH, RANDOM_STATE, TRANSFORMER_MODEL, VADER_NEG, VADER_POS
from src.preprocessing import clean_text, preprocess_to_string

_vader = SentimentIntensityAnalyzer()


def score_to_label(score, binary=False):
    if binary:
        return "positive" if score >= 0 else "negative"
    if score >= VADER_POS:
        return "positive"
    if score <= VADER_NEG:
        return "negative"
    return "neutral"


# ---------- 1. Lexicon models ----------
def vader(text, binary=False):
    scores = _vader.polarity_scores(clean_text(text))
    return {"label": score_to_label(scores["compound"], binary), "score": scores["compound"],
            "pos": scores["pos"], "neu": scores["neu"], "neg": scores["neg"]}


def textblob(text, binary=False):
    s = TextBlob(clean_text(text)).sentiment
    return {"label": score_to_label(s.polarity, binary), "polarity": s.polarity,
            "subjectivity": s.subjectivity}


def subjectivity_label(subjectivity):
    return "subjective (opinion)" if subjectivity >= 0.5 else "objective (fact)"


# ---------- 2. TF-IDF + classical ML ----------
CLASSIFIERS = {
    "nb": lambda: MultinomialNB(alpha=1.0),
    "lr": lambda: LogisticRegression(max_iter=2000, C=4.0, random_state=RANDOM_STATE),
    "svm": lambda: LinearSVC(C=0.5, random_state=RANDOM_STATE),
}
CLASSIFIER_NAMES = {"nb": "Naive Bayes", "lr": "Logistic Regression", "svm": "Linear SVM"}


def build_pipeline(classifier="lr", ngram_range=(1, 2), max_features=200_000):
    """TF-IDF features -> classifier. Input must already be preprocessed strings."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=ngram_range, min_df=2, max_df=0.9,
                                  sublinear_tf=True, max_features=max_features)),
        ("clf", CLASSIFIERS[classifier]()),
    ])


def save_model(pipeline, path=BEST_MODEL_PATH):
    joblib.dump(pipeline, path)


@lru_cache(maxsize=1)
def load_model(path=BEST_MODEL_PATH):
    """Returns the trained pipeline, or None if `scripts/experiments.py ml` has not been run yet."""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None


def ml_predict(text, pipeline=None):
    pipeline = pipeline or load_model()
    if pipeline is None:
        return None
    processed = preprocess_to_string(text)
    label = pipeline.predict([processed])[0]
    result = {"label": label}
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([processed])[0]
        result["confidence"] = float(np.max(probs))
    return result


def explain_words(text, pipeline=None, top=10):
    """Words that pushed the prediction towards positive (+) or negative (-).

    Works for linear models (LR / SVM): contribution = tfidf value x coefficient.
    """
    pipeline = pipeline or load_model()
    if pipeline is None or not hasattr(pipeline.named_steps["clf"], "coef_"):
        return []
    vec, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    x = vec.transform([preprocess_to_string(text)])
    coefs = clf.coef_[0]
    # coef_ points towards classes_[1]; flip so that + always means "positive"
    sign = 1 if clf.classes_[1] == "positive" else -1
    names = vec.get_feature_names_out()
    contrib = [(names[j], sign * x[0, j] * coefs[j]) for j in x.nonzero()[1]]
    contrib.sort(key=lambda p: abs(p[1]), reverse=True)
    return contrib[:top]


# ---------- 3. Transformer (optional) ----------
@lru_cache(maxsize=1)
def get_transformer():
    """Returns a HuggingFace pipeline, or None if transformers/torch are not installed."""
    try:
        from transformers import pipeline
    except ImportError:
        return None
    return pipeline("sentiment-analysis", model=TRANSFORMER_MODEL, truncation=True, max_length=512)


def transformer_predict(texts):
    model = get_transformer()
    if model is None:
        return None
    if isinstance(texts, str):
        texts = [texts]
    outputs = model([clean_text(t) for t in texts], batch_size=16)
    return [{"label": o["label"].lower(), "confidence": o["score"]} for o in outputs]
