"""Rule-based aspect sentiment using dependency parsing (builds on Exp 5).

Example: "The battery is great but the camera is not good."
    -> battery: positive, camera: negative

Rules:
  1. amod   - adjective directly modifies a noun:        "great camera"
  2. acomp  - adjective after a verb, noun is its nsubj:  "battery is good"
  3. conj   - adjective joined by "and" to one above:     "cheap and good phone"
Negation ("not", "never") and intensifiers ("very") are kept in the opinion
phrase, and VADER scores the phrase, so "not good" comes out negative.

Known limitation: implicit aspects ("it dies by noon" = battery) are missed.
"""
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()
NOUN_POS = {"NOUN", "PROPN"}
SKIP_ASPECTS = {"thing", "lot", "way", "one", "time", "bit", "product", "movie", "film", "item"}


def _aspect_phrase(noun):
    """Include compound words: 'battery life', 'customer service'."""
    parts = [c for c in noun.children if c.dep_ == "compound" and c.i < noun.i]
    return " ".join([c.lemma_.lower() for c in parts] + [noun.lemma_.lower()])


def _opinion_phrase(adj, extra_heads=()):
    """Adjective plus its negation/intensifier modifiers, e.g. 'not very good'."""
    mods = [c for c in adj.children if c.dep_ in ("neg", "advmod")]
    for head in extra_heads:
        mods += [c for c in head.children if c.dep_ == "neg"]
    words = sorted(mods + [adj], key=lambda t: t.i)
    return " ".join(w.text.lower() for w in words)


def _score(phrase):
    return _vader.polarity_scores(phrase)["compound"]


def _pairs_for_adjective(adj):
    """Return (noun, extra_heads) pairs that this adjective describes."""
    if adj.dep_ == "amod" and adj.head.pos_ in NOUN_POS:
        return [(adj.head, ())]
    if adj.dep_ in ("acomp", "attr"):
        verb = adj.head
        subjects = [c for c in verb.children if c.dep_ in ("nsubj", "nsubjpass") and c.pos_ in NOUN_POS]
        return [(s, (verb,)) for s in subjects]
    if adj.dep_ == "conj" and adj.head.pos_ == "ADJ":
        return _pairs_for_adjective(adj.head)
    return []


def extract_aspects(doc):
    """Return a list of dicts: aspect, opinion, score, sentence."""
    found = []
    for token in doc:
        if token.pos_ != "ADJ":
            continue
        for noun, extra_heads in _pairs_for_adjective(token):
            aspect = _aspect_phrase(noun)
            if aspect in SKIP_ASPECTS or noun.is_stop:
                continue
            opinion = _opinion_phrase(token, extra_heads)
            found.append({
                "aspect": aspect,
                "opinion": opinion,
                "score": _score(opinion),
                "sentence": token.sent.text.strip(),
            })
    return found


def aspect_summary(docs, min_mentions=1):
    """Aggregate aspects across many reviews -> one row per aspect."""
    rows = [a for doc in docs for a in extract_aspects(doc)]
    if not rows:
        return pd.DataFrame(columns=["aspect", "mentions", "avg_score", "positive", "negative", "example_opinions"])
    df = pd.DataFrame(rows)
    summary = df.groupby("aspect").agg(
        mentions=("score", "size"),
        avg_score=("score", "mean"),
        positive=("score", lambda s: int((s > 0.05).sum())),
        negative=("score", lambda s: int((s < -0.05).sum())),
        example_opinions=("opinion", lambda s: ", ".join(dict.fromkeys(s).keys())[:80]),
    ).reset_index()
    summary = summary[summary["mentions"] >= min_mentions]
    return summary.sort_values("mentions", ascending=False).reset_index(drop=True)
