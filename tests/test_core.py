"""Quick checks that the core pieces work. Run: python -m pytest -q"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import aspects, linguistics, sentiment, summarize
from src.ngram_lm import NGramLM, PerplexityClassifier
from src.preprocessing import preprocess


def test_preprocess_keeps_negations():
    assert "not" in preprocess("This phone isn't good at all")
    assert "not" not in preprocess("This phone isn't good at all", keep_negations=False)


def test_preprocess_strips_html_and_lemmatizes():
    assert preprocess("Loved the movies!<br /><br />Great acting.") == ["love", "movie", "great", "act"]


def test_ngram_probabilities_sum_to_one():
    lm = NGramLM(n=2, k=1.0, min_count=1).fit([["good", "phone"], ["bad", "phone"], ["good", "camera"]])
    total = sum(lm.prob(w, ["good"]) for w in lm.vocab)
    assert math.isclose(total, 1.0, rel_tol=1e-9)


def test_smoothing_gives_unseen_bigram_nonzero_probability():
    lm = NGramLM(n=2, k=1.0, min_count=1).fit([["good", "phone"], ["bad", "camera"]])
    assert lm.prob("camera", ["good"]) > 0
    assert math.isfinite(lm.perplexity(["good", "camera"]))


def test_perplexity_classifier():
    pos = [["great", "movie", "loved", "it"]] * 5
    neg = [["terrible", "movie", "hated", "it"]] * 5
    clf = PerplexityClassifier(n=2, k=0.1, min_count=1).fit(pos + neg, ["positive"] * 5 + ["negative"] * 5)
    assert clf.predict([["loved", "it"], ["hated", "it"]]) == ["positive", "negative"]


def test_aspects_with_negation():
    doc = linguistics.parse("The battery is great but the camera is not good.")
    found = {a["aspect"]: a["score"] for a in aspects.extract_aspects(doc)}
    assert found["battery"] > 0
    assert found["camera"] < 0


def test_aspects_amod_and_compound():
    doc = linguistics.parse("It has an amazing battery life and a terrible screen.")
    found = {a["aspect"]: a["score"] for a in aspects.extract_aspects(doc)}
    assert found["battery life"] > 0
    assert found["screen"] < 0


def test_lexicon_models():
    assert sentiment.vader("I absolutely love this!")["label"] == "positive"
    assert sentiment.vader("This is awful and broken.")["label"] == "negative"
    assert 0 <= sentiment.textblob("A wonderful film")["subjectivity"] <= 1


def test_entities():
    doc = linguistics.parse("I bought a Samsung phone from Flipkart in Mumbai.")
    labels = {label for _, label in linguistics.entities(doc)}
    assert "GPE" in labels


def test_summaries():
    text = ("The phone is fast. The battery lasts long. The camera is excellent in daylight. "
            "Night photos are noisy. The price is fair for what you get.")
    for method in ("lexrank", "textrank", "lsa", "luhn", "frequency", "lead"):
        out = summarize.summarize(text, 2, method)
        assert out and len(out) < len(text)
    assert summarize.rouge("the battery lasts long", "battery lasts long")["rouge1"] > 0.8
