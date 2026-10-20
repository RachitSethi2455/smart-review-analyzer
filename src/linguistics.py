"""spaCy-based analysis: morphology (Exp 3), NER (Exp 4), POS tagging and chunking (Exp 5)."""
from collections import Counter
from functools import lru_cache

import pandas as pd
import spacy

from src.config import SPACY_MODEL

# Entity types that matter for product/movie reviews
REVIEW_ENTITY_TYPES = {"ORG", "PRODUCT", "PERSON", "GPE", "WORK_OF_ART", "MONEY", "DATE", "NORP", "LOC"}


@lru_cache(maxsize=1)
def get_nlp():
    try:
        return spacy.load(SPACY_MODEL)
    except OSError as exc:
        raise OSError(
            f"spaCy model '{SPACY_MODEL}' is missing. Run: python -m spacy download {SPACY_MODEL}"
        ) from exc


def parse(text):
    return get_nlp()(text)


def parse_many(texts, batch_size=64):
    return list(get_nlp().pipe(texts, batch_size=batch_size))


# ---------- Exp 3.2: morphological analysis ----------
def morphology_table(doc):
    rows = []
    for t in doc:
        if t.is_punct or t.is_space:
            continue
        rows.append({
            "word": t.text,
            "lemma": t.lemma_,
            "pos": t.pos_,
            "morph_features": str(t.morph) or "-",
            "is_stopword": t.is_stop,
        })
    return pd.DataFrame(rows)


# ---------- Exp 4.2: named entity recognition ----------
def entities(doc, types=REVIEW_ENTITY_TYPES):
    return [(ent.text, ent.label_) for ent in doc.ents if types is None or ent.label_ in types]


def entity_table(docs, types=REVIEW_ENTITY_TYPES):
    """Count entities across many docs -> DataFrame(entity, label, count)."""
    counts = Counter()
    for doc in docs:
        for text, label in entities(doc, types):
            counts[(text.strip(), label)] += 1
    rows = [{"entity": e, "label": l, "count": c} for (e, l), c in counts.most_common()]
    return pd.DataFrame(rows, columns=["entity", "label", "count"])


# ---------- Exp 5.2: POS tagging and chunking ----------
def pos_table(doc):
    return pd.DataFrame(
        [{"word": t.text, "pos": t.pos_, "tag": t.tag_, "explanation": spacy.explain(t.tag_) or "",
          "dependency": t.dep_, "head": t.head.text} for t in doc if not t.is_space]
    )


def pos_counts(docs):
    counts = Counter(t.pos_ for doc in docs for t in doc if not (t.is_punct or t.is_space))
    return pd.DataFrame(counts.most_common(), columns=["pos", "count"])


def noun_chunks(doc):
    return [chunk.text for chunk in doc.noun_chunks]


def top_words_by_pos(docs, pos="ADJ", n=15):
    """Most frequent adjectives (or nouns, verbs...) - shows *how* reviewers describe things."""
    counts = Counter(t.lemma_.lower() for doc in docs for t in doc if t.pos_ == pos and t.is_alpha)
    return pd.DataFrame(counts.most_common(n), columns=["word", "count"])
