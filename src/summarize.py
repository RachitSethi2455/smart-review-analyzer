"""Extractive text summarization (Lab Exp 8).

- frequency_summary: our own implementation (sentence score = sum of word frequencies)
- sumy_summary:      LexRank / TextRank / LSA / Luhn from the sumy library
- pros_cons_summary: for many reviews - summarise liked and disliked points separately
- rouge:             ROUGE-1/2/L against a reference summary
"""
from collections import Counter

from rouge_score import rouge_scorer
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.utils import get_stop_words
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.preprocessing import clean_text, preprocess, sentences

SUMY_METHODS = {
    "lexrank": LexRankSummarizer,
    "textrank": TextRankSummarizer,
    "lsa": LsaSummarizer,
    "luhn": LuhnSummarizer,
}
_vader = SentimentIntensityAnalyzer()
_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def frequency_summary(text, n_sentences=3):
    sents = sentences(text)
    if len(sents) <= n_sentences:
        return " ".join(sents)
    freq = Counter(preprocess(text))
    if not freq:
        return " ".join(sents[:n_sentences])
    top = max(freq.values())
    scores = []
    for i, s in enumerate(sents):
        words = preprocess(s)
        # normalise by length so long sentences don't always win
        score = sum(freq[w] / top for w in words) / (len(words) ** 0.5 or 1)
        scores.append((score, i))
    chosen = sorted(i for _, i in sorted(scores, reverse=True)[:n_sentences])
    return " ".join(sents[i] for i in chosen)


def sumy_summary(text, n_sentences=3, method="lexrank"):
    parser = PlaintextParser.from_string(clean_text(text), Tokenizer("english"))
    summarizer = SUMY_METHODS[method](Stemmer("english"))
    summarizer.stop_words = get_stop_words("english")
    return " ".join(str(s) for s in summarizer(parser.document, n_sentences))


def lead_summary(text, n_sentences=3):
    """Baseline: first n sentences. Any real method should beat this on ROUGE."""
    return " ".join(sentences(text)[:n_sentences])


def summarize(text, n_sentences=3, method="lexrank"):
    if method == "frequency":
        return frequency_summary(text, n_sentences)
    if method == "lead":
        return lead_summary(text, n_sentences)
    return sumy_summary(text, n_sentences, method)


def pros_cons_summary(reviews, n_sentences=4, method="lexrank", threshold=0.3, max_sentences=400):
    """Split every review into sentences, group by VADER polarity, summarise each group."""
    pros, cons = [], []
    for review in reviews:
        for s in sentences(review):
            score = _vader.polarity_scores(s)["compound"]
            if score >= threshold:
                pros.append((score, s))
            elif score <= -threshold:
                cons.append((score, s))
    # keep the strongest sentences so LexRank stays fast on big CSVs
    pros = [s for _, s in sorted(pros, reverse=True)[:max_sentences]]
    cons = [s for _, s in sorted(cons)[:max_sentences]]
    return {
        "liked": summarize(" ".join(pros), n_sentences, method) if pros else "",
        "disliked": summarize(" ".join(cons), n_sentences, method) if cons else "",
        "n_positive_sentences": len(pros),
        "n_negative_sentences": len(cons),
    }


def rouge(summary, reference):
    scores = _rouge.score(reference, summary)
    return {name: round(s.fmeasure, 4) for name, s in scores.items()}
