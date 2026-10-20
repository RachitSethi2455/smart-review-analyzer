# Smart Review Analyzer: Report Outline

Target length is about 15–25 pages. Fill each section in, and copy the tables and figures from `results/`.

---

## Title page
- Title: *Smart Review Analyzer: Comparing Lexicon, Classical ML, N-gram and Transformer Approaches to Review Sentiment, with Aspect Extraction and Summarization*
- Names, roll numbers, course (Computational Linguistics and NLP Lab), guide, date

## Abstract (about 150 words)
- The problem, what you built, the dataset, the best result (e.g. "TF-IDF + LR reached X% accuracy on IMDB, versus Y% for VADER"), and one finding about aspects or summaries.

## 1. Introduction
- Why review analysis matters (e-commerce, movies, too many reviews to read)
- Objectives: list the six features of the app
- Contribution: one same-split comparison of lexicon, ML, n-gram and transformer models, plus aspect sentiment and pros/cons summaries in a single tool

## 2. Related work (1–2 pages)
- Lexicon methods: VADER (Hutto & Gilbert 2014), TextBlob
- Classical ML on IMDB: Maas et al. 2011 (the dataset paper); Wang & Manning 2012, NB-SVM at 91.2%
- Transformers: DistilBERT and RoBERTa on IMDB
- Aspect-based sentiment: SemEval-2014 Task 4, dependency-rule methods
- Extractive summarization: LexRank (Erkan & Radev 2004), TextRank
- Existing GitHub/Streamlit review analyzers, and what they lack (no evaluation, only one or two features)

## 3. Datasets
- IMDB: 25k train / 25k test, balanced. Figure: `eda_labels.png`, `eda_lengths.png`
- Flipkart / Amazon Fine Food, if used. Explain why the star rating is the ground truth.
- Table: dataset, size, labels, source

## 4. Methodology
### 4.1 System architecture
Diagram: input → preprocessing → [sentiment | NER | POS/aspects | summary | n-gram LM] → Streamlit UI
### 4.2 Preprocessing (Exp 2, 3)
- Steps, with one worked example (screenshot of the Preprocessing tab)
- Why negations are kept in the stopword list
### 4.3 Sentiment models (Exp 7)
- VADER (compound thresholds ±0.05), TextBlob (polarity and subjectivity)
- TF-IDF formula, the settings used (sublinear TF, min_df=2, max_df=0.9, n-grams), NB / LR / SVM
- DistilBERT (optional)
### 4.4 N-gram language model (Exp 6)
- The add-k smoothing formula, perplexity, and the perplexity classifier (one LM per class, shared vocabulary)
### 4.5 NER, POS and aspect extraction (Exp 4, 5)
- spaCy pipeline; the three dependency rules with example parses (displaCy screenshot)
### 4.6 Summarization (Exp 8)
- Frequency method (our own), LexRank, TextRank, LSA, Luhn; pros/cons grouping; ROUGE

## 5. Experiments and results
| # | Experiment | Output file |
|---|---|---|
| E1 | Lexicon baselines | `results/lexicon_results.csv` |
| E2 | TF-IDF × classifier × n-gram range | `results/ml_results.csv` |
| E3 | Preprocessing ablation (keeping stopwords, removing negations, no lemmatization) | `results/ml_results.csv` (ablation rows) |
| E4 | N-gram LM perplexity classifier: n and k | `results/ngram_results.csv` |
| E5 | DistilBERT on a 1k subset (optional) | `results/transformer_results.csv` |
| E6 | Summarization ROUGE vs lead baseline | `results/summary_results.csv` |
| E7 | Aspect extraction precision on 50–100 hand-labelled sentences | make this table by hand |
| E8 | 3-class Flipkart experiment (optional) | |

- Main table: `results/comparison_table.csv` and chart `comparison_best_per_family.png`
- Confusion matrices: `results/figures/cm_*.png`
- Discuss: on 2k reviews the bigram LM was best, but on 25k reviews the trigram LM wins. Explain this with data sparsity: higher-order n-grams need more data. Also discuss why larger k (more smoothing) helps here, why VADER does worse on long movie reviews, and what the ablation shows (removing negations hurts; keeping stopwords actually helped LR).

## 6. Application demo
- One screenshot per tab, with a caption saying which lab experiment it shows
- A "trap" example, e.g. "not bad at all" or sarcasm, showing where models disagree

## 7. Limitations and future work
- The known limitations from the README
- Future work: transformer aspect sentiment, fine-tuning, Hindi/Hinglish reviews, live scraping

## 8. Conclusion

## References
- Maas et al. 2011, Learning Word Vectors for Sentiment Analysis (IMDB)
- Hutto & Gilbert 2014, VADER
- Wang & Manning 2012, Baselines and Bigrams (NB-SVM)
- Erkan & Radev 2004, LexRank
- Mihalcea & Tarau 2004, TextRank
- Lin 2004, ROUGE
- Sanh et al. 2019, DistilBERT
- Pontiki et al. 2014, SemEval-2014 Task 4
- Jurafsky & Martin, *Speech and Language Processing*, Ch. 3 (n-gram LMs)
- spaCy, NLTK, scikit-learn, sumy documentation

## Appendix
- How to run (copy from the README)
- Key code excerpts: smoothing, aspect rules
