# Smart Review Analyzer

NLP Lab Mini-Project by **Rachit** and **Manan**. Submission: 5 Nov 2026.

Paste a review or upload a CSV of product or movie reviews. The app shows:

- sentiment from three kinds of model (lexicon, classical ML, transformer)
- subjectivity
- word clouds and top words
- named entities
- aspect-level sentiment ("battery 👍, camera 👎")
- pros/cons summaries
- an n-gram language model

## How it covers the lab syllabus

| Lab experiment | Where it is in this project |
|---|---|
| 1. CSV + graphs | `src/data.py`, `src/text_stats.py`, every chart in the app, `experiments.py eda` |
| 2. Tokenization | `src/preprocessing.py` (`sentences`, `tokens`), **Preprocessing** tab |
| 3. Preprocessing, morphology | `preprocess()` (stopwords, lemmatization, stemming), `linguistics.morphology_table` |
| 4. Word count, word cloud, NER | `text_stats.py`, `linguistics.entity_table`, **Words** and **Entities** tabs |
| 5. POS tagging, chunking | `linguistics.pos_table`, `noun_chunks`, `src/aspects.py` (dependency rules) |
| 6. N-gram LM + smoothing | `src/ngram_lm.py`, written from scratch: add-k smoothing, perplexity, next-word suggestions, perplexity classifier |
| 7. Sentiment + subjectivity | `src/sentiment.py`: VADER, TextBlob, TF-IDF + NB/LR/SVM, DistilBERT (optional) |
| 8. Summarization | `src/summarize.py`: our own frequency method, LexRank, TextRank, LSA, Luhn, ROUGE |

## Setup (once, on each laptop)

```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate
```
```bash
pip install -r requirements.txt
```
```bash
python -m spacy download en_core_web_sm
```
```bash
python scripts/download_data.py
```

Optional: add DistilBERT as the transformer baseline. It is about a 1 GB download.
```bash
pip install -r requirements-optional.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Run

The trained model (`models/sentiment_best.joblib`) is committed, so the app works straight after setup. You only need `download_data.py` to re-run the experiments.

```bash
streamlit run app.py
```
```bash
python -m pytest -q
```

## Experiments (results go to `results/`)

Each command writes a CSV to `results/` and its plots to `results/figures/`. Add `--sample 2000` for a quick test run.

| Command | What it does |
|---|---|
| `python scripts/experiments.py eda` | Dataset plots: class balance, review lengths, top words, word clouds |
| `python scripts/experiments.py lexicon` | VADER and TextBlob on the IMDB test set |
| `python scripts/experiments.py ml` | TF-IDF with NB/LR/SVM (unigram vs bigram), the preprocessing ablation, and saves the best model for the app |
| `python scripts/experiments.py ngram` | 1-, 2- and 3-gram LM perplexity classifier, k = 1 / 0.1 / 0.01 |
| `python scripts/experiments.py transformer` | DistilBERT on 1,000 test reviews (optional) |
| `python scripts/experiments.py summaries --csv data/raw/Reviews.csv` | ROUGE scores against Amazon Fine Food summaries |
| `python scripts/experiments.py table` | Merges everything into `results/comparison_table.csv` and a chart |

Full IMDB runs take a few minutes each on a laptop. Preprocessed text is cached in `data/processed/`, so later runs are faster.

## Datasets

- **IMDB 50K:** downloaded by `scripts/download_data.py`. It has positive/negative labels only, with no neutral class.
- **Flipkart product reviews (Kaggle, manual download):** use the **star rating** as the ground truth. The dataset's "sentiment" column was made by running VADER, so testing VADER against it would be circular.
- **Amazon Fine Food Reviews (Kaggle, manual download):** its `Summary` column is written by people, so it can serve as the reference for ROUGE.
- **`data/sample_reviews.csv`:** 18 made-up demo reviews for quick testing of the app.

Put downloaded CSVs in `data/raw/`. That folder is git-ignored, so don't commit datasets.

## Project structure

```
app.py                  Streamlit app (8 tabs)
src/
  config.py             paths and settings
  data.py               dataset loading, star rating -> label
  preprocessing.py      cleaning, tokenization, stopwords (keeps "not"), lemmatization
  linguistics.py        spaCy: POS, chunks, morphology, NER
  aspects.py            rule-based aspect sentiment (amod / acomp + nsubj / conj)
  sentiment.py          VADER, TextBlob, TF-IDF models, explanations, DistilBERT
  ngram_lm.py           n-gram LM with add-k smoothing and perplexity classifier
  summarize.py          extractive summaries, pros/cons, ROUGE
  text_stats.py         word counts, word clouds, plots
  evaluation.py         metrics, confusion matrices, saving results
scripts/
  download_data.py      gets IMDB
  experiments.py        every experiment for the report
tests/test_core.py      quick checks
report/report_outline.md
```

## Who does what

| Dates | Rachit: models and evaluation | Manan: app and linguistics |
|---|---|---|
| 20–21 Oct | Dataset download, EDA plots | Get familiar with the app; add Flipkart data |
| 22–24 Oct | `ml` experiments, ablation, choose the best model | Improve the NER and Words tabs |
| 25–27 Oct | `lexicon`, `ngram`, optional `transformer`; comparison table | Aspect rules: add more patterns, test on 50 sentences |
| 28–29 Oct | Summaries + ROUGE; 3-class Flipkart experiment | Summary tab, polish the UI |
| 30 Oct | Both: hand-label 50 sentences for aspect precision | Integration, testing with big CSVs |
| 31 Oct – 3 Nov | Report: results and discussion | Report: system design and screenshots |
| 4 Nov | Buffer: freeze code, rehearse the demo and viva | Same |

**Freeze features on 27 Oct.** After that, only fix bugs and evaluate. Each person should be able to explain the other's code in the viva.

## Known limitations (put these in the report)

- **Aspect rules:** they miss implicit aspects ("it dies by noon" means the battery). They sometimes pick wrong nouns ("low light"). VADER also does not know some domain words, so "slow" scores 0.
- **IMDB has no neutral class:** 3-class results need star-rated product data.
- **Lexicon models:** they struggle with sarcasm and with long reviews that mix positive and negative points.
