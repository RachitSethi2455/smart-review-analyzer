# Smart Review Analyzer: Feasibility and Build Plan for a 2-Person NLP Lab Mini-Project (due 5 Nov 2026)

Scope: a Streamlit app where the user pastes one review or uploads a CSV of reviews. It shows sentiment (pos/neg/neutral plus subjectivity), word cloud, top words, NER, POS-based adjective and aspect extraction, and a summary of many reviews, with charts. These notes are meant to help judge feasibility and plan the build. Research date: 20 Oct 2026.

Note on sources: Papers with Code, the usual IMDB leaderboard, was shut down on 24 Jul 2025 and now redirects to Hugging Face Trending Papers. Its archives are static, and no new leaderboard results appear after July 2025 ([hyper.ai](https://hyper.ai/en/news/42900); [Codesota, a competitor, so treat with caution](https://www.codesota.com/papers-with-code)). For "state of the art" numbers, NLP-progress and model cards were used instead. Baselight, a Kaggle mirror used to check one Flipkart dataset, also shut down (July 2026). Its page now shows only a closure notice, so some Flipkart details could not be re-verified.

---

## Q1. Which public datasets are best (IMDB 50K, Amazon, Flipkart, Yelp, etc.)? Sizes, labels, licensing, links

### Takeaway
Use **IMDB 50K** as the main benchmark for training and evaluating the sentiment classifier. It matches lab experiment 7, is balanced and is standard. Add **one product-review dataset with star ratings and a short "Summary/Title" field** for the product and aspect demo: a Flipkart CSV (Indian context) or Amazon Fine Food Reviews. Star ratings give a real 3-class (neutral) label, and the Summary/Title column can serve as a pseudo-reference for ROUGE. Keep the product subset small (5k–20k rows) for laptop speed.

### Cited Findings
**IMDB (Maas et al., 2011)**
- HF `stanfordnlp/imdb`: 25,000 train, 25,000 test and 50,000 unlabeled ("unsupervised") reviews. Binary labels (0 = neg, 1 = pos), "highly polar" movie reviews. License is tagged "other" with "More Information Needed". Download is about 84 MB (217 MB on disk). Citation: Maas, Daly, Pham, Huang, Ng, Potts, ACL-HLT 2011, pp. 142–150 — [HF dataset card](https://huggingface.co/datasets/stanfordnlp/imdb)
- The Kaggle version "IMDB Dataset of 50K Movie Reviews" (kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) is a single CSV with 50,000 rows and 2 columns (review, sentiment). Users report it is noisy (HTML `<br />` tags etc.) and needs cleaning — [Medium walkthrough](https://khalidsiddique2401.medium.com/imdb-sentiment-analysis-using-python-63a5d494f42c)
- The Kaggle mirror is described as perfectly balanced, 25k positive and 25k negative — [Kaggle mirror listing (via search snippet; Baselight now defunct)](https://baselight.app/u/kaggle/dataset/atulanandjha_imdb_50k_movie_reviews_test_your_bert)
- IMDB has **only positive and negative labels, no neutral class** — [HF card](https://huggingface.co/datasets/stanfordnlp/imdb)

**Flipkart (Indian e-commerce)**
- "Flipkart Products Review Dataset" by Mansi Thummar on Kaggle is widely used, with many community notebooks (VADER, RNN). Last updated about 3 years ago — [Kaggle](https://kaggle.com/datasets/mansithummar67/flipkart-product-review-dataset/code)
- "Flipkart Product Reviews With Sentiment Dataset" (Kaggle user niraliivaghani): 205,053 rows and 6 columns, scraped from flipkart.com with BeautifulSoup in Dec 2022. Positive/neutral/negative labels were assigned **by running VADER on the Summary column**, not by humans — [Baselight mirror via search snippet (site now shut down)](https://baselight.app/u/kaggle/dataset/niraliivaghani_flipkart_product_customer_reviews_dataset)
- Zenodo "Flipkart Review Dataset" (Oct 2025, v1): 104 product types, 194,276 rows and 5 columns. Row counts differ from the Kaggle listing, so verify the file — [Zenodo](https://zenodo.org/records/17612521)
- Analytics Vidhya has a tutorial on Flipkart review sentiment with Pandas, NLTK and WordCloud — [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2022/09/sentiment-analysis-on-flipkart-dataset/)
- Licensing for the Flipkart datasets was not stated in the sources found — [Kaggle](https://kaggle.com/datasets/mansithummar67/flipkart-product-review-dataset/code)

**Amazon**
- Amazon Reviews'23 (McAuley Lab, UCSD): 571.54M reviews, 54.51M users, 48.19M items, 33 categories, May 1996 – Sep 2023. Fields: rating (1–5), title, text, helpful_vote, verified_purchase, etc. Small categories include All_Beauty (701.5K ratings) and Subscription_Boxes (16.2K). Electronics (43.9M) and Cell_Phones (20.8M) are huge. No license is stated on the site. Cite Hou et al. 2024, arXiv:2403.03952 — [Amazon Reviews'23 site](https://amazon-reviews-2023.github.io/); HF repo `McAuley-Lab/Amazon-Reviews-2023` loads per-category configs such as `raw_review_All_Beauty` — [HF](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/benchmark)
- `amazon_polarity` (HF): binary neg/pos, 3.6M train and 400K test, fields title, content, label — [HF README](https://huggingface.co/datasets/fancyzhx/amazon_polarity/blob/main/README.md)
- Amazon Fine Food Reviews (Kaggle `snap/amazon-fine-food-reviews`): 568,454 reviews, 256,059 users, 74,258 products, Oct 1999 – Oct 2012. 10 columns, including Score (1–5) and **Summary** (a short digest of each review) — [GitHub notebook README](https://github.com/manish-vi/supervised_ml_notebooks). The Summary column has been used as the target for seq2seq review summarization — [jalammar/Text-Summarization-with-Amazon-Reviews](https://github.com/jalammar/Text-Summarization-with-Amazon-Reviews). The usual labeling is 4–5 = positive, 1–2 = negative, and 3 = neutral (often dropped) — [Towards Data Science](https://towardsdatascience.com/sentiment-analysis-on-amazon-food-reviews-from-eda-to-deployment-f985c417b0c/). A 3,000-row sample exists on Kaggle — [Baselight listing via snippet](https://baselight.app/u/kaggle/dataset/davemarietti_amazonfinefoods3k_csv)

**Yelp**
- `Yelp/yelp_review_full`: 650,000 train and 50,000 test, 5 classes (stars 1–5; 130k train / 10k test per class). License "other", governed by the Yelp dataset agreement (Yelp Dataset Challenge 2015) — [HF card](https://huggingface.co/datasets/Yelp/yelp_review_full)
- `yelp_polarity`: 560,000 train and 38,000 test, binary, derived from star ratings — [HF dataset_infos](https://huggingface.co/datasets/fancyzhx/yelp_polarity/raw/8c69647bc35c8f2415a3007a16e52a305cd8537d/dataset_infos.json)

**Aspect-annotated (to evaluate aspect extraction)**
- SemEval-2014 Task 4 (ABSA): laptop and restaurant reviews, over 6K training sentences with aspect-level human annotations. Polarity labels are positive/negative/neutral/conflict, and many papers drop "conflict". Official page: alt.qcri.org/semeval2014/task4 — [SemEval task page](https://alt.qcri.org/semeval2014/task4/); [label note, arXiv 2310.18025](https://arxiv.org/pdf/2310.18025); overview paper (163 submissions from 32 teams) — [Pontiki et al., S14-2004](https://preview.aclanthology.org/moar-dois/S14-2004.pdf)

### Inferences
- **Recommended combo:** (a) IMDB 50K Kaggle CSV to train and evaluate the TF-IDF classifiers (it directly continues lab experiments 7 and 8). (b) A 5k–20k-row sample of a star-rated product dataset (Flipkart for Indian relevance, or Amazon Fine Food / an Amazon'23 small category) for the product demo and the 3-class neutral experiment (1–2★ neg, 3★ neutral, 4–5★ pos). (c) Optionally, about 100 SemEval-2014 laptop sentences to measure aspect-extraction precision. Laptops suit an electronics review demo ("battery", "screen", "keyboard").
- If the Flipkart niraliivaghani labels are used, the report must say they are **VADER-generated pseudo-labels**. Evaluating VADER against them is circular. Use the star `Rate` column as ground truth instead.
- Do not download the whole Amazon'23 dataset. It is tens of GB. Use one small category or `amazon_polarity` subsets.
- All of these are free and need no paid API. Most are "research use" with unclear licenses, so the report should cite each dataset and state "used for academic purposes only".

### Gaps
- Exact current license text on the Kaggle IMDB 50K and Flipkart pages could not be fetched (Kaggle pages are JS-rendered, and the Baselight mirror is now dead). Column names of the niraliivaghani CSV could not be re-verified. From memory they are likely product_name, product_price, Rate, Review, Summary, Sentiment, but this is unverified.
- The exact Kaggle URL for the niraliivaghani dataset was not confirmed. It is likely kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset.
- No verified HF mirror of SemEval-2014 ABSA was confirmed. Several community copies exist, with varying label handling.

---

## Q2. Which libraries and models fit each feature? Typical IMDB accuracies for classic ML vs VADER vs transformers

### Takeaway
Rough IMDB accuracy ladder: lexicon tools (VADER/TextBlob) **about 66–77%** → TF-IDF + Logistic Regression / Linear SVM / NB **about 87–89%** → NB-SVM with bigrams **about 91%** → off-the-shelf DistilBERT-SST2 **about 81%** (domain shift) → IMDB-tuned RoBERTa-large **about 96%** (XLNet SOTA 96.21%). For a lab project: build the TF-IDF models yourself (main contribution), use VADER and TextBlob as lexicon baselines, and optionally run one pretrained transformer via the `pipeline()` API on a **subset** of the test set for comparison.

### Cited Findings
**Lexicon tools**
- VADER: MIT license, `pip install vaderSentiment`. "Specifically attuned to sentiments expressed in social media" but works on other domains. Standard thresholds: compound ≥ 0.05 positive, ≤ −0.05 negative, otherwise neutral — [vaderSentiment GitHub](https://github.com/cjhutto/vaderSentiment)
- TextBlob: `.sentiment` returns polarity in [−1, 1] and subjectivity in [0, 1] (0 = very objective, 1 = very subjective). It also provides `.noun_phrases`, `.tags` (POS), tokenization, n-grams, lemmatization and spelling correction (about 70% accurate) — [TextBlob quickstart](https://textblob.readthedocs.io/en/dev/quickstart.html)
- On movie reviews, a cited study (Bonta et al.) found VADER 77%, TextBlob 74% and NLTK 62% accuracy — [CEUR-WS Vol-3935 Paper15 (secondary citation)](https://ceur-ws.org/Vol-3935/Paper15.pdf)
- An Amrita survey of tools on IMDB and Rotten Tomatoes reports 73% accuracy and F1 0.78 on IMDB for its best (TextBlob Naive Bayes) analyzer — [Amrita](https://www.amrita.edu/publication/sentiment-analysis-tools-for-movie-review-evaluation-a-survey/) (attribution to the exact tool is unclear across search snippets)
- VADER "degrades with domain-specific jargon or multi-aspect sentiment common in longer movie reviews" — [arXiv 2601.07235 survey](https://arxiv.org/pdf/2601.07235). On product reviews, sentiment tools were more accurate on short and medium reviews, and ML beat VADER only on long reviews about experience goods — [AMCIS 2018](https://aisel.aisnet.org/amcis2018/DataScience/Presentations/27). VADER reached 66.25% on 80 hard (mixed) Weimar-era film reviews — [UvA DARE](https://dare.uva.nl/id/826fe2bc-316a-4eef-8bba-3f48ae6de3ea)

**Classic ML (scikit-learn TF-IDF)**
- TF-IDF + Logistic Regression on IMDB: 88.98% accuracy, 89.13% F1, ROC-AUC about 0.957 (compared against MNB, linear SVM, DT, KNN, RF) — [CAUCHY journal 2025](https://ejournal.uin-malang.ac.id/index.php/Math/article/view/33809); 87.29% accuracy and 0.948 AUC — [ITATS Kernel journal](https://ejurnal.itats.ac.id/kernel/article/view/8567); 89.20% accuracy (P 88.8 / R 89.8) with default parameters — [SIMAD repository](https://repository.simad.edu.so/id/eprint/31)
- NB-SVM (Wang & Manning 2012, "Baselines and Bigrams"): bigram features give consistent gains, and the SVM over NB log-count ratios reaches 91.22% on IMDB. A reproduction got 91.55% (bigrams) and 91.82% (trigrams) — [ACL P12-2018](https://aclanthology.org/P12-2018/); [jeanvc/nbsvm reproduction](https://github.com/jeanvc/nbsvm)

**Transformers (HuggingFace, free, local)**
- `distilbert-base-uncased-finetuned-sst-2-english`: 67M params, Apache-2.0, 91.3% on SST-2 dev (vs 92.7% BERT-base). Binary only. The card notes bias issues (e.g., country names shift the positive probability from 0.89 to 0.08) — [HF model card](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
- `siebert/sentiment-roberta-large-english` (RoBERTa-large fine-tuned on 15 datasets) on Maas et al. (2011) = IMDB: **96.0% vs 80.6% for DistilBERT-SST2**. Yelp 96.5 vs 84.8, Amazon reviews (McAuley & Leskovec) 98.0 vs 84.7, average 93.2 vs 78.1 — [HF model card](https://huggingface.co/siebert/sentiment-roberta-large-english) (Maas 2011 = IMDB per [HF IMDB card](https://huggingface.co/datasets/stanfordnlp/imdb))
- `cardiffnlp/twitter-roberta-base-sentiment-latest`: **3 classes (Negative/Neutral/Positive)**, RoBERTa-base trained on about 124M tweets (2018–2021) and fine-tuned on TweetEval, CC-BY-4.0. Designed for tweets — [HF model card](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- DistilBERT embeddings + custom feed-forward network on IMDB: 86.6% test accuracy (not full fine-tuning) — [NHSJS paper](https://nhsjs.wpsite.pro/?p=36941)
- XLNet-Large IMDB error rate 3.79% (96.21% accuracy) vs BERT-Large 4.51%, per the NLP-progress leaderboard — [NLP-progress](https://nlpprogress.com/english/sentiment_analysis.html)

**NER / POS / parsing (spaCy)**
- `en_core_web_sm` v3.7.1 (MIT): pipeline tok2vec, tagger, parser, senter, attribute_ruler, lemmatizer, ner. 18 NER labels including **ORG, PERSON, GPE, PRODUCT, LOC, MONEY, WORK_OF_ART**. NER F 84.56, POS tag accuracy 97.25, dependency UAS 91.75 / LAS 89.87 — [HF spacy/en_core_web_sm](https://huggingface.co/spacy/en_core_web_sm)

**Summarization**
- `sumy` (Apache-2.0, Python 3.8+, `pip install sumy`, about 3.7k stars, CI active): LexRank, LSA, Luhn, Edmundson, plus others listed in docs/summarizators.md. Ships a `sumy_eval` CLI to compare against a reference summary — [sumy GitHub](https://github.com/miso-belica/sumy)

### Inferences
Feature-to-tool map (recommended):
| Feature | Tool | Syllabus link |
|---|---|---|
| CSV upload, charts | pandas + matplotlib/plotly in Streamlit (`st.file_uploader`, `st.pyplot`/`st.bar_chart`) | Exp 1 |
| Tokenize, clean | NLTK `word_tokenize`/`sent_tokenize`, or spaCy | Exp 2 |
| Stopwords, lemmatize, morphology | NLTK stopwords + WordNetLemmatizer, or spaCy `token.lemma_`, `token.morph` | Exp 3 |
| Word count, top-10, word cloud | `collections.Counter`, `wordcloud` lib | Exp 4 |
| NER (brands/people/places) | spaCy `en_core_web_sm` (ORG/PRODUCT/PERSON/GPE) | Exp 4 |
| POS, chunking, adjectives | spaCy POS + `noun_chunks`, or NLTK `pos_tag` + RegexpParser | Exp 5 |
| N-gram LM / smoothing | NLTK `nltk.lm` (Laplace/KneserNey) on positive vs negative reviews, e.g., "which class LM gives lower perplexity" as a toy classifier, or bigram features in TF-IDF | Exp 6 |
| Sentiment + subjectivity | VADER (3-class via thresholds), TextBlob (polarity + subjectivity), TF-IDF+LR/MNB/LinearSVC (trained), optional DistilBERT/RoBERTa | Exp 7 |
| Summarization | sumy LexRank/TextRank/LSA/Luhn, or your own frequency-based extractive summarizer (as in lab) | Exp 8 |
- Using the N-gram LM as a classifier (train a bigram Laplace LM on positive reviews and one on negative, then classify by lower perplexity) is a cheap way to cover experiment 6 in the project. It is an inference and its accuracy is untested here. Expect it below TF-IDF+LR.
- Neutral class: VADER thresholds and TextBlob polarity bands (e.g., |polarity| < 0.05–0.1) give a neutral label for free. The trained classifier can be made 3-class only on star-rated product data. The cardiffnlp model is a free 3-class transformer, but it is tweet-trained, so mention the domain mismatch.
- An honest expectation for the report: VADER on IMDB probably lands around 65–75%, below TF-IDF+LR (about 88–90%). That gap is a good talking point: lexicons miss sarcasm, negation scope and long mixed reviews.

### Gaps
- No single primary source benchmarks VADER, TextBlob, TF-IDF+LR and DistilBERT on the **same** IMDB split. The cited numbers come from different papers and preprocessing. The students' own comparison table fills exactly this gap, which is a selling point.
- No clean, primary VADER-on-IMDB-50K accuracy figure was found.
- TextBlob's default analyzer (PatternAnalyzer vs NaiveBayesAnalyzer) was not confirmed from the quickstart page.
- Laptop CPU inference speed for DistilBERT/RoBERTa on 25k long reviews was not measured in any source found.

---

## Q3. How can aspect-based sentiment ("battery good, camera bad") be done simply with POS/dependency parsing? Is it feasible in 2 weeks?

### Takeaway
Yes. A **rule-based, dependency-parse ABSA** is very feasible in 3–4 days with spaCy. Extract noun aspects and pair them with adjectives through `amod` (adjectival modifier, "great camera") and `acomp`/`nsubj` ("battery is good"). Flip polarity on negation (`neg` child, "not good"). Score each opinion word with VADER/TextBlob, then aggregate per aspect into a bar chart. Expect moderate precision and low recall on implicit aspects. Report this honestly, ideally with a small manual evaluation.

### Cited Findings
- `amod` = adjectival modifier. In "There are slow and repetitive parts", "slow" and "repetitive" modify "parts", which is the core aspect–opinion pattern — [arXiv 2512.00515](https://arxiv.org/pdf/2512.00515)
- A food-review system uses spaCy POS and dependency labels to extract adjective–noun pairs. It keeps adjectives whose dependency is `amod`. For `acomp` it goes to the head verb and finds the `nsubj` noun. If a negation word is found in the head's subtree, it is added to the pair (polarity flip) — [arXiv 2201.10978](https://arxiv.org/pdf/2201.10978)
- In a spaCy parse of "spaCy is great", "great" is `acomp` and "spaCy" is `nsubj` — [spacyr docs](https://spacyr.quanteda.io/articles/using_spacyr.html)
- An aspect-extraction paper considers opinions sharing `amod` with candidate aspects, and for `nsubj` includes only adjectives modifying nouns, which reduces false matches — [search result summary; source cited as BERT-ASC / related work](https://arxiv.org/pdf/2203.11702) (attribution uncertain)
- Limitations: rule building takes a lot of effort and many dependency patterns per domain, and "large numbers of aspects are not extracted by the rules" — [SlideShare, 2018 conference deck](https://www.slideshare.net/slideshow/aspect-extraction-performance-with-common-pattern-of-dependency-relation-in-multi-aspect-sentiment-analysis/91662569)
- A 2023 unsupervised method (DS-RN) uses spaCy's dependency parser with ROOT-node rules plus lexicon pruning for explicit aspect extraction — [INFLIBNET Shodhganga](https://sgbeta.inflibnet.ac.in/items/04c56585-7604-4c41-a3a9-e45b2d21b522)
- An existing HF Space ("Aspect-Product-Sentiment-Analysis", Streamlit) loops over reviews, detects aspects, runs transformer sentiment and keeps positive/negative/neutral counts per aspect, plus a word cloud — [HF Space source](https://huggingface.co/spaces/Saima0/Aspect-Product-Sentiment-Analysis/blob/main/src/streamlit_app.py)
- SemEval-2014 laptop and restaurant data provide human aspect annotations for evaluation — [SemEval-2014 Task 4](https://alt.qcri.org/semeval2014/task4/)

### Inferences
- Minimal algorithm (about 60 lines, spaCy `en_core_web_sm`):
  1. Split into sentences (`doc.sents`).
  2. For each token with `pos_ == "ADJ"`:
     - if `dep_ == "amod"` and its head is a NOUN, then aspect = head (use the compound noun if `compound` children exist, e.g., "battery life");
     - if `dep_ in ("acomp", "attr")` and it has a head verb (e.g., "is"), then aspect = the `nsubj` child of that verb.
  3. Negation: if the adjective or its head has a child with `dep_ == "neg"`, invert the score.
  4. Score = VADER `polarity_scores(adj)["compound"]` (or TextBlob polarity of "adj aspect").
  5. Normalize aspects by lemma. Optionally map them to a small hand-made aspect dictionary per domain (phone: battery, camera, display/screen, price, delivery…; movie: acting, plot, story, music, direction).
  6. Aggregate across reviews: mention count, % positive, average score. Show a diverging bar chart.
- Extensions if time allows: conjunctions (`conj`: "battery is good and cheap"), adverb intensifiers (`advmod`: "very good"), and "but" clause splitting.
- Evaluation: hand-label 50–100 product-review sentences (or use SemEval laptop sentences) and report aspect-extraction precision/recall and aspect-polarity accuracy. Even a small table with error analysis adds credibility.
- Risks: implicit aspects ("it dies by noon" means battery) are missed. Verb opinions ("I love the camera") need an extra rule (`dobj` of sentiment verbs). Hinglish or misspelled Flipkart text hurts parsing. Movie reviews are long and multi-aspect, so ABSA demos are cleaner on product reviews.
- Zero-shot or transformer ABSA (e.g., PyABSA, SetFit ABSA) exist but add dependency weight and risk. Keep them as "future work", not a core feature.

### Gaps
- No authoritative, maintained step-by-step spaCy ABSA tutorial with code was found. Students will write the traversal themselves (straightforward). The "Rule 1: amod children" student report (anlp.vmv.re) was only seen as a heading.
- No published precision/recall figure for simple amod/acomp rules on SemEval-2014 was found in this search.

---

## Q4. Notable existing GitHub projects or tutorials (Streamlit review analyzers): what do they include, and how can students differentiate?

### Takeaway
Many projects exist, but most do **one thing**: sentiment classification of a single review or a CSV, sometimes with a word cloud. No found open-source project combines classical-vs-lexicon-vs-transformer comparison, NER, dependency-based aspect sentiment and multi-review summarization in one Streamlit app with an evaluation section. That combination, plus Indian (Flipkart) data and syllabus-mapped modules, is the differentiation.

### Cited Findings
- `amri-tah/Amazon-Reviews-Sentiment-Analysis`: Streamlit app with single-review, CSV batch and Amazon-URL (scraper) modes, plus a word cloud. Models are tagged as logistic regression / SVM (pickled `models.p`). No accuracy metrics are reported in the README. Uses the Kaggle Amazon full-score dataset (1–5★; 3M train / 650k test) — [GitHub](https://github.com/amri-tah/Amazon-Reviews-Sentiment-Analysis)
- An HF Space does aspect-based product sentiment with per-aspect pos/neg/neu counts and a word cloud using a transformer — [HF Space](https://huggingface.co/spaces/Saima0/Aspect-Product-Sentiment-Analysis/blob/main/src/streamlit_app.py)
- A ConsoleFlare tutorial builds a Streamlit app that takes an uploaded text file and returns an overall TextBlob sentiment score plus a word cloud (no NER, no aspects) — [ConsoleFlare](https://consoleflare.com/blog/?p=4076)
- Analytics Vidhya Flipkart tutorial: Pandas + NLTK + WordCloud sentiment analysis — [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2022/09/sentiment-analysis-on-flipkart-dataset/)
- A Towards Data Science article covers Amazon food review sentiment "from EDA to deployment" — [TDS](https://towardsdatascience.com/sentiment-analysis-on-amazon-food-reviews-from-eda-to-deployment-f985c417b0c/)
- Commercial freelancers sell "sentiment + spaCy NER in Streamlit" as a service (GPT-4 based), which shows the feature combination is seen as valuable — [Upwork listing](https://www.upwork.com/services/product/development-it-review-sentiment-ner-actionable-insights-from-customer-feedback-1956744902365210293)
- Other repos exist (e.g., [2003HARSH/Sentiment-Analysis-on-movie-reviews](https://github.com/2003HARSH/Sentiment-Analysis-on-movie-reviews), the [GitHub text-sentiment-analysis topic](https://github.com/topics/text-sentiment-analysis)), but their contents were not inspected.

### Inferences
Differentiation and extension ideas (pick 2–3, ranked by value per effort):
1. **Model comparison dashboard inside the app**: a "Compare" tab that runs VADER, TextBlob, TF-IDF+LR and (optional) DistilBERT on the same input and shows agreement or disagreement. Highlight cases where lexicons and the trained model disagree. Cheap and impressive.
2. **Aspect sentiment chart** ("Battery 78% positive, Camera 35% positive") from dependency rules, with example sentences shown per aspect (explainability).
3. **Explainability**: show the top positive and negative TF-IDF+LR coefficients (words) and highlight words in the input review by contribution, in green/red. It is easy with linear models and examiners like it.
4. **Rating vs sentiment mismatch detector**: flag reviews where the star rating disagrees with text sentiment (e.g., 5★ but negative text). Possible sarcasm or spam. Uses the product dataset's star column.
5. **Pros/Cons summary**: run extractive summarization separately on the positive and negative reviews, giving "What people liked / disliked".
6. **N-gram LM tie-in** (syllabus 6): show top bigrams/trigrams per sentiment class, and a toy perplexity-based classifier.
7. **Indian context**: Flipkart data, plus noting Hinglish/code-mixed reviews as a limitation or future work.
8. **Downloadable report**: let the user download the CSV with per-review predictions (`st.download_button`).
- Avoid scraping live Amazon/Flipkart URLs. It is fragile, against site terms and breaks at demo time. Twitter/X API is paid, so avoid it.

### Gaps
- GitHub star counts and the feature completeness of the best-starred "review analyzer" Streamlit repos were not systematically surveyed. Only a few were inspected.
- No Real Python Streamlit review-analyzer tutorial was found in this search.

---

## Q5. Evaluation: which metrics and experiments make results credible for a lab report?

### Takeaway
A credible report needs: (1) a **model comparison table** on the IMDB 25k test set (accuracy, precision, recall, F1, plus ROC-AUC for probabilistic models) covering VADER, TextBlob, MNB, LR, LinearSVC (± bigrams) and optionally a pretrained transformer on a subset. (2) **Confusion matrices**. (3) A **preprocessing ablation** (stopwords/lemmatization on vs off, unigrams vs uni+bigrams). (4) A **3-class experiment** on star-rated product data. (5) **ROUGE** for summaries, using product "Summary/Title" fields as references. (6) A small **manual evaluation** of aspect extraction.

### Cited Findings
- Published IMDB TF-IDF+LR studies report accuracy, F1 and ROC-AUC/PR-AUC, and compare against MNB, linear SVM, decision tree, KNN and random forest. This is a good template — [CAUCHY 2025](https://ejournal.uin-malang.ac.id/index.php/Math/article/view/33809); [ITATS](https://ejurnal.itats.ac.id/kernel/article/view/8567)
- Bigrams give consistent gains for sentiment (NB-SVM 91.22% on IMDB). This justifies an n-gram ablation — [Wang & Manning 2012](https://aclanthology.org/P12-2018/); [reproduction](https://github.com/jeanvc/nbsvm)
- Reference points for the "related work" table: XLNet 96.21% (SOTA-era), RoBERTa-large (siebert) 96.0%, DistilBERT-SST2 zero-shot 80.6% on IMDB — [NLP-progress](https://nlpprogress.com/english/sentiment_analysis.html); [siebert card](https://huggingface.co/siebert/sentiment-roberta-large-english)
- `rouge-score` (PyPI, v0.1.2, Jul 2022): pure-Python reimplementation of ROUGE-1.5.5, supporting ROUGE-N, ROUGE-L and ROUGE-Lsum (newline = sentence boundary), with an optional stemmer and bootstrap confidence intervals. Use `rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)` — [PyPI rouge-score](https://pypi.org/project/rouge-score/)
- `sumy_eval` can compare summarizer output to a reference summary — [sumy](https://github.com/miso-belica/sumy)
- The Amazon Fine Food `Summary` field is a human-written short digest of each review, already used as a summarization target — [jalammar repo](https://github.com/jalammar/Text-Summarization-with-Amazon-Reviews); [manish-vi notebooks](https://github.com/manish-vi/supervised_ml_notebooks)
- SemEval-2014 ABSA offers gold aspect annotations for laptops and restaurants — [SemEval task page](https://alt.qcri.org/semeval2014/task4/)

### Inferences
Suggested experiments (all CPU-friendly):
| # | Experiment | Data | Metric | Output |
|---|---|---|---|---|
| E1 | Lexicon baselines: VADER, TextBlob (drop neutral or map to nearest class) | IMDB test 25k | Acc, P, R, F1, confusion matrix | Table row + CM plot |
| E2 | TF-IDF + MNB / LR / LinearSVC (unigram vs 1–2 gram) | IMDB train/test | Acc, macro-F1, ROC-AUC, train time | Main table |
| E3 | Preprocessing ablation (raw vs stopword removal vs lemmatization) | IMDB | Acc/F1 | Small table. Note that removing stopwords like "not" can *hurt* sentiment. |
| E4 | Transformer zero-shot (DistilBERT-SST2) on a 1k–2k random subset | IMDB test subset | Acc, F1, sec/review | Table row (note subset) |
| E5 | 3-class sentiment (1–2★/3★/4–5★) with LR vs VADER thresholds | Flipkart/Amazon sample | Macro-F1, per-class F1, CM | Shows that neutral is hardest |
| E6 | Extractive summarization: LexRank vs LSA vs Luhn vs own frequency method vs lead-1 baseline | Product reviews where Summary/Title exists (single-review title generation as proxy), or grouped reviews | ROUGE-1/2/L F1 | Table + example outputs |
| E7 | Aspect extraction precision (manual) | 50–100 product sentences / SemEval laptops | Precision, recall, polarity accuracy | Table + error analysis |
- Keep a fixed `random_state`, report the train/test split, and use 5-fold cross-validation on the training set for hyperparameter C (LR/SVM). Show training time, a strong point for classic ML vs transformers.
- ROUGE caveat: titles and summaries are abstractive and very short, so absolute ROUGE scores will be low. Always include a simple baseline (first sentence / lead-k) so the numbers have context. For multi-review summarization without references, add a small human rating (1–5 for informativeness and readability by both students or classmates) and be explicit that this is a qualitative evaluation.
- Mandatory plots: class distribution bar chart, review-length histogram, top-20 words per class, word clouds pos vs neg, confusion matrices, model comparison bar chart, aspect sentiment bar chart.

### Gaps
- No standard reference-summary benchmark for **multi-review summarization of IMDB** exists in the sources found. IMDB has no reference summaries, so ROUGE on IMDB itself is not possible without hand-written references.
- No published ROUGE numbers for sumy algorithms on Amazon review titles were found.

---

## Q6. Realistic difficulty, time estimate, risks, and splitting work between 2 people

### Takeaway
**Difficulty: moderate. Feasible in about 2 weeks** if the scope is fixed early: classic ML + lexicons + spaCy rules, transformers optional. Most components are direct extensions of the 8 lab experiments. The main risks are transformer compute, scope creep, the neutral-class mismatch in IMDB, and Streamlit performance on 50k reviews. All are manageable with caching, sampling and pre-trained pickles.

### Cited Findings
- The DistilBERT SST-2 model has 67M params, and RoBERTa-large-based siebert is far larger (param count not stated on its card). The siebert card gives only a simple `pipeline()` usage — [DistilBERT card](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english); [siebert card](https://huggingface.co/siebert/sentiment-roberta-large-english)
- IMDB has no neutral label (binary only) — [HF IMDB card](https://huggingface.co/datasets/stanfordnlp/imdb)
- Streamlit Community Cloud (free) limits as of Feb 2024: about 690 MB–2.7 GB memory, "may change at any time". Apps over the limit get throttled or stop working. Recommended fixes: `st.cache_data` / `st.cache_resource`, capping cache size, profiling memory — [Streamlit docs](https://docs.streamlit.io/knowledge-base/deploy/resource-limits). A 2026 third-party guide says 1 GB, which conflicts with the docs — [livemy.app](https://livemy.app/blog/deploy-streamlit-app)
- The Kaggle IMDB CSV needs cleaning (HTML tags, noise) — [Medium](https://khalidsiddique2401.medium.com/imdb-sentiment-analysis-using-python-63a5d494f42c)
- Rule-based aspect extraction misses many aspects and needs per-domain patterns — [SlideShare 2018](https://www.slideshare.net/slideshow/aspect-extraction-performance-with-common-pattern-of-dependency-relation-in-multi-aspect-sentiment-analysis/91662569)
- VADER struggles with long, mixed movie reviews — [arXiv 2601.07235](https://arxiv.org/pdf/2601.07235)

### Inferences
**Effort estimate (2 people, about 3–4 h/day each, 20 Oct → 5 Nov):**
| Days | Person A ("Models and Evaluation") | Person B ("App and Linguistics") |
|---|---|---|
| 1–2 (20–21 Oct) | Download IMDB + product dataset, cleaning script, EDA plots | Streamlit skeleton: text box, CSV uploader, tabs; word count/top-10/word cloud |
| 3–5 (22–24 Oct) | TF-IDF + MNB/LR/SVM, CV, ablations, pickle best model (`joblib`) | VADER + TextBlob integration (3-class + subjectivity), spaCy NER tab (ORG/PRODUCT/PERSON/GPE counts) |
| 6–8 (25–27 Oct) | VADER/TextBlob/DistilBERT-subset evaluation, confusion matrices, comparison table | Dependency-rule ABSA + aspect chart; POS/adjective frequency; noun chunks |
| 9–10 (28–29 Oct) | Summarization experiments (sumy + own) + ROUGE; 3-class product experiment | Summary tab (overall + pros/cons), explainability highlighting, download button |
| 11 (30 Oct) | Manual ABSA evaluation (both label 50 sentences each) | Integration, caching, error handling, test with 5–10k row CSV |
| 12–14 (31 Oct – 3 Nov) | Report: results, discussion, related work | Report: system design, screenshots, architecture diagram; README, requirements.txt |
| 15 (4 Nov) | Buffer: viva prep, demo rehearsal, freeze code | same |
- Both people should understand every module for the viva. Swap and review each other's code on days 10–11.
- **Risks and mitigations:**
  - *Transformer compute on laptops:* PyTorch + model download is about 0.3–1.5 GB. CPU inference on long IMDB reviews is slow, and inputs are truncated at 512 tokens. Mitigation: evaluate on a 1–2k subset, load with `@st.cache_resource`, make it an optional toggle, never fine-tune (fine-tuning on CPU over 25k reviews is impractical in this timeline; Colab free GPU is a possible but optional route).
  - *Neutral class missing in IMDB:* state it explicitly. Use thresholds for lexicons, and a product dataset with 3★ = neutral for 3-class training. Optionally use the cardiffnlp 3-class model (tweet domain caveat).
  - *Stopword removal deleting negations* ("not", "no", "never" are in NLTK stopwords): keep a custom stopword list that excludes negations. This is a good viva point.
  - *Streamlit slowness on big CSVs:* cap uploads (e.g., analyze the first 5,000 rows or a sample), cache the spaCy model and vectorizer, use `nlp.pipe(batch_size=…)` with unused components disabled.
  - *Library/version breakage:* pin versions in `requirements.txt`. Use the spaCy 3.7.x model to match spaCy >= 3.7.2, < 3.8 (per the model card), or use matching newer versions together. Download NLTK data (`punkt`/`punkt_tab`, `stopwords`, `wordnet`, `vader_lexicon`) in a setup script.
  - *Pseudo-labeled datasets* (Flipkart VADER labels): use star ratings as ground truth instead.
  - *Scope creep:* freeze features by day 8. Transformer ABSA, scraping and multilingual support are "future work".
- Overall difficulty: about 6/10 for undergrads. Nothing needs a GPU, and every core technique was covered in lab. The novelty is in integration plus systematic evaluation.

### Gaps
- No measured benchmarks of DistilBERT/RoBERTa CPU throughput on typical Indian student laptops (e.g., i5, 8 GB RAM) were found. The time estimates above are judgment, not sourced.
- Current (2026) Streamlit Community Cloud limits are unconfirmed. The official docs date from Feb 2024 and a third-party source conflicts with them.

---

## Q7. Likely viva and examiner questions

### Takeaway
Examiners will probe (a) **why** each technique was chosen, with its limitations, (b) how the numbers were obtained (split, metrics, baselines), and (c) whether each student understands the internals of TF-IDF, LR/NB, VADER, dependency parsing and extractive summarization. Prepare short, honest answers backed by your own comparison table.

### Cited Findings
- VADER is lexicon- and rule-based, tuned for social media, with fixed compound thresholds (±0.05) — [vaderSentiment](https://github.com/cjhutto/vaderSentiment)
- TextBlob subjectivity ranges over 0 (objective) to 1 (subjective) — [TextBlob docs](https://textblob.readthedocs.io/en/dev/quickstart.html)
- Bigrams help sentiment (NB-SVM) — [Wang & Manning 2012](https://aclanthology.org/P12-2018/)
- DistilBERT-SST2 drops to 80.6% on IMDB, versus 96% for a model trained on diverse review data (domain shift) — [siebert card](https://huggingface.co/siebert/sentiment-roberta-large-english)
- spaCy NER label set and accuracy (F 84.56) — [spaCy en_core_web_sm](https://huggingface.co/spacy/en_core_web_sm)
- ROUGE-1/2/L definitions are implemented in `rouge-score` — [PyPI](https://pypi.org/project/rouge-score/)

### Inferences
Likely questions (with answer hints):
1. *Why TF-IDF over bag-of-words? What do TF and IDF mean? Why use sublinear TF / max_df?* Hint: they down-weight common words.
2. *Why does Logistic Regression beat Naive Bayes (or vice versa) here? What is the NB independence assumption?*
3. *Why is VADER worse on IMDB than your trained model?* Hint: social-media lexicon, long multi-aspect reviews, sarcasm, no domain training.
4. *IMDB has no neutral class. How do you predict "neutral"? How did you evaluate it?*
5. *Did you remove "not" as a stopword? What happens to "not good"?* How do you handle negation (bigrams, VADER rules, `neg` dependency)?
6. *Difference between stemming and lemmatization? Why lemmatize? Show a morphological analysis example.*
7. *Explain your aspect-extraction rule. What are `amod`, `acomp`, `nsubj`? Give a sentence where it fails.*
8. *What is subjectivity vs polarity? Can a review be objective and negative?*
9. *Which NER labels did you use for brands? Why does spaCy tag some brands as PERSON or GPE?* Hint: the small model makes errors (F about 85%) and lacks domain fine-tuning.
10. *How does your summarizer pick sentences (frequency scoring / LexRank PageRank on a sentence-similarity graph)? Extractive vs abstractive?*
11. *What is ROUGE-1 vs ROUGE-L? Why are your ROUGE scores low?*
12. *What is smoothing in an N-gram model? Why do you need Laplace / Kneser-Ney? What is perplexity?*
13. *Accuracy vs F1: when does accuracy mislead? Read your confusion matrix: which errors dominate?*
14. *Why not just use BERT? What would fine-tuning cost? What is the 512-token limit?*
15. *How did you avoid data leakage (fit the vectorizer on train only)?*
16. *What is novel in your project compared with existing GitHub projects?*
17. *How would you extend it to Hindi/Hinglish reviews?*
18. *Demo request:* "type a sarcastic review" or "not bad at all". Be ready to explain the failure gracefully.

### Gaps
- No published list of viva questions for this specific project type was found. The list above is expert judgment based on the syllabus and the methods involved.
