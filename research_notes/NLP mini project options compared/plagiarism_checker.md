# Plagiarism / Text Similarity Checker — feasibility and build plan (2-person UG NLP mini-project, due 5 Nov 2026)

## Q1. Which public benchmark datasets exist for evaluation, and which are easiest for students?

### Takeaway
For a 2-week project, use **two small datasets**: the **Clough & Stevenson short-answer plagiarism corpus** (about 100 texts with graded plagiarism labels, built for exactly this task) for document-level plagiarism, and **MRPC** (about 5.8k sentence pairs, binary labels, one line to load from Hugging Face) for sentence-level paraphrase detection. PAWS (adversarial, high word overlap) is a good optional "stress test" that shows where n-gram methods fail. The PAN corpora are the real research benchmarks, but they are large, XML-annotated, passage-level alignment tasks. Treat them as stretch goals only.

### Cited Findings
**Clough & Stevenson corpus (2011)**
- Citation: Clough, P. & Stevenson, M. (2011) "Developing a corpus of plagiarised short answers", *Language Resources and Evaluation* 45(1): 5–24, DOI 10.1007/s10579-009-9112-1. It is a corpus of answers to short questions with simulated plagiarism, designed to cover plagiarism types missing from existing corpora — [White Rose eprint](https://eprints.whiterose.ac.uk/id/eprint/42922/); [Springer](https://link.springer.com/article/10.1007/s10579-009-9112-1)
- Five short-answer questions on computer-science curriculum topics. A Wikipedia article is the source text for each. There are four levels: near copy (copy-paste from Wikipedia), light revision (synonym substitution, grammar changes), heavy revision, and non-plagiarised — [arXiv 2605.15110 quoting the corpus paper](https://arxiv.org/pdf/2605.15110); [search summary of same](https://eprints.whiterose.ac.uk/42922)
- The original download page was given as http://www.ir.shef.ac.uk/cloughie/resources/plagiarism_corpus.html — [Springer search snippet](https://link.springer.com/article/10.1007/s10579-009-9112-1). **That host did not resolve when I tried it (ENOTFOUND, Oct 2026).** Students will need a mirror.
- A mirror is bundled in the CLiPS *Pattern* library test corpora as `plagiarism-clough&stevenson.csv`: "100 texts: authentic (0), heavy (1) or light revision (2), cut & paste (3)" — [git.xpub.nl mirror of pattern-master/test/corpora](https://git.xpub.nl/Bohye.Woo/bo-graduation/src/commit/52ca77ce4db530594376462d9c69a6f81193ae26/nltk-book/pattern-master/test/corpora)
- A 2017 paraphrase-plagiarism paper reports a combined-method F1 of 89.746 on this corpus. The same paper notes the corpus is small and may contain annotation errors — [arXiv 1712.10309 (via search snippet; the abstract page itself gives no numbers)](https://arxiv.org/pdf/1712.10309)

**MRPC (Microsoft Research Paraphrase Corpus)**
- In the GLUE collection on Hugging Face: about 5.8k rows, labels `not_equivalent`(0) / `equivalent`(1), 1.5 MB. The license is listed as "other", and the original MRPC terms apply — [HF nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue)
- The MRPC test set is imbalanced (about 67% paraphrase, 33% not), so accuracy must be compared with a majority-class baseline of about 0.67 — [CEUR Vol-3202 parmex-paper2](https://ceur-ws.org/Vol-3202/parmex-paper2.pdf)

**Quora Question Pairs (QQP)**
- In GLUE: about 795k rows (this total includes the unlabeled test split), labels `not_duplicate`/`duplicate`, 107 MB. Quora's own terms apply — [HF nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue)

**STS Benchmark (STS-B)**
- In GLUE: about 8.63k rows, a float similarity score from 0 to 5 (regression, not binary), 1.2 MB — [HF nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue)

**PAWS**
- 108,463 human-labeled paraphrase/non-paraphrase pairs with *high lexical overlap*, built by controlled word swapping and back-translation. There are also 656k noisily labeled pairs. Models trained on earlier datasets scored under 40% accuracy on PAWS, and adding PAWS training data raised this to about 85% — [arXiv 1904.01130](https://arxiv.org/abs/1904.01130?context=cs); [TFDS paws_wiki](https://www.tensorflow.org/datasets/catalog/paws_wiki)
- PAWS-Wiki labeled_final splits: train 49,401 / dev 8,000 / test 8,000. PAWS-QQP cannot be redistributed because of the QQP license. You have to rebuild it from the original QQP plus an index file and script (11,988 train / 677 dev+test examples) — [GitHub google-research-datasets/paws](https://github.com/google-research-datasets/paws)

**PAN plagiarism corpora (Webis)**
- PAN-PC-11 (Zenodo DOI 10.5281/zenodo.3250094) is free for research use. It contains automatically inserted plagiarism (from a "random plagiarist" program) and manually written plagiarism collected via Amazon Mechanical Turk — [DataCite/Zenodo record](https://commons.datacite.org/doi.org/10.5281/zenodo.3250094)
- The PAN text-alignment task (2012–2015) asks systems to find reused passages between suspicious–source document pairs under various obfuscations. Systems output one XML file per pair listing detected cases — [PAN 2012 text alignment](https://pan.webis.de/clef12/pan12-web/text-alignment.html); [PAN 2013](https://pan.webis.de/clef13/pan13-web/text-alignment.html); [PAN 2015 corpus construction](https://pan.webis.de/clef15/pan15-web/text-alignment.html)
- Real plagiarism cases cannot be published as evaluation collections because they cannot be properly anonymised. This is why every benchmark is simulated or synthetic — [arXiv 1603.03014 / search summary](https://arxiv.org/pdf/1603.03014)

### Inferences
- **Easiest path:** use `datasets.load_dataset("nyu-mll/glue","mrpc")` for MRPC. For Clough, grab the 100-text CSV from the Pattern-library mirror (or ask the lab instructor / search Kaggle for a copy). With both datasets the team can report document-level and sentence-level results within the first 2–3 days.
- **Clough labels:** collapse them to binary (plagiarised = cut/light/heavy vs. non) for P/R/F1 and ROC. Then show a per-category score boxplot. The expected ordering, cut > light > heavy > non, makes a strong report figure because it shows the methods degrade as obfuscation increases.
- **PAWS** is the best dataset for showing *why* embeddings are needed and also where they still fail. Its pairs share almost all words but can differ in meaning, so Jaccard/TF-IDF will rate them highly similar regardless of label.
- **PAN** is too heavy for 2 weeks (large corpus, passage-level offsets, XML evaluation). At most, mention it in related work or run a tiny subset.

### Gaps
- I could not verify the exact Clough corpus counts from the primary source (the paper PDF text would not extract). Commonly cited figures are 95 answers by 19 participants plus 5 Wikipedia originals, but **students should verify these from the downloaded files** before quoting them.
- I did not confirm the exact licenses for PAWS (the repo links a LICENSE file I did not open), STS-B, or the Clough corpus. Students should check each dataset page and state "used for academic/research purposes" in the report.
- I did not verify per-split MRPC counts (HF shows only the 5.8k total). The commonly cited split is 3,668 train / 408 validation / 1,725 test; students should confirm it when loading.

---

## Q2. Techniques & libraries: how do they compare on paraphrase detection, and what accuracy/F1 is typical?

### Takeaway
Lexical methods (word n-gram/shingle Jaccard or containment, winnowing fingerprints, TF-IDF cosine, LCS via difflib) work very well for copy-paste and light edits, and they are fast and explainable. They break down on heavy paraphrase. Sentence embeddings (sentence-transformers MiniLM) capture meaning and handle paraphrase much better, but they are slower and their scores are less interpretable. Embedding-based paraphrase classifiers on MRPC sit around 70–76% accuracy against a 67% majority baseline, so the realistic message is that MRPC is hard and that a combined/hybrid score is best.

### Cited Findings
**N-gram containment / LCS**
- Containment C_n(A,B) for n = 1..10 is the size of the shared n-gram set divided by the size of one document's n-gram set. It is a standard assignment-style feature for plagiarism detection — [UBC COSC 419 assignment](https://cmps-people.ok.ubc.ca/bowenhui/analytics/asgns/a6.html)
- LCS allows gaps: matched tokens need not be consecutive, unlike n-grams. A longer LCS suggests more similar texts, and an F-measure can be computed from LCS — [arXiv 1001.3487 / 1003.4065 (search summary)](https://arxiv.org/pdf/1003.4065)
- A feature-engineering repo for this exact corpus style exists: grantgasser/plagiarism-detection-nlp (tagged "containment") — [GitHub topic: plagiarism-detector](https://github.com/topics/plagiarism-detector)

**Winnowing (MOSS)**
- Schleimer, Wilkerson & Aiken, "Winnowing: Local Algorithms for Document Fingerprinting", SIGMOD 2003. The paper introduces local fingerprinting algorithms, proves a lower bound for them, and shows winnowing is within 33% of that bound — [author page](https://sschleimer.warwick.ac.uk/Maths/winnowing.html); [PDF](https://sschleimer.warwick.ac.uk/Maths/winnowing.pdf)
- Guarantee: winnowing detects at least one shared k-gram in any shared substring of length ≥ w + k − 1, where w is the window size and k is the k-gram length. Matches shorter than k (the noise threshold) are never detected — [winnowing paper via search](https://www.cs.princeton.edu/courses/archive/spring05/cos598E/bib/p76-schleimer.pdf); [Ruby winnow gem docs](https://www.rubydoc.info/gems/winnow)
- Patent note: some applications are patented by the UC Regents, but the claims cover only comparing a query document against web content — [author page](https://sschleimer.warwick.ac.uk/Maths/winnowing.html)

**MinHash / LSH (datasketch)**
- `MinHash(num_perm=128)` per set and `MinHashLSH(threshold=0.5, num_perm=128)`, with `insert` and `query`, approximate Jaccard similarity. A set is returned with probability 1 − (1 − s^r)^b. Results are approximate (false positives and negatives), queries return unranked candidates, and there is no top-k (use LSH Forest for that). Larger num_perm gives more accuracy at the cost of more memory — [datasketch LSH docs](https://ekzhu.com/datasketch/lsh.html)
- In the datasketch benchmark, precision was about 0.69–0.77 and recall about 0.69–0.84 on corpora with many pairs near the threshold. It performs much better on near-duplicates — [datasketch LSH docs](https://ekzhu.com/datasketch/lsh.html)

**Sentence embeddings (sentence-transformers)**
- all-mpnet-base-v2 gives the best quality, while all-MiniLM-L6-v2 is "5 times faster and still offers good quality". paraphrase-multilingual-MiniLM-L12-v2 covers 50+ languages without specifying the input language — [SBERT pretrained models docs](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
- all-MiniLM-L6-v2: 22.7M parameters, 384-dim embeddings, Apache-2.0 license — [HF model card derivative (search summary)](https://huggingface.co/codersan/all-MiniLM-L6-v2-Fa-v3)
- `util.paraphrase_mining` returns (score, i, j) triples of the highest-cosine pairs and avoids quadratic brute force on big collections. This fits the "find matching sentences across documents" feature well — [SBERT paraphrase mining docs](https://www.sbert.net/examples/applications/paraphrase-mining/README.html)
- SBERT paper SentEval MRPC accuracy (logistic regression on frozen embeddings, 10-fold CV): Avg GloVe 72.87, Avg fastText 74.49, Avg BERT 69.45, BERT CLS 71.13, InferSent 75.77, USE 70.14, SBERT-NLI-base 76.00, SBERT-NLI-large 75.94 — [Reimers & Gurevych 2019, arXiv 1908.10084, Table 5](https://arxiv.org/pdf/1908.10084)
- The same paper's unsupervised STS-B Spearman correlations: Avg GloVe 58.02, BERT CLS 16.50, USE 74.92, SBERT-NLI-base 77.03. Fine-tuned on STS-B: about 84–86 — [arXiv 1908.10084, Tables 1–2](https://arxiv.org/pdf/1908.10084)

**Threshold-based MRPC numbers**
- A cosine baseline on MRPC with threshold 0.93 reached 0.85 accuracy but only 0.54 positive-class F1 — [CEUR Vol-3202 parmex-paper2 (via search summary; unclear whether the threshold was tuned on labels)](https://ceur-ws.org/Vol-3202/parmex-paper2.pdf)
- A 2025 *Computers* paper found that optimal sentence-embedding cosine thresholds on MRPC varied widely across models (0.334–0.867), so no single generalisable cut-off exists — [DOAJ, Computers Sep 2025 (via search snippet; page returned 403)](https://doaj.org/article/07b42863f3ea496cb3572f121c7543cc)
- Classical TF-IDF cosine fed into an SVM "performs better than expected" on lexically diverse corpora — [arXiv 2303.13989](https://arxiv.org/pdf/2303.13989)

### Inferences
- **Recommended method set (5 scores, all free and laptop-friendly):**

| Method | Library | Catches | Misses |
|---|---|---|---|
| Word n-gram (3–5) shingle Jaccard + containment | plain Python / NLTK `ngrams` | copy-paste | paraphrase, reordering |
| Winnowing fingerprints | ~40 lines of custom code | copy-paste with small edits, scales | paraphrase |
| TF-IDF cosine (word + char n-grams) | scikit-learn | light revision, reordering | synonyms |
| LCS ratio | `difflib.SequenceMatcher` | ordered copying | reordering, synonyms |
| Sentence-embedding cosine | sentence-transformers all-MiniLM-L6-v2 | paraphrase, synonyms | negation/word-swap traps (PAWS) |

- **Combined score:** use a simple weighted average, or a logistic regression trained on the 5 scores as features. Training on Clough/MRPC usually beats every single method, and it gives a "trained model" result for the report.
- MiniLM runs on CPU in seconds for a few documents, and in a few minutes for the full MRPC test set (1.7k pairs). No GPU is needed. The model is roughly 90 MB to download once.
- **Expected results to sanity-check against** (estimates, not cited): on Clough binary, lexical methods should reach high F1 (likely 0.85–0.95) because most plagiarised answers are cut/light. On MRPC, all unsupervised thresholds will land roughly in the 0.70–0.82 F1 range, close to the "always paraphrase" baseline (F1 ≈ 0.80). Report this honestly.

### Gaps
- I found no authoritative published table of all-MiniLM-L6-v2 threshold accuracy/F1 on MRPC, or of TF-IDF threshold F1 on MRPC. The students must compute these themselves, which is actually a strength for the project.
- I could not confirm a clean published per-method (n-gram vs LCS vs TF-IDF) results table on the Clough corpus from a primary source.

---

## Q3. How to present results credibly

### Takeaway
Treat each similarity method as a binary classifier ("plagiarised if score ≥ τ"). Tune τ on a training/validation split, report precision, recall, F1 and accuracy on a held-out test split next to a majority-class baseline, and plot ROC/PR curves with AUC for all methods on one figure. A method comparison table plus a per-category boxplot (Clough) is the core results section.

### Cited Findings
- MRPC is about 67/33 imbalanced, so 0.85 accuracy has to be compared with the about-0.67 majority baseline, and positive-class F1 can be low (0.54) even when accuracy looks high. Report F1 for the paraphrase class plus the majority baseline, with τ chosen on the training split — [CEUR parmex-paper2 (search summary)](https://ceur-ws.org/Vol-3202/parmex-paper2.pdf)
- Optimal cosine thresholds vary from 0.334 to 0.867 across embedding models on MRPC, so the threshold must be tuned per method, never fixed at 0.5 — [Computers 2025 via DOAJ snippet](https://doaj.org/article/07b42863f3ea496cb3572f121c7543cc)
- PAN-style evaluation adds passage-level measures (plagdet, granularity), which are overkill here — [PAN 2013 text alignment](https://pan.webis.de/clef13/pan13-web/text-alignment.html)

### Inferences
- **Results checklist:**
  1. Method comparison table (rows = methods; columns = best τ, P, R, F1, Acc, ROC-AUC) for Clough and MRPC, with a majority-baseline row.
  2. One ROC plot and one PR plot overlaying all methods (`sklearn.metrics.roc_curve`, `precision_recall_curve`, matplotlib). This covers syllabus experiment 1 (graphs).
  3. Boxplot/violin of scores per Clough category (cut/light/heavy/non) for each method.
  4. F1-vs-threshold curves showing how sensitive each method is to τ.
  5. Error analysis: 5–10 false positives/negatives (e.g., PAWS-style word swaps fooling Jaccard, heavy paraphrases missed by n-grams).
  6. Runtime table (ms per document pair) on a laptop.
- Use stratified train/test or 5-fold CV for Clough, since 100 texts is small. Mention the small-sample caveat.

### Gaps
- None material. These are standard ML evaluation practices.

---

## Q4. Existing GitHub projects/tutorials and how to differentiate

### Takeaway
Many plagiarism checkers already exist (TF-IDF cosine scripts, side-by-side sentence highlighters, code tools like JPlag/Dolos/MOSS). A plain "TF-IDF cosine of two files" will look generic. Differentiation should come from (a) **multi-method comparison with proper evaluation on benchmarks**, (b) **sentence-level side-by-side highlighting with a match-type label** (exact vs paraphrase), and (c) **an N×N multi-document heatmap**. Cross-lingual detection via multilingual MiniLM is a cheap, impressive extension (e.g., English vs Hindi). AI-generated-text detection should only appear as a clearly caveated experiment or be left out.

### Cited Findings
- GitHub topic "plagiarism-detector": JPlag (about 2k stars, Java, code), dodona-edu/dolos (353 stars, code), Wazzabeee/copy-spotter (58 stars, Python, side-by-side similar-sentence view), grantgasser/plagiarism-detection-nlp (containment feature engineering), Karthik-02/plagiarism-detection (cosine), scriptographers/UnPlag (TF-IDF + AST for C++/Python/text), a-matahen/plagiarism_detection (TF-IDF, cosine, Jaccard) — [GitHub topics/plagiarism-detector](https://github.com/topics/plagiarism-detector)
- An existing Streamlit app compares a source sentence with up to three others using a sentence-transformer and cosine, without highlighting. A Flask system (IRJET) does TF-IDF cosine at document and sentence level with highlighting — [contra.com Text Similarity](https://contra.com/p/7Y6BvWmq-text-similarity); [IRJET via issuu](https://issuu.com/irjet/docs/an_intelligent_plagiarism_detection_system_using_t)
- The multilingual model paraphrase-multilingual-MiniLM-L12-v2 is trained on parallel data for 50+ languages, so cross-language similarity works without specifying the language — [SBERT docs](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
- **AI-detection caveat:** OpenAI withdrew its AI Text Classifier on 20 July 2023 because of its "low rate of accuracy". It flagged only 26% of AI-written text as likely AI and mislabelled 9% of human text as AI, and it was very unreliable under 1,000 characters — [AI Business](https://aibusiness.com/responsible-ai/openai-quietly-shuts-ai-plagiarism-detector-tool); [Search Engine Journal](https://searchenginejournal.com/openai-shuts-down-flawed-ai-detector/492565)
- A Stanford study of 7 GPT detectors found 61% of TOEFL essays by non-native (Chinese) English writers were misclassified as AI-generated, while US 8th-grade essays were classified well — [AI Business / Stanford study coverage](https://aibusiness.com/responsible-ai/openai-quietly-shuts-ai-plagiarism-detector-tool)

### Inferences
- **Differentiators, ranked by value per effort:**
  1. *(High value, low effort)* **Side-by-side highlighter:** split each document into sentences, compute a sentence×sentence similarity matrix, and colour pairs red (exact/near-copy: n-gram Jaccard ≥ τ1) or orange (paraphrase: embedding cosine ≥ τ2 but low Jaccard). The explicit "paraphrase vs copy" label is the novel UX piece.
  2. *(High, low)* **Multi-document N×N similarity heatmap** (seaborn/plotly) for uploading a whole class's assignments, with clustering of suspicious groups. MinHash LSH can serve as a scalable candidate-filter demo.
  3. *(High, medium)* **Benchmark + ablation:** per-method results on Clough + MRPC + PAWS. Few student repos evaluate at all.
  4. *(Medium, low)* **Cross-lingual check** with the multilingual MiniLM, e.g., an English source vs a Hindi/Marathi translation. Demo-quality only; there is no labelled benchmark here. The students could make 20–30 pairs themselves with free translation.
  5. *(Medium, medium)* **Linguistic "obfuscation report"** reusing the syllabus: shows how many matched content words were replaced by synonyms (WordNet), POS-pattern changes, etc.
  6. *(Low priority)* **AI-generated text detection:** present it only as "perplexity under our own N-gram LM", tied to syllabus experiment 6, as an exploratory signal with an explicit unreliability disclaimer citing OpenAI's withdrawn classifier and the Stanford bias study. Never present it as a verdict.
  7. *(Out of scope)* **Code plagiarism:** this is a different problem (tokens/ASTs; JPlag/Dolos/MOSS already do it). Mention it only as future work.

### Gaps
- I did not verify code quality or licenses of the listed repos. Students should cite any code they reuse.
- Star counts are as shown on the GitHub topic page at the time of fetching (Oct 2026).

---

## Q5. Realistic difficulty, time estimate, risks, and work split

### Takeaway
Difficulty is **low-to-moderate**. The core (5 similarity methods + Streamlit upload/compare + evaluation on Clough and MRPC) is achievable in about 8–10 working days by two students. The main risks are dataset access (the Clough host is down), threshold tuning and honest MRPC numbers, PDF/DOCX text extraction, and over-scoping the AI-detection or code-plagiarism extensions.

### Cited Findings
- all-MiniLM-L6-v2 is the fast option (5× faster than mpnet) with good quality, which suits CPU-only laptops — [SBERT docs](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
- The Clough corpus's original Sheffield URL is unreachable (my fetch failed). A mirror exists in the Pattern library's test corpora — [git.xpub.nl mirror](https://git.xpub.nl/Bohye.Woo/bo-graduation/src/commit/52ca77ce4db530594376462d9c69a6f81193ae26/nltk-book/pattern-master/test/corpora)
- LSH queries are approximate, with false positives and negatives. Use it only as a candidate filter and verify the candidates — [datasketch docs](https://ekzhu.com/datasketch/lsh.html)

### Inferences
- **Timeline (20 Oct → 5 Nov 2026, about 16 days):**
  - Day 1–2: set up the environment (Python 3.10+, nltk, spacy, scikit-learn, sentence-transformers, datasketch, streamlit, pypdf, python-docx). Download MRPC/PAWS from HF and get the Clough CSV. Do the EDA plots.
  - Day 3–5: preprocessing pipeline plus the 5 similarity functions, with unit tests on toy pairs.
  - Day 6–8: evaluation harness (threshold tuning, P/R/F1, ROC, table) on Clough + MRPC (+ PAWS).
  - Day 7–11 (in parallel): Streamlit UI with upload (txt/pdf/docx), score cards, side-by-side highlighting, N×N heatmap.
  - Day 12–13: extensions (cross-lingual demo, N-gram LM perplexity panel, combined logistic-regression score).
  - Day 14–16: report, slides, demo video, viva prep. Keep 1–2 buffer days.
- **Split:**
  - Person A (NLP/eval): preprocessing, n-gram/winnowing/LCS/TF-IDF, N-gram LM, dataset loading, evaluation, results figures, error analysis.
  - Person B (semantic/UI): sentence-transformers integration, sentence alignment and highlighting, Streamlit app, file parsing, heatmap, cross-lingual demo.
  - Shared: report chapters (each writes their own modules), README, the combined-score model, slides.
- **Risks and mitigations:**
  - Dataset unavailability: mirror plus an instructor fallback. Worst case, build a small self-made corpus (e.g., 10 source paragraphs × 4 obfuscation levels written by friends, or paraphrased with a free local tool), clearly labelled as synthetic.
  - First model download needs internet (about 90 MB). Cache it before the demo.
  - PDF extraction noise (hyphenation, headers). Show a .txt demo path as a backup.
  - Quadratic sentence comparisons on long documents. Cap the length, or use `paraphrase_mining` / LSH.
  - Over-claiming: never claim to "detect AI text" or that MRPC numbers are state of the art.

### Gaps
- The time estimates are my inference from typical UG skill levels, not from a cited source.

---

## Q6. How well does it map to the syllabus experiments?

### Takeaway
The mapping is strong for experiments 1, 2, 3, 4 and 6 (**N-grams are central**: shingling, containment, winnowing k-grams, plus an optional N-gram LM with smoothing). Experiment 5 (POS/chunking) is partial: it can be used for content-word filtering and for showing syntactic-restructuring obfuscation. Experiments 7 (sentiment/subjectivity) and 8 (summarisation) are weak and need deliberate small hooks.

### Cited Findings
- Containment over n-grams for n = 1..10 is a canonical plagiarism feature — [UBC COSC 419](https://cmps-people.ok.ubc.ca/bowenhui/analytics/asgns/a6.html)
- Winnowing operates over k-grams (character or token n-grams) with window w — [Schleimer et al. 2003](https://sschleimer.warwick.ac.uk/Maths/winnowing.html)
- The light-revision category in Clough involves synonym substitution and grammatical restructuring, which links naturally to lemmatisation/WordNet and POS analysis — [arXiv 2605.15110 quoting Clough & Stevenson](https://arxiv.org/pdf/2605.15110)

### Inferences
| Syllabus expt | Coverage | How it appears in the project |
|---|---|---|
| 1. CSV + matplotlib | Strong | Load the Clough CSV / MRPC TSV with pandas; ROC/PR curves, score boxplots, heatmaps |
| 2. Word/sentence tokenisation | Strong | Sentence splitting for side-by-side alignment; word tokens for shingles |
| 3. Stopwords, lemmatisation, morphology | Strong | Preprocessing ablation (with/without stopwords/lemmas) and its effect on Jaccard/TF-IDF; lemmatisation defeats inflection changes |
| 4. Word count, top-10, word cloud, NER | Moderate | Document stats panel; word clouds of *shared* vs *unique* words; NER overlap (shared named entities as a plagiarism signal) |
| 5. POS tagging, chunking | Partial | Restrict shingles to content-word POS (NOUN/VERB/ADJ); compare noun-phrase chunks between documents; detect POS-sequence changes in light revisions |
| 6. N-gram LM + smoothing | Strong | Word n-gram shingles are the core method. Extension: train a Laplace/Kneser-Ney smoothed n-gram LM (NLTK `nltk.lm`) on the source and compute the suspicious text's perplexity (a lower-perplexity passage suggests copying). The same LM can serve as a caveated "AI-like text" signal |
| 7. Sentiment/subjectivity | Weak | Optional: TextBlob polarity/subjectivity of matched pairs as a meaning-preservation check (paraphrase keeps sentiment; e.g., a negation flip reverses it). PAWS-style traps |
| 8. Summarisation | Weak/optional | Optional: an extractive summary (TF-IDF sentence scoring) of the matched region, or detection of "summary obfuscation" (PAN 2013+ includes summary-based obfuscation) |

### Gaps
- None. This mapping is an inference from the project design.

---

## Q7. Likely viva/examiner questions

### Takeaway
Expect questions on the math of each similarity measure, why multiple methods are needed, the choice of n/k/w and threshold, the evaluation methodology and dataset limits, how embeddings work, and the honesty of the AI-detection claims.

### Cited Findings
- Winnowing guarantee (w + k − 1) and noise threshold k — [Schleimer et al.](https://sschleimer.warwick.ac.uk/Maths/winnowing.html)
- MinHash LSH probability 1 − (1 − s^r)^b and its approximate nature — [datasketch](https://ekzhu.com/datasketch/lsh.html)
- Threshold variability across models — [Computers 2025 snippet](https://doaj.org/article/07b42863f3ea496cb3572f121c7543cc)
- AI detector unreliability — [Search Engine Journal](https://searchenginejournal.com/openai-shuts-down-flawed-ai-detector/492565)

### Inferences
Likely questions, with answer pointers:
1. Define Jaccard vs containment vs cosine. When is containment better? (When a short suspicious text is copied from a long source.)
2. Why word 3-grams/5-grams and not unigrams? (Unigrams overlap on any same-topic text.) Show the F1-vs-n plot.
3. How does winnowing pick fingerprints, and what does the w + k − 1 guarantee mean?
4. TF-IDF: what does IDF do, and why use sublinear tf or char n-grams? Why can TF-IDF miss synonyms?
5. What does LCS/difflib measure? What is its time complexity (O(mn) for classic LCS)? Why does reordering defeat it?
6. How does a sentence-transformer produce a sentence vector (mean pooling, contrastive/siamese training)? Why is cosine used? Why MiniLM over BERT-CLS? (BERT-CLS STS-B Spearman is only 16.5 vs 77 for SBERT.)
7. How did you choose the threshold? Did you tune it on the test set? (It must be a train/val split.)
8. Why report F1 rather than accuracy on MRPC? What is the majority baseline?
9. What are the limitations of the Clough corpus (small, simulated, 5 topics)? Can you generalise to real student essays?
10. Where does your system fail? (PAWS word swaps; heavy paraphrase; translated plagiarism if the monolingual model is used; common phrases/boilerplate causing false positives.)
11. How would you scale to 10,000 documents? (MinHash LSH/inverted index candidate filtering, then exact verification.)
12. What role do stopword removal and lemmatisation play? Show the ablation.
13. Can your tool detect ChatGPT-written text? (Answer honestly: not reliably. OpenAI withdrew its own classifier; detectors are biased against non-native writers.)
14. Ethics: false accusations, so the tool should *flag for human review*, not decide. There is also privacy of uploaded documents (process locally).
15. How is this different from Turnitin/MOSS? (No web-scale source index; only pairwise/within-batch comparison; open and explainable multi-method scores.)

### Gaps
- Specific examiner question lists from Indian university NLP labs were not searched. The list above is inferred from the project's technical content.
