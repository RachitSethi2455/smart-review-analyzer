# Resume–Job Description Matcher (2-person undergrad NLP mini-project, due 5 Nov 2026)

## Which public datasets exist (sizes, labels, licensing, links)?

### Takeaway
There are enough free datasets. Two Kaggle resume-category datasets (962 and about 2,484 resumes) support a classification experiment. One HF dataset of resume–JD pairs labelled Good/Potential/No fit gives the only ready-made ground truth for the match score. ESCO and O*NET supply free skill dictionaries, and LinkedIn postings data (124k+) supplies real JDs. Licences are mostly unconfirmed, so the students must read each Kaggle/HF page before use.

### Cited Findings
- **Kaggle "Resume Dataset" (Snehaan Bhawal)**: "2400+ resumes" in string and PDF format. Listings show 2,484 rows and **24 categories** (HR, Designer, Information-Technology, Teacher, Advocate, Business-Development, Healthcare, Fitness, Agriculture, BPO, Sales, Consultant, Digital-Media, Automobile, Chef, Finance, Apparel, Engineering, Accountant, Construction, Public-Relations, Banking, Arts, Aviation). Columns: ID, Resume_str, Resume_html, Category. PDFs sit in folders named by category. Scraped from livecareer.com. — [Baselight mirror listing (search snippet)](https://baselight.app/u/kaggle/dataset/snehaanbhawal_resume_dataset); also used in [arXiv 2406.18125](https://www.arxiv.org/pdf/2406.18125)
  - Kaggle page: kaggle.com/datasets/snehaanbhawal/resume-dataset. Licence not confirmed (the Baselight mirror shut down in July 2026, so it can't be checked there) — [baselight.ai](https://baselight.ai/u/kaggle/dataset/snehaanbhawal_resume_dataset)
- **UpdatedResumeDataSet.csv** (Kaggle, e.g. "jillanisofttech/updated-resume-dataset"): **962 resumes, 25 categories**, columns Category + Resume. The Analytics Vidhya tutorial uses it. — [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/06/resume-screening-with-natural-language-processing-in-python/); [Baselight listing](https://baselight.app/u/kaggle/dataset/jillanisofttech_updated_resume_dataset)
- **HF `cnamuangtoun/resume-job-description-fit`**: CSV text data, size class 1K–10K rows, 81 likes. A University of Toronto student report describes it as resume text + JD text + label in {Good Fit, Potential Fit, No Fit}. — [HF dataset page](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit); [UofT "ResuMatch" report](https://www.eecg.utoronto.ca/~jayar/1786_24_projects/reports/resumatch.pdf)
  - A DistilBERT model fine-tuned on it reports only about **50% accuracy / macro-F1 about 0.43**, and its card notes that "a TF-IDF baseline matched or beat this model on macro-F1." The task is hard and noisy. — [domynom/resume-job-distilbert](https://huggingface.co/domynom/resume-job-distilbert)
- **LinkedIn Job Postings 2023–2024 (Kaggle, arshkon)**: 124,000+ postings, with separate files for skills, benefits, and industries per posting, plus description and salary fields. Licence not confirmed in search. Check kaggle.com/datasets/arshkon/linkedin-job-postings. — [Baselight listing](https://baselight.app/u/kaggle/dataset/arshkon_linkedin_job_postings)
  - Related Kaggle sets: asaniczka "data science job postings and skills" and "software engineer job postings LinkedIn". The rajatraj0502 "LinkedIn job 2023" set is ODbL. — [Baselight](https://baselight.app/u/kaggle/dataset/asaniczka_data_science_job_postings_and_skills); [Baselight](https://baselight.app/u/kaggle/dataset/rajatraj0502_linkedin_job_2023)
- **ESCO (EU skills taxonomy) v1.2** (May 2024): **13,939 skills, 3,039 occupations**, 28 languages. Downloadable as CSV/ODS/SKOS-RDF. The main file is `skills_en.csv`, and an API is also available. — [ESCO v1.2 launch slides](https://esco.ec.europa.eu/system/files/2024-05/V1.2%20Launch%20-%20ESCO%20SEC%20combined%20-%20final%20clean.pdf); [ESCO downloadable dataset structure](https://esco.ec.europa.eu/en/structure-esco-downloadable-datasets); [Wikipedia](https://en.wikipedia.org/wiki/European_Skills,_Competences,_Qualifications_and_Occupations)
- **O*NET Technology Skills file**: about 31,078 rows (v26.0), mapping O*NET-SOC occupations to software/technology examples, with a "Hot Technology" Y/N flag for tools frequently in job postings. Available as Excel/text/SQL. — [O*NET Technology Skills dictionary](https://www.onetcenter.org/dictionary/26.0/excel/technology_skills.html)
- **Lightcast (formerly EMSI) Open Skills**: 32,000–35,000+ skills (figures vary by page). The free API tier covers skill names/IDs/types/categories, autocomplete, and limited extraction (one source says 50 extraction requests/month). Access needs registration, and the docs say free access is for nonprofit/public-good use with approval. — [Lightcast free API features](https://docs.lightcast.io/lightcast-api/docs/free-api-features); [API access options](https://docs.lightcast.io/lightcast-api/docs/api-access); [skills.lightcast.io](https://skills.lightcast.io/)
- **SkillNER** (open-source, spaCy PhraseMatcher + the EMSI/Lightcast skill DB bundled as SKILL_DB) gives an offline skill dictionary without API registration. Needs `en_core_web_lg`. About 207 stars, last push Jan 2024. — [GitHub AnasAito/SkillNER](https://github.com/AnasAito/SkillNER); [PyPI skillner](https://pypi.org/project/skillner); [SkillNER paper arXiv 2101.11431](https://arxiv.org/pdf/2101.11431)

### Inferences
- Recommended combination: **UpdatedResumeDataSet (962) or the Snehaan 2,484 set** for the category-classification experiment. **cnamuangtoun pairs** give a labelled match-score evaluation. **A curated skill list** comes from O*NET hot technologies + ESCO filtered to IT/digital skills (or SkillNER's DB), trimmed to about 500–2,000 terms, plus Indian-context additions (e.g., "B.Tech", "NPTEL", "Tally", "SAP").
- The Snehaan PDFs folder is useful for testing the PDF-extraction step on real PDFs without collecting real people's resumes.
- Both Kaggle resume sets are scraped/non-Indian and US-centric. That is worth noting in the report as a domain limitation.

### Gaps
- Licences for the Snehaan dataset, UpdatedResumeDataSet, arshkon LinkedIn, and the cnamuangtoun HF set could not be confirmed (Kaggle/HF card text not retrievable). Students should screenshot the licence on each page for the report.
- Exact train/test row counts and label distribution of cnamuangtoun were not retrievable (the README fetch returned 404).
- ESCO and O*NET reuse licence text was not confirmed in this research (both are generally understood to be free to reuse, but this is unverified here).

## Techniques & libraries (extraction, skill NER, similarity, classification)

### Takeaway
A fully offline, laptop-friendly stack works: pdfplumber/PyMuPDF + python-docx for text, spaCy `en_core_web_sm` + PhraseMatcher/EntityRuler with a skill list, TF-IDF cosine as baseline, and `all-MiniLM-L6-v2` (22.7M params, 384-d, CPU-friendly) as the semantic upgrade. Don't use pyresparser, which is abandoned (last release 2019). Classic ML (TF-IDF + LinearSVC/LogReg/KNN) gets about 99% on the 962-resume category dataset, which is suspiciously easy.

### Cited Findings
- **PDF extraction**: PyMuPDF is **AGPL-3.0**, and pdfplumber is **MIT**. A vendor benchmark (PDF Oxide, a competitor, so treat with caution) reports pdfplumber about 23 ms/doc vs PyMuPDF about 4.6 ms. pdfplumber leads on table extraction. — [pdf.oxide.fyi comparison](https://pdf.oxide.fyi/docs/comparison/pdfplumber-vs-pymupdf)
- A test on synthetic PDFs found pdfplumber was the only one of four libraries (vs PyMuPDF, pypdf, pypdfium2) that failed a plain **two-column PDF at default settings**. All four "scrambled" at least one page. — [Python in Plain English article (summary)](https://python.plainenglish.io/pymupdf-vs-pdfplumber-vs-pypdf-vs-pypdfium2-all-4-got-every-word-all-4-scrambled-one-page-d12054be37bd); see also [Nutrient 2026 Python PDF library comparison](https://www.nutrient.io/blog/best-python-pdf-libraries/)
- **pyresparser**: latest version 1.0.6, released **15 Dec 2019**. Classifiers cover Python 3.3–3.7 only. On Windows it supports .docx/.pdf only (.doc needs textract). No releases since. — [PyPI pyresparser](https://pypi.org/project/pyresparser/)
- **spaCy rule-based matching**: PhraseMatcher takes Doc patterns and "is much faster" than the token Matcher for large terminology lists. `attr="LOWER"` gives case-insensitive matching, and patterns can be built with `nlp.make_doc` (tokenizer only). EntityRuler adds labelled spans from patterns, and the longest match wins on overlap. Placed **before `ner`**, the statistical NER respects the ruler's spans. Pattern `id`s let synonyms share one ID (`ent_id_`), which suits normalising "JS"/"JavaScript". Batched pattern adding gives a 10–20x speedup for 5k–100k patterns. — [spaCy rule-based matching docs](https://spacy.io/usage/rule-based-matching)
- **SkillNER** = curated skills DB + spaCy PhraseMatcher. It returns full matches and n-gram (partial) matches with skill IDs, and covers hard and soft skills. — [GitHub AnasAito/SkillNER](https://github.com/AnasAito/SkillNER); [CodeCut tutorial](https://codecut.ai/skillner-automating-skill-extraction-in-python/)
- **all-MiniLM-L6-v2**: 384-dim embeddings, about **22.7M parameters**, Apache-2.0. Inputs over **256 word pieces are truncated** (trained with 128). Intended for sentence/short-paragraph similarity, clustering, and IR. Trained on more than 1B sentence pairs. — [HF model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **TF-IDF vs SBERT**: a 2025 NCI master's thesis comparing SBERT and TF-IDF for resume–job matching (nDCG, MRR, P@K, against HR-experienced human judgments) found SBERT ranked better, while TF-IDF was faster. — [Power 2025, NCI](https://norma.ncirl.ie/9765/1/alexpower.pdf)
- On the cnamuangtoun pairs, a TF-IDF baseline matched or beat fine-tuned DistilBERT on macro-F1. — [domynom/resume-job-distilbert](https://huggingface.co/domynom/resume-job-distilbert)
- **Category classification**: on UpdatedResumeDataSet (962/25), the Analytics Vidhya tutorial gets about 0.99 train/test accuracy with KNeighborsClassifier (OneVsRest) on TF-IDF. Per-class test supports are tiny (3–9). — [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/06/resume-screening-with-natural-language-processing-in-python/)
- A Thai thesis on the same 962-row set compared 8 classifiers. SVC/LogReg/RandomForest reached 99.48%, and SVC got 99.50% with cross-validation. — [SWU thesis](https://ir-ithesis.swu.ac.th/handle/123456789/2762)

### Inferences
- **The roughly 99% on UpdatedResumeDataSet is likely inflated by near-duplicate resumes** (the dataset is widely known to contain many repeated rows; not verified here). Students should deduplicate (`df.drop_duplicates('Resume')`) and report accuracy before and after. That gives the report an honest, differentiating result. The 24-class Snehaan set is the harder and more credible benchmark. It is reported in the literature with much lower accuracy, but those numbers are unverified here.
- Because MiniLM truncates at 256 word pieces, embedding a whole resume only "sees" the first part. Chunk by section or sentence, embed each, and take max/mean similarity against JD requirement sentences. This is also a good viva point.
- Prefer **pdfplumber (MIT)** with PyMuPDF as a fallback for messy/two-column PDFs. AGPL doesn't matter for a non-distributed student project, but mentioning the licence shows awareness.
- Pipeline sketch: extract text → clean (regex for emails/phones/URLs) → sentence/word tokenize → stopwords + lemmatize → section split by heading regex (Education/Experience/Skills/Projects) → PhraseMatcher skills (LOWER attr, lemma-normalised) + spaCy NER (ORG for colleges/companies, DATE for experience span) + noun-chunk keywords for the JD → scores: (a) TF-IDF cosine, (b) MiniLM cosine, (c) skill-overlap = |R∩J|/|J| → weighted final score → missing skills = J − R.
- Skip pyresparser and build the parser yourselves. That is also more defensible in a viva.

### Gaps
- No independent, non-vendor benchmark on real resume PDFs (two-column, icons, tables) was found for pdfplumber vs PyMuPDF.
- No verified accuracy figure was found for the Snehaan 24-class dataset with classic ML.
- python-docx was not specifically researched. It is standard and reliable for paragraph text, but text inside DOCX tables/text boxes needs extra handling (unverified).

## How to evaluate "match score" credibly without ground truth?

### Takeaway
Combine three things: (1) a labelled benchmark (the cnamuangtoun Good/Potential/No-fit pairs) where you threshold or classify each score and report accuracy/macro-F1 plus score distributions per label; (2) a small hand-annotated ranking set (e.g., 5 JDs × 10 resumes rated 0–3 by both students), evaluated with Spearman correlation, nDCG@k, and P@k, with inter-annotator agreement; (3) skill-extraction precision/recall on about 20–30 manually tagged resumes. Published research uses exactly these metrics.

### Cited Findings
- Research uses human labels from recruiters/consultants as ground truth. conSultantBERT (Randstad) uses consultants' decisions on resume–vacancy pairs, compares against TF-IDF and pretrained-BERT cosine baselines, and finds a fine-tuned Siamese SBERT best. — [conSultantBERT, arXiv 2109.06501](https://arxiv.org/pdf/2109.06501)
- Ranking metrics nDCG, MRR, and Precision@K against judgments from HR-experienced participants are used to compare SBERT and TF-IDF. — [Power 2025, NCI thesis](https://norma.ncirl.ie/9765/1/alexpower.pdf)
- ConFit reports nDCG@10 for resume–job matching. A TF-IDF + XGBoost variant is competitive with BERT-based person–job-fit methods. — [ConFit, arXiv 2401.16349](https://arxiv.org/pdf/2401.16349)
- The cnamuangtoun 3-class fit labels give a classification-style evaluation. The fine-tuned DistilBERT reference point is about 50% accuracy, macro-F1 about 0.43. — [domynom/resume-job-distilbert](https://huggingface.co/domynom/resume-job-distilbert)

### Inferences
- Concrete eval plan within 2 weeks:
  1. **Benchmark**: on about 500–1,000 cnamuangtoun pairs, compute TF-IDF, MiniLM, skill-overlap, and hybrid scores. Show box plots of score by label (matplotlib, which also covers syllabus item 1). Report AUC (Good vs No) and macro-F1 of a tiny LogisticRegression over the 3 scores. Compare to the DistilBERT 0.43 macro-F1 reference.
  2. **Manual ranking set**: both students independently rate 50 pairs (5 JDs × 10 resumes from Kaggle data) on a 0–3 scale. Report Cohen's kappa between them, Spearman ρ between each method and the mean human rating, and nDCG@5 per JD.
  3. **Skill extraction**: hand-tag skills in 20 resumes and report precision/recall/F1 of the PhraseMatcher vs noun-chunk-only vs SkillNER.
  4. **Ablation**: hybrid weights (e.g., 0.4 semantic + 0.4 skills + 0.2 TF-IDF) vs each component alone.
- A "sanity" test also makes a good viva answer: score each resume against JDs from its own category vs other categories (using the Kaggle category labels). The within-category mean should be higher, which gives a weak-supervision check for free.

### Gaps
- No standard public Indian resume–JD benchmark was found.
- cnamuangtoun label provenance (human or synthetic/LLM-generated) is unknown. It should be stated as a limitation.

## Notable existing GitHub projects/tutorials and how to differentiate

### Takeaway
The space is crowded: Resume-Matcher (28.8k stars, LLM-oriented), many Streamlit "ATS score" clones on HF Spaces, and the Analytics Vidhya category-classifier tutorial. Students differentiate through rigorous evaluation, explainable section-wise scoring, a skill-gap view tied to a taxonomy, multi-resume ranking, and a bias/fairness probe. They should not try to compete on UI features.

### Cited Findings
- **srbhr/Resume-Matcher**: about 28.8k stars, Apache-2.0, v1.3. FastAPI + Next.js + LiteLLM. Features: resume tailoring (needs an LLM; Ollama is the local option), cover letters, and an ATS score /100 with keyword highlighting that makes **no LLM calls**. Too heavy to copy in 2 weeks. — [GitHub srbhr/Resume-Matcher](https://github.com/srbhr/Resume-Matcher)
- **HF Space "Resume_ATS_Score_Checker"** (Streamlit): embedding-based semantic match % plus spaCy POS-based keyword extraction (NOUN/PROPN/VERB) from the JD to list missing keywords, with bands >75 high and >50 medium. — [HF Space Bodyless](https://huggingface.co/spaces/Bodyless/Resume_ATS_Score_Checker/blob/main/src/streamlit_app.py)
- **HF Space "muddasser/resume"**: Streamlit + spaCy + scikit-learn. Parses PDF/DOCX, finds user-defined skills, estimates years of experience, and computes cosine similarity. — [HF Space muddasser](https://huggingface.co/spaces/muddasser/resume/blob/main/README.md)
- **ATSCrewAI**: Streamlit/spaCy/NLTK scoring plus LLM rewrite when the score is <85 (uses agents/LLMs). — [HF Space sunithalv/ATSCrewAI](https://huggingface.co/spaces/sunithalv/ATSCrewAI/blob/main/README.md)
- GitHub topic "resume-scoring" lists many similar recruiter tools. — [GitHub topics/resume-scoring](https://github.com/topics/resume-scoring)
- Analytics Vidhya "Resume Screening with NLP in Python": category classification on 962 resumes, word cloud, and frequency analysis. Closest to a lab-style baseline. — [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/06/resume-screening-with-natural-language-processing-in-python/)
- **Bias**: UW study (Wilson & Caliskan, 2024) tested Massive Text Embedding (MTE) retrieval models on resume screening across 9 occupations and about 3M comparisons. White-associated names were favoured 85% of the time and female-associated names only 11%. Black men were the most disadvantaged. The authors note that removing names doesn't fully fix it, because identity leaks via education, location, and word choice. — [arXiv 2407.20371](https://www.arxiv.org/pdf/2407.20371); [GeekWire coverage](https://www.geekwire.com/2024/ai-overwhelmingly-prefers-white-and-male-job-candidates-in-new-test-of-resume-screening-bias/)

### Inferences
- Differentiators that suit a 2-week scope:
  1. **Section-wise scoring**: separate Skills / Experience / Education / Projects scores plus a weighted total, with a radar or bar chart. This is explainable, unlike a single ATS number.
  2. **Skill-gap report**: missing skills grouped by ESCO/O*NET category, marked "hot technology" via the O*NET flag, and linked to free learning resources (a static mapping, e.g., NPTEL/SWAYAM course names).
  3. **Multi-resume ranking**: upload N resumes for one JD and get a ranked table with CSV export. This is the recruiter view.
  4. **Bias probe (mini-replication)**: swap Indian names signalling gender/region/caste/religion (e.g., common Hindu/Muslim/North/South names) on otherwise identical resumes and measure the score change for TF-IDF vs MiniLM. TF-IDF should be name-insensitive once names are removed in preprocessing. This is a strong novel angle for India.
  5. **Lab-syllabus "analysis tab"**: word cloud, top-10 words, POS distribution, n-gram keyword phrases, JD subjectivity, and a JD summary (see syllabus mapping).
  6. **Honest dataset finding**: the dedup effect on the 99% classifier accuracy.

### Gaps
- Star counts for the smaller student repos were not verified.
- No Indian-context replication of the name-bias study was found. That would be novel but is not established.

## Realistic difficulty, time estimate, risks, and 2-person work split

### Takeaway
Feasible in about 2 weeks at about 3–4 h/day each if the scope is fixed early: TF-IDF + MiniLM + dictionary skills + classifier + Streamlit, with LLMs and fine-tuning ruled out. Main risks: messy PDF text (multi-column/tables), skill dictionary coverage/synonyms, MiniLM truncation, the inflated classifier accuracy, and privacy if real classmates' resumes are used.

### Cited Findings
- Two-column PDFs can break default extraction (pdfplumber), and all tested libraries scrambled at least one page. — [Python in Plain English](https://python.plainenglish.io/pymupdf-vs-pdfplumber-vs-pypdf-vs-pypdfium2-all-4-got-every-word-all-4-scrambled-one-page-d12054be37bd)
- MiniLM truncates beyond 256 word pieces. — [HF model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- SkillNER requires `en_core_web_lg` (a large download). — [CodeCut tutorial](https://codecut.ai/skillner-automating-skill-extraction-in-python/)
- pyresparser targets Python ≤3.7 and is unmaintained since 2019, so it is not usable on a modern Python setup. — [PyPI](https://pypi.org/project/pyresparser/)
- Resume-matching models embed demographic bias. — [arXiv 2407.20371](https://www.arxiv.org/pdf/2407.20371)

### Inferences
- **Difficulty: moderate** (about 6.5/10 for undergrads). The NLP pieces are standard. Most of the effort goes into integration and evaluation.
- **Timeline (20 Oct → 5 Nov)**:
  - Days 1–2: pick datasets, download, EDA (CSV + matplotlib), freeze scope, set up the repo/venv (`spacy`, `en_core_web_sm`, `nltk`, `scikit-learn`, `sentence-transformers`, `pdfplumber`, `python-docx`, `streamlit`, `wordcloud`).
  - Days 3–5: text extraction + preprocessing + section splitting (A). Skill dictionary build + PhraseMatcher/EntityRuler + NER (B).
  - Days 6–8: scoring module (TF-IDF, MiniLM, skill overlap, hybrid) (A). Category classifier with dedup experiment, plus confusion matrix (B).
  - Days 9–11: Streamlit UI (single + multi-resume), analysis tab (A). Evaluation: cnamuangtoun benchmark, manual annotation of 50 pairs, skill P/R, bias probe (B; both annotate).
  - Days 12–14: report, slides, README, demo video, buffer. Leave 2 days of buffer before 5 Nov.
- **Split**: Person A = "pipeline + app" (extraction, preprocessing, scoring, Streamlit). Person B = "knowledge + ML + evaluation" (skill dictionary, NER/chunking, classifier, all metrics, bias probe). Both annotate the manual set, which gives the inter-annotator kappa, and both write the report sections they own.
- **Risk mitigations**: fall back to PyMuPDF when pdfplumber output looks garbled; test on the Snehaan PDFs plus 5 self-made resumes (including a two-column template); handle scanned PDFs (no text layer) by saying "OCR not supported" rather than adding Tesseract. Precompute JD/resume embeddings and cache the model with `st.cache_resource`. MiniLM's CPU footprint (22.7M params) suits ordinary laptops.
- **Privacy**: use only public datasets and your own/synthetic resumes. If classmates' resumes are used, get consent and anonymise (strip names/phones/emails with regex + spaCy PERSON). Process in memory without storing uploads. India's DPDP Act 2023 is a reasonable thing to mention (not researched here).

### Gaps
- No empirical timing data for student builds was found. The estimates above are judgement.
- DPDP Act applicability to student projects was not researched.

## How well does it map to the syllabus experiments?

### Takeaway
It maps well. Items 1–5 and 8 are naturally covered. Item 6 (n-gram LM + smoothing) and item 7 (sentiment/subjectivity) are weakly relevant and need deliberate "bolt-on" uses so the examiner sees every experiment.

### Cited Findings
- spaCy PhraseMatcher/EntityRuler (rule-based NER) and the statistical NER pipeline interplay are documented. — [spaCy docs](https://spacy.io/usage/rule-based-matching)
- The Analytics Vidhya screening tutorial already combines CSV loading, cleaning, word frequency/word cloud, and classification on resume data. — [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/06/resume-screening-with-natural-language-processing-in-python/)
- A POS-based keyword extraction (NOUN/PROPN/VERB) for JD keywords is used in an existing Streamlit matcher. — [HF Space Bodyless](https://huggingface.co/spaces/Bodyless/Resume_ATS_Score_Checker/blob/main/src/streamlit_app.py)

### Inferences
| Syllabus item | Coverage | How |
|---|---|---|
| 1. CSV + matplotlib | Strong | Load Kaggle CSVs; category distribution bar chart, score box plots by fit label, confusion matrix |
| 2. Tokenization | Strong | Sentence tokenize resume/JD (for sentence embeddings and section detection), word tokenize for TF-IDF |
| 3. Stopwords, lemmatization, morphology | Strong | Normalise "managed/managing → manage", "databases → database" before skill matching. Morphological analysis via spaCy `token.morph` (tense/number) can show why lemma matching helps |
| 4. Word count, top-10, word cloud, NER | Strong | Analysis tab: resume vs JD word clouds, top-10 words, NER (ORG = companies/colleges, DATE = experience, GPE = location) + custom SKILL entity |
| 5. POS tagging + chunking | Strong | Noun-phrase chunking (spaCy `noun_chunks` or NLTK RegexpParser `NP: {<JJ>*<NN.*>+}`) to propose candidate skills/keywords missing from the dictionary |
| 6. N-gram LM + smoothing | Weak, needs a bolt-on | (a) Use bigram/trigram TF-IDF (`ngram_range=(1,2)`) so "machine learning" is one feature. (b) Train a bigram LM with Laplace/Kneser-Ney (NLTK `nltk.lm`) on JD text per category and score a resume by perplexity under each category's LM, as an alternative "which domain is this resume" classifier. (c) Collocation/bigram detection to discover multi-word skills |
| 7. Sentiment + subjectivity | Weak, needs a bolt-on | TextBlob subjectivity/polarity on resume summary/achievement lines (flag vague "passionate, hardworking" fluff vs objective, quantified statements). Or JD tone analysis. Present it as a "resume writing feedback" feature |
| 8. Summarization | Moderate | Extractive summary (TF-IDF/frequency sentence scoring or TextRank) of a long JD into its key requirements, or a 3-line candidate summary for the recruiter ranking view |

- The perplexity-based category classifier (item 6) compared against TF-IDF + SVM is a neat, syllabus-faithful experiment that most online clones lack.

### Gaps
- No source evaluating n-gram-LM perplexity for resume categorisation was found. Its accuracy is unknown and it may perform worse than SVM, which is still a valid finding.

## Likely viva/examiner questions

### Takeaway
Expect questions on why TF-IDF vs embeddings, how the score is validated, how skills are extracted and normalised, dataset issues (duplicates, leakage), PDF parsing limits, bias/ethics, and how each syllabus experiment appears in the project.

### Cited Findings
- TF-IDF can match or beat fine-tuned transformers on noisy fit labels. — [domynom/resume-job-distilbert](https://huggingface.co/domynom/resume-job-distilbert)
- SBERT outperforms TF-IDF on human-judged ranking metrics, while TF-IDF is faster. — [Power 2025](https://norma.ncirl.ie/9765/1/alexpower.pdf)
- The embedding model truncates at 256 word pieces. — [HF model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- EntityRuler placement before `ner` and PhraseMatcher `attr="LOWER"`. — [spaCy docs](https://spacy.io/usage/rule-based-matching)
- Name-based bias in embedding-based resume retrieval. — [arXiv 2407.20371](https://www.arxiv.org/pdf/2407.20371)

### Inferences
Likely questions (with answer pointers):
1. What is TF-IDF and why cosine similarity rather than Euclidean distance? (Length normalisation; resumes and JDs differ in length.)
2. Why does TF-IDF fail on synonyms ("ML" vs "machine learning")? How do embeddings fix this? What does all-MiniLM-L6-v2 output (384-d), and what happens to text over 256 tokens?
3. How did you build the skill dictionary? Coverage? How do you handle synonyms/abbreviations and case (`attr="LOWER"`, pattern IDs)? What are the precision/recall of skill extraction?
4. Difference between the rule-based EntityRuler/PhraseMatcher and the statistical NER? Why put the ruler before `ner`?
5. How do you know your match score is "correct"? (Benchmark labels, human annotation, Spearman/nDCG, kappa.)
6. Why is classification accuracy ~99%? Did you check for duplicate rows/leakage? What happens after deduplication? Which classes get confused (confusion matrix)?
7. Lemmatization vs stemming: why lemmatize for skill matching? Show a morphological analysis example.
8. What is chunking? Show your regex grammar. How does it find skills not in the dictionary?
9. How is the n-gram LM used? Why smoothing? Laplace vs Kneser-Ney? What is perplexity?
10. What does subjectivity mean in TextBlob, and why is it useful on a resume?
11. How do you handle two-column/scanned PDFs? Which library and why (licence: AGPL vs MIT)?
12. Bias: could your system discriminate by name/gender/college? How did you test it? Is name removal sufficient?
13. Privacy: where are uploaded resumes stored? Consent for real resumes?
14. How do you set the weights of the hybrid score? Did you tune them on the test data? (Avoid that.)
15. How would you scale to 10,000 resumes? (Precompute embeddings, vector index such as FAISS.)
16. Limitations and future work (fine-tuning SBERT on pairs like conSultantBERT, OCR, multilingual/Hindi resumes, LLM-free explainability).

### Gaps
- No published list of actual viva questions from Indian university NLP labs was found. The list above is inferred from the project's technical content.
