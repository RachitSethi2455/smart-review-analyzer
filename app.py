"""Smart Review Analyzer - Streamlit app.

Run:  streamlit run app.py
"""
import html

import pandas as pd
import streamlit as st

from src import aspects, linguistics, sentiment, summarize, text_stats
from src.config import SAMPLE_CSV
from src.data import load_reviews_csv
from src.ngram_lm import NGramLM
from src.preprocessing import preprocess, preprocessing_steps

MAX_CSV_ROWS = 2000
DEMO_TEXT = ("The battery is great and lasts two days. The camera is not good in low light though. "
             "Bought it from Flipkart and delivery to Delhi was fast, but the charger is very slow.")

st.set_page_config(page_title="Smart Review Analyzer", page_icon="💬", layout="wide")


# ---------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading spaCy model...")
def nlp():
    return linguistics.get_nlp()


@st.cache_resource
def ml_model():
    return sentiment.load_model()


@st.cache_resource(show_spinner="Parsing reviews with spaCy...", max_entries=5)
def docs_for(texts):
    return list(nlp().pipe(texts, batch_size=64))


@st.cache_data(show_spinner="Scoring sentiment...")
def score_reviews(texts):
    model = ml_model()
    rows = []
    for t in texts:
        v, tb = sentiment.vader(t), sentiment.textblob(t)
        row = {"text": t, "vader": v["label"], "vader_score": round(v["score"], 3),
               "textblob": tb["label"], "polarity": round(tb["polarity"], 3),
               "subjectivity": round(tb["subjectivity"], 3)}
        if model is not None:
            row["tfidf_model"] = sentiment.ml_predict(t, model)["label"]
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner="Training n-gram language model...")
def ngram_model(texts, n):
    return NGramLM(n=n, k=0.1, min_count=1).fit([preprocess(t, remove_stopwords=False, use_lemmatize=False)
                                                  for t in texts])


# ---------------------------------------------------------------- helpers
def badge(label):
    color = text_stats.COLORS.get(label, "#555")
    return f"<span style='background:{color};color:white;padding:2px 10px;border-radius:10px'>{label}</span>"


def highlight(text, weights):
    """Colour words green/red by their contribution to the TF-IDF model's decision."""
    lookup = {w: s for w, s in weights if " " not in w}
    out = []
    for word in text.split():
        key = preprocess(word)
        s = lookup.get(key[0]) if key else None
        safe = html.escape(word)
        if s is None:
            out.append(safe)
        else:
            alpha = min(abs(s) * 3, 0.85) + 0.15
            rgb = "46,125,50" if s > 0 else "198,40,40"
            out.append(f"<span style='background:rgba({rgb},{alpha:.2f});padding:1px 3px;border-radius:3px'>{safe}</span>")
    return " ".join(out)


# ---------------------------------------------------------------- sidebar: input
st.sidebar.title("💬 Smart Review Analyzer")
st.sidebar.caption("NLP Lab Mini-Project")
mode = st.sidebar.radio("Input", ["Single review", "Upload CSV", "Sample reviews"])

texts, ratings = [], None
if mode == "Single review":
    texts = [st.sidebar.text_area("Paste a review", DEMO_TEXT, height=180)]
else:
    source = st.sidebar.file_uploader("Review CSV", type="csv") if mode == "Upload CSV" else SAMPLE_CSV
    if source is not None:
        raw = pd.read_csv(source, encoding_errors="replace", nrows=50_000)
        if hasattr(source, "seek"):
            source.seek(0)
        cols = list(raw.columns)
        text_col = st.sidebar.selectbox("Text column", cols, index=0)
        rating_col = st.sidebar.selectbox("Star rating column (optional)", ["(none)"] + cols,
                                          index=(cols.index("rating") + 1) if "rating" in cols else 0)
        df_in = load_reviews_csv(source, text_col, None if rating_col == "(none)" else rating_col,
                                 max_rows=MAX_CSV_ROWS)
        texts = df_in["text"].tolist()
        ratings = df_in.get("rating")
        st.sidebar.success(f"{len(texts):,} reviews loaded" + (f" (capped at {MAX_CSV_ROWS})" if len(raw) > MAX_CSV_ROWS else ""))

model = ml_model()
if model is not None:
    st.sidebar.info("TF-IDF model loaded ✔")
else:
    st.sidebar.warning("No trained model yet. Run `python scripts/experiments.py ml` to add the TF-IDF model.")

texts = [t for t in texts if t and t.strip()]
if not texts:
    st.info("Enter a review or upload a CSV to start.")
    st.stop()

single = len(texts) == 1
docs = docs_for(tuple(texts))
scores = score_reviews(tuple(texts))

tabs = st.tabs(["😊 Sentiment", "🔤 Words", "🏷️ Entities", "🧩 Aspects", "📝 Summary",
                "🧠 Linguistics", "🔮 N-gram LM", "⚙️ Preprocessing"])

# ---------------------------------------------------------------- 1. Sentiment (Exp 7)
with tabs[0]:
    if single:
        text = texts[0]
        v, tb = sentiment.vader(text), sentiment.textblob(text)
        cols = st.columns(4)
        cols[0].markdown(f"**VADER**<br>{badge(v['label'])}<br>compound = {v['score']:.3f}", unsafe_allow_html=True)
        cols[1].markdown(f"**TextBlob**<br>{badge(tb['label'])}<br>polarity = {tb['polarity']:.3f}", unsafe_allow_html=True)
        if model is not None:
            ml = sentiment.ml_predict(text, model)
            conf = f"<br>confidence = {ml['confidence']:.2f}" if "confidence" in ml else ""
            cols[2].markdown(f"**TF-IDF model**<br>{badge(ml['label'])}{conf}", unsafe_allow_html=True)
        if st.session_state.get("use_transformer"):
            tr = sentiment.transformer_predict(text)
            if tr:
                cols[3].markdown(f"**DistilBERT**<br>{badge(tr[0]['label'])}<br>confidence = {tr[0]['confidence']:.2f}",
                                 unsafe_allow_html=True)
        st.checkbox("Also run DistilBERT (needs requirements-optional.txt)", key="use_transformer")

        st.subheader("Subjectivity")
        st.progress(tb["subjectivity"], text=f"{tb['subjectivity']:.2f} → {sentiment.subjectivity_label(tb['subjectivity'])}")

        if model is not None:
            weights = sentiment.explain_words(text, model, top=25)
            if weights:
                st.subheader("Why did the TF-IDF model decide this?")
                st.caption("Green words pushed towards positive, red towards negative.")
                st.markdown(highlight(text, weights), unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(weights, columns=["feature", "contribution"]).round(3), hide_index=True)

        st.subheader("Sentence by sentence")
        st.dataframe(pd.DataFrame([{"sentence": s.text, **{k: sentiment.vader(s.text)[k] for k in ("label", "score")}}
                                   for s in docs[0].sents]), hide_index=True, use_container_width=True)
    else:
        label_col = "tfidf_model" if "tfidf_model" in scores else "vader"
        c1, c2 = st.columns(2)
        c1.pyplot(text_stats.label_distribution_figure(scores["vader"], "VADER sentiment"))
        c2.pyplot(text_stats.label_distribution_figure(scores["textblob"], "TextBlob sentiment"))
        st.metric("Average subjectivity", f"{scores['subjectivity'].mean():.2f}")

        if ratings is not None and ratings.notna().any():
            st.subheader("⭐ Rating vs text mismatch detector")
            st.caption("Reviews whose star rating disagrees with what the text says.")
            check = scores.assign(rating=ratings.values)
            check["rating_label"] = check["rating"].map(lambda r: None if pd.isna(r) else
                                                        ("negative" if r <= 2 else "neutral" if r < 4 else "positive"))
            mismatch = check[(check["rating_label"].isin(["positive", "negative"])) &
                             (check["vader"].isin(["positive", "negative"])) &
                             (check["rating_label"] != check["vader"])]
            st.write(f"{len(mismatch)} of {len(check)} reviews look mismatched.")
            st.dataframe(mismatch[["text", "rating", "vader", "vader_score"]], hide_index=True, use_container_width=True)

        st.subheader("All reviews")
        st.dataframe(scores, hide_index=True, use_container_width=True)
        st.download_button("Download results CSV", scores.to_csv(index=False), "review_sentiment.csv", "text/csv")

# ---------------------------------------------------------------- 2. Words (Exp 4.1)
with tabs[1]:
    counts = text_stats.basic_counts(texts)
    c = st.columns(3)
    c[0].metric("Reviews", counts["documents"])
    c[1].metric("Total words", f"{counts['total_words']:,}")
    c[2].metric("Avg words / review", counts["avg_words_per_doc"])
    freqs = text_stats.word_frequencies(texts)
    top10 = pd.DataFrame(freqs.most_common(10), columns=["word", "count"])
    left, right = st.columns([1, 1.4])
    left.pyplot(text_stats.bar_figure(top10, "word", "count", "Top 10 words (after preprocessing)"))
    right.pyplot(text_stats.wordcloud_figure(freqs, "Word cloud"))
    if not single:
        st.subheader("Positive vs negative vocabulary")
        lab = scores["tfidf_model"] if "tfidf_model" in scores else scores["vader"]
        p, n = st.columns(2)
        pos_texts = [t for t, l in zip(texts, lab) if l == "positive"]
        neg_texts = [t for t, l in zip(texts, lab) if l == "negative"]
        p.pyplot(text_stats.wordcloud_figure(text_stats.word_frequencies(pos_texts), "Positive reviews", "Greens"))
        n.pyplot(text_stats.wordcloud_figure(text_stats.word_frequencies(neg_texts), "Negative reviews", "Reds"))

# ---------------------------------------------------------------- 3. Entities (Exp 4.2)
with tabs[2]:
    ents = linguistics.entity_table(docs)
    if ents.empty:
        st.info("No named entities found.")
    else:
        if single:
            from spacy import displacy
            st.markdown(displacy.render(docs[0], style="ent"), unsafe_allow_html=True)
        left, right = st.columns(2)
        left.dataframe(ents, hide_index=True, use_container_width=True)
        by_label = ents.groupby("label")["count"].sum().sort_values(ascending=False).reset_index()
        right.pyplot(text_stats.bar_figure(by_label, "label", "count", "Entities by type", "#6a1b9a"))
        st.caption("ORG = companies/brands, PRODUCT = products, PERSON = people, GPE = cities/countries.")

# ---------------------------------------------------------------- 4. Aspects (Exp 5 + 7)
with tabs[3]:
    st.caption("Which features people like or dislike, found with dependency-parsing rules (amod / acomp + nsubj).")
    found = [a for d in docs for a in aspects.extract_aspects(d)]
    if not found:
        st.info("No aspect–opinion pairs found. Try a review like 'The battery is great but the camera is bad'.")
    else:
        summary = aspects.aspect_summary(docs)
        top = summary.head(15).sort_values("avg_score")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 0.35 * len(top) + 1.2))
        ax.barh(top["aspect"], top["avg_score"],
                color=["#2e7d32" if s > 0.05 else "#c62828" if s < -0.05 else "#9e9e9e" for s in top["avg_score"]])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("average sentiment (VADER)")
        ax.set_title("Aspect sentiment")
        fig.tight_layout()
        st.pyplot(fig)
        st.dataframe(summary.round(3), hide_index=True, use_container_width=True)
        with st.expander("Every aspect–opinion pair"):
            st.dataframe(pd.DataFrame(found).round(3), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- 5. Summary (Exp 8)
with tabs[4]:
    method = st.selectbox("Method", ["lexrank", "textrank", "lsa", "luhn", "frequency"],
                          help="'frequency' is our own implementation; the rest come from sumy.")
    n_sent = st.slider("Sentences", 1, 8, 3)
    if single:
        st.write(summarize.summarize(texts[0], n_sent, method))
    else:
        pc = summarize.pros_cons_summary(texts, n_sent, method)
        g, b = st.columns(2)
        g.subheader(f"👍 What people liked ({pc['n_positive_sentences']} sentences)")
        g.write(pc["liked"] or "—")
        b.subheader(f"👎 What people disliked ({pc['n_negative_sentences']} sentences)")
        b.write(pc["disliked"] or "—")

# ---------------------------------------------------------------- 6. Linguistics (Exp 3.2, 5)
with tabs[5]:
    idx = 0 if single else st.number_input("Review #", 0, len(texts) - 1, 0)
    doc = docs[idx]
    if not single:
        st.write(texts[idx])
    st.subheader("POS tags")
    st.dataframe(linguistics.pos_table(doc), hide_index=True, use_container_width=True)
    st.subheader("Noun-phrase chunks")
    st.write(", ".join(linguistics.noun_chunks(doc)) or "—")
    st.subheader("Morphological analysis")
    st.dataframe(linguistics.morphology_table(doc), hide_index=True, use_container_width=True)
    with st.expander("Dependency tree"):
        from spacy import displacy
        sent = next(doc.sents).as_doc() if not single else doc
        st.markdown(f"<div style='overflow-x:auto'>{displacy.render(sent, style='dep', options={'compact': True})}</div>",
                    unsafe_allow_html=True)
    if not single:
        left, right = st.columns(2)
        left.pyplot(text_stats.bar_figure(linguistics.pos_counts(docs), "pos", "count", "POS distribution"))
        right.pyplot(text_stats.bar_figure(linguistics.top_words_by_pos(docs, "ADJ", 10), "word", "count",
                                           "Most used adjectives", "#ef6c00"))

# ---------------------------------------------------------------- 7. N-gram LM (Exp 6)
with tabs[6]:
    st.caption("A bigram/trigram language model with add-k smoothing, trained on the loaded reviews "
               "(or the sample reviews if only one is given).")
    n = st.radio("n", [2, 3], horizontal=True)
    corpus = texts if len(texts) > 5 else pd.read_csv(SAMPLE_CSV)["review"].tolist() + texts
    lm = ngram_model(tuple(corpus), n)
    prompt = st.text_input("Start typing", "the battery is")
    toks = preprocess(prompt, remove_stopwords=False, use_lemmatize=False)
    sugg = lm.suggest(toks, top=5)
    st.write("**Next-word suggestions:** " + (", ".join(f"`{w}` ({p:.3f})" for w, p in sugg) or "none (unseen context)"))
    test_sentence = st.text_input("Sentence to score", "the camera is good")
    st.metric("Perplexity", f"{lm.perplexity(preprocess(test_sentence, remove_stopwords=False, use_lemmatize=False)):.1f}")
    st.caption(f"Vocabulary size V = {lm.V}. Lower perplexity = the model finds the sentence more likely.")

# ---------------------------------------------------------------- 8. Preprocessing (Exp 2, 3)
with tabs[7]:
    idx = 0 if single else st.number_input("Review # ", 0, len(texts) - 1, 0)
    steps = preprocessing_steps(texts[idx])
    for name, value in steps.items():
        st.markdown(f"**{name.replace('_', ' ').title()}**")
        st.write(value if isinstance(value, str) else (value if name == "sentences" else " | ".join(value)))
    st.caption("Negation words like 'not' and 'never' are deliberately kept: removing them flips sentiment.")
