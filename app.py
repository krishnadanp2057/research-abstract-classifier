"""
Research Abstract Classifier - Streamlit demo for Project-I.

Run:  streamlit run app.py
Needs model.joblib (created by the notebook) in the same folder.
metrics.json (also created by the notebook) is optional; it powers the "Model insights" tab.
"""
import html
import json
import os
import re

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Abstract Classifier", page_icon="📚", layout="wide")

# ----------------------------------------------------------------------------
# Look and feel (CSS + animations)
# ----------------------------------------------------------------------------
COLORS = {"Science": "#2E86DE", "Business": "#F39C12", "Social Science": "#16A085"}
ICONS = {"Science": "🔬", "Business": "💼", "Social Science": "🌍"}

CSS = """
<style>
@keyframes fadeUp   {from {opacity:0; transform:translateY(18px);} to {opacity:1; transform:none;}}
@keyframes popIn    {0% {opacity:0; transform:scale(.85);} 70% {transform:scale(1.04);} 100% {opacity:1; transform:scale(1);}}
@keyframes growBar  {from {width:0;} to {width:var(--w);}}
@keyframes gradMove {0% {background-position:0% 50%;} 50% {background-position:100% 50%;} 100% {background-position:0% 50%;}}
@keyframes floaty   {0%,100% {transform:translateY(0);} 50% {transform:translateY(-6px);}}
@keyframes pulse    {0% {box-shadow:0 0 0 0 var(--glow);} 100% {box-shadow:0 0 0 22px rgba(0,0,0,0);}}

.hero {
  background: linear-gradient(120deg,#1e3c72,#2a5298,#6a3093,#2a5298);
  background-size: 300% 300%; animation: gradMove 12s ease infinite;
  border-radius: 18px; padding: 28px 32px; color: #fff; margin-bottom: 18px;
}
.hero h1 {margin:0; font-size: 2.1rem; color:#fff;}
.hero p  {margin:6px 0 0 0; opacity:.92; font-size:1.02rem;}
.hero .icon {display:inline-block; animation: floaty 3s ease-in-out infinite; margin-right:8px;}

.result-card {
  animation: popIn .6s ease both, pulse 1.4s ease-out 1;
  border-radius: 18px; padding: 22px 26px; color:#fff; margin: 6px 0 14px 0;
}
.result-card .lbl  {font-size: 1.9rem; font-weight: 700;}
.result-card .sub  {opacity:.92; font-size: .98rem;}

.bar-wrap {margin: 10px 0; animation: fadeUp .6s ease both;}
.bar-head {display:flex; justify-content:space-between; font-size:.92rem; margin-bottom:4px;}
.bar-bg   {background: rgba(128,128,128,.22); border-radius: 10px; height: 16px; overflow:hidden;}
.bar-fill {height:100%; border-radius:10px; width: var(--w); animation: growBar 1.1s cubic-bezier(.2,.8,.2,1) both;}

.abstract-box {
  animation: fadeUp .7s ease both; line-height: 1.85; padding: 16px 18px; border-radius: 12px;
  border: 1px solid rgba(128,128,128,.3); background: rgba(128,128,128,.06);
}
.abstract-box mark {padding: 1px 4px; border-radius: 5px; color: inherit; transition: transform .2s;}
.abstract-box mark:hover {transform: scale(1.12);}

.metric-tile {
  animation: fadeUp .6s ease both; border-radius: 14px; padding: 14px 16px;
  border: 1px solid rgba(128,128,128,.28); background: rgba(128,128,128,.07); text-align:center;
}
.metric-tile .v {font-size: 1.7rem; font-weight: 700;}
.metric-tile .k {font-size: .82rem; opacity: .75; text-transform: uppercase; letter-spacing: .05em;}

.chip {display:inline-block; padding: 3px 12px; border-radius: 999px; font-size:.82rem; font-weight:600;
       color:#fff; margin: 2px 4px 2px 0; animation: popIn .5s ease both;}
.stTabs [data-baseweb="tab"] {font-size: 1.02rem; font-weight: 600;}
.stButton > button {transition: transform .15s ease, box-shadow .15s ease; border-radius: 10px;}
.stButton > button:hover {transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,.18);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Model helpers
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))


def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"(©|copyright|\(c\)).{0,80}", " ", t)
    t = re.sub(r"^\s*abstract[:.\s]*", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@st.cache_resource(show_spinner=False)
def load_model():
    path = os.path.join(HERE, "model.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_metrics():
    path = os.path.join(HERE, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def predict_proba(model, texts):
    """Return an (n, classes) probability matrix for any of the six model types."""
    cleaned = [clean_text(t) for t in texts]
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        p = model.predict_proba(cleaned)
    else:  # LinearSVC: turn decision scores into probabilities with a softmax
        d = model.decision_function(cleaned)
        e = np.exp(d - d.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
    return p


def word_weights(model, class_index):
    """Per-word influence towards one class (linear / naive Bayes models only)."""
    clf = model.named_steps["clf"]
    vocab = model.named_steps["tfidf"].get_feature_names_out()
    if hasattr(clf, "coef_"):
        W = clf.coef_
    elif hasattr(clf, "feature_log_prob_"):
        W = clf.feature_log_prob_
    else:
        return None
    others = np.delete(W, class_index, axis=0).mean(axis=0)
    return dict(zip(vocab, W[class_index] - others))


def highlight_html(text, weights, scale):
    """Colour every word by how strongly it pushes towards the predicted class."""
    out = []
    for tok in re.split(r"(\s+)", text):
        key = re.sub(r"[^a-z]", "", tok.lower())
        w = weights.get(key) if key else None
        safe = html.escape(tok)
        if w is None or abs(w) < 0.15 * scale:
            out.append(safe)
        else:
            a = min(abs(w) / scale, 1.0) * 0.55 + 0.12
            colour = f"rgba(46,204,113,{a:.2f})" if w > 0 else f"rgba(231,76,60,{a:.2f})"
            out.append(f'<mark style="background:{colour}" title="{w:+.2f}">{safe}</mark>')
    return "".join(out)


def bars_html(labels, probs):
    order = np.argsort(probs)[::-1]
    rows = []
    for rank, i in enumerate(order):
        c = COLORS.get(labels[i], "#7f8c8d")
        pct = probs[i] * 100
        rows.append(
            f'<div class="bar-wrap" style="animation-delay:{rank * 0.12:.2f}s">'
            f'<div class="bar-head"><span>{ICONS.get(labels[i], "")} <b>{labels[i]}</b></span>'
            f"<span>{pct:.1f}%</span></div>"
            f'<div class="bar-bg"><div class="bar-fill" style="--w:{pct:.1f}%;background:{c};'
            f'animation-delay:{rank * 0.12:.2f}s"></div></div></div>'
        )
    return "".join(rows)


def confidence_word(p):
    if p >= 0.85:
        return "high confidence"
    if p >= 0.65:
        return "moderate confidence"
    return "low confidence - this paper may sit between fields"


# ----------------------------------------------------------------------------
# Example abstracts (written for this demo, not taken from the training data)
# ----------------------------------------------------------------------------
EXAMPLES = {
    "🔬 Science": (
        "We report a nickel-based molecular catalyst for the selective hydrogenation of unsaturated "
        "organic compounds under mild conditions. Kinetic experiments and density functional theory "
        "calculations suggest a mechanism involving heterolytic cleavage of hydrogen at the metal "
        "centre. The catalyst tolerates a wide range of functional groups and can be recycled several "
        "times without loss of activity. Spectroscopic characterisation of reaction intermediates "
        "supports the proposed catalytic cycle."
    ),
    "💼 Business": (
        "This study investigates how supply chain resilience influences financial performance in "
        "manufacturing firms. Using survey and archival data from several hundred companies, we show "
        "that supplier diversification and inventory flexibility raise profitability during periods of "
        "disruption. Management commitment and information sharing with customers strengthen the "
        "effect, suggesting that managers should invest in visibility across the supply network."
    ),
    "🌍 Social Science": (
        "This article examines how social media use relates to political participation among young "
        "adults. Drawing on a national survey, we find that exposure to political content online "
        "increases voting intention and protest activity, but only when users also discuss politics "
        "with peers offline. Differences by education and income suggest that digital platforms may "
        "deepen existing inequalities in civic engagement."
    ),
    "🤔 Borderline": (
        "Large language models are increasingly used in computer science education, helping students "
        "write, debug and understand code. We present a systematic literature review of their impact "
        "on the learning experience, personalised education and curriculum development, and discuss "
        "challenges and directions for future research."
    ),
}


def set_example(name):
    st.session_state["abstract_text"] = EXAMPLES[name]


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
bundle = load_model()
metrics = load_metrics()

st.markdown(
    '<div class="hero"><h1><span class="icon">📚</span>Research Abstract Classifier</h1>'
    "<p>Paste an abstract or upload a file of abstracts and the model predicts whether the paper "
    "belongs to <b>Science</b>, <b>Business</b> or <b>Social Science</b>.</p></div>",
    unsafe_allow_html=True,
)

if bundle is None:
    st.error("`model.joblib` was not found next to `app.py`. Run every cell of the notebook first "
             "(the last cells save the model), then restart this app.")
    st.stop()

model, labels = bundle["model"], list(bundle["labels"])
clf_classes = list(model.named_steps["clf"].classes_)

with st.sidebar:
    st.header("⚙️ About this model")
    st.write(f"**Model:** {bundle['name']}")
    st.write("**Features:** TF-IDF, word 1-2-grams")
    st.markdown("".join(f'<span class="chip" style="background:{COLORS.get(l, "#7f8c8d")}">'
                        f"{ICONS.get(l, '')} {l}</span>" for l in labels), unsafe_allow_html=True)
    if metrics:
        best_row = next((r for r in metrics["results"] if r["Model"] == metrics["best"]), None)
        if best_row:
            st.metric("Test accuracy", f"{best_row['Test accuracy']:.1%}")
            st.metric("Macro F1", f"{best_row['F1 (macro)']:.3f}")
        st.caption(f"Trained on {metrics['n_train']:,} abstracts, tested on {metrics['n_test']:,}.")
    st.divider()
    low_conf = st.slider("Low-confidence warning below", 0.40, 0.95, 0.65, 0.05,
                         help="Predictions under this probability are flagged for manual review.")
    st.caption("Tip: green words push towards the predicted class, red words push against it.")

tab_single, tab_batch, tab_insights, tab_help = st.tabs(
    ["✍️ Single abstract", "📂 Upload a file", "📊 Model insights", "ℹ️ How it works"])

# ---------------------------- Single abstract --------------------------------
with tab_single:
    st.write("**Try an example**")
    cols = st.columns(len(EXAMPLES))
    for col, name in zip(cols, EXAMPLES):
        col.button(name, key=f"ex_{name}", on_click=set_example, args=(name,))

    text = st.text_area("Abstract text", key="abstract_text", height=200,
                        placeholder="Paste a paper abstract here (at least a few sentences)...")
    n_words = len(text.split())
    st.caption(f"{n_words} words" + ("  ·  very short, the prediction may be unreliable" if 0 < n_words < 40 else ""))

    if st.button("🔍 Classify", type="primary", disabled=n_words == 0):
        with st.spinner("Reading the abstract..."):
            probs = predict_proba(model, [text])[0]
        top = int(np.argmax(probs))
        label = clf_classes[top]
        colour = COLORS.get(label, "#7f8c8d")
        st.markdown(
            f'<div class="result-card" style="background:linear-gradient(135deg,{colour},#2c3e50);--glow:{colour}66">'
            f'<div class="sub">Predicted field</div>'
            f'<div class="lbl">{ICONS.get(label, "")} {label}</div>'
            f'<div class="sub">{probs[top]:.1%} - {confidence_word(probs[top])}</div></div>',
            unsafe_allow_html=True,
        )
        if probs[top] < low_conf:
            st.warning("The model is not very sure about this one. Papers on technology in education, "
                       "or business and the environment, often sit between two fields.")
        left, right = st.columns([1, 1.25])
        with left:
            st.write("**Probability for each class**")
            st.markdown(bars_html(clf_classes, probs), unsafe_allow_html=True)
        with right:
            st.write("**Which words drove the decision?**")
            weights = word_weights(model, top)
            if weights is None:
                st.info("Word highlighting is available for linear and Naive Bayes models only.")
            else:
                scale = max(np.percentile(np.abs(list(weights.values())), 99), 1e-6)
                st.markdown(f'<div class="abstract-box">{highlight_html(text, weights, scale)}</div>',
                            unsafe_allow_html=True)
                clean_words = set(clean_text(text).split())
                contrib = sorted(((w, v) for w, v in weights.items() if w in clean_words),
                                 key=lambda x: -x[1])
                pos = [w for w, v in contrib[:6] if v > 0]
                if pos:
                    st.markdown("**Strongest evidence:** " + "".join(
                        f'<span class="chip" style="background:{colour}">{html.escape(w)}</span>' for w in pos),
                        unsafe_allow_html=True)

# ---------------------------- Batch upload -----------------------------------
with tab_batch:
    st.write("Upload a CSV or Excel file that has a column of abstracts. If it also has a `label` column "
             "(Science / Business / Social Science) the app will measure accuracy for you.")
    c1, c2 = st.columns([2, 1])
    up = c1.file_uploader("CSV or Excel file", type=["csv", "xlsx"], label_visibility="collapsed")
    c2.download_button("⬇️ Download a sample file",
                       open(os.path.join(HERE, "new_test_abstracts.csv"), "rb").read()
                       if os.path.exists(os.path.join(HERE, "new_test_abstracts.csv")) else b"title,abstract,label\n",
                       "sample_abstracts.csv", "text/csv")

    if up is not None:
        try:
            df = pd.read_csv(up, encoding="utf-8-sig") if up.name.lower().endswith(".csv") else pd.read_excel(up)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the file: {e}")
            st.stop()
        if df.empty:
            st.error("The file has no rows.")
            st.stop()
        guess = next((c for c in df.columns if "abstract" in str(c).lower()), df.columns[0])
        col = st.selectbox("Column that holds the abstracts", list(df.columns), index=list(df.columns).index(guess))

        if st.button("🚀 Predict all", type="primary"):
            texts = df[col].fillna("").astype(str).tolist()
            bar = st.progress(0.0, text="Classifying...")
            chunks, step = [], max(1, len(texts) // 20)
            for i in range(0, len(texts), step):
                chunks.append(predict_proba(model, texts[i:i + step]))
                bar.progress(min((i + step) / len(texts), 1.0), text=f"Classifying... {min(i + step, len(texts))}/{len(texts)}")
            bar.empty()
            probs = np.vstack(chunks)
            res = df.copy()
            res["predicted_label"] = [clf_classes[i] for i in probs.argmax(axis=1)]
            res["confidence"] = probs.max(axis=1).round(3)
            res["needs_review"] = res["confidence"] < low_conf
            for j, c in enumerate(clf_classes):
                res[f"prob_{c}"] = probs[:, j].round(3)
            st.session_state["batch_result"] = (res, col)
            st.toast(f"Classified {len(res)} abstracts", icon="✅")
            if len(res) >= 50:
                st.balloons()

    if "batch_result" in st.session_state:
        res, col = st.session_state["batch_result"]
        res = res.copy()
        res["needs_review"] = res["confidence"] < low_conf

        tiles = st.columns(4)
        tile_data = [("Abstracts", f"{len(res):,}"),
                     ("Average confidence", f"{res['confidence'].mean():.1%}"),
                     ("Need review", f"{int(res['needs_review'].sum()):,}"),
                     ("Top class", res["predicted_label"].mode().iat[0])]
        has_truth = "label" in res.columns
        if has_truth:
            tile_data[3] = ("Accuracy vs your labels", f"{(res['label'] == res['predicted_label']).mean():.1%}")
        for t, (k, v) in zip(tiles, tile_data):
            t.markdown(f'<div class="metric-tile"><div class="v">{v}</div><div class="k">{k}</div></div>',
                       unsafe_allow_html=True)
        st.write("")

        domain = list(COLORS)
        rng = [COLORS[d] for d in domain]
        g1, g2 = st.columns(2)
        counts = res["predicted_label"].value_counts().rename_axis("Class").reset_index(name="Abstracts")
        g1.altair_chart(
            alt.Chart(counts).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
                x=alt.X("Class:N", sort="-y", axis=alt.Axis(labelAngle=0)), y="Abstracts:Q",
                color=alt.Color("Class:N", scale=alt.Scale(domain=domain, range=rng), legend=None),
                tooltip=["Class", "Abstracts"]).properties(title="Predicted classes", height=280),
        )
        g2.altair_chart(
            alt.Chart(res).mark_bar(opacity=.85).encode(
                x=alt.X("confidence:Q", bin=alt.Bin(maxbins=20), title="Confidence"),
                y=alt.Y("count()", title="Abstracts"),
                color=alt.Color("predicted_label:N", scale=alt.Scale(domain=domain, range=rng), title="Class"),
                tooltip=["count()"]).properties(title="How sure is the model?", height=280),
        )

        if has_truth:
            cm = pd.crosstab(res["label"], res["predicted_label"]).reset_index().melt("label")
            cm.columns = ["True label", "Predicted", "Count"]
            heat = alt.Chart(cm).mark_rect(cornerRadius=6).encode(
                x=alt.X("Predicted:N", axis=alt.Axis(labelAngle=0)), y="True label:N",
                color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")), tooltip=["True label", "Predicted", "Count"])
            txt = heat.mark_text(fontSize=16).encode(text="Count:Q", color=alt.value("black"))
            st.altair_chart((heat + txt).properties(title="Confusion matrix on your file", height=260))

        st.subheader("Results")
        f1, f2, f3 = st.columns([2, 1.3, 1.3])
        pick = f1.multiselect("Show classes", clf_classes, default=clf_classes)
        only_review = f2.toggle("Only low confidence", value=False)
        search = f3.text_input("Search text", placeholder="word in abstract...")
        view = res[res["predicted_label"].isin(pick)]
        if only_review:
            view = view[view["needs_review"]]
        if search:
            view = view[view[col].astype(str).str.contains(search, case=False, na=False)]
        st.caption(f"Showing {len(view):,} of {len(res):,} rows")
        st.dataframe(view, column_config={
            "confidence": st.column_config.ProgressColumn("confidence", min_value=0.0, max_value=1.0, format="%.2f"),
            col: st.column_config.TextColumn(col, width="large"),
        }, hide_index=True)
        st.download_button("⬇️ Download predictions (CSV)", res.to_csv(index=False).encode("utf-8"),
                           "predictions.csv", "text/csv")

# ---------------------------- Model insights ---------------------------------
with tab_insights:
    if not metrics:
        st.info("Run the notebook to create `metrics.json`; it adds the model comparison and confusion "
                "matrix to this tab.")
    else:
        mt = st.columns(4)
        info = [("Abstracts", f"{metrics['n_total']:,}"), ("Training set", f"{metrics['n_train']:,}"),
                ("Test set", f"{metrics['n_test']:,}"), ("Best model", metrics["best"])]
        for t, (k, v) in zip(mt, info):
            t.markdown(f'<div class="metric-tile"><div class="v" style="font-size:1.25rem">{v}</div>'
                       f'<div class="k">{k}</div></div>', unsafe_allow_html=True)
        st.write("")
        res_df = pd.DataFrame(metrics["results"])
        metric_choice = st.radio("Metric", ["Test accuracy", "F1 (macro)", "Precision (macro)", "Recall (macro)"],
                                 horizontal=True)
        bar_df = res_df.sort_values(metric_choice, ascending=False)
        best_name = metrics["best"]
        chart = alt.Chart(bar_df).mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8).encode(
            y=alt.Y("Model:N", sort="-x", title=None),
            x=alt.X(f"{metric_choice}:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.condition(alt.datum.Model == best_name, alt.value("#2E86DE"), alt.value("#95A5A6")),
            tooltip=["Model", alt.Tooltip(f"{metric_choice}:Q", format=".3f")])
        labels_txt = chart.mark_text(align="left", dx=4).encode(
            text=alt.Text(f"{metric_choice}:Q", format=".3f"), color=alt.value("gray"))
        st.altair_chart((chart + labels_txt).properties(height=300, title=f"Model comparison - {metric_choice}"))

        cm = np.array(metrics["confusion_matrix"])
        cm_df = pd.DataFrame(cm, index=metrics["labels"], columns=metrics["labels"]).reset_index().melt("index")
        cm_df.columns = ["True label", "Predicted", "Count"]
        heat = alt.Chart(cm_df).mark_rect(cornerRadius=6).encode(
            x=alt.X("Predicted:N", axis=alt.Axis(labelAngle=0)), y="True label:N",
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["True label", "Predicted", "Count"])
        txt = heat.mark_text(fontSize=18).encode(text="Count:Q", color=alt.value("black"))
        st.altair_chart((heat + txt).properties(height=300, title=f"Confusion matrix ({best_name}, held-out test set)"))
        with st.expander("Full results table"):
            st.dataframe(res_df, hide_index=True)

    st.subheader("Top words per class")
    clf = model.named_steps["clf"]
    vocab = model.named_steps["tfidf"].get_feature_names_out()
    if hasattr(clf, "coef_") or hasattr(clf, "feature_log_prob_"):
        cols = st.columns(len(clf_classes))
        for j, (c, name) in enumerate(zip(cols, clf_classes)):
            w = word_weights(model, j)
            top = sorted(w.items(), key=lambda x: -x[1])[:12]
            d = pd.DataFrame(top, columns=["Word", "Weight"])
            c.altair_chart(
                alt.Chart(d).mark_bar(color=COLORS.get(name, "#7f8c8d"), cornerRadiusTopRight=6,
                                      cornerRadiusBottomRight=6).encode(
                    y=alt.Y("Word:N", sort="-x", title=None), x=alt.X("Weight:Q", title=None),
                    tooltip=["Word", alt.Tooltip("Weight:Q", format=".2f")]
                ).properties(title=f"{ICONS.get(name, '')} {name}", height=330))
    else:
        st.info("Top words are shown for linear and Naive Bayes models.")

# ---------------------------- Help ------------------------------------------
with tab_help:
    st.markdown(
        """
**Pipeline**

1. The abstract is lower-cased and cleaned (HTML, copyright lines, numbers and punctuation removed).
2. TF-IDF turns it into numbers using single words and word pairs.
3. The trained classifier gives a probability for each of the three classes.

**Reading the result**

- *Confidence* is the probability of the winning class. Low confidence usually means the paper sits
  between fields, for example technology in education, or business and the environment.
- *Green / red words* show which words pushed the prediction towards or away from the chosen class.
- Training labels come from Scopus subject areas, which tag journals and not individual papers, so a
  few labels are noisy. The model can sometimes be right when a label looks wrong.

**Limits**

The model only knows three broad fields, sees only the abstract, and cannot say that a paper belongs to none of them.
        """
    )
