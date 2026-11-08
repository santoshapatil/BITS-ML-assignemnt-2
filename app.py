"""Credit Card Default Risk - model comparison app (Streamlit).

Loads the five pre-trained classifiers from model/ and scores an uploaded test CSV.
No retraining happens here: the app only calls predict / predict_proba / decision_function.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from metrics_utils import MODEL_FILES, TARGET_COL, evaluate

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"

st.set_page_config(page_title="Credit Default Risk Classifier", page_icon="💳", layout="wide")


@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}


@st.cache_data
def load_feature_columns():
    return json.loads((MODEL_DIR / "feature_columns.json").read_text())


def validate(df, features):
    missing = [c for c in features + [TARGET_COL] if c not in df.columns]
    if missing:
        st.error(f"Uploaded CSV is missing required column(s): {', '.join(missing)}")
        st.stop()
    labels = set(df[TARGET_COL].dropna().unique())
    if not labels <= {0, 1}:
        st.error(f"Target column `{TARGET_COL}` must be 0/1; found {sorted(labels)}")
        st.stop()
    clean = df[features + [TARGET_COL]].dropna()
    if len(clean) < len(df):
        st.warning(f"Dropped {len(df) - len(clean)} row(s) with missing values.")
    return clean[features], clean[TARGET_COL].astype(int)


# ---------------- Sidebar ----------------
st.sidebar.title("💳 Default Risk Lab")
st.sidebar.caption("UCI · Default of Credit Card Clients (Taiwan, 2005)")

uploaded = st.sidebar.file_uploader("1 · Upload test CSV", type=["csv"])
use_sample = st.sidebar.checkbox("…or use the bundled test_data.csv", value=uploaded is None)
if SAMPLE_CSV.exists():
    st.sidebar.download_button("Download sample test_data.csv", SAMPLE_CSV.read_bytes(),
                               file_name="test_data.csv", mime="text/csv")

models = load_models()
model_name = st.sidebar.selectbox("2 · Choose a model", list(models.keys()))

# ---------------- Main ----------------
st.title("Credit Card Default Prediction — Model Comparison")
st.write("Predicts whether a credit-card client defaults on next month's payment "
         "(`DEFAULT` = 1) using five pre-trained classifiers. Upload held-out test data to "
         "evaluate them — the models are **not** retrained on your upload.")

if uploaded is not None:
    raw = pd.read_csv(uploaded)
    source = uploaded.name
elif use_sample and SAMPLE_CSV.exists():
    raw = pd.read_csv(SAMPLE_CSV)
    source = "test_data.csv (bundled)"
else:
    st.info("Upload a CSV in the sidebar (it must contain all 23 feature columns plus `DEFAULT`).")
    st.stop()

features = load_feature_columns()
X, y = validate(raw, features)

st.caption(f"Data source: **{source}**")
c1, c2 = st.columns(2)
c1.metric("Rows evaluated", f"{len(X):,}")
c2.metric("Default rate", f"{y.mean():.1%}")

with st.expander("Preview uploaded data"):
    st.dataframe(raw.head(20), use_container_width=True)

tab_single, tab_compare = st.tabs([f"🔍 {model_name}", "📊 Compare all 5 models"])

with tab_single:
    metrics, y_pred = evaluate(models[model_name], X, y)
    st.subheader(f"Evaluation metrics — {model_name}")
    items = list(metrics.items())
    for row in (items[:3], items[3:]):
        for col, (k, v) in zip(st.columns(3), row):
            col.metric(k, f"{v:.4f}")

    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(3.6, 3.0), dpi=90)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No default", "Default"], yticklabels=["No default", "Default"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, y_pred, labels=[0, 1],
                                       target_names=["No default (0)", "Default (1)"],
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

with tab_compare:
    st.subheader("All models on the same uploaded data")
    rows = [{"Model": n, **evaluate(m, X, y)[0]} for n, m in models.items()]
    table = pd.DataFrame(rows).set_index("Model")
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#c6f0d0"),
                 use_container_width=True)
    metric_to_plot = st.selectbox("Metric to chart", list(table.columns), index=5)
    st.bar_chart(table[metric_to_plot])
    best = table["MCC"].idxmax()
    st.success(f"Best by MCC (robust to the ~78/22 class imbalance): **{best}** "
               f"(MCC = {table.loc[best, 'MCC']:.4f})")
