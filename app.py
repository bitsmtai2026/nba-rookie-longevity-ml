"""Streamlit front-end: NBA rookie career-longevity classifiers.

The app only LOADS the models saved by train_models.py and scores the uploaded
test CSV with them; nothing is retrained here.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from evaluation import METRIC_NAMES, compute_metrics, positive_scores
from features import RAW_FEATURES, TARGET_COL

APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "model"
SAMPLE_CSV = APP_DIR / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
CLASS_LABELS = ["Career < 5 yrs (0)", "Career >= 5 yrs (1)"]

st.set_page_config(page_title="Rookie Longevity Lab", page_icon="🏀", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    div[data-testid="stMetric"] {
        background: #f4f6fb; border-left: 5px solid #c8102e;
        padding: 0.7rem 1rem; border-radius: 6px;
    }
    @media (prefers-color-scheme: dark) {
        div[data-testid="stMetric"] {background: #1c2233;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model(file_name: str):
    return joblib.load(MODEL_DIR / file_name)


@st.cache_data(show_spinner=False)
def read_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV)


def validate(df: pd.DataFrame):
    """Return (clean_df, error_message). Needs the 19 raw stats and the target."""
    needed = RAW_FEATURES + [TARGET_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return None, f"The CSV is missing required column(s): {', '.join(missing)}"
    df = df.dropna(subset=[TARGET_COL]).copy()
    if not set(df[TARGET_COL].unique()) <= {0, 1}:
        return None, f"`{TARGET_COL}` must contain only 0 and 1."
    if df[TARGET_COL].nunique() < 2:
        return None, f"`{TARGET_COL}` has only one class in this file, so AUC and MCC can't be computed."
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df, None


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload a test CSV", type="csv")
    st.caption(
        "Needs the 19 rookie-stat columns plus `TARGET_5Yrs` (0/1). "
        "Upload test rows only: the models are already trained."
    )
    st.header("2 · Model")
    model_name = st.selectbox("Choose a classifier", list(MODEL_FILES))
    st.divider()
    st.caption("Models are loaded from the `model/` folder and are never retrained on your upload.")

# -------------------------------------------------------------------- body
st.title("🏀 Rookie Longevity Lab")
st.write(
    "Will an NBA rookie still be in the league five years later? Pick a saved "
    "model, feed it held-out rookie stats and see how well it does."
)

if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as exc:  # unreadable / not a CSV
        st.error(f"Could not read that file as CSV: {exc}")
        st.stop()
    source_note = f"Using your upload: **{uploaded.name}**"
else:
    raw_df = read_sample()
    source_note = "No file uploaded yet, so the bundled held-out `test_data.csv` is shown. Upload your own CSV in the sidebar."

data, problem = validate(raw_df)
if problem:
    st.error(problem)
    st.stop()

st.info(f"{source_note}  ·  {len(data)} rows")

model = load_model(MODEL_FILES[model_name])
y_true = data[TARGET_COL]
metrics = compute_metrics(model, data, y_true)

tab_metrics, tab_cm, tab_compare, tab_data = st.tabs(
    ["Metrics", "Confusion matrix & report", "Compare all models", "Data & predictions"]
)

with tab_metrics:
    st.subheader(f"{model_name}: evaluation metrics")
    cols = st.columns(len(METRIC_NAMES))
    for col, name in zip(cols, METRIC_NAMES):
        col.metric(name, f"{metrics[name]:.4f}")

    fpr, tpr, _ = roc_curve(y_true, positive_scores(model, data))
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot(fpr, tpr, color="#c8102e", lw=2, label=f"AUC = {metrics['AUC']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab_cm:
    y_pred = model.predict(data)
    left, right = st.columns([1, 1.2])
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.4, 3.8))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], labels=["Pred < 5", "Pred >= 5"])
        ax.set_yticks([0, 1], labels=["Actual < 5", "Actual >= 5"])
        for (i, j), value in pd.DataFrame(cm).stack().items():
            ax.text(j, i, int(value), ha="center", va="center",
                    color="white" if value > cm.max() / 2 else "black", fontsize=14)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Classification report")
        report = classification_report(
            y_true, y_pred, target_names=CLASS_LABELS, output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

with tab_compare:
    st.subheader("All five saved models on this data")
    rows = []
    for name, file_name in MODEL_FILES.items():
        rows.append({"ML Model Name": name, **compute_metrics(load_model(file_name), data, y_true)})
    table = pd.DataFrame(rows).set_index("ML Model Name")
    st.dataframe(
        table.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8cf"),
        width="stretch",
    )
    st.caption("Green = best value in each column.")

with tab_data:
    st.subheader("Rows scored by the selected model")
    out = data.copy()
    out["predicted"] = model.predict(data)
    out["P(5+ yrs) / score"] = positive_scores(model, data).round(4)
    st.dataframe(out, width="stretch", height=360)
    st.download_button(
        "Download predictions (CSV)",
        out.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{MODEL_FILES[model_name].split('.')[0]}.csv",
        mime="text/csv",
    )
