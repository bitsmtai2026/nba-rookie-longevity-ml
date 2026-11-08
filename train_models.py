"""Train the five classifiers on the NBA rookie-longevity data and save everything
the Streamlit app needs (models, metrics, held-out test CSV).

Run:  python train_models.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from evaluation import compute_metrics
from features import TARGET_COL, prepare_features

BASE = Path(__file__).parent
RAW_CSV = BASE / "data" / "nba_raw.csv"
MODEL_DIR = BASE / "model"
SPLIT_SEED = 7
CV_SEED = 11

MODEL_DIR.mkdir(exist_ok=True)


def build_search_spaces():
    """(display name, file stem, pipeline, parameter grid) for each classifier."""
    feat = lambda: FunctionTransformer(prepare_features)  # noqa: E731
    return [
        (
            "Logistic Regression", "logistic_regression",
            Pipeline([("feat", feat()), ("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=3000))]),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        (
            "Decision Tree", "decision_tree",
            Pipeline([("feat", feat()),
                      ("clf", DecisionTreeClassifier(random_state=SPLIT_SEED))]),
            {"clf__max_depth": [3, 4, 5, 7, None], "clf__min_samples_leaf": [5, 10, 20]},
        ),
        (
            "KNN", "knn",
            Pipeline([("feat", feat()), ("scale", StandardScaler()),
                      ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [5, 9, 15, 25], "clf__weights": ["uniform", "distance"]},
        ),
        (
            "Naive Bayes", "naive_bayes",
            Pipeline([("feat", feat()), ("scale", StandardScaler()),
                      ("clf", GaussianNB())]),
            {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]},
        ),
        (
            "Support Vector Machine", "svm",
            Pipeline([("feat", feat()), ("scale", StandardScaler()),
                      ("clf", SVC(kernel="rbf"))]),
            {"clf__C": [0.5, 1, 2, 5], "clf__gamma": ["scale", 0.01, 0.05]},
        ),
    ]


def main():
    raw = pd.read_csv(RAW_CSV)
    print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} columns")

    before = len(raw)
    raw = raw.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {before - len(raw)} exact duplicate rows -> {len(raw)} rows")

    train_df, test_df = train_test_split(
        raw, test_size=0.20, stratify=raw[TARGET_COL], random_state=SPLIT_SEED
    )
    y_train = train_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)

    # Held-out test file used by the Streamlit app (raw columns + target).
    test_df.to_csv(BASE / "test_data.csv", index=False)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows -> test_data.csv")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    rows, params_log = [], {}

    for name, stem, pipe, grid in build_search_spaces():
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(train_df, y_train)
        best = search.best_estimator_

        metrics = compute_metrics(best, test_df, y_test)
        rows.append({
            "ML Model Name": name,
            **metrics,
            "CV AUC (train, 5-fold)": search.best_score_,
        })
        params_log[name] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        joblib.dump(best, MODEL_DIR / f"{stem}.joblib")
        print(f"[{name}] best params: {params_log[name]}  CV AUC={search.best_score_:.4f}")

    results = pd.DataFrame(rows)
    results.round(4).to_csv(MODEL_DIR / "metrics_summary.csv", index=False)
    with open(MODEL_DIR / "best_params.json", "w") as fh:
        json.dump(params_log, fh, indent=2, default=str)

    print("\n=== Hold-out test metrics ===")
    print(results.round(4).to_string(index=False))
    try:  # optional pretty table; needs the `tabulate` package
        print("\nMarkdown table:\n")
        print(results.round(4).to_markdown(index=False))
    except ImportError:
        print("\n(Install 'tabulate' to also print a markdown table.)")


if __name__ == "__main__":
    main()
