"""Metric helpers shared by train_models.py and the Streamlit app."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef,
)

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def positive_scores(model, X):
    """Continuous score for the positive class (probability if available, otherwise
    the SVM decision value; AUC only needs a ranking, not calibrated probabilities)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return np.asarray(model.decision_function(X))


def compute_metrics(model, X, y_true):
    """Return the six assignment metrics for a fitted model on (X, y_true)."""
    y_pred = model.predict(X)
    scores = positive_scores(model, X)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, scores),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
