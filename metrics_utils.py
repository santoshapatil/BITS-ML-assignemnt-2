"""Shared evaluation helpers used by both the training script and the Streamlit app."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET_COL = "DEFAULT"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def positive_scores(model, X):
    """Score for the positive class, used for AUC.

    SVC is trained without probability=True (Platt scaling on 24k rows is slow),
    so fall back to decision_function, which ranks samples just as well for AUC.
    """
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


def evaluate(model, X, y):
    y_pred = model.predict(X)
    scores = positive_scores(model, X)
    auc = roc_auc_score(y, scores) if len(np.unique(y)) > 1 else float("nan")
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }, y_pred
