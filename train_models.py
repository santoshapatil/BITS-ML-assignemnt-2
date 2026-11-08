"""Train the five classifiers on the UCI 'Default of Credit Card Clients' dataset.

Run:  python train_models.py
Outputs:
  model/*.joblib            - one fitted pipeline per classifier
  model/feature_columns.json- expected input columns (order) for the app
  model/metrics.csv         - held-out test metrics for all models
  test_data.csv             - the 20% stratified hold-out split (features + target)
"""
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from metrics_utils import MODEL_FILES, TARGET_COL, evaluate

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "credit_default_full.csv"
MODEL_DIR = ROOT / "model"
SEED = 42


def build_models():
    scaled = lambda clf: Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return {
        "Logistic Regression": scaled(LogisticRegression(max_iter=2000, random_state=SEED)),
        # Trees don't need scaling; depth is capped to stop it memorising the training set.
        "Decision Tree": Pipeline([("clf", DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=50, random_state=SEED))]),
        "KNN": scaled(KNeighborsClassifier(n_neighbors=25, weights="distance")),
        "Naive Bayes": scaled(GaussianNB()),
        "Support Vector Machine": scaled(SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED)),
    }


def main():
    df = pd.read_csv(DATA)
    X, y = df.drop(columns=[TARGET_COL]), df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"train={X_train.shape}, test={X_test.shape}, default rate={y.mean():.3f}")

    MODEL_DIR.mkdir(exist_ok=True)
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test
    test_df.to_csv(ROOT / "test_data.csv", index=False)
    (MODEL_DIR / "feature_columns.json").write_text(json.dumps(list(X.columns)))

    rows = []
    for name, model in build_models().items():
        t0 = time.time()
        model.fit(X_train, y_train)
        metrics, _ = evaluate(model, X_test, y_test)
        joblib.dump(model, MODEL_DIR / MODEL_FILES[name], compress=3)
        rows.append({"Model": name, **metrics})
        print(f"{name:<24} trained in {time.time() - t0:5.1f}s  " +
              "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    out = pd.DataFrame(rows).set_index("Model").round(4)
    out.to_csv(MODEL_DIR / "metrics.csv")
    print("\n" + out.to_markdown())


if __name__ == "__main__":
    main()
