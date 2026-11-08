# Credit Card Default Prediction — ML Assignment 2

Live app: `<STREAMLIT_APP_URL>`  ·  Repository: https://github.com/santoshapatil/BITS-ML-assignemnt-2

## a. Problem statement

Banks issuing credit cards need to know, ahead of time, which clients are likely to miss next
month's payment so they can adjust limits, prioritise collections, or offer restructuring.
The task is a **binary classification** problem: given a client's credit limit, demographics,
and six months of repayment status, bill amounts and payment amounts, predict whether the
client will **default on the next month's payment** (`DEFAULT` = 1) or not (`DEFAULT` = 0).

Five classifiers (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM) are
trained on the same data, compared on six metrics, and served through an interactive
Streamlit app that scores an uploaded test CSV with the saved models (no retraining).

## b. Dataset description

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository — *Default of Credit Card Clients* (Yeh & Lien, 2009), id 350 |
| URL | https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients |
| Instances | 30,000 clients (Taiwan, April–September 2005) |
| Features | 23 (all numeric / integer-coded) — the `ID` column is dropped |
| Target | `DEFAULT` (renamed from `default payment next month`): 1 = default, 0 = no default |
| Class balance | 23,364 no-default (77.9%) / 6,636 default (22.1%) — imbalanced |
| Missing values | None |

Feature groups:

- `LIMIT_BAL` — credit limit (NT dollars), individual + family credit
- `SEX` (1 male, 2 female), `EDUCATION` (1 grad school, 2 university, 3 high school, 4 others), `MARRIAGE` (1 married, 2 single, 3 others), `AGE`
- `PAY_0, PAY_2 … PAY_6` — repayment status Sep→Apr (-1 paid duly, 1..9 months delayed)
- `BILL_AMT1 … BILL_AMT6` — bill statement amount Sep→Apr
- `PAY_AMT1 … PAY_AMT6` — amount actually paid Sep→Apr

**Split:** stratified 80/20 train/test (`random_state=42`) → 24,000 train / 6,000 test.
The 6,000-row hold-out set is saved as [`test_data.csv`](test_data.csv) and is what the app
expects to be uploaded.

**Preprocessing:** `StandardScaler` inside a scikit-learn `Pipeline` for LR, KNN, NB and SVM
(distance/margin-based models are scale-sensitive); the Decision Tree uses raw features. The
scaler is fitted on the training split only and is saved inside each pipeline, so the app
applies identical preprocessing.

## c. GitHub repository link

https://github.com/santoshapatil/BITS-ML-assignemnt-2

```
ml-assignment-2/
├── app.py                  # Streamlit app (upload CSV → pick model → metrics, confusion matrix, report)
├── train_models.py         # trains + evaluates all 5 models, writes model/ and test_data.csv
├── metrics_utils.py        # shared metric computation (training and app use the same code)
├── requirements.txt
├── README.md
├── test_data.csv           # 6,000-row stratified hold-out set (features + DEFAULT)
├── data/credit_default_full.csv   # full UCI dataset converted from .xls to .csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── feature_columns.json
    └── metrics.csv         # the comparison table below
```

## d. Models used

| Model | Configuration |
|---|---|
| Logistic Regression | `StandardScaler` → `LogisticRegression(max_iter=2000)` |
| Decision Tree | `DecisionTreeClassifier(max_depth=6, min_samples_leaf=50)` |
| KNN | `StandardScaler` → `KNeighborsClassifier(n_neighbors=25, weights="distance")` |
| Naive Bayes | `StandardScaler` → `GaussianNB()` |
| Support Vector Machine | `StandardScaler` → `SVC(kernel="rbf", C=1.0, gamma="scale")` (AUC from `decision_function`) |

### Comparison table (6,000-row hold-out test set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8077 | 0.7076 | 0.6868 | 0.2396 | 0.3553 | 0.3244 |
| Decision Tree | **0.8167** | **0.7508** | 0.6697 | 0.3376 | 0.4489 | **0.3828** |
| KNN | 0.8095 | 0.7402 | 0.6314 | 0.3331 | 0.4361 | 0.3592 |
| Naive Bayes | 0.7525 | 0.7249 | 0.4515 | **0.5539** | **0.4975** | 0.3386 |
| Support Vector Machine | 0.8160 | 0.7210 | 0.6652 | 0.3384 | 0.4486 | 0.3809 |

Precision, Recall and F1 are for the positive class (`DEFAULT` = 1). A trivial "always predict
no default" model scores 0.779 accuracy, so accuracy alone is misleading here; MCC and AUC
are the more informative metrics for this imbalanced problem.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Lowest AUC (0.708) and MCC (0.324). It is conservative: high precision (0.687) but catches only 24% of defaulters. The default risk depends non-linearly on the delay codes in `PAY_0…PAY_6` (e.g. "-1 paid duly" vs "2 months late"), which a single linear boundary on scaled values cannot capture. |
| Decision Tree | Best accuracy (0.817), AUC (0.751) and MCC (0.383). Splits on `PAY_0` / `PAY_2` thresholds model the "how many months late" rule directly. Capping depth at 6 and leaf size at 50 kept it from overfitting; it is also the fastest and most interpretable model. |
| KNN | Second-best AUC (0.740) with k = 25 and distance weighting, but MCC (0.359) trails the tree and SVM. Performance is limited by many skewed bill/payment amount features that dominate the distance even after scaling, and prediction is the slowest at inference time because it stores all 24k training rows. |
| Naive Bayes | Very different trade-off: highest recall (0.554) and F1 (0.498) but the lowest accuracy (0.753) and precision (0.452). The independence assumption is badly violated (the six `BILL_AMT` columns are strongly correlated), so its probabilities are over-confident and it flags many non-defaulters. Useful only if missing a defaulter is far costlier than a false alarm. |
| Support Vector Machine | Close second on MCC (0.381) and accuracy (0.816); the RBF kernel captures the non-linear signal that Logistic Regression misses. AUC (0.721) is lower because the SVM's decision scores rank less well than they classify. Slowest to train (~15 s vs < 1 s for the others). |
| **Overall winner** | **Decision Tree** — best MCC, AUC and accuracy at the lowest cost, with interpretable rules. All models have modest recall because of the 78/22 class imbalance; class weighting or threshold tuning would be the next improvement. If catching defaulters matters most, Naive Bayes offers the highest recall. |

## Streamlit app features

1. **CSV upload** of test data (or one-click use of the bundled `test_data.csv`, also downloadable from the sidebar)
2. **Model selection dropdown** for the five saved models
3. **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1, MCC
4. **Confusion matrix** heat-map and **classification report**
5. **Compare-all tab**: all five models scored on the uploaded data, best value per metric highlighted, bar chart by metric

The app validates that the upload has all 23 feature columns plus `DEFAULT`, and only calls
`predict` / `predict_proba` / `decision_function` on the saved pipelines; it never retrains.

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py     # optional: re-creates model/*.joblib and test_data.csv
streamlit run app.py
```

## Deployment (Streamlit Community Cloud)

New app → this repository → branch `main` → main file `app.py` → Advanced settings → Python
**3.12** (the pinned numpy 1.26.4 has no 3.13 wheel) → Deploy.
