# Rookie Longevity Lab: predicting NBA career length from rookie-season stats

**Course:** Machine Learning, M.Tech (AIML), BITS Pilani WILP: Assignment 2

| Link | URL |
|---|---|
| GitHub repository | `<PASTE-YOUR-GITHUB-REPO-URL>` |
| Live Streamlit app | `<PASTE-YOUR-STREAMLIT-APP-URL>` |

---

## a. Problem statement

Making it to the NBA is hard, and staying is harder. Given a player's **rookie-season
statistics**, the task is to predict a **binary outcome**: will the player still have a
career of **five or more years** in the league (`1`) or not (`0`)?

Five classification models (Logistic Regression, Decision Tree, K-Nearest Neighbours,
Gaussian Naive Bayes and Support Vector Machine) are trained on the same data, compared
with six evaluation metrics, and exposed through an interactive Streamlit app that loads
the saved models and scores an uploaded test CSV **without retraining**.

## b. Dataset description

| Item | Value |
|---|---|
| Name | NBA rookie career longevity (5-year target) |
| Type | Binary classification |
| Instances | 1,340 rows (1,328 after removing 12 exact duplicate rows) |
| Features | **19** numeric rookie-season inputs (minimum required: 12) |
| Target | `TARGET_5Yrs`: 1 = career of 5+ years, 0 = shorter |
| Class balance (after cleaning) | 824 positive (62.1%) / 504 negative (37.9%) |
| Source | Public NBA rookie-longevity dataset (Kaggle / data.world); this copy was obtained from the public `pycaret/datasets` GitHub repository (`nba.csv`) |

**Raw features (per-game averages unless noted):** `GP` (games played in the rookie
season), `MIN`, `PTS`, `FGM`, `FGA`, `FG%`, `3P Made`, `3PA`, `3P%`, `FTM`, `FTA`,
`FT%`, `OREB`, `DREB`, `REB`, `AST`, `STL`, `BLK`, `TOV`. The `Name` column is an
identifier and is not used as a feature.

**Preprocessing and feature engineering** (`features.py`, shared by training and the app):

1. `3P%` is missing only for players with zero three-point attempts (10 rows), so it is set to 0.
2. Five engineered columns are added: `pts_per_min`, `reb_per_min`, `ast_tov_ratio`,
   `three_attempt_share` (3PA / FGA) and `ft_attempt_rate` (FTA / FGA), giving 24 model inputs.
3. Features are standardised (`StandardScaler`) for Logistic Regression, KNN, Naive Bayes
   and SVM. The Decision Tree uses unscaled features.
4. All preprocessing is stored **inside** each saved scikit-learn `Pipeline`, so the
   app can score a raw CSV directly.

**Evaluation protocol:** stratified 80/20 split (`random_state=7`) → 1,062 training rows
and a held-out test set of **266 rows** (165 positive / 101 negative, saved as
`test_data.csv`). Hyper-parameters were tuned with 5-fold stratified cross-validation
(scoring = ROC-AUC) on the **training rows only**; the test set was used once for the
final metrics below.

## c. GitHub repository

`<PASTE-YOUR-GITHUB-REPO-URL>`

```
project-folder/
├── app.py               # Streamlit app (loads saved models, never retrains)
├── features.py          # shared preprocessing / feature engineering
├── evaluation.py        # shared metric code (training + app)
├── train_models.py      # trains, tunes, evaluates and saves all 5 models
├── requirements.txt
├── README.md
├── test_data.csv        # held-out test set (266 rows, raw columns + target)
├── data/nba_raw.csv     # full dataset used for training
└── model/               # saved pipelines (.joblib), metrics_summary.csv, best_params.json
```

**Run locally**

```bash
pip install -r requirements.txt
python train_models.py        # optional: regenerates model/ and test_data.csv
streamlit run app.py
```

## d. Models used

### Comparison table (held-out test set, 266 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:--|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 0.7293 | 0.7247 | 0.7487 | 0.8485 | 0.7955 | 0.4069 |
| Decision Tree | 0.6579 | 0.6710 | 0.7229 | 0.7273 | 0.7251 | 0.2724 |
| KNN | 0.6729 | 0.6971 | 0.7167 | 0.7818 | 0.7478 | 0.2873 |
| Naive Bayes | 0.6203 | 0.7122 | 0.8333 | 0.4848 | 0.6130 | 0.3299 |
| Support Vector Machine | 0.7030 | 0.7195 | 0.7263 | 0.8364 | 0.7775 | 0.3454 |

Precision, Recall and F1 are for the positive class (career of 5+ years). Always
predicting "5+ years" would score 0.6203 accuracy on this test set.

Selected hyper-parameters (5-fold CV): Logistic Regression `C=0.1`; Decision Tree
`max_depth=3, min_samples_leaf=20`; KNN `k=25, uniform weights`; Naive Bayes
`var_smoothing=1e-9`; SVM (RBF) `C=0.5, gamma=0.01`.

### Observations on model performance

| ML Model Name | Observation about model performance |
|:--|:--|
| Logistic Regression | Best model on accuracy (0.729), AUC (0.725), F1 (0.796) and MCC (0.407), and also the best 5-fold CV AUC (0.775). It finds 140 of the 165 durable careers (recall 0.849) but is optimistic: 47 of 101 short careers are wrongly flagged as 5+ years. The largest standardised coefficient is `GP` (games played), followed by offensive rebounds and rebounds per minute. A strongly regularised linear model (`C=0.1`) suits correlated box-score stats, and the signal appears to be largely monotonic in playing time and activity. |
| Decision Tree | Weakest model overall (accuracy 0.658, AUC 0.671, MCC 0.272). Cross-validation chose a very shallow tree (depth 3, 8 leaves), and about 71% of its importance sits on `GP`, so it is essentially a rule on games played plus a few refinements. Few leaves give coarse probability scores, which limits AUC. Deeper trees were not selected, which suggests they overfit this small dataset. It is the most interpretable model, at the cost of accuracy. |
| KNN | Mid-table: accuracy 0.673, AUC 0.697, MCC 0.287. Tuning picked a large neighbourhood (k = 25), which means local neighbourhoods are noisy and heavy smoothing helps. With 24 correlated, scaled inputs, distance-based similarity is not very discriminative, and every training row must be kept at prediction time. |
| Naive Bayes | Accuracy (0.620) is identical to the always-"5+ years" baseline, so on accuracy alone it adds nothing. It has the highest precision (0.833) but the lowest recall (0.485): it predicts "5+ years" for only 36% of players, missing 85 of the 165 durable careers. Its AUC (0.712) is third best, so it ranks players reasonably well, and the weakness is mainly the 0.5 decision threshold. The independence assumption is likely violated by strongly related stats (FGM/FGA/PTS, OREB/DREB/REB), which probably makes its probabilities over-confident. |
| Support Vector Machine | Close second to Logistic Regression: accuracy 0.703, AUC 0.720, F1 0.778, MCC 0.345, recall 0.836, with 52 false positives. The chosen RBF settings (small `gamma = 0.01`, `C = 0.5`) give a very smooth, almost linear boundary, which is consistent with the linear model performing best. The extra flexibility of the kernel brought no gain here. |
| **Overall, winner for your dataset?** | **Logistic Regression.** It leads on accuracy, AUC, recall, F1 and MCC on the test set and on cross-validated AUC, and it is also the simplest and most interpretable of the strong models. Caveat: the test set has only 266 rows, so the gap to the SVM (about 7 players' worth of accuracy) is within sampling noise; the consistent ranking across test and CV results is what supports choosing Logistic Regression. All AUCs sit between 0.67 and 0.72, so rookie-season box-score stats are only moderately predictive of career length. |

## Streamlit app features

- **Dataset upload (CSV):** upload test rows only (the models are already trained). The file must contain the 19 raw stat columns and `TARGET_5Yrs` (0/1); clear errors are shown otherwise. If nothing is uploaded, the bundled `test_data.csv` is used so the app opens with results.
- **Model selection dropdown:** all five classifiers.
- **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1 and MCC, plus an ROC curve.
- **Confusion matrix and classification report** for the selected model.
- **Compare all models** table on the uploaded data, and a downloadable predictions CSV.

The app loads the saved `.joblib` pipelines from `model/` and never refits on uploaded data.

## Deployment notes (Streamlit Community Cloud)

- `requirements.txt` pins the exact library versions used for training. This matters because pickled scikit-learn models are not guaranteed to load under a different scikit-learn version.
- In *Advanced settings* when deploying, choose **Python 3.13** (the version used to train the models). Main file: `app.py`.
