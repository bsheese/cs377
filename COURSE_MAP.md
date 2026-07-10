# cs377 Course Map

One page showing the whole arc: what each unit teaches, what it assumes, and
where each core concept is *first* taught versus reprised. Per-unit detail
lives in each unit's `*_outline.md`; this file is the view across units.

The follow-on deep-learning course (PyTorch, CNNs, NLP) is a separate repo
(`cs387`); this course ends with ensembles and a bridge to neural networks.

---

## The Arc

```
16  Tools          →  17  Regression         →  18  Classification
    (NumPy, pandas)       (predict a number)        (predict a category)

17_0 → 17_1 → 17_2 → 17_3        18_1 → 18_2 → 18_5 → 18_6
prelim  SLR    MLR    inter-      eval   logreg  multi-  trees &
                       actions    first          class   ensembles
```

**The through-line of the whole course is honest evaluation.** Every unit
advances one storyline: how do you know your model will work on data it has
never seen? Baselines → train/test split → cross-validation → pipelines (no
leakage) → hyperparameter tuning → nested CV. Models change; that question
never does.

## Units at a Glance

| Unit | Role | Primary datasets | Assumes |
|---|---|---|---|
| `16_ml_intro` | Tooling + what ML is. 16_4/16_5 are **reference notebooks** (skim, return on demand) | Song data (DJ problem), Auto MPG | Intro Python (lists, loops, functions) |
| `17_0_Preliminaries` | Statistical foundations, built by hand: spread, association, residuals, R² | Heights, penguins, Anscombe | 16 |
| `17_1_SLR` | Simple linear regression end-to-end: inference, assumptions, influence, transformations, generalization | Penguins, Auto MPG, Ames (preview), Gapminder | 17_0 |
| `17_2_MLR` | The Ames five-part serial: cleaning → selection → regularization → tuning → nested CV; plus regression trees (17_2_2) | Ames Housing | 17_1 |
| `17_3_Interactions` | One case study: when one feature's effect depends on another | Medical insurance costs | 17_1 (statsmodels formulas) |
| `18_1_Classification_Basics` | **Evaluation-first**: metrics machinery using XGBoost as a black box; three end-of-unit projects (worked / optional / capstone) | German Credit, credit-card fraud, Bank Marketing, Hotel, Telco | 17_2 (pipelines, CV, nested CV) |
| `18_2_LogisticRegression` | Opens the hood on a classifier; interpretability; bridge to neural networks | Titanic, German Credit | 18_1 |
| `18_5_MutliClassClassification` | Two classes → K classes: softmax probabilities, K×K confusion matrix, macro vs. weighted, imbalance | Penguins, fetal health (CTG), wine quality | 18_1 |
| `18_6_Ensemble` | Tree mechanics for classification: bagging, random forests, boosting, head-to-head nested CV | Wisconsin breast cancer | 17_2_2 (tree basics), 18_1 (metrics) |

**Sequencing notes.**
- Unit 18 is deliberately *evaluation-first*: XGBoost is used as a black box
  throughout 18_1; its mechanics arrive in 18_6, and logistic regression (18_2)
  is the first model whose insides are taught. 18_1_1 says this to students
  explicitly.
- Tree mechanics are taught **in the regression unit** (17_2_2) and assumed by
  18_6 — a student skipping 17_2_2 will miss the foundation for ensembles.
- 17_3 can float: it needs 17_1's statsmodels fluency but nothing from 17_2.

## Where Concepts Are First Taught (and Reprised)

| Concept | First taught | Reprised / deepened |
|---|---|---|
| Baseline model (predict the mean / majority class) | 17_0_1 | 17_1, 18_1_1 (accuracy paradox) |
| TSS, variance, standard deviation | 17_0_1 | throughout |
| Correlation (Pearson's r) | 17_0_3 | 17_2_1_2 (as a shortlist filter, with its bias) |
| Residuals, RSS, R² | 17_0_4, 17_0_5 | 17_1_1, 17_2 (in dollars) |
| sklearn vs. statsmodels (predict vs. explain) | 17_1_1 | 18_2_2 (Logit for inference) |
| p-values, standard error, confidence intervals | 17_1_2 | 18_2_2 (CIs on odds ratios) |
| Regression assumptions (LINE), residual diagnostics | 17_1_3 | 17_2_1_3 (residual analysis) |
| Leverage, Cook's distance, ethics of dropping data | 17_1_4 | 17_2_1_1 (outlier policy) |
| Log transforms & interpretation | 17_1_5 | 17_2_1_1 (Ames), Log-Dollar Illusion (17_2_1_3) |
| Train/test split, overfitting, bias-variance | 17_1_6 | everywhere; trees version in 17_2_2 |
| Data cleaning, leakage, deterministic-vs-statistical rule | 17_2_1_1 | 18_1_9_x (leakage-aware capstones) |
| Feature selection (forward/backward), VIF | 17_2_1_2 | 17_2_1_3 (regularization as the alternative) |
| Pipelines (scaler inside CV) | 17_2_1_3 | all later modeling |
| Regularization (Ridge/Lasso/ElasticNet) | 17_2_1_3 | 18_6_3 (XGBoost's built-in reg), weight decay pointer |
| Hyperparameters, GridSearchCV, validation curve | 17_2_1_4 | 18_1_6, 18_6_2/3 |
| Nested cross-validation | 17_2_1_4 → deep dive 17_2_1_5 | 18_1_6, 18_1_9_x, 18_6_4 |
| Decision trees (regression) | 17_2_2 | 18_6_1 (classification version) |
| Interaction effects | 17_3_1 | implicitly: trees "capture interactions" (17_2_2, 18_6) |
| Class imbalance & the accuracy paradox | 18_1_1 | 18_5_3 (multiclass), 18_6_2 (class_weight) |
| Confusion matrix, precision/recall/F1 | 18_1_2 | 18_5_1/2 (K×K, macro/weighted), 18_6 |
| ROC/AUC, PR curves, threshold tuning | 18_1_4 | 18_1_5 (why ROC misleads under extreme imbalance), 18_2_1 |
| Cost-sensitive decisions (business thresholds) | 18_1_4/18_1_5 | 18_1_9_x capstones |
| Odds, odds ratios, sigmoid, calibration | 18_2_1/18_2_2 | 18_6_3 (calibration of boosted models) |
| Neurons, hidden layers, XOR, activation functions | 18_2_3 | continued in cs387 |
| Softmax, one-hot targets, macro vs. weighted F1 | 18_5_1/18_5_2 | continued in cs387 |
| Sample weights for imbalance | 18_5_3 | 18_6_2 (`class_weight`) |
| Bagging, OOB error, random forests | 18_6_2 | — |
| Boosting (AdaBoost, gradient), BART | 18_6_3 | — |

## Ethics Thread (deliberate, worth naming to students)

- 17_1_4 — when you may and may not drop a data point
- 17_3_1 — omitted interactions concentrate pricing errors on specific groups
- 18_1_4/5 — decision thresholds as business/cost choices, not math constants
- 18_2_2 — regulated domains require interpretable models (credit, medicine)
- 18_5_3 — the rare class is usually the high-stakes class

## Maintenance Notes

- Support files per unit: `*_outline.md`, `*_glossary.md`,
  `*_discussion_questions.md`, `*_practice_quiz.ipynb`; quiz site source in
  `quizzes/`, deployed via GitHub Pages.
- Smoke-test all notebooks: `python smoke_test.py` (see file header for usage).
- Shared cleaning modules: `17_2_MLR/ames_cleaning.py`,
  `18_classification/classification_cleaning.py`.
- Datasets are vendored in `data/` and loaded from this repo's raw GitHub URL
  (course-controlled hosting; changes take effect on push to `main`). Only
  `sns.load_dataset`, `fetch_openml`, and sklearn-bundled data still come from
  outside the repo.
