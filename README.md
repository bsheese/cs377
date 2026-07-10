# cs377 — Machine Learning

Course materials for CS 377. Each unit is a folder of numbered Jupyter
notebooks with explanations before every code cell, plus an outline, glossary,
discussion questions, and a practice quiz.

## Units

| Unit | Topic |
|---|---|
| `16_ml_intro` | What ML is; NumPy arrays; Pandas Series and DataFrames |
| `17_regression_crossval` | Regression and cross-validation: spread, associations, residuals, R² (17_0); simple linear regression (17_1); multiple linear regression on the Ames Housing data — cleaning, feature selection, regularization, hyperparameter tuning, nested CV (17_2); interaction terms (17_3) |
| `18_classification` | Classification basics, confusion matrices, ROC/AUC, imbalanced data (18_1); logistic regression (18_2); multi-class classification (18_5); trees and ensembles — bagging, random forests, boosting (18_6) |

Notebooks within a unit are meant to be read in numeric order; `*_9_*`
notebooks are student exercises. The five `17_2_1_*` notebooks are a sequential
pipeline — each depends on decisions made in the one before it.

## Practice quizzes

Self-serve practice quizzes for every unit are published at
**https://bsheese.github.io/cs377/** — no install needed. (Quiz sources live in
`quizzes/`; see `quiz_site/README.md` for how the site is built.)

## Running the notebooks

**In Colab:** open a notebook from GitHub. Datasets are fetched from the web at
runtime, so most notebooks just run. Notebooks that import a shared cleaning
module (`ames_cleaning.py`, `classification_cleaning.py`) include a Colab
download cell near the top.

**Locally:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Shared modules

- `17_regression_crossval/17_2_MLR/ames_cleaning.py` — full Ames Housing
  cleaning pipeline (`load_and_clean_ames`), used by the 17_2 notebooks.
- `18_classification/classification_cleaning.py` — Titanic and breast-cancer
  loaders used across unit 18.
