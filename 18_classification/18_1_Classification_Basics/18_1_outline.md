# 18_1 Classification Basics — Topic Outline

This document provides a complete outline of all topics covered across the six notebooks in the 18_1 Classification Basics series.

---

## 18_1_1: Classification Foundations

**Dataset:** South German Credit (`credit-g`) — 1000 rows, 21 features, binary target (good/bad credit), ~70/30 class distribution.

### Regression vs. Classification
- Linear regression predicts continuous values; classification predicts discrete categories
- Classification requires bounded probabilities (0 to 1)
- XGBoost uses gradient boosting: sequential trees that correct errors
- Each tree makes threshold-based splits on features
- Ensemble of trees produces probability predictions

### Class Imbalance and the Accuracy Paradox
- Class distribution visualization (70/30 split)
- The naive baseline: always predicting the majority class
- Why accuracy is deceptive on imbalanced data
- Absolute improvement vs. relative improvement

### Data Preparation for XGBoost
- Binary encoding of the target variable (good=0, bad=1)
- One-hot encoding of categorical features (`drop_first=True`)
- Stratified train-test split (preserving class ratios)
- Note: Tree-based models do NOT require feature scaling

### Training the Model
- Gradient boosting: sequential tree correction
- `scale_pos_weight`: compensating for class imbalance (ratio of negatives/positives)
- Key parameters: n_estimators, max_depth, learning_rate
- Training with and without class weighting for comparison

### Interpreting Feature Importance
- Importance based on gain (loss reduction), cover (samples affected), frequency (times used)
- Top 5 and bottom 5 features by importance
- Unlike log-odds, importance doesn't tell direction (positive/negative effect)
- Direction requires additional analysis (e.g., partial dependence plots)

### Probabilities and the Decision Threshold
- Hard predictions (`.predict()`) vs. soft predictions (`.predict_proba()`)
- The default 0.5 threshold
- Overlapping probability distributions by actual class
- Quantifying false negative and false positive rates at the threshold

### Basic Model Evaluation
- Model accuracy vs. naive baseline
- Absolute improvement (e.g., +6%) vs. relative improvement (e.g., +9%)
- When to use each interpretation

---

## 18_1_2: Confusion Matrix and Basic Metrics

**Dataset:** South German Credit (`credit-g`)

### The Confusion Matrix
- Terminology: TP, TN, FP, FN
- Type I error (false alarm) vs. Type II error (missed case)
- Visualizing the confusion matrix with labeled cells
- Business interpretation: cost of denying good customers vs. approving defaulters

### Precision and Recall
- Precision: reliability of positive predictions (TP / TP + FP)
- Recall: sensitivity — how many actual positives were caught (TP / TP + FN)
- The precision-recall trade-off
- Word association trick: "Precision = positive predictive value"
- The "denominator trick" — precision denominator is predicted positives, recall denominator is actual positives

### The F1-Score
- Harmonic mean of precision and recall
- Why harmonic mean (not arithmetic): punishing extreme imbalances
- Concrete example: precision=100%, recall=1% → F1≈2%

### The Classification Report
- Per-class precision, recall, and f1-score
- Support: number of samples per class
- Macro average vs. weighted average: when to use each
- How to read per-class metrics and identify model weaknesses

### The Threshold Trade-Off (Preview)
- Moving the threshold: conservative vs. sensitive
- Teaser for ROC curves and systematic threshold optimization

---

## 18_1_3: Examples and Practice

**Dataset:** Adult Census Income — binary target (income >50K or ≤50K).

### Applying Concepts from Notebooks 1 & 2
- Load and explore a new dataset
- Examine target variable distribution (imbalance check)
- Data preparation: encoding, splitting, class weighting

### Modeling Pipeline
- Train XGBoost on the Adult dataset
- Confusion matrix analysis with business interpretation
- Precision, recall, and F1-score evaluation
- Full classification report

### Feature Importance
- Interpreting which features drive income predictions
- Top contributing features and their practical meaning

### Probability Distribution Analysis
- Reading the probability distribution by actual class
- Understanding model confidence and separation quality

### Summary
- End-to-end application of concepts from the first two notebooks
- Confirmation that the classification workflow generalizes across datasets

---

## 18_1_4: ROC, AUC, and Threshold Tuning

**Dataset:** South German Credit (`credit-g`)

### The ROC Curve
- True Positive Rate (TPR = recall) vs. False Positive Rate (FPR = 1 − specificity)
- Reading the curve: perfect model (top-left), random model (diagonal)
- AUC interpretation: probability of correctly ranking a random positive above a random negative
- AUC grading scale: 0.5 (random) → 0.7–0.8 (acceptable) → 0.8–0.9 (excellent) → 1.0 (perfect)

### Precision-Recall Curves
- Why ROC can be over-optimistic on imbalanced data
- PR curve baseline: the positive class prevalence
- When to prefer PR curves over ROC curves

### Finding the Optimal Decision Threshold
- Youden's J Statistic: J = TPR − FPR (maximizes distance from diagonal)
- Comparing Youden's J to the default 0.5 threshold

### Business Cost Sensitivity
- Assigning dollar costs to false positives and false negatives
- Computing total cost at each threshold
- The cost curve visualization
- Comparing three thresholds: default (0.5), Youden's J, cost-optimal

### Summary
- Three threshold selection methods: default, statistical, business-driven
- Evaluation tools: ROC curve, PR curve, cost curve

---

## 18_1_5: Credit Card Fraud Detection

**Dataset:** Credit Card Fraud (OpenML 1597) — 284,807 transactions, 31 features, extreme imbalance (~0.17% fraudulent).

### The Imbalance Problem
- Extreme class imbalance: far more severe than the credit dataset
- Naive baseline accuracy exceeds 99.8%
- XGBoost struggles: high precision but near-zero recall for fraud

### Model Evaluation on Fraud Data
- Confusion matrix reveals the imbalance trap
- Performance on fraud class (Class 1) is very poor without intervention
- Macro vs. weighted averages diverge dramatically
- Practical business impact: most fraudulent transactions go undetected

### Precision-Recall Analysis
- Skipping ROC in favor of PR curves for extreme imbalance
- PR AUC and the PR curve shape
- The F-Beta Score: weighting recall more heavily than precision
  - F2-Score: beta=2 means recall counts twice as much as precision
  - Finding the "sweet spot" where F-beta peaks

### Business Cost Analysis
- Consistent cost values for false positives vs. false negatives
- Out-of-Fold (OOF) probabilities for honest evaluation
- Three approaches to threshold selection:
  - Youden's J Statistic (statistical balanced approach)
  - Cost-based optimization (business-driven approach)
  - The "generalization gap" between OOF and test set
- Comparing recommendations from data-driven vs. business-driven thresholds

### Recommendations for Improvement
- Strategies to address the extreme imbalance
- Connection to hyperparameter tuning (next notebook)

---

## 18_1_6: Credit Card Fraud — Hyperparameter Tuning with Nested Cross-Validation

**Dataset:** Credit Card Fraud (OpenML 1597)

### Hyperparameter Tuning with GridSearchCV
- Systematically searching for optimal XGBoost parameters
- Parameter grid: max_depth, learning_rate, n_estimators, scale_pos_weight, subsample, etc.
- Why tuning matters more for the fraud dataset than the credit dataset

### Nested Cross-Validation
- Why standard CV overestimates performance when hyperparameters are tuned on the same data
- Inner loop: grid search within each outer training fold
- Outer loop: honest evaluation on held-out test folds
- Interpreting nested CV results for the fraud model

### Final Model Evaluation
- Building the final production model with optimal parameters
- Evaluating on the held-out test set
- Summary of improvement over the untuned baseline

### Key Takeaways
- Hyperparameter tuning can significantly improve recall on the minority class
- Nested CV provides honest performance estimates
- The complete classification workflow: from raw data to tuned, evaluated model

---

## Supporting Materials

| File | Description |
|------|-------------|
| `18_1_practice_quiz.md` | Practice quiz for assessment |
| `18_1_discussion_questions.md` | Discussion questions for the topic |
| `18_1_glossary.md` | Key terminology definitions |

---

## Cross-Cutting Themes

| Theme | Notebooks |
|---|---|
| **Class imbalance** | 18_1_1, 18_1_2, 18_1_3, 18_1_4, 18_1_5, 18_1_6 |
| **Threshold tuning** | 18_1_1, 18_1_2, 18_1_4, 18_1_5 |
| **Precision/Recall trade-off** | 18_1_2, 18_1_3, 18_1_4, 18_1_5 |
| **Confusion matrices** | 18_1_2, 18_1_3, 18_1_5 |
| **Classification report** | 18_1_2, 18_1_3, 18_1_5 |
| **Feature importance** | 18_1_1, 18_1_3, 18_1_5 |
| **Business cost analysis** | 18_1_2, 18_1_4, 18_1_5 |
| **Out-of-fold probabilities** | 18_1_5 |
| **Nested cross-validation** | 18_1_6 |
| **F-Beta Score** | 18_1_5 (fraud detection) |
