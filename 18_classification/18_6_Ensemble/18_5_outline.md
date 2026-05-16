# 18_5 Ensemble Methods — Topic Outline

This document provides a complete outline of all topics covered across the five notebooks in the 18_5 Ensemble Methods series.

---

## 18_5_0: Intro to Ensemble Methods

**Dataset:** Wisconsin Breast Cancer — 569 samples, 30 features, binary target (malignant/benign), ~63/37 class distribution.

### Introduction: Building on What We Know
- Acknowledges prior tree coverage from 18_1_4 and 18_1_5
- Frames 18_5 as a deeper dive into ensemble methods with classification-specific concerns
- Focus on precision, recall, false negatives in medical diagnosis, and ROC curves

### The Ensemble Hierarchy
- All ensemble methods share the core idea: combine multiple weak models to create a strong one
- Comparison table: Single Tree, Bagging, Random Forest, Boosting, BART
  - How trees are built (parallel vs. sequential)
  - How predictions are combined (majority vote vs. weighted sum vs. posterior distribution)
  - Primary goal (interpretability vs. variance reduction vs. bias reduction vs. uncertainty quantification)
- Key insight: Bagging/RF reduce variance; Boosting reduces bias; BART addresses both + uncertainty

### The Dataset: Wisconsin Breast Cancer
- 569 samples, 30 features (mean, SE, and "worst" for 10 cell nucleus characteristics)
- Binary target: Malignant (1) or Benign (0)
- Class distribution: ~63% benign, ~37% malignant — moderately imbalanced
- High-stakes classification: false negatives (missed cancers) are far more costly than false positives (unnecessary biopsies)
- Why accuracy alone is insufficient for medical diagnosis

---

## 18_5_1: Decision Trees — The Foundation of Ensembles

**Dataset:** Wisconsin Breast Cancer

### Data Preparation
- Loading and exploring the breast cancer dataset
- Flipping target encoding to match original UCI (1=malignant, 0=benign)
- Visualizing key feature distributions with violin plots
- Identifying the most discriminative features (e.g., mean concave points)
- Train/test split with stratification

### A Single Decision Tree
- **Gini impurity explained:** Formula, numerical example, how trees choose splits
- Gini = 0 (pure) vs. Gini = 0.5 (maximally mixed)
- Alternative: entropy (information gain) — produces nearly identical trees
- Visualizing the tree flowchart with `plot_tree`
- Reading tree nodes: condition, Gini, samples, value, class, color
- Regression vs. classification tree comparison table (leaf predictions, split criteria, evaluation metrics)

### The Overfitting Problem: Depth vs. Generalization
- Training accuracy vs. testing accuracy across depths 1-20
- Adding malignant recall to the depth experiment (dual-axis plot)
- Best accuracy depth vs. best recall depth — they may differ
- The gap between training and testing curves = overfitting penalty
- Same bias-variance pattern as 17_2_4_5 with Ames data, now in classification context

### Beyond Accuracy: Classification Metrics
- Confusion matrix with labeled cells (TN, FP, FN, TP)
- Medical interpretation: missed cancers vs. unnecessary biopsies
- Classification report: precision, recall, F1 per class
- ROC curve and AUC: probability of correctly ranking a random malignant above a random benign

### Robust Evaluation: K-Fold Cross-Validation
- 10-fold CV reporting accuracy, recall, and F1
- Standard deviation as a measure of model stability
- Clinical interpretation: "1 in X cancers would be missed"

---

## 18_5_2: Bagging and Random Forests

**Dataset:** Wisconsin Breast Cancer

### Bagging: Fixing Instability
- How bagging works: bootstrap sampling → deep trees → majority vote
- Why it works: each tree overfits differently; averaging cancels noise
- **Why deep trees?** Contrast with boosting's shallow trees
  - Bagging: each tree needs to be strong on its bootstrap sample; variance is reduced by averaging
  - Boosting: each tree only corrects residual errors; deep trees overfit faster
- Out-of-Bag (OOB) error: ~1/3 of data held out per tree serves as built-in validation
- OOB vs. test accuracy comparison

### Handling Class Imbalance
- The `class_weight='balanced'` parameter
- Demonstrating the precision-recall tradeoff: default vs. balanced weights
- In medical context: catching more cancers at the cost of more false positives

### Random Forests: Decorrelating the Trees
- Bagging's flaw: correlation — dominant features used by every tree
- The Random Forest fix: random feature subsets at each split (`max_features='sqrt'`)
- Forces trees to explore alternative paths, decorrelating them

### Classification Metrics: Beyond Accuracy
- Confusion matrix with medical interpretation
- Classification report: precision, recall, F1 per class
- ROC curve and AUC

### Hyperparameter Tuning with GridSearchCV
- Tuning `n_estimators`, `max_depth`, `max_features`
- Comparing default (500 trees) vs. tuned model
- More trees ≠ always better; tuning finds simpler, equally effective models

### Feature Importance and Its Limitations
- Feature importance based on total impurity reduction
- The correlated features problem: importance split among correlated features
- Permutation importance as an alternative

### Comparing Stability: Single Tree vs. Ensembles
- 10-fold CV comparing single tree, bagging, and random forest
- Accuracy, F1, and malignant recall for each
- Standard deviation as stability measure

---

## 18_5_3: Boosting and BART

**Dataset:** Wisconsin Breast Cancer

### AdaBoost: Adaptive Boosting
- How AdaBoost works: equal weights → train stump → reweight misclassified → repeat → weighted voting
- **Why stumps (max_depth=1)?** Weak learners only slightly better than random; power comes from combining hundreds
- Contrast with bagging's deep trees: different strategies for different goals
- AdaBoost accuracy, F1, and malignant recall

### Gradient Boosting: Predicting the Residuals
- How Gradient Boosting works: log-odds prediction → calculate residuals → train tree on residuals → add scaled predictions → repeat
- Same concept as 17_2_4_5 with Ames Housing data
- Key hyperparameters: n_estimators, learning_rate (shrinkage), max_depth
- Gradient Boosting accuracy, F1, and malignant recall

### Classification Metrics: Full Evaluation
- Confusion matrix with medical interpretation
- Classification report
- ROC curve and AUC

### Hyperparameter Tuning with GridSearchCV
- Tuning n_estimators, learning_rate, max_depth
- The shrinkage trade-off: smaller learning rate needs more trees but often generalizes better
- Comparing default vs. tuned Gradient Boosting

### Comparing All Methods
- 10-fold CV: Single Tree, Random Forest, AdaBoost, Gradient Boosting
- Accuracy, F1, malignant recall, and AUC for each
- Gradient Boosting typically highest accuracy/F1; Random Forest most stable; AdaBoost sensitive to noise

### Probability Calibration
- Why calibration matters in medical contexts
- Random Forests: typically well-calibrated
- Gradient Boosting: often produces extreme probabilities
- Calibration curves: predicted probability vs. actual fraction of positives

### BART: Bayesian Additive Regression Trees
- How BART works: ensemble of shallow trees + MCMC perturbation + prior beliefs
- Key advantage: uncertainty quantification (credible intervals)
- Why BART isn't in the comparison (requires specialized libraries)

---

## 18_5_4: Model Comparison — All Ensemble Methods Head-to-Head

**Dataset:** Wisconsin Breast Cancer

### Nested Cross-Validation for Unbiased Comparison
- Recall from 17_2_4_4: inner loop tunes, outer loop evaluates
- Outer loop: 5-fold CV; Inner loop: 3-fold CV
- Computational cost quantified: 450 total model fits across 4 models
- Hyperparameter grids per model

### Visualizing the Comparison
- 4-panel boxplots: accuracy, F1, malignant recall, AUC
- Strip plot overlay showing actual 5 outer fold data points
- Honest visualization of limited sample size (5 points per model)

### Interpreting the Results
- **The Bias-Variance Story:** Decision Tree (high variance) → Bagging → Random Forest → Gradient Boosting (lowest bias)
- **The Malignant Recall Story:** Most clinically relevant metric; mean recall and variance across folds
- **Which Model Should You Choose?** Evidence-based decision table
- **Nested CV vs. Single Train/Test Split:** Comparison table showing optimistic bias gap

### Final Model: Best Performer on Full Data
- Selecting best model by malignant recall (not accuracy or F1)
- Training on full dataset with GridSearchCV
- Confusion matrix with FN/FP counts on full data (5-fold CV)
- Classification report on full data
- Feature importance from the final model

### Conclusion: The Ensemble Series Recap
- Single Decision Tree → Bagging → Random Forest → Gradient Boosting → BART
- Key takeaways: ensembles beat single trees, bagging reduces variance, boosting reduces bias, tuning matters, recall > accuracy in medical contexts, nested CV gives honest estimates

---

## Cross-Cutting Themes

These concepts appear throughout multiple notebooks:

| Theme | Notebooks |
|---|---|
| **Malignant recall** | 18_5_1 (depth experiment, CV), 18_5_2 (class_weight demo, CV comparison), 18_5_3 (all metrics, 4-model CV), 18_5_4 (nested CV, final model selection) |
| **Confusion matrix interpretation** | 18_5_1, 18_5_2, 18_5_3, 18_5_4 |
| **ROC/AUC** | 18_5_1, 18_5_2, 18_5_3, 18_5_4 |
| **GridSearchCV tuning** | 18_5_2 (RF), 18_5_3 (GB), 18_5_4 (all models) |
| **Feature importance** | 18_5_2 (correlated features problem), 18_5_4 (final model) |
| **Bias-variance tradeoff** | 18_5_1 (depth experiment), 18_5_2 (bagging vs. RF), 18_5_3 (boosting vs. bagging), 18_5_4 (final comparison) |
| **Cross-validation** | 18_5_1 (10-fold), 18_5_2 (10-fold), 18_5_3 (10-fold), 18_5_4 (nested 5×3) |
| **OOB error** | 18_5_2 (bagging and RF) |
| **Probability calibration** | 18_5_3 (RF vs. GB) |
| **Class imbalance handling** | 18_5_0 (introduction), 18_5_2 (class_weight demo), 18_5_4 (recall-based model selection) |
