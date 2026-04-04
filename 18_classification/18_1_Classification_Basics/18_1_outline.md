# 18_1 Classification Basics — Topic Outline

This document provides a complete outline of all topics covered across the six notebooks in the 18_1 Classification Basics series.

---

## 18_1_0: Classification Foundations

**Dataset:** South German Credit (`credit-g`) — 1000 rows, 21 features, binary target (good/bad credit), ~70/30 class distribution.

### Why Not Linear Regression?
- Unbounded predictions vs. bounded probabilities
- The sigmoid (logistic) function: formula and properties
- Visualizing the S-curve: squashing any input into [0, 1]
- Bridging linear regression and probability

### Class Imbalance and the Accuracy Paradox
- Class distribution visualization
- The naive baseline: always predicting the majority class
- Why accuracy is deceptive on imbalanced data

### Data Preparation for Logistic Regression
- Binary encoding of the target variable
- One-hot encoding of categorical features (`drop_first=True`)
- Stratified train-test split (preserving class ratios)
- Feature scaling with `StandardScaler` (why gradient descent needs it)

### Training the Model
- Maximum Likelihood Estimation (MLE): making observed data "most likely"
- `class_weight='balanced'`: compensating for class imbalance

### Interpreting Model Coefficients
- Coefficients in log-odds units
- Converting to odds ratios: `exp(β)`
- Standardized features → directly comparable coefficients
- Top 5 positive and top 5 negative drivers of default risk

### Probabilities and the Decision Threshold
- Hard predictions (`.predict()`) vs. soft predictions (`.predict_proba()`)
- The default 0.5 threshold
- Overlapping probability distributions by actual class
- Quantifying false negative and false positive rates at the threshold

### Basic Model Evaluation
- Model accuracy vs. naive baseline

---

## 18_1_1: Confusion Matrix and Basic Metrics

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

### The F1-Score
- Harmonic mean of precision and recall
- Why harmonic mean (not arithmetic): punishing extreme imbalances
- Concrete example: precision=100%, recall=1% → F1≈2%

### The Classification Report
- Per-class precision, recall, and f1-score
- Support: number of samples per class
- Macro average vs. weighted average: when to use each

### The Threshold Trade-Off (Preview)
- Moving the threshold: conservative vs. sensitive
- Teaser for ROC curves and systematic threshold optimization

---

## 18_1_2: ROC, AUC, and Threshold Tuning

**Dataset:** South German Credit (`credit-g`)

### The ROC Curve
- True Positive Rate (TPR = recall) vs. False Positive Rate (FPR = 1 − specificity)
- Reading the curve: perfect model, random model, "knee" of the curve
- AUC interpretation: probability of correctly ranking a random positive above a random negative
- Concrete AUC example

### Precision-Recall Curves
- Why ROC can be over-optimistic on imbalanced data
- PR curve baseline: the positive class prevalence
- When to prefer PR curves over ROC curves

### Youden's J Statistic
- Formula: J = TPR − FPR
- Finding the mathematically optimal threshold
- Comparing Youden's J to the default 0.5 threshold

### Business Cost Sensitivity
- Assigning dollar costs to false positives and false negatives
- Computing total cost at each threshold
- Comparing three thresholds: default (0.5), Youden's J, cost-optimal
- The cost curve visualization

---

## 18_1_3: Model Selection via Cross-Validation

**Dataset:** South German Credit (`credit-g`)

### Model Competition
- Why cross-validation over a single train/test split
- Four competitors: Logistic Regression, Decision Tree, Random Forest, SVM (RBF)
- Scoring on both accuracy and F1
- Boxplot visualization of CV score distributions
- Interpreting variance: model stability across folds
- Which model won and why

### Regularization and Feature Selection
- L2 (Ridge): shrinks all coefficients, keeps all features
- L1 (Lasso): zeroes out coefficients, performs automatic feature selection
- ElasticNet: mix of L1 and L2
- Comparing coefficient plots across penalty types
- How many features each penalty zeroes out
- Feature selection sensitivity to penalty type

---

## 18_1_4: Multiclass Classification

**Dataset:** Wine (`load_wine`) — 178 rows, 13 chemical features, 3 cultivar classes.

### The Multiclass Problem
- Transitioning from binary to 3+ classes
- Dataset properties and class distribution

### Multiclass Strategies
- One-vs-Rest (OvR): K independent binary classifiers
- Softmax (Multinomial): K outputs that sum to 1.0
- Comparing OvR and Softmax accuracy
- Visualizing OvR probability distributions per classifier
- Reading the OvR histograms: overlap = confusion

### Multiclass Evaluation
- 3×3 confusion matrix: diagonal = correct, off-diagonal = confusion
- Classification report for 3 classes
- Macro vs. weighted vs. micro averaging
- When each averaging strategy matters (concrete 90/10 example)
- Identifying which classes are hardest to distinguish

---

## 18_1_5: Decision Boundaries and Feature Importance

**Dataset:** Wine (`load_wine`)

### PCA for Dimensionality Reduction
- Why we can't visualize 13 dimensions
- PCA: finding directions of maximum variance
- Explained variance ratio: how much structure is preserved in 2D
- Caveat: boundaries in PC space are approximations of true 13D boundaries

### Visualizing Decision Boundaries
- The meshgrid prediction technique
- Three models compared:
  - **Logistic Regression:** straight-line (linear) boundaries
  - **Decision Tree:** axis-aligned rectangular boundaries ("staircase")
  - **KNN:** wavy, local, non-linear boundaries
- KNN explained: classifying by nearest neighbors
- Training accuracy labels on each subplot
- Geometric model behaviors summary table

### Multiclass Confusion Matrix (Random Forest)
- Training on all 13 features (not PCA-reduced)
- Interpreting off-diagonal confusion: which cultivars share chemical properties?

### Feature Importance
- `feature_importances_` from Random Forest
- Percentage labels on each bar
- Top 3 features and their chemical interpretation
- Proline, color intensity, flavanoids: why they differentiate cultivars

### Full Series Recap
- Parts 1–5 summary
- Practical takeaway: model selection workflow
- Forward look to 18_2 (Logistic Regression deep dive) and 18_5 (Ensemble methods)

---

## Cross-Cutting Themes

These concepts appear throughout multiple notebooks:

| Theme | Notebooks |
|---|---|
| **Class imbalance** | 18_1_0, 18_1_1, 18_1_2, 18_1_3 |
| **Threshold tuning** | 18_1_0, 18_1_1, 18_1_2 |
| **Precision/Recall trade-off** | 18_1_1, 18_1_2, 18_1_4 |
| **Cross-validation** | 18_1_3 |
| **Regularization** | 18_1_0, 18_1_3 |
| **Confusion matrices** | 18_1_1, 18_1_4, 18_1_5 |
| **Macro vs. weighted averaging** | 18_1_1, 18_1_4 |
| **Feature importance** | 18_1_0 (coefficients), 18_1_5 (tree importances) |
