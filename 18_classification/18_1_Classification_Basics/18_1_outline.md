# 18_1 Classification Basics — Topic Outline

This document provides a complete outline of all topics covered across the six notebooks in the 18_1 Classification Basics series.

---

## 18_1_1: Classification Foundations

**Dataset:** South German Credit (`credit-g`) — 1000 rows, 21 features, binary target (good/bad credit), ~70/30 class distribution.

### Why Gradient Boosting?
- Unlike linear regression, classification requires bounded probabilities
- XGBoost uses gradient boosting: sequential trees that correct errors
- Each tree makes threshold-based splits on features
- Ensemble of trees produces probability predictions

### Class Imbalance and the Accuracy Paradox
- Class distribution visualization
- The naive baseline: always predicting the majority class
- Why accuracy is deceptive on imbalanced data
- The "accuracy paradox" explained with concrete example

### Data Preparation for XGBoost
- Binary encoding of the target variable (good=0, bad=1)
- One-hot encoding of categorical features (`drop_first=True`)
- Stratified train-test split (preserving class ratios)
- Note: Tree-based models do NOT require feature scaling

### Training the Model
- Gradient boosting: sequential tree correction
- `scale_pos_weight`: compensating for class imbalance (ratio of negatives/positives)
- Key parameters: n_estimators, max_depth, learning_rate

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
- Interpretation: what the 3% improvement actually means

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

## 18_1_3: ROC, AUC, and Threshold Tuning

**Dataset:** South German Credit (`credit-g`)

### The ROC Curve
- True Positive Rate (TPR = recall) vs. False Positive Rate (FPR = 1 − specificity)
- Reading the curve: perfect model, random model, "knee" of the curve
- AUC interpretation: probability of correctly ranking a random positive above a random negative
- Concrete AUC example with 0.79

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

## 18_1_4: Model Selection via Cross-Validation

**Dataset:** South German Credit (`credit-g`)

### Model Competition
- Why cross-validation over a single train/test split
- Four competitors: XGBoost, Decision Tree, Random Forest, SVM (RBF)
- Scoring on both accuracy and F1
- Boxplot visualization of CV score distributions
- Interpreting variance: model stability across folds
- Which model won and why

### Note on Regularization
- Regularization (L1/L2/ElasticNet) is NOT covered in this notebook
- For XGBoost regularization parameters, see 18_5 (Ensemble Methods)

---

## 18_1_5: Multiclass Classification

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

## 18_1_6: Decision Boundaries and Feature Importance

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

### Multiclass Confusion Matrix (XGBoost/Random Forest)
- Training on all 13 features (not PCA-reduced)
- Interpreting off-diagonal confusion: which cultivars share chemical properties?

### Feature Importance
- `feature_importances_` from tree-based models
- Percentage labels on each bar
- Top 3 features and their chemical interpretation
- Proline, color intensity, flavanoids: why they differentiate cultivars

### Full Series Recap
- Parts 1–6 summary
- Practical takeaway: model selection workflow
- Forward look to 18_2 (Logistic Regression deep dive) and 18_5 (Ensemble methods)

---

## Cross-Cutting Themes

These concepts appear throughout multiple notebooks:

| Theme | Notebooks |
|---|---|
| **Class imbalance** | 18_1_1, 18_1_2, 18_1_3, 18_1_4 |
| **Threshold tuning** | 18_1_1, 18_1_2, 18_1_3 |
| **Precision/Recall trade-off** | 18_1_2, 18_1_3, 18_1_5 |
| **Cross-validation** | 18_1_4 |
| **Confusion matrices** | 18_1_2, 18_1_5, 18_1_6 |
| **Macro vs. weighted averaging** | 18_1_2, 18_1_5 |
| **Feature importance** | 18_1_1 (XGBoost gain), 18_1_6 (tree importances) |