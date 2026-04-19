# 18_1 Classification Basics — Topic Outline

This document provides a complete outline of all topics covered across the notebooks in the 18_1 Classification Basics series.

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
- Feature Engineering Pipeline: one-hot encoding with `pd.get_dummies()`, column name cleaning with `re.sub()`, `drop_first=True` for multicollinearity avoidance
- Model.score() method usage for comprehensive evaluation

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

### Confusion Matrix Visualization and Business Cost
- Confusion matrix visualization using `plt.imshow()` and cell annotations
- Business cost framework: assigning costs to false positives and false negatives

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

## 18_1_4: Multiclass Classification

**Dataset:** Cardiotocography (`fetch_openml` data_id=1560) — 2126 rows, 35 features (V1-V35), 3 fetal health classes.

### Bridge from Binary to Multiclass
- Recap of Parts 1-3 binary toolkit (confusion matrix, precision/recall, ROC, thresholds)
- Transition to 3+ classes on a new clinical dataset
- Dataset properties and class distribution (Normal/Suspect/Pathological)
- Class imbalance: ~78% Normal, ~14% Suspect, ~8% Pathological
- Explicit naive baseline computation (always predict Normal = 77.8%)

### How Multiclass Classification Works
- Softmax: model outputs probability for each class, predicted class = highest probability
- `multi:softprob` objective in XGBoost
- One-vs-Rest (OvR) as an alternative strategy (conceptual mention)

### A Simple Model Falls Into the Trap
- Decision Tree (max_depth=2): limited capacity, biased toward majority class
- 93% accuracy but misses 31% of Pathological cases ("Fatal Misses")
- Comparison to naive baseline: accuracy sounds good but hides dangerous errors

### The 3×3 Confusion Matrix
- Diagonal = correct predictions
- Off-diagonal = misclassifications
- Clinical interpretation: False Alarms vs. Fatal Misses
- Reading specific rows to trace where each class's errors go

### XGBoost Comparison
- XGBoost (100 trees): 99% accuracy, catches nearly all minority class cases
- Side-by-side confusion matrix comparison
- Ensemble approach recovers minority class patterns that a shallow tree misses

### Per-Class Metrics in Multiclass
- Precision and recall computed separately for each class
- How to read per-class metrics: "Recall for Pathological = (correctly predicted Pathological) / (all actual Pathological)"
- Guided walkthrough of both classification reports

### Multiclass Averaging Strategies
- Per-class precision, recall, f1-score
- Support: number of samples per class
- Manual calculation of Macro F1 vs. Weighted F1
- Macro vs. weighted vs. micro averaging
- When each averaging strategy matters: macro treats all classes equally, weighted accounts for class imbalance, micro aggregates globally
- The gap between Weighted F1 and Macro F1 as the "signature of the imbalance trap"

### Addressing Imbalance with Sample Weights
- `compute_sample_weight('balanced', y)` from sklearn
- Weights inversely proportional to class frequency
- Retraining a Decision Tree with balanced weights: improved Macro F1
- Brief mention of other strategies: oversampling (SMOTE), undersampling, cost-sensitive learning
- Connection to `scale_pos_weight` from Part 1

### Practical Checklist
- Check class distribution first
- Compare accuracy to naive baseline
- Inspect confusion matrix for dangerous misclassifications
- Report Macro F1 alongside accuracy
- Consider sample weighting for minority classes

---

## Supporting Materials

| File | Description |
|------|-------------|
| `18_1_2_x_exercise.ipynb` | Practice notebook applying concepts from Notebooks 1 & 2 on Adult Census Income dataset |
| `18_1_4_x_Exercise_Two_Parts.ipynb` | Two-part exercise: binary (Thoracic Surgery) and multiclass (Contraceptive Method Choice) |
| `18_1_5_x_credit_card_fraud.ipynb` | Capstone exercise: credit card fraud detection with extreme class imbalance |
| `18_1_practice_quiz.md` | Practice quiz for assessment |
| `18_1_discussion_questions.md` | Discussion questions for the topic |
| `18_1_glossary.md` | Key terminology definitions |

---

## Cross-Cutting Themes

These concepts appear throughout multiple notebooks:

| Theme | Notebooks |
|-------|------------|
| **Class imbalance** | 18_1_1, 18_1_2, 18_1_3, 18_1_4 |
| **Threshold tuning** | 18_1_1, 18_1_2, 18_1_3 |
| **Precision/Recall trade-off** | 18_1_2, 18_1_3, 18_1_4 |
| **Confusion matrices** | 18_1_2, 18_1_4 |
| **Macro vs. weighted averaging** | 18_1_2, 18_1_4 |
| **Sample weighting / imbalance mitigation** | 18_1_1 (`scale_pos_weight`), 18_1_4 (`compute_sample_weight`) |
| **Feature importance** | 18_1_1 |
| **Feature Engineering Pipeline** | All notebooks |

### Feature Engineering Pipeline
A unified view of preprocessing steps across all notebooks:
1. Target encoding: `good=0, bad=1`
2. Categorical encoding: `pd.get_dummies()` with `drop_first=True`
3. Column name cleaning: `re.sub(r"[<>[\]]", "_", col)` for XGBoost compatibility
4. Train-test split: `train_test_split(..., stratify=y)`
5. Scale_pos_weight calculation: `(negatives/positives)` ratio |