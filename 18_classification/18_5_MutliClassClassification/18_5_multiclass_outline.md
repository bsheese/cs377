# 18_5 Multi-Class Classification — Topic Outline

This document outlines the topics covered across the three teaching notebooks and the exercise in the 18_5 Multi-Class Classification series.

**Data by notebook:**
- **18_5_1:** Palmer Penguins (three species)
- **18_5_2:** Palmer Penguins (three species)
- **18_5_3:** Fetal health / cardiotocography (Normal / Suspect / Pathological — heavily imbalanced)
- **18_5_9 (exercise):** Wine Quality (red), binned into a three-class target

**Model throughout:** XGBoost with `objective="multi:softprob"` — the same classifier family from 18_1, extended to K classes.

---

## 18_5_1: When Two Classes Are Not Enough

**Topics:** Extending binary classification to K classes: the problem setup, probability output, and the K×K confusion matrix.

### Getting to Know the Problem
- The scenario: identifying penguin species from measurements
- The three species; visual separability of the classes

### Preparing the Data and Training
- Integer-coded targets; one-hot encoding as the convention deep-learning frameworks use (exactly one 1 per row — one class per observation)
- `XGBClassifier(objective="multi:softprob", num_class=3)`

### Probabilities for Three Classes
- `predict_proba` returns an N×3 matrix; each row sums to 1 (softmax)
- The predicted label is the argmax of the row

### The 3×3 Confusion Matrix
- From 2×2 to 3×3: rows actual, columns predicted, diagonal correct
- Every off-diagonal cell is one *specific* confusion — which pairs the model mixes up

### A First Look at the Classification Report
- Per-class precision/recall/F1/support; preview of the averaging question

---

## 18_5_2: Measuring Performance Class by Class

**Topics:** Per-class metrics, F1, and the macro-vs-weighted averaging problem.

### The Limits of Accuracy
- One number for K classes hides which class is failing
- Precision and recall revisited (18_1 definitions, applied per class); a tiny worked example

### Per-Class Performance on Penguins
- Reading precision/recall per species; where the errors concentrate

### The F1 Score and the Averaging Problem
- F1 as the harmonic mean of precision and recall, per class
- Macro averaging: all classes equal
- Weighted averaging: frequent classes count more
- The macro/weighted gap as a diagnostic for rare-class failure

### Choosing the Right Metric
- Match the average to the question: does every class matter equally, or does volume matter?

---

## 18_5_3: When One Class Is Rare

**Topics:** Imbalanced multiclass classification on real medical data; sample weighting; honest cross-validation.

### The Problem: Fetal Health Monitoring
- Cardiotocography data; Pathological is rare and is the class you cannot afford to miss

### Measuring the Problem, and the Imbalance Trap in Action
- Class proportions; a weak model posts high accuracy while missing Pathological cases
- Macro F1 vs. weighted F1: why the gap matters here

### Remedies
- XGBoost's flexibility; `compute_sample_weight(class_weight="balanced")` — rare classes get equal total voice in training, no rows added or removed

### Honest Evaluation
- `StratifiedKFold` so every fold preserves the rare class's proportion
- Which metric to use: macro F1 plus the rare class's own recall

---

## 18_5_9: Exercise — Wine Quality Prediction

**Topics:** The full multiclass pipeline, end to end, on a new dataset.

- Load and explore; bin quality scores into a three-class target
- Visualize class separation
- Baseline XGBoost model; confusion matrix
- Handle imbalance with balanced sample weights
- Honest evaluation with stratified cross-validation
- Reflection questions
