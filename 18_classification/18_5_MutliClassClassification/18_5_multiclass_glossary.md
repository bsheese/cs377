# 18_5 Multi-Class Classification — Glossary

This document defines the technical and conceptual terms used across the three notebooks in the 18_5 Multi-Class Classification series.

---

## C

### Classification Report (Multiclass)
Sklearn's per-class table of precision, recall, F1, and support, plus the macro and weighted averages at the bottom. In the multiclass setting this is the primary evaluation tool: overall accuracy compresses K classes into one number, while the report shows exactly which class the model is failing.

### Confusion Matrix (K×K)
The 2×2 confusion matrix from 18_1 generalized to K classes: rows are actual classes, columns are predicted classes. Correct predictions sit on the diagonal; each off-diagonal cell counts one specific confusion (row Adelie, column Gentoo = actual Adelies predicted as Gentoo). With K classes there are $K^2 - K$ distinct ways to be wrong, and the matrix shows every one.

---

## I

### Imbalance Trap
The multiclass version of 18_1's accuracy paradox: when one class dominates (most fetal-health recordings are Normal), a model can post high accuracy while nearly ignoring the rare classes — the ones that usually matter most. Diagnosed by the gap between weighted and macro F1, and by the rare class's own recall.

---

## M

### Macro Average
The unweighted mean of a per-class metric: compute F1 for each class, then average, treating a 40-row class exactly like a 1,600-row class. Use it when every class matters equally regardless of frequency — the standard scoring choice for imbalanced multiclass problems in this series.

### Multiclass Classification
Classification with more than two possible labels (three penguin species; Normal/Suspect/Pathological fetal health). Everything from binary classification carries over — probabilities, confusion matrix, precision/recall — but each piece grows from 2 to K, and evaluation must now be read class by class.

### `multi:softprob`
The XGBoost objective used throughout the series: the model outputs a full probability distribution over all K classes for every observation (via softmax), rather than just a label. `predict_proba` then returns an N×K matrix whose rows sum to 1.

---

## O

### One-Hot Encoded Target
Representing a K-class label as a length-K vector with a single 1 (Adelie = [1, 0, 0]) — exactly one because each observation belongs to exactly one class. Deep-learning frameworks often expect targets in this form; sklearn and XGBoost handle integer labels directly, but the convention is worth recognizing.

---

## P

### Per-Class Precision / Recall
Precision and recall computed treating one class as "positive" and everything else as "negative." Precision for Chinstrap: of everything predicted Chinstrap, what fraction really is. Recall for Chinstrap: of the actual Chinstraps, what fraction the model found. The 18_1 definitions, applied K times.

---

## S

### Sample Weights (`compute_sample_weight(class_weight="balanced")`)
Per-observation training weights inversely proportional to class frequency, so each class carries equal total influence on the loss. Passed to `fit(..., sample_weight=...)`, this makes XGBoost stop treating rare-class errors as cheap — the main imbalance remedy used in 18_5_3. Contrast with resampling: no rows are added or removed.

### Softmax
The function that converts a model's K raw scores into K probabilities that are each in (0, 1) and sum to 1 — the multiclass generalization of the sigmoid. It is what puts the "softprob" in `multi:softprob`.

### Stratified K-Fold
Cross-validation that preserves each class's proportion inside every fold. Essential for imbalanced multiclass data: ordinary K-fold can produce folds with almost no rare-class rows, making fold scores meaningless for exactly the class you care about.

### Support
The number of true instances of each class in the evaluation set — the rightmost column of the classification report. Low-support classes have noisier metrics and are where the macro/weighted gap comes from.

---

## W

### Weighted Average
The per-class metric averaged with each class weighted by its support. It tracks overall experience (most rows are the common class) but can look healthy while a rare class fails. Reading macro and weighted *together* — and especially the gap between them — is the series' core diagnostic.
