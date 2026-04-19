# 18_1 Classification Basics — Glossary

This document defines all technical and conceptual terms used across the four notebooks in the 18_1 Classification Basics series.

---

## A

### Accuracy
The proportion of correct predictions (both true positives and true negatives) out of all predictions. Can be misleading on imbalanced datasets — a model that always predicts the majority class can achieve high accuracy while being useless.

### Accuracy Paradox
The phenomenon where a model with high accuracy is actually performing poorly on the task of interest. This occurs on imbalanced datasets where a naive model that always predicts the majority class achieves high accuracy but fails to detect the minority class.

### AUC (Area Under the Curve)
A single-number summary of a model's ranking ability, computed as the area under the ROC curve. Represents the probability that the model will assign a higher predicted probability to a randomly chosen positive instance than to a randomly chosen negative instance. Ranges from 0.5 (random) to 1.0 (perfect).

---

## B

### Baseline (Naive)
The performance of the simplest possible model — typically one that always predicts the majority class. Any useful model must significantly exceed the baseline. For a 70/30 split, the naive baseline accuracy is 70%.

### Business Cost Sensitivity
The practice of assigning real-world dollar costs to different types of classification errors (false positives vs. false negatives) and choosing the decision threshold that minimizes total expected cost, rather than using a mathematically optimal but cost-blind threshold.

---

## C

### Class Imbalance
A situation where the classes in a dataset are not equally represented (e.g., 70% good credit, 30% bad credit). Imbalance can cause models to be biased toward the majority class and makes accuracy a poor evaluation metric.

### Classification Report
A scikit-learn function that prints precision, recall, f1-score, and support for each class, along with macro, weighted, and (optionally) micro averages. Provides a comprehensive view of per-class model performance.

### Confusion Matrix
A table that cross-tabulates actual class labels against predicted class labels. For binary classification, it has four cells: TP, TN, FP, FN. For K-class classification, it is a K×K grid. The diagonal contains correct predictions; off-diagonal cells show which classes the model confuses.

---

## D

### Decision Threshold
The probability cutoff used to convert a soft prediction (probability) into a hard prediction (class label). By default, sklearn uses 0.5: if P(positive) ≥ 0.5, predict positive; otherwise predict negative. Moving the threshold changes the precision-recall trade-off.

### Drop First
A parameter in one-hot encoding (`drop_first=True`) that removes the first category column to avoid perfect multicollinearity (the dummy variable trap). With K categories, this produces K−1 binary columns instead of K.

---

## E

### Ensemble
A machine learning technique that combines multiple models to produce better predictions than any single model. Random Forest and XGBoost are ensemble methods that aggregate predictions from many decision trees.

---

## F

### F1-Score
The harmonic mean of precision and recall: `F1 = 2 × (precision × recall) / (precision + recall)`. Provides a single number that balances both metrics. Punishes extreme imbalances — a model with perfect precision but near-zero recall gets a near-zero F1.

### False Negative (FN) / Type II Error
A case where the model predicts negative (Good credit) but the actual label is positive (Bad credit / Default). In the credit context: approving a loan to someone who will default.

### False Positive (FP) / Type I Error
A case where the model predicts positive (Bad credit / Default) but the actual label is negative (Good credit). In the credit context: wrongly denying a loan to a good customer.

### Feature Importance
A measure of how much each input feature contributes to a model's predictions. For tree-based models like XGBoost and Random Forest, it is computed based on gain (loss reduction), cover (samples affected), or frequency (times used). Normalized to sum to 1.0.

### Feature Scaling
Transforming features so they are on a comparable scale (typically mean=0, std=1). Required for logistic regression (gradient descent converges faster) and SVM (distance-based kernels). Not required for tree-based models like XGBoost.

---

## G

### Gain (Feature Importance)
In XGBoost, a measure of feature importance based on how much a feature contributes to reducing the loss function (improving predictions) when making splits. Higher gain means the feature has a larger impact on the model's performance.

### Gradient Boosting
A machine learning technique that builds models sequentially, where each new model corrects the errors of the previous ones. XGBoost is a popular gradient boosting implementation that uses decision trees as base learners.

---

## H

### Hard Prediction
A discrete class label (0 or 1, Good or Bad) produced by applying the decision threshold to a probability. Contrasts with soft prediction (probability).

---

## L

### Learning Rate
In gradient boosting (XGBoost), a hyperparameter that controls how much each tree contributes to the final prediction. A lower learning rate requires more trees but typically produces better generalization.

---

## M

### Macro Average
An averaging strategy that computes the metric (precision, recall, or f1) separately for each class, then takes the simple (unweighted) average. Treats all classes as equally important, regardless of how many samples each has. Useful when you care equally about performance on minority and majority classes.

### Micro Average
An averaging strategy that computes the metric globally across all samples, aggregating TP, FP, and FN across all classes before calculating precision/recall. For single-label classification, micro precision and micro recall are both equal to accuracy.

### Multiclass Classification
A classification problem with more than two classes (e.g., Normal/Suspect/Pathological fetal health). Requires strategies like One-vs-Rest or Softmax to extend binary classifiers.

---

## n_estimators
In ensemble models (XGBoost, Random Forest), the number of trees in the ensemble. More trees typically improve performance but increase computation time.

### Naive Baseline
See **Baseline (Naive)**.

---

## O

### One-Hot Encoding
A technique for converting categorical variables into binary (0/1) columns. Each category becomes its own column, with 1 indicating presence and 0 indicating absence.

### One-vs-Rest (OvR)
A strategy for extending binary classifiers to multiclass problems. For K classes, K separate binary classifiers are trained (e.g., Normal vs. Not-Normal, Suspect vs. Not-Suspect, etc.), and the class with the highest confidence wins. An alternative to the softmax approach.

---

## P

### Precision
The proportion of positive predictions that are actually correct: `TP / (TP + FP)`. Measures reliability — when the model says "positive," how often is it right?

### Precision-Recall Curve
A plot of precision vs. recall as the decision threshold varies from 0 to 1. More informative than the ROC curve for imbalanced datasets because it ignores true negatives. The baseline is the positive class prevalence.

---

## R

### Recall (Sensitivity)
The proportion of actual positives that the model correctly identifies: `TP / (TP + FN)`. Measures coverage — of all the people who actually defaulted, what percentage did we catch?

### ROC Curve (Receiver Operating Characteristic)
A plot of the True Positive Rate (recall) against the False Positive Rate (1 − specificity) as the decision threshold varies from 0 to 1. The curve shows the trade-off between catching positives and generating false alarms. A perfect model hugs the top-left corner; a random model follows the diagonal.

---

## S

### Sample Weight
A per-sample weight assigned during training to control how much each sample contributes to the model's learning. Used to handle class imbalance by giving higher weights to minority class samples. In scikit-learn, `compute_sample_weight('balanced', y)` automatically calculates weights inversely proportional to class frequency. This is the general-purpose version of `scale_pos_weight` that works with any model and any number of classes.

### scale_pos_weight
A parameter in XGBoost used to handle class imbalance in binary classification. It scales the weight of the positive class (minority) relative to the negative class. Calculated as: (number of negatives) / (number of positives). For multiclass imbalance, use `sample_weight` instead.

### Soft Prediction
A probability value between 0 and 1 produced by `.predict_proba()`. Represents the model's confidence that an instance belongs to the positive class. Contrasts with hard prediction (class label).

### Softmax
A function that converts a vector of raw model outputs into a valid probability distribution (all values between 0 and 1, summing to 1.0). Used by XGBoost (`objective='multi:softprob'`) and neural networks for multiclass classification. Each class gets a probability, and the predicted class is whichever has the highest value.

### Specificity
The proportion of actual negatives that the model correctly identifies: `TN / (TN + FP)`. Equal to 1 − FPR. Measures how well the model avoids false alarms.

### Stratified Split
A train-test split that preserves the class distribution in both the training and test sets. For a 70/30 dataset, both splits will have approximately 70% negative and 30% positive samples.

### Support
The number of actual instances of each class in the dataset (or test set). Reported in the classification report. Used to compute weighted averages.

---

## T

### Threshold
See **Decision Threshold**.

### True Negative (TN)
A case where the model correctly predicts negative (Good credit).

### True Positive (TP)
A case where the model correctly predicts positive (Bad credit / Default).

### Type I Error
See **False Positive (FP)**.

### Type II Error
See **False Negative (FN)**.

---

## W

### Weighted Average
An averaging strategy that computes the metric for each class, then takes a weighted average where each class is weighted by its support (number of samples). Reflects overall performance on the dataset population. When classes are balanced, weighted and macro averages are similar.

---

## X

### XGBoost
eXtreme Gradient Boosting — a powerful tree-based ensemble algorithm. Instead of fitting a single model, it builds an ensemble of decision trees sequentially, where each new tree corrects the errors of the previous ones. Uses scale_pos_weight to handle class imbalance.

---

## Y

### Youden's J Statistic
A metric for finding the optimal decision threshold: `J = TPR − FPR`. Maximizes the distance between the ROC curve and the random chance line. Finds the threshold that gives the best balance between catching positives and avoiding false alarms, assuming both error types are equally costly.