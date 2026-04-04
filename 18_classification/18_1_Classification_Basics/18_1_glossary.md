# 18_1 Classification Basics — Glossary

This document defines all technical and conceptual terms used across the six notebooks in the 18_1 Classification Basics series.

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

### C Parameter
The inverse of regularization strength in logistic regression. A smaller C means stronger regularization (more penalty on coefficients); a larger C means weaker regularization. Controls the trade-off between fitting the training data well and keeping coefficients small.

### Class Imbalance
A situation where the classes in a dataset are not equally represented (e.g., 70% good credit, 30% bad credit). Imbalance can cause models to be biased toward the majority class and makes accuracy a poor evaluation metric.

### Class Weight (Balanced)
A parameter in scikit-learn classifiers that automatically adjusts the penalty for misclassifying each class inversely proportional to its frequency. Gives the minority class more weight during training, helping the model pay more attention to it.

### Classification Report
A scikit-learn function that prints precision, recall, f1-score, and support for each class, along with macro, weighted, and (optionally) micro averages. Provides a comprehensive view of per-class model performance.

### Confusion Matrix
A table that cross-tabulates actual class labels against predicted class labels. For binary classification, it has four cells: TP, TN, FP, FN. For K-class classification, it is a K×K grid. The diagonal contains correct predictions; off-diagonal cells show which classes the model confuses.

### Cross-Validation (k-Fold)
A technique for evaluating model performance by splitting the training data into k folds, training on k−1 folds and validating on the remaining fold, then repeating k times so each fold serves as the validation set once. The average across all k folds gives a more reliable performance estimate than a single train/test split.

---

## D

### Decision Boundary
The surface in feature space that separates regions predicted as different classes. For logistic regression, this is a straight line (or hyperplane). For decision trees, it is a set of axis-aligned rectangles. For KNN, it can be highly irregular and non-linear.

### Decision Threshold
The probability cutoff used to convert a soft prediction (probability) into a hard prediction (class label). By default, sklearn uses 0.5: if P(positive) ≥ 0.5, predict positive; otherwise predict negative. Moving the threshold changes the precision-recall trade-off.

### Drop First
A parameter in one-hot encoding (`drop_first=True`) that removes the first category column to avoid perfect multicollinearity (the dummy variable trap). With K categories, this produces K−1 binary columns instead of K.

---

## E

### ElasticNet
A regularization penalty that combines L1 (Lasso) and L2 (Ridge) penalties, controlled by the `l1_ratio` parameter. Can zero out some coefficients (like Lasso) while maintaining stability when features are correlated (like Ridge).

---

## F

### F1-Score
The harmonic mean of precision and recall: `F1 = 2 × (precision × recall) / (precision + recall)`. Provides a single number that balances both metrics. Punishes extreme imbalances — a model with perfect precision but near-zero recall gets a near-zero F1.

### False Negative (FN) / Type II Error
A case where the model predicts negative (Good credit) but the actual label is positive (Bad credit / Default). In the credit context: approving a loan to someone who will default.

### False Positive (FP) / Type I Error
A case where the model predicts positive (Bad credit / Default) but the actual label is negative (Good credit). In the credit context: wrongly denying a loan to a good customer.

### Feature Importance
A measure of how much each input feature contributes to a model's predictions. For tree-based models, it is computed as the total reduction in impurity (Gini or entropy) attributed to splits on that feature across all trees, normalized to sum to 1.0.

### Feature Scaling
Transforming features so they are on a comparable scale (typically mean=0, std=1). Required for logistic regression (gradient descent converges faster) and SVM (distance-based kernels). Not required for tree-based models.

### Feature Selection
The process of selecting a subset of input features that are most relevant to the prediction task. L1 regularization (Lasso) performs automatic feature selection by zeroing out coefficients for irrelevant features.

---

## H

### Harmonic Mean
A type of average that is more sensitive to small values than the arithmetic mean. For two numbers a and b: `2ab / (a + b)`. Used in the F1-score because it ensures that both precision and recall must be high for the F1 to be high.

### Hard Prediction
A discrete class label (0 or 1, Good or Bad) produced by applying the decision threshold to a probability. Contrasts with soft prediction (probability).

---

## K

### K-Nearest Neighbors (KNN)
A classification algorithm that predicts the class of a new point by looking at the K closest points in the training set and taking a majority vote. Makes local decisions based on nearby points rather than learning a global rule. No training phase — all computation happens at prediction time.

---

## L

### L1 Regularization (Lasso)
A penalty that adds the sum of absolute coefficient values to the loss function. Can set coefficients exactly to zero, effectively performing automatic feature selection.

### L2 Regularization (Ridge)
A penalty that adds the sum of squared coefficient values to the loss function. Shrinks all coefficients toward zero but rarely sets them exactly to zero. Keeps all features in the model.

### Log-Odds
The natural logarithm of the odds ratio. The unit in which logistic regression coefficients are expressed. A coefficient of β means that a one-unit increase in the feature changes the log-odds of the positive class by β.

### Logistic Regression
A linear classification model that uses the sigmoid function to map a linear combination of features into a probability between 0 and 1. Despite the name, it is a classifier, not a regressor.

---

## M

### Macro Average
An averaging strategy that computes the metric (precision, recall, or f1) separately for each class, then takes the simple (unweighted) average. Treats all classes equally regardless of how many samples each has. Useful when you care equally about performance on minority and majority classes.

### Maximum Likelihood Estimation (MLE)
The method used to train logistic regression. It finds the coefficients that make the observed labels in the training data the "most likely" outcome — maximizing the product of predicted probabilities for the actual classes.

### Micro Average
An averaging strategy that computes the metric globally across all samples, aggregating TP, FP, and FN across all classes before calculating precision/recall. For single-label classification, micro precision and micro recall are both equal to accuracy.

### Multiclass Classification
A classification problem with more than two classes (e.g., 3 wine cultivars). Requires strategies like One-vs-Rest or Softmax to extend binary classifiers.

---

## N

### Naive Baseline
See **Baseline (Naive)**.

---

## O

### Odds Ratio
The ratio of the probability of an event occurring to the probability of it not occurring. Computed as `e^β` where β is the logistic regression coefficient. An odds ratio of 1.65 means the odds of the positive class increase by 65% for each standard deviation increase in the feature.

### One-Hot Encoding
A technique for converting categorical variables into binary (0/1) columns. Each category becomes its own column, with 1 indicating presence and 0 indicating absence.

### One-vs-Rest (OvR)
A multiclass strategy that trains K separate binary classifiers, one for each class (Class i vs. Not-Class i). The final prediction is the class whose classifier outputs the highest probability.

---

## P

### PCA (Principal Component Analysis)
A dimensionality reduction technique that finds the directions (principal components) in the data that capture the most variance. PC1 captures the most variance; PC2 captures the most remaining variance orthogonal to PC1, and so on. Used to project high-dimensional data into 2D for visualization.

### Precision
The proportion of positive predictions that are actually correct: `TP / (TP + FP)`. Measures reliability — when the model says "positive," how often is it right?

### Precision-Recall Curve
A plot of precision vs. recall as the decision threshold varies from 0 to 1. More informative than the ROC curve for imbalanced datasets because it ignores true negatives. The baseline is the positive class prevalence.

---

## R

### Recall (Sensitivity)
The proportion of actual positives that the model correctly identifies: `TP / (TP + FN)`. Measures coverage — of all the people who actually defaulted, what percentage did we catch?

### Regularization
A technique that adds a penalty on coefficient size to the loss function, preventing overfitting. L1 (Lasso) and L2 (Ridge) are the two most common types.

### ROC Curve (Receiver Operating Characteristic)
A plot of the True Positive Rate (recall) against the False Positive Rate (1 − specificity) as the decision threshold varies from 0 to 1. The curve shows the trade-off between catching positives and generating false alarms. A perfect model hugs the top-left corner; a random model follows the diagonal.

---

## S

### Sigmoid Function (Logistic Function)
The S-shaped function `σ(z) = 1 / (1 + e^(-z))` that maps any real number to a value between 0 and 1. Used in logistic regression to convert a linear combination of features into a probability.

### Soft Prediction
A probability value between 0 and 1 produced by `.predict_proba()`. Represents the model's confidence that an instance belongs to the positive class. Contrasts with hard prediction (class label).

### Softmax (Multinomial)
A multiclass strategy that produces K output probabilities that sum to 1.0, trained as a single optimization problem. Unlike OvR, the classes compete against each other directly. This is the default multiclass strategy in modern scikit-learn.

### Specificity
The proportion of actual negatives that the model correctly identifies: `TN / (TN + FP)`. Equal to 1 − FPR. Measures how well the model avoids false alarms.

### StandardScaler
A scikit-learn transformer that standardizes features to have mean=0 and standard deviation=1.

### Stratified Split
A train-test split that preserves the class distribution in both the training and test sets. For a 70/30 dataset, both splits will have approximately 70% negative and 30% positive samples.

### Support
The number of actual instances of each class in the dataset (or test set). Reported in the classification report. Used to compute weighted averages.

### SVM (Support Vector Machine)
A classification algorithm that finds the hyperplane that maximally separates classes. With an RBF (radial basis function) kernel, it can capture complex non-linear boundaries.

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

## V

### Variance (Model)
In the context of cross-validation, the spread of scores across the k folds. High variance means the model's performance depends heavily on which specific samples end up in each fold — a sign of instability. Low variance means the model performs consistently regardless of the data split.

---

## W

### Weighted Average
An averaging strategy that computes the metric for each class, then takes a weighted average where each class is weighted by its support (number of samples). Reflects overall performance on the dataset population. When classes are balanced, weighted and macro averages are similar.

---

## Y

### Youden's J Statistic
A metric for finding the optimal decision threshold: `J = TPR − FPR`. Maximizes the distance between the ROC curve and the random chance line. Finds the threshold that gives the best balance between catching positives and avoiding false alarms, assuming both error types are equally costly.
