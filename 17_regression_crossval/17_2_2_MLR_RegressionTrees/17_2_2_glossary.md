# 17_2_2 Regression Trees — Glossary

Terms used in 17_2_2 (regression trees, random forests, gradient boosting, XGBoost). Linear-model terms (cross-validation, pipelines, regularization, nested CV) are in the 17_2_1 glossary in `17_2_1_MLR/`.

---

## B

### Bagging (Bootstrap Aggregating)
An ensemble technique where multiple models (typically decision trees) are trained independently on random bootstrap samples of the training data, and their predictions are averaged. Used in Random Forests to reduce variance.

### Bootstrap Sampling
Random sampling with replacement from the training data. Each bootstrap sample is the same size as the original dataset but contains some duplicate observations and omits others. Used in Random Forests and for estimating uncertainty.
---

## D

### Decision Tree
A model that makes predictions by asking a series of yes/no questions about the features, splitting the data at each node until reaching a leaf node. The prediction at each leaf is the average target value of training samples in that leaf. Prone to overfitting if not constrained by `max_depth`.
---

## E

### Ensemble Learning
A technique that combines multiple models to produce a single, more accurate prediction. Examples include Random Forests (bagging: averaging many trees) and Gradient Boosting (boosting: sequentially correcting errors).
---

## F

### Feature Importance
For tree-based models, the total reduction in error (impurity or gain) attributed to splits on each feature across all trees. It says *which* features the model relies on, not the direction or shape of their effect. When two features are highly correlated, the model may split on one and give the other little credit.
---

## G

### Gradient Boosting
An ensemble technique that builds trees sequentially, where each new tree predicts the residuals (errors) of the ensemble so far. Reduces bias by iteratively correcting mistakes. Variants include HistGradientBoostingRegressor and XGBoost.
---

## H

### HistGradientBoostingRegressor
Scikit-learn's fast implementation of gradient boosting that uses histogram binning to speed up split-finding. Handles categorical features and missing values natively.
---

## L

### Leaf Node
An endpoint in a decision tree where no further splits occur. The prediction for any sample reaching a leaf is the average target value of the training samples in that leaf.
---

## M

### Max Depth
A hyperparameter that limits the maximum depth (number of levels) of a decision tree. Prevents overfitting by stopping the tree from creating too many splits.

### Max Features
In Random Forests, the number of features each tree considers at each split. `max_features='sqrt'` means each tree considers approximately the square root of the total number of features. Increases tree diversity.
---

## N

### N Estimators
The number of trees in an ensemble model (Random Forest, Gradient Boosting, XGBoost). More trees generally improve performance up to a point, after which returns diminish.

### Node (Decision Node)
A point in a decision tree where a question is asked about a feature, splitting the data into two branches.
---

## R

### Random Forest
An ensemble of decision trees trained on bootstrap samples with random feature subsets at each split. Predictions are averaged across all trees. Reduces variance compared to a single tree. Does not require feature scaling.

### Root Node
The first and most important split in a decision tree. Determines the single most informative feature for separating the data.
---

## X

### XGBoost (eXtreme Gradient Boosting)
An optimized implementation of gradient boosting with built-in regularization (L1 and L2 on leaf weights), parallelized feature sorting, and native missing data handling. Often the most accurate model for structured tabular data, but less interpretable than linear models.
