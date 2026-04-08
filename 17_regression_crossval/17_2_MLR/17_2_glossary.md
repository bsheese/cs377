# 17_2 Regression Cross-Validation — Glossary

This document defines all technical and conceptual terms used across the five notebooks in the 17_2 Regression Cross-Validation series.

---

## A

### Alpha (α)
The regularization strength parameter in Ridge, Lasso, and ElasticNet regression. A higher alpha means stronger penalty on coefficient size, leading to more shrinkage. Alpha = 0 is equivalent to ordinary least squares (no regularization).

### AIC (Akaike Information Criterion)
A metric for model selection that balances goodness of fit against model complexity. Lower AIC indicates a better model. Penalizes the number of parameters to discourage overfitting.

### Accuracy
The proportion of correct predictions out of all predictions. In regression, this is typically measured by R² or MSE rather than classification accuracy.

### Ames Housing Dataset
A dataset of residential property sales in Ames, Iowa, containing 2,930 observations and 82 features. Used throughout this series as the primary dataset for demonstrating regression techniques.

---

## B

### Backward Selection
A feature selection algorithm that starts with all features and iteratively removes the least useful one at each step, based on cross-validation performance. Stops when removing any further feature would significantly hurt performance.

### Bagging (Bootstrap Aggregating)
An ensemble technique where multiple models (typically decision trees) are trained independently on random bootstrap samples of the training data, and their predictions are averaged. Used in Random Forests to reduce variance.

### Baseline Model
The simplest reasonable model used as a reference point for comparison. In regression, this is often a model that always predicts the mean of the target variable.

### Bias
Error introduced by overly simplistic assumptions in a model. High bias causes underfitting — the model fails to capture the underlying patterns in the data.

### Bias-Variance Tradeoff
The fundamental tension in model building: reducing bias (by making the model more complex) typically increases variance (sensitivity to training data noise), and vice versa. The goal is to find the balance that minimizes total error on unseen data.

### Bootstrap Sampling
Random sampling with replacement from the training data. Each bootstrap sample is the same size as the original dataset but contains some duplicate observations and omits others. Used in Random Forests and for estimating uncertainty.

### BIC (Bayesian Information Criterion)
Similar to AIC but with a stronger penalty for model complexity. BIC tends to select simpler models than AIC, especially with larger datasets.

---

## C

### Coefficient (β)
In linear regression, the weight assigned to each feature. A positive coefficient means the feature increases the predicted target; a negative coefficient means it decreases the prediction. In regularized regression, coefficients are shrunk toward zero.

### Correlation
A measure of the linear relationship between two variables, ranging from -1 (perfect negative) to +1 (perfect positive). High correlation between features (multicollinearity) can destabilize regression coefficients.

### Cross-Validation (k-Fold)
A technique for evaluating model performance by splitting the training data into k folds, training on k−1 folds and validating on the remaining fold, then repeating k times. The average across all k folds gives a more reliable performance estimate than a single train/test split.

---

## D

### Data Leakage
When information from the validation or test set inadvertently influences the training process, leading to overly optimistic performance estimates. Common causes include scaling before splitting or using future information.

### Decision Tree
A model that makes predictions by asking a series of yes/no questions about the features, splitting the data at each node until reaching a leaf node. Prone to overfitting if not constrained.

### Decision Threshold
In classification, the probability cutoff used to convert a soft prediction into a hard class label. In regression, analogous decisions are made about which features to include or which alpha value to use.

### Dummy Variable Trap
The problem of perfect multicollinearity that occurs when one-hot encoding creates K binary columns for a K-category feature. One column is perfectly predictable from the others. Solved by dropping one category (`drop_first=True`).

---

## E

### ElasticNet
A regularization method that combines L1 (Lasso) and L2 (Ridge) penalties, controlled by the `l1_ratio` parameter. Can perform feature selection (like Lasso) while maintaining stability with correlated features (like Ridge).

### Ensemble Learning
A technique that combines multiple models to produce a single, more accurate prediction. Examples include Random Forests (averaging many trees) and Gradient Boosting (sequentially correcting errors).

---

## F

### Feature Importance
A measure of how much each input feature contributes to a model's predictions. For tree-based models, it is computed as the total reduction in impurity attributed to splits on that feature across all trees.

### Feature Scaling
Transforming features so they are on a comparable scale (typically mean=0, std=1 using StandardScaler). Required for regularization and gradient-based optimization. Not required for tree-based models.

### Feature Selection
The process of identifying and retaining only the most useful input features. Methods include forward selection, backward selection, and Lasso regularization.

### Forward Selection
A feature selection algorithm that starts with no features and iteratively adds the most useful one at each step, based on cross-validation performance. Stops when adding any further feature no longer improves performance.

### F-Test
A statistical test used in stepwise selection to determine whether adding or removing a feature significantly improves the model.

---

## G

### Gradient Boosting
An ensemble technique that builds trees sequentially, where each new tree predicts the residuals (errors) of the ensemble so far. Reduces bias by iteratively correcting mistakes.

### GridSearchCV
A scikit-learn tool that systematically tests all combinations of hyperparameters from a defined grid, using cross-validation to evaluate each combination. Returns the best-performing hyperparameter set.

---

## H

### Heteroscedasticity
A condition where the variance of the errors is not constant across the range of predicted values. Violates an assumption of linear regression and can make confidence intervals unreliable.

### HistGradientBoostingRegressor
Scikit-learn's fast implementation of gradient boosting that uses histogram binning to speed up split-finding. Handles categorical features and missing values natively.

### Hyperparameter
A configuration setting that is chosen before training begins, as opposed to parameters (coefficients) that are learned during training. Examples: alpha, max_depth, n_estimators, learning_rate.

---

## I

### Imputation
The process of filling in missing values. Common methods include median imputation (for numeric features) and mode imputation (for categorical features).

### Inner Loop (Nested CV)
In nested cross-validation, the inner loop is the cross-validation performed within each outer training fold to tune hyperparameters. It finds the best hyperparameters for that specific training subset.

### Intercept (β₀)
The baseline prediction of a regression model when all features are zero. In regularized regression, the intercept is typically not penalized.

---

## K

### k-Fold Cross-Validation
See **Cross-Validation (k-Fold)**.

### KNN (K-Nearest Neighbors)
A classification/regression algorithm that predicts based on the K closest training examples. Makes local decisions rather than learning a global rule.

---

## L

### L1 Regularization (Lasso)
A penalty that adds the sum of absolute coefficient values to the loss function. Can set coefficients exactly to zero, performing automatic feature selection.

### L2 Regularization (Ridge)
A penalty that adds the sum of squared coefficient values to the loss function. Shrinks all coefficients toward zero but rarely sets them exactly to zero.

### l1_ratio
In ElasticNet, the mixing parameter between L1 and L2 penalties. l1_ratio = 1 is pure Lasso; l1_ratio = 0 is pure Ridge; l1_ratio = 0.5 is an equal mix.

### Learning Curve
A plot showing how model performance (training and validation scores) changes as the training set size increases. Used to diagnose whether a model would benefit from more data, more complexity, or less complexity.

### Leaf Node
An endpoint in a decision tree where no further splits occur. The prediction for any sample reaching a leaf is the average target value of the training samples in that leaf.

### Log Transformation
Applying the natural logarithm to a variable, typically the target. Used to reduce right-skew, stabilize variance, and convert multiplicative relationships into additive ones.

### Log-Odds
The natural logarithm of the odds ratio. The unit in which logistic regression coefficients are expressed. A coefficient of β means that a one-unit increase in the feature changes the log-odds of the positive class by β.

---

## M

### Max Depth
A hyperparameter that limits the maximum depth (number of levels) of a decision tree. Prevents overfitting by stopping the tree from creating too many splits.

### Max Features
In Random Forests, the number of features each tree considers at each split. `max_features='sqrt'` means each tree considers approximately the square root of the total number of features.

### Mean Squared Error (MSE)
The average of the squared differences between predicted and actual values. Sensitive to outliers because errors are squared.

### Median Imputation
Filling missing numeric values with the median of the observed values. Preferred over mean imputation when the distribution is skewed, as the median is robust to outliers.

### Multicollinearity
A condition where two or more independent variables are highly correlated with each other. Makes individual regression coefficients unstable and hard to interpret, even if overall predictions remain accurate.

---

## N

### N Estimators
The number of trees in an ensemble model (Random Forest, Gradient Boosting, XGBoost). More trees generally improve performance up to a point, after which returns diminish.

### Naive Baseline
The simplest possible model — typically one that always predicts the mean of the target variable. Any useful model must significantly exceed the baseline.

### Nested Cross-Validation
A two-layer cross-validation approach: the inner loop tunes hyperparameters within each outer training fold, and the outer loop evaluates the tuned model on an independent holdout fold. Provides an unbiased estimate of model performance.

### Node (Decision Node)
A point in a decision tree where a question is asked about a feature, splitting the data into two branches.

---

## O

### One-Hot Encoding
A technique for converting categorical variables into binary (0/1) columns. Each category becomes its own column, with 1 indicating presence and 0 indicating absence.

### Optimistic Bias
The upward bias in a performance estimate that occurs when hyperparameters are selected based on their score on the same data used for evaluation. The selected hyperparameters are partly tuned to noise.

### Outer Loop (Nested CV)
In nested cross-validation, the outer loop splits the data into folds for evaluation. Each outer fold uses a model tuned by the inner loop, providing an unbiased performance estimate.

### Overfitting
When a model learns the noise and specificities of the training data rather than the underlying patterns. Characterized by high training performance but poor performance on unseen data.

### Outlier
An observation that is significantly different from the majority of the data. Can disproportionately influence regression models, especially those using squared error.

---

## P

### Partial Dependence Plot
A visualization that shows the marginal effect of a feature on the predicted outcome, averaging out the effects of all other features. Used to understand how a feature affects predictions in complex models.

### Pipeline
A scikit-learn tool that chains together multiple processing steps (e.g., scaling, feature selection, modeling) into a single object. Ensures that each step is applied correctly within cross-validation, preventing data leakage.

### Polynomial Features
Features created by raising existing features to a power (e.g., X², X³) or multiplying features together (interaction terms). Allow linear models to capture non-linear relationships.

### Precision-Recall Curve
A plot of precision vs. recall as the decision threshold varies. More informative than ROC for imbalanced classification problems.

### Principal Component Analysis (PCA)
A dimensionality reduction technique that finds the directions in the data that capture the most variance. Used to project high-dimensional data into 2D for visualization.

---

## R

### R² (R-squared)
The proportion of variance in the target variable explained by the model. Ranges from negative infinity to 1, where 1 is a perfect fit and 0 means the model is no better than predicting the mean.

### Random Forest
An ensemble of decision trees trained on bootstrap samples with random feature subsets at each split. Predictions are averaged across all trees. Reduces variance compared to a single tree.

### Recall
In classification, the proportion of actual positives that the model correctly identifies. In regression contexts, analogous to how well the model captures the full range of the target.

### Refit
In GridSearchCV, the `refit=True` parameter means that after finding the best hyperparameters, the model is retrained on the entire training set so it is ready for predictions.

### Regularization
A technique that adds a penalty on coefficient size to the loss function, preventing overfitting. L1 (Lasso) and L2 (Ridge) are the two most common types.

### Residual
The difference between the actual and predicted values for a single observation. In gradient boosting, each new tree is trained to predict the residuals of the ensemble so far.

### Residual Plot
A scatter plot of residuals against predicted values. Used to diagnose model problems: a funnel shape indicates heteroscedasticity; a pattern indicates non-linearity; random scatter indicates a well-fitted model.

### Ridge Regression
See **L2 Regularization (Ridge)**.

### RMSE (Root Mean Squared Error)
The square root of MSE. In the same units as the target variable, making it more interpretable than MSE.

### Root Node
The first and most important split in a decision tree. Determines the single most informative feature for separating the data.

---

## S

### Safe Drop Pattern
A coding practice where columns are dropped only if they exist in the DataFrame, preventing errors when the cleaning pipeline is run on different versions of the dataset.

### Sigmoid Function
The S-shaped function σ(z) = 1 / (1 + e^(-z)) that maps any real number to a value between 0 and 1. Used in logistic regression to convert a linear combination of features into a probability.

### Softmax
A multiclass strategy that produces K output probabilities that sum to 1.0, trained as a single optimization problem. The classes compete against each other directly.

### StandardScaler
A scikit-learn transformer that standardizes features to have mean=0 and standard deviation=1.

### Stratified Split
A train-test split that preserves the class distribution in both the training and test sets. Important for imbalanced datasets.

### Stepwise Selection
A family of feature selection methods (forward, backward, or bidirectional) that iteratively add or remove features based on statistical criteria. Described as "binary and harsh" because features are either fully included or fully excluded.

### Structural Multicollinearity
Multicollinearity created by the way features are constructed, such as including both X and X² in a model without centering X first. Can be resolved by standardizing features before creating polynomial terms.

### Support Vector Machine (SVM)
A classification/regression algorithm that finds the hyperplane that maximally separates classes. With an RBF kernel, it can capture complex non-linear boundaries.

---

## T

### Train-Test Split
Dividing the dataset into a training set (typically 80%) used to fit the model and a test set (typically 20%) used to evaluate generalization performance.

### True Positive Rate (TPR)
See **Recall**.

### Type I Error
In classification, a false positive — predicting the positive class when the actual label is negative.

### Type II Error
In classification, a false negative — predicting the negative class when the actual label is positive.

---

## V

### Variance
How much a model's predictions vary when trained on different subsets of the data. High variance leads to overfitting — the model is too sensitive to the specific training samples.

### Variance Inflation Factor (VIF)
A measure of how much a coefficient's variance is inflated due to multicollinearity with other features. VIF = 1 means no correlation; VIF > 5 indicates moderate concern; VIF > 10 indicates severe multicollinearity.

---

## X

### XGBoost (eXtreme Gradient Boosting)
An optimized implementation of gradient boosting with built-in regularization (L1 and L2 on leaf weights), parallelized feature sorting, and native missing data handling. Often the most accurate model for structured tabular data.

---

## Y

### Yolked Variables
Paired features where a categorical feature being "None" always corresponds to a numeric feature being 0. Creates redundancy and can confuse the model. Resolved by dropping or combining the paired features.

### Youden's J Statistic
A metric for finding the optimal decision threshold: J = TPR − FPR. Maximizes the distance between the ROC curve and the random chance line.
