# 17_2 MLR and Regularization — Glossary

This document defines the technical and conceptual terms used in the 17_2_1 Ames series (17_2_1_1 to 17_2_1_5). Tree terms are in the 17_2_2 glossary in `17_2_2_MLR_RegressionTrees/`.

---

## A

### Alpha (α)
The regularization strength parameter in Ridge, Lasso, and ElasticNet regression. A higher alpha means stronger penalty on coefficient size, leading to more shrinkage. Alpha = 0 is equivalent to ordinary least squares (no regularization). Found via grid search in Part 4.

### Ames Housing Dataset
A dataset of residential property sales in Ames, Iowa, containing 2,930 observations and 82 features. Used throughout this series as the primary dataset for demonstrating data cleaning, feature selection, regularization, and tree-based methods.

---

## B

### Backward Selection
A feature selection algorithm that starts with all features and iteratively removes the least useful one at each step, based on cross-validation performance. Stops when removing any further feature would significantly hurt performance. More computationally expensive than forward selection but can catch interactions.

### Baseline Model
The simplest reasonable model used as a reference point for comparison. In regression, this is often Ordinary Least Squares (OLS) without regularization.

### Beta Weights (Standardized Coefficients)
Coefficients transformed to a common scale by standardizing all features to have mean 0 and standard deviation 1 before fitting. Allow direct comparison of feature importance regardless of original units. In regularized regression, beta weights show which features have the largest practical impact.

### Bias
Error introduced by overly simplistic assumptions in a model. High bias causes underfitting — the model fails to capture the underlying patterns in the data.

### Bias-Variance Tradeoff
The fundamental tension in model building: reducing bias (by making the model more complex) typically increases variance (sensitivity to training data noise), and vice versa. The goal is to find the balance that minimizes total error on unseen data.

### Box-Cox Transformation
A family of power transformations parameterized by lambda ($\lambda$) that includes log, square root, and reciprocal as special cases. Used in data cleaning to normalize skewed distributions.

---

## C

### Coefficient ($\beta$)
In linear regression, the weight assigned to each feature. A positive coefficient means the feature increases the predicted target; a negative coefficient means it decreases the prediction. In regularized regression, coefficients are shrunk toward zero. In tree-based models, coefficients are not used.

### Correlation
A measure of the linear relationship between two variables, ranging from −1 (perfect negative) to +1 (perfect positive). High correlation between features (multicollinearity) can destabilize regression coefficients.

### Cross-Validation (k-Fold)
A technique for evaluating model performance by splitting the training data into k folds, training on k−1 folds and validating on the remaining fold, then repeating k times. The average across all k folds gives a more reliable performance estimate than a single train/test split.

---

## D

### Data Leakage
When information from the validation or test set inadvertently influences the training process, leading to overly optimistic performance estimates. Common causes include scaling before splitting, imputing missing values on the full dataset, or using future information. Pipelines prevent this.

### Double Shrinkage Bias
A phenomenon in ElasticNet where both the L1 and L2 penalties shrink coefficients simultaneously, potentially leading to more bias than either Ridge or Lasso alone. Relevant when choosing between regularization strategies.

### Dummy Variable Trap
The problem of perfect multicollinearity that occurs when one-hot encoding creates K binary columns for a K-category feature. One column is perfectly predictable from the others. Solved by dropping one category (`drop_first=True`).

---

## E

### ElasticNet
A regularization method that combines L1 (Lasso) and L2 (Ridge) penalties, controlled by the `l1_ratio` parameter. Can perform feature selection (like Lasso) while maintaining stability with correlated features (like Ridge). Requires a 2D grid search for tuning.

---

## F

### Feature Engineering
Creating new features from existing ones to improve model performance. Common techniques: interaction terms, polynomial features, binning continuous variables, domain-specific constructions. Applied during data cleaning in Part 1.

### Feature Importance
A measure of how much each input feature contributes to a model's predictions. For regularized linear models with standardized features, it can be read from the size of the standardized coefficients (beta weights). Tree-based models measure it differently; see the 17_2_2 glossary.

### Feature Scaling
Transforming features so they are on a comparable scale (typically mean=0, std=1 using StandardScaler). Required for regularization and gradient-based optimization. Not required for tree-based models. **Critical:** the scaler must be fit on training data only to prevent data leakage.

### Feature Selection
The process of identifying and retaining only the most useful input features. Methods covered: forward selection, backward selection, Lasso regularization (automatic).

### Forward Selection
A feature selection algorithm that starts with no features and iteratively adds the most useful one at each step, based on cross-validation performance. Stops when adding any further feature no longer improves performance. Computationally efficient but cannot remove features once added.

### F-Test
A classical statistical test for whether adding or removing a feature significantly improves a model — the criterion used in *traditional* stepwise selection. Note: the forward/backward selection in 17_2_1_2 does **not** use an F-test; it uses scikit-learn's `SequentialFeatureSelector`, which adds or removes features based on cross-validated score instead.

---

## G

### GridSearchCV
A scikit-learn tool that systematically tests all combinations of hyperparameters from a defined grid, using cross-validation to evaluate each combination. Returns the best-performing hyperparameter set and can refit the model on the full training data.

---

## H

### Heteroscedasticity
A condition where the variance of the errors is not constant across the range of predicted values. Violates an assumption of linear regression and can make confidence intervals unreliable.

### Hyperparameter
A configuration setting that is chosen before training begins, as opposed to parameters (coefficients) that are learned during training. Examples: alpha, max_depth, n_estimators, learning_rate, l1_ratio.

---

## I

### Imputation
The process of filling in missing values. Common methods include median imputation (for numeric features) and mode imputation (for categorical features). **Critical:** imputation must be fit on the training set and applied to the test set separately.

### Inner Loop (Nested CV)
In nested cross-validation, the inner loop performs grid search or hyperparameter tuning within each outer training fold. Finds the best hyperparameters for that specific training subset without seeing the outer test fold.

### Intercept ($\beta_0$)
The baseline prediction of a regression model when all features are zero. In regularized regression, the intercept is typically not penalized.

---

## L

### L1 Regularization (Lasso)
A penalty that adds the sum of absolute coefficient values to the loss function. Can set coefficients exactly to zero, performing automatic feature selection.

### L2 Regularization (Ridge)
A penalty that adds the sum of squared coefficient values to the loss function. Shrinks all coefficients toward zero but rarely sets them exactly to zero. Handles multicollinearity by distributing weight across correlated features.

### l1_ratio
In ElasticNet, the mixing parameter between L1 and L2 penalties. l1_ratio = 1 is pure Lasso; l1_ratio = 0 is pure Ridge; l1_ratio = 0.5 is an equal mix.

### Learning Curve
A plot showing how model performance (training and validation scores) changes as the training set size increases. Used to diagnose whether a model would benefit from more data, more complexity, or less:
- **Overfitting:** training score high, validation low, large gap
- **Underfitting:** both scores converge at a low value
- **Need More Data:** validation score still climbing

### Log Transformation
Applying the natural logarithm to a variable, typically the target (SalePrice). Used to reduce right-skew, stabilize variance, and convert multiplicative relationships into additive ones. After log transformation, coefficients are interpreted in log-units rather than original dollar units.

---

## M

### Mean Squared Error (MSE)
The average of the squared differences between predicted and actual values. Sensitive to outliers because errors are squared.

### Median Imputation
Filling missing numeric values with the median of the observed values. Preferred over mean imputation when the distribution is skewed, as the median is robust to outliers.

### Multicollinearity
A condition where two or more independent variables are highly correlated with each other. Makes individual regression coefficients unstable and hard to interpret, even if overall predictions remain accurate. Detected with VIF; handled by Ridge regression.

---

## N

### Naive Baseline
The simplest possible model — typically one that always predicts the mean of the target variable. Any useful model must significantly exceed the baseline.

### Nested Cross-Validation
A two-layer cross-validation approach: the inner loop tunes hyperparameters within each outer training fold, and the outer loop evaluates the tuned model on an independent holdout fold. Provides an unbiased estimate of model performance. The gold standard for honest evaluation.

---

## O

### One-Hot Encoding
A technique for converting categorical variables into binary (0/1) columns. Each category becomes its own column, with 1 indicating presence and 0 indicating absence. After one-hot encoding, the Ames dataset grows from ~38 to ~225+ features.

### Optimistic Bias
The upward bias in a performance estimate that occurs when hyperparameters are selected based on their score on the same data used for evaluation. The selected hyperparameters are partly tuned to noise. Nested cross-validation corrects this.

### Outer Loop (Nested CV)
In nested cross-validation, the outer loop splits the data into folds for evaluation. Each outer fold uses a model tuned by the inner loop, providing an unbiased performance estimate.

### Overfitting
When a model learns the noise and specificities of the training data rather than the underlying patterns. Characterized by high training performance but poor performance on unseen data.

### Outlier
An observation that is significantly different from the majority of the data. Can disproportionately influence regression models, especially those using squared error.

---

## P

### Pipeline
A scikit-learn tool that chains together multiple processing steps (e.g., scaling, feature selection, modeling) into a single estimator. Ensures that each step is applied correctly within cross-validation, preventing data leakage. The "Double Underscore" rule accesses nested parameters: `estimator__param`.

### Polynomial Features
Features created by raising existing features to a power (e.g., X², X³) or multiplying features together (interaction terms). Allow linear models to capture non-linear relationships.

### Production Model
The final model trained on 100% of the available data after all tuning and evaluation is complete. In the nested CV workflow, the production model is obtained by running a final grid search on the full dataset and using `refit=True`.

---

## R

### $R^2$ (R-squared)
The proportion of variance in the target variable explained by the model. Ranges from negative infinity to 1, where 1 is a perfect fit and 0 means the model is no better than predicting the mean.

### Refit
In GridSearchCV, the `refit=True` parameter means that after finding the best hyperparameters, the model is retrained on the entire training set so it is ready for predictions.

### Regularization
A technique that adds a penalty on coefficient size to the loss function, preventing overfitting. L1 (Lasso) and L2 (Ridge) are the two most common types. Requires feature scaling.

### Residual
The difference between the actual and predicted values for a single observation.

### Residual Plot
A scatter plot of residuals against predicted values. Used to diagnose model problems: a funnel shape indicates heteroscedasticity; a pattern indicates non-linearity; random scatter indicates a well-fitted model.

### Ridge Regression
See **L2 Regularization (Ridge)**.

### RMSE (Root Mean Squared Error)
The square root of MSE. In the same units as the target variable, making it more interpretable than MSE.

---

## S

### Safe Drop Pattern
A coding practice where columns are dropped only if they exist in the DataFrame, preventing errors when the cleaning pipeline is run on different versions of the dataset.

### StandardScaler
A scikit-learn transformer that standardizes features to have mean=0 and standard deviation=1. Fitted on the training set and applied to both training and test sets. Required for regularization.

### Stepwise Selection
A family of feature selection methods (forward, backward, or bidirectional) that iteratively add or remove features based on cross-validation performance. Implemented via scikit-learn's `SequentialFeatureSelector`.

### Structural Multicollinearity
Multicollinearity created by the way features are constructed, such as including both X and X² in a model without centering X first. Can be resolved by standardizing features before creating polynomial terms.

---

## T

### Train-Test Split
Dividing the dataset into a training set (typically 80%) used to fit the model and a test set (typically 20%) used to evaluate generalization performance. The test set must remain completely unseen during training, tuning, and feature selection.

---

## U

### Underfitting
When a model is too simple to capture the underlying patterns in the data. Results in both training and test scores being low. On a learning curve, both training and validation scores converge at a low value.

---

## V

### Variance
How much a model's predictions vary when trained on different subsets of the data. High variance leads to overfitting — the model is too sensitive to the specific training samples.

### Variance Inflation Factor (VIF)
A measure of how much a coefficient's variance is inflated due to multicollinearity with other features. VIF = 1 means no correlation; VIF > 5 indicates moderate concern; VIF > 10 indicates severe multicollinearity.

---

## Y

### Yoked Variables
Paired features where a categorical feature being "None" always corresponds to a numeric feature being 0. Creates redundancy and can confuse the model. Resolved by dropping or combining the paired features.
