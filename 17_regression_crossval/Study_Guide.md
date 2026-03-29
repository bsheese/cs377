# Study Guide: Predicting Housing Prices in Ames, Iowa

This guide summarizes the key concepts from each part of the regression series. Refer back to each part's notebook for more details.

---

## Part Summaries

### Part 1: Data Cleaning
- **Yolked Variables**: Paired features where one is categorical (e.g., "None") and the other is numeric (often 0 when categorical is "None")
- **One-Hot Encoding**: Converting categorical variables into binary (0/1) columns
- **Dummy Variable Trap**: The problem of perfect multicollinearity when using one-hot encoding without `drop_first=True`
- **Missing Value Imputation**: Filling NaN values, typically with median for numeric features
- **Outlier Removal**: Removing extreme data points that could skew model results
- **Safe Drop Pattern**: Dropping columns only if they exist in the DataFrame

*See [Part 1 Notebook](./17_2_4_1_MLR_Ames_Part1_Revised.ipynb) for detailed data cleaning steps.*

---

### Part 2: Forward and Backward Selection
- **Forward Selection**: Iteratively adding the most significant feature at each step
- **Backward Selection**: Iteratively removing the least significant feature at each step
- **SequentialFeatureSelector**: sklearn tool for automated feature selection
- **K-Fold Cross-Validation**: Splitting data into k folds to train k times, each time using k-1 folds for training and 1 for validation
- **Train-Test Split**: Dividing data into training (typically 80%) and test (20%) sets
- **VIF (Variance Inflation Factor)**: Measure of multicollinearity; VIF > 5 indicates moderate correlation concerns
- **Multicollinearity**: When independent variables are highly correlated with each other
- **Polynomial Features**: Creating squared, cubed, or interaction terms to capture non-linear relationships
- **Centering/Standardizing**: Subtracting mean and dividing by std to avoid structural multicollinearity with polynomials
- **Log Transformation**: Applying log() to right-skewed target variable to normalize distribution and improve model fit

*See [Part 2 Notebook](./17_2_4_2_MLR_Ames_Part2_Revised.ipynb) for feature selection methods.*

---

### Part 3: Regularization
- **Ridge Regression (L2)**: Penalizes the sum of squared coefficients; shrinks coefficients toward zero but rarely sets them exactly to zero
- **Lasso Regression (L1)**: Penalizes the sum of absolute coefficients; can set coefficients exactly to zero (feature selection)
- **ElasticNet**: Combines both L1 and L2 penalties
- **Alpha (α)**: The regularization strength parameter (higher = more penalty/more shrinkage)
- **l1_ratio**: In ElasticNet, controls the mix: 0 = pure Ridge, 1 = pure Lasso, 0.5 = equal mix
- **Feature Scaling Requirement**: StandardScaler is REQUIRED for regularization to work properly
- **Bias-Variance Tradeoff**: High alpha = high bias / low variance; low alpha = low bias / high variance

*See [Part 3 Notebook](./17_2_4_3_MLR_Ames_Part3_Revised.ipynb) for regularization methods.*

---

### Part 4: Hyperparameter Tuning
- **GridSearchCV**: Systematically testing all combinations of hyperparameters from a defined grid
- **Pipeline**: sklearn Pipeline that bundles scaling, feature selection, and modeling to prevent data leakage
- **Learning Curves**: Plots showing how model performance changes with training set size; diagnose overfitting vs. underfitting
- **Nested Cross-Validation**: Using CV both for hyperparameter tuning (inner loop) AND for unbiased evaluation (outer loop)
- **Optimistic Test Scores**: Test scores after GridSearch are biased because hyperparameters were tuned on the same training data
- **Bias-Variance Visualization**: Showing how training/validation scores change with alpha values

*See [Part 4 Notebook](./17_2_4_4_MLR_Ames_Part4_Revised.ipynb) for hyperparameter tuning methods.*

---

### Part 5: Tree-Based Methods
- **Decision Tree**: Splits data by asking yes/no questions at each node; prone to overfitting
- **Random Forest (Bagging)**: Ensemble of many decision trees, each trained on a bootstrap sample
- **Gradient Boosting**: Sequential ensemble where each tree corrects the errors of the previous trees
- **XGBoost (eXtreme Gradient Boosting)**: Optimized implementation of gradient boosting
- **HistGradientBoostingRegressor**: sklearn's fast histogram-based gradient boosting that handles categoricals natively
- **Feature Importance (Impurity-Based)**: Measures how much each feature contributes to error reduction across all trees
- **Hyperparameters (Tree Models)**: `n_estimators`, `learning_rate`, `max_depth`, `min_samples_leaf`, etc.
- **Categorical Handling**: Tree models don't require one-hot encoding; they can handle categoricals directly

*See [Part 5 Notebook](./17_2_4_5_MLR_Ames_Part5_Revised.ipynb) for tree-based methods.*

---

## Glossary of Terms

| Term | Definition |
|------|------------|
| **Bias** | Error from overly simplistic assumptions in the model; high bias leads to underfitting |
| **Bootstrap Sampling** | Random sampling with replacement, used to create diverse training sets for ensemble methods |
| **Cross-Validation (CV)** | Technique to evaluate models by training/validating on different subsets of data |
| **Data Leakage** | When information from validation data inadvertently influences training |
| **Decision Tree** | A model that splits data based on feature thresholds to make predictions |
| **ElasticNet** | Regularization combining L1 and L2 penalties |
| **Ensemble Learning** | Combining multiple models to improve predictive performance |
| **Feature Importance** | Measure of how much each feature contributes to the model's predictions |
| **Gradient Boosting** | Sequential ensembles where each tree corrects previous errors |
| **Hyperparameter** | External configuration set before training (e.g., alpha, max_depth) |
| **L1 Regularization (Lasso)** | Penalty using sum of absolute coefficient values |
| **L2 Regularization (Ridge)** | Penalty using sum of squared coefficient values |
| **Learning Curve** | Plot showing model performance vs. training set size |
| **Multicollinearity** | When independent variables are highly correlated |
| **One-Hot Encoding** | Converting categorical variables to binary columns |
| **Overfitting** | Model learns noise in training data; performs poorly on new data |
| **Pipeline** | Chain of transformations applied sequentially to data |
| **R² (R-squared)** | Proportion of variance explained by the model; higher is better |
| **MSE (Mean Squared Error)** | Average squared difference between predicted and actual values; sensitive to outliers |
| **RMSE (Root Mean Squared Error)** | Square root of MSE; in same units as the target variable |
| **StandardScaler** | Transforms features to have mean=0 and std=1 |
| **Underfitting** | Model is too simple to capture patterns in data |
| **Variance** | How much model predictions vary with different training data; high variance leads to overfitting |
| **VIF (Variance Inflation Factor)** | Measure of how much a coefficient's variance is inflated by multicollinearity |

---

## Short-Answer Questions

### Part 1: Data Cleaning

1. What is a "yolked" variable in the context of the Ames Housing dataset? Give an example.
2. Why do we use `drop_first=True` when one-hot encoding? What problem does this prevent?
3. Why do we remove houses with `Gr Liv Area` > 4000 square feet from the dataset?

### Part 2: Forward/Backward Selection

4. Explain the difference between forward selection and backward selection. When might you choose one over the other?
5. What does a VIF value greater than 5 indicate? What about greater than 10?
6. In the context of polynomial features, what is "structural multicollinearity" and how do you prevent it?
7. Why does the dummy variable trap require us to drop one category when one-hot encoding?
8. Why do we apply a log transformation to the SalePrice target variable? What problem does it solve?

### Part 3: Regularization

8. Compare Ridge and Lasso regression: How do their penalty terms differ? What happens to coefficients in each case?
9. Why is feature scaling required before applying regularization?
10. Configure an ElasticNet model to be: (a) Pure Ridge, (b) Pure Lasso, (c) Equal mix of L1 and L2.
11. Higher alpha values result in models with more bias or less bias? Explain why.

### Part 4: Hyperparameter Tuning

12. What is the purpose of using a Pipeline in GridSearchCV? What would break if you didn't use one?
13. If the learning curve shows training score high but validation score low, is the model overfitting or underfitting? What about if both are low?
14. Why are the test R² scores reported after GridSearch considered "optimistic"?
15. Briefly explain what nested cross-validation accomplishes that a simple train-test split does not.

### Part 5: Tree-Based Methods

16. Why don't tree-based models require feature scaling?
17. Unlike linear model coefficients, feature importance scores tell you *what* the model uses but not *how* it affects the prediction. Explain why.
18. What problem can arise when two highly correlated features are included in a tree model? How might this affect feature importance scores?
19. Compare Random Forests (bagging) and Gradient Boosting: How do they differ in how they build their ensembles?

---

## Answer Key

### Part 1 Answers

1. **Yolked variables**: Paired categorical+numeric features where the numeric is 0 whenever the categorical is "None" (e.g., Basement Quality and Basement Square Footage)
2. **Dummy variable trap**: Without dropping one category, you have perfect multicollinearity (one column is perfectly predictable from others)
3. **Outliers removed**: Large houses sold for very little were inherited - these are non-representative outliers

### Part 2 Answers

4. **Forward**: Start with no features, add one at a time. **Backward**: Start with all, remove one at a time. Forward is better when there are many features and you want to avoid the combinatorial explosion of backward.
5. **VIF > 5**: Moderate correlation concerns. **VIF > 10**: Very high correlation, likely problematic.
6. **Structural multicollinearity**: Created when squaring or cubing un-standardized features (X and X² are now correlated). Prevent by standardizing before creating polynomials.
7. **Dummy variable trap**: One category column is perfectly predictable from the others (if all are 0, the dropped category must be 1).
8. **Log transformation**: Applied to right-skewed SalePrice to reduce the effect of extreme values; helps normalize the distribution, improves model fit, and stabilizes variance (especially important for predicting high-value homes)

### Part 3 Answers

8. **Ridge (L2)**: Penalizes sum of squared coefficients; shrinks toward zero but rarely exactly zero. **Lasso (L1)**: Penalizes sum of absolute coefficients; can set coefficients exactly to zero (feature selection).
9. **Feature scaling required**: Regularization applies penalty equally to all coefficients - without scaling, features with larger ranges dominate the penalty.
10. **ElasticNet config**: (a) l1_ratio=0, (b) l1_ratio=1, (c) l1_ratio=0.5
11. **Higher alpha = more bias**: The penalty forces coefficients closer to zero, reducing variance but also underfitting

### Part 4 Answers

12. **Pipeline purpose**: Prevents data leakage by ensuring scaler/transformer is fit only on training data in each CV fold, not on validation fold
13. **Training high/validation low**: Overfitting. **Both low**: Underfitting.
14. **Optimistic test R²**: Hyperparameters were tuned to maximize performance on the training data, so test set performance appears better than true generalization
15. **Nested CV**: Uses inner loop for hyperparameter tuning and outer loop for unbiased evaluation; provides less biased estimate of out-of-sample performance

### Part 5 Answers

16. **No scaling needed**: Trees make splits based on threshold values, not distances; scaling doesn't affect split decisions
17. **Feature importance limitation**: Importance tells you how much the model used the feature for splitting, not whether the effect is positive or negative on price
18. **Correlated features**: Model can only pick one for primary split - that feature gets ALL the importance while correlated feature gets NONE, even though both are useful
19. **Random Forest vs Gradient Boosting**: RF trains trees in parallel on bootstrap samples independently. Boosting trains trees sequentially, each correcting previous trees' errors.