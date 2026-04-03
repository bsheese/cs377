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

**Manual Tuning & GridSearchCV:**
- **Bias-Variance Visualization**: Plotting training and CV R² across alpha values to find the "Goldilocks" zone where CV score peaks
- **GridSearchCV**: Systematically testing all combinations of hyperparameters from a defined grid; reports total fits = (number of candidates) × (number of CV folds)
- **Pipeline**: Bundles scaling and modeling to prevent data leakage — the scaler is fit only on each fold's training data, not the validation data
- **refit=True**: After finding the best hyperparameters, GridSearchCV retrains the model on the full training set

**The Optimistic Bias Problem:**
- After GridSearchCV picks the best alpha, the test set score is slightly inflated because the alpha was selected for performing well on *those specific* CV folds
- This is the same overfitting problem seen in feature selection: when you search through many options and pick the best, you're partly picking up noise

**Nested Cross-Validation:**
- **Inner Loop**: GridSearchCV tunes hyperparameters within each outer training fold
- **Outer Loop**: Evaluates the tuned model on an independent holdout fold that was never involved in tuning
- **Key Insight**: `cross_val_score` treats `GridSearchCV` as a single estimator — calling `.fit()` on it triggers the full inner search
- **Result**: Each outer fold gets its own independently tuned hyperparameters; the final score is the average across all outer folds
- **Why It Matters**: Provides an unbiased estimate with quantifiable variance (± standard deviation across outer folds)

**Learning Curves:**
- **Overfitting (High Variance)**: Training score high, validation score much lower; gap between curves
- **Underfitting (High Bias)**: Both scores converge at a low value; adding more data won't help
- **Need More Data**: Validation score still climbing and hasn't plateaued; model would benefit from additional examples

*See [Part 4 Notebook](./17_2_4_4_MLR_Ames_Part4_Revised.ipynb) for hyperparameter tuning methods.*

---

### Part 5: Tree-Based Methods

**Why Trees vs. Linear Models:**
- Linear models assume straight-line relationships; trees partition data into regions and predict the average within each region
- Trees naturally capture diminishing returns, threshold effects, and feature interactions without manual engineering
- Trees don't require one-hot encoding or extensive data cleaning

**Decision Trees:**
- **"20 Questions" Analogy**: The model asks binary yes/no questions, splitting data at each node until reaching a leaf
- **Node**: Where the tree asks a question. **Branch**: The yes/no path. **Leaf**: The endpoint where the prediction is the average of training samples in that group
- **Split Selection**: At each node, the tree tries every possible split on every feature and picks the one that reduces error the most
- **Overfitting**: An unconstrained tree achieves R² = 1.0 on training data by memorizing it; `max_depth` limits complexity
- **Depth Experiment**: Test R² peaks at an optimal depth (e.g., 7), then declines as the tree overfits — classic bias-variance tradeoff

**Random Forests (Bagging):**
- **Bootstrap Sampling**: Each tree trains on a random sample with replacement; different trees see different data
- **Feature Randomness**: At each node, each tree considers only a random subset of features (e.g., `max_features='sqrt'`); prevents every tree from splitting on the same dominant feature
- **Variance Reduction**: Averaging many diverse, uncorrelated trees cancels out individual overfitting
- **Why `max_depth=None` Works**: The averaging process controls overfitting, so deeper trees can capture more nuance without harming generalization

**Gradient Boosting:**
- **Sequential Correction**: Each new tree predicts the *residuals* (errors) of the ensemble so far, not the original target
- **Process**: Start with average prediction → calculate residuals → train tree on residuals → update predictions → repeat
- **Bias Reduction**: Unlike Random Forests (which reduce variance), boosting reduces bias by chipping away at underfitting
- **HistGradientBoostingRegressor**: Uses histogram binning to speed up split-finding; handles categoricals and missing values natively

**XGBoost:**
- **Built-in Regularization**: Applies L1 (`reg_alpha`) and L2 (`reg_lambda`) penalties to leaf weights and tree complexity (connecting back to Ridge/Lasso from Part 3)
- **Speed**: Parallelizes the internal work of building each tree (feature sorting), despite boosting being inherently sequential
- **Missing Data Handling**: Tests sending missing values left or right at each split and learns which direction reduces error

**Feature Importance:**
- **What It Measures**: How much each split on a feature reduced prediction error across all trees, expressed as a percentage
- **Limitation**: Tells you *what* the model prioritizes but not *how* the feature affects the target (direction or shape)
- **Correlated Features Problem**: If two features are correlated, the model picks one for splits and the other gets no credit — even though both are useful
- **Next Steps**: Partial dependence plots or SHAP values can recover the "how" that importance scores alone cannot

**Data Robustness:**
- Running HGB on raw (uncleaned) Ames data achieved nearly the same R² as on carefully cleaned data
- Trees don't require one-hot encoding, handle non-linear relationships automatically, and some implementations handle missing values natively

**Model Comparison Progression:**
Decision Tree (0.83) → Random Forest (0.88) → HGB (0.94) → XGBoost (0.94) → XGBoost Nested CV (0.92 ± 0.02)

*See [Part 5 Notebook](./17_2_4_5_MLR_Ames_Part5_Revised.ipynb) for tree-based methods.*

---

## Glossary of Terms

| Term | Definition |
|------|------------|
| **Bias** | Error from overly simplistic assumptions; high bias leads to underfitting |
| **Bootstrap Sampling** | Random sampling with replacement; creates diverse training sets for ensembles |
| **Correlated Features Problem** | When two features are correlated, the tree picks one for splits and the other gets no importance credit |
| **Cross-Validation (CV)** | Technique to evaluate models by training/validating on different subsets of data |
| **Data Leakage** | When information from validation data inadvertently influences training |
| **Decision Tree** | A model that splits data based on feature thresholds to make predictions |
| **ElasticNet** | Regularization combining L1 and L2 penalties |
| **Ensemble Learning** | Combining multiple models to improve predictive performance |
| **Feature Importance** | Measure of how much each feature contributes to error reduction across all trees |
| **Feature Randomness** | At each node, a tree considers only a random subset of features; decorrelates trees in a forest |
| **Gradient Boosting** | Sequential ensembles where each tree corrects previous errors |
| **Histogram Binning** | Grouping continuous data into discrete bins to speed up split-finding in boosting |
| **Hyperparameter** | External configuration set before training (e.g., alpha, max_depth, n_estimators) |
| **L1 Regularization (Lasso)** | Penalty using sum of absolute coefficient values |
| **L2 Regularization (Ridge)** | Penalty using sum of squared coefficient values |
| **Learning Curve** | Plot showing model performance vs. training set size |
| **Leaf Node** | An endpoint in a tree where the prediction is the average of samples in that group |
| **`learning_rate`** | In boosting, scales each tree's contribution; smaller values need more trees but are more stable |
| **Multicollinearity** | When independent variables are highly correlated |
| **`n_estimators`** | Number of trees in a Random Forest or boosting ensemble |
| **Nested Cross-Validation** | Inner loop tunes hyperparameters; outer loop evaluates on independent folds |
| **One-Hot Encoding** | Converting categorical variables to binary columns |
| **Optimistic Bias** | Inflated test scores after hyperparameter tuning because the best params were selected on the same data |
| **Overfitting** | Model learns noise in training data; performs poorly on new data |
| **Partial Dependence Plots** | Show the direction and shape of a feature's effect on predictions |
| **Pipeline** | Chain of transformations applied sequentially to prevent data leakage |
| **R² (R-squared)** | Proportion of variance explained by the model; higher is better |
| **Residuals (Boosting)** | The errors left by the ensemble; each new boosting tree predicts these residuals |
| **Root Node** | The first and most important split in a decision tree |
| **SHAP Values** | Advanced method for explaining individual predictions and feature effects |
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

9. Compare Ridge and Lasso regression: How do their penalty terms differ? What happens to coefficients in each case?
10. Why is feature scaling required before applying regularization?
11. Configure an ElasticNet model to be: (a) Pure Ridge, (b) Pure Lasso, (c) Equal mix of L1 and L2.
12. Higher alpha values result in models with more bias or less bias? Explain why.

### Part 4: Hyperparameter Tuning

13. What is the purpose of using a Pipeline in GridSearchCV? What would break if you didn't use one?
14. If the learning curve shows training score high but validation score low, is the model overfitting or underfitting? What about if both are low?
15. Why are the test R² scores reported after GridSearch considered "optimistic"?
16. Briefly explain what nested cross-validation accomplishes that a simple train-test split does not.
17. In the nested CV code, `cross_val_score` is called with a `GridSearchCV` object as its estimator. What happens during each outer fold?
18. If the learning curve's validation score is still climbing at the largest training size, what does this suggest?

### Part 5: Tree-Based Methods

19. Why don't tree-based models require feature scaling?
20. A single unconstrained decision tree achieves R² = 1.0 on training data but only 0.80 on test data. What is happening?
21. Why does a Random Forest use bootstrap sampling (drawing with replacement) for each tree?
22. What does `max_features='sqrt'` mean at each split in a Random Forest?
23. Why is `max_depth=None` acceptable in a Random Forest but catastrophic for a single tree?
24. In gradient boosting, what does each new tree predict? How does this differ from a Random Forest?
25. The raw-data HGB model achieved nearly the same R² as the cleaned-data version. What does this demonstrate about tree-based models?
26. How does XGBoost's built-in regularization differ from Ridge and Lasso regularization?
27. Feature importance tells you *what* the model prioritizes but not *how* the feature affects the prediction. Explain why, and contrast with a linear regression coefficient.
28. `Gr Liv Area` has a VIF of 118 but only 2.98% feature importance. Why does a highly predictive feature receive such a low importance score?
29. Compare Random Forests and Gradient Boosting: How do they differ in how they build their ensembles, and what does each one reduce (bias or variance)?

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

9. **Ridge (L2)**: Penalizes sum of squared coefficients; shrinks toward zero but rarely exactly zero. **Lasso (L1)**: Penalizes sum of absolute coefficients; can set coefficients exactly to zero (feature selection).
10. **Feature scaling required**: Regularization applies penalty equally to all coefficients - without scaling, features with larger ranges dominate the penalty.
11. **ElasticNet config**: (a) l1_ratio=0, (b) l1_ratio=1, (c) l1_ratio=0.5
12. **Higher alpha = more bias**: The penalty forces coefficients closer to zero, reducing variance but also underfitting

### Part 4 Answers

13. **Pipeline purpose**: Prevents data leakage by ensuring scaler/transformer is fit only on training data in each CV fold, not on validation fold
14. **Training high/validation low**: Overfitting. **Both low**: Underfitting.
15. **Optimistic test R²**: Hyperparameters were tuned to maximize performance on the training data, so test set performance appears better than true generalization
16. **Nested CV**: Uses inner loop for hyperparameter tuning and outer loop for unbiased evaluation; each outer fold gets independently tuned hyperparameters, providing a less biased estimate with quantifiable variance
17. **What happens each outer fold**: `cross_val_score` calls `.fit()` on the GridSearchCV, which runs the full inner search (trying all hyperparameter combinations with inner CV), finds the best params, refits, and then predicts on the outer test fold
18. **Validation still climbing**: The model has not yet plateaued and would likely benefit from additional training data

### Part 5 Answers

19. **No scaling needed**: Trees make splits based on threshold values, not distances; scaling doesn't affect split decisions
20. **Overfitting**: The tree has created a separate leaf for nearly every training sample, memorizing the training data including its noise
21. **Bootstrap sampling**: Gives each tree a different perspective on the data so they make different errors; averaging these diverse errors cancels out the noise
22. **max_features='sqrt'**: Each tree considers only about √n features (where n = total features) at each individual node, forcing diversity
23. **Averaging controls overfitting**: In a forest, the averaging process across diverse trees cancels out individual overfitting, so deeper trees can capture more nuance without harming generalization
24. **Residuals**: Each new tree predicts the errors left by the ensemble so far, not the original target. RF trains trees independently in parallel; boosting trains them sequentially, each correcting the last
25. **Data robustness**: Tree-based models are remarkably robust to messy, unprocessed data and don't require the extensive preprocessing that linear models need
26. **Leaf weights vs. coefficients**: XGBoost applies penalties to leaf weights and tree complexity, not to linear coefficients. The concept is the same (penalize complexity to prevent overfitting) but applied at the tree level
27. **What vs. how**: Importance tells you how much the model used the feature for splitting, not whether the effect is positive or negative. A linear coefficient tells you both: "each additional square foot adds $50 to the price"
28. **Correlated features**: XGBoost chose correlated features like Overall Qual for splits, leaving Gr Liv Area with little importance credit — even though living area is genuinely predictive
29. **Random Forest vs Gradient Boosting**: RF trains trees in parallel on bootstrap samples independently, reducing variance. Boosting trains trees sequentially, each correcting previous trees' errors, reducing bias
