# 17_2 Regression Cross-Validation — Topic Outline

This document provides a complete outline of all topics covered across the five notebooks in the 17_2 Regression Cross-Validation series.

---

## 17_2_4_1: Data Cleaning

**Dataset:** Ames Housing — 2,930 observations, 82 features. Goal: clean and prepare for multiple linear regression.

### Examine the Dataframe
- Using `.info()` to inspect column types and non-null counts
- Identifying the mix of numeric and categorical features

### Remove Obviously Uninformative Features
- Dropping unique identifiers (`Order`, `PID`)
- Removing columns with only one unique value (constant features)
- Removing duplicate rows and rows where all values are NaN

### Yolked Variables
- Definition: paired features where a categorical "None" always corresponds to a numeric 0
- Why yolked variables create redundancy and confuse the model
- Resolving yolked variables:
  - Dropping redundant pairs entirely (Pool QC/Pool Area, Garage Yr Blt)
  - Combining into binary indicators (`garage_attached`, `garage_unfinished`)
  - Handling false positives in automated detection

### Cleaning Categorical Features
- Dropping categories where one category accounts for >70% of values
- Collapsing dominant categories (>50%) into binary columns instead of full one-hot encoding
- Grouping rare categories into "Other" (e.g., Foundation → PConc, CBlock, Other)
- Dropping features with too many categories to avoid exploding one-hot columns (Exterior 1st/2nd)

### Cleaning Numeric Features
- Converting highly zero-dominant features (>90%) to boolean
- Dropping features with one value accounting for >90% of observations
- Handling features that are mostly zero but potentially valuable (Mas Vnr Area)
- Median imputation for missing numeric values

### Conclusion
- DataFrame reduced from 82 to ~38 columns
- Why different modeling techniques require different levels of data cleaning
- Trade-offs of aggressive vs. conservative feature dropping

---

## 17_2_4_2: Forward/Backward Selection, Cross-Validation, and Feature Engineering

**Dataset:** Ames Housing (cleaned from Part 1). Goal: select features and diagnose model problems.

### One-Hot Encoding
- Converting categorical features to binary columns with `drop_first=True`
- The Dummy Variable Trap: perfect multicollinearity from redundant category columns

### How Forward and Backward Selection Work
- **Forward Selection:** Start with no features, add one at a time based on CV improvement
  - Mini-example with 3 features (A, B, C)
  - `n_features_to_select='auto'`: stops when no feature improves the score
- **Backward Selection:** Start with all features, remove the least useful one at a time
  - Trade-offs: forward is faster; backward can catch missed interactions

### The Selection and Evaluation Function
- Pipeline: `StandardScaler` → `SequentialFeatureSelector` → `LinearRegression`
- Why the pipeline prevents data leakage (scaler fit only on training folds)
- Using only 2 CV folds to keep computation manageable

### The Modeling Workflow
- Encapsulating train-test split, CV, feature selection, evaluation, and residual plotting
- Running both forward and backward selection to compare results

### Interpreting Results
- **Negative coefficient for Bedroom AbvGr:** holding square footage constant, more bedrooms means smaller rooms → lower value
- **Heteroscedastic residuals:** funnel-shaped error pattern (errors grow with price)
  - Caused by right-skewed SalePrice distribution
  - Violates linear regression assumptions; confidence intervals unreliable
- **Multicollinearity teaser:** correlated features may confuse the selection process

### Improving the Model: Log Transform
- Why log-transforming the target helps: converts absolute errors to percentage errors
- Before vs. after: CV R² improves modestly (0.86 → 0.89), test R² jumps (0.88 → 0.93)
- Why the test score improvement is larger: stabilized variance across price ranges
- Coefficients now in log-units (~12.0) rather than dollars (~180,000)

### Add Polynomials to Model Non-Linear Features
- Why X² allows the regression line to bend while still being "linear" regression
- Three examples: Overall Qual (accelerating premium), Bedroom AbvGr (cramped room penalty), Garage Area (diminishing returns)
- Always include the base linear term with the squared term
- Standardize before squaring to avoid structural multicollinearity
- Forward selection drops all polynomial features — doesn't mean they're useless, just that they don't add enough *additional* power

### Assessing Multicollinearity with VIF
- VIF interpretation: 1 = no correlation, >5 = moderate concern, >10 = severe
- Full feature set: Gr Liv Area VIF = 118 (extreme)
- Reduced feature set: no VIF above 5 (forward selection filtered redundancy)
- But multicollinearity may have affected the selection process itself
- Correlation heatmap: Garage Cars vs. Garage Area at r = 0.89

### Summary
- Model achieves R² ≈ 0.93 with log-transformed target
- Three remaining issues: multicollinearity, possible missed features, polynomial features dropped
- Tease: Part 3 addresses all three with regularization

---

## 17_2_4_3: Regularization

**Dataset:** Ames Housing (cleaned). Goal: use Ridge, Lasso, and ElasticNet to handle multicollinearity and prevent overfitting.

### Preliminaries
- Feature selection and evaluation function (same pipeline as Part 2)
- Modeling workflow function
- Baseline: OLS with forward selection (for comparison)

### Ridge Regression (L2 Regularization)
- Penalty: sum of squared coefficients
- Shrinks coefficients toward zero but never exactly to zero
- Benefits:
  - Handles multicollinearity by distributing weight across correlated features
  - More stable than OLS when features are correlated
  - Keeps all features in the model
- Limitations:
  - Doesn't perform feature selection (all features retained)
  - Coefficients lose traditional interpretability (can't say "each sq ft adds $X")
  - Requires feature scaling

### Lasso Regression (L1 Regularization)
- Penalty: sum of absolute coefficients
- Can set coefficients exactly to zero (automatic feature selection)
- Benefits:
  - Performs feature selection automatically
  - Produces simpler, more interpretable models
  - Handles high-dimensional data well
- Limitations:
  - With correlated features, picks one at random and drops the rest
  - Unstable feature selection when features are highly correlated
  - If alpha is too high, drops ALL features (predicts only the intercept)

### ElasticNet
- Combines L1 and L2 penalties, controlled by `l1_ratio`
- l1_ratio = 0 → pure Ridge; l1_ratio = 1 → pure Lasso; l1_ratio = 0.5 → equal mix
- Benefits:
  - Guarantees performance at least as good as Ridge or Lasso alone
  - Handles correlated features better than Lasso (Ridge component)
  - Still performs feature selection (Lasso component)
  - Popular in bioinformatics, genomics, econometrics
- Limitations:
  - Two hyperparameters to tune (alpha + l1_ratio)
  - Still assumes linear relationships
  - Less interpretable than simple linear regression

### Model Comparison Summary
- Ridge: keeps all 38 features, stable coefficients
- Lasso: keeps only 22 features, similar test R² to Ridge
- ElasticNet: balances between the two
- On full one-hot encoded data (276 features): Ridge keeps 272, Lasso/ElasticNet much more selective
- Trade-off: predictive accuracy vs. interpretability

---

## 17_2_4_4: Grid Search, Nested Cross-Validation, and Learning Curves

**Dataset:** Ames Housing (cleaned). Goal: systematically find optimal hyperparameters and get unbiased performance estimates.

### Why Tune Hyperparameters?
- Alpha controls regularization strength: too low = overfitting, too high = underfitting
- The "Goldilocks" zone: not too much, not too little penalty

### 1. Visualizing the Bias-Variance Tradeoff (Manual Tuning)
- Looping through alpha values and plotting training R² vs. CV R²
- Training R² falls steadily as alpha increases (more constraint)
- CV R² rises then falls, peaking at the optimal alpha
- The gap between curves indicates overfitting

### 2. Automated Tuning with GridSearchCV
- Systematically testing all hyperparameter combinations
- Pipeline with scaler prevents data leakage
- `refit=True`: retrains on full training set with best params
- Ridge: 8 alphas × 5 folds = 40 fits
- Lasso: 8 alphas × 5 folds = 40 fits
- ElasticNet: 4 alphas × 5 l1_ratios × 5 folds = 100 fits

### 3. Final Evaluation on Test Set
- Ridge, Lasso, and ElasticNet achieve nearly identical test R² scores
- Why: remaining features after dropping strongest predictors are weakly informative

### The Problem with Our Test Set Score
- The best alpha was selected because it performed well on *those specific* CV folds
- Test set score is slightly inflated (optimistic bias)
- Same overfitting problem as feature selection: picking the best option partly picks up noise

### Nested Cross-Validation
- **Inner Loop:** GridSearchCV tunes hyperparameters within each outer training fold
- **Outer Loop:** Evaluates the tuned model on an independent holdout fold
- Visual diagram: 5 outer folds, each with its own inner CV search
- Concrete trace: Outer Fold 1 trains on 2,340 houses, inner loop finds best alpha, evaluates on 585 held-out houses
- Key insight: `cross_val_score` treats `GridSearchCV` as a single estimator
- Each outer fold gets independently tuned hyperparameters
- Comparison table: train/test + GridSearchCV vs. nested CV (scores, bias, variance, cost)

### Learning Curves
- **Overfitting (High Variance):** Training score high, validation score much lower; gap between curves
- **Underfitting (High Bias):** Both scores converge at a low value; adding data won't help
- **Need More Data:** Validation score still climbing, hasn't plateaued

### Conclusion
- Moved from guessing alpha to systematically finding optimal values
- GridSearchCV automates the search; nested CV gives honest performance estimates
- Learning curves diagnose whether the model needs more data, more complexity, or less

---

## 17_2_4_5: Tree-Based Methods

**Dataset:** Ames Housing (cleaned, then raw for robustness test). Goal: explore tree-based models as an alternative to linear regression.

### Why Trees?
- Linear models assume straight-line relationships; trees partition data into regions
- Trees naturally capture diminishing returns, threshold effects, and feature interactions
- No manual polynomial engineering needed; minimal data cleaning required

### How a Regression Tree Works
- "20 Questions" analogy: binary yes/no questions split data at each node
- Node: where a question is asked. Branch: yes/no path. Leaf: endpoint prediction (average of samples)
- Split selection: tries every possible split on every feature, picks the one that reduces error most
- Overfitting problem: unconstrained tree achieves R² = 1.0 on training by memorizing
- `max_depth` limits complexity

### Interpreting the Depth Experiment
- Test R² peaks at depth 7 (0.8319), then declines
- Depth 3: moderate scores (~0.71-0.75) — underfitting
- Depth None: training R² = 1.0, test R² = 0.7986 — severe overfitting
- Same bias-variance pattern as polynomial features in Part 2, but controlled by a single parameter

### Visualizing a Single Tree
- Root node: the first and most important split
- Feature repetition at different depths captures interactions
- Small-leaf warnings: predictions based on few samples are unreliable
- Even at depth 5, visualization is becoming difficult to read

### Random Forests: Bagging and Parallelization
- **Bootstrap Sampling:** each tree trains on a random sample with replacement
- **Feature Randomness:** `max_features='sqrt'` — each tree considers ~√n features at each split
- Variance reduction: averaging diverse, uncorrelated trees cancels out individual overfitting
- Why `max_depth=None` works in forests: averaging controls overfitting
- Tuned RF achieves CV R² = 0.8771

### Gradient Boosting: Sequential Refinement
- Each new tree predicts the *residuals* (errors) of the ensemble so far
- Process: average prediction → calculate residuals → train tree on residuals → update → repeat
- Bias reduction (vs. RF's variance reduction)
- HistGradientBoostingRegressor: histogram binning for speed, native categorical/missing value support
- Raw-data HGB achieves R² = 0.9449 — nearly identical to cleaned version

### XGBoost: eXtreme Gradient Boosting
- Built-in L1/L2 regularization on leaf weights (connecting back to Ridge/Lasso from Part 3)
- Parallelized feature sorting despite sequential nature
- Native missing data handling (tests left vs. right at each split)
- XGBoost test R² = 0.9418

### Feature Importance
- Measures total error reduction attributed to each feature across all trees
- Overall Qual: 45.76% of model's learning
- Limitation: tells you *what* not *how* (no direction or shape information)
- Correlated features problem: model picks one, the other gets no credit
- Gr Liv Area: VIF = 118 but only 2.98% importance


### XGBoost Hyperparameter Tuning: Nested Cross-Validation
- Inner loop: 3 folds, 12 hyperparameter combinations
- Outer loop: 5 folds
- Total: 180 model fits
- Nested CV R² = 0.9177 ± 0.0195
- Comparison to Part 4's Ridge nested CV (0.7897): ~13-point improvement

### Model Comparison Summary
- Decision Tree (0.83) → Random Forest (0.88) → HGB (0.94) → XGBoost (0.94) → XGBoost Nested CV (0.92 ± 0.02)
- Trade-off: accuracy vs. interpretability vs. computational cost

### Final Thought
- OLS = interpretable but sensitive to data quality
- XGBoost = not interpretable but robust to messy data
- Practical takeaway: start with Random Forest, move to boosting if you need the last bit of accuracy

---

## Cross-Cutting Themes

These concepts appear throughout multiple notebooks:

| Theme | Notebooks |
|---|---|
| **Multicollinearity** | 17_2_4_2 (VIF), 17_2_4_3 (Ridge handles it), 17_2_4_5 (feature importance problem) |
| **Bias-Variance Tradeoff** | 17_2_4_3 (regularization), 17_2_4_4 (alpha tuning), 17_2_4_5 (tree depth) |
| **Cross-Validation** | 17_2_4_2 (feature selection), 17_2_4_4 (GridSearchCV, nested CV), 17_2_4_5 (nested CV for XGBoost) |
| **Overfitting** | 17_2_4_2 (polynomials), 17_2_4_3 (regularization), 17_2_4_4 (learning curves), 17_2_4_5 (tree depth) |
| **Feature Selection** | 17_2_4_2 (forward/backward), 17_2_4_3 (Lasso), 17_2_4_5 (feature importance) |
| **Log Transformation** | 17_2_4_2 (target), 17_2_4_5 (used throughout tree models) |
| **Pipelines** | 17_2_4_2 (scaling + selection), 17_2_4_3 (scaling + regularization), 17_2_4_4 (scaling + GridSearchCV) |
| **Model Comparison** | 17_2_4_3 (Ridge vs. Lasso vs. EN), 17_2_4_4 (tuned models), 17_2_4_5 (tree progression) |
