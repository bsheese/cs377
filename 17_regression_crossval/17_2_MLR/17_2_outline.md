# 17_2 MLR and Regularization — Topic Outline

This document provides a complete outline of all topics covered across the six notebooks in the 17_2 MLR and Regularization series.

---

## 17_2_1_1: Data Cleaning

**Dataset:** Ames Housing — 2,930 observations, 82 features. Goal: clean and prepare for multiple linear regression.

### Purpose of this Notebook
- Why data cleaning matters for regression quality
- Under-cleaning (noise in, garbage out) vs. over-cleaning (data loss)

### Typical Checks
- Using `.info()`, `.describe()`, and boxplots to inspect types, ranges, and outliers
- Identifying numeric vs. categorical feature mix

### Common Remediation Approaches
- Consistent units/scales: converting between square feet, acres, etc.
- Handling impossible values: zero living area, negative values
- Types of outliers and diagnostics for detecting them

### Common Transformations
- Log, square root, Box-Cox for skewed distributions
- Why transformations matter for inference
- Important note: interpretation changes after transformation

### Cleaning Specifically with MLR in Mind
- Unlike simple regression, MLR assumes linear additivity
- Feature interactions and multicollinearity awareness

### Handling Missing and Extreme Values (Train-Set Only)
- Median imputation for missing numeric values
- Why you must fit imputation on the training set only

### Feature Engineering
- Consolidating correlated floor-area columns into `Total_Square_Footage`
- The deterministic vs. statistical preprocessing distinction (what is safe before the split)

### Encoding Categoricals for the Algorithm
- Ordinal encoding for ranked quality scales; one-hot encoding (`OneHotEncoder`, `drop='first'`, `handle_unknown='ignore'`) for nominal variables
- The Dummy Variable Trap: perfect multicollinearity from redundant columns
- Unseen test-set categories encoded as all zeros

### Target Variable Housekeeping
- Log-transforming the target (SalePrice) to reduce skew
- Why log transforms stabilize variance across price ranges

### Summary
- After one-hot encoding, the dataset expands to roughly 225 features
- Why different modeling techniques require different levels of cleaning
- Trade-offs of aggressive vs. conservative feature dropping
- Cleaned data feeds into Part 2 for feature selection

---

## 17_2_1_2: Forward/Backward Selection and VIF

**Dataset:** Ames Housing (cleaned from Part 1). Goal: select features, assess multicollinearity.

### Feature Shortlisting
- Speeding up selection by pre-filtering the feature set
- Removing near-constant and highly correlated features first

### Forward Selection
- Start with no features, add one at a time based on CV improvement
- Test each feature individually, add the best one, repeat
- `n_features_to_select='auto'`: stops when improvement becomes small
- Mini-example: stepping through feature-by-feature addition
- Advantage: computationally efficient for large feature sets
- Limitation: once a feature is added, it can never be removed

### Backward Selection
- Start with all features, remove the least useful one at a time
- Test each feature individually, remove the worst one, repeat
- Stop when further removal hurts performance
- Advantage: can catch interactions that forward selection misses
- Limitation: more computationally expensive

### The Evaluation Pipeline
- Pipeline: `StandardScaler` → `SequentialFeatureSelector` → `LinearRegression`
- Why the pipeline prevents data leakage (scaler fit only on training folds)

### Interpreting Results
- Exceptional model performance with selected features
- The "Big Three" value drivers: Overall Qual, Gr Liv Area, Garage Area
- Logic check: positive vs. negative coefficient signs
- Selection efficiency: how many features were kept

### Assessing Multicollinearity with VIF
- VIF interpretation: 1 = no correlation, >5 = moderate, >10 = severe
- "The overlapping signals" problem: correlated features inflate variance
- Recommended steps: drop one of a correlated pair, monitor remaining VIF values
- Drop one of the "size" variables (Gr Liv Area vs. Garage Area)
- Monitor `Overall Qual` and `Year Built` for remaining collinearity

### Summary
- Model achieves strong $R^2$ with a compact feature set
- Multicollinearity still present — teases regularization as a solution

---

## 17_2_1_3: Regularization (Ridge, Lasso, ElasticNet)

**Dataset:** Ames Housing (cleaned). Goal: use regularization to handle multicollinearity and prevent overfitting.

### The 225-Feature Headache
- After one-hot encoding, the feature set explodes
- OLS becomes unstable or impossible with more features than observations
- Regularization provides a solution

### Building a Professional Evaluation Workflow
- Preventing data leakage with Pipelines
- Solving the "Log-Dollar Illusion": evaluating in original dollar units
- The professional evaluation function: pipeline + CV + scoring
- Establishing the OLS baseline

### Ridge Regression (L2 Regularization)
- Penalty: sum of squared coefficients
- Shrinks coefficients toward zero but never exactly to zero
- Benefits: handles multicollinearity, stable coefficients, keeps all features
- Limitations: no feature selection, coefficients lose interpretability, requires scaling

### Lasso Regression (L1 Regularization)
- Penalty: sum of absolute coefficients
- Can set coefficients exactly to zero (automatic feature selection)
- Benefits: performs feature selection, simpler interpretable models
- Limitations: with correlated features, picks one at random, unstable

### ElasticNet (Hybrid)
- Combines L1 and L2 penalties, controlled by `l1_ratio`
- l1_ratio = 0 → pure Ridge; l1_ratio = 1 → pure Lasso
- Benefits: guaranteed performance, handles correlated features
- Limitations: two hyperparameters to tune, "double shrinkage" bias

### Model Comparison
- Ridge is the new champion on this dataset
- Why Lasso and ElasticNet lag: feature set already well-filtered
- Computational cost of 2D search for ElasticNet

### Diagnostics
- Residual analysis: checking for remaining bias
- Identifying model weaknesses

### Feature Interpretation
- Understanding standardized coefficients (Beta weights)
- Extracting the "top move-makers" for business
- Visualizing feature importance
- Business impact and recommendations

### Interim Reflection: Choosing a Regularization Strategy
- Computational cost of 2D search
- "Double shrinkage" bias
- Explicit business or design constraints
- Occam's Razor in $N \gg P$ scenarios
- Deep learning and optimization defaults

---

## 17_2_1_4: Hyperparameter Tuning and Nested Cross-Validation

**Dataset:** Ames Housing (cleaned). Goal: tune hyperparameters and understand nested CV. (Learning curves are practiced in the 17_2_1_9 capstone exercise.)

### What Are Hyperparameters?
- Configuration settings chosen before training (alpha, l1_ratio)
- Distinguished from parameters (coefficients) learned during training
- The goal: balancing the bias-variance tradeoff
- Finding the "Goldilocks" zone: not too much, not too little penalty

### Grid Search
- Systematically testing all hyperparameter combinations
- Pipeline with scaler prevents data leakage
- The "Double Underscore" Pipeline Rule: `estimator__param`
- `refit=True`: retrains on full training set with best params

### Tuning Ridge
- Ridge test performance: modest gain from tuning (metric barely moves)
- Interpreting the flat plateau: model is already well-calibrated

### Tuning Lasso — The Redemption Arc
- Looping through alpha values, plotting training $R^2$ vs. CV $R^2$
- Training $R^2$ falls as alpha increases
- CV $R^2$ rises then falls, peaking at the optimal alpha
- The gap between curves indicates overfitting

### Evaluating the Tuned ElasticNet
- The computational cost of a 2D grid
- Interpreting the ElasticNet mix

### Final Leaderboard and Test Evaluation
- Ridge, Lasso, ElasticNet on the unseen test set
- Why Ridge consistently leads

### Nested Cross-Validation
- **The Problem:** tuning hyperparameters on the same data used for evaluation creates optimistic bias
- **The Solution:** an outer loop for evaluation + an inner loop for tuning
- Visual overview: 5 outer folds, each with its own inner CV grid search
- What actually happens in each outer fold: trains on one subset, inner loop finds best params, evaluates on held-out fold
- Why this is unbiased: the test fold is never seen during tuning
- Comparing the two approaches: standard CV vs. nested CV
- Code implementation and result interpretation

### Learning Curves *(practiced in the 17_2_1_9 capstone exercise)*
- Plotting training and validation scores as training set size increases
- **Overfitting (High Variance):** training score high, validation low, large gap
- **Underfitting (High Bias):** both scores converge at a low value
- **Need More Data:** validation score still climbing, hasn't plateaued

### Conclusion
- Hyperparameter tuning: marginal gain on this particular dataset
- Nested CV: honest, unbiased performance estimate
- Learning curves: diagnose whether more data helps

---

## 17_2_1_5: Nested Cross-Validation Deep Dive

**Dataset:** Ames Housing (cleaned). Goal: thorough implementation and deployment of nested CV.

### The Three Levels of Evaluation
- Level 1: Basic train-test split
- Level 2: Train/test with cross-validation (standard CV)
- Level 3: Nested cross-validation

### The Mechanics
- The Core Idea: separate the data used for tuning from the data used for evaluation
- Visualizing the nested loops
- What actually happens in each outer fold
- Why this is the gold standard for unbiased evaluation

### Implementation
- Step-by-step implementation of nested cross-validation in Python
- `cross_val_score` treats `GridSearchCV` as a single estimator
- Each outer fold gets independently tuned hyperparameters
- Interpreting the code and results
- Extracting the "winning" alphas from each outer fold

### Building and Deploying the Final Production Model
- Step 1: The final grid search (100% of data, not just training folds)
- Step 2: The "refit" magic — `GridSearchCV` retrains on all data
- Step 3: Extracting final business insights from the production model
- Step 4: Saving the model for future deployment

### Conclusion of the Ames MLR Series
- Recap of the end-to-end MLR pipeline
- From raw data to production-ready regularized model

---

## 17_2_2: Tree-Based Methods (Regression Trees, Random Forest, Gradient Boosting, XGBoost)

**Dataset:** Ames Housing (cleaned, then raw for robustness test). Goal: explore tree-based models as an alternative to linear regression.

### Why Trees?
- Linear models assume straight-line relationships; trees partition data into regions
- Trees naturally capture non-linearities, threshold effects, and interactions
- Minimal data cleaning required — robust to messy data
- The "20 Questions" analogy

### How a Regression Tree Works
- Key terminology: root node, decision node, leaf node, branch
- Split selection: tries every possible split on every feature, picks the one that reduces error most
- Overfitting problem: unconstrained tree memorizes the training data
- `max_depth` limits complexity

### Interpreting Depth Experiment
- Test $R^2$ peaks at moderate depth, then declines
- Depth 3: underfitting; Depth None: severe overfitting (train $R^2 = 1.0$)
- Same bias-variance pattern as polynomial features

### Visualizing a Single Tree
- Reading the tree structure from root to leaves
- Feature repetition at different depths captures interactions
- Small-leaf warnings: predictions based on few samples are unreliable

### Random Forests (Bagging + Parallelization)
- **Bootstrap sampling:** each tree trains on a random sample with replacement
- **Feature randomness:** `max_features='sqrt'` — each tree considers ~√n features per split
- Variance reduction: averaging diverse, uncorrelated trees cancels out individual overfitting
- Why `max_depth=None` works in forests: averaging controls overfitting
- Tuned RF achieves strong CV performance

### Gradient Boosting (Sequential Refinement)
- Each new tree predicts the *residuals* (errors) of the ensemble so far
- Process: average prediction → calculate residuals → train tree on residuals → update → repeat
- Bias reduction (vs. RF's variance reduction)
- Robustness test: training on raw, uncleaned data — nearly identical performance

### XGBoost (eXtreme Gradient Boosting)
- Built-in L1/L2 regularization on leaf weights
- Parallelized feature sorting despite sequential nature
- Native missing data handling
- Comparing three boosting approaches: GBM vs. HGB vs. XGBoost

### Feature Importance
- Measures total error reduction attributed to each feature across all trees
- Overall Qual dominates (~45% of model's learning)
- Limitation: tells you *what* not *how* (no direction or shape)
- Correlated features problem: model picks one, the other gets no credit

### XGBoost Hyperparameter Tuning (Nested Cross-Validation)
- Inner loop: 3 folds, 12 hyperparameter combinations
- Outer loop: 5 folds, 180 total model fits
- Nested CV $R^2$: competitive with regularized linear models

### Model Comparison Summary
- Decision Tree → Random Forest → HGB → XGBoost
- Trade-off: accuracy vs. interpretability vs. computational cost

### Final Thought
- OLS + Regularization = interpretable but sensitive to data quality
- XGBoost = not interpretable but robust to messy data
- Practical takeaway: start with Random Forest, move to boosting if you need more accuracy

---

## Cross-Cutting Themes

| Theme | Notebooks |
|---|---|
| **Multicollinearity** | 17_2_1_2 (VIF), 17_2_1_3 (Ridge handles it), 17_2_2 (feature importance problem) |
| **Bias-Variance Tradeoff** | 17_2_1_3 (regularization), 17_2_1_4 (alpha tuning, nested CV), 17_2_2 (tree depth) |
| **Cross-Validation** | 17_2_1_2 (feature selection), 17_2_1_4 (GridSearchCV, nested CV), 17_2_1_5 (nested CV deep dive), 17_2_2 (nested CV for XGBoost) |
| **Overfitting** | 17_2_1_3 (regularization prevents it), 17_2_1_9 (learning curves), 17_2_2 (tree depth) |
| **Feature Selection** | 17_2_1_2 (forward/backward), 17_2_1_3 (Lasso), 17_2_2 (feature importance) |
| **Log Transformation** | 17_2_1_1 (target cleaning), 17_2_1_2 (log interpretation) |
| **Pipelines** | 17_2_1_2 (scaling + selection), 17_2_1_3 (scaling + regularization), 17_2_1_4 (scaling + GridSearchCV) |
| **Model Comparison** | 17_2_1_3 (Ridge vs. Lasso vs. EN), 17_2_1_4 (tuned models), 17_2_2 (tree progression) |
| **Data Cleaning** | 17_2_1_1 (full pipeline), 17_2_2 (robustness to messy data) |
| **One-Hot Encoding** | 17_2_1_1, 17_2_1_3 (feature explosion) |
