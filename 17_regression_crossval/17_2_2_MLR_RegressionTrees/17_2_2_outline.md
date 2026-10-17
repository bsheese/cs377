# 17_2_2 Regression Trees — Topic Outline

This document outlines the topics in 17_2_2, which revisits the Ames data from the 17_2_1 series with tree-based models. Tree mechanics introduced here are assumed by 18_6 (Ensembles).

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
- Inner loop: 3 folds, 8 hyperparameter combinations
- Outer loop: 5 folds, 120 total model fits
- Nested CV $R^2$: competitive with regularized linear models

### Model Comparison Summary
- Decision Tree → Random Forest → HGB → XGBoost
- Trade-off: accuracy vs. interpretability vs. computational cost

### Final Thought
- OLS + Regularization = interpretable but sensitive to data quality
- XGBoost = not interpretable but robust to messy data
- Practical takeaway: one test split is one draw; report the nested CV estimate
