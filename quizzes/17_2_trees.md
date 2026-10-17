# 17.2.2 · Regression Trees

## 17_2_2: Tree-Based Methods

### Linear models required manual polynomial features to capture non-linear patterns. How do tree-based models handle non-linearity?
- [ ] They apply logarithmic transformations to all features automatically.
- [x] They partition the data into regions and predict the average within each region.
- [ ] They add squared terms to the regression equation during training.
- [ ] They use gradient descent to find the optimal non-linear curve.

### In a regression tree, what value does each leaf node predict?
- [ ] The median of the target values of training samples in that leaf.
- [x] The average of the target values of training samples in that leaf.
- [ ] The most frequent target value among training samples in that leaf.
- [ ] A weighted sum of coefficients applied to the features in that leaf.

### How does a regression tree decide which feature and split point to use at each node?
- [ ] It randomly selects a feature and tries all possible split points on that feature.
- [ ] It selects the feature with the highest correlation to the target variable.
- [x] It tries every possible split on every feature and picks the one that reduces error the most.
- [ ] It uses the feature that was most important in the parent node's split.

### An unconstrained decision tree achieves R² = 1.0 on training data but R² = 0.80 on test data. What is happening?
- [ ] The model is underfitting because it has too few splits to capture the patterns.
- [x] The model is overfitting by creating a separate leaf for nearly every training sample.
- [ ] The training data contains outliers that the model is correctly ignoring.
- [ ] The test set is fundamentally different from the training set due to data leakage.

### In the depth experiment, the test R² peaked at depth 5 and then declined as depth increased. What does this pattern represent?
- [x] The bias-variance tradeoff, where deeper trees reduce bias but increase variance.
- [ ] A computational limitation where deeper trees run out of memory.
- [ ] The learning rate decay, where deeper trees contribute less to the prediction.
- [ ] A data quality issue where deeper splits encounter noisier samples.

### Why does a Random Forest use bootstrap sampling (drawing with replacement) for each tree?
- [ ] To ensure every tree sees the exact same data but in a different order.
- [ ] To reduce the computational cost of training each individual tree.
- [x] To give each tree a different perspective on the data so they make different errors.
- [ ] To eliminate the need for a separate test set during model evaluation.

### In a Random Forest, what does max_features='sqrt' mean at each split?
- [ ] Each tree considers only the square root of the total features across the entire tree.
- [x] Each tree considers a random subset of about √n features at each individual node.
- [ ] The forest builds exactly √n trees, where n is the number of features.
- [ ] Each tree uses the top √n most important features and ignores the rest.

### Why does feature randomness in Random Forests prevent every tree from splitting on the same dominant feature first?
- [ ] Because the dominant feature is removed from the dataset before training begins.
- [x] Because each tree is forced to consider only a random subset of features at each node.
- [ ] Because the dominant feature's values are randomly shuffled before each split.
- [ ] Because the algorithm penalizes trees that use the same feature twice.

### The Random Forest selected max_depth=None as optimal. Why is unlimited depth acceptable in a forest but catastrophic for a single tree?
- [ ] Because the forest uses a different splitting criterion that prevents overfitting.
- [x] Because the averaging process across diverse trees cancels out individual overfitting.
- [ ] Because each tree in the forest sees fewer features, making deep trees simpler.
- [ ] Because the forest automatically prunes trees that show signs of overfitting.

### What is the primary objective of a Random Forest from a bias-variance perspective?
- [ ] Reducing bias by building deeper and deeper trees sequentially.
- [x] Reducing variance by averaging many diverse, uncorrelated predictions.
- [ ] Reducing both bias and variance equally through random feature selection.
- [ ] Eliminating bias entirely by using an infinite number of trees.

### In gradient boosting, what does each new tree predict?
- [ ] The original target variable (e.g., house price) directly.
- [x] The residuals (errors) left by the ensemble of all previous trees.
- [ ] The average of all previous trees' predictions.
- [ ] A weighted combination of the target variable and the previous residuals.

### Gradient boosting starts with an initial prediction equal to the average target value. What is the role of the first tree?
- [ ] It predicts the target variable directly, ignoring the initial prediction.
- [x] It predicts the residuals from the initial prediction, correcting the average.
- [ ] It replaces the initial prediction entirely with its own predictions.
- [ ] It determines which features are most important for all subsequent trees.

### What is the key difference between Random Forests and Gradient Boosting in how they build their ensembles?
- [ ] Random Forests build trees sequentially; Gradient Boosting builds them in parallel.
- [x] Random Forests average independent trees; Gradient Boosting corrects errors sequentially.
- [ ] Random Forests use deep trees; Gradient Boosting uses only single-split trees.
- [ ] Random Forests reduce bias; Gradient Boosting reduces variance.

### Why does HistGradientBoostingRegressor use histogram binning?
- [ ] To convert categorical features into numeric codes automatically.
- [x] To group continuous data into discrete bins, speeding up the split-finding process.
- [ ] To reduce the number of features by combining correlated ones into bins.
- [ ] To handle missing values by placing them into a separate histogram bin.

### The raw-data HGB model achieved R² = 0.9449, nearly identical to the cleaned-data version. What does this demonstrate?
- [ ] That data cleaning is never necessary when using tree-based models.
- [x] That tree-based models are robust to messy data and don't require extensive preprocessing.
- [ ] That the Ames dataset was already clean enough that no cleaning was needed.
- [ ] That HGB automatically performs the same cleaning steps as the notebook's cleaning function.

### How does XGBoost's built-in regularization differ from Ridge and Lasso regularization?
- [x] XGBoost applies penalties to leaf weights and tree complexity, not to linear coefficients.
- [ ] XGBoost only uses L1 regularization, while Ridge and Lasso use both L1 and L2.
- [ ] XGBoost applies regularization after training, while Ridge and Lasso apply it during training.
- [ ] XGBoost does not use regularization; it relies solely on tree depth limits.

### Boosting is inherently sequential, making it hard to parallelize. How does XGBoost achieve speed improvements?
- [ ] It builds multiple trees in parallel and averages their predictions.
- [x] It parallelizes the internal work of building each tree, such as feature sorting.
- [ ] It skips the residual calculation step for trees after the first 50.
- [ ] It uses a single CPU core but with a more efficient optimization algorithm.

### How does XGBoost handle missing values at each split?
- [ ] It imputes missing values with the median before training begins.
- [ ] It removes any sample with a missing value from the training set.
- [x] It tests sending missing values left or right and learns which direction reduces error.
- [ ] It assigns missing values to a separate leaf node that predicts the global average.

### Feature importance in tree models measures how much each split reduced prediction error. What is a key limitation of this metric?
- [ ] It cannot handle categorical features, so their importance is always zero.
- [x] It tells you what the model prioritizes but not how the feature affects the target.
- [ ] It only works for the top 5 features and ignores all remaining features.
- [ ] It requires the model to be retrained after importance scores are calculated.

### In the feature importance plot, Overall Qual accounts for 45.76% of the model's learning. What does this percentage mean?
- [ ] Overall Qual alone can predict 45.76% of house prices without any other features.
- [x] Splits on Overall Qual were responsible for 45.76% of the total error reduction across all trees.
- [ ] 45.76% of the trees in the ensemble used Overall Qual as their first split.
- [ ] Overall Qual has a coefficient of 0.4576 in the underlying linear approximation.

### In Part 2, Log_Gr Liv Area sat at the top of the VIF table (highly redundant with other size variables), yet XGBoost gives Gr Liv Area only 2.98% feature importance. Why?
- [ ] Living area is genuinely unimportant for predicting house prices in Ames.
- [x] XGBoost chose correlated features like Overall Qual for splits, leaving Gr Liv Area with little credit.
- [ ] The VIF calculation is only relevant for linear models and has no bearing on tree models.
- [ ] The feature importance calculation is biased against continuous features.

### XGBoost with fixed settings scored 0.9418 on the notebook's single test split, but 0.9177 under nested CV. What best explains most of the gap?
- [ ] The nested CV used a more conservative scoring metric
- [ ] Tuning on the test split inflated the single-split score
- [ ] The nested CV used fewer trees
- [x] The single split happened to be an easy one; nested CV averages over five different test folds

### The nested CV standard deviation for XGBoost was ±0.0195. What does this tell you?
- [x] The model's performance varies by about 2 percentage points across different data splits.
- [ ] The hyperparameter search found 19.5% of the possible combinations to be optimal.
- [ ] The model is overfitting by approximately 1.95 percentage points on each fold.
- [ ] The inner loop's best score was 1.95 percentage points higher than the outer loop's score.

### The comparison table describes the random forest as 'stable, robust, hard to mess up'. Why is a forest a sensible first ensemble to try?
- [ ] It always produces the highest accuracy of any model
- [x] Averaging deep trees controls overfitting with little tuning, so it gives a strong result with few ways to go wrong
- [ ] It is the only tree method that handles missing values natively
- [ ] Its feature importances show the direction of each effect

### What tradeoff do you accept when moving from linear regression to XGBoost?
- [ ] You gain interpretability but lose the ability to handle non-linear patterns.
- [x] You gain predictive accuracy but lose the ability to explain individual coefficient effects.
- [ ] You gain speed but lose the ability to handle categorical features.
- [ ] You gain robustness to outliers but lose the ability to use cross-validation.

### In the nested CV for XGBoost, the inner loop used 3 folds and the outer loop used 5 folds with 8 hyperparameter combinations. How many inner-loop model fits occurred?
- [ ] 40, because 5 outer folds times 8 hyperparameter combinations equals 40.
- [x] 120, because 5 outer folds each run a full inner search of 8 combinations times 3 inner folds.
- [ ] 15, because 5 outer folds times 3 inner folds equals 15 total evaluations.
- [ ] 24, because 8 hyperparameter combinations times 3 inner folds equals 24.

### Central Air achieved 7.86% feature importance despite being a binary feature. Why might a binary feature score highly?
- [ ] Binary features are always more important than continuous features in tree models.
- [x] The presence or absence of central air is a major price differentiator in the Ames market.
- [ ] Binary features are used more frequently as split points because they are easier to compute.
- [ ] The model automatically upweights binary features to compensate for their simplicity.

### Feature importance tells you which features a tree ensemble relied on. What does it NOT tell you?
- [ ] Which features the model used most for its splits
- [x] The direction and shape of a feature's effect on the prediction
- [ ] How the importance is shared out across features (it sums to 1)
- [ ] That Overall Qual mattered most in this model

### Decision trees convert categorical features to numeric codes using .cat.codes. What risk does this introduce?
- [x] The model treats categories as having an ordinal relationship that may not exist.
- [ ] The model loses all information about the categorical feature during conversion.
- [ ] The model becomes unable to handle new categories that appear in the test set.
- [ ] The model requires one-hot encoding to work correctly with numeric codes.

### The notebook's final model comparison shows Decision Tree (0.82) < Random Forest (0.88) < XGBoost (0.94) ≈ HGB (0.94). What does this progression illustrate?
- [ ] That more complex models always produce better results regardless of the dataset.
- [x] That each ensemble method builds on the previous one's weaknesses to improve accuracy.
- [ ] That the differences between models are negligible and any tree model would suffice.
- [ ] That the hyperparameter grids were progressively larger, giving later models an unfair advantage.
