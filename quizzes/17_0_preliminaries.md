# 17.0 · Regression Preliminaries

## 17_0_1: Spread & TSS

### Why does the sum of raw deviations from the mean always equal zero?
- [ ] Deviations are undefined for values below the mean
- [x] Positive and negative deviations cancel because the mean is the balance point
- [ ] The mean is always zero, so all deviations are also zero
- [ ] Raw deviations are computed as absolute values and therefore always positive

### Why is squaring deviations preferred over taking their absolute values?
- [ ] Squaring is computationally faster than the absolute value operation
- [x] Squaring penalizes large errors more heavily and produces a differentiable function
- [ ] Absolute value fails to prevent the cancellation problem for symmetric data
- [ ] Squared deviations remain in the same units as the original data

### TSS for penguin body mass is computed in g². Why is standard deviation a better summary of spread?
- [ ] Standard deviation is always smaller than variance for real datasets
- [x] Taking the square root of variance returns the spread to the original units
- [ ] Variance cannot be compared across datasets; standard deviation always can
- [ ] TSS grows with sample size; standard deviation is normalized by sample size

### The mean is described as the 'baseline prediction that minimizes squared error.' What does this mean?
- [x] Predicting the mean for every observation minimizes the sum of squared residuals
- [ ] The mean equals zero when its own squared value is computed
- [ ] Baseline predictions are set equal to the sample mean by mathematical definition
- [ ] The mean minimizes prediction error only for normally distributed target variables

### If a dataset has TSS = 0, what must be true?
- [x] All values in the dataset are equal to each other
- [ ] The dataset contains exactly one observation with no variance
- [ ] The mean of the dataset is exactly equal to zero
- [ ] The standard deviation is undefined because of division by zero

## 17_0_2: Visualizing Associations

### Anscombe's Quartet has four datasets with nearly identical means, variances, and correlations. What does this demonstrate?
- [ ] Summary statistics fully capture the structure of any real dataset
- [x] Summary statistics can be identical while the underlying patterns are completely different
- [ ] Correlation is the single most important summary statistic for bivariate data
- [ ] All linear models fit equally well when summary statistics are the same

### Anscombe Dataset II is a perfect parabola. Fitting a linear model to it produces R² ≈ 0.67. What is the problem?
- [ ] R² = 0.67 is too low for the model to be considered useful at all
- [x] The linear model misses the curved pattern; predictions are systematically wrong outside the middle range
- [ ] A parabolic relationship requires more than 11 data points to detect properly
- [ ] The model is correct; R² = 0.67 means 67% of variance is explained by the line

### What is the difference between a y-outlier and a high-leverage point?
- [ ] A y-outlier has an unusual x-value; a high-leverage point has an unusual y-value
- [x] A y-outlier has an unusual y-value given x; a high-leverage point has an extreme x-value
- [ ] They describe the same type of observation using different terminology
- [ ] y-outliers affect the slope; high-leverage points only shift the intercept

### Dataset IV in Anscombe's Quartet has all x-values near 8 except one point at x=19. Why is this single point dangerous?
- [ ] It is a y-outlier that inflates the residual sum of squares
- [x] It is a high-leverage point that can pull the slope toward it with little resistance
- [ ] The model ignores observations that fall far from the mean x-value
- [ ] It forces the intercept to equal zero regardless of the other data points

### The Datasaurus Dozen shows 13 datasets — including a dinosaur shape — all with identical summary statistics. What is the key takeaway?
- [ ] Automated ML pipelines should compute summary statistics before visualizing data
- [x] Visualization is essential; summary statistics alone cannot reveal the true data structure
- [ ] The Datasaurus proves that all datasets follow the same underlying distribution
- [ ] Summary statistics are only reliable when computed on more than 100 data points

## 17_0_3: Covariance & Correlation

### In the quadrant trick, why do upper-right points contribute positively to covariance?
- [x] Upper-right points have both x and y above the mean, so (x-x̄)(y-ȳ) = (+)(+) = +
- [ ] Upper-right points have the largest absolute values and dominate the calculation
- [ ] Positive quadrants are defined as any region above the x-axis by convention
- [ ] Covariance counts points per quadrant and weights the upper-right region as positive

### Covariance between flipper length (mm) and mass (g) is 9,796 mm·g. In kg, it becomes 9.796. Is the relationship stronger in kg?
- [ ] Yes — a smaller covariance value indicates a weaker linear relationship
- [x] No — the relationship strength is unchanged; only the units differ
- [ ] Yes — covariance in kg is a more standardized measure than covariance in grams
- [ ] No — covariance values cannot be meaningfully compared across different unit systems

### Pearson's r = 0 for a dataset where y = x². Why doesn't this mean x and y are unrelated?
- [ ] r can only be computed for datasets with more than 30 observations
- [x] r measures only linear association; a perfect quadratic relationship is non-linear
- [ ] y = x² would produce r = -1 because the parabola curves downward
- [ ] r cannot equal exactly 0 for any real-world dataset

### Dividing covariance by the product of standard deviations (σ_x · σ_y) produces a unitless r. Why does this division cancel units?
- [ ] Dividing by standard deviations converts the data to z-scores in both dimensions
- [ ] Standard deviations are always equal to 1 in normalized datasets
- [x] Multiplying two standard deviations produces the same units as the covariance, canceling them
- [ ] The product σ_x · σ_y equals the variance, which removes all measurement units

### r = 0.87 between flipper length and body mass. Does this mean flipper length causes heavier penguins?
- [ ] Yes — strong correlation indicates a causal relationship between two variables
- [x] No — correlation measures association; causation requires additional experimental evidence
- [ ] Only if the p-value for r is less than 0.05
- [ ] Yes — biological measurements of the same organism are always causally linked

## 17_0_4: Residuals

### A residual is e_i = y_i - ŷ_i. What does a large positive residual indicate?
- [ ] The model overpredicted — it placed the regression line above the actual point
- [x] The model underpredicted — the actual value is well above the fitted line
- [ ] The point is a high-leverage observation that should be removed from the data
- [ ] The prediction is exactly correct for that particular observation

### Any line passing through (x̄, ȳ) has residuals that sum to zero. Why can't you use this sum as a model quality metric?
- [ ] A sum of zero means the model predicted every point exactly
- [x] Many terrible lines also have residuals summing to zero; positive and negative errors cancel
- [ ] Residuals from lines not through (x̄, ȳ) also sum to zero for linear data
- [ ] The sum equals zero only for perfectly linear relationships in the data

### Residuals are measured vertically (in y-units) rather than perpendicularly to the line. Why?
- [ ] Perpendicular distances are undefined for lines with non-integer slope values
- [x] We want to measure prediction error in the target variable's units, not geometric distance
- [ ] Vertical residuals are always smaller and therefore easier to minimize numerically
- [ ] Perpendicular residuals would make the resulting regression coefficients uninterpretable

### A residual plot shows a U-shaped curve (not random scatter). What does this tell you about the model?
- [ ] The model has high variance but low bias overall
- [x] The model systematically misses a non-linear pattern in the data
- [ ] The residuals are normally distributed, which is a desirable diagnostic sign
- [ ] There are influential outliers that should be removed before refitting the model

## 17_0_5: OLS & R-squared

### Plotting RSS vs. candidate slopes produces a U-shaped bowl with one minimum. Why exactly one minimum?
- [x] RSS is a convex quadratic function of the slope, guaranteeing a single global minimum
- [ ] The dataset is too small in size to produce multiple local minima
- [ ] Grid search converges to one answer regardless of the underlying landscape shape
- [ ] The minimum is guaranteed only when the regression line passes through the origin

### The optimal slope formula is m = r · (σ_y / σ_x). If r = 0 (no correlation), what is the optimal slope?
- [x] 0 — the best prediction is the mean of y regardless of x
- [ ] σ_y / σ_x — the ratio of standard deviations becomes the slope when r = 0
- [ ] Undefined — the slope formula breaks down when correlation equals zero
- [ ] 1 — a unit slope is the safest default when the correlation is unknown

### R² = 0.50 for the Ames Housing model. In plain English, what does this mean?
- [ ] The model predicts house prices with 50% accuracy on the test set
- [x] The model explains 50% of the variance in price that existed before fitting
- [ ] 50% of predictions fall within one standard deviation of the true value
- [ ] The model's slope captures exactly half of the true underlying relationship

### R² is defined as 1 - RSS/TSS. What does R² = 0 tell you about the model?
- [ ] RSS = 0, meaning the model perfectly fits all data points
- [x] RSS = TSS, meaning the model performs no better than predicting the mean
- [ ] TSS = 0, meaning all y-values in the dataset are identical
- [ ] The model has failed to converge and needs to be retrained

### For simple linear regression, R² = r². If r = 0.70, what is R²?
- [ ] 0.70 — the correlation and R² are equal for any linear model
- [x] 0.49 — R² equals the squared correlation
- [ ] 0.35 — R² equals half the correlation for bivariate regression
- [ ] 1.40 — R² equals twice the correlation in simple linear regression

### Can R² ever be negative? What would a negative R² mean?
- [ ] No — R² is bounded between 0 and 1 by mathematical definition
- [x] Yes — the model performs worse than simply predicting the mean on test data
- [ ] Yes — it indicates that the regression slope points in the wrong direction
- [ ] No — RSS can never exceed TSS for a model that has been properly fitted
