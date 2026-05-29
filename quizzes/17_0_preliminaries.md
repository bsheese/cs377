# 17.0 · Regression Preliminaries

## 17_0_1: Spread & TSS

### Why does the sum of raw deviations from the mean always equal zero?
- [ ] Deviations are undefined for values below the mean
- [x] Positive and negative deviations cancel because the mean is the balance point of the data
- [ ] The mean is always zero, so all deviations are also zero
- [ ] Raw deviations are always positive when computed correctly

### Why is squaring deviations preferred over taking their absolute values?
- [ ] Squaring is computationally faster than absolute value
- [x] Squaring penalizes large errors more heavily and produces a differentiable function
- [ ] Absolute value cannot prevent the cancellation problem
- [ ] Squared deviations are always in the same units as the data

### TSS for penguin body mass is computed in g². Why is standard deviation a better summary of spread?
- [ ] Standard deviation is always smaller than variance
- [x] Taking the square root of variance returns the spread to the original units
- [ ] Variance cannot be compared across datasets; standard deviation can
- [ ] TSS grows with sample size; standard deviation does not

### The mean is described as the 'baseline prediction that minimizes squared error.' What does this mean?
- [x] Predicting the mean for every observation minimizes the sum of squared residuals
- [ ] The mean is the only statistic that equals zero when squared
- [ ] Baseline predictions are always equal to the sample mean by definition
- [ ] The mean minimizes prediction error only for normally distributed data

### If a dataset has TSS = 0, what must be true?
- [x] All values in the dataset are equal to each other
- [ ] The dataset has exactly one observation
- [ ] The mean of the dataset is exactly zero
- [ ] The standard deviation is undefined because of division by zero

## 17_0_2: Visualizing Associations

### Anscombe's Quartet has four datasets with nearly identical means, variances, and correlations. What does this demonstrate?
- [ ] Summary statistics fully capture the structure of any dataset
- [x] Summary statistics can be identical while the underlying patterns are completely different
- [ ] Correlation is the most important summary statistic for bivariate data
- [ ] All linear regression models are equally accurate for the same summary statistics

### Anscombe Dataset II is a perfect parabola. Fitting a linear model to it produces R² ≈ 0.67. What is the problem?
- [ ] R² = 0.67 is too low for any model to be useful
- [x] The linear model misses the curved pattern; predictions will be systematically wrong outside the middle range
- [ ] A parabolic relationship cannot be detected without more than 11 data points
- [ ] The model is correct; R² = 0.67 means 67% of variance is explained by the line

### What is the difference between a y-outlier and a high-leverage point?
- [ ] A y-outlier has an unusual x-value; a high-leverage point has an unusual y-value
- [x] A y-outlier has an unusual y-value given x; a high-leverage point has an extreme x-value
- [ ] They describe the same type of point using different terminology
- [ ] y-outliers affect the slope; high-leverage points only affect the intercept

### Dataset IV in Anscombe's Quartet has all x-values near 8 except one point at x=19. Why is this single point dangerous?
- [ ] It is a y-outlier that inflates the residual sum of squares
- [x] It is a high-leverage point that can pull the slope toward it with little resistance
- [ ] The model ignores points far from the mean x-value
- [ ] It causes the intercept to equal zero regardless of other data

### The Datasaurus Dozen shows 13 datasets — including a dinosaur shape — all with identical summary statistics. What is the key takeaway?
- [ ] Automated ML pipelines should always compute summary statistics before visualizing
- [x] Visualization is essential; summary statistics alone cannot reveal the true data structure
- [ ] The Datasaurus proves that all datasets follow the same underlying distribution
- [ ] Summary statistics are only reliable when computed on more than 100 data points

## 17_0_3: Covariance & Correlation

### In the quadrant trick, why do upper-right points contribute positively to covariance?
- [x] Upper-right points have both x and y above the mean, so (x-x̄)(y-ȳ) = (+)(+) = +
- [ ] Upper-right points have the largest absolute values and dominate the calculation
- [ ] Positive quadrants are defined as any region above the x-axis
- [ ] Covariance counts the number of points per quadrant, weighting upper-right as positive

### Covariance between flipper length (mm) and mass (g) is 9,796 mm·g. In kg, it becomes 9.796. Is the relationship stronger in kg?
- [ ] Yes — a smaller covariance value indicates a weaker relationship
- [x] No — the relationship strength is unchanged; only the units differ
- [ ] Yes — covariance in kg is always more accurate than in grams
- [ ] No — covariance cannot be meaningfully compared across unit systems

### Pearson's r = 0 for a dataset where y = x². Why doesn't this mean x and y are unrelated?
- [ ] r can only be computed for datasets with more than 30 observations
- [x] r measures only linear association; a perfect quadratic relationship is non-linear
- [ ] y = x² would produce r = -1 because the curve is concave
- [ ] r cannot equal exactly 0 for any real dataset

### Dividing covariance by the product of standard deviations (σ_x · σ_y) produces a unitless r. Why does this division cancel units?
- [ ] Dividing by standard deviations converts the data to z-scores in both dimensions
- [ ] Standard deviations are always equal to 1 in normalized data
- [x] Multiplying two standard deviations produces the same units as the covariance, canceling them
- [ ] The product σ_x · σ_y equals the variance, which removes all units

### r = 0.87 between flipper length and body mass. Does this mean flipper length causes heavier penguins?
- [ ] Yes — strong correlation always indicates a causal relationship
- [x] No — correlation measures association; causation requires additional experimental evidence
- [ ] Only if the p-value for r is less than 0.05
- [ ] Yes — biological measurements are always causal

## 17_0_4: Residuals

### A residual is e_i = y_i - ŷ_i. What does a large positive residual indicate?
- [ ] The model overpredicted — it placed the line above the actual point
- [x] The model underpredicted — the actual value is well above the line
- [ ] The point is a high-leverage observation that should be removed
- [ ] The prediction is exactly correct for that observation

### Any line passing through (x̄, ȳ) has residuals that sum to zero. Why can't you use this sum as a model quality metric?
- [ ] A sum of zero means the model predicted every point exactly
- [x] Many terrible lines also have residuals summing to zero; positive and negative errors cancel
- [ ] Residuals from lines not through (x̄, ȳ) always sum to zero too
- [ ] The sum is always zero only for perfectly linear relationships

### Residuals are measured vertically (in y-units) rather than perpendicularly to the line. Why?
- [ ] Perpendicular distances are undefined for lines with non-integer slopes
- [x] We want to measure prediction error in the target variable's units, not geometric distance
- [ ] Vertical residuals are always smaller and easier to minimize
- [ ] Perpendicular residuals would make the resulting coefficients non-interpretable

### A residual plot shows a U-shaped curve (not random scatter). What does this tell you about the model?
- [ ] The model has high variance but low bias
- [x] The model systematically misses a non-linear pattern in the data
- [ ] The residuals are normally distributed, which is desirable
- [ ] There are influential outliers that should be removed before refitting

## 17_0_5: OLS & R-squared

### Plotting RSS vs. candidate slopes produces a U-shaped bowl with one minimum. Why exactly one minimum?
- [x] RSS is a convex quadratic function of the slope, guaranteeing a single global minimum
- [ ] The dataset is too small to produce multiple local minima
- [ ] Grid search always converges to one answer regardless of the landscape shape
- [ ] The minimum is guaranteed only when the data passes through the origin

### The optimal slope formula is m = r · (σ_y / σ_x). If r = 0 (no correlation), what is the optimal slope?
- [x] 0 — the best prediction is always the mean of y, regardless of x
- [ ] σ_y / σ_x — the ratio of standard deviations is the slope
- [ ] Undefined — slope cannot be computed when correlation is zero
- [ ] 1 — a unit slope is the safest default when r is unknown

### R² = 0.50 for the Ames Housing model. In plain English, what does this mean?
- [ ] The model predicts house prices with 50% accuracy
- [x] The model explains 50% of the variance in price that existed before fitting the line
- [ ] 50% of predictions are within one standard deviation of the true value
- [ ] The model's slope is exactly half the true relationship slope

### R² is defined as 1 - RSS/TSS. What does R² = 0 tell you about the model?
- [ ] RSS = 0, meaning the model perfectly fits all data points
- [x] RSS = TSS, meaning the model is no better than predicting the mean
- [ ] TSS = 0, meaning all y-values are identical
- [ ] The model has failed to converge and should be retrained

### For simple linear regression, R² = r². If r = 0.70, what is R²?
- [ ] 0.70 — the correlation and R² are equal for linear models
- [x] 0.49 — R² equals the squared correlation
- [ ] 0.35 — R² equals half the correlation for bivariate data
- [ ] 1.40 — R² is two times the correlation in simple regression

### Can R² ever be negative? What would a negative R² mean?
- [ ] No — R² is bounded between 0 and 1 by mathematical definition
- [x] Yes — the model performs worse than simply predicting the mean on test data
- [ ] Yes — it indicates that the slope is in the wrong direction
- [ ] No — RSS can never exceed TSS for a fitted model
