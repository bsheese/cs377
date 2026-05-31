# 17_0 Regression Preliminaries — Topic Outline

This document provides a complete outline of all topics covered across the five notebooks in the 17_0 Regression Preliminaries series.

---

## 17_0_1: Univariate Spread

**Topics:** Descriptive statistics, the mean as a model, Total Sum of Squares (TSS), variance, and standard deviation.

### Before We Begin
- Understanding what a "model" means in the context of regression
- The simplest model: predict the mean for every observation

### The Simplest Model (The Mean)
- Scenario: guessing the weight of a randomly selected avocado
- The mean as our first and simplest predictor
- Why the mean minimizes absolute error but not squared error

### Measuring the Error of the Mean
- Calculating deviations from the mean: $y_i - \bar{y}$
- Why deviations sum to zero (the cancellation problem)
- The intuition: positive and negative errors cancel out

### Total Sum of Squares (TSS)
- Why we square the errors instead of summing raw deviations
- Squaring removes cancellation and penalizes large errors more heavily
- TSS formula: $\sum(y_i - \bar{y})^2$
- Student exercise: implement TSS from scratch

### Variance and Standard Deviation
- Variance: dividing TSS by N — average squared distance from the mean (sample estimates use N−1, *Bessel's correction*; see the notebook aside)
- Standard deviation: square root of variance — returns to original units
- Relationship: standard deviation is the typical distance of a point from the mean

### Why This Matters for Regression
- TSS measures total variability in the target
- Regression tries to explain some of that variability
- Future notebooks compare explained vs. unexplained variability

---

## 17_0_2: Visualizing Associations

**Topics:** Scatterplots, Anscombe's Quartet, outliers vs. high-leverage points, the Datasaurus Dozen.

### From 1D to 2D — The Scatterplot
- Moving from univariate to bivariate analysis
- The scatterplot as the fundamental visualization tool
- X-axis (predictor) vs. Y-axis (target)

### Anscombe's Trap
- Anscombe's Quartet: four datasets with nearly identical summary statistics
- Mean, variance, correlation, and regression line are almost the same
- The trap: identical numbers, completely different visual stories
- The lesson: always visualize your data before computing statistics

### Outliers vs. High-Leverage Points
- **Outlier:** a point with an unusual y-value given its x-value (large residual)
- **High leverage:** a point with an extreme x-value that can strongly influence the regression line
- Dataset III: a genuine outlier with moderate leverage
- Dataset IV: a high-leverage point that distorts the regression slope
- Physics analogy: leverage as a fulcrum — points far from the center have more rotational force
- The core principle: extreme x-values allow points to "pull" the regression line toward themselves

### The Datasaurus Dozen
- Extension of Anscombe's idea: twelve datasets with identical summary statistics
- Visual shapes: dinosaur, circle, star, lines, etc.
- Reinforces the same lesson: summary statistics alone are insufficient

### Where We're Going Next
- Scatterplots reveal whether a linear relationship exists
- Next: quantifying the strength of that relationship with covariance and correlation

---

## 17_0_3: Covariance and Correlation

**Topics:** Quadrant method, covariance, Pearson's r, unit interpretation.

### The Quadrant Trick
- Visual pattern: data points in the top-right and bottom-left quadrants support a positive relationship
- Points in top-left and bottom-right support a negative relationship
- The "aha" moment: counting which quadrants have more points reveals direction of association

### Covariance
- Formula: $\frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{N}$ (sample estimates use $N-1$; `np.cov` defaults to $N-1$)
- Positive covariance: variables move in the same direction
- Negative covariance: variables move in opposite directions
- Covariance has a problem: dependent on the units of measurement
  - Height in inches vs. height in feet produces different covariance values
  - Cannot compare covariance across different measurement scales

### Pearson's $r$ (Correlation)
- Solution: divide covariance by the product of standard deviations
- Formula: $r = \frac{\text{cov}(x,y)}{s_x s_y}$
- $r$ is unitless and ranges from −1 to +1
- $r = +1$: perfect positive linear relationship
- $r = -1$: perfect negative linear relationship
- $r = 0$: no linear relationship (but may have non-linear relationship)
- Student exercise: implement Pearson's $r$ from scratch
- Important warning: $r$ measures only **linear** association

### Where We're Going Next
- Correlation quantifies association strength
- Next: what happens when we draw a line through the data (residuals)

---

## 17_0_4: Residuals

**Topics:** Predicted values ($\hat{y}$), drawing a regression line, residuals, the trap of summing errors.

### Drawing an (Arbitrary) Line
- $y$ vs. $\hat{y}$: actual observed value vs. model-predicted value
- The guessing game: picking a line by sight (slope and intercept)
- A line that is reasonable but not optimal

### The Residual
- Definition: $e_i = y_i - \hat{y}_i$ (actual minus predicted)
- Positive residual: model underpredicts (point is above the line)
- Negative residual: model overpredicts (point is below the line)
- Zero residual: perfect prediction (point is exactly on the line)

### The Trap of "Total Error"
- Tempting approach: sum all residuals to measure total error
- Problem: positive and negative residuals cancel each other
- Same issue as deviations from the mean in Notebook 1
- Leads to the need for **RSS** (Residual Sum of Squares)

### Where We're Going Next
- Squaring residuals solves the cancellation problem
- Next: Residual Sum of Squares and $R^2$

---

## 17_0_5: RSS and $R^2$

**Topics:** Residual Sum of Squares (RSS), $R^2$ interpretation, the shortcut formula for simple linear regression.

### Residual Sum of Squares (RSS)
- Formula: $\sum(y_i - \hat{y}_i)^2$
- Squaring solves the cancellation problem from Notebook 4
- The U-shaped bowl: plotting RSS against slope values for a fixed intercept
- Visual intuition: the bottom of the bowl is the best slope
- "The optimal line is the one that minimizes RSS"

### The True Meaning of $R^2$
- Comparing RSS to TSS: how much variability remains after accounting for the model
- TSS = total variability in $y$ (from Notebook 1)
- RSS = variability left over after the model
- Formula: $R^2 = 1 - \frac{\text{RSS}}{\text{TSS}}$
- Interpretation:
  - $R^2 = 0$: model explains none of the variance (RSS = TSS)
  - $R^2 = 1$: model explains all variance (RSS = 0)
  - $R^2 = 0.75$: model explains 75% of the variance
- Intuition: the fraction of TSS that the model "soaks up"

### The Shortcut — Simple Linear Regression in One Line
- No need for a brute-force search over slopes
- Formula for the optimal slope: $\hat{\beta}_1 = r \cdot \frac{s_y}{s_x}$
- Formula for the optimal intercept: $\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$
- Does the formula actually hit the bottom of the bowl? Yes.
- Bonus: for simple linear regression, $R^2 = r^2$
  - Connection back to Notebook 3's correlation coefficient
  - If $|r|$ is large, $R^2$ is large; if $|r|$ is small, $R^2$ is small

### Where We Go From Here
- Simple linear regression fundamentals are complete
- Next topics: Multiple Linear Regression, cross-validation, regularization

---

## Cross-Cutting Themes

| Theme | Notebooks |
|---|---|
| **TSS, RSS, $R^2$ progression** | 17_0_1 (TSS), 17_0_5 (RSS, $R^2$) |
| **Always visualize first** | 17_0_2 (Anscombe, Datasaurus) |
| **Squared errors solve cancellation** | 17_0_1 (TSS), 17_0_4 (residual trap), 17_0_5 (RSS) |
| **Correlation and regression connection** | 17_0_3 ($r$), 17_0_5 ($R^2 = r^2$) |
| **Mean as a baseline model** | 17_0_1 (TSS vs. mean), 17_0_5 (TSS vs. RSS) |
| **Leverage and influence** | 17_0_2 (outliers vs. leverage) |
