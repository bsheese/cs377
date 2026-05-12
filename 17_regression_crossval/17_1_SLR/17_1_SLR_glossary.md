# 17_1 Simple Linear Regression — Glossary

This document defines all technical and conceptual terms used across the six notebooks in the 17_1 Simple Linear Regression series.

---

## A

### Adjusted $R^2$
A modified version of $R^2$ that penalizes the addition of unnecessary predictors. Will only increase if a new feature improves the model more than expected by chance. Used in Multiple Linear Regression to compare models with different numbers of variables.

---

## B

### Bias
Error introduced by overly simplistic assumptions in a model. High bias causes underfitting — the model fails to capture the underlying patterns. In the bias-variance tradeoff, simple models have high bias but low variance.

### Bias–Variance Tradeoff
The fundamental tension: reducing bias (by making the model more complex) typically increases variance (sensitivity to training data noise), and vice versa. The best model is the simplest one that still fits the data well.

### Box-Cox Transformation
A family of power transformations parameterized by lambda ($\lambda$). Includes log ($\lambda = 0$), square root ($\lambda = 0.5$), reciprocal ($\lambda = -1$), and no transformation ($\lambda = 1$) as special cases. Automatically finds the best transformation to make data more normal.

---

## C

### Confidence Interval (95%)
A range of plausible values for a population parameter (e.g., the true slope). Calculated as $\hat{\beta}_1 \pm t^* \cdot SE(\hat{\beta}_1)$. If the interval does not contain zero, the result is statistically significant at the $\alpha = 0.05$ level. Does **not** mean there is a 95% probability the true value lies in that interval — it means 95% of similarly constructed intervals would contain the true value.

### Cook's Distance
A measure of how much the regression coefficients change when a single observation is removed. A value greater than 1 is commonly flagged for investigation. Formula accounts for both the residual size and the leverage of the point. Calculated in Python with `.get_influence().cooks_distance`.

---

## D

### Data Leakage
When information from outside the training set influences the model during fitting or evaluation, leading to overly optimistic performance estimates. The train/test split prevents this by keeping the test set completely hidden during training.

### Durbin–Watson Statistic
A test statistic for autocorrelation in residuals. Values near 2 indicate independence (no autocorrelation). Values near 0 indicate positive autocorrelation; values near 4 indicate negative autocorrelation. Relevant for the **I** (Independence) assumption of linear regression.

### Drop Test
Manually removing a high-influence point and refitting the model to see how much the coefficients change. Used alongside Cook's Distance to assess practical impact of influential observations.

---

## E

### Elasticity
The percentage change in y for a 1% change in x. In a log-log model ($\log y \sim \log x$), the slope coefficient is directly interpreted as elasticity.

---

## H

### Heteroscedasticity
A violation of the **E** (Equal Variance) assumption where the spread of residuals changes across the range of fitted values. Often appears as a fan or funnel shape in the residuals vs. fitted plot. Can be addressed with transformations or robust standard errors.

### Homoscedasticity
The assumption that the variance of residuals is constant across all levels of the fitted values. When true, the residuals vs. fitted plot shows a random scatter with constant spread.

---

## I

### Influence
The degree to which a single observation changes the regression coefficients when removed. A function of both **leverage** (extremeness of x) and the **residual** (deviation from the line). A point must have both to be influential. See **Cook's Distance**.

### Intercept ($\beta_0$)
The predicted value of y when all predictors are zero. In statsmodels, must be added explicitly with `sm.add_constant()`. In scikit-learn, included by default.

---

## L

### Leverage
A measure of how far a data point's x-value is from the mean of x. Points with extreme x-values have high leverage — they have more "rotational force" to influence the regression slope. Distinct from **outlier** (unusual y-value) and **influence** (actual change to coefficients).

### LINE Assumptions
The four assumptions of linear regression:
- **L**inearity: the relationship between predictors and target is linear
- **I**ndependence: observations are independent
- **N**ormality: residuals are normally distributed (for valid inference)
- **E**qual Variance (Homoscedasticity): constant spread of residuals

### Log-Linear Model
A model where the target is log-transformed ($\log y \sim x$). The slope is interpreted as: a one-unit increase in x is associated with a $100 \times \hat{\beta}_1$ percent change in y.

### Linear-Log Model
A model where the predictor is log-transformed ($y \sim \log x$). The slope is interpreted as: a 1% increase in x is associated with a $\hat{\beta}_1 / 100$ unit change in y.

### Log-Log Model
A model where both target and predictor are log-transformed ($\log y \sim \log x$). The slope is interpreted as elasticity: a 1% increase in x is associated with a $\hat{\beta}_1$ percent change in y.

---

## N

### Null Hypothesis ($H_0$)
The assumption that there is no relationship between the predictor and the target (the true slope is zero). Statistical significance testing asks: if $H_0$ is true, how likely is it that random data would produce a slope at least as extreme as the one we observed?

---

## O

### Overfitting
When a model learns the noise and specificities of the training data rather than the underlying pattern. Detected by a large gap between training performance and test performance. In the polynomial example, training $R^2$ rises steadily while test $R^2$ eventually falls.

---

## P

### P-Value
The probability of observing a test statistic (e.g., slope) at least as extreme as the one obtained, assuming the null hypothesis is true. A p-value less than 0.05 is conventionally considered "statistically significant." Does not measure the probability that the null hypothesis is true, nor the practical importance of the result.

### Polynomial Features
Creating powers of a predictor ($x^2$, $x^3$, etc.) to allow linear regression to model curves. Adding polynomial features of increasing degree demonstrates overfitting: training $R^2$ approaches 1.0 while test $R^2$ eventually decreases.

---

## Q

### Q-Q Plot (Quantile-Quantile Plot)
A diagnostic plot for checking the normality assumption. Plots the quantiles of the residuals against the quantiles of a normal distribution. Points following the diagonal line indicate normally distributed residuals. S-curves or systematic deviations indicate skew or heavy tails.

---

## R

### $R^2$ (R-Squared)
The proportion of variance in the target explained by the model. Can be negative on test data — this happens when the model performs worse than simply predicting the mean.

### Robust Standard Errors
Standard errors that are corrected for heteroscedasticity. Produce valid confidence intervals and p-values even when the equal variance assumption is violated. An alternative to transforming the data.

---

## S

### Scikit-learn (`sklearn`)
A Python machine learning library focused on prediction. API: `.fit()` trains, `.predict()` predicts. Requires 2D input arrays. Preferred for building prediction pipelines. Less informative about statistical inference than statsmodels.

### Slope ($\beta_1$)
The change in the predicted y for a one-unit increase in x. Estimated by ordinary least squares. Statistical significance is assessed via its standard error, p-value, and confidence interval.

### Standard Error of the Slope ($SE(\hat{\beta}_1)$)
A measure of how much the estimated slope would vary across different samples from the same population. Calculated from the residual standard deviation and the spread of x. A small standard error means the slope is estimated precisely.

### Statsmodels
A Python library focused on statistical inference. API: formula-based `ols()` or array-based with `sm.add_constant()`. `.summary()` produces a detailed table with coefficients, standard errors, p-values, confidence intervals, $R^2$, F-statistic, and more.

---

## T

### Train/Test Split
Dividing the dataset into a training set (used to fit the model) and a test set (used to evaluate generalization). Prevents data leakage and provides an honest estimate of real-world performance. Typically 70–80% train, 20–30% test.

### Transformation
Applying a mathematical function to a variable (log, square root, reciprocal, etc.) to achieve linearity, stabilize variance, or reduce skew. Linear regression requires linearity in the parameters, not in the variables — transformations bend the data, not the line.

---

## V

### Variance
Error introduced by a model's sensitivity to the specific training data. High variance causes overfitting — small changes in training data produce large changes in predictions. Complex models have low bias but high variance.

---

## W

### White Test
A statistical test for heteroscedasticity. Tests whether the variance of residuals is related to the fitted values or predictors. A significant result indicates heteroscedasticity, violating the **E** assumption.

---

## X

### X-shape ($x^2$)
See **Polynomial Features**.
