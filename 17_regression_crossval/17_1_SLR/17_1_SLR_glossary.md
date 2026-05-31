# 17_1 Simple Linear Regression — Glossary

This document defines all technical and conceptual terms used across the six notebooks in the 17_1 Simple Linear Regression series.

---

## A

### `add_constant`
A statsmodels function that prepends a column of 1's to the feature matrix so the model estimates an intercept. Required because `sm.OLS` does not include an intercept by default. Contrast with sklearn's `LinearRegression`, which includes one automatically.

---

## B

### Bias
Error introduced by overly simplistic assumptions in a model. High bias causes underfitting — the model fails to capture the underlying patterns. In the bias-variance tradeoff, simple models have high bias but low variance.

### Bias–Variance Tradeoff
The decomposition of expected test error into three components:

$$\text{Test Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

Reducing bias (by making the model more complex) typically increases variance, and vice versa. The best model is at the complexity level where their sum is minimized.

### Bootstrap
A resampling technique: draw $N$ observations *with replacement* from a dataset of size $N$, then compute the statistic of interest on the resampled data. Repeating this many times produces an empirical sampling distribution. Used in 17_1_2 to estimate the standard error of the slope.

### Box-Cox Transformation
A family of power transformations parameterized by $\lambda$:

$$y^{(\lambda)} = \begin{cases} \frac{y^\lambda - 1}{\lambda} & \lambda \neq 0 \\ \log y & \lambda = 0 \end{cases}$$

Special cases: $\lambda = 0$ is log, $\lambda = 0.5$ is a scaled square root, $\lambda = -1$ is reciprocal, $\lambda = 1$ is no transformation. Automatically finds the $\lambda$ that makes data most normal.

---

## C

### Central Limit Theorem (CLT)
The result that the sampling distribution of a sample mean (or sum) approaches a normal distribution as the sample size grows, regardless of the population's shape. In regression it is why inference (p-values, confidence intervals) stays approximately valid even when residuals are only mildly non-normal, provided $n$ is reasonably large (a rough rule of thumb is $n > 30$). Relevant to the **N** (Normality) assumption.

### Confidence Interval (95%)
A range of plausible values for a population parameter (e.g., the true slope). Calculated as $\hat{\beta}_1 \pm t^* \cdot SE(\hat{\beta}_1)$ or via bootstrap percentiles. If the interval does not contain zero, the result is statistically significant at $\alpha = 0.05$.

**Correct interpretation:** If we repeated the experiment many times, about 95% of similarly constructed intervals would contain the true slope. Does **not** mean there is a 95% probability the true value lies in the interval.

### Convergence (of sklearn and statsmodels)
The fact that `LinearRegression().fit()` and `sm.OLS().fit()` produce identical coefficients and $R^2$ because both minimize the same RSS objective. They differ in API and output, not in math.

### Cook's Distance ($D_i$)
A measure of how much all fitted values would change if observation $i$ were deleted. Combines leverage and residual size. Computed in Python with `model.get_influence().cooks_distance`. A common heuristic flags $D_i > 4/n$ for investigation.

---

## D

### Durbin–Watson Statistic
A test for autocorrelation in residuals (ordered as the data arrive). Values near 2 suggest independence; values below roughly 1.7 suggest positive autocorrelation; values above roughly 2.3 suggest negative autocorrelation. Critical values depend on sample size and number of predictors. Relevant for the **I** (Independence) assumption.

---

## E

### Elasticity
The percentage change in $y$ for a 1% change in $x$. In a log-log model ($\log y \sim \log x$), the slope is directly interpreted as elasticity.

### Empirical p-value
A p-value estimated by simulation rather than from a theoretical distribution. Count how many shuffled/null statistics are at least as extreme as the observed value, then divide by the number of simulations. Resolution is limited by the number of simulations (e.g., 5000 shuffles can only detect $p \ge 1/5000$).

---

## H

### Heteroscedasticity
A violation of the **E** (Equal Variance) assumption: the spread of residuals changes across fitted values. Often appears as a fan or funnel shape in the residuals vs. fitted plot. Can be addressed with transformations or robust standard errors.

### Homoscedasticity
The assumption that residual variance is constant across all fitted values. When true, the residuals vs. fitted plot shows random scatter with constant spread.

### Huber Regression
A robust regression method that down-weights observations with large residuals, making it less sensitive to outliers than ordinary least squares. Available in sklearn as `HuberRegressor`. An alternative to manually dropping influential points.

---

## I

### Influence
The degree to which removing a single observation changes the regression coefficients. A function of both **leverage** (extremeness of $x$) and the **residual** (deviation from the line). A point must have both to be influential. Measured by **Cook's Distance**.

### Intercept ($\beta_0$)
The predicted value of $y$ when all predictors are zero. Often not interpretable directly (e.g., a penguin with zero-length flipper). In statsmodels, must be added with `sm.add_constant()`. In scikit-learn, included by default.

---

## J

### Jarque–Bera Test
A statistical test for the normality of residuals, based on their skewness and kurtosis. The null hypothesis is that the residuals are normally distributed, so a *small* p-value is evidence *against* normality. Reported in the statsmodels `.summary()` output and read alongside the Q-Q plot to assess the **N** (Normality) assumption. Like any normality test, it can flag trivial departures in large samples, so it complements the plot rather than replacing it.

### Jensen's Correction
An adjustment applied when back-transforming predictions from a log model to the original scale. Because the mean of a log is not the log of the mean (a consequence of Jensen's inequality), naively exponentiating a log-scale prediction ($e^{\hat{y}}$) systematically *underestimates* the mean on the original scale. Adding a correction term (related to the residual variance) compensates. Relevant whenever you fit a log–level or log–log model and need predictions in original units.

---

## L

### Leverage ($h_{ii}$)
A measure of how far a data point's $x$ value is from the mean of $x$. Points with extreme $x$ have high leverage — more potential to influence the regression slope. Distinct from **outlier** (unusual $y$) and **influence** (actual change to coefficients). Computed as the diagonal of the hat matrix via `influence.hat_matrix_diag`.

### LINE Assumptions
The four assumptions of linear regression:
- **L**inearity: the relationship between predictors and target is linear
- **I**ndependence: observations are independent of each other
- **N**ormality of residuals: residuals are normally distributed (needed for valid inference)
- **E**qual Variance (Homoscedasticity): constant spread of residuals across fitted values

### Log Transformations (three flavors)
| Model | Formula | Interpretation |
|---|---|---|
| **Level–log** ($\log x$ only) | $y = \beta_0 + \beta_1 \log x$ | +1% in $x \to +\beta_1/100$ units of $y$ |
| **Log–level** ($\log y$ only) | $\log y = \beta_0 + \beta_1 x$ | +1 unit of $x \to +100\beta_1\%$ change in $y$ |
| **Log–log** (both) | $\log y = \beta_0 + \beta_1 \log x$ | +1% in $x \to +\beta_1\%$ change in $y$ (elasticity) |

### Lowess Smoother
A locally weighted scatterplot smoother overlaid on residual plots to highlight patterns (curves, funnels). Used in `sns.residplot(..., lowess=True)`.

---

## N

### Null Hypothesis ($H_0$)
The assumption that there is no relationship between predictor and target (the true slope is zero). Significance testing asks: if $H_0$ is true, how likely would random data produce a slope at least as extreme as the one observed?

---

## O

### Overfitting
When a model learns noise and idiosyncrasies of the training data rather than the underlying pattern. Detected by a large gap between training and test performance. In polynomial regression, training $R^2$ rises steadily while test $R^2$ peaks then falls (sometimes to negative values). See also **Bias–Variance Tradeoff**.

### Outlier
An observation with an unusual $y$ value given its $x$ (a large residual). Not the same as **leverage** or **influence**. A point must have both high leverage and a large residual to be influential.

---

## P

### P-Value
The probability of observing a test statistic (e.g., slope) at least as extreme as the one obtained, assuming the null hypothesis is true. A p-value less than 0.05 is conventionally called "statistically significant." Does **not** measure the probability that $H_0$ is true, nor the practical importance of the result.

### Permutation Test
A resampling technique for generating the null distribution: randomly shuffle the $y$ values (breaking any $x$-$y$ relationship), refit the model, and record the slope. Repeating many times shows what slopes would look like if $H_0$ were true. Used in 17_1_2 to estimate p-values.

### Polynomial Features
Powers of a predictor ($x^2, x^3, \dots, x^d$) added as extra columns to allow linear regression to model curves. Creating polynomial features of increasing degree demonstrates overfitting: training $R^2$ approaches 1.0 while test $R^2$ eventually falls.

---

## Q

### Q-Q Plot (Quantile-Quantile Plot)
A diagnostic plot for the normality assumption. Plots sorted residuals against the quantiles of a normal distribution. Points on the diagonal = normal residuals; curves off the line = heavy tails or skew.

---

## R

### $R^2$ (R-Squared)
The proportion of variance in the target explained by the model: $1 - \text{RSS}/\text{TSS}$. Can be negative on test data — this happens when the model performs worse than predicting the mean.

### Residual
The difference between an observed value and its predicted value: $e_i = y_i - \hat{y}_i$.

### Residual Plot
A scatterplot of residuals ($y$-axis) vs. fitted values ($x$-axis). The primary diagnostic tool for the **L** (linearity) and **E** (equal variance) assumptions. A horizontal band of noise = assumptions hold; curves or funnels = violations.

### `reshape(-1, 1)`
A NumPy operation that converts a 1D array into a 2D column vector. Required by scikit-learn because its API expects a 2D feature matrix even for a single predictor. Using a 1D Series or array produces a `ValueError`.

### Robust Standard Errors
Standard errors corrected for heteroscedasticity, producing valid confidence intervals and p-values even when the equal variance assumption is violated. An alternative to transforming the data.

### Robust Regression
A family of regression methods (e.g., Huber regression, RANSAC) that are less sensitive to outliers than OLS. Useful when influential points cannot be legitimately dropped.

---

## S

### Scikit-learn (`sklearn`)
A Python library focused on **prediction**. API: `.fit(X, y)` trains, `.predict(X)` predicts. Requires `X` to be 2D (use `.reshape(-1, 1)` for a single feature). Preferred for building prediction pipelines and model comparison. Less informative about inference than statsmodels.

### Slope ($\beta_1$)
The change in predicted $y$ for a one-unit increase in $x$. Estimated by ordinary least squares (minimizing RSS). Statistical significance is assessed via its standard error, t-statistic, p-value, and confidence interval.

### Standard Error of the Slope ($SE(\hat{\beta}_1)$)
The estimated standard deviation of the slope across hypothetical repeat samples. A small standard error means the slope is estimated precisely. Can be estimated via the bootstrap (resampling pairs) or from the classical formula.

### Statsmodels (`sm`)
A Python library focused on **inference**. API: `sm.OLS(y, X).fit()` with `(y, X)` argument order (response first). Requires an intercept column added manually via `sm.add_constant()`. The `.summary()` method produces a detailed table with coefficients, standard errors, t-statistics, p-values, confidence intervals, and diagnostics.

### Studentized Residual
A residual divided by its estimated standard deviation, expressed in t-statistic units. Values with absolute magnitude greater than ~3 are considered unusual. Computed via `influence.resid_studentized_external`. The term "studentized" refers to "Student" (W.S. Gosset), the discoverer of the t-distribution.

---

## T

### t-Statistic
The slope divided by its standard error: $t = \hat{\beta}_1 / SE(\hat{\beta}_1)$. Measures how many standard errors the estimate is from zero. A large |t| (e.g., 32.6 in the penguins example) indicates strong evidence against the null hypothesis.

### Train/Test Split
Dividing the dataset into a training set (used to fit the model) and a test set (used to evaluate generalization). Prevents data leakage and provides an honest estimate of real-world performance. Convention: `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`. Typical split: 70–80% train, 20–30% test.

### Transformation
Applying a mathematical function to a variable (log, reciprocal, square root, Box-Cox) to achieve linearity, stabilize variance, or reduce skew. Linear regression requires linearity *in the parameters*, not in the variables — transformations bend the data, not the line.

---

## U

### Underfitting
When a model is too simple to capture the real structure of the data. Both training and test performance are poor. The fix: increase model complexity (add features, increase polynomial degree, etc.). Opposite of **overfitting**.

---

## V

### Variance (in bias-variance tradeoff)
Error introduced by a model's sensitivity to the specific training data. High variance causes overfitting — small changes in the training data produce large changes in predictions. Complex models have low bias but high variance.
