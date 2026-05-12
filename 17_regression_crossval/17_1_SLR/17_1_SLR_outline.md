# 17_1 Simple Linear Regression — Topic Outline

This document provides a complete outline of all topics covered across the six notebooks in the 17_1 Simple Linear Regression series.

**Data:** Automobile fuel efficiency (MPG) dataset throughout; Ames Housing introduced in Notebook 4 for influence analysis.

---

## 17_1_1: The Two Paradigms (Scikit-Learn vs. Statsmodels)

**Topics:** Fitting a regression line in sklearn and statsmodels, understanding the two different philosophies, verifying they produce the same result.

### What This Notebook Is About
- Two Python libraries, two philosophies, one math
- Scikit-learn: prediction-focused "fit/predict" API
- Statsmodels: inference-focused "summary table" API

### Paradigm A — Scikit-Learn (The Predictor)
- `LinearRegression` from `sklearn.linear_model`
- The #1 beginner error: passing a Series instead of a DataFrame (2D requirement)
- `.fit()` and `.predict()` workflow
- Accessing coefficients: `.coef_`, `.intercept_`

### Paradigm B — Statsmodels (The Explainer)
- `ols` from `statsmodels.formula.api`
- The statsmodels quirk: you must add the intercept yourself using `sm.add_constant()`
- `.fit()` and `.summary()` workflow
- Reading the summary table: coefficients, standard errors, p-values, $R^2$

### The Convergence
- Both libraries produce identical coefficients and $R^2$
- Scikit-learn is preferred for prediction pipelines
- Statsmodels is preferred for inference and diagnostics

### Where We're Going Next
- The summary table includes p-values — what do they mean?

---

## 17_1_2: Statistical Significance (Did We Just Get Lucky?)

**Topics:** Null hypothesis, standard error of the slope, p-values, 95% confidence intervals.

### The Null Hypothesis
- $H_0$: the true slope is zero (no relationship between x and y)
- If the null were true, what's the chance we'd see a slope this extreme?

### The Standard Error — How Much Does the Slope Wiggle?
- Simulating 2000 parallel universes: resampling to see how much the slope varies
- The standard error is the standard deviation of the sampling distribution of the slope
- A small standard error → precise estimate; a large one → noisy estimate

### The P-Value — Simulating the Null World
- If the true slope were zero, how often would random data produce a slope at least as extreme as ours?
- The 0.05 convention (and its discontents)
- p < 0.05: "statistically significant" — but not necessarily practically significant

### The 95% Confidence Interval
- Range of plausible values for the true slope
- Formula: $\hat{\beta}_1 \pm t^* \cdot SE(\hat{\beta}_1)$
- If the CI doesn't include zero, the result is statistically significant
- One careful word about interpretation: not a "95% chance the true value is in here"

### Where We're Going Next
- Significance relies on assumptions — next we check those assumptions

---

## 17_1_3: The LINE Assumptions (Diagnosing the Model)

**Topics:** The four LINE assumptions, diagnostic plots, Durbin-Watson test, Q-Q plots, what to do when assumptions fail.

### The LINE Assumptions
- **L**inearity: the relationship between x and y is linear
- **I**ndependence: observations are independent of each other
- **N**ormality: residuals are normally distributed (for inference)
- **E**qual Variance (Homoscedasticity): residual spread is constant across fitted values

### L — Linearity
- Checking with a residuals vs. fitted plot
- Patterns (curves, fans) indicate violations
- No pattern = linearity is plausible

### I — Independence
- Most commonly violated with time series data
- The Durbin–Watson statistic: values near 2 indicate independence
- Values near 0 indicate positive autocorrelation; near 4 indicate negative

### N — Normality of Residuals
- Checking with a histogram and a Q-Q plot
- Reading a Q-Q plot in 30 seconds: points along the diagonal = normal
- S-shaped curves indicate skew or heavy tails
- Normality matters for p-values and confidence intervals, not for coefficient estimates

### E — Equal Variance (Homoscedasticity)
- Checking with a residuals vs. fitted plot
- Funnel shape indicates heteroscedasticity (unequal variance)
- What a failure looks like: residuals spread wider as fitted values increase

### What Do You Do When Assumptions Fail?
- Transform the variables (log, square root, etc.)
- Use robust standard errors
- Consider non-linear models

### Where We're Going Next
- Before transforming, check if a few data points are driving the violations

---

## 17_1_4: Influence, Leverage, and Cook's Distance

**Topics:** Distinguishing leverage from outliers from influence, Cook's Distance, the drop test, ethics of dropping data.

### The Needle in the Haystack
- A single data point can dominate the regression line
- Need tools to identify such points systematically

### Leverage ≠ Outlier ≠ Influence
- **Leverage:** extreme x-value (potential to influence)
- **Outlier:** unusual y-value given x (large residual)
- **Influence:** actually changes the coefficients when removed
- A point can have high leverage but zero influence (if it follows the trend)

### Hunting in Ames with `.get_influence()`
- Cook's Distance in one sentence: "how much does the line move if I delete this point?"
- Cook's D > 1 is a common threshold for concern
- Finding the suspect programmatically with `.get_influence().cooks_distance`

### The Drop Test
- Manually drop the high-influence point and refit
- Compare coefficients before and after
- How much did the slope and intercept change?

### The Ethics of Dropping Data
- Never drop data just to get a "better" result
- Legitimate reasons to drop: data entry errors, equipment malfunction, outside the domain of interest
- What to do with genuinely influential points you can't legitimately drop:
  - Report both models (with and without)
  - Use robust regression methods
  - Collect more data in the influential region
  - Transform the variable

### Where We're Going Next
- If assumptions fail due to non-linearity, we can transform the variables

---

## 17_1_5: Transformations (Bending the Line)

**Topics:** Log transformations, log-linear interpretation, the transformation toolkit, fixing a failing model.

### The Failing Model
- A model that violates LINE assumptions
- Residuals show a clear pattern: heteroscedasticity, non-linearity, or both

### Bend the Data, Not the Line
- Linear regression requires linearity in the parameters, not in the variables
- Transform x, y, or both to achieve linearity
- Most common: log transformation

### Interpreting a Log-Linear Slope
- The math in one line: a 1% change in x is associated with a $(\hat{\beta}_1 / 100)$ unit change in y
- The vocabulary: "elasticity" — the percentage change in y for a 1% change in x
- Three flavors of log transformation:
  - **Log-Linear:** log(y) ~ x (interpret: % change in y per unit x)
  - **Linear-Log:** y ~ log(x) (interpret: change in y per % change in x)
  - **Log-Log:** log(y) ~ log(x) (interpret: % change in y per % change in x — elasticity)

### The Transformation Toolkit
- Log, square root, reciprocal, Box-Cox, polynomial
- When to use each: skew, variance, curvature patterns

### Promise Kept — Fixing the MPG Model from 17_1_3
- Applying a log transformation to the problematic variable
- Rechecking the LINE assumptions: residuals now well-behaved
- Before-and-after comparison

### A Few Honest Warnings
- Transformations change the question you're answering
- Interpretability decreases as transformation complexity increases
- Over-transforming can hide rather than fix problems

### Where We're Going Next
- Does the model perform well on data it hasn't seen before?

---

## 17_1_6: The Generalization Test (Train/Test Split)

**Topics:** Train/test split, preventing data leakage, overfitting, the bias-variance tradeoff.

### The Cheating Problem
- Using the same data to fit and evaluate overstates performance
- The model "cheats" by memorizing the training data

### The Train/Test Split
- `train_test_split` from `sklearn.model_selection`
- Hold out a portion (typically 20–30%) of the data
- Fit on train, evaluate on test
- Test performance is the honest estimate of generalization

### Fit on Train, Score on Both
- Training $R^2$ is typically higher than test $R^2$
- Gap between train and test = overfitting signal
- "But if it's the same in train and test, what's the big deal?" — it shouldn't be the same, the test should be worse

### Making Overfitting Visible
- Adding polynomial features of increasing degree
- Training $R^2$ steadily rises toward 1.0
- Test $R^2$ rises then falls (the U-shaped generalization curve)
- Wait — $R^2$ can be negative? Yes — the model can do worse than the mean

### The Bias–Variance Tradeoff in One Page
- **Bias:** error from oversimplifying (underfitting)
- **Variance:** error from being too sensitive to training data (overfitting)
- Simple models: high bias, low variance
- Complex models: low bias, high variance
- A one-sentence rule: "The best model is the simplest one that still fits the data well."

### End of the SLR Arc
- From starting with a single variable to checking significance, assumptions, influence, transformations, and generalization
- All these ideas scale directly to Multiple Linear Regression

---

