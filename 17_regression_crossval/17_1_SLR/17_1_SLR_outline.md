# 17_1 Simple Linear Regression — Topic Outline

This document provides a complete outline of all topics covered across the six notebooks in the 17_1 Simple Linear Regression series.

**Data by notebook:**
- **17_1_1:** Palmer Penguins (flipper length vs. body mass)
- **17_1_2:** Palmer Penguins (flipper length vs. body mass)
- **17_1_3:** Auto MPG (displacement vs. mpg)
- **17_1_4:** Ames Housing (area vs. price), with synthetic poisoned row
- **17_1_5:** Gapminder 2007 (GDP vs. life expectancy); Auto MPG revisited
- **17_1_6:** Ames Housing (baseline); synthetic sin(x) + noise (overfitting demo)

---

## 17_1_1: The Two Paradigms (Scikit-Learn vs. Statsmodels)

**Topics:** Fitting a regression line in sklearn and statsmodels, understanding the two different philosophies, verifying they produce the same result.

### What This Notebook Is About
- Two Python libraries, two philosophies, one math
- Scikit-learn: prediction-focused "fit/predict" API
- Statsmodels: inference-focused "summary table" API
- How to choose between them (prediction vs. inference)

### Paradigm A — Scikit-Learn (The Predictor)
- `LinearRegression` from `sklearn.linear_model`
- The #1 beginner error: passing a 1D Series instead of a 2D array (`.reshape(-1, 1)`)
- `.fit()` and `.predict()` workflow
- Accessing coefficients: `.coef_`, `.intercept_`

### Paradigm B — Statsmodels (The Explainer)
- `OLS` from `statsmodels.api`
- The statsmodels quirk: you must add the intercept yourself using `sm.add_constant()`
- `.fit()` and `.summary()` workflow
- Reading the summary table: coefficients, standard errors, p-values, $R^2$
- Argument order: `OLS(y, X)` vs. sklearn's `fit(X, y)` — and why

### The Convergence
- Both libraries produce identical coefficients and $R^2$
- Closed-form formula from 17_0_5 matches both
- The difference is in the interface and what they report, not in the math

### Where We're Going Next
- The summary table includes p-values — what do they mean?

---

## 17_1_2: Statistical Significance (Did We Just Get Lucky?)

**Topics:** Null hypothesis, standard error of the slope, t-statistic, p-values, 95% confidence intervals — all built from simulation.

### The Null Hypothesis
- $H_0$: the true slope is zero (no relationship between x and y)
- If the null were true, what's the chance we'd see a slope this extreme?

### The Standard Error — How Much Does the Slope Wiggle?
- Bootstrap: resampling pairs with replacement to simulate repeat samples
- The standard error is the standard deviation of the bootstrap sampling distribution

### The t-Statistic — Bridging SE and P-Value
- $t = \hat{\beta}_1 / SE(\hat{\beta}_1)$
- Confirmed against the summary table value

### The P-Value — Simulating the Null World
- Permutation test: shuffling y to destroy the x-y relationship
- Distinction between bootstrap (preserves relationship) and permutation (breaks it)
- Empirical p-value vs. theoretical p-value
- The 0.05 convention (and its discontents)
- Statistical significance vs. practical significance

### The 95% Confidence Interval
- Bootstrap percentile interval
- Formula-based CI: $\hat{\beta}_1 \pm t^* \cdot SE(\hat{\beta}_1)$
- Both methods match statsmodels
- Correct interpretation of a CI (demonstrated via coverage simulation)
- Bootstrap percentile method caveat: works best for symmetric distributions

### How Sample Size Affects Everything
- Larger $n$ shrinks the standard error of the slope, narrowing confidence intervals and inflating the t-statistic
- The same true effect looks "more significant" (smaller p-value) as $n$ grows
- Reinforces the distinction between statistical significance and practical significance

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
- Checking with `sns.residplot(x=x, y=y, lowess=True)`
- U-shape or curve in residuals = violation
- No pattern = linearity is plausible

### I — Independence
- Most commonly violated with time series data
- The Durbin–Watson statistic: values near 2 indicate independence
- Rough critical values for single predictor (~1.7 and ~2.3); zone of indecision
- DW of 0.926 in MPG data is an artifact of row ordering, not real autocorrelation

### N — Normality of Residuals
- Checking with a Q-Q plot (`sm.qqplot`)
- Reading a Q-Q plot: points on the diagonal = normal; curves = heavy tails or skew
- A Q-Q plot is more informative than a histogram for assessing normality
- Central Limit Theorem makes inference robust to mild non-normality with large n

### E — Equal Variance (Homoscedasticity)
- Checking with residuals vs. fitted scatterplot
- Funnel shape = heteroscedasticity (unequal variance)
- Synthetic example isolating a pure funnel (no non-linearity)
- Applying a log transform to reduce the funnel
- Quantitative comparison of residual spread at low vs. high fitted values

### What Do You Do When Assumptions Fail?
- Transform variables (log, reciprocal, square root)
- Use robust standard errors
- Consider non-linear models

### Where We're Going Next
- Before transforming, check if a few data points are driving the violations

---

## 17_1_4: Influence, Leverage, and Cook's Distance

**Topics:** Distinguishing leverage from outliers from influence, Cook's Distance, the drop test, robust regression as an alternative, ethics of dropping data.

### The Needle in the Haystack
- A single data point can dominate the regression line
- Visual inspection fails with large datasets

### Leverage ≠ Outlier ≠ Influence
- **Leverage:** extreme x-value (potential to influence)
- **Outlier:** unusual y-value given x (large residual)
- **Influence:** actually changes the coefficients when removed
- A point must have both high leverage and a large residual to be influential
- High leverage alone can still inflate standard errors

### Hunting in Ames with `.get_influence()`
- Leverage: `influence.hat_matrix_diag`
- Studentized residuals: `influence.resid_studentized_external`
- Cook's Distance: `influence.cooks_distance`
- Common heuristic: $D_i > 4/n$ flags investigation (not a hard cutoff)

### The Ethics of Dropping Data
- Cook's Distance is a detective, not an executioner
- Legitimate reasons to drop: data entry errors, wrong units, instrument failure, different population
- Illegitimate reasons: "it was hurting my $R^2$" or "it was flagged by Cook's"
- What to do with genuinely influential points you can't drop:
  - Report both models (with and without)
  - Use robust regression (Huber regression demonstrated)
  - Model what's actually going on (separate models, extra features)
- Ethics discussion comes **before** the drop test mechanics

### The Drop Test
- Manually drop the high-influence point and refit
- Compare coefficients, standard errors, and $R^2$ before and after
- Huber regression as a third option between "keep" and "drop"

### Where We're Going Next
- If assumptions fail due to non-linearity, we can transform the variables

---

## 17_1_5: Transformations (Bending the Line)

**Topics:** Log transformations, why log works (diminishing returns and the derivative of log), log-linear interpretation, Box-Cox, fixing a failing model.

### The Failing Model
- Gapminder GDP vs. life expectancy: curved scatterplot, U-shaped residuals
- Two out of four LINE assumptions broken: L and E

### Why Log?
- Diminishing returns → the slope decreases as $x$ increases
- $d/dx \log(x) = 1/x$: log has the same property (steep when small, flat when large)
- This is why log is the natural choice, not a guess

### Bend the Data, Not the Line
- Linear regression requires linearity in the parameters, not in the variables
- Transform $x$ or $y$ or both to achieve linearity
- Most common: log transformation

### Interpreting a Log-Linear Slope
- Calculus version: $dy = \beta_1 \cdot dx/x$
- Algebra version (no calculus needed): $\beta_1 \cdot \log(1.01) \approx \beta_1 / 100$
- A 1% increase in $x \to +\beta_1/100$ unit change in $y$
- Three flavors: level-log, log-level, log-log
- Back-transforming predictions to the original scale: Jensen's correction

### Box-Cox — Let the Data Choose
- Automatically finds the best power transformation $\lambda$
- Special cases: $\lambda = 0$ (log), $\lambda = 0.5$ (sqrt), $\lambda = -1$ (reciprocal)
- Applied to Gapminder GDP ($\lambda \approx 0$, confirming log) and MPG displacement ($\lambda \approx -0.3$, close to reciprocal)

### Fixing the MPG Model from 17_1_3
- Physics suggests $1/\text{mpg} \propto \text{displacement}$, so $\text{mpg} \approx a + b / \text{displacement}$
- Verify: $1/\text{mpg}$ is linear in displacement
- Reciprocal transform eliminates the U-shaped residuals
- The right metric for a successful transformation is the residual plot, not the $R^2$

### A Few Honest Warnings
- Transformations change the question you're answering
- Interpretability decreases as transformation complexity increases
- Over-transforming can hide rather than fix problems
- $R^2$ values across different $y$-scales are not directly comparable

### Where We're Going Next
- Does the model perform well on data it hasn't seen before?

---

## 17_1_6: The Generalization Test (Train/Test Split)

**Topics:** Train/test split, why test performance differs from training (sampling variation), overfitting, the bias-variance tradeoff.

### The Cheating Problem
- Using the same data to fit and evaluate overstates performance
- The model "cheats" by optimizing for the data it already saw

### The Train/Test Split
- `train_test_split` from `sklearn.model_selection`
- Arguments: `X, y, test_size, random_state`
- The golden rule: test data is off-limits for fitting and model selection
- (Data quality inspection is still allowed)

### Fit on Train, Score on Both
- Simple model (Ames: 2 parameters, 2930 houses): train and test $R^2$ are essentially equal
- Multiple random splits show the gap is centered near zero with small variance
- **Simple models generalize well** — this is the baseline
- Test $R^2$ can be slightly higher than train $R^2$ due to sampling variation

### Making Overfitting Visible
- Synthetic sin(x) data with polynomial regression of increasing degree
- What are polynomial features? Design matrix visualization
- Degree 1: underfitting (high bias)
- Degree 3: Goldilocks zone
- Degree 10–20: overfitting (high variance)
- Training $R^2$ climbs to 1.0; test $R^2$ peaks then collapses
- $R^2$ can be negative on test data (model does worse than the mean)

### The Bias–Variance Tradeoff
- Formula: $\text{Test Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$
- Empirical decomposition across degrees 1–12
- Bias falls, variance rises, their sum is U-shaped
- The minimum of the sum corresponds to the best test $R^2$

### End of the SLR Arc
- From fitting a line through significance, assumptions, influence, transformations, and generalization
- All these ideas scale directly to Multiple Linear Regression
- Next: cross-validation and regularization

---

