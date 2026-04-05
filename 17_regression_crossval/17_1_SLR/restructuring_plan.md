# 17_1 SLR Notebook Restructuring Plan

**Goal:** Align the 17_1_SLR Colab notebooks with OpenIntro IMS Chapter 7 (Linear Regression with a Single Predictor), covering every topic and term while preserving the existing hands-on problem-solving format.

---

## Topic Gap Analysis

| Chapter 7 Topic | Current Coverage | Gap |
|---|---|---|
| 7.1.1 Model equation, perfect vs. imperfect, predictor/outcome | 17.1.0 (superficial) | Model form $y = b_0 + b_1x + e$, population vs. sample parameters |
| 7.1.2 Predictions as averages, third-variable context | Not covered | Missing entirely |
| 7.1.3 Residuals, residual plots, model assessment | Mentioned only | No residual plots, no pattern diagnosis |
| 7.1.4 Correlation formula, properties, unitless | Mentioned | Formula not shown, unit invariance not demonstrated |
| 7.2.2 Why least squares, absolute vs. squared residuals | Not covered | Missing entirely |
| 7.2.3 Interpreting slope/intercept, manual computation from summary stats | sklearn output only | No manual $b_1 = (s_y/s_x)r$ computation, no interpretation practice |
| 7.2.4 Extrapolation warnings | One prediction problem | Not explicitly taught or warned against |
| 7.2.5 R², SST, SSE, coefficient of determination | `.score()` only | No SST/SSE decomposition, no R² interpretation |
| 7.2.6 Categorical predictors (indicator/dummy variables) | Not covered | Missing entirely |
| 7.3 Outliers, leverage, influential points | Not covered | Missing entirely |
| 7.4 Key terms | Not covered | Missing |

**Notebook 17.1.5 is broken** — ~50 cells are AI-generated review output that needs to be stripped and replaced.

---

## Proposed Notebook Structure (8 notebooks)

### 17.1.0 — Introduction to SLR (revise existing)

**Keep:**
- Correlation vs. regression comparison
- Scatterplots and Seaborn `lmplot`
- Visualizing uncertainty (confidence bands)
- Warning about spurious trends from random data

**Add:**
- Formal model equation: $y = b_0 + b_1x + e$
- Predictor vs. outcome terminology (and equivalents: feature/target, independent/dependent)
- Population parameters ($\beta_0, \beta_1$) vs. sample statistics ($b_0, b_1$)
- Perfect vs. imperfect vs. non-linear relationships (with visual examples)
- Predictions as *averages*, not certainties
- Third-variable context (coloring by group) as preview of multiple regression

### 17.1.1 — SLR from Scratch (revise existing)

**Keep:**
- All 10 problems (my_mean, my_distances, slope, intercept, prediction, plotting)
- Sugar/sweetness made-up data

**Add:**
- Residual computation after fitting ($e_i = y_i - \hat{y}_i$)
- Residual plot generation
- Computing $R^2$ from scratch using SST and SSE decomposition
- Comparing sum of absolute residuals vs. sum of squared residuals (demonstrate why least squares is preferred)

### 17.1.2 — SLR with Scikit-Learn (revise existing)

**Keep:**
- sklearn LinearRegression workflow (.fit(), .coef_, .intercept_, .predict(), .score())
- Sugar/sweetness demo data
- Boston Housing problem set

**Add:**
- Computing slope/intercept manually from summary statistics ($b_1 = \frac{s_y}{s_x}r$, $b_0 = \bar{y} - b_1\bar{x}$) and verifying against sklearn
- Detailed interpretation of slope and intercept in context (Elmhurst-style example)
- Explicit $R^2$ interpretation as "proportion of variance explained"
- Extracting and plotting residuals from sklearn model
- Extrapolation warning demonstration

### 17.1.3 — Residuals and Model Assessment (NEW)

**Topics:**
- Residual definition: $e_i = y_i - \hat{y}_i$, Data = Fit + Residual
- Computing and plotting residuals (residual vs. predicted, residual vs. $x$)
- Diagnosing residual plots:
  - Random scatter around 0 → linear model appropriate
  - Curved pattern → non-linear relationship, consider transformations
  - No trend, no pattern → linear model reasonable but slope may be zero
- Using residual plots to decide whether a linear model is appropriate

**Hands-on:**
- Fit models to 3 fabricated datasets (linear, curved, no relationship)
- Compute residuals and create residual plots for each
- Diagnose each plot and justify whether SLR is appropriate

### 17.1.4 — Correlation and R-Squared (NEW)

**Topics:**
- Correlation formula: $r = \frac{1}{n-1} \sum \frac{x_i - \bar{x}}{s_x} \cdot \frac{y_i - \bar{y}}{s_y}$
- Computing correlation from scratch
- Visual gallery: scatterplots with $r$ values from −1 to +1
- Correlation is unitless: demonstrate with same data in different units (kg→lbs, cm→inches)
- Correlation only measures linear relationships: show strong non-linear patterns (quadratic, cyclic) with near-zero $r$
- $R^2$ as coefficient of determination: $R^2 = r^2$ for SLR
- SST, SSE decomposition: $SST = \sum(y_i - \bar{y})^2$, $SSE = \sum(y_i - \hat{y}_i)^2$
- $R^2 = \frac{SST - SSE}{SST} = 1 - \frac{SSE}{SST}$
- Interpreting $R^2$ in context ("X% of variation in Y is explained by...")

### 17.1.5 — Visualizing SLR by Category (complete rewrite)

**Strip:** All AI-generated review/output cells (~50 cells of garbage)

**Keep/Rebuild:**
- Seaborn `lmplot` with `hue` for grouped regression lines
- Seaborn `lmplot` with `col` for side-by-side panels
- De-emphasizing scatter points with `alpha`
- Regression-only plots (no scatter) for clean comparison
- Seaborn `jointplot` with marginal distributions

**Add (from §7.2.6):**
- Categorical predictors with two levels — indicator/dummy variables
- Encode binary category as 0/1
- Fit SLR with indicator variable
- Interpret intercept as reference group mean, slope as group difference
- Connect visualization to algebra: the regression line with an indicator predictor is just two horizontal lines (group means)
- Mario Kart eBay auction example or similar

### 17.1.6 — Outliers, Leverage, and Influence (NEW)

**Topics:**
- Definitions:
  - **Outlier**: point that stands out from the rest of the data
  - **Leverage point**: outlier with extreme $x$ value (horizontally far from center)
  - **Influential point**: leverage point that actually changes the slope of the line
- Being outlying in $x$ or $y$ alone does NOT make a point influential
- What matters: is the point outlying relative to the *bivariate* model?

**Six illustrative scenarios (fabricated datasets):**
1. Outlier in $y$ only → slightly influences line
2. Outlier in $x$ and $y$ but *on* the trend → not influential
3. Outlier in $x$ and $y$ *off* the trend → pulls the line, influential
4. Secondary cloud of outliers → distorts fit across entire range
5. Single outlier creating a trend where none exists → problematically controls slope
6. Outlier far away but *on* the line → high leverage but not influential

**Best practices:**
- Don't remove outliers without a very good reason
- Run two analyses: with and without outliers
- Present both and discuss
- Exceptional cases are often interesting and informative

### 17.1.8 — Predicting Startup Profits (revise existing)

**Keep:**
- Data cleaning workflow (duplicates, missing values, datatypes)
- Descriptives, histograms, skew/kurtosis transforms, z-score standardization
- Correlation examination
- SLR loop with visualizations
- Regressions by state

**Add:**
- Residual diagnostics for each model
- $R^2$ interpretation for each model
- Extrapolation warning when making predictions outside data range

### 17.1.9 — Predicting MPG (revise existing)

**Keep:**
- Data retrieval and cleaning (handling `?` missing values)
- SLR analysis of all predictors vs. MPG
- Conclusion writing
- Optional aggregation-by-manufacturer problem

**Add:**
- Residual plots to assess linearity for each predictor
- Identify and discuss outliers/leverage points in the MPG data
- $R^2$ comparison across predictors
- Extrapolation discussion

---

## Summary of Changes

| Notebook | Action | Key Additions |
|---|---|---|
| 17.1.0 | Revise | Model equation, population vs. sample, predictions as averages, third-variable context |
| 17.1.1 | Revise | Residuals, R² from scratch, absolute vs. squared residuals comparison |
| 17.1.2 | Revise | Manual computation from summary stats, interpretation practice, residual extraction, extrapolation warning |
| **17.1.3** | **NEW** | Residual plots, model assessment, pattern diagnosis on 3 fabricated datasets |
| **17.1.4** | **NEW** | Correlation formula, unitless property, R²/SST/SSE decomposition, non-linear correlation failures |
| 17.1.5 | **Rewrite** | Strip AI garbage, rebuild visualization content, add indicator variable / categorical predictor section |
| **17.1.6** | **NEW** | Outliers, leverage, influential points, 6 illustrative scenarios, best practices |
| 17.1.8 | Revise | Residual diagnostics, R² interpretation, extrapolation warning |
| 17.1.9 | Revise | Residual plots, outlier identification, R² comparison, extrapolation discussion |

---

## Key Terms to Integrate Across Notebooks

All 16 terms from §7.4.2 should appear in context at least once:

- coefficient of determination
- correlation
- extrapolation
- high leverage
- indicator variable
- influential point
- least squares line
- leverage point
- outcome
- outlier
- predictor
- R-squared
- regression sum of squares
- residuals
- sum of squared error (SSE)
- total sum of squares (SST)
