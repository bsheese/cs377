# 17.1 Simple Linear Regression - Course Outline

Based on *Introduction to Modern Statistics (OpenIntro)* Chapter 7.

# Chapter 7: Linear Regression with a Single Predictor

**Source:** *Introduction to Modern Statistics (2e)* — OpenIntro  
**URL:** https://openintro-ims.netlify.app/model-slr

---

## 7.1 Fitting a Line, Residuals, and Correlation

### 7.1.1 Fitting a Line to Data

- **Perfect vs. imperfect linear relationships**
  - Perfect linear relationship: knowing $x$ gives exact $y$ (e.g., stock purchase cost = price × quantity) — rare in natural processes
  - Imperfect linear relationships: data appear as a "cloud of points" around a line
  - Non-linear relationships: a straight line is not helpful even when variables are clearly related (e.g., quadratic/parabolic patterns)

- **The linear regression model equation**
  - $y = b_0 + b_1 x + e$
  - $b_0$ = intercept (sample statistic), estimates population parameter $\beta_0$
  - $b_1$ = slope (sample statistic), estimates population parameter $\beta_1$
  - $e$ = error term (often dropped when writing the model for prediction)
  - **Predictor variable** ($x$): the variable used to make predictions (a.k.a. explanatory, independent variable, feature)
  - **Outcome** ($y$): the variable being predicted (a.k.a. response, dependent variable)

- **Strength of linear trends**
  - Strong: data tightly cluster around a line, small remaining variability
  - Moderate: clear trend but noticeable spread around the line
  - Weak: slight trend, difficult to discern from random scatter

### 7.1.2 Using Linear Regression to Predict Possum Head Lengths

- **Case study: Brushtail possums** (n = 104, Australia)
  - Predictor: total length (cm, head to tail)
  - Outcome: head length (mm)
  - Moderately strong positive linear association

- **Fitting a line by eye**
  - Example equation: $\hat{y} = 41 + 0.59x$
  - The "hat" ($\hat{y}$) signifies an *estimated/predicted* value
  - Predictions represent *average* outcomes for a given $x$ value
  - Example: a possum with 80 cm total length → predicted head length = 88.2 mm

- **Adding a third variable for context**
  - Coloring points by a third variable (e.g., sex, age) reveals whether the relationship differs across groups
  - Male possums tend to be larger in both total length and head length than females
  - Sets the stage for multiple regression (Chapter 8)

### 7.1.3 Residuals

- **Definition of a residual**
  - Residual = observed outcome − predicted outcome
  - $e_i = y_i - \hat{y}_i$
  - Data = Fit + Residual (decomposition of each observation)

- **Interpreting residuals**
  - Positive residual: observation is *above* the regression line (model underpredicted)
  - Negative residual: observation is *below* the regression line (model overpredicted)
  - Residual size discussed in terms of absolute value
  - Goal: residuals should be as small as possible

- **Residual plots**
  - Horizontal axis: predicted $\hat{y}$ values (or $x$ values)
  - Vertical axis: residual values ($e_i$)
  - Conceptually: "tipping the scatterplot over so the regression line is horizontal"
  - Dashed reference line at residual = 0

- **Using residual plots to assess model appropriateness**
  - **No pattern** (random scatter around 0): linear model is appropriate
  - **Curved pattern in residuals**: the true relationship is non-linear; a straight line is not appropriate; consider variable transformations
  - **No trend and no pattern**: linear model is reasonable, but it may be unclear whether the slope differs from zero (requires inference, Chapter 24)

### 7.1.4 Describing Linear Relationships with Correlation

- **Correlation ($r$)**: measures the strength and *direction* of the *linear* relationship between two variables
  - Always between −1 and +1
  - $r = +1$: perfect positive linear relationship
  - $r = -1$: perfect negative linear relationship
  - $r \approx 0$: no apparent linear relationship
  - Strong positive: $r$ near +1; strong negative: $r$ near −1

- **Formula for correlation**
  - $r = \frac{1}{n-1} \sum_{i=1}^{n} \frac{x_i - \bar{x}}{s_x} \cdot \frac{y_i - \bar{y}}{s_y}$
  - Uses standardized (z-score) versions of both variables

- **Key properties of correlation**
  - **Unitless**: not affected by linear changes in units (e.g., kg → lbs, cm → inches)
  - Only measures *linear* relationships — strong non-linear relationships (quadratic, cyclic) can produce correlations near zero
  - Does not distinguish between predictor and outcome (symmetric)

- **Visual interpretation**
  - Scatterplots with labeled $r$ values ranging from −1 to +1
  - Non-linear patterns (parabola, sine wave) with weak correlations despite strong relationships

---

## 7.2 Least Squares Regression

### 7.2.1 Gift Aid for First-Year Students at Elmhurst College

- **Case study**: random sample of 50 first-year students
  - Predictor: family income (in $1,000s)
  - Outcome: gift aid (financial aid that does not need to be repaid, in $1,000s)
  - Negative trend: higher family income → lower gift aid

### 7.2.2 An Objective Measure for Finding the Best Line

- **Two criteria for "best" line**
  1. Minimize sum of absolute residuals: $|e_1| + |e_2| + \dots + |e_n|$
  2. Minimize sum of squared residuals: $e_1^2 + e_2^2 + \dots + e_n^2$ ← **least squares**

- **Four reasons to prefer least squares**
  1. Most commonly used method (tradition/convenience)
  2. Widely supported in statistical software
  3. A residual twice as large is *more than* twice as bad — squaring penalizes larger errors more heavily
  4. Analyses linking the model to population inference are most straightforward with least squares

### 7.2.3 Finding and Interpreting the Least Squares Line

- **Model equation**: $\widehat{\text{aid}} = \beta_0 + \beta_1 \times \text{family\_income}$
  - $\beta_0$ and $\beta_1$ are population parameters
  - $b_0$ and $b_1$ are their sample-based point estimates

- **Software output** (regression table)
  - Columns: term, estimate, std.error, statistic, p.value
  - For Elmhurst data: $b_0 = 24.319$, $b_1 = -0.043$

- **Interpreting the slope ($b_1$)**
  - For each additional $1,000 of family income, expected aid decreases by $43.10 on average
  - Association ≠ causation (observational data)

- **Interpreting the intercept ($b_0$)**
  - Average aid for a student with $0 family income = $24,319
  - Meaningful here because $x = 0$ exists in the data
  - In many applications, the intercept has no practical meaning if $x = 0$ is not observed or relevant

- **General interpretation template**
  - **Slope**: estimated difference in the predicted average outcome of $y$ for a one-unit increase in $x$
  - **Intercept**: average outcome of $y$ when $x = 0$, *if* the model is valid at $x = 0$

- **Computing the least squares line from summary statistics (by hand)**
  - Two key properties:
    1. $b_1 = \frac{s_y}{s_x} \cdot r$ (slope from correlation and standard deviations)
    2. The point $(\bar{x}, \bar{y})$ always falls on the least squares line
  - Using point-slope form: $y - \bar{y} = b_1(x - \bar{x})$
  - Simplifies to: $b_0 = \bar{y} - b_1 \bar{x}$

- **Predictions as estimates, not certainties**
  - Model provides an imperfect estimate — individual outcomes vary around the line
  - Data from one cohort may not generalize to other years

### 7.2.4 Extrapolation Is Treacherous

- **Extrapolation**: applying a model estimate to values *outside* the range of the original data
- Example: predicting aid for a family with $1M income yields −$18,800 (nonsensical)
- Linear models are approximations — the relationship may not hold beyond observed data
- **Warning**: extrapolation is an unreliable bet that the linear trend continues

### 7.2.5 Describing the Strength of a Fit

- **R-squared ($R^2$)**, a.k.a. **coefficient of determination**
  - Describes the proportion of variation in the outcome variable explained by the least squares line
  - Always between 0 and 1
  - For SLR: $R^2 = r^2$ (square of the correlation)

- **Computing $R^2$ from variances**
  - Total variance of outcome: $s_y^2$
  - Residual variance after model: $s_{RES}^2$
  - Proportion of variation explained: $\frac{s_y^2 - s_{RES}^2}{s_y^2}$

- **Sums of squares framework**
  - **Total Sum of Squares (SST)**: $\sum (y_i - \bar{y})^2$ — total variability in $y$ around its mean
  - **Sum of Squared Errors (SSE)**: $\sum (y_i - \hat{y}_i)^2 = \sum e_i^2$ — leftover variability after the model
  - **$R^2 = \frac{SST - SSE}{SST} = 1 - \frac{SSE}{SST}$**

- **Interpretation example**: $R^2 = 0.25$ means 25% of the variation in gift aid is explained by family income

### 7.2.6 Categorical Predictors with Two Levels

- **Case study**: eBay auctions of Mario Kart (Nintendo Wii)
  - Predictor: game condition (used vs. new) — categorical with two levels
  - Outcome: total auction price ($)

- **Indicator (dummy) variable**
  - `condnew` = 1 if new, 0 if used
  - Model: $\widehat{\text{price}} = b_0 + b_1 \times \text{condnew}$

- **Interpreting parameters for binary predictors**
  - **Intercept ($b_0$)**: average outcome for the reference category (condnew = 0, i.e., used games)
    - Example: average price of a used game = $42.87
  - **Slope ($b_1$)**: average *difference* in outcome between the two categories
    - Example: new games sell for $10.90 more than used games, on average

- **Key insight**: the intercept and slope interpretations do not fundamentally change; with a binary predictor, the coefficients are directly interpretable as group means and mean differences

---

## 7.3 Outliers in Linear Regression

- **Outliers in regression**: observations that fall far from the main cloud of points in the bivariate ($x$, $y$) space
  - Being outlying in $x$ alone or $y$ alone does *not* make a point influential
  - What matters is whether the point is outlying *relative to the bivariate model*

- **Types of unusual points**
  - **Outlier**: a point (or group of points) that stands out from the rest of the data
  - **Leverage point**: an outlier that falls horizontally far from the center of the cloud (extreme $x$ value)
    - These points "pull harder" on the regression line
  - **Influential point**: a leverage point that actually changes the slope of the least squares line
    - A point is influential if, had the line been fitted without it, the point would be unusually far from the line

- **Six illustrative scenarios**
  - Outlier in $y$ direction only: slightly influences the line
  - Outlier in $x$ and $y$ but *on* the trend line: not influential (not an outlier of the bivariate model)
  - Outlier in $x$ and $y$ *off* the trend line: pulls the line, influential
  - Secondary cloud of outliers: can distort the fit across the entire range
  - Single outlier creating a trend where none exists: problematically controls the slope
  - Outlier far away but *on* the line: high leverage but not influential

- **Best practice for handling outliers**
  - Do **not** remove outliers without a very good reason
  - Produce **two analyses**: one with and one without the outlying observations
  - Present both and discuss the role of the outliers for a holistic understanding
  - Exceptional cases are often interesting and informative (e.g., financial market crashes)

---

## 7.4 Chapter Review

### 7.4.1 Summary

- Linear models can use numerical predictors (e.g., possum total length) and categorical predictors with two levels (e.g., game condition)
- Residuals are essential for evaluating model fit
- High leverage points and influential points can substantially impact the least squares line
- Correlation measures strength and direction of linear association without specifying predictor vs. outcome
- Future chapters generalize from sample estimates to population inference

### 7.4.2 Key Terms

| Term | Definition |
|---|---|
| **Coefficient of determination** ($R^2$) | Proportion of variation in $y$ explained by the linear model |
| **Correlation** ($r$) | Strength and direction of the linear relationship between two variables (−1 to +1) |
| **Extrapolation** | Applying a model estimate outside the range of the original data |
| **High leverage** | Points horizontally far from the center of the data cloud |
| **Indicator variable** | A binary (0/1) variable used to represent a two-level categorical predictor |
| **Influential point** | An outlier that changes the slope of the least squares line |
| **Least squares line** | The line that minimizes the sum of squared residuals |
| **Leverage point** | An outlier with an extreme $x$ value |
| **Outcome** | The variable being predicted ($y$, dependent variable, response) |
| **Outlier** | A point that stands out from the rest of the data |
| **Predictor** | The variable used to make predictions ($x$, independent variable, explanatory) |
| **R-squared** | Same as coefficient of determination |
| **Regression sum of squares** | Variability explained by the model (SST − SSE) |
| **Residuals** | Differences between observed and predicted values ($e_i = y_i - \hat{y}_i$) |
| **Sum of squared error** (SSE) | $\sum e_i^2$, leftover variability after the model |
| **Total sum of squares** (SST) | $\sum (y_i - \bar{y})^2$, total variability in the outcome |

---

## 7.5 Exercises (Topics Covered)

The chapter exercises address the following skills and concepts:

1. **Visualizing residuals** — sketching/describing what residual plots look like given scatterplots with fitted lines
2. **Trends in residuals** — identifying patterns (random vs. curved) and determining whether a linear model is appropriate
3. **Identifying relationships** — assessing strength (weak/moderate/strong) and linearity from scatterplots
4. **Midterms and final** — comparing correlations between two predictors and a common outcome; reasoning about why one correlation is stronger
5. **Meat consumption and life expectancy** — describing relationships, comparing correlation strength, understanding unit invariance of $r$
6. **Matching correlation to scatterplot** — associating numerical $r$ values with visual patterns
7. **Body measurements (correlation)** — describing relationships and the effect of unit changes on $r$
8. **Comparing correlations across unit systems** — recognizing that $r$ is unchanged by linear unit transformations
9. **Urban homeowners' income and age** — interpreting a regression equation, slope, intercept, and $R^2$
10. **GDP and poverty rate** — computing slope and intercept from summary statistics, interpreting parameters, making predictions
11. **Tourist spending** — interpreting regression output, computing $R^2$ from $r$, predicting values
12. **Catching errors** — identifying impossible or inconsistent regression results (e.g., $R^2 > 1$, sign mismatches between $r$ and slope)
13. **Interpreting $R^2$ in context** — explaining what a given $R^2$ value means in a real-world scenario
14. **Climatologists and cricket chirps** — building a regression model from summary statistics, making predictions, assessing extrapolation
15. **Distance and age of drivers** — interpreting slope and intercept, assessing practical meaning of intercept
16. **Fitting a regression by eye** — understanding the subjectivity of visual line-fitting vs. least squares
17. **Estimating equations from scatterplots** — reading approximate slope and intercept from a graph
18. **Marine iguanas** — interpreting regression output in a biological context
19. **Computing $R^2$ from SST and SSE** — applying the formula $R^2 = 1 - SSE/SST$
20. **Residual calculation** — computing individual residuals from observed and predicted values
21. **Outlier identification** — classifying points as outliers, leverage points, and/or influential points
22. **Effect of removing influential points** — predicting how the regression line changes when an influential point is removed
