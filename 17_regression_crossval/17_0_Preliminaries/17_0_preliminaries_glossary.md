# 17_0 Regression Preliminaries — Glossary

This document defines all technical and conceptual terms used across the five notebooks in the 17_0 Regression Preliminaries series.

---

## A

### Anscombe's Quartet
Four datasets created by Francis Anscombe that have nearly identical summary statistics (mean, variance, correlation, regression line) but dramatically different visual appearances when plotted. Used to demonstrate why data visualization is essential before statistical analysis.

---

## C

### Correlation (Pearson's $r$)
A unitless measure of the strength and direction of a linear relationship between two variables, ranging from −1 (perfect negative) to +1 (perfect positive). Formula: $r = \frac{\text{cov}(x,y)}{s_x s_y}$. Unlike covariance, correlation is independent of measurement scale. Only measures **linear** relationships — a variable can be strongly related to another in a non-linear way and still have $r \approx 0$.

### Correlation vs. Causation
Correlation measures only that two variables move together — it does **not** establish that one causes the other. Two variables can be correlated because of a third (confounding) variable, or by coincidence. Establishing causation takes more than correlation (for example, a controlled experiment or a careful causal argument). The penguins' flipper length and body mass are strongly correlated ($r \approx 0.87$), but longer flippers do not *cause* greater mass — both increase together as the animal grows.

### Covariance
A measure of how two variables change together. Positive covariance means they tend to move in the same direction; negative covariance means they tend to move in opposite directions. Formula: $\frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{N}$. Limitation: depends on the units of measurement, so different scales produce incomparable values. (As with variance, sample estimates often divide by $N-1$ instead; NumPy's `np.cov` defaults to $N-1$, while the 17_0 notebooks divide by $N$ to keep the focus conceptual.)

---

## D

### Datasaurus Dozen
A set of thirteen datasets created by Justin Matejka and George Fitzmaurice (2017) that, like Anscombe's Quartet, have nearly identical summary statistics but look very different when plotted — including one shaped like a dinosaur (based on Alberto Cairo's original Datasaurus drawing). Demonstrates that summary statistics alone cannot capture data structure.

### Deviation
The difference between an individual observation and the mean: $y_i - \bar{y}$. Deviations sum to zero (positive and negative cancel out), which is why we square them to measure total variability.

---

## H

### High-Leverage Point
A data point with an extreme x-value relative to the rest of the dataset. Because of its distance from the center of the data, it can exert disproportionate influence on the slope of a regression line. Analogous to a heavy weight on a lever — the farther from the fulcrum, the more rotational force.

---

## I

### Intercept ($\beta_0$)
The predicted value of $y$ when $x = 0$. In simple linear regression, the optimal intercept is $\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$.

---

## L

### Leverage
A measure of how far a data point's x-value is from the mean of x. High-leverage points have the potential to strongly influence the regression line. An outlier combined with high leverage is especially dangerous.

---

## M

### Mean ($\bar{y}$)
The arithmetic average of a set of values. In regression, the mean serves as the simplest possible model — a baseline that we must improve upon. Total Sum of Squares measures the total error of this baseline model.

---

## O

### Outlier
A data point with an unusual y-value given its x-value (a large residual). Unlike a high-leverage point, an outlier's x-value may be near the center of the data, but its y-value deviates significantly from the pattern.

---

## P

### Pearson's $r$
See **Correlation**.

### Predicted Value ($\hat{y}$)
The value output by a regression model for a given input. Pronounced "y-hat." Distinguished from the actual observed value $y$.

---

## Q

### Quadrant Trick (Covariance Intuition)
A visual method for estimating the direction of association: draw vertical and horizontal lines at the means of x and y, dividing the scatterplot into four quadrants. Points in the top-right and bottom-left quadrants support a positive relationship; points in the top-left and bottom-right quadrants support a negative relationship.

---

## R

### $R^2$ (R-Squared)
The proportion of variance in the target variable explained by the model. Formula: $R^2 = 1 - \frac{\text{RSS}}{\text{TSS}}$. Ranges from 0 (model explains nothing) to 1 (model explains everything). For simple linear regression, $R^2 = r^2$, the square of the correlation coefficient.

### Residual ($e_i$)
The difference between the actual value and the model's predicted value: $e_i = y_i - \hat{y}_i$. A positive residual means the model underpredicted; a negative residual means it overpredicted.

### Residual Sum of Squares (RSS)
The sum of squared residuals: $\sum(y_i - \hat{y}_i)^2$. Measures the total error remaining after fitting the model. Squaring solves the cancellation problem (positive and negative residuals would otherwise cancel). The optimal regression line minimizes RSS.

---

## S

### Scatterplot
A two-dimensional plot where each point represents one observation, with the x-value on the horizontal axis and the y-value on the vertical axis. The primary visualization tool for examining the relationship between two quantitative variables.

### Slope ($\beta_1$)
The change in the predicted value of $y$ for a one-unit increase in $x$. In simple linear regression, the optimal slope is $\hat{\beta}_1 = r \cdot \frac{s_y}{s_x}$.

### Standard Deviation ($s$)
The square root of the variance. Measures the typical distance of a data point from the mean. In the same units as the original data, making it more interpretable than variance.

---

## T

### Total Sum of Squares (TSS)
The sum of squared deviations from the mean: $\sum(y_i - \bar{y})^2$. Measures the total variability in the target variable. Serves as the baseline error that any regression model must improve upon. TSS is the error of the mean-only model.

---

## V

### Variance ($s^2$)
The average squared distance of data points from the mean. Formula: $\frac{\sum(y_i - \bar{y})^2}{N} = \frac{\text{TSS}}{N}$. Units are the square of the original data's units. (You will also see the **sample** formula with $N-1$ in the denominator — *Bessel's correction*. The 17_0 notebooks divide by $N$ to keep the focus on the *average* squared deviation. Watch the library defaults: NumPy's `np.var`/`np.std` divide by $N$, while pandas' `.var()`/`.std()` divide by $N-1$, so they can return slightly different numbers.)

### Visualization
The practice of plotting data to reveal patterns, anomalies, and relationships that summary statistics alone cannot capture. Central lesson of the Anscombe's Quartet and Datasaurus Dozen demonstrations.
