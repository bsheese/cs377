# Discussion Questions: 17_0 — Regression Preliminaries

---

## 17_0_1: Univariate Spread (Mean, Variance, TSS)

### The Mean as a Baseline Model

1. The notebook frames the mean as a "baseline prediction" that minimizes squared error. What does "minimizes squared error" mean, and why would you want a prediction that minimizes this quantity?

2. If deviations from the mean always sum to zero, why does this make the raw sum of errors useless as an error metric?

3. Why is squaring deviations preferred over taking absolute values as a way to prevent cancellation? What are the tradeoffs of each choice?

4. TSS is computed in squared grams (g²). Why is this a strange unit for measuring "spread in body mass"? How does standard deviation solve this problem?

5. The standard deviation for penguin body mass is ~801g. Given that the mean is ~4,207g, what does this tell you about how much variation exists in penguin sizes? Is this a lot or a little?

6. Why is the mean the best single-number summary for squared error, but not necessarily for other error functions (e.g., absolute error)? What would you use instead?

### TSS and Baseline Comparison

7. TSS is called "the baseline error we need to beat." Beat it at what? What does reducing TSS tell you about a model?

8. If a dataset has TSS = 0, what must be true about the data? Is such a dataset useful for modeling?

9. A student claims: "A model with TSS = 5 million is worse than one with TSS = 500." What's wrong with this reasoning?

---

## 17_0_2: Visualizing Associations (Anscombe's Quartet & Datasaurus)

### The Importance of Visualization

1. Anscombe's Quartet consists of four datasets with nearly identical means, variances, correlations, and regression lines. Yet their scatterplots look completely different. What does this demonstrate about the limits of summary statistics?

2. Dataset II in Anscombe's Quartet is a perfect parabola. The correlation is ~0.816 and a linear regression line fits through it. Is the linear model appropriate here? What harm could come from using it?

3. Dataset IV has all x-values near 8 except one point at x=19. Why does this single high-leverage point have such an outsized effect on the regression line?

4. "Always visualize before fitting a model" is the main lesson of this notebook. Give a real-world example where fitting a model without visualizing first could lead to a costly mistake.

5. The Datasaurus Dozen shows 13 datasets — including a dinosaur — all with identical summary statistics. What does this imply about the reliability of automated modeling pipelines that skip visualization?

### Outliers vs. High-Leverage Points

6. Define the difference between a y-outlier and a high-leverage point. Which is more dangerous to a regression model, and why?

7. In Dataset III of Anscombe's Quartet, there is one y-outlier but all other points fall exactly on a line. How does this single outlier affect the slope and intercept? What would the model look like without it?

8. A point with high leverage but a small residual (it falls near the regression line) is said to be "harmless." Why? Could it ever become a problem?

9. "A high-leverage point is like a fulcrum on a lever." Explain this physics analogy in terms of how the point affects the regression line's slope.

---

## 17_0_3: Covariance and Correlation

### The Quadrant Trick

1. The quadrant trick divides the scatterplot into four regions using mean lines. Explain why points in the upper-right and lower-left quadrants contribute positively to covariance.

2. If most data points fall in the upper-left and lower-right quadrants, what sign will the covariance have? What does this mean about the relationship between x and y?

3. A dataset has equal numbers of points in all four quadrants. What would you predict for the covariance? What would the scatterplot look like?

### From Covariance to Correlation

4. Covariance between flipper length (mm) and body mass (g) is ~9,796 mm·g. When converted to kg, it becomes 9.796. Is the relationship stronger or weaker in kilograms? Why is this unit-dependence a problem?

5. Dividing covariance by the product of standard deviations produces a unitless correlation. Explain why dividing by standard deviations cancels the units.

6. Pearson's r is bounded between -1 and +1. What does r = +1 look like geometrically? What does r = 0 look like?

7. The notebook warns that r only measures linear relationships. Describe a dataset where r ≈ 0 but x and y are strongly related. (Hint: think about the shape of the relationship.)

8. For the palmer penguins, r ≈ 0.87 between flipper length and body mass. What does this mean in plain English? Does it mean flipper length *causes* body mass to change?

### Units and Interpretation

9. If you measured flipper length in millimeters vs. centimeters, would the correlation change? Would the covariance change? Why the difference?

10. Two analysts compute covariance on the same dataset but get different numbers. One used N as the denominator, one used N-1. Which is the "population" formula and which is the "sample" formula? Why does N-1 matter?

---

## 17_0_4: Residuals

### Defining Residuals

1. A residual is defined as $e_i = y_i - \hat{y}_i$ (actual minus predicted). Why is the sign of the residual meaningful? What does a large positive residual tell you about the model's prediction for that point?

2. The notebook shows that "any line through $(\bar{x}, \bar{y})$" has residuals summing to zero. Why does passing through the mean guarantee this cancellation?

3. A "terrible line" with slope -10 and intercept 102.5 has residuals of [-42.5, -27.5, -7.5, 7.5, 27.5, 42.5] that sum to zero. Does this mean the model is equally good as the better-fitting line? What metric distinguishes them?

4. If a residual is zero, the model predicted that point perfectly. Does this mean the model is good? What caution is warranted?

### Residuals vs. Deviations

5. Deviations from the mean (notebook 17_0_1) and residuals from the regression line (this notebook) are computed the same way: actual minus predicted. What's the conceptual difference between them?

6. If you fit a regression line to data that has no relationship between x and y, what would you expect the residuals to look like?

7. Vertical lines from data points to the fitted line represent residuals geometrically. Why are residuals measured *vertically* rather than at a right angle to the line?

### From Residuals to RSS

8. The notebook shows that summing raw residuals fails as an error metric. What is the solution, and why does it work?

9. If two different lines both have residuals summing to zero, how do you choose which line is better? What quantity tells you?

10. A residual plot (residuals vs. fitted values) that shows a U-shape would indicate what problem? How does this connect to the LINE assumptions in 17_1?

---

## 17_0_5: R-squared and OLS

### The RSS Bowl and Optimization

1. When RSS is plotted against candidate slopes, a U-shaped bowl appears. Why is this bowl U-shaped? Why does it have exactly one minimum?

2. A "grid search" tests 100 candidate slopes. Why is this inefficient? Why would it become even more inefficient with a second predictor?

3. The closed-form optimal slope formula is $m = r \cdot \frac{\sigma_y}{\sigma_x}$. Why does the correlation r appear in the slope formula?

4. If $r = 1$ (perfect positive relationship), what would the slope be? If $r = 0$ (no relationship), what would the slope be?

### Understanding R²

5. R² = 0.50 for the Ames Housing model (area predicting price). In plain English, what does this mean?

6. R² is defined as $1 - \frac{RSS}{TSS}$. What is the intuition behind this formula? What does it measure?

7. R² = 0 means the model is no better than predicting the mean. Can R² be negative? What would that mean about the model?

8. R² = 1 means perfect fit (RSS = 0). Is this always desirable? When might R² = 1 be a warning sign?

9. The identity $R^2 = r^2$ holds only for simple linear regression. Why does squaring the correlation give you R²?

### Connecting the Pieces

10. The series started with TSS (baseline error) and ended with R² as the fraction of TSS explained. Trace this chain in your own words: what does each step in the series add to your understanding?

11. If you added a completely random feature (random numbers) to a simple linear regression, would R² go up, down, or stay the same? Why does this matter for multiple regression?

12. A model with R² = 0.95 sounds impressive. What other information would you want before trusting this model for real decisions?

