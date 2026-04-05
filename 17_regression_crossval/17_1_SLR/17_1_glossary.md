# 17.1 Simple Linear Regression - Glossary

| Term | Definition |
|---|---|
| **Coefficient of Determination** ($R^2$) | The proportion of variation in the outcome variable explained by the linear model. For SLR, $R^2 = r^2$. |
| **Correlation** ($r$) | A unitless measure of the strength and direction of the linear relationship between two variables, ranging from -1 to +1. |
| **Extrapolation** | Applying a regression model to predict values outside the range of the observed data. |
| **Heteroscedasticity** | A condition where the variance of the residuals is not constant across the range of the predictor (often appearing as a "funnel" shape). |
| **High Leverage** | Describes points that have an extreme value in the predictor variable ($x$), giving them more potential to influence the regression line. |
| **Indicator Variable** | A binary variable (coded 0 or 1) used to include a two-level categorical predictor in a regression model. |
| **Influential Point** | A point (usually with high leverage) that, if removed, would substantially change the estimated slope or intercept of the regression line. |
| **Least Squares Line** | The unique line that minimizes the sum of squared residuals (SSE). |
| **Leverage Point** | An outlier in the $x$-direction. |
| **Outcome Variable** ($y$) | The variable being predicted (also called the dependent, response, or target variable). |
| **Outlier** | An observation that falls far from the main cloud of points in bivariate space (often having a large residual). |
| **Predictor Variable** ($x$) | The variable used to make predictions (also called the independent, explanatory variable, or feature). |
| **R-squared** | See Coefficient of Determination. |
| **Regression Sum of Squares** ($SSR$) | The variability in the outcome explained by the regression model: $\sum(\hat{y}_i - ar{y})^2$. |
| **Residual** ($e$) | The vertical distance from an observed data point to the regression line: $e = y - \hat{y}$. |
| **Sum of Squared Error** ($SSE$) | The sum of the squared residuals: $\sum e_i^2$. Measures the unexplained variability. |
| **Total Sum of Squares** ($SST$) | The total variability in the outcome variable around its mean: $\sum(y_i - ar{y})^2$. Note that $SST = SSR + SSE$. |
