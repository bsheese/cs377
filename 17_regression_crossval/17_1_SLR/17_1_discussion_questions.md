# 17.1 Simple Linear Regression - Discussion Questions

1. Why do we minimize the *squared* residuals instead of the *absolute* residuals in OLS regression? What are the practical trade-offs?
2. Explain the fundamental decomposition: **Data = Fit + Residual**. How does this relate to the Sum of Squares identity ($SST = SSR + SSE$)?
3. If two variables have a strong non-linear relationship (like a parabola), why might their correlation coefficient ($r$) be close to zero?
4. What are the risks of extrapolation? Give a real-world example where a linear model would fail if extrapolated too far.
5. In an indicator variable regression predicting income from gender (Male=0, Female=1), how do you interpret a negative slope?
6. Describe three specific patterns you might see in a residual plot and explain what each one indicates about your model.
7. How can a single data point be an outlier in both the $x$ and $y$ directions but NOT be an influential point?
8. Explain the difference between "leverage" and "influence". Can a point have high leverage without being influential?
9. Why is correlation called "unitless"? How does this property make it more useful than the slope for comparing relationships across different studies?
10. If a model has an $R^2$ of 0.95, does that mean the model is "good"? What else would you need to check before trusting its predictions?
