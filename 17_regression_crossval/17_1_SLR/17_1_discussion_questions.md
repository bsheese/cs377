# Discussion Questions: 17_1 Simple Linear Regression

---

## 17_1_1: The Two Paradigms (Scikit-Learn vs. Statsmodels)

### APIs and Libraries

1. Why does Scikit-Learn require X to be a 2D array (using `.reshape(-1, 1)`) while Statsmodels can accept a 1D array directly?

2. Scikit-Learn's philosophy is "prediction-focused" while Statsmodels is "inference-focused." What does this fundamental difference mean in terms of what information each library prioritizes?

3. The notebook uses `model.coef_[0]` in Scikit-Learn but `model.params[1]` in Statsmodels to extract the slope. Why the different indexing? What does index 0 vs. index 1 represent?

4. Why must we manually add an intercept with `sm.add_constant(X)` in Statsmodels but not in Scikit-Learn?

5. Both libraries produce identical slope, intercept, and R² values. Why is this convergence important as evidence that both are solving the same underlying math problem?

### Interpreting the Model

6. The model predicts that a penguin with a 200mm flipper has a body mass of 9,130g. A penguin with a 0mm flipper would weigh −5,872g. Why is this second prediction conceptually nonsensical, and what does it tell us about extrapolation beyond the data range?

7. The intercept represents the expected y-value when x = 0. In what scenarios is the intercept meaningful versus when is it "just a mathematical artifact"?

8. The R² = 0.762 means about 76% of variation in body mass is explained by flipper length. What accounts for the remaining 24%? Name at least two plausible biological factors.

### Pedagogical Progression

9. Why does the notebook verify the fitted slope against a closed-form formula? What does this verification prove about the relationship between different approaches?

10. If you were teaching a student linear regression for the first time, which library would you start with: Scikit-Learn or Statsmodels? Justify your choice.

---

## 17_1_2: Statistical Significance

### Building Intuition with Simulation

1. The bootstrap resampling procedure draws pairs (x, y) with replacement from the original data. Why is it important to draw *pairs* together rather than resampling x and y independently?

2. The bootstrap SE (~1.49) matches the Statsmodels SE (~1.54) even though they are computed using completely different methods. What does this agreement tell you?

3. A permutation test shuffles the y-values randomly and refits the model. Why does this procedure simulate a world where x and y have no relationship?

4. In the permutation test with 5,000 shuffles, the observed slope of 50.15 never appeared in the null distribution. What is the empirical p-value, and why is it so small?

### Hypothesis Testing and P-Values

5. The null hypothesis H₀ states that the true population slope is zero. Under what circumstances would you reject H₀ as implausible?

6. A p-value of < 0.001 means there's less than 0.1% chance of observing this slope if the null hypothesis were true. Does this mean there's a 99.9% chance the alternative hypothesis is true? Explain the subtle difference.

7. Small sample sizes produce wide confidence intervals and large standard errors. Why does sample size matter so much for detecting real relationships?

### Confidence Intervals

8. The three methods for constructing a 95% CI (bootstrap percentile, formula-based, Statsmodels) all converge on [47.1, 53.2]. What does this convergence mean?

9. The notebook runs a coverage simulation: repeated 1,000 times, roughly 95% of intervals contain the true slope. Does this mean we should expect 95% of our real-world intervals to contain the truth? Why or why not?

10. If you lowered the confidence level from 95% to 90%, would the confidence interval get wider or narrower? Why?

### Practical Significance vs. Statistical Significance

11. A regression has p < 0.001 (highly significant) but R² = 0.05 (very weak effect). How would you explain this paradox to someone unfamiliar with statistics?

12. Is a slope of 50g per mm "practically significant" for penguin biology? What information beyond the p-value would you need to answer this?

---

## 17_1_3: The LINE Assumptions

### Linearity

1. What does "linearity" mean as an assumption, and how would you check it visually?

2. In the Auto MPG data, the residual plot shows a clear U-shape. What does this tell you about the relationship between displacement and mpg?

3. A LOWESS smoother is overlaid on the residual plot. What does a smooth curve that deviates from the zero line indicate?

4. If a residual plot shows perfect randomness around zero, does that guarantee the model is correct? What other assumptions must still hold?

### Independence

5. What is the Durbin–Watson statistic, and what does a value near 2.0 indicate?

6. The Durbin–Watson test flagged the Auto MPG data (DW = 0.926) but the notebook explains this is likely due to year-ordering, not true autocorrelation. How would you distinguish between these two cases?

7. In what types of datasets would you expect independence to be violated? Give examples beyond time-series data.

8. If observations are clustered (e.g., multiple measurements from the same subject), what does this violate about independence, and how could it affect your model?

### Normality

9. A Q–Q plot shows points that curve upward at the high end (heavy right tail). What does this tell you about the distribution of residuals?

10. The Central Limit Theorem makes inference "robust to mild non-normality" at n > 30. What does robustness mean here—does it mean non-normality doesn't matter at all?

11. Why is a Q–Q plot considered more informative than a histogram for detecting non-normality?

12. The Jarque-Bera test has a p-value of 0.087. Should you conclude the residuals are normally distributed? Why or why not?

### Equal Variance (Homoscedasticity)

13. A "funnel" or "megaphone" shape in the residual plot indicates heteroscedasticity. What does this mean, and why does it matter?

14. How would you quantitatively test for equal variance without relying solely on visual inspection?

15. If heteroscedasticity is detected, what are two strategies to address it? (Hint: one involves the target variable.)

### Honest Warnings

16. The notebook emphasizes that LINE violations compromise p-values, CIs, and SEs, but not the slope itself. What does this mean? Can you trust the point estimate even if assumptions fail?

17. If you violate linearity (fit a line to curved data), what happens to your predictions outside the data range compared to predictions in the middle of the data?

18. Is it ever acceptable to violate one or more LINE assumptions? Under what circumstances?

---

## 17_1_4: Influence, Leverage, and Cook's Distance

### Conceptual Distinctions

1. Define "leverage," "outlier," and "influence" in your own words. Give an example of a point that is high-leverage but not influential.

2. A point has high leverage (unusual x-value) but a residual near zero (predicted well). Why is this point harmless to the regression line?

3. A point has low leverage but a large positive residual. Why is this point also relatively harmless?

4. Cook's Distance combines leverage and residual disagreement into a single metric. Why is this combination necessary to identify truly problematic points?

### Detecting Influential Points

5. The Ames Housing dataset had one "poisoned" row (10,000 sq ft, $500). When removed, the slope changed from 0.0997 to 0.1126. What does this change tell you about how much that single row dominated the fit?

6. When visualizing Cook's Distance, the notebook uses a log scale and a stem plot. Why are these visualization choices helpful?

7. Is there an "official" threshold for Cook's Distance above which you should automatically remove a point? Why or why not?

8. Studentized residuals differ from standardized residuals. What additional information do studentized residuals account for?

### The Ethics of Data Dropping

9. The notebook lists "legitimate" and "illegitimate" reasons to drop data. What makes a reason legitimate?

10. A point meets the threshold for high Cook's Distance. You investigate and find it's a correct data entry. Should you drop it? Why or why not?

11. What is the risk of dropping multiple points iteratively based on Cook's Distance?

12. Huber regression is presented as an alternative to dropping outliers. How does it handle influential points differently?

### Practical Alternatives

13. Instead of dropping a high-influence point, what other approaches could you take? List at least three.

14. If you remove an influential point, how would you communicate this decision in a report?

---

## 17_1_5: Transformations

### Motivation and Intuition

1. The Gapminder data shows diminishing returns: each additional dollar of GDP predicts less additional life expectancy. How does a log transformation straighten this relationship?

2. Why does a log transformation often fix *both* linearity and heteroscedasticity violations simultaneously?

3. Box-Cox transformation finds the optimal λ (lambda). What does λ = 0 suggest? What does λ = 1 suggest?

### Interpreting Log-Transformed Models

4. In the Gapminder model, the coefficient for log(GDP) is approximately 5.8. How do you interpret this slope in plain English?

5. A model uses `log(y) = β₀ + β₁ * x`. A 1-unit increase in x is associated with how much percent change in y?

6. The three log models (level-log, log-level, log-log) have different interpretations. When would you use each one?

7. If β₁ = 0.05 in a log-level model (log(y) = β₀ + 0.05 * x), how much percent change in y corresponds to a 1-unit increase in x?

### Over-Transforming and Limitations

8. The Auto MPG reciprocal model (1/mpg vs. displacement) has a very different interpretation than a linear model. What does the reciprocal relationship suggest about the underlying physics?

9. Why is it dangerous to transform both the predictor AND the target variable in the same model?

10. After log-transformation, R² = 0.65 (vs. 0.46 before). Is this a "real" improvement in model fit, or is R² incomparable across different y-scales?

11. The notebook warns that "transformations can hide rather than fix problems." Give an example of when a transformation might mask an underlying issue.

### Applying Transformations

12. When you fit a log-transformed model, your predictions are on the log scale. How do you back-transform to the original scale, and what is Jensen's correction?

13. Why is Jensen's correction important when back-transforming log-level or log-log models?

---

## 17_1_6: Train/Test Split and Generalization

### The Purpose of Train/Test Splits

1. Training error is always optimistic. Why? What is the model optimizing for?

2. The notebook emphasizes "never peeking at test performance to make modeling decisions." Why is this rule so critical?

3. In cross-validation, each fold serves as test data once and training data multiple times. Why is this more efficient than a single train/test split?

### Observing the Bias-Variance Tradeoff

4. In the synthetic sin(x) experiment, what does each degree of polynomial represent in terms of model complexity?

5. At degree 1 (line), both train and test R² are low. Is this underfitting or overfitting? How do you know?

6. At degree 20, train R² ≈ 0.98 and test R² ≈ -0.5. Why is a negative test R² possible, and what does it mean?

7. Negative test R² means the model performs worse than predicting the mean. When would this occur in practice?

### The Train-Test Gap as a Diagnostic

8. A model has train R² = 0.90 and test R² = 0.87. Is this overfitting? What gap size suggests overfitting?

9. Why does the train-test gap grow as model complexity increases?

10. The bias-variance curve is U-shaped. At what degree (approximately) do you find the sweet spot in the synthetic data?

### Repeated Splits and Distribution

11. The notebook runs a 200-split experiment with varying random seeds. What is the distribution of test R² telling you about model stability?

12. If the distribution of test R² is very wide, what does that suggest about your dataset or model?

### Practical Implications

13. You fit a model that achieves 85% accuracy on a single test set. How confident should you be in this number? What would increase your confidence?

14. Should you always use nested cross-validation instead of a single train/test split? When is a simple split sufficient?

15. In the real-world scenario described in 17_1_6, why does the reciprocal transformation generalize better than the raw model?

---

## 17_1_9: The Restaurant Tips Exercise

### Integrating All Concepts

1. In the tips dataset, the relationship between total_bill and tip is approximately linear. Why might this make sense from a behavioral economics perspective?

2. The exercise asks you to run six parts mirroring notebooks 17_1_1 through 17_1_6. Why is this integration important?

3. After building the full pipeline (fitting, checking assumptions, removing influential points, transforming, and testing), how has your understanding of the tip-bill relationship deepened?

### Model Refinement

4. When you compared the raw model to the log-transformed model, did both assumptions *and* R² improve? Why might R² improvements not always accompany assumption corrections?

5. When you computed bootstrap confidence intervals for the slope, why did they turn out wider than Statsmodels CIs? (Hint: think about heteroscedasticity.)

6. After dropping or handling influential points, did the slope change substantially? If so, what would this tell you?

### Reflection Questions

7. Which LINE assumption was most violated in the tips data, and how did it affect your modeling choices?

8. Would a prediction of tip for a $200 restaurant bill be reliable? Why or why not?

9. If you wanted to predict the exact tip for a specific bill amount, what additional information would help you make a better prediction?

10. Summarize in 3–4 sentences: What is the relationship between bill size and tip, and how confident are you in that relationship?

