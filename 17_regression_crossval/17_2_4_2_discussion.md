# Discussion Questions: Forward/Backward Selection, Cross-Validation, and Feature Engineering

## One-Hot Encoding

1. The notebook asks: "Why do we use `drop_first=True`?" What is the "Dummy Variable Trap," and what problem does dropping the first category column solve for linear regression?

## Pipelines and Data Leakage

2. The feature selection and evaluation function wraps `StandardScaler`, `SequentialFeatureSelector`, and `LinearRegression` together in a `Pipeline`. Why is this necessary rather than scaling the data first and then running feature selection separately?

3. The notebook uses `cross_validate` to evaluate the model. During cross-validation, the data is split into training and validation folds. If the scaler were fit on the entire dataset before cross-validation, what information would "leak" from the validation fold into the training fold?

4. The function returns a dictionary containing the pipeline, selected features, and CV metrics. After cross-validation is complete, the pipeline is refit on the entire dataset passed to the function. Why is this done, and what data should that "entire dataset" typically be?

## Forward and Backward Selection

5. Forward selection starts with no features and adds them one at a time. Backward selection starts with all features and removes them one at a time. Under what circumstances might these two approaches select different sets of features?

6. The notebook uses `n_features_to_select='auto'`, which lets the algorithm decide how many features to keep. What are the tradeoffs of letting the algorithm choose versus specifying the number yourself?

7. The notebook notes that only 2 CV folds are used "to keep computation time manageable." What is the tradeoff when using fewer folds? Why would 5 or 10 folds give more reliable estimates?

## Interpreting Coefficients

8. The forward selection model produces a negative coefficient for `Bedroom AbvGr`. The notebook explains this by saying the model is "holding total square footage constant." In your own words, why would adding a bedroom while keeping square footage the same actually decrease a home's value?

9. The residuals plot shows that the model's errors get larger as home price increases. The notebook attributes this partly to the "massive right skew" of SalePrice. What does it mean for a distribution to be right-skewed, and why does this cause problems for linear regression?

## Log Transformation

10. Applying a log transform to SalePrice changes the distribution from right-skewed to approximately normal. Why does linear regression perform better when the target variable is approximately normally distributed?

11. After log-transforming SalePrice, the coefficients are no longer in raw dollars. If a coefficient for `Gr Liv Area` is 0.14 in the log-transformed model, how should you interpret its effect on the original dollar-scale price?

12. The notebook shows that log-transforming SalePrice improves the test R² from 0.8834 to 0.9346. Why does this transformation improve the model so dramatically, even though no new information was added to the dataset?

## Polynomial Features

13. The notebook explains that adding $X^2$ allows the regression line to bend, yet the model is still called "linear" regression. Why is a model with squared terms still considered linear?

14. Three examples of potential polynomials are given: `Overall Qual_Sqr`, `Bedroom AbvGr_Sqr`, and `Garage Area_Sqr`. For each, the notebook predicts a different curve shape (upward hockey stick, inverted U, diminishing returns). What determines whether the squared term's coefficient is positive or negative?

15. The notebook warns that you "should almost always include the base linear term" when adding a squared term. Why is it problematic to include $X^2$ without $X$?

16. Forward selection did not choose any of the three polynomial features when they were added to the full model. Does this mean polynomials are never useful, or could there be another explanation?

## Multicollinearity

17. The Variance Inflation Factor (VIF) measures how much a feature's coefficient variance is inflated due to correlation with other features. Why does high VIF make individual coefficients unreliable, even if the overall model predictions are still accurate?

18. `Gr Liv Area` has a VIF of 118 when all numeric features are included, but only 2.84 in the reduced feature set selected by forward selection. What does this tell you about how forward selection interacts with multicollinearity?

19. The correlation analysis reveals that `Garage Cars` and `Garage Area` are correlated at 0.89. If both features were included in a model, what would you expect to happen to their coefficients compared to including only one?

20. The notebook notes that multicollinearity "may have affected the forward selection process to begin with." Why might multicollinearity cause forward selection to make suboptimal choices about which features to include?

## The Modeling Workflow

21. The notebook defines a `run_modeling_workflow` function that handles train-test split, cross-validation, feature selection, evaluation, and residual plotting in one call. What are the advantages of encapsulating this workflow in a function rather than running each step separately?

22. The workflow evaluates the model using both cross-validation R² (on training data) and test R² (on held-out data). Why are both metrics reported, and what does it mean if they differ significantly?

23. The notebook ultimately shows that forward selection with log-transformed SalePrice produces a strong model (R² = 0.93). However, it also reveals significant multicollinearity among the original features. What limitation does this create for interpreting the selected features, and how might you address it in Part 3?
