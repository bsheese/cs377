# Discussion Questions: Regularization (Ridge, Lasso, ElasticNet)

## The Problem with Unregularized Models

1. The notebook states that "a regression model wants to minimize its error on the training data, but it will happily assign massive weights in pursuit of the best fit." Why are massive weights a problem? What do they indicate about the model?

2. In Part 2, we used forward and backward selection to reduce features. The notebook says regularization is "widely considered a much more modern and powerful approach." What are the four reasons given for why regularization is preferred over stepwise selection?

3. Stepwise selection is described as "binary and harsh" — a feature is either 100% in or 100% out. How does regularization handle this differently, and why is that an advantage?

## Ridge Regression (L2)

4. Ridge regression adds a penalty based on the squared value of the coefficients (the L2 penalty). Why does squaring the coefficients mean that Ridge "heavily punishes abnormally large coefficients"?

5. The notebook explains that Ridge shrinks coefficients "asymptotically close to zero, but it never forces them to exactly 0.0." Why does this matter? What are the consequences for model interpretability?

6. Ridge is described as particularly good at handling multicollinearity. When two features are highly correlated, what does Ridge do with their coefficients, and why is this better than what OLS or stepwise selection might do?

7. The notebook states that Ridge requires mandatory feature scaling. Without scaling, what would happen to features measured in large units (like square footage) versus features measured in small units (like number of bathrooms)?

8. The notebook says Ridge coefficients lose "traditional interpretability." If a Ridge coefficient for `Square Feet` is \$40, why can't you interpret that as "each additional square foot adds exactly \$40 to the price"?

## Lasso Regression (L1)

9. Lasso uses the absolute value of coefficients (L1 penalty) instead of the squared value (L2). How does this small mathematical change lead to such a dramatic difference in behavior — specifically, the ability to zero out coefficients entirely?

10. When Lasso encounters two highly correlated features, the notebook says it "will essentially pick one at random to keep and drop the other." Why might this be problematic for interpretation, even if it doesn't hurt predictive accuracy?

11. What happens if alpha is set too high in Lasso? The notebook warns about a specific scenario — what is it?

## ElasticNet

12. ElasticNet combines L1 and L2 penalties. The notebook says it "guarantees that your model will perform at least as well as Ridge or Lasso." Why is this true? Under what circumstances would ElasticNet effectively become Ridge or Lasso?

13. The `l1_ratio` parameter controls the mix between Lasso and Ridge. If you set `l1_ratio=0.3`, what does that mean in terms of how the penalty is applied?

14. The notebook identifies "double the hyperparameter tuning" as the biggest cost of ElasticNet. Why does having two hyperparameters instead of one make such a significant difference in practice?

## Feature Scaling and Pipelines

15. Every regularization model in the notebook is wrapped in a `Pipeline` with `StandardScaler`. What would go wrong if you ran Lasso or Ridge on unscaled data?

16. The OLS model in the notebook uses a `SequentialFeatureSelector` step in the pipeline. How does the OLS approach to feature selection differ fundamentally from how Lasso performs feature selection?

## Comparing Models

17. Looking at the model comparison table, Ridge keeps all 38 features while Lasso keeps only 22, yet their test R² scores are very similar. What does this tell you about the value of the features that Lasso dropped?

18. The notebook then runs the same models on a dataset with 276 features (full one-hot encoding). Ridge keeps 272 of 276 features. What is the practical consequence of keeping nearly all features, and why might Lasso or ElasticNet be more useful in this scenario?

19. ElasticNet is described as popular in fields like "bioinformatics, genomics, and econometrics." Why would these fields specifically benefit from a method that combines Ridge's handling of correlated features with Lasso's ability to zero out coefficients?

## Overfitting and the Bias-Variance Tradeoff

20. Regularization is described as "intentionally trad[ing] a tiny bit of accuracy on the training data (introducing bias) to achieve a massive improvement in predicting new, unseen data (reducing variance)." Explain this tradeoff in your own words. Why would adding bias to a model sometimes improve it?

21. The alpha parameter controls the strength of regularization. If you could see the training error and validation error plotted against alpha, what pattern would you expect to see as alpha increases from 0 to a very large value?

22. The notebook notes that the alpha values used (e.g., α=10.0 for Ridge, α=0.01 for Lasso) are arbitrary. What risk does this introduce, and how would you systematically find a better alpha value?

## Limitations and Modern Context

23. The notebook states that ElasticNet "still assumes relationships are straight lines." What kinds of patterns in data would a linear model with regularization fail to capture?

24. The final section mentions that researchers who "only care about predictive accuracy" often move to tree-based models like Random Forests or XGBoost. Under what circumstances would you still prefer a regularized linear model over a tree-based model?
