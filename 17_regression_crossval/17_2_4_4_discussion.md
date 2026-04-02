# Discussion Questions: Grid Search, Nested Cross-Validation, and Learning Curves

## Bias-Variance Tradeoff (Section 1)

1. In the manual alpha tuning plot, the training R² is always higher than the cross-validation R². Why?
2. What happens to both the training and cross-validation scores as alpha gets very large? What does this tell you about the model?
3. The notebook says we are looking for a "Goldilocks" zone. If you had to pick an alpha based on the plot, which one would you choose and why?

## GridSearchCV (Section 2)

4. The `tune_model` function wraps everything in a `Pipeline` that includes a `StandardScaler`. Why can't we just scale the data before calling GridSearchCV? What problem does the pipeline solve?
5. When the param grid is passed to `tune_model`, the keys are like `'alpha'`, but inside the function they get prefixed to `'model__alpha'`. Why is this necessary?
6. The output says "Fitting 5 folds for each of 8 candidates, totalling 40 fits." Where does the number 40 come from? How would this change if you used 10-fold cross-validation instead of 5?
7. ElasticNet has two hyperparameters (`alpha` and `l1_ratio`). How many total combinations does GridSearchCV try with the grid used in the notebook? How many total model fits does that produce with 5-fold CV?

## Test Set Evaluation (Section 3)

8. All three optimized models (Ridge, Lasso, ElasticNet) produce very similar Test R² scores. Why might that be?
9. The notebook evaluates models on the test set after using GridSearchCV to tune on the training set. What is the risk of looking at the test set scores and then going back to re-tune your hyperparameters?

## Nested Cross-Validation

10. In your own words, what is "optimistic bias" and why does it occur when you select hyperparameters based on cross-validation scores?
11. The notebook explains that nested CV uses the full dataset (not a train/test split). Why is this an advantage? What does it buy you compared to the train/test approach used in Sections 2-3?
12. In the nested CV code, `GridSearchCV` is passed as the estimator to `cross_val_score`. Trace through what happens on a single outer fold: what does the inner loop do, and what does the outer loop do with the result?
13. The notebook states that nested CV is computationally expensive. For a 5-fold outer loop, 5-fold inner loop, and ElasticNet with 4 alpha values and 5 l1_ratio values, how many total Ridge/Lasso fits occur? How many ElasticNet fits? (Hint: count carefully -- each inner GridSearchCV fit involves all parameter combinations across all inner folds.)
14. Why might different outer folds select different "best" hyperparameters? What does it mean if they do?

## Comparing Approaches

15. Under what circumstances would you prefer a simple train/test split over nested cross-validation? When would nested CV be the better choice?
16. The comparison table says nested CV gives "quantifiable" variance of the estimate. What does this mean, and how does the code produce it?

## Learning Curves

17. Looking at the learning curve plot, what do the shaded bands around each line represent?
18. If the training score is much higher than the validation score at the largest training size, what does that suggest about the model? What would you try next?
19. If both curves converge at a low R² value and adding more data doesn't help, what does that tell you? Is this a data problem or a model problem?
20. The learning curve is plotted using `best_ridge` -- the pipeline returned by GridSearchCV with the best alpha already selected. Why does it make sense to use the tuned model for learning curves rather than an untuned one?

## Putting It All Together

21. A colleague tells you: "I just ran my model on the test set and got an R² of 0.92. I tuned my hyperparameters using GridSearchCV with 5-fold CV on the training set." What follow-up questions would you ask before trusting that 0.92 number?
22. Imagine you have a very small dataset (200 rows). Would you use train/test split, plain cross-validation, or nested cross-validation? Justify your answer.
23. The notebook moves from "guessing" alpha values to "optimizing" them. But is GridSearchCV truly optimization, or is it still a form of search? What are its limitations?
