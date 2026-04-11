# Discussion Questions: 17_2 Regression Cross-Validation Series

---

## 17_2_4_1: Data Cleaning for Multiple Linear Regression

### Initial Data Inspection

1. The notebook begins by using `.info()` to examine the DataFrame. What information does this provide that is critical before starting any modeling work? Why is the "Non-Null Count" column especially important?

2. When removing rows where a numeric feature has extreme outliers (e.g., >= 4000), what kind of data problem can these extreme values create for a linear regression model?

### Removing Uninformative Features

3. `Order` and `PID` are unique identifiers. If you accidentally left a unique identifier in a regression model, what would happen? What would the model "learn" from it?

4. The notebook checks for columns with only one unique value. Why are monotonic (constant) features useless for modeling?

5. Duplicate rows and rows where all values are NaN are removed. What could cause duplicate rows to appear in a real dataset? Why might rows with all NaN values exist?

### Yolked Variables

6. "Yolked features" occur where a categorical feature being "None" always corresponds to a numeric feature being 0 (e.g., no garage means garage area = 0). Why does this create a problem for linear regression?

7. When resolving yolked variables, why might you choose to drop some features entirely while combining others into binary features?

8. Some automated yolked variable detection might produce false positives. Why might an algorithm incorrectly identify non-yolked features as yolked?

### Cleaning Categorical Features

9. Categorical features where one category accounts for more than 70% of values are dropped. Why are highly unbalanced categorical features problematic for regression models?

10. For categorical features where one category is above 50%, the notebook creates a binary column (the top category vs. all others). What are the tradeoffs of this approach compared to full one-hot encoding?

11. Foundation is collapsed from multiple categories into three: PConc, CBlock, and Other. Why might grouping rare categories into an "Other" category be preferable to keeping them separate?

12. What does "explode the one_hots" mean, and why is having too many one-hot encoded columns a concern?

### Cleaning Numeric Features

13. Numeric features where one value accounts for more than 90% of observations are dropped entirely. Features where one value accounts for more than 80% are converted to boolean (0/1). What is the logic behind treating these two thresholds differently?

14. A numeric feature that is 60% zero might be dropped, but it "would be worth holding onto for non-basic models." Why might a feature that is mostly zero still be valuable in a more sophisticated model?

15. Missing numeric values are filled with the column median. Why is median imputation often preferred over mean imputation? Under what circumstances might median imputation introduce bias?

### Broader Data Cleaning Decisions

16. The notebook makes many subjective decisions: which features to drop, which categories to collapse, which thresholds to use. Two analysts cleaning the same dataset might make different choices. How could different cleaning decisions affect the final model's performance and interpretability?

17. A conclusion states: "If we planned on using other techniques, we'd leave in the more and clean a less." Why would different modeling techniques require different levels of data cleaning?

18. Throughout the notebook, the author drops features to keep the exercise shorter and simpler. In a real-world project, what are the risks of dropping features too aggressively during cleaning versus keeping too many?

### Feature Engineering Decisions

19. When collapsing a multi-category feature into a binary one (e.g., "attached garage" vs. "all others"), what information is lost?

20. A helper function checks whether columns exist before trying to drop them. Why is this kind of defensive coding important when writing data cleaning pipelines?

### Preparation for Modeling

21. At the end of data cleaning, the DataFrame has been reduced from many columns to approximately 38. The remaining features are a mix of numeric and boolean types. Why is having a clean, consistent set of feature types important before feeding data into a scikit-learn pipeline?

22. Data cleaning is separated from model building rather than doing both simultaneously. Why is this critical?

23. Identify at least two data cleaning decisions that would likely be different if the goal were to build a tree-based model instead of linear regression.

---

## 17_2_4_2: Forward/Backward Selection, Cross-Validation, and Feature Engineering

### One-Hot Encoding

1. Why do we use `drop_first=True`? What is the "Dummy Variable Trap," and what problem does dropping the first category column solve for linear regression?

### Pipelines and Data Leakage

2. The feature selection and evaluation function wraps StandardScaler, SequentialFeatureSelector, and LinearRegression together in a Pipeline. Why is this necessary rather than scaling the data first and then running feature selection separately?

3. During cross-validation, the data is split into training and validation folds. If the scaler were fit on the entire dataset before cross-validation, what information would "leak" from the validation fold into the training fold?

4. After cross-validation is complete, the pipeline is refit on the entire dataset passed to the function. Why is this done, and what data should that "entire dataset" typically be?

### Forward and Backward Selection

5. Forward selection starts with no features and adds them one at a time. Backward selection starts with all features and removes them one at a time. Under what circumstances might these two approaches select different sets of features?

6. Using `n_features_to_select='auto'` lets the algorithm decide how many features to keep. What are the tradeoffs of letting the algorithm choose versus specifying the number yourself?

7. Only 2 CV folds are used to keep computation time manageable. What is the tradeoff when using fewer folds? Why would 5 or 10 folds give more reliable estimates?

### Interpreting Coefficients

8. A forward selection model produces a negative coefficient for a bedroom-related feature. The model is "holding total square footage constant." In your own words, why would adding a bedroom while keeping square footage the same actually decrease a home's value?

9. A residuals plot shows that the model's errors get larger as home price increases. This is partly due to the "massive right skew" of the target variable. What does it mean for a distribution to be right-skewed, and why does this cause problems for linear regression?

### Log Transformation

10. Applying a log transform to the target variable changes the distribution from right-skewed to approximately normal. Why does linear regression perform better when the target variable is approximately normally distributed?

11. After log-transforming the target, the coefficients are no longer in raw dollars. If a coefficient for square footage is 0.14 in the log-transformed model, how should you interpret its effect on the original dollar-scale price?

12. Log-transforming the target variable can dramatically improve the test R². Why does this transformation improve the model so dramatically, even though no new information was added to the dataset?

### Polynomial Features

13. Adding X² allows the regression line to bend, yet the model is still called "linear" regression. Why is a model with squared terms still considered linear?

14. For potential polynomial features, the model might predict three different curve shapes: upward hockey stick, inverted U, and diminishing returns. What determines whether the squared term's coefficient is positive or negative?

15. You should "almost always include the base linear term" when adding a squared term. Why is it problematic to include X² without X?

16. Forward selection did not choose any polynomial features when they were added to the full model. Does this mean polynomials are never useful, or could there be another explanation?

### Multicollinearity

17. The Variance Inflation Factor (VIF) measures how much a feature's coefficient variance is inflated due to correlation with other features. Why does high VIF make individual coefficients unreliable, even if the overall model predictions are still accurate?

18. One feature has a VIF of 118 when all numeric features are included, but only 2.84 in a reduced feature set selected by forward selection. What does this tell you about how forward selection interacts with multicollinearity?

19. Two features are correlated at 0.89. If both features were included in a model, what would you expect to happen to their coefficients compared to including only one?

20. Multicollinearity "may have affected the forward selection process to begin with." Why might multicollinearity cause forward selection to make suboptimal choices about which features to include?

### The Modeling Workflow

21. A `run_modeling_workflow` function handles train-test split, cross-validation, feature selection, evaluation, and residual plotting in one call. What are the advantages of encapsulating this workflow in a function rather than running each step separately?

22. The workflow evaluates the model using both cross-validation R² (on training data) and test R² (on held-out data). Why are both metrics reported, and what does it mean if they differ significantly?

23. A strong model (R² = 0.93) is produced, but significant multicollinearity exists among the original features. What limitation does this create for interpreting the selected features?

---

## 17_2_4_3: Regularization (Ridge, Lasso, ElasticNet)

### The Problem with Unregularized Models

1. A regression model wants to minimize its error on the training data, but it will happily assign massive weights in pursuit of the best fit. Why are massive weights a problem? What do they indicate about the model?

2. Forward and backward selection reduce features. Regularization is "widely considered a much more modern and powerful approach." What are the four reasons for why regularization is preferred over stepwise selection?

3. Stepwise selection is "binary and harsh" — a feature is either 100% in or 100% out. How does regularization handle this differently?

### Ridge Regression (L2)

4. Ridge regression adds a penalty based on the squared value of the coefficients (the L2 penalty). Why does squaring the coefficients mean that Ridge "heavily punishes abnormally large coefficients"?

5. Ridge shrinks coefficients "asymptotically close to zero, but it never forces them to exactly 0.0." Why does this matter? What are the consequences for model interpretability?

6. Ridge is particularly good at handling multicollinearity. When two features are highly correlated, what does Ridge do with their coefficients?

7. Ridge requires mandatory feature scaling. Without scaling, what would happen to features measured in large units versus features measured in small units?

8. Ridge coefficients lose "traditional interpretability." If a Ridge coefficient for square feet is $40, why can't you interpret that as "each additional square foot adds exactly $40 to the price"?

### Lasso Regression (L1)

9. Lasso uses the absolute value of coefficients (L1 penalty) instead of the squared value (L2). How does this small mathematical change lead to the ability to zero out coefficients entirely?

10. When Lasso encounters two highly correlated features, it "will essentially pick one at random to keep and drop the other." Why might this be problematic for interpretation?

11. What happens if alpha is set too high in Lasso? What specific scenario does this warn about?

### ElasticNet

12. ElasticNet combines L1 and L2 penalties and "guarantees that your model will perform at least as well as Ridge or Lasso." Why is this true? Under what circumstances would ElasticNet effectively become Ridge or Lasso?

13. The `l1_ratio` parameter controls the mix between Lasso and Ridge. If you set `l1_ratio=0.3`, what does that mean in terms of how the penalty is applied?

14. "Double the hyperparameter tuning" is identified as the biggest cost of ElasticNet. Why does having two hyperparameters instead of one make such a significant difference?

### Feature Scaling and Pipelines

15. Every regularization model is wrapped in a Pipeline with StandardScaler. What would go wrong if you ran Lasso or Ridge on unscaled data?

16. OLS uses SequentialFeatureSelector in the pipeline. How does the OLS approach to feature selection differ fundamentally from how Lasso performs feature selection?

### Comparing Models

17. One model keeps all features while another keeps only 22, yet their test R² scores are very similar. What does this tell you about the value of the features that were dropped?

18. Another model keeps nearly all features. What is the practical consequence of keeping nearly all features, and why might regularization be more useful in this scenario?

19. ElasticNet is popular in fields like "bioinformatics, genomics, and econometrics." Why would these fields specifically benefit from a method that combines Ridge's handling of correlated features with Lasso's ability to zero out coefficients?

### Overfitting and the Bias-Variance Tradeoff

20. Regularization is "intentionally trading a tiny bit of accuracy on the training data (introducing bias) to achieve a massive improvement in predicting new, unseen data (reducing variance)." Explain this tradeoff in your own words. Why would adding bias to a model sometimes improve it?

21. The alpha parameter controls the strength of regularization. If you could see the training error and validation error plotted against alpha, what pattern would you expect to see as alpha increases from 0 to a very large value?

22. The alpha values used are arbitrary. What risk does this introduce, and how would you systematically find a better alpha value?

### Limitations and Modern Context

23. ElasticNet "still assumes relationships are straight lines." What kinds of patterns in data would a linear model with regularization fail to capture?

24. Researchers who "only care about predictive accuracy" often move to tree-based models. Under what circumstances would you still prefer a regularized linear model over a tree-based model?

---

## 17_2_4_4: Grid Search, Nested Cross-Validation, and Learning Curves

### Bias-Variance Tradeoff

1. In a manual alpha tuning plot, the training R² is always higher than the cross-validation R². Why?

2. What happens to both the training and cross-validation scores as alpha gets very large? What does this tell you about the model?

3. We are looking for a "Goldilocks" zone. If you had to pick an alpha based on the plot, which one would you choose and why?

### GridSearchCV

4. The `tune_model` function wraps everything in a Pipeline that includes a StandardScaler. Why can't we just scale the data before calling GridSearchCV? What problem does the pipeline solve?

5. When the param grid is passed to the function, the keys are like 'alpha', but inside the function they get prefixed to 'model__alpha'. Why is this necessary?

6. The output says "Fitting 5 folds for each of 8 candidates, totalling 40 fits." Where does the number 40 come from? How would this change if you used 10-fold cross-validation instead of 5?

7. A model has two hyperparameters. How many total combinations does GridSearchCV try with the given grid? How many total model fits does that produce with 5-fold CV?

### Test Set Evaluation

8. All three optimized models produce very similar Test R² scores. Why might that be?

9. The notebook evaluates models on the test set after using GridSearchCV to tune on the training set. What is the risk of looking at the test set scores and then going back to re-tune hyperparameters?

### Nested Cross-Validation

10. In your own words, what is "optimistic bias" and why does it occur when you select hyperparameters based on cross-validation scores?

11. Nested CV uses the full dataset (not a train/test split). Why is this an advantage? What does it buy you compared to the train/test approach?

12. In the nested CV code, GridSearchCV is passed as the estimator to cross_val_score. Trace through what happens on a single outer fold: what does the inner loop do, and what does the outer loop do with the result?

13. Nested CV is computationally expensive. For a 5-fold outer loop, 5-fold inner loop, and a model with 4 alpha values and 5 l1_ratio values, how many total fits occur?

14. Why might different outer folds select different "best" hyperparameters? What does it mean if they do?

### Comparing Approaches

15. Under what circumstances would you prefer a simple train/test split over nested cross-validation? When would nested CV be the better choice?

16. Nested CV gives "quantifiable" variance of the estimate. What does this mean?

### Learning Curves

17. Looking at a learning curve plot, what do the shaded bands around each line represent?

18. If the training score is much higher than the validation score at the largest training size, what does that suggest about the model? What would you try next?

19. If both curves converge at a low R² value and adding more data doesn't help, what does that tell you? Is this a data problem or a model problem?

20. The learning curve is plotted using a tuned model. Why does it make sense to use the tuned model for learning curves rather than an untuned one?

### Putting It All Together

21. A colleague says: "I just ran my model on the test set and got an R² of 0.92. I tuned my hyperparameters using GridSearchCV with 5-fold CV on the training set." What follow-up questions would you ask before trusting that 0.92 number?

22. Imagine you have a very small dataset (200 rows). Would you use train/test split, plain cross-validation, or nested cross-validation? Justify your answer.

23. The notebook moves from "guessing" alpha values to "optimizing" them. But is GridSearchCV truly optimization, or is it still a form of search? What are its limitations?

---

## 17_2_4_5: Tree-Based Methods

### Why Trees?

1. Linear models required manual polynomial features to capture non-linear patterns. How do tree-based models handle non-linearity without this manual effort?

2. Tree-based models capture three examples automatically: diminishing returns, threshold effects, and feature interactions. For each, describe a real estate scenario where a linear model would struggle but a tree would succeed.

3. Why are tree-based models described as taking a "completely different approach" from linear regression? What is the fundamental difference in how they make predictions?

### How a Regression Tree Works

4. In the "20 Questions" analogy, each question splits the data into two groups. What does the tree predict for a house that lands in a particular leaf node?

5. The tree does not ask questions at random. At each node, how does it decide which feature and which split point to use?

6. If we let a tree keep splitting without any limits, it achieves R² = 1.0 on training data. Why is this a problem? What is the tree actually learning at this point?

7. The key hyperparameter for controlling tree complexity is `max_depth`. What does this parameter limit, and why is it effective at preventing overfitting?

### Interpreting the Depth Experiment

8. In a depth experiment, the test R² peaks at a certain depth and then declines. Why does the training R² continue to climb while the test R² falls?

9. At a moderate depth, both training and test scores are moderate. Is this overfitting, underfitting, or a good fit? How do you know?

10. At unlimited depth, the test R² is actually worse than at a moderate depth. How can a more complex model perform worse than a simpler one on test data?

11. The notebook says this is "the same pattern we observed with polynomial features." What is the shared pattern, and what is the key difference in how complexity is controlled?

### Reading the Tree Visualization

12. The root node is described as the tree's "first and most important split." Why is the first split more important than later splits?

13. The same features appear multiple times at different depths. How does this allow trees to capture feature interactions?

14. Leaves with very few samples are described as a "warning sign." Why should we be cautious about predictions based on a small number of training examples?

15. Even at moderate depth, the tree visualization is "already becoming difficult to read." What does this tell you about the interpretability of single trees as they grow deeper?

### Random Forests: Bagging and Parallelization

16. The Random Forest introduces two sources of randomness: bootstrap sampling and feature randomness. What problem does each one solve?

17. If every tree in a Random Forest saw the exact same data and considered all features, what would happen to the ensemble's predictions? Why wouldn't averaging help?

18. A tree in the forest "might never see the most expensive mansion in the dataset." How does bootstrap sampling (drawing with replacement) make this possible?

19. What does `max_features='sqrt'` mean in practice? If there are 37 features, approximately how many does each tree consider at each split?

20. Random Forest chose unlimited depth as optimal. Why is unlimited depth acceptable in a forest but catastrophic for a single tree?

21. From a bias-variance perspective, what is the Random Forest's primary job? How does averaging diverse trees accomplish this?

### Gradient Boosting: Sequential Refinement

22. In gradient boosting, the first prediction is the average of all training targets. What does the first tree predict, and what does the second tree predict?

23. A concrete example: a house was predicted at 12.0 but the actual value was 11.2 (residual = -0.8). After a tree corrects by -0.7, the new prediction is 11.3. Why doesn't the tree correct by the full -0.8?

24. The assembly line analogy describes each worker "sanding down the exact rough edges left by the person before them." How does this map to the boosting process?

25. Compare Random Forests and Gradient Boosting using the bias-variance framework. Which reduces variance and which reduces bias? Why?

26. What does `HistGradientBoostingRegressor` do differently from a standard gradient boosting implementation, and why does this make it faster?

### Robustness to Messy Data

27. In earlier parts, significant data cleaning was required. A raw-data tree-based model achieved nearly identical performance to the cleaned version. What does this tell you about tree-based models vs. linear models?

28. What specific cleaning steps were not needed for tree-based models that were needed for linear models?

29. Does this result mean data cleaning is never necessary? When might thoughtful feature engineering still help a tree-based model?

### XGBoost

30. XGBoost applies L1 and L2 regularization to leaf weights, not to linear coefficients. How is this similar to Ridge and Lasso, and how is it different?

31. Boosting is inherently sequential, making it hard to parallelize. How does XGBoost achieve speed improvements despite this constraint?

32. At each split, XGBoost tests sending missing values left or right. Why is this better than imputing missing values with the mean or median?

### Feature Importance

33. Feature importance tells you what the model prioritizes but not how the feature affects the prediction. Contrast this with a linear regression coefficient. What can a coefficient tell you that feature importance cannot?

34. A feature accounts for a large percentage of the model's learning. What does this percentage actually measure? (Hint: it is not the percentage of samples correctly predicted.)

35. A simple binary feature achieved high importance despite being a simple yes/no. Why might a simple binary feature score so highly?

### Correlated Features and Feature Importance

37. A feature has very high VIF (extreme multicollinearity) but only a small feature importance. Why does a highly predictive feature receive such a low importance score?

38. If two features are highly correlated, the tree model picks one for its splits and the other gets no credit. How is this similar to the multicollinearity problem in linear regression, and how is it different?

39. The notebook suggests comparing feature importance across multiple models. Why would this help identify the correlated-features problem?

### Model Comparison Summary

40. The model comparison shows a progression from Decision Tree to Random Forest to Gradient Boosting to XGBoost. What does this progression tell you about the tradeoff between model complexity and accuracy?

41. One model's single-split score was higher than its nested CV estimate. What causes this gap, and why should you trust the nested CV number more?

42. Even the conservative nested CV estimate is well above the best linear model from earlier parts. What does this difference suggest about the suitability of tree-based models for structured tabular data?

### Putting It All Together

43. A colleague says: "I don't need to clean my data or engineer features — I'll just throw it into XGBoost." Based on what you've learned, how would you respond?

44. The notebook recommends starting with a Random Forest for most real-world problems. Why not start with XGBoost, which is more accurate?

45. You have a dataset with 500 rows and 30 features. Your goal is to build the most accurate predictive model possible. Would you use a single decision tree, a Random Forest, or XGBoost? Justify your choice considering both accuracy and the risk of overfitting.