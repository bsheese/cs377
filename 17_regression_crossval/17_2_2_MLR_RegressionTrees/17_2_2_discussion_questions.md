# Discussion Questions: 17_2_2 Regression Trees

---

## 17_2_2: Tree-Based Methods

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