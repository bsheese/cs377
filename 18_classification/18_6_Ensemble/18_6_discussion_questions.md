# Discussion Questions: 18_6 — Trees and Ensemble Methods

---

## 18_6_1: Decision Trees

### How Trees Learn

1. A decision tree chooses splits by minimizing Gini impurity. In plain English, what does a node with Gini impurity of 0 contain? What about a node with the maximum possible impurity for two classes?
2. Decision trees require no feature scaling and handle non-linear relationships naturally. What is it about the splitting mechanism that makes both of these true?
3. The notebook's complexity curve shows training accuracy rising toward 100% as `max_depth` grows while test accuracy stalls and then degrades. Explain what the tree is doing at depth 15 that it is not doing at depth 3.
4. You can read a fitted decision tree as a set of if/then rules. Why does this interpretability advantage disappear when you move to an ensemble of hundreds of trees? Is anything left in its place?

### Evaluating on Medical Data

5. On the Wisconsin breast cancer data, which error is worse: classifying a benign tumor as malignant, or a malignant tumor as benign? Which metric from 18_1 tracks the error you care about most?
6. Why does the notebook bother with k-fold cross-validation when it already has a train/test split? What can 5 folds tell you that one split cannot?

## 18_6_2: Bagging and Random Forests

### Why Averaging Works

7. Bagging trains each tree on a bootstrap sample (drawn with replacement) of the training data. Why does deliberately training on *noisier* versions of the data produce a *more* stable model?
8. Single decision trees are described as "high variance." What does variance mean here — variance of what, across what? How does averaging many trees reduce it?
9. Bagged trees are grown deep (unpruned) on purpose, even though a deep single tree overfits. Why is overfitting in the individual trees acceptable — even desirable — inside a bagging ensemble?
10. Out-of-bag (OOB) error gives a validation estimate "for free." Where does the free validation data come from, and why is each tree only scored on some of the rows?

### The Random Forest Fix

11. If bagging already averages many trees, what problem is left for random forests to solve? What does restricting each split to a random subset of features accomplish that bootstrapping alone does not?
12. Suppose one feature (say, `worst concave points`) is by far the strongest predictor. Describe what happens to a bagged ensemble's trees without feature subsampling, and why that hurts the ensemble.
13. The notebook shows that feature importance is unreliable when features are correlated. If two nearly identical features split the importance credit between them, what mistake might an analyst make when reading the importance chart?

## 18_6_3: Boosting

### Sequential Learning

14. Bagging trains trees independently and in parallel; boosting trains them sequentially. What information does boosting pass from one tree to the next, and why does that make the trees deliberately *weak* (stumps or shallow)?
15. AdaBoost re-weights misclassified samples; gradient boosting fits each new tree to the current residuals. Explain how these are two versions of the same idea: "concentrate on what we're still getting wrong."
16. Boosting reduces bias, while bagging reduces variance. Connect this to the models each one starts from: why does it make sense that an ensemble of stumps needs bias reduction while an ensemble of deep trees needs variance reduction?
17. The learning rate shrinks each tree's contribution. Why do a *smaller* learning rate and *more* trees often generalize better than a large learning rate with few trees? What is the cost?
18. Boosted models can achieve excellent accuracy while producing poorly calibrated probability estimates. Why does calibration matter if all you report to the clinician is a probability of malignancy?

## 18_6_4: Model Comparison

### Comparing Fairly

19. The final notebook uses *nested* cross-validation to compare tree, forest, and boosting models — the same machinery as 17_2's Part 5. Why would tuning hyperparameters and reporting the score from the same cross-validation loop overstate every model's performance? Does it overstate all models equally?
20. Two models finish within a fraction of a percentage point of each other, but their fold-to-fold standard deviations overlap heavily. What should you conclude, and what would you look at next to choose between them (consider the malignant-recall story and interpretability)?
21. After nested CV selects a winner, the final model is refit on the full dataset. Why is it legitimate to train the shipped model on all the data when every performance number you quoted came from held-out folds?
22. Across the whole series — single tree, bagging, random forest, boosting — state each method's one-sentence answer to the question "where do you get your edge?"
