# 18_5 Ensemble Methods — Glossary

This document defines all technical and conceptual terms used across the five notebooks in the 18_5 Ensemble Methods series. Terms already defined in the 18_1 Classification Basics glossary are noted with cross-references.

---

## A

### AdaBoost (Adaptive Boosting)
The original boosting algorithm. It trains weak learners (typically decision stumps) sequentially, increasing the weights of misclassified samples after each round so that subsequent learners focus on the hardest cases. Final predictions are made by weighted voting, where better-performing learners get more weight.

### Accuracy Paradox
See **18_1 Glossary: Accuracy Paradox**. In the medical context of 18_5, a model can achieve high accuracy by correctly classifying most benign tumors while missing many malignant ones — making it clinically useless.

### AUC (Area Under the ROC Curve)
See **18_1 Glossary: AUC**. In 18_5, an AUC of 0.99 means the model correctly ranks a random malignant tumor above a random benign tumor 99% of the time.

---

## B

### Bagging (Bootstrap Aggregating)
An ensemble technique that reduces variance by training multiple models independently on bootstrap samples (random samples drawn with replacement) and averaging their predictions. Each model overfits to its specific sample, but the errors differ across models, so averaging cancels out the noise.

### BART (Bayesian Additive Regression Trees)
A Bayesian ensemble method that combines an ensemble of shallow trees with MCMC (Markov Chain Monte Carlo) sampling. Unlike boosting, BART does not build trees sequentially to correct errors — instead, it randomly perturbs the entire ensemble (adding branches, removing splits, changing thresholds) and accepts changes that improve the fit. The key advantage is **uncertainty quantification**: BART provides credible intervals for each prediction.

### Bias
See **18_1 Glossary**. In the ensemble context: single trees have low bias (they can fit complex patterns) but high variance. Boosting reduces bias by sequentially correcting errors.

### Bootstrap Sampling
Random sampling with replacement from the training data. Each bootstrap sample is the same size as the original dataset but contains some duplicate observations and omits others (~37% of samples are left out on average). Used in bagging and random forests to create diverse training sets.

---

## C

### Calibration (Probability Calibration)
The degree to which a model's predicted probabilities match actual outcome frequencies. A well-calibrated model that predicts 80% probability of malignancy should be correct about 80% of the time. Random Forests tend to be well-calibrated; Gradient Boosting often produces probabilities that are too extreme.

### Class Weight (Balanced)
See **18_1 Glossary: Class Weight (Balanced)**. In 18_5, using `class_weight='balanced'` with Random Forests shifts the decision boundary to catch more malignant tumors at the cost of more false positives (unnecessary biopsies).

### Confusion Matrix
See **18_1 Glossary: Confusion Matrix**. In 18_5, the medical interpretation is critical: false negatives = missed cancers, false positives = unnecessary biopsies.

### Correlated Features Problem
When two or more features are highly correlated, tree-based models tend to use one for splits and ignore the others. This splits the importance score among correlated features, making none appear as dominant as it truly is. Permutation importance is a more reliable alternative.

---

## D

### Decision Stump
A decision tree with a single split (max_depth=1). It is a "weak learner" — only slightly better than random guessing. Used as the base learner in AdaBoost because the sequential correction process adds value only when individual learners are weak.

### Decision Tree
See **18_1 Glossary**. In 18_5, we contrast classification trees (majority class prediction at leaves, Gini impurity for splits) with the regression trees from 17_2_4_5 (mean prediction at leaves, MSE for splits).

---

## E

### Ensemble Learning
A technique that combines multiple models (typically decision trees) to produce a single, more accurate prediction. The core idea is the "wisdom of the crowd": averaging diverse opinions cancels out individual errors. Main types: bagging (parallel, variance reduction), boosting (sequential, bias reduction), and Bayesian ensembles (uncertainty quantification).

---

## F

### F1-Score
See **18_1 Glossary: F1-Score**. In 18_5, the F1-score is used as the scoring metric for GridSearchCV because it balances precision and recall — important when both missed cancers and unnecessary biopsies matter.

### False Negative (FN) / Type II Error
In the medical context of 18_5: the model predicts benign but the tumor is actually malignant. A **missed cancer**. This is typically the more costly error in clinical settings.

### False Positive (FP) / Type I Error
In the medical context of 18_5: the model predicts malignant but the tumor is actually benign. An **unnecessary biopsy**. Less costly than a false negative but still has clinical and financial consequences.

### Feature Importance
See **18_1 Glossary: Feature Importance**. Computed as the total reduction in Gini impurity attributed to splits on each feature across all trees in the ensemble. Limitation: correlated features share importance, so none appears as dominant as it truly is.

---

## G

### Gini Impurity
A measure of node purity used by decision trees to choose splits. `Gini = 1 - Σ(p_i²)` where p_i is the proportion of class i in the node. Gini = 0 means the node is pure (all one class); Gini = 0.5 means maximally mixed (50/50 for binary classification). The tree picks the split that produces the greatest reduction in Gini impurity.

### Gradient Boosting
An ensemble technique that builds trees sequentially, where each new tree predicts the residuals (errors) of the ensemble so far. The tree's predictions are added to the ensemble, scaled by the learning rate (shrinkage). Reduces bias more effectively than bagging but is more prone to overfitting if not carefully tuned.

### GridSearchCV
See **18_1 Glossary** (indirectly, via 17_2_4_4). In 18_5, used to tune hyperparameters for Random Forests (n_estimators, max_depth, max_features) and Gradient Boosting (n_estimators, learning_rate, max_depth).

---

## H

### Harmonic Mean
See **18_1 Glossary: Harmonic Mean**. Used in the F1-score to ensure that both precision and recall must be high for the F1 to be high.

---

## L

### Learning Rate (Shrinkage)
In Gradient Boosting, a factor that scales each tree's contribution to the ensemble. Smaller learning rates (e.g., 0.01 vs. 0.1) require more trees but often produce more robust models that generalize better. Think of it as learning slowly and deliberately vs. rushing to conclusions.

---

## M

### Malignant Recall
The recall score computed specifically for the malignant (positive) class. In the breast cancer context, this is the most clinically relevant metric: it tells us what fraction of actual cancers the model catches.

### MCMC (Markov Chain Monte Carlo)
A family of algorithms used in Bayesian inference to sample from complex probability distributions. Used by BART to randomly perturb the tree ensemble and explore different model configurations.

---

## N

### Nested Cross-Validation
A two-layer cross-validation approach: the inner loop tunes hyperparameters within each outer training fold, and the outer loop evaluates the tuned model on an independent holdout fold. Provides an unbiased estimate of model performance, accounting for the fact that hyperparameter selection itself introduces optimistic bias. Used in 18_5_4 for the final model comparison.

### N Estimators
The number of trees in an ensemble model (bagging, random forest, boosting). More trees generally improve performance up to a point, after which returns diminish. Tuning often finds that fewer trees (e.g., 100) are sufficient.

---

## O

### OOB Score (Out-of-Bag Score)
A built-in validation estimate available for bagging-based methods. Because bootstrap sampling leaves out ~37% of data for each tree, those "out-of-bag" samples serve as a validation set. The OOB score is the accuracy on these held-out samples, averaged across all trees. Useful because it eliminates the need for a separate validation set.

### One-vs-Rest (OvR)
See **18_1 Glossary: One-vs-Rest (OvR)**. Not directly used in 18_5 (all problems are binary), but mentioned in the context of multiclass extensions.

### Overfitting
See **18_1 Glossary**. In 18_5, demonstrated through the depth experiment: training accuracy reaches 1.0 while test accuracy plateaus or drops. Ensembles (bagging, RF, boosting) are designed to combat overfitting.

---

## P

### Permutation Importance
An alternative to impurity-based feature importance. It measures how much model performance drops when a feature's values are randomly shuffled. More reliable than impurity-based importance when features are correlated, because it evaluates each feature's contribution independently.

### Precision
See **18_1 Glossary: Precision**. In the medical context: when the model says "malignant," how often is it right? High precision means few unnecessary biopsies.

### Probability Calibration
See **Calibration (Probability Calibration)**.

---

## R

### Random Forest
An ensemble of decision trees where each tree is trained on a bootstrap sample and, at each split, considers only a random subset of features (typically √p). This decorrelates the trees, making the averaging more effective than plain bagging. Reduces variance.

### Recall (Sensitivity)
See **18_1 Glossary: Recall**. In the medical context of 18_5: of all actual malignant tumors, what percentage did the model catch? This is the most critical metric for clinical deployment.

### ROC Curve (Receiver Operating Characteristic)
See **18_1 Glossary: ROC Curve**. In 18_5, used to compare the discrimination ability of different ensemble methods.

---

## S

### Shrinkage
See **Learning Rate (Shrinkage)**.

### Softmax (Multinomial)
See **18_1 Glossary: Softmax (Multinomial)**. Not used in 18_5 (all problems are binary), but mentioned in the context of multiclass extensions.

### Specificity
See **18_1 Glossary: Specificity**. Equal to 1 − FPR. In the medical context: of all benign tumors, what percentage did the model correctly identify as benign?

### Stratified Split
See **18_1 Glossary: Stratified Split**. Used in 18_5 to ensure the 63/37 class distribution is preserved in both training and test sets.

### Stump
See **Decision Stump**.

### Support
See **18_1 Glossary: Support**. In 18_5, the malignant class has fewer samples (~37%), so its metrics have higher variance.

---

## T

### True Negative (TN)
The model correctly predicts benign and the tumor is actually benign.

### True Positive (TP)
The model correctly predicts malignant and the tumor is actually malignant.

---

## V

### Variance (Model)
See **18_1 Glossary: Variance (Model)**. In 18_5, bagging and random forests reduce variance by averaging many diverse trees. The standard deviation of cross-validation scores is a practical measure of variance.

---

## W

### Weak Learner
A model that performs only slightly better than random guessing (e.g., a decision stump with accuracy ~55-60%). AdaBoost combines hundreds of weak learners to create a strong ensemble. In contrast, bagging uses strong learners (deep trees) and averages them.

### Weighted Average
See **18_1 Glossary: Weighted Average**.

### Wisdom of the Crowd
The principle that averaging the predictions of many diverse, independent models produces a more accurate result than any single model. The foundation of all ensemble methods.
