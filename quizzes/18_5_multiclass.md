# 18.5 · Multi-Class Classification

## 18_5_1: Multiclass Foundations

### With three penguin species as the target, what does predict_proba return for a single test row?
- [x] A row of three probabilities, one per species, that sums to 1
- [ ] A single probability between 0 and 1 for the most likely species
- [ ] Three independent yes/no probabilities that do not need to sum to 1
- [ ] The Gini impurity of the leaf node each species landed in

### What does objective='multi:softprob' tell XGBoost to do?
- [x] Output a full probability distribution across all classes for each observation
- [ ] Apply a softmax penalty to overconfident trees during training
- [ ] Predict only the single most probable class label with no probabilities
- [ ] Convert the target to one-hot columns before growing any trees

### In the 3×3 confusion matrix (rows = actual, columns = predicted), what does the cell in row Adelie, column Gentoo count?
- [x] Actual Adelie penguins the model misclassified as Gentoo
- [ ] Actual Gentoo penguins the model misclassified as Adelie
- [ ] Penguins the model correctly classified as either Adelie or Gentoo
- [ ] The number of features shared by Adelie and Gentoo penguins

### In one-hot encoding of a three-class target (Adelie = [1, 0, 0], Chinstrap = [0, 1, 0], Gentoo = [0, 0, 1]), why does each vector contain exactly one 1?
- [x] Each observation belongs to exactly one class
- [ ] It keeps the feature matrix sparse to save memory
- [ ] XGBoost requires binary inputs for every column
- [ ] It guarantees the three classes stay equally frequent

## 18_5_2: Per-Class Metrics and Averaging

### A model scores high overall accuracy on the penguins, yet performs noticeably worse on one species. What does the classification report provide that accuracy alone hides?
- [x] Precision, recall, and F1 broken out separately for every class
- [ ] The probability that the accuracy estimate is statistically significant
- [ ] The feature importances responsible for each misclassification
- [ ] A corrected accuracy value that removes the effect of class size

### In the multiclass setting, what is the precision for the Chinstrap class?
- [x] Of all penguins the model predicted to be Chinstrap, the fraction that actually are
- [ ] Of all actual Chinstrap penguins, the fraction the model found
- [ ] The fraction of all penguins the model classified correctly
- [ ] The fraction of Chinstrap predictions that the model made with high confidence

### What is the key difference between macro-averaged and weighted-averaged F1?
- [x] Macro gives every class equal weight; weighted counts frequent classes more
- [ ] Macro uses the median F1; weighted uses the mean F1
- [ ] Macro is computed on the test set; weighted is computed on the training set
- [ ] Macro applies only to balanced data; weighted applies only to imbalanced data

### A model's weighted F1 is high but its macro F1 is much lower. What does this gap diagnose?
- [x] The model performs well on the common classes but poorly on the rare ones
- [ ] The model is overfitting the training data and needs regularization
- [ ] The test set is too small for reliable evaluation
- [ ] The model's probabilities are poorly calibrated across classes

## 18_5_3: Imbalanced Multiclass

### In the fetal health data, most recordings are Normal. A weak model posts high accuracy while missing most Pathological cases. Why is accuracy misleading here?
- [x] Predicting the dominant class for nearly everything already yields high accuracy
- [ ] Accuracy is undefined when a dataset has more than two classes
- [ ] Accuracy double-counts observations from the majority class
- [ ] Medical datasets require regression metrics rather than classification metrics

### What does compute_sample_weight(class_weight='balanced') do?
- [x] Gives each observation a weight inversely proportional to its class frequency, so rare classes carry equal total influence in training
- [ ] Removes observations from the majority class until all classes are the same size
- [ ] Duplicates minority-class rows until the dataset is balanced
- [ ] Adjusts the decision threshold of the trained model to favor rare classes

### Why use StratifiedKFold rather than ordinary KFold when cross-validating on imbalanced multiclass data?
- [x] It preserves each class's proportion in every fold, so rare classes appear in all of them
- [ ] It shuffles the data more thoroughly than ordinary KFold
- [ ] It creates larger validation folds for the rare classes
- [ ] It guarantees every fold produces exactly the same score

### When the rare class is the clinically critical one (Pathological fetal health), which evaluation focus does the notebook recommend over accuracy?
- [x] Macro F1 together with the rare class's own recall
- [ ] Weighted F1, because it reflects the true class distribution
- [ ] Overall accuracy measured on a rebalanced copy of the test set
- [ ] Training-set log-loss, since it is what XGBoost optimizes
