# 18.1 · Classification Basics

## Part 1: Classification Foundations

### In XGBoost, what does the scale_pos_weight parameter do?
- [ ] Increases the learning rate for predictions on the positive class
- [x] Scales the weight of the positive class to handle class imbalance
- [ ] Normalizes all predicted probabilities to fall between 0 and 1
- [ ] Rescales feature values to have zero mean and unit variance

### For the South German Credit dataset with 70% good and 30% bad credit, what is the naive baseline accuracy?
- [ ] 30%
- [ ] 50%
- [x] 70%
- [ ] 100%

### XGBoost builds an ensemble of decision trees:
- [x] Sequentially, each one correcting the errors of the prior ensemble
- [ ] Sequentially, with each tree trained on an independent bootstrap sample
- [ ] In parallel, with the final output determined by majority vote

### Why is stratified splitting important for imbalanced datasets?
- [ ] It forces the training and test sets to have equal total sizes
- [x] It ensures both train and test sets preserve the same class proportions
- [ ] It automatically adjusts class weights during model training
- [ ] It removes minority class samples to make the split more manageable

## Part 2: Confusion Matrix & Basic Metrics

### In a confusion matrix, what does a False Negative represent?
- [ ] Predicting default when the customer is actually a good payer
- [x] Predicting good when the customer actually defaults
- [ ] Correctly identifying a customer who defaults
- [ ] Correctly identifying a customer who is a good payer

### Precision measures:
- [ ] Of all actual positives, how many were correctly identified
- [x] Of all positive predictions, how many are actually correct
- [ ] The proportion of total predictions that are correct overall
- [ ] The ratio of true positives to total true negatives

### Recall measures:
- [ ] Of all positive predictions, how many are actually correct
- [x] Of all actual positives, how many were correctly identified
- [ ] The proportion of total predictions that are correct overall
- [ ] The ratio of true positives to total predictions made

### Why does F1-score use harmonic mean instead of arithmetic mean?
- [ ] Harmonic mean is computationally less expensive to calculate
- [x] Harmonic mean gives more weight to lower values, penalizing extreme imbalances between precision and recall
- [ ] Arithmetic mean cannot be applied to percentage-valued metrics
- [ ] Harmonic mean produces higher scores than arithmetic mean for balanced classifiers

### If precision = 100% and recall = 1%, what is the F1-score?
- [ ] About 50% — the average of perfect precision and near-zero recall
- [x] About 2% — the harmonic mean heavily penalizes the near-zero recall
- [ ] About 100% — perfect precision dominates the score
- [ ] About 25% — the geometric mean of precision and recall

### When would you prefer weighted average over macro average?
- [ ] When all classes are equally important regardless of their sample size
- [x] When you want overall performance weighted by the dataset's class distribution
- [ ] When you want to treat all classes equally regardless of sample size
- [ ] When the dataset has a perfectly balanced class distribution

### If you raise the decision threshold from 0.5 to 0.7, what happens to precision and recall?
- [ ] Both precision and recall increase
- [ ] Both precision and recall decrease
- [x] Precision increases, recall decreases
- [ ] Precision decreases, recall increases

### Type I error is another name for:
- [ ] True Negative
- [ ] False Negative
- [x] False Positive
- [ ] True Positive

### What does the ROC curve plot?
- [ ] Precision vs. Recall at varying decision thresholds
- [x] True Positive Rate vs. False Positive Rate at varying thresholds
- [ ] Accuracy vs. F1-Score at varying decision thresholds
- [ ] Training loss vs. validation loss across training epochs

## Part 4: ROC, AUC & Threshold Tuning

### An AUC of 0.79 means:
- [ ] The model correctly classifies 79% of all samples in the dataset
- [x] The model correctly ranks a random positive above a random negative 79% of the time
- [ ] 79% of all model predictions are true positives
- [ ] The model achieves 79% precision on the positive class

### Why is AUC described as threshold-independent?
- [ ] It assumes a fixed decision threshold of 0.5 for all calculations
- [x] It evaluates performance across all possible decision thresholds simultaneously
- [ ] It only applies to binary classification problems with balanced classes
- [ ] It does not require any threshold to compute the score

### Youden's J statistic is calculated as:
- [ ] TPR + FPR
- [x] TPR - FPR
- [ ] Precision + Recall
- [ ] Accuracy - Baseline accuracy

### Why might ROC curves be over-optimistic on imbalanced data?
- [x] ROC includes true negatives in the FPR calculation, which inflates performance
- [ ] ROC curves ignore false positives when computing the true positive rate
- [ ] ROC is unaffected by class imbalance and always gives accurate results
- [ ] ROC uses precision in place of recall, which benefits the majority class

### What is the baseline for a Precision-Recall curve?
- [ ] 0.5, representing a random classifier on balanced data
- [x] The positive class prevalence in the dataset
- [ ] The majority class proportion in the dataset
- [ ] 1.0, representing a perfect classifier

### If false negatives cost more than false positives, where should you set the threshold?
- [ ] At 0.5, the standard default threshold
- [ ] Higher than 0.5, to make the classifier more conservative
- [x] Lower than 0.5, to flag more positives and reduce missed cases
- [ ] At 1.0, to eliminate all false positives

### A perfect ROC curve would:
- [ ] Follow the diagonal line from (0,0) to (1,1)
- [x] Hug the top-left corner of the plot
- [ ] Form a horizontal line at y = 0.5
- [ ] Form a vertical line at x = 0

### A wide box in a cross-validation boxplot indicates:
- [ ] The model achieves high accuracy on every fold
- [x] The model's performance varies significantly across different data splits
- [ ] The model has high bias and underfits every fold
- [ ] The model is underfitting the training data

### Why do we score models on both accuracy AND F1?
- [ ] F1 is always more important and reliable than accuracy
- [x] Accuracy can be misleading on imbalanced data; F1 reveals minority class performance
- [ ] Accuracy is unreliable on large datasets; F1 is more robust to dataset size
- [ ] They always produce the same result on balanced datasets

### If Random Forest has the highest mean F1 but also the widest spread, you should:
- [ ] Choose it because it has the highest mean performance
- [x] Consider both mean performance and stability; a more consistent model may be preferable
- [ ] Choose the lowest variance model regardless of mean performance
- [ ] Replace cross-validation with a single train/test split instead
