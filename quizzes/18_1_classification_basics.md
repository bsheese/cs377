# 18.1 · Classification Basics

## Part 1: Classification Foundations

### In XGBoost, what does the scale_pos_weight parameter do?
- [ ] Increases the learning rate for positive class predictions
- [x] Scales the weight of the positive class to handle class imbalance
- [ ] Normalizes all predictions to be between 0 and 1
- [ ] Scales feature values to have zero mean and unit variance

### For the South German Credit dataset with 70% good and 30% bad credit, what is the naive baseline accuracy?
- [ ] 30%
- [ ] 50%
- [x] 70%
- [ ] 100%

### XGBoost builds an ensemble of decision trees:
- [x] Sequentially, correcting errors from prior trees
- [ ] Sequentially, fixing errors from earlier trees
- [ ] Random selection without specific order

### Why is stratified splitting important for imbalanced datasets?
- [ ] It makes training and test sets proportionally equal
- [x] It ensures both train and test sets have the same class proportions
- [ ] It automatically balances the class weights during training
- [ ] It removes minority class samples to simplify training

## Part 2: Confusion Matrix & Basic Metrics

### In a confusion matrix, what does a False Negative represent?
- [ ] Predicting default when the customer is actually good
- [x] Predicting good when the customer actually defaults
- [ ] Correctly identifying a defaulter
- [ ] Correctly identifying a good customer

### Precision measures:
- [ ] Of all actual positives, how many were correctly identified
- [x] Of all positive predictions, how many are actually correct
- [ ] The total number of correct predictions
- [ ] The ratio of true positives to true negatives

### Recall measures:
- [ ] Of all positive predictions, how many are actually correct
- [x] Of all actual positives, how many were correctly identified
- [ ] The total number of correct predictions
- [ ] The ratio of true positives to total predictions

### Why does F1-score use harmonic mean instead of arithmetic mean?
- [ ] Harmonic mean is faster to compute
- [x] Harmonic mean gives more weight to lower values, penalizing extreme imbalances
- [ ] Arithmetic mean cannot handle percentages
- [ ] Harmonic mean always gives higher scores

### If precision = 100% and recall = 1%, what is the F1-score?
- [ ] About 50%
- [x] About 2%
- [ ] About 100%
- [ ] About 25%

### When would you prefer weighted average over macro average?
- [ ] When all classes are equally important
- [x] When you want overall performance on the dataset population
- [ ] When you want to treat all classes equally regardless of sample size
- [ ] When the dataset is perfectly balanced

### If you raise the decision threshold from 0.5 to 0.7, what happens to precision and recall?
- [ ] Both increase
- [ ] Both decrease
- [x] Precision increases, recall decreases
- [ ] Precision decreases, recall increases

### Type I error is another name for:
- [ ] True Negative
- [ ] False Negative
- [x] False Positive
- [ ] True Positive

### What does the ROC curve plot?
- [ ] Precision vs Recall
- [x] True Positive Rate vs False Positive Rate
- [ ] Accuracy vs F1-Score
- [ ] Loss vs Epoch

## Part 4: ROC, AUC & Threshold Tuning

### An AUC of 0.79 means:
- [ ] The model correctly classifies 79% of samples
- [x] The model correctly ranks a random positive above a random negative 79% of the time
- [ ] 79% of predictions are true positives
- [ ] The model has 79% precision

### Why is AUC described as threshold-independent?
- [ ] It uses a fixed threshold of 0.5
- [x] It evaluates performance across all possible thresholds
- [ ] It only works with binary classification
- [ ] It doesn't require a threshold to compute

### Youden's J statistic is calculated as:
- [ ] TPR + FPR
- [x] TPR - FPR
- [ ] Precision + Recall
- [ ] Accuracy - Baseline

### Why might ROC curves be over-optimistic on imbalanced data?
- [x] ROC includes true negatives in the calculation
- [ ] ROC curves ignore false positives
- [ ] ROC is not affected by class imbalance
- [ ] ROC uses precision instead of recall

### What is the baseline for a Precision-Recall curve?
- [ ] 0.5 (random)
- [x] The positive class prevalence
- [ ] The majority class proportion
- [ ] 1.0 (perfect)

### If false negatives cost more than false positives, where should you set the threshold?
- [ ] At 0.5 (default)
- [ ] Higher than 0.5 (more conservative)
- [x] Lower than 0.5 (more aggressive)
- [ ] At 1.0

### A perfect ROC curve would:
- [ ] Follow the diagonal line
- [x] Hug the top-left corner
- [ ] Be a horizontal line at y=0.5
- [ ] Be a vertical line at x=0

### A wide box in a cross-validation boxplot indicates:
- [ ] The model is always accurate
- [x] The model's performance varies significantly across different data splits
- [ ] The model has high bias
- [ ] The model is underfitting

### Why do we score models on both accuracy AND F1?
- [ ] F1 is always more important than accuracy
- [x] Accuracy can be misleading on imbalanced data; F1 reveals how well the model handles the minority class
- [ ] Accuracy is not always reliable; F1 provides a better measure for imbalanced classes
- [ ] They always give the same result

### If Random Forest has the highest mean F1 but also the widest spread, you should:
- [ ] Always choose it because it has the highest mean
- [x] Consider both mean performance and stability; a slightly lower but more consistent model might be preferable
- [ ] Choose the lowest variance model regardless of mean
- [ ] Use a single train/test split instead of CV
