# 18.1 · Classification Basics

## Part 1: Classification Foundations

### In XGBoost, what does the scale_pos_weight parameter do?
- [ ] Increases the learning rate for positive class predictions
- [x] Scales the weight of the positive class to handle class imbalance
- [ ] Normalizes all predictions to be between 0 and 1
- [ ] Scales feature values to have zero mean and unit variance

### For the German Credit dataset with 70% good and 30% bad credit, what is the naive baseline accuracy?
- [ ] 30%
- [ ] 50%
- [x] 70%
- [ ] 100%

### XGBoost builds an ensemble of decision trees:
- [x] Sequentially, correcting errors from prior trees
- [ ] In parallel, averaging predictions across all trees
- [ ] Randomly, selecting the best tree at the end

### Why is stratified splitting important for imbalanced datasets?
- [ ] It makes training and test sets proportionally equal in size
- [x] It ensures both train and test sets have the same class proportions
- [ ] It automatically balances the class weights during training
- [ ] It removes minority class samples to simplify training

### Why do tree-based models like XGBoost NOT require feature scaling?
- [ ] They use gradient descent, which is scale-invariant
- [x] They make threshold-based splits, so relative rank matters more than magnitude
- [ ] They normalize features internally before training
- [ ] They only work with binary features

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

### Using the Denominator Trick: precision's denominator is ___; recall's denominator is ___.
- [ ] actual positives; predicted positives
- [x] predicted positives; actual positives
- [ ] all samples; all positives
- [ ] true positives; all samples

### Why does F1-score use harmonic mean instead of arithmetic mean?
- [ ] Harmonic mean is faster to compute
- [x] Harmonic mean penalizes extreme imbalances between precision and recall
- [ ] Arithmetic mean cannot handle percentages
- [ ] Harmonic mean always gives higher scores

### If precision = 100% and recall = 1%, what is the F1-score?
- [ ] About 50%
- [x] About 2%
- [ ] About 100%
- [ ] About 25%

### When would you prefer weighted average over macro average?
- [ ] When all classes are equally important regardless of size
- [x] When you want overall performance reflecting the dataset's class distribution
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

## Part 3: Examples and Practice

### In the Adult Census Income dataset (~76% earn <=50K, ~24% earn >50K), what is the naive baseline accuracy?
- [ ] 24%
- [ ] 50%
- [x] 76%
- [ ] 100%

### After training XGBoost on the Adult Census dataset, the '>50K' class has lower precision and recall than the '<=50K' class. What is the most likely reason?
- [ ] XGBoost cannot handle more than two classes
- [x] The model sees fewer positive examples during training, making the minority class harder to predict
- [ ] The features in the dataset are not relevant to income
- [ ] scale_pos_weight was not set correctly

### For the Adult Census task, a false negative means the model predicted '<=50K' for someone who actually earns '>50K'. In a marketing context, what is the business cost of this error?
- [ ] Wasted marketing budget on an uninterested prospect
- [x] A missed high-income prospect who would have responded to a premium offer
- [ ] Lower model accuracy on the training set
- [ ] Higher precision on the positive class

## Part 4: ROC, AUC & Threshold Tuning

### What does the ROC curve plot?
- [ ] Precision vs Recall
- [x] True Positive Rate vs False Positive Rate
- [ ] Accuracy vs F1-Score
- [ ] Loss vs number of trees

### An AUC of 0.79 means:
- [ ] The model correctly classifies 79% of samples
- [x] The model correctly ranks a random positive above a random negative 79% of the time
- [ ] 79% of predictions are true positives
- [ ] The model has 79% precision

### Why is AUC described as threshold-independent?
- [ ] It uses a fixed threshold of 0.5
- [x] It evaluates performance across all possible thresholds simultaneously
- [ ] It only works with binary classification
- [ ] It doesn't require a threshold to compute

### Youden's J statistic is calculated as:
- [ ] TPR + FPR
- [x] TPR - FPR
- [ ] Precision + Recall
- [ ] Accuracy - Baseline

### Why might ROC curves be over-optimistic on imbalanced data?
- [x] The large number of true negatives keeps FPR artificially low even with many false alarms
- [ ] ROC curves ignore false positives entirely
- [ ] ROC is not affected by class imbalance
- [ ] ROC uses precision instead of recall

### What is the baseline for a Precision-Recall curve?
- [ ] 0.5 (random)
- [x] The positive class prevalence
- [ ] The majority class proportion
- [ ] 1.0 (perfect)

### If false negatives cost more than false positives, where should the threshold be set relative to 0.5?
- [ ] At 0.5 (default)
- [ ] Higher than 0.5 (more conservative)
- [x] Lower than 0.5 (flag more positives, catch more true cases)
- [ ] At 1.0

### Youden's J is the most appropriate threshold selection method when:
- [ ] False positives cost more than false negatives
- [ ] False negatives cost more than false positives
- [x] False positives and false negatives are equally costly
- [ ] The dataset has no class imbalance

### Why do we use out-of-fold (OOF) training probabilities to select the threshold, rather than test-set probabilities?
- [ ] OOF probabilities are more accurate than test-set probabilities
- [x] Using the test set for threshold selection leaks information and makes the final evaluation dishonest
- [ ] The test set does not contain enough samples for reliable threshold estimation
- [ ] OOF probabilities produce a smoother ROC curve

## Part 5: Credit Card Fraud Detection

### The credit card fraud dataset has 0.17% fraud. A model that labels every transaction 'Not Fraud' achieves 99.83% accuracy. Why is this model useless?
- [ ] Its precision is too low
- [x] It catches zero fraud cases — the thing we actually care about
- [ ] It has a high false positive rate
- [ ] It has not been trained on enough data

### With 99.8% legitimate transactions, the weighted average F1 in the classification report is nearly indistinguishable from accuracy. Why?
- [ ] Weighted average is always equal to accuracy
- [x] The legitimate class dominates the support counts, so its performance overwhelms the fraud class in the weighted calculation
- [ ] F1 and accuracy use the same formula
- [ ] The model performs equally well on both classes

### For the fraud dataset, the ROC AUC looks impressive (e.g., ~0.97). Why is this misleading?
- [ ] AUC is always high when the dataset is large
- [x] With 85,000+ legitimate transactions, even thousands of false alarms barely move the FPR denominator
- [ ] The ROC curve only measures recall, not precision
- [ ] XGBoost always produces high AUC on fraud data

### The F2-score (beta=2) weights recall four times as heavily as precision. When is this the right choice?
- [ ] When false positives are more costly than false negatives
- [x] When missing positive cases (e.g., fraud) is more costly than generating false alarms
- [ ] When the dataset is perfectly balanced
- [ ] When you want to maximize overall accuracy

### On the fraud dataset, the PR curve baseline is approximately 0.0017 (0.17%). A model whose PR curve barely rises above this line would tell you:
- [ ] The model is performing well on the minority class
- [x] The model is barely better than randomly guessing 'fraud' with 0.17% probability
- [ ] The model has high recall but low precision
- [ ] The positive class prevalence is too high to detect fraud

### Why must you use out-of-fold (OOF) probabilities — not test-set probabilities — when selecting the optimal threshold?
- [ ] OOF probabilities are always more accurate
- [x] Using test-set probabilities to choose a threshold means your final evaluation is no longer on truly unseen data
- [ ] The test set is too small for reliable threshold estimation
- [ ] OOF avoids overfitting the model itself

## Part 6: Cost-Weighted Training & Nested CV

### For the fraud dataset, the class ratio is ~499:1 but the cost ratio is only 4.5:1 ($450 FN / $100 FP). Why would scale_pos_weight = 499 be a poor choice?
- [x] It would over-weight the minority class by two orders of magnitude relative to the actual costs
- [ ] XGBoost cannot accept scale_pos_weight values above 100
- [ ] It would make training too slow on 284,807 transactions
- [ ] scale_pos_weight only works for balanced datasets

### What does the custom cost_weighted_objective change about how XGBoost trains?
- [x] It scales the gradient and hessian by dollar costs, so missed fraud produces a stronger correction signal than a false alarm
- [ ] It changes the decision threshold from 0.5 to the cost-optimal value during training
- [ ] It removes legitimate transactions from the training set until the classes are balanced
- [ ] It replaces the trees with a linear model that minimizes dollar cost directly

### The flawed total_cost_scorer converts probabilities to predictions with a hardcoded threshold (0.95) inside GridSearchCV. Why does this produce misleading model comparisons?
- [x] The cost-optimal threshold differs for every hyperparameter combination, so the search rewards models that happen to score well at that one cutoff
- [ ] 0.95 is too high a threshold for any practical fraud model
- [ ] GridSearchCV cannot maximize negative values
- [ ] The scorer ignores false positives entirely

### What is the correct GridSearch scoring metric in notebook 6, and why?
- [x] average_precision (PR-AUC), because it measures ranking quality across all thresholds and is therefore threshold-independent
- [ ] accuracy, because it is the most interpretable metric
- [ ] recall, because missing fraud is the most expensive error
- [ ] total dollar cost at the 0.5 threshold, because it directly encodes business priorities

### In nested cross-validation, what is the role of the outer loop?
- [x] To evaluate the complete tuning pipeline on holdout folds it never influenced, giving an honest generalization estimate
- [ ] To find the best hyperparameters for the final model
- [ ] To generate out-of-fold probabilities for threshold tuning
- [ ] To speed up GridSearchCV by parallelizing the parameter grid

### The tuned model saved very little over the baseline XGBoost. According to the notebook, why is this an expected result rather than a failure?
- [x] Default XGBoost hyperparameters are strong starting points, so grid search typically retrieves only the last few percent of attainable performance
- [ ] The grid search used the wrong scoring metric
- [ ] Nested CV systematically underestimates the benefit of tuning
- [ ] The dataset is too small for hyperparameter tuning to matter

### After nested CV, the final production model is trained on 100% of the data. How should its performance be reported?
- [x] Report the nested CV mean PR-AUC — never evaluate the final model on the data it was trained on
- [ ] Evaluate the final model on the full dataset, since more data gives a more reliable score
- [ ] Re-split the data and evaluate on a fresh 30% test set
- [ ] Report the inner-loop GridSearchCV best score
