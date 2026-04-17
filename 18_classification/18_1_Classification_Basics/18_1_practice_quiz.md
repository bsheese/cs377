# 18_1 Practice Quiz Questions

## 18_1_1: Classification Foundations

### Question 1
**In XGBoost, what does the scale_pos_weight parameter do?**

**Options:**
- Increases the learning rate for positive class predictions
- Scales the weight of the positive class to handle class imbalance
- Normalizes all predictions to be between 0 and 1
- Scales feature values to have zero mean and unit variance

**Answer:** Scales the weight of the positive class to handle class imbalance

---

### Question 2
**For the South German Credit dataset with 70% good and 30% bad credit, what is the naive baseline accuracy?**

**Options:**
- 30%
- 50%
- 70%
- 100%

**Answer:** 70%


---

### Question 7
**XGBoost builds an ensemble of decision trees:**

**Options:**
- In parallel, all at once
- Sequentially, with each tree correcting the errors of previous trees
- Randomly, with no particular order
- From the bottom up, starting with the root node

**Answer:** Sequentially, with each tree correcting the errors of previous trees

---

### Question 8
**Why is stratified splitting important for imbalanced datasets?**

**Options:**
- It makes the training set larger than the test set
- It ensures both train and test sets have the same class proportions
- It automatically balances the class weights
- It removes all minority class samples to simplify training

**Answer:** It ensures both train and test sets have the same class proportions

---

## 18_1_2: Confusion Matrix and Basic Metrics

### Question 1
**In a confusion matrix, what does a False Negative represent?**

**Options:**
- Predicting default when the customer is actually good
- Predicting good when the customer actually defaults
- Correctly identifying a defaulter
- Correctly identifying a good customer

**Answer:** Predicting good when the customer actually defaults

---

### Question 2
**Precision measures:**

**Options:**
- Of all actual positives, how many were correctly identified
- Of all positive predictions, how many are actually correct
- The total number of correct predictions
- The ratio of true positives to true negatives

**Answer:** Of all positive predictions, how many are actually correct

---

### Question 3
**Recall measures:**

**Options:**
- Of all positive predictions, how many are actually correct
- Of all actual positives, how many were correctly identified
- The total number of correct predictions
- The ratio of true positives to total predictions

**Answer:** Of all actual positives, how many were correctly identified

---

### Question 4
**Why does F1-score use harmonic mean instead of arithmetic mean?**

**Options:**
- Harmonic mean is faster to compute
- Harmonic mean gives more weight to lower values, penalizing extreme imbalances
- Arithmetic mean cannot handle percentages
- Harmonic mean always gives higher scores

**Answer:** Harmonic mean gives more weight to lower values, penalizing extreme imbalances

---

### Question 5
**If precision = 100% and recall = 1%, what is the F1-score?**

**Options:**
- About 50%
- About 2%
- About 100%
- About 25%

**Answer:** About 2%

---

### Question 6
**When would you prefer weighted average over macro average?**

**Options:**
- When all classes are equally important
- When you want overall performance on the dataset population
- When you want to treat all classes equally regardless of sample size
- When the dataset is perfectly balanced

**Answer:** When you want overall performance on the dataset population

---

### Question 7
**If you raise the decision threshold from 0.5 to 0.7, what happens to precision and recall?**

**Options:**
- Both increase
- Both decrease
- Precision increases, recall decreases
- Precision decreases, recall increases

**Answer:** Precision increases, recall decreases

---

### Question 8
**Type I error is another name for:**

**Options:**
- True Negative
- False Negative
- False Positive
- True Positive

**Answer:** False Positive

---

### Question 9
**What does the ROC curve plot?**

**Options:**
- Precision vs Recall
- True Positive Rate vs False Positive Rate
- Accuracy vs F1-Score
- Loss vs Epoch

**Answer:** True Positive Rate vs False Positive Rate

---

## 18_1_3: ROC, AUC, and Threshold Tuning

### Question 1
**An AUC of 0.79 means:**

**Options:**
- The model correctly classifies 79% of samples
- The model correctly ranks a random positive above a random negative 79% of the time
- 79% of predictions are true positives
- The model has 79% precision

**Answer:** The model correctly ranks a random positive above a random negative 79% of the time

---

### Question 2
**Why is AUC described as threshold-independent?**

**Options:**
- It uses a fixed threshold of 0.5
- It evaluates performance across all possible thresholds
- It only works with binary classification
- It doesn't require a threshold to compute

**Answer:** It evaluates performance across all possible thresholds

---

### Question 3
**Youden's J statistic is calculated as:**

**Options:**
- TPR + FPR
- TPR - FPR
- Precision + Recall
- Accuracy - Baseline

**Answer:** TPR - FPR

---

### Question 4
**Why might ROC curves be over-optimistic on imbalanced data?**

**Options:**
- ROC includes true negatives in the calculation
- ROC curves ignore false positives
- ROC is not affected by class imbalance
- ROC uses precision instead of recall

**Answer:** ROC includes true negatives in the calculation

---

### Question 5
**What is the baseline for a Precision-Recall curve?**

**Options:**
- 0.5 (random)
- The positive class prevalence
- The majority class proportion
- 1.0 (perfect)

**Answer:** The positive class prevalence

---

### Question 6
**If false negatives cost more than false positives, where should you set the threshold?**

**Options:**
- At 0.5 (default)
- Higher than 0.5 (more conservative)
- Lower than 0.5 (more aggressive)
- At 1.0

**Answer:** Lower than 0.5 (more aggressive)

---

### Question 7
**A perfect ROC curve would:**

**Options:**
- Follow the diagonal line
- Hug the top-left corner
- Be a horizontal line at y=0.5
- Be a vertical line at x=0

**Answer:** Hug the top-left corner

---

### Question 8
**A wide box in a cross-validation boxplot indicates:**

**Options:**
- The model is always accurate
- The model's performance varies significantly across different data splits
- The model has high bias
- The model is underfitting

**Answer:** The model's performance varies significantly across different data splits

---

### Question 9
**Why do we score models on both accuracy AND F1?**

**Options:**
- F1 is always more important than accuracy
- Accuracy can be misleading on imbalanced data; F1 reveals how well the model handles the minority class
- Accuracy is required by law
- They always give the same result

**Answer:** Accuracy can be misleading on imbalanced data; F1 reveals how well the model handles the minority class

---

### Question 12
**If Random Forest has the highest mean F1 but also the widest spread, you should:**

**Options:**
- Always choose it because it has the highest mean
- Consider both mean performance and stability; a slightly lower but more consistent model might be preferable
- Choose the lowest variance model regardless of mean
- Use a single train/test split instead of CV

**Answer:** Consider both mean performance and stability; a slightly lower but more consistent model might be preferable

---

## 18_1_4: Multiclass Classification

### Question 1
**Why don't Decision Trees and Random Forests require feature scaling?**

**Options:**
- They are immune to overfitting
- They use information gain, not distances
- Scaling is done automatically inside the algorithm
- Scaling would reduce accuracy

**Answer:** They use information gain, not distances

---

### Question 2
**Which XGBoost parameter controls how many trees are in the ensemble?**

**Options:**
- max_depth
- learning_rate
- n_estimators
- subsample

**Answer:** n_estimators

---

### Question 5
**In a 3x3 confusion matrix, the diagonal contains:**

**Options:**
- False positives
- False negatives
- Correct predictions
- Uncertain predictions

**Answer:** Correct predictions

---

### Question 6
**Macro average treats all classes:**

**Options:**
- Equally, regardless of sample size
- Based on their support (sample count)
- Based on their precision scores only
- Randomly

**Answer:** Equally, regardless of sample size

---

### Question 7
**When would macro and weighted averages give very different results?**

**Options:**
- When the dataset is perfectly balanced
- When classes have very different sample sizes and performance varies significantly across classes
- When using binary classification
- When using OvR strategy

**Answer:** When classes have very different sample sizes and performance varies significantly across classes



### Question 11
**Decision tree boundaries are:**

**Options:**
- Straight lines (hyperplanes)
- Axis-aligned rectangles (horizontal and vertical steps)
- Wavy and irregular curves
- Diagonal lines only

**Answer:** Axis-aligned rectangles (horizontal and vertical steps)

---

### Question 14
**Does feature importance tell you the direction (positive/negative) of the effect?**

**Options:**
- Yes, always
- No, it only tells you which features matter, not how they affect predictions
- Only for binary features
- Only for numerical features

**Answer:** No, it only tells you which features matter, not how they affect predictions

---

