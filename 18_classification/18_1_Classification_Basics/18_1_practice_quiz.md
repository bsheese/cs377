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

Sequentially, correcting errors from prior trees - sequential approach fixing all prior mistakes
- Sequentially, correcting errors from prior trees
- Sequentially, fixing errors from earlier trees
- Random selection without specific order

**Answer:** Sequentially, with each tree correcting the errors of previous trees

---

### Question 8
**Why is stratified splitting important for imbalanced datasets?**

**Options:**
- It makes training and test sets proportionally equal
- It ensures both train and test sets have the same class proportions
- It automatically balances the class weights during training
- It removes minority class samples to simplify training

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

Harmonic mean gives more weight to lower values, penalizing extreme imbalances significantly
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

The model correctly ranks a random positive above a random negative 79% of the time
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

The model's performance varies significantly across different data splits because tree depth matters
- The model is always accurate
- The model's performance varies significantly across different data splits
- The model has high bias
- The model is underfitting

**Answer:** The model's performance varies significantly across different data splits

---

### Question 9
**Why do we score models on both accuracy AND F1?**

Accuracy is often misleading on imbalanced datasets; F1 reveals minority class handling quality
- F1 is always more important than accuracy
- Accuracy can be misleading on imbalanced data; F1 reveals how well the model handles the minority class
- Accuracy is not always reliable; F1 provides a better measure for imbalanced classes
- They always give the same result

**Answer:** Accuracy can be misleading on imbalanced data; F1 reveals how well the model handles the minority class

---

### Question 12
**If Random Forest has the highest mean F1 but also the widest spread, you should:**

Consider mean performance and stability; choose slightly lower but highly consistent models
- Always choose it because it has the highest mean
- Consider both mean performance and stability; a slightly lower but more consistent model might be preferable
- Choose the lowest variance model regardless of mean
- Use a single train/test split instead of CV

**Answer:** Consider both mean performance and stability; a slightly lower but more consistent model might be preferable

---

## 18_1_4: Multiclass Classification

### Question 1
**In a 3x3 confusion matrix, the diagonal contains:**

**Options:**
- False positives
- False negatives
- Correct predictions
- Uncertain predictions

**Answer:** Correct predictions

---

### Question 2
**Macro average treats all classes:**

**Options:**
- Equally, regardless of sample size
- Based on their support (sample count)
- Based on their precision scores only
- Randomly

**Answer:** Equally, regardless of sample size

---

### Question 3
**When would macro and weighted averages give very different results?**

**Options:**
- When classes have very different sample sizes and performance varies significantly across classes
- When the dataset is perfectly balanced
- When classes have similar sample sizes and performance
- When using binary classification only

**Answer:** When classes have very different sample sizes and performance varies significantly across classes

---

### Question 4
**In multiclass classification, how does XGBoost produce predictions for 3+ classes?**

**Options:**
- It uses a single probability and multiple thresholds
- It uses softmax to output a probability for each class, and predicts the class with the highest probability
- It always uses One-vs-Rest with separate binary models
- It randomly assigns classes based on feature values

**Answer:** It uses softmax to output a probability for each class, and predicts the class with the highest probability

---

### Question 5
**A model achieves 93% accuracy on a dataset where 78% of samples belong to one class. The Macro F1 is 0.87 and the Weighted F1 is 0.93. Which metric best reveals the model's weakness on minority classes?**

**Options:**
- Accuracy (93%)
- Weighted F1 (0.93)
- Macro F1 (0.87)
- All three metrics tell the same story

**Answer:** Macro F1 (0.87)

---

### Question 6
**What does `compute_sample_weight('balanced', y)` do?**

**Options:**
- Removes minority class samples to balance the dataset
- Assigns higher weights to rare class samples so the model pays more attention to them during training
- Duplicates minority class samples to create a balanced dataset
- Scales feature values based on class frequency

**Answer:** Assigns higher weights to rare class samples so the model pays more attention to them during training

---

### Question 7
**In a fetal health classification model, which error is the most dangerous?**

**Options:**
- Predicting Normal when actually Suspect (under-monitoring)
- Predicting Pathological when actually Normal (false alarm)
- Predicting Normal when actually Pathological (fatal miss)
- Predicting Suspect when actually Normal (minor false alarm)

**Answer:** Predicting Normal when actually Pathological (fatal miss)

---

