# Discussion Questions: 18_1 Classification Basics

---

## 18_1_1: Classification Foundations

### Why Gradient Boosting?

1. The notebook uses XGBoost (gradient boosting) rather than logistic regression. Why might gradient boosting be more effective for complex classification tasks like credit default prediction?
2. XGBoost builds an ensemble of decision trees sequentially, with each tree correcting the errors of the previous ones. How does this differ from a single decision tree?
3. What is the purpose of the `scale_pos_weight` parameter in XGBoost? How does it help with class imbalance?
4. If you have 700 good credit customers and 300 bad credit customers, what should `scale_pos_weight` be set to? Why does this help the model learn better from the minority class?

### Class Imbalance and the Accuracy Paradox

5. In the South German Credit dataset, 70% of applicants have good credit and 30% have bad credit. What would be the accuracy of a model that always predicts "Good"? Why is this model useless despite its 70% accuracy?
6. If a dataset had a 95/5 class split, what would the naive baseline accuracy be? How much better than this baseline would a model need to be before you'd consider it useful?
7. Why is the accuracy paradox especially dangerous in real-world applications like fraud detection or rare disease diagnosis?
8. The notebook states: "If your model's accuracy isn't significantly better than the majority class percentage, your model hasn't learned anything useful." What does "significantly better" mean in practice? Is 72% vs. 70% significant? What about 80% vs. 70%?

### Data Preparation

9. Why must the target variable ('good'/'bad') be converted to numeric (0/1) before training XGBoost?
10. What is the purpose of `drop_first=True` in one-hot encoding? What would happen if you omitted it?
11. Why is a stratified train-test split important for imbalanced datasets? What could go wrong with a simple random split?
12. The notebook notes that tree-based models like XGBoost do NOT require feature scaling. Why is feature scaling unnecessary for tree-based models, but required for logistic regression and SVM?

### Interpreting Feature Importance

13. XGBoost provides feature importance scores based on "gain" (loss reduction), "cover" (samples affected), and "frequency" (times used). What does a high gain score tell you about a feature?
14. If a feature has a very low importance score (near zero), does this mean the feature is useless? Explain.
15. How would you determine whether a high-importance feature increases or decreases credit risk? What additional analysis would you need?

### Probabilities and the Decision Threshold

16. What is the difference between `.predict()` and `.predict_proba()`? When would you use each?
17. The overlap analysis shows that actual defaulters scored below 0.5. What does this mean in business terms?
18. The mean probability for good customers is around 0.30, while for defaulters it's around 0.65. What does the gap between these two means tell you about the model's discriminative ability?
19. If the two probability distributions overlapped completely (both centered at 0.5), what would this tell you about the model?

---

## 18_1_2: Confusion Matrix and Basic Metrics

### The Confusion Matrix

20. Given a confusion matrix `[[180, 20], [30, 70]]`, identify the TP, TN, FP, and FN values. Which class is the "positive" class?
21. In the credit default context, what is the real-world consequence of a false positive? Of a false negative?
22. Why is it called a "confusion" matrix? What is the model "confused" about?
23. If a model produces a confusion matrix of `[[210, 0], [0, 90]]`, what does this tell you about its performance? Is this realistic for a credit risk model?

### Precision and Recall

24. Precision answers the question: "When the model says 'default,' how often is it right?" Recall answers: "Of all actual defaulters, how many did we catch?" Why are these two questions different, and why do they often conflict?
25. Can a model have 100% precision? Can it have 100% recall? Can it have both simultaneously? Under what conditions?
26. If a bank's priority is to "never approve a loan to someone who will default," which metric should they optimize: precision or recall? Why?
27. If a bank's priority is to "never turn away a good customer," which metric should they optimize? What is the trade-off?

### The F1-Score

28. Why does the F1-score use the harmonic mean instead of the arithmetic mean? What would happen if we used the arithmetic mean of precision and recall?
29. A model has precision = 100% and recall = 1%. What is its F1-score? What does this tell you about the model's actual usefulness?
30. A model has precision = 60% and recall = 60%. Another has precision = 90% and recall = 30%. Which has the higher F1-score? Which would you prefer for a credit risk application?
31. When would you choose to optimize F1 over optimizing precision or recall individually?

### The Classification Report

32. The classification report shows separate precision, recall, and F1 for each class, plus macro and weighted averages. Why are there two different averages? When would they give very different results?
33. If the 'Good' class has recall = 0.85 and the 'Bad' class has recall = 0.55, what does this tell you about the model's behavior?
34. What does the "support" column tell you? Why is it important context for interpreting the per-class metrics?
35. In an extremely imbalanced dataset (99% negative, 1% positive), the weighted average F1 might be 0.95 while the macro average F1 is 0.50. Which one should you trust, and why?

### The Threshold Trade-Off

36. If you raise the decision threshold from 0.5 to 0.7, what happens to precision? To recall? To the number of false positives? To the number of false negatives?
37. Why does the notebook defer the full threshold analysis to the next notebook rather than covering it here?

---

## 18_1_3: ROC, AUC, and Threshold Tuning

### The ROC Curve

38. The ROC curve plots TPR (recall) against FPR (1 − specificity). What does it mean for a point on the curve to be in the top-left corner? In the bottom-right corner? On the diagonal?
39. Why does the ROC curve always start at (0, 0) and end at (1, 1)?
40. If a model's ROC curve is very close to the diagonal line, what does this tell you about its predictive ability?
41. Can a model have an ROC curve below the diagonal? What would this mean, and what could you do about it?

### AUC

42. AUC = 0.79 means "the model correctly ranks a random defaulter above a random good customer 79% of the time." Explain this interpretation in your own words with a concrete example.
43. If Model A has AUC = 0.85 and Model B has AUC = 0.82, can you conclude that Model A is better? What additional information would you need?
44. Why is AUC described as "threshold-independent"? What advantage does this give you when comparing models?
45. An AUC of 0.50 means the model is no better than random. What would an AUC of 1.00 mean? Is this achievable in practice?

### Precision-Recall Curves

46. Why can the ROC curve be over-optimistic on highly imbalanced datasets? What role do true negatives play in this?
47. The PR curve baseline is the positive class prevalence. Why is this the baseline, and what does it mean if your PR curve hugs close to this line?
48. When would you choose to look at a PR curve instead of an ROC curve?

### Youden's J Statistic

49. Youden's J = TPR − FPR. Why does maximizing this quantity find the "best" threshold? What trade-off does it implicitly make?
50. Youden's J assumes that false positives and false negatives are equally costly. In the credit default context, is this a reasonable assumption? Why or why not?

### Business Cost Sensitivity

51. If a false negative costs $5,000 and a false positive costs $500, would you expect the cost-optimal threshold to be higher or lower than 0.5? Why?
52. The notebook compares three thresholds: default (0.5), Youden's J, and cost-optimal. In what scenario would all three give the same threshold?
53. If you doubled the cost of a false negative from $5,000 to $10,000, what would happen to the cost-optimal threshold? Would it move up or down?
54. Why is the cost curve approach more practically useful than Youden's J for real-world decision-making?

---

## 18_1_4: Model Selection via Cross-Validation

### Cross-Validation Model Competition

55. Why is 5-fold cross-validation preferred over a single train/test split for model comparison? What does it tell you that a single split cannot?
56. In the model competition, each model is scored on both accuracy and F1. Why is it important to look at both metrics? What would it mean if a model had high accuracy but low F1?
57. The boxplots show the spread of scores across the 5 folds. What does a wide box tell you about a model's stability? Why might a Decision Tree have a wider box than a Random Forest?
58. If the Random Forest has the highest mean F1 but also the widest spread, would you still choose it over a slightly lower but more consistent model? What factors would influence your decision?
59. The model competition compares XGBoost, Decision Tree, Random Forest, and SVM (RBF). Why might XGBoost perform well on credit default data?

### Note on Regularization

60. The notebook does NOT cover L1/L2/ElasticNet regularization. Why might this topic have been omitted? Where would you go to learn about regularization for XGBoost?

---

## 18_1_5: Multiclass Classification

### The Multiclass Problem

61. What fundamentally changes when you move from binary classification (2 classes) to multiclass classification (3+ classes)? Why can't you just use the same binary approach?

### OvR vs. Softmax

62. In One-vs-Rest, each binary classifier asks "Is this class i or not class i?" What happens if two classifiers both output high probabilities for the same sample? How is the final decision made?
63. Softmax forces all K output probabilities to sum to 1.0. How does this change the way the model thinks about the classes compared to OvR?
64. XGBoost uses Softmax (multiclass) natively. How does this differ from using OvR explicitly?
65. If OvR and Softmax give very different accuracy scores on the same dataset, what might this suggest about the relationships between the classes?

### Multiclass Evaluation

66. In a 3×3 confusion matrix, the diagonal contains correct predictions. What do the off-diagonal cells represent?
67. Macro average treats all classes equally. Weighted average weights by sample count. Micro average computes globally. Why are macro and weighted averages similar when classes are roughly balanced?
68. Imagine a 4-class dataset with distribution 200/50/50/50. A model gets all 200 of class 0 right but misses every sample from the other three classes. What are the macro, weighted, and micro recall scores?
69. When would micro averaging give a very different result from macro averaging? Which one should you report in a scientific paper, and which one should you report to a business stakeholder?

---

## 18_1_6: Decision Boundaries and Feature Importance

### PCA and Dimensionality Reduction

70. PCA finds the directions in the data that capture the most variance. What does "variance" mean in this context, and why is it a useful criterion for choosing projection directions?
71. If PC1 captures 55% of the variance and PC2 captures 25%, together they capture 80%. What does the missing 20% represent, and could it contain important information for classification?
72. If you increased PCA to 3 components, how much more variance would you capture? Could you still visualize decision boundaries? How?

### Decision Boundaries

73. Describe how decision boundaries differ in shape for: Logistic Regression, Decision Tree, and KNN.
74. Decision tree boundaries are "axis-aligned" — composed of horizontal and vertical steps. Why does this happen? What would a diagonal boundary require from a tree?
75. KNN produces wavy, irregular boundaries. Why is KNN able to capture such complex shapes? What is the risk of boundaries that are too complex?
76. If you increased K in KNN from 5 to 50, what would happen to the decision boundaries? Would they become smoother or more jagged? Why?

### Feature Importance

77. The notebook uses XGBoost's feature importance (based on gain). How does this differ from Random Forest's impurity-based importance?
78. If a feature accounts for 35% of the total feature importance, does this mean the feature alone can classify with 35% accuracy? Explain why or why not.
79. Feature importance tells you *which* features matter but not *how* they affect the prediction. If you wanted to know whether higher values of a feature increase or decrease the probability of a class, what additional analysis would you need?
80. Two features that are highly correlated will share importance between them — neither will appear as dominant. How does this affect your interpretation of the importance chart?