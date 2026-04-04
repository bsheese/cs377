# Discussion Questions: 18_1 Classification Basics

---

## 18_1_0: Classification Foundations

### The Sigmoid Function

1. Linear regression can predict values from negative infinity to positive infinity. Why does this make it unsuitable for binary classification problems?
2. The sigmoid function is described as a "soft switch." What does this mean, and how does it differ from a "hard" on/off switch?
3. If the linear input (z = β₀ + β₁X₁ + ...) equals 0, what probability does the sigmoid function return? Why is this significant?
4. What happens to the sigmoid output as the linear input approaches very large positive values (e.g., z = 100)? Very large negative values (e.g., z = −100)? Why is this behavior desirable for classification?
5. The notebook shows that sigmoid(2) ≈ 0.881 and sigmoid(−2) ≈ 0.119. What does this symmetry tell you about the function?

### Class Imbalance and the Accuracy Paradox

6. In the South German Credit dataset, 70% of applicants have good credit and 30% have bad credit. What would be the accuracy of a model that always predicts "Good"? Why is this model useless despite its 70% accuracy?
7. If a dataset had a 95/5 class split, what would the naive baseline accuracy be? How much better than this baseline would a model need to be before you'd consider it useful?
8. Why is the accuracy paradox especially dangerous in real-world applications like fraud detection or rare disease diagnosis?
9. The notebook states: "If your model's accuracy isn't significantly better than the majority class percentage, your model hasn't learned anything useful." What does "significantly better" mean in practice? Is 72% vs. 70% significant? What about 80% vs. 70%?

### Data Preparation

10. Why must the target variable ('good'/'bad') be converted to numeric (0/1) before training logistic regression?
11. What is the purpose of `drop_first=True` in one-hot encoding? What would happen if you omitted it?
12. Why is a stratified train-test split important for imbalanced datasets? What could go wrong with a simple random split?
13. The notebook explains that feature scaling helps gradient descent converge faster. What would happen if you trained logistic regression without scaling, on a dataset where one feature ranges from 0–1 and another ranges from 0–100,000?
14. After one-hot encoding, the feature count jumps from 20 to 48. Why does this happen, and what are the implications for model complexity?

### Maximum Likelihood Estimation

15. Linear regression minimizes "squared error." What does logistic regression maximize instead, and why can't it use squared error?
16. The notebook gives a concrete example: 3 defaults out of 10 applicants. Walk through how MLE would adjust the model's weights for these 10 people. What would happen to the predicted probabilities for the 3 defaulters? For the 7 non-defaulters?
17. What does `class_weight='balanced'` actually do to the training process? How does it change the way the model penalizes mistakes?

### Interpreting Coefficients

18. Logistic regression coefficients are in "log-odds" units. What does a coefficient of +0.5 actually mean in practical terms?
19. Why do we convert log-odds to odds ratios using `exp(β)`? What does an odds ratio of 1.65 tell you that a log-odds of 0.5 does not?
20. The notebook states that because features are standardized, coefficients are "directly comparable in magnitude." Why is this true? What would happen if features were not standardized?
21. If a feature has a coefficient of −0.3, does this mean the feature is unimportant? Explain.
22. In the top 3 default risk drivers, `housing_rent` has a positive coefficient. What does this suggest about the relationship between renting and credit risk? What confounding factors might explain this?

### Probabilities and the Decision Threshold

23. What is the difference between `.predict()` and `.predict_proba()`? When would you use each?
24. The overlap analysis shows that 23.3% of actual defaulters scored below 0.5. What does this mean in business terms? How many of the ~90 actual defaulters in the test set were missed?
25. The mean probability for good customers is around 0.30, while for defaulters it's around 0.65. What does the gap between these two means tell you about the model's discriminative ability?
26. If the two probability distributions overlapped completely (both centered at 0.5), what would this tell you about the model?
27. The model achieves 73.3% accuracy, beating the 70% baseline by 3.3 percentage points. Is this a good model? What additional information would you need to answer this question properly?

---

## 18_1_1: Confusion Matrix and Basic Metrics

### The Confusion Matrix

28. Given a confusion matrix `[[180, 20], [30, 70]]`, identify the TP, TN, FP, and FN values. Which class is the "positive" class?
29. In the credit default context, what is the real-world consequence of a false positive? Of a false negative?
30. Why is it called a "confusion" matrix? What is the model "confused" about?
31. If a model produces a confusion matrix of `[[210, 0], [0, 90]]`, what does this tell you about its performance? Is this realistic for a credit risk model?

### Precision and Recall

32. Precision answers the question: "When the model says 'default,' how often is it right?" Recall answers: "Of all actual defaulters, how many did we catch?" Why are these two questions different, and why do they often conflict?
33. Can a model have 100% precision? Can it have 100% recall? Can it have both simultaneously? Under what conditions?
34. If a bank's priority is to "never approve a loan to someone who will default," which metric should they optimize: precision or recall? Why?
35. If a bank's priority is to "never turn away a good customer," which metric should they optimize? What is the trade-off?

### The F1-Score

36. Why does the F1-score use the harmonic mean instead of the arithmetic mean? What would happen if we used the arithmetic mean of precision and recall?
37. A model has precision = 100% and recall = 1%. What is its F1-score? What does this tell you about the model's actual usefulness?
38. A model has precision = 60% and recall = 60%. Another has precision = 90% and recall = 30%. Which has the higher F1-score? Which would you prefer for a credit risk application?
39. When would you choose to optimize F1 over optimizing precision or recall individually?

### The Classification Report

40. The classification report shows separate precision, recall, and F1 for each class, plus macro and weighted averages. Why are there two different averages? When would they give very different results?
41. If the 'Good' class has recall = 0.85 and the 'Bad' class has recall = 0.55, what does this tell you about the model's behavior?
42. What does the "support" column tell you? Why is it important context for interpreting the per-class metrics?
43. In an extremely imbalanced dataset (99% negative, 1% positive), the weighted average F1 might be 0.95 while the macro average F1 is 0.50. Which one should you trust, and why?

### The Threshold Trade-Off

44. If you raise the decision threshold from 0.5 to 0.7, what happens to precision? To recall? To the number of false positives? To the number of false negatives?
45. Why does the notebook defer the full threshold analysis to the next notebook rather than covering it here?

---

## 18_1_2: ROC, AUC, and Threshold Tuning

### The ROC Curve

46. The ROC curve plots TPR (recall) against FPR (1 − specificity). What does it mean for a point on the curve to be in the top-left corner? In the bottom-right corner? On the diagonal?
47. Why does the ROC curve always start at (0, 0) and end at (1, 1)?
48. If a model's ROC curve is very close to the diagonal line, what does this tell you about its predictive ability?
49. Can a model have an ROC curve below the diagonal? What would this mean, and what could you do about it?

### AUC

50. AUC = 0.80 means "the model correctly ranks a random defaulter above a random good customer 80% of the time." Explain this interpretation in your own words with a concrete example.
51. If Model A has AUC = 0.85 and Model B has AUC = 0.82, can you conclude that Model A is better? What additional information would you need?
52. Why is AUC described as "threshold-independent"? What advantage does this give you when comparing models?
53. An AUC of 0.50 means the model is no better than random. What would an AUC of 1.00 mean? Is this achievable in practice?

### Precision-Recall Curves

54. Why can the ROC curve be over-optimistic on highly imbalanced datasets? What role do true negatives play in this?
55. The PR curve baseline is the positive class prevalence (~30% for our dataset). Why is this the baseline, and what does it mean if your PR curve hugs close to this line?
56. When would you choose to look at a PR curve instead of an ROC curve?

### Youden's J Statistic

57. Youden's J = TPR − FPR. Why does maximizing this quantity find the "best" threshold? What trade-off does it implicitly make?
58. Youden's J assumes that false positives and false negatives are equally costly. In the credit default context, is this a reasonable assumption? Why or why not?
59. If Youden's J identifies an optimal threshold of 0.35 (lower than the default 0.5), what does this tell you about the model's behavior at the default threshold?

### Business Cost Sensitivity

60. If a false negative costs $5,000 and a false positive costs $500, would you expect the cost-optimal threshold to be higher or lower than 0.5? Why?
61. The notebook compares three thresholds: default (0.5), Youden's J, and cost-optimal. In what scenario would all three give the same threshold?
62. If you doubled the cost of a false negative from $5,000 to $10,000, what would happen to the cost-optimal threshold? Would it move up or down?
63. Why is the cost curve approach more practically useful than Youden's J for real-world decision-making?

---

## 18_1_3: Model Selection via Cross-Validation

### Cross-Validation Model Competition

64. Why is 5-fold cross-validation preferred over a single train/test split for model comparison? What does it tell you that a single split cannot?
65. In the model competition, each model is scored on both accuracy and F1. Why is it important to look at both metrics? What would it mean if a model had high accuracy but low F1?
66. The boxplots show the spread of scores across the 5 folds. What does a wide box tell you about a model's stability? Why might a Decision Tree have a wider box than a Random Forest?
67. If the Random Forest has the highest mean F1 but also the widest spread, would you still choose it over a slightly lower but more consistent model? What factors would influence your decision?
68. Why do Logistic Regression and SVM require scaled features while Decision Trees and Random Forests do not?

### Regularization and Feature Selection

69. L2 (Ridge) regularization shrinks coefficients toward zero but rarely sets them exactly to zero. L1 (Lasso) can zero out coefficients entirely. Why does L1 have this property while L2 does not?
70. If L1 regularization zeroes out 15 of 48 features, does this mean those 15 features are unimportant? Or could they still be useful in a different model?
71. ElasticNet combines L1 and L2 penalties. In what scenario would you prefer ElasticNet over pure L1 or pure L2?
72. The notebook notes that L1 and ElasticNet zero out *different* features. What does this tell you about the reliability of any single feature selection method?
73. If you increased the regularization strength (decreased C) for L1, what would happen to the number of zeroed features? Why?
74. Why can't we compute AIC/BIC for Random Forests or SVMs? What makes logistic regression special in this regard?

---

## 18_1_4: Multiclass Classification

### The Multiclass Problem

75. What fundamentally changes when you move from binary classification (2 classes) to multiclass classification (3+ classes)? Why can't you just use the same binary approach?
76. The wine dataset has 3 classes with roughly balanced distribution (59/71/48). How would the analysis change if the distribution were 150/20/8?

### OvR vs. Softmax

77. In One-vs-Rest, each binary classifier asks "Is this class i or not class i?" What happens if two classifiers both output high probabilities for the same sample? How is the final decision made?
78. Softmax forces all K output probabilities to sum to 1.0. How does this change the way the model thinks about the classes compared to OvR?
79. In the OvR probability histograms, Classifier 1 shows the most overlap between blue and red distributions. What does this tell you about class 1's distinguishability from the other classes?
80. If OvR and Softmax give very different accuracy scores on the same dataset, what might this suggest about the relationships between the classes?

### Multiclass Evaluation

81. In a 3×3 confusion matrix, the diagonal contains correct predictions. What do the off-diagonal cells represent? If class 0 and class 2 are frequently confused, what might this mean chemically about those two wine cultivars?
82. Macro average treats all classes equally. Weighted average weights by sample count. Micro average computes globally. For the wine dataset (roughly balanced), why are macro and weighted averages similar?
83. Imagine a 4-class dataset with distribution 200/50/50/50. A model gets all 200 of class 0 right but misses every sample from the other three classes. What are the macro, weighted, and micro recall scores?
84. When would micro averaging give a very different result from macro averaging? Which one should you report in a scientific paper, and which one should you report to a business stakeholder?
85. If the classification report shows that class 2 has precision = 0.95 but recall = 0.60, what does this tell you about how the model handles class 2?

---

## 18_1_5: Decision Boundaries and Feature Importance

### PCA and Dimensionality Reduction

86. PCA finds the directions in the data that capture the most variance. What does "variance" mean in this context, and why is it a useful criterion for choosing projection directions?
87. If PC1 captures 55% of the variance and PC2 captures 25%, together they capture 80%. What does the missing 20% represent, and could it contain important information for classification?
88. The decision boundaries shown in the plots are in PCA-reduced 2D space, not the original 13-dimensional space. Why might the true boundaries in 13D look different from what we see here?
89. If you increased PCA to 3 components, how much more variance would you capture? Could you still visualize decision boundaries? How?

### Decision Boundaries

90. Logistic regression produces straight-line boundaries in 2D (hyperplanes in higher dimensions). Why is this a limitation, and what kinds of data patterns would a linear boundary fail to capture?
91. Decision tree boundaries are "axis-aligned" — composed of horizontal and vertical steps. Why does this happen? What would a diagonal boundary require from a tree?
92. KNN produces wavy, irregular boundaries. Why is KNN able to capture such complex shapes? What is the risk of boundaries that are too complex?
93. If you increased K in KNN from 5 to 50, what would happen to the decision boundaries? Would they become smoother or more jagged? Why?
94. The geometric model behaviors table lists strengths and weaknesses for each algorithm. For the wine classification task, which model's boundary shape seems most appropriate? Why?

### Feature Importance

95. Random Forest feature importance is based on "total reduction in impurity" attributed to splits on each feature. What does "impurity" mean, and why does reducing it indicate a feature is important?
96. If `proline` accounts for 35% of the total feature importance, does this mean proline alone can classify wines with 35% accuracy? Explain why or why not.
97. Feature importance tells you *which* features matter but not *how* they affect the prediction. If you wanted to know whether higher proline levels increase or decrease the probability of class 0, what additional analysis would you need?
98. Two features that are highly correlated will share importance between them — neither will appear as dominant. How does this affect your interpretation of the importance chart?
99. If you removed the top 3 most important features and retrained the model, which features would you expect to rise in importance? Why?

### Series-Wide Reflection

100. Across the entire 18_1 series, we moved from a single logistic regression model (Part 1) to comparing multiple algorithms (Part 3) to multiclass problems (Part 4) to visualizing how models carve up feature space (Part 5). What is the single most important insight you've gained about classification from this progression?
101. The series emphasizes that "the best model depends on your business costs, not just mathematical optimality." Give a concrete example from the credit default context where the mathematically optimal model would be the wrong business choice.
102. If you had to build a production credit risk system tomorrow, which model from this series would you deploy? Justify your choice considering accuracy, interpretability, computational cost, and regulatory requirements.
103. The series covers logistic regression, decision trees, random forests, SVM, and KNN. Which of these would you *not* use for a credit risk application, and why?
104. Looking back at the naive baseline (70% accuracy from always predicting "Good"), our best model achieves ~73%. Is a 3-point improvement worth the complexity of all the techniques we've learned? Under what circumstances would a small improvement be valuable, and when would it not?
