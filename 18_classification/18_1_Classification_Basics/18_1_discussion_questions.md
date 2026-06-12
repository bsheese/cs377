# Discussion Questions: 18_1 Classification Basics

---

## 18_1_1: Classification Foundations

### Why Gradient Boosting?

1. The notebook uses XGBoost (gradient boosting) rather than logistic regression. Why might gradient boosting be more effective for complex classification tasks like credit default prediction?
2. How does XGBoost differ from a single decision tree?
3. What is the purpose of the `scale_pos_weight` parameter in XGBoost? How does it help with class imbalance?
4. If you have 700 good credit customers and 300 bad credit customers, what should `scale_pos_weight` be set to? Why does this help the model learn better from the minority class?

### Class Imbalance and the Accuracy Paradox

5. In the German Credit dataset, 70% of applicants have good credit and 30% have bad credit. What would be the accuracy of a model that always predicts "Good"? Why is this model useless despite its 70% accuracy?
6. If a dataset had a 95/5 class split, what would the naive baseline accuracy be? How much better than this baseline would a model need to be before you'd consider it useful?
7. Why is the accuracy paradox especially dangerous in real-world applications like fraud detection or rare disease diagnosis?
8. The notebook states: "If your model's accuracy isn't significantly better than the majority class percentage, your model hasn't learned anything useful." What does "significantly better" mean in practice? Is 72% vs. 70% significant? What about 80% vs. 70%?

### Data Preparation

9. Why must the target variable ('good'/'bad') be converted to numeric (0/1) before training XGBoost?
10. Notebook 1 passes categorical columns to XGBoost natively (`enable_categorical=True`), while Notebook 3 one-hot encodes them with `drop_first=True` instead. Why are both approaches valid for XGBoost, and what is the purpose of `drop_first=True` when you do encode? What would happen if you omitted it?
11. Why is a stratified train-test split important for imbalanced datasets? What could go wrong with a simple random split?
12. The notebook notes that tree-based models like XGBoost do NOT require feature scaling. Why is feature scaling unnecessary for tree-based models?

### Interpreting Feature Importance

13. XGBoost provides feature importance scores based on "gain" (loss reduction), "cover" (samples affected), and "frequency" (times used). What does a high gain score tell you about a feature?
14. If a feature has a very low importance score (near zero), does this mean the feature is useless? Explain.

### Probabilities and the Decision Threshold

15. What is the difference between `.predict()` and `.predict_proba()`? When would you use each?
16. The mean predicted default probability for good customers is around 0.26, while for actual defaulters it's around 0.57. What does the gap between these two means tell you about the model's discriminative ability?
17. If the two probability distributions overlapped completely (both centered at 0.5), what would this tell you about the model?

---

## 18_1_2: Confusion Matrix and Basic Metrics

### The Confusion Matrix

18. Given a confusion matrix `[[180, 20], [30, 70]]`, identify the TP, TN, FP, and FN values. Which class is the "positive" class?
19. In the credit default context, what is the real-world consequence of a false positive? Of a false negative?
20. If a model produces a confusion matrix of `[[210, 0], [0, 90]]`, what does this tell you about its performance? Is this realistic for a credit risk model?

### Precision and Recall

21. Precision answers the question: "When the model says 'default,' how often is it right?" Recall answers: "Of all actual defaulters, how many did we catch?" Why are these two questions different, and why do they often conflict?
22. Can a model have 100% precision? Can it have 100% recall? Can it have both simultaneously? Under what conditions?
23. If a bank's priority is to "never approve a loan to someone who will default," which metric should they optimize: precision or recall? Why?
24. If a bank's priority is to "never turn away a good customer," which metric should they optimize? What is the trade-off?

### The F1-Score

25. Why does the F1-score use the harmonic mean instead of the arithmetic mean? What would happen if we used the arithmetic mean of precision and recall?
26. A model has precision = 100% and recall = 1%. What is its F1-score? What does this tell you about the model's actual usefulness?
27. A model has precision = 60% and recall = 60%. Another has precision = 90% and recall = 30%. Which has the higher F1-score? Which would you prefer for a credit risk application?
28. When would you choose to optimize F1 over optimizing precision or recall individually?

### The Classification Report

29. The classification report shows separate precision, recall, and F1 for each class, plus macro and weighted averages. Why are there two different averages? When would they give very different results?
30. If the 'Good' class has recall = 0.85 and the 'Bad' class has recall = 0.55, what does this tell you about the model's behavior?
31. What does the "support" column tell you? Why is it important context for interpreting the per-class metrics?
32. In an extremely imbalanced dataset (99% negative, 1% positive), the weighted average F1 might be 0.95 while the macro average F1 is 0.50. Which one should you trust, and why?

### The Threshold Trade-Off

33. If you raise the decision threshold from 0.5 to 0.7, what happens to precision? To recall? To the number of false positives? To the number of false negatives?

---

## 18_1_4: ROC, AUC, and Threshold Tuning

### The ROC Curve

34. The ROC curve plots TPR (recall) against FPR (1 − specificity). What does it mean for a point on the curve to be in the top-left corner? In the bottom-right corner? On the diagonal?
35. Why does the ROC curve always start at (0, 0) and end at (1, 1)?
36. If a model's ROC curve is very close to the diagonal line, what does this tell you about its predictive ability?
37. Can a model have an ROC curve below the diagonal? What would this mean, and what could you do about it?

### AUC

38. AUC = 0.79 means "the model correctly ranks a random defaulter above a random good customer 79% of the time." Explain this interpretation in your own words with a concrete example.
39. If Model A has AUC = 0.85 and Model B has AUC = 0.82, can you conclude that Model A is better? What additional information would you need?
40. Why is AUC described as "threshold-independent"? What advantage does this give you when comparing models?
41. An AUC of 0.50 means the model is no better than random. What would an AUC of 1.00 mean? Is this achievable in practice?

### Precision-Recall Curves

42. Why can the ROC curve be over-optimistic on highly imbalanced datasets? What role do true negatives play in this?
43. The PR curve baseline is the positive class prevalence. Why is this the baseline, and what does it mean if your PR curve hugs close to this line?
44. When would you choose to look at a PR curve instead of an ROC curve?

### Youden's J Statistic

45. Youden's J = TPR − FPR. Why does maximizing this quantity find the "best" threshold? What trade-off does it implicitly make?
46. Youden's J assumes that false positives and false negatives are equally costly. In the credit default context, is this a reasonable assumption? Why or why not?

### Business Cost Sensitivity

47. If a false negative costs $5,000 and a false positive costs $500, would you expect the cost-optimal threshold to be higher or lower than 0.5? Why?
48. In what scenario would the default (0.5), Youden's J, and cost-optimal all give the same threshold?
49. If you doubled the cost of a false negative from $5,000 to $10,000, what would happen to the cost-optimal threshold? Would it move up or down?
50. Why is the cost curve approach more practically useful than Youden's J for real-world decision-making?

---

## 18_1_5: Credit Card Fraud Detection

### Extreme Imbalance and the Accuracy Trap

51. The Credit Card Fraud dataset has a 0.17% fraud rate. A model that labels every transaction as "Not Fraud" achieves 99.83% accuracy. In your own words, why is this model completely useless despite its near-perfect accuracy?
52. In the classification report for the untuned fraud model at the default threshold, accuracy is ~1.00 and weighted average F1 is also very high, yet fraud recall is only ~0.74. Which number most honestly describes the model's usefulness? Why is the weighted average nearly meaningless here?
53. The fraud notebook deliberately avoids using ROC AUC as the primary evaluation tool. Explain why: what specific property of the False Positive Rate makes the ROC curve look deceptively good when 99.8% of samples are legitimate?

### Precision-Recall Curves and F-Beta Score

54. On the fraud dataset, the PR curve baseline is approximately 0.0017 (0.17%). What does this baseline represent? What would it mean if your model's PR curve barely rose above this line?
55. The F2-score (beta=2) weights recall four times as heavily as precision (the weight is beta-squared). In fraud detection, why would you prioritize recall over precision? Give a concrete business example of what a false negative (missed fraud) costs the bank vs. a false positive (incorrectly flagging a legitimate transaction).
56. The F-Beta score peaks at a specific threshold — below that threshold, recall is high but precision is very low; above it, the reverse is true. What does the peak of the F-Beta curve tell you about the right decision threshold for your specific cost priorities?

### Threshold Selection and OOF Probabilities

57. The notebook computes out-of-fold (OOF) probabilities on the training set and uses them to select the cost-optimal threshold, rather than using the test set. Why must threshold selection use training data only? What would go wrong if you selected the threshold using the test set?
58. If the threshold optimized on OOF training data produces slightly different precision and recall when evaluated on the held-out test set, what does this "generalization gap" tell you? Is some gap expected, and what would a very large gap suggest?

---

## 18_1_6: Cost-Weighted Training and Nested CV

### Handling Imbalance: Three Approaches

59. By the end of Notebook 6 you have seen three tools for addressing class imbalance: `scale_pos_weight` (training time), a custom cost-weighted objective (training time), and threshold tuning (prediction time). What is the fundamental difference between the training-time approaches and the prediction-time approach? When would you use all three together?
60. For the fraud dataset, the class ratio is about 499:1 (negative to positive), but the cost ratio is approximately 4.5:1 ($450 FN vs. $100 FP). Why would setting `scale_pos_weight = 499` be a poor choice here? What does the custom cost-weighted objective do differently?

### GridSearch Scorer Design

61. The `total_cost_scorer` function is deliberately shown as a flawed example. The flaw is a hardcoded threshold (`y_prob >= 0.95`) inside the scoring function. Why does using any fixed threshold inside a GridSearch scorer produce misleading model comparisons?
62. What is the correct scoring metric when using GridSearch to optimize model selection for PR-based performance? Why does `average_precision_score` avoid the hardcoded-threshold problem?

### Nested Cross-Validation

63. In nested cross-validation, what is the purpose of the inner loop? What is the purpose of the outer loop? Why does running `cross_val_score` on a model that was already tuned with GridSearch *overestimate* true generalization performance?
64. After the full nested CV and hyperparameter tuning pipeline on the fraud dataset, the improvement over the default XGBoost model was very small. Give at least two reasons why small gains over well-tuned defaults are expected even with a systematic grid search.
