Here is a detailed, step-by-step plan for a single Jupyter Notebook utilizing the **Two-Act Structure**. 

This plan is built on a "Guided Workshop" model. The notebook will contain pre-written scaffolding code for visualizations,and the core modeling code, students must modify parameters (with direction), and answer markdown reflection questions at every step.

---

# Notebook Title: The Classification Toolkit: From Binary Thresholds to Multiclass Matrices

**The Premise:** The student plays the role of a consulting Data Scientist tackling two consecutive projects for different clients. 
* **Client 1 (Finance):** A bank that needs a loan approval model. They care deeply about the *dollar cost* of mistakes.
* **Client 2 (Healthcare):** A hospital that needs a fetal monitor alert system. They care deeply about *fatal misses* across multiple risk categories.

---

## ACT I: The Binary Problem (Credit-g Dataset)
**Learning Focus:** `scale_pos_weight`, Probabilities vs. Hard Predictions, ROC/PR Curves, Youden's J, and Business Cost-Curves.

### Step 1: The Baseline and the Accuracy Paradox
*   **Narrative:** The bank wants to predict "Bad Credit" (1) vs "Good Credit" (0). The data is naturally imbalanced (70% Good, 30% Bad).
*   **Code Action:** 
    *   Load `credit-g`, encode target, perform a stratified train/test split.
    *   Train a baseline `XGBClassifier`.
    *   Print Accuracy and display the 2x2 Confusion Matrix.
*   **Visual:** A 2x2 Heatmap showing high True Negatives (Good) but missing a lot of True Positives (Bad).
*   **Student Reflection:** "My model has an accuracy of 74%. What would the accuracy be if I wrote a one-line Python script that *always* predicted 'Good Credit'? Why is the bank's Chief Risk Officer unhappy with this model?"

### Step 2: Peeking Under the Hood (Probabilities)
*   **Narrative:** Before tuning, we need to understand *how* the model is making its mistakes.
*   **Code Action:** 
    *   Use `.predict_proba()` to get the raw probabilities.
    *   Plot a histogram showing the distribution of predicted probabilities, color-coded by the *actual* class.
*   **Visual:** Two overlapping histograms (Blue for Good Credit, Red for Bad Credit). A vertical line is drawn at the default `0.5` threshold.
*   **Student Reflection:** "Look at the overlapping area in the histogram. If we leave the threshold at 0.5, are we making more False Positives or False Negatives? What direction should we move the line to catch more 'Bad Credit' customers?"

### Step 3: Tuning via Thresholds (ROC & Youden's J)
*   **Narrative:** Let's find the mathematically optimal threshold without retraining the model.
*   **Code Action:** 
    *   Generate False Positive Rates (FPR) and True Positive Rates (TPR) using `roc_curve()`.
    *   Calculate Youden's J (`TPR - FPR`) for every threshold.
    *   Find the threshold that maximizes Youden's J.
*   **Visual:** The ROC Curve with a star plotting the default 0.5 threshold, and a red dot plotting the Youden's J optimal threshold.
*   **Student Reflection:** "Youden's J suggests a threshold of ~0.32. Apply this new threshold to your probabilities and print a new confusion matrix. How did the TP, TN, FP, and FN numbers shift compared to Step 1?"

### Step 4: The Business Cost Curve
*   **Narrative:** The bank doesn't care about math; they care about money. Approving a bad loan (False Positive) costs $5,000. Rejecting a good loan (False Negative) costs $1,000 in lost revenue.
*   **Code Action:** 
    *   Write a custom function that calculates total dollar cost based on the FP and FN counts at *every* threshold from 0.0 to 1.0.
*   **Visual:** A Cost Curve (X-axis: Threshold, Y-axis: Total Dollar Cost).
*   **Student Reflection:** "Did the business-optimal threshold match Youden's J? Why or why not? Write a 2-sentence recommendation to the bank on where to set their system."

---

## ACT II: The Multiclass Problem (Cardiotocography Dataset)
**Learning Focus:** The 3x3 Matrix, Macro vs Weighted F1, and `sample_weights`.

### Step 5: Transitioning to Multiclass (The Trap)
*   **Narrative:** You are reassigned to a hospital predicting fetal health: Normal (0), Suspect (1), Pathological (2). The data is heavily imbalanced (78%, 15%, 7%). 
*   **Code Action:**
    *   Load dataset, apply train/test split.
    *   Train a default `XGBClassifier(objective="multi:softprob")`.
    *   Print the Classification Report and a 3x3 Confusion Matrix.
*   **Visual:** A 3x3 Matrix where the model overwhelmingly guesses "Normal".
*   **Student Reflection:** "You cannot easily use a single Threshold or `scale_pos_weight` here because there are 3 classes. Look at your Classification Report. Why is the 'Weighted Avg F1' score so high (~0.85) while the 'Macro Avg F1' score is so low (~0.60)?"

### Step 6: Forcing Fairness with Sample Weights
*   **Narrative:** We must force the trees to care about the "Pathological" minority class during training. 
*   **Code Action:** 
    *   Calculate balanced weights using `compute_sample_weight('balanced', y_train)`.
    *   Train a new XGBoost model, passing `sample_weight=weights` into the `.fit()` method.
    *   Display the new 3x3 Confusion Matrix next to the old one.
*   **Visual:** Side-by-side heatmaps. The new matrix shows errors spreading out, but significantly more correct hits in the bottom-right "Pathological" corner.
*   **Student Reflection:** "By adding sample weights, our overall accuracy actually dropped. However, look at the cells of the matrix. Explain why the hospital will prefer this 'lower accuracy' model. Use the terms 'False Alarm' (Predicting Pathological when Normal) and 'Fatal Miss' (Predicting Normal when Pathological)."

### Step 7: Interpreting the Black Box
*   **Narrative:** The doctors want to know *why* the model is making these decisions before they trust it.
*   **Code Action:** 
    *   Extract feature importances from the weighted model using `plot_importance(importance_type='gain')`.
*   **Visual:** A horizontal bar chart showing the top 5 most important clinical features.
*   **Student Reflection:** "List the top 3 features by Gain. Remember from 18_1_1 that XGBoost importance doesn't tell us *direction*. How might you explain this limitation to a doctor who asks, 'Does a high value of [Top Feature] mean the fetus is Pathological?'"

---

## The Final Output

At the end of the notebook, the student will have:
1. Re-evaluated accuracy in the face of imbalance.
2. Learned to tune a binary model post-training (Thresholds/Business Costs).
3. Learned to tune a multiclass model pre-training (Sample Weights).
4. Navigated both 2x2 and 3x3 confusion matrices.
5. Understood the distinct use cases of Weighted vs. Macro averaging.
