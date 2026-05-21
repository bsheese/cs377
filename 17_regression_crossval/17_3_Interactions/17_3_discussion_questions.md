# Discussion Questions: 17_3 — Interaction Terms

---

## 17_3_1: Interaction Effects in Medical Cost Data

### Recognizing the Need for Interactions

1. The notebook uses `sns.lmplot(hue="smoker")` to plot separate regression lines for smokers and non-smokers *before* fitting any model. Why is this visualization step critical? What would you miss if you went straight to model fitting?

2. The two regression lines (smokers vs. non-smokers) have dramatically different slopes. Why does this "crossing" or "diverging" pattern in the plot specifically indicate an interaction, rather than just a main effect of smoking?

3. An additive model `charges ~ bmi + smoker` assumes BMI and smoking have independent effects. What exactly does "independent" mean here? Draw a quick sketch of what two parallel lines would look like if this assumption held.

4. A student fits an additive model first and gets R² ≈ 0.65. They then add an interaction term and R² jumps to ≈ 0.83. The student says "the model improved dramatically." Is R² improvement alone sufficient justification for adding complexity? What else should they check?

5. Name two other pairs of variables in a health insurance dataset that you'd expect to interact. For each, explain the real-world mechanism that creates the interaction.

### Interpreting Interaction Coefficients

6. In the interaction model `charges ~ bmi * smoker`, three coefficients are produced: `bmi`, `smoker[T.yes]`, and `bmi:smoker[T.yes]`. What does each coefficient represent in plain English?

7. The `bmi` coefficient (~7) represents the cost increase per BMI point for **non-smokers**. The `bmi:smoker[T.yes]` coefficient (~1,400) represents the *additional* increase per BMI point for smokers. What is the total increase per BMI point for a smoker?

8. If a non-smoker gains 10 BMI points, how much does their predicted cost increase? If a smoker gains the same 10 BMI points, how much does their predicted cost increase? Why is this difference alarming from a public health perspective?

9. The `smoker[T.yes]` coefficient in the interaction model has a different value and interpretation than in the additive model. Why does its meaning change when an interaction term is included?

10. How would you interpret the interaction coefficient if it were zero? What does a zero interaction coefficient tell you about the relationship between BMI and smoking?

### The `*` vs. `:` Formula Notation

11. In Statsmodels formula API, `bmi * smoker` expands to `bmi + smoker + bmi:smoker`. Why is it better practice to use `*` rather than manually writing out all three terms? When might you deliberately use `:` alone?

12. A student writes `charges ~ bmi:smoker` without including the main effects. What is wrong with this model specification? What would the intercept mean?

13. If `bmi * smoker` is used and `smoker` has three levels (never, former, current), how many total terms are added to the model? List them.

### Model Comparison

14. AIC (Akaike Information Criterion) penalizes model complexity. If the interaction model has a much lower AIC than the additive model, what does this confirm? If the AIC is only slightly lower, how would that change your decision?

15. R² always increases when you add terms. AIC can increase or decrease. Explain why R² is an unreliable guide for model selection while AIC is more reliable.

16. The notebook compares additive vs. interaction models on the same training data. Why might you also want to compare their test R² values?

### Generalizing the Concept

17. Feature engineering is presented as a way to improve model performance. How is creating an interaction term a form of feature engineering? What "new information" is the interaction term encoding?

18. The notebook mentions that this interaction insight "often beats simply adding more variables." Why would one well-chosen interaction term outperform adding several new independent features?

19. Interaction effects are present in many real-world datasets but are often overlooked. What is the risk of building a policy decision (e.g., insurance pricing) on an additive model when a true interaction exists?

20. If you were presenting these results to an insurance company's board, how would you explain the interaction effect in one or two non-technical sentences?

