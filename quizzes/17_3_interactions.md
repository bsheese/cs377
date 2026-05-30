# 17.3 · Interaction Terms

## 17_3_1: Interaction Terms

### The lmplot with hue='smoker' shows two lines with dramatically different slopes. What does this specifically indicate?
- [ ] Smoking is a confounding variable that should be excluded from the model
- [x] The effect of BMI on charges differs between smokers and non-smokers — an interaction exists
- [ ] The data has heteroscedasticity that a log transformation would correct
- [ ] Smokers and non-smokers require entirely separate modeling pipelines

### In the additive model 'charges ~ bmi + smoker', what does 'independent effects' mean?
- [ ] BMI and smoking status are statistically uncorrelated in this dataset
- [x] BMI's effect on charges is the same regardless of smoking status
- [ ] Each predictor is independently significant according to the model summary output
- [ ] The model was estimated on an independently drawn random sample of patients

### In 'charges ~ bmi * smoker', three coefficients are produced. What does the 'bmi:smoker[T.yes]' coefficient represent?
- [ ] The baseline cost difference between smokers and non-smokers at BMI = 0
- [x] The additional per-BMI-unit cost increase for smokers above the non-smoker rate
- [ ] The total cost per BMI unit regardless of the patient's smoking status
- [ ] The probability that a high-BMI patient in the dataset is a smoker

### The non-smoker BMI coefficient is ~7 and the interaction coefficient is ~1,400. What is the total cost per BMI unit for a smoker?
- [ ] 7 — the interaction term replaces the main effect for smokers entirely
- [ ] 1,400 — the main effect is irrelevant when an interaction term is present
- [x] ~1,407 — the main effect plus the interaction coefficient
- [ ] 700 — the average of the main effect and the interaction coefficient

### Why is 'bmi * smoker' preferable to manually writing 'bmi + smoker + bmi:smoker' in the formula?
- [ ] The * syntax runs faster because it skips the main effects calculation
- [x] The * syntax ensures main effects are always included with the interaction term
- [ ] Manually specifying terms causes statsmodels to double-count the main effects
- [ ] The * syntax automatically centers BMI to reduce multicollinearity

### R² increases from ~0.65 (additive) to ~0.83 (interaction). Is R² improvement alone sufficient justification for the interaction model?
- [ ] Yes — any R² improvement, however small, justifies adding a new term
- [x] No — R² rises with any additional term; AIC and domain logic are also needed
- [ ] Yes — only R² measures out-of-sample predictive accuracy directly
- [ ] No — R² improvements only matter when they exceed 50 percentage points

### If the 'bmi:smoker[T.yes]' interaction coefficient were exactly zero, what would this tell you?
- [ ] BMI is unimportant for predicting charges in this dataset
- [x] The BMI slope is the same for smokers and non-smokers — no interaction exists
- [ ] Smoking has no effect on charges when controlling for BMI
- [ ] The interaction term was not correctly encoded in the formula string

### AIC (Akaike Information Criterion) penalizes model complexity. If both models have similar AIC, what does this suggest?
- [ ] The interaction model is preferred whenever AIC values are equal
- [x] The interaction term's added complexity is not sufficiently repaid by the improvement in fit
- [ ] AIC cannot distinguish between additive and interaction model structures
- [ ] Both models will produce identical predictions on any held-out test data

### A student writes 'charges ~ bmi:smoker' without including main effects. What is the problem with this model?
- [ ] The interaction operator : is not valid without the * operator in statsmodels
- [x] The model has no term for smokers with BMI = 0, making the intercept uninterpretable
- [ ] Statsmodels automatically adds main effects when interaction terms are specified
- [ ] The model would overfit because interaction terms require large sample sizes

### An insurance company builds an additive model (no interaction) for pricing. Why could this lead to unfair pricing?
- [ ] Additive models overcharge non-smokers and systematically undercharge smokers
- [x] The model charges smokers and non-smokers the same rate per BMI unit, ignoring the true cost difference
- [ ] Additive models cannot include categorical variables like smoking status
- [ ] The model would ignore BMI entirely once smoker status is included
