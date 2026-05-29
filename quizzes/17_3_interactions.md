# 17.3 · Interaction Terms

## 17_3_1: Interaction Terms

### The lmplot with hue='smoker' shows two lines with dramatically different slopes. What does this specifically indicate?
- [ ] Smoking is a confounding variable that should be excluded from the model
- [x] The effect of BMI on charges differs between smokers and non-smokers — an interaction exists
- [ ] The data has heteroscedasticity that a transformation would fix
- [ ] Smokers and non-smokers should be modeled in completely separate datasets

### In the additive model 'charges ~ bmi + smoker', what does 'independent effects' mean?
- [ ] BMI and smoking are statistically uncorrelated in the dataset
- [x] BMI's effect on charges is the same whether someone smokes or not
- [ ] Each predictor is independently significant in the model summary
- [ ] The model was fit on a random independent sample of patients

### In 'charges ~ bmi * smoker', three coefficients are produced. What does the 'bmi:smoker[T.yes]' coefficient represent?
- [ ] The base cost difference between smokers and non-smokers at BMI=0
- [x] The additional per-BMI-unit cost increase for smokers above the non-smoker rate
- [ ] The total cost per BMI unit regardless of smoking status
- [ ] The probability that a high-BMI patient is a smoker

### The non-smoker BMI coefficient is ~7 and the interaction coefficient is ~1,400. What is the total cost per BMI unit for a smoker?
- [ ] 7 — the interaction term replaces the main effect for smokers
- [ ] 1,400 — the main effect is irrelevant when an interaction is present
- [x] ~1,407 — the main effect plus the interaction coefficient
- [ ] 700 — the average of the main effect and the interaction term

### Why is 'bmi * smoker' preferable to manually writing 'bmi + smoker + bmi:smoker' in the formula?
- [ ] The * syntax runs faster because it skips the main effects calculation
- [x] The * syntax ensures main effects are always included with the interaction term
- [ ] Manually specifying terms causes statsmodels to double-count the main effects
- [ ] The * syntax automatically centers BMI to reduce multicollinearity

### R² increases from ~0.65 (additive) to ~0.83 (interaction). Is R² improvement alone sufficient justification for the interaction model?
- [ ] Yes — any R² improvement justifies adding the interaction term
- [x] No — R² always increases with more terms; AIC and domain logic are also needed
- [ ] Yes — only R² measures out-of-sample predictive accuracy
- [ ] No — R² improvements only matter when they exceed 50 percentage points

### If the 'bmi:smoker[T.yes]' interaction coefficient were exactly zero, what would this tell you?
- [ ] BMI is unimportant for predicting charges in this model
- [x] The BMI slope is the same for smokers and non-smokers — no interaction exists
- [ ] Smoking has no effect on charges when controlling for BMI
- [ ] The interaction term was not properly encoded in the formula

### AIC (Akaike Information Criterion) penalizes model complexity. If both models have similar AIC, what does this suggest?
- [ ] The interaction model is always preferred when AIC values are equal
- [x] The complexity of the interaction term does not pay off sufficiently in fit improvement
- [ ] AIC cannot distinguish between additive and interaction models
- [ ] Both models will produce identical predictions on test data

### A student writes 'charges ~ bmi:smoker' without including main effects. What is the problem with this model?
- [ ] The interaction operator : is not valid without the * operator
- [x] The model has no term for smokers who have BMI=0, making the intercept uninterpretable
- [ ] Statsmodels automatically adds main effects when interaction terms are specified
- [ ] The model would overfit because interactions always require large sample sizes

### An insurance company builds an additive model (no interaction) for pricing. Why could this lead to unfair pricing?
- [ ] Additive models always overcharge non-smokers and undercharge smokers
- [x] The model would charge smokers and non-smokers the same rate per BMI unit, ignoring the true cost difference
- [ ] Additive models cannot include categorical variables like smoker status
- [ ] The model would ignore BMI entirely when smoker is in the model
