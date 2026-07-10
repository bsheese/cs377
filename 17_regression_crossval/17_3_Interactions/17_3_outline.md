# 17_3 Interaction Terms — Topic Outline

This document outlines the single case-study notebook in the 17_3 Interactions unit.

**Data:** Medical insurance costs (1,338 customers; `charges` predicted from `bmi` and `smoker`), fetched at runtime from the Machine-Learning-with-R-datasets GitHub repository.

---

## 17_3_1: Beyond Additive Models — Interaction Effects in Medical Costs

**Topics:** The additive assumption and when it breaks; visual detection of interactions; fitting and interpreting an interaction model with the statsmodels formula API; comparing models with $R^2$ and AIC.

### What This Notebook Is About
- Every model so far assumed features contribute independently (additively)
- The alternative to "add more columns": let one feature's effect depend on another
- Driving question: is a BMI point worth the same dollars for a smoker as a non-smoker?

### 1. The Data
- 1,338 insurance customers; target is annual medical `charges`
- Key features for this study: `bmi` (continuous) and `smoker` (categorical)

### 2. See It Before You Model It
- `sns.lmplot(x="bmi", y="charges", hue="smoker")` — separate regression line per group
- Parallel lines ⇒ additive is fine; non-parallel lines ⇒ interaction
- The plot shows a nearly flat non-smoker line and a steeply climbing smoker line

### 3. The Additive Model
- `ols('charges ~ bmi + smoker')` — one shared BMI slope, one vertical offset
- $R^2 \approx 0.66$; shared slope ≈ \$388 per BMI point
- Why the shared slope is a compromise that is wrong for both groups

### 4. Letting the Slope Change: the Interaction Model
- `bmi * smoker` expands to `bmi + smoker + bmi:smoker`
- The interaction term as an engineered feature (product of two columns)
- `*` vs. `:` — why main effects must accompany the interaction

### 5. Reading the Coefficients
- `bmi` ≈ \$83 — the slope *for non-smokers* (reference group)
- `bmi:smoker[T.yes]` ≈ \$1,390 — the *change* in slope for smokers; total ≈ \$1,473 (≈ 18× steeper)
- Why `smoker[T.yes]` turns negative: it's the offset at BMI = 0, which no longer means much once slopes differ
- Model comparison: $R^2$ 0.66 → 0.74; AIC drops — the term earns its keep
- $R^2$ always rises with added terms; AIC and the plot make the real case

### 6. Why This Matters
- One engineered interaction beat the "add more columns" instinct from Ames
- Fairness stakes: an additive pricing model concentrates its errors on specific groups
- How to hunt for interactions: mechanism suggests amplification → plot slopes by group first
