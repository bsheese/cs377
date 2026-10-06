# 17_3 Interaction Terms — Topic Outline

This document outlines the single case-study notebook in the 17_3 Interactions unit.

**Data:** Medical insurance costs (1,338 customers; `charges` predicted from `bmi` and `smoker`), loaded at runtime from this repo's `data/insurance.csv`.

---

## 17_3_1: Beyond Additive Models — Interaction Effects in Medical Costs

**Topics:** The additive assumption and when it breaks; visual detection of interactions; the statsmodels formula interface and its automatic dummy coding; fitting and interpreting an interaction model; comparing models with $R^2$ and AIC.

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
- Introduces the formula interface: `target ~ feature + feature`, intercept added automatically
- Text columns are dummy-coded automatically: `smoker[T.yes]`, with non-smokers as the reference group
- $R^2 \approx 0.66$; shared slope ≈ \$388 per BMI point
- Why the shared slope is a compromise that is wrong for both groups

### 4. Letting the Slope Change: the Interaction Model
- `bmi * smoker` expands to `bmi + smoker + bmi:smoker`
- The interaction term as an engineered feature (product of two columns)
- `*` vs. `:` — why main effects must accompany the interaction (`bmi:smoker` alone forces both lines through the same point at BMI = 0)

### 5. Reading the Coefficients
- `bmi` ≈ \$83 — the slope *for non-smokers* (reference group)
- `bmi:smoker[T.yes]` ≈ \$1,390 — the *change* in slope for smokers; total ≈ \$1,473 (≈ 18× steeper)
- Why `smoker[T.yes]` turns negative: it's the offset at BMI = 0, which no longer means much once slopes differ
- Model comparison: $R^2$ 0.66 → 0.74; AIC 27,526 → 27,152 — the term earns its keep
- What AIC is: fit plus a penalty per coefficient; lower is better; only differences on the same data mean anything
- $R^2$ never falls when terms are added; AIC and the plot make the real case
- Caveat: both models are fit and scored on all the data, so these are not estimates of predictive accuracy

### 6. Why This Matters
- Ames was about which columns to keep; this is about how columns relate to each other
- Fairness stakes: the additive model overcharges some groups (low-BMI smokers, high-BMI non-smokers) and undercharges others
- How to hunt for interactions: mechanism suggests amplification → plot slopes by group first
