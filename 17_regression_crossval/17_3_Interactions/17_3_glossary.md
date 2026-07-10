# 17_3 Interaction Terms — Glossary

This document defines the technical and conceptual terms used in the 17_3 Interactions notebook.

---

## A

### Additive Model
A regression model in which each feature contributes its effect independently and the prediction is the sum of those contributions: `charges ~ bmi + smoker`. Geometrically, a categorical feature shifts the line up or down but every group shares the same slope — parallel lines. The additive assumption is the default in every model from 17_1 and 17_2; this unit is about when it breaks.

### AIC (Akaike Information Criterion)
A model-comparison score that rewards fit but penalizes complexity; lower is better. Unlike $R^2$, which always improves when a term is added, AIC only improves if the added term earns back its complexity cost. In the medical-cost case study, AIC drops substantially when the interaction term is added — evidence the term is not just noise-chasing.

---

## I

### Interaction Effect
A relationship in which the effect of one feature on the target *depends on the value of another feature*. In the case study, each BMI point costs a non-smoker roughly \$83 but a smoker roughly \$1,473 — the BMI slope depends on smoking status. Visually: non-parallel group regression lines.

### Interaction Term
The engineered feature that lets a model capture an interaction — literally the product of two columns (`bmi × smoker`). Its coefficient measures how much the slope of one variable *changes* across levels of the other. A zero interaction coefficient means the slopes are identical and the additive model was adequate.

---

## M

### Main Effect
The stand-alone term for a feature in a model that also contains interactions (`bmi` and `smoker` in `charges ~ bmi * smoker`). With an interaction present, a main effect is interpreted *for the reference group*: the `bmi` coefficient is the slope for non-smokers only. Main effects should virtually always be kept in the model alongside their interaction — dropping them makes the remaining coefficients nearly uninterpretable.

---

## R

### Reference Group
The category statsmodels absorbs into the intercept when encoding a categorical variable — for `smoker`, the non-smokers (`smoker[T.yes]` measures the difference *from* them). In an interaction model, main-effect slopes belong to the reference group, and interaction coefficients measure the change relative to it.

---

## S

### Slope-Shift Interpretation
The plain-English reading of an interaction coefficient: `bmi:smoker[T.yes] ≈ 1,390` means "being a smoker adds about \$1,390 to the per-BMI-point slope." The total slope for the non-reference group is main effect + interaction (≈ \$83 + \$1,390 ≈ \$1,473).

### Statsmodels Formula Operators (`*` vs. `:`)
In a formula, `a:b` adds only the interaction term, while `a * b` expands to `a + b + a:b` — main effects plus interaction. Prefer `*`: it guarantees the main effects ride along. `a:b` alone produces a model with no term for the reference group's slope and an uninterpretable intercept.
