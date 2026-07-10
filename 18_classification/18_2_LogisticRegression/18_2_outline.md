# 18_2 Logistic Regression — Topic Outline

This document outlines the topics covered across the 18_2 Logistic Regression series.

**Data by notebook:**
- **18_2_0_1 (reading):** none — conceptual primer with required StatQuest video
- **18_2_1:** Titanic (survival)
- **18_2_2:** Titanic (survival), logistic regression vs. XGBoost
- **18_2_3:** XOR toy data; Titanic (MLP vs. logistic regression)
- **18_2_9 (exercise):** German Credit (`credit-g` from OpenML)

---

## 18_2_0_1: Logistic Regression Basics (reading + video)

**Topics:** What logistic regression is for; the sigmoid; reading the formula.

- Required StatQuest video introduction
- Binary classification predicts a *probability*, then thresholds it
- Linear regression's output line vs. the sigmoid's (0, 1) squash
- The logistic function formula and what each $\beta$ means

---

## 18_2_1: Introduction with Titanic

**Topics:** Full end-to-end logistic regression workflow: EDA, pipeline, cross-validation, baselines, interpretation, evaluation.

### How Logistic Regression Works
- Why a straight line fails for a 0/1 target
- The sigmoid function; probability output
- Preview: this single sigmoid unit *is already a neural network* (bridge to 18_2_3)

### Building the Model
- Feature engineering on Titanic; train-test split
- `Pipeline` to prevent leakage (house convention)
- Cross-validation: is the model stable across folds?

### Is the Model Any Good?
- Establishing baselines before celebrating an accuracy number
- Reading the baselines vs. the model's ~77%

### Interpreting and Evaluating
- Odds ratios: the interpretable currency of logistic regression
- Predicted probabilities and calibration
- Classification report; ROC curve and AUC; precision-recall curve (reprise from 18_1)
- Threshold tuning

### When to Choose Logistic Regression
- Strengths: calibrated probabilities, interpretability, inference
- Limits: linear decision boundary

---

## 18_2_2: Interpretability — When the Model Must Explain Itself

**Topics:** Statistical inference on logistic regression; what interpretable models offer that black boxes cannot; regulatory context.

### Setup: Same Data, Two Models
- Sklearn logistic regression and XGBoost trained on identical Titanic features

### Confidence Intervals on Odds Ratios
- Why statsmodels (`Logit`) instead of sklearn: standard errors, p-values, CIs
- Forest plot of odds ratios with confidence intervals; the reference line at 1.0

### What Each Model Can and Cannot Tell You
- XGBoost feature importance vs. logistic regression coefficients — importance says *that* a feature matters, not *how* or *in which direction*

### When Interpretability Is Required
- The regulatory context (credit, medicine): adverse-action reasons, auditability
- Explaining an individual prediction by decomposing its linear score

---

## 18_2_3: From One Neuron to Many — The Bridge to Neural Networks

**Topics:** Logistic regression as a single neuron; the XOR limitation; hidden layers and activation functions; MLPClassifier as a black box; the interpretability tax.

### Logistic Regression Is a Single Neuron
- Numerical verification: $\sigma(w \cdot x + b)$ reproduces `predict_proba`

### The Problem Logistic Regression Cannot Solve
- XOR: four points no line can separate; the linear-boundary ceiling

### From One Neuron to a Network
- Architecture: input, hidden layer, output
- Activation functions: sigmoid, tanh, ReLU — and why hidden layers need nonlinearity
- `MLPClassifier` solves XOR with a small hidden layer

### A Neural Network on Titanic
- MLP vs. logistic regression on the same features — similar performance on tabular data of this size

### What the Neural Network Loses
- The interpretability tax: no odds ratios, no CIs, no simple per-feature story
- Three-part module summary; forward pointer to the deep-learning course

---

## 18_2_9: Exercise — Credit Risk with Interpretability Constraints

**Topics:** Applying the full 18_2 workflow under a realistic constraint: the model must explain itself.

- German Credit data (`fetch_openml('credit-g')`)
- Pipeline build, training, evaluation
- Odds ratios with confidence intervals; interpretation
- Explaining an individual applicant's prediction
- LR vs. XGBoost comparison: is any accuracy gain worth the interpretability tax here?
