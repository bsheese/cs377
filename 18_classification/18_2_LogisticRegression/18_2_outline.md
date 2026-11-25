# 18_2 Logistic Regression: Topic Outline

This document outlines the topics covered across the 18_2 Logistic Regression series.

**Data by notebook:**
- **18_2_0_2 and 18_2_0_3 (readings):** Challenger O-ring data (23 launches before Challenger); published referral rates from a 1999 medical study (18_2_0_2 only)
- **18_2_0_1 (older reading, replaced by 18_2_0_2 and 18_2_0_3):** none
- **18_2_1:** Titanic (survival)
- **18_2_2:** Titanic (survival), logistic regression vs. XGBoost
- **18_2_3:** XOR toy data; Titanic (MLP vs. logistic regression)
- **18_2_9 (exercise):** German Credit (`credit-g` from OpenML)

---

## 18_2_0_2: Logistic Regression in Plain Language (reading + videos)

**Topics:** What logistic regression does and how to read it, introduced through real cases. Each section begins with a StatQuest video.

- Yes/no outcomes: why a straight line fails and what the S-shaped curve does
- Probability, odds and log-odds; converting among them
- Odds ratios, and why an odds ratio is not a statement about how likely something is
- Coefficients: making a prediction, reading $e^{\beta}$, 0/1 features and the reference group
- The same odds ratio can change the probability by different amounts
- Maximum likelihood: choosing the curve that gives the most probability to what happened; link to log loss
- Gradient descent (optional section)

---

## 18_2_0_3: The Math Behind Logistic Regression (reading + videos)

**Topics:** Where the formulas in 18_2_0_2 come from. Calculus is optional.

- Rules for logs and exponents
- Solving the log-odds equation for $p$ to get the sigmoid; thresholds on the log-odds scale; decision boundary
- Why $e^{\beta}$ is the same odds ratio for every starting point; other unit sizes; the intercept; the $\beta/4$ limit
- Probability vs. likelihood; the likelihood and log-likelihood formulas
- Log loss (cross-entropy) for each observation and for the model; comparison with a baseline
- Gradient descent: the first step by hand, convergence, learning rate
- Assumptions and limits: linearity in the log-odds, interactions, independence, perfect separation, two classes

---

## 18_2_0_1: Logistic Regression Basics (older reading)

Replaced by 18_2_0_2 and 18_2_0_3, which cover the same material. Kept for students who have already studied from it.

---

## 18_2_1: Introduction with Titanic

**Topics:** Full end-to-end logistic regression workflow: EDA, pipeline, cross-validation, baselines, interpretation, evaluation.

### How Logistic Regression Works
- Why a straight line fails for a 0/1 target
- The sigmoid function; probability output
- Preview: this single sigmoid unit has the same structure as one neuron in a neural network (developed in 18_2_3)

### Building the Model
- Feature engineering on Titanic; train-test split
- `Pipeline` to prevent leakage (house convention)
- Cross-validation: is the model stable across folds?

### Is the Model Any Good?
- Establishing baselines before judging an accuracy number
- Reading the baselines vs. the model's ~77%

### Interpreting and Evaluating
- Odds ratios: the standard way to interpret logistic regression coefficients
- From coefficients to a predicted probability: the intercept and contributions worked through for two passengers
- Odds are not probabilities: what multiplying the odds does to a passenger's probability
- Predicted probabilities and calibration
- Classification report; ROC curve and AUC; precision-recall curve (reprise from 18_1)
- Threshold tuning

### When to Choose Logistic Regression
- Strengths: calibrated probabilities, interpretability, inference
- Limits: linear decision boundary

---

## 18_2_2: Interpretability: When the Model Must Explain Its Predictions

**Topics:** Statistical inference on logistic regression; what interpretable models offer that harder-to-explain models do not; regulatory context.

### Setup: Same Data, Two Models
- Sklearn logistic regression and XGBoost trained on identical Titanic features

### Confidence Intervals on Odds Ratios
- Why statsmodels (`Logit`) instead of sklearn: standard errors, p-values, CIs
- Forest plot of odds ratios with confidence intervals; the reference line at 1.0

### What Each Model Can and Cannot Tell You
- XGBoost feature importance vs. logistic regression coefficients: importance says *that* a feature matters, not *how* or *in which direction*

### When Interpretability Is Required
- The regulatory context (credit, medicine): adverse-action reasons, auditability
- Explaining an individual prediction by decomposing its linear score

---

## 18_2_3: From One Neuron to Many: Connecting Logistic Regression to Neural Networks

**Topics:** Logistic regression as a single neuron; the XOR limitation; hidden layers and activation functions; MLPClassifier as a black box; the interpretability trade-off.

### Logistic Regression Is a Single Neuron
- Numerical verification: $\sigma(w \cdot x + b)$ reproduces `predict_proba`

### The Problem Logistic Regression Cannot Solve
- XOR: four points no line can separate; the limits of a linear boundary

### From One Neuron to a Network
- Architecture: input, hidden layer, output
- Activation functions: sigmoid, tanh, ReLU, and why hidden layers need nonlinearity
- `MLPClassifier` solves XOR with a small hidden layer

### A Neural Network on Titanic
- MLP vs. logistic regression on the same features: similar performance on tabular data of this size

### What the Neural Network Loses
- The interpretability trade-off: no odds ratios, no CIs, no simple per-feature explanation
- Three-part module summary; forward pointer to the deep-learning course

---

## 18_2_9: Exercise: Credit Risk with Interpretability Constraints

**Topics:** Applying the full 18_2 workflow under a realistic constraint: the model must be able to explain its predictions.

- German Credit data (`fetch_openml('credit-g')`)
- Pipeline build, training, evaluation
- Odds ratios with confidence intervals; interpretation
- Explaining an individual applicant's prediction
- LR vs. XGBoost comparison: is any accuracy gain worth the loss of interpretability here?
