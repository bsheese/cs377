# 18_2 Logistic Regression — Glossary

This document defines the technical and conceptual terms used across the notebooks in the 18_2 Logistic Regression series.

---

## A

### Activation Function
The nonlinear function applied to a neuron's weighted sum before passing it on. Logistic regression uses the sigmoid; hidden layers in modern networks usually use ReLU. Without a nonlinearity between layers, stacking neurons collapses back into a single linear model — the nonlinearity is what buys expressive power.

---

## B

### Baseline (Dummy) Classifier
A deliberately trivial model — e.g., "always predict the majority class" — used as a floor for judging a real model's accuracy. On imbalanced data a high accuracy can be meaningless; a model is only as impressive as its distance above the baseline.

---

## C

### Calibration
How well predicted probabilities match observed frequencies: among all passengers given a 0.7 survival probability, about 70% should survive. Logistic regression is naturally well-calibrated because it directly models probability; many other classifiers (including boosted trees) output scores that need recalibration before being read as probabilities.

### Confidence Interval (on an Odds Ratio)
The range of odds-ratio values consistent with the data. Statsmodels reports these; sklearn does not. If the interval contains 1.0, the data cannot rule out "no effect" for that feature. Visualized in the forest plot of 18_2_2.

---

## D

### Decision Boundary
The surface in feature space where the model's predicted probability equals the threshold (usually 0.5), separating predicted classes. Logistic regression's boundary is always a straight line (hyperplane); a neural network with hidden layers can bend it — which is exactly what solving XOR requires.

---

## F

### Forest Plot
A plot displaying each feature's odds ratio as a point with a horizontal line for its confidence interval, with a vertical reference line at 1.0. Features whose intervals cross 1.0 have effects the data cannot distinguish from zero.

---

## H

### Hidden Layer
A layer of neurons between input and output. Each hidden neuron computes its own weighted sum and activation; the output neuron then combines *their* outputs. Hidden layers are what let a network represent functions (like XOR) that no single neuron can.

---

## I

### Interpretability Tax
The trade-off incurred when moving from logistic regression to more flexible models: accuracy may improve, but per-feature effect sizes, directions, confidence intervals, and simple individual-prediction explanations are lost. In regulated domains (credit, medicine), that loss can be disqualifying.

---

## L

### Log-Odds (Logit)
The quantity logistic regression is actually linear in: $\log\frac{p}{1-p} = \beta_0 + \beta_1 X_1 + \dots + \beta_k X_k$. Each coefficient is the change in log-odds per unit of its feature; exponentiating a coefficient gives its odds ratio.

### Logistic (Sigmoid) Function
$\sigma(z) = \frac{1}{1 + e^{-z}}$ — squashes any real number into (0, 1), turning the linear combination of features into a probability. The reason a straight-line fit fails for binary targets and logistic regression does not.

### Logistic Regression
A classifier that models the *probability* of the positive class by passing a linear combination of the features through the sigmoid. Despite the name, it is a classification method. Its strengths: calibrated probabilities, interpretable coefficients, statistical inference; its limit: a linear decision boundary.

---

## M

### MLPClassifier (Multi-Layer Perceptron)
Sklearn's feedforward neural network. Used in 18_2_3 as a black box to show that adding hidden layers solves XOR and to compare against logistic regression on Titanic. Its internals (training loop, backpropagation) are deliberately left opaque in this course.

---

## N

### Neuron
The unit of computation in a neural network: a weighted sum of inputs plus a bias, passed through an activation function — $\sigma(w \cdot x + b)$. Logistic regression *is* a single neuron with a sigmoid activation; 18_2_3 verifies this numerically.

---

## O

### Odds
The ratio of the probability an event happens to the probability it doesn't: $\text{odds} = \frac{p}{1-p}$. A probability of 0.75 is odds of 3 ("three to one").

### Odds Ratio
$e^{\beta}$ for a coefficient $\beta$: the multiplicative change in odds per one-unit increase in that feature, holding the others fixed. An odds ratio of 2 doubles the odds; 0.5 halves them; 1.0 means no effect. The standard interpretable currency of logistic regression.

---

## R

### ReLU (Rectified Linear Unit)
$\text{ReLU}(z) = \max(0, z)$ — the default hidden-layer activation in modern networks. Unlike sigmoid, it does not saturate for large positive inputs, which makes deep networks much easier to train. Sigmoid remains the right choice for the *output* of a binary classifier, where a probability is needed.

---

## S

### Statsmodels `Logit`
The inference-oriented implementation of logistic regression: same coefficients as sklearn (when regularization is off), but with standard errors, p-values, and confidence intervals. The 18_2 series' rule of thumb mirrors 17_1: sklearn to *predict*, statsmodels to *explain*.

---

## T

### Threshold (Decision Threshold)
The probability cutoff for converting predicted probabilities into class labels, 0.5 by default. Moving it trades precision against recall — reprised from 18_1's ROC and precision-recall material, now applied to logistic regression's calibrated probabilities.

---

## X

### XOR Problem
The four-point dataset — (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0 — that no linear decision boundary can separate. The classic proof that a single neuron is not enough and the motivating example for hidden layers in 18_2_3.
