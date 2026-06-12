# 19_0 PyTorch Basics — Glossary

This document defines all technical and conceptual terms used across the three notebooks
in the 19_0 PyTorch Basics series. Terms already defined in earlier units are noted with
cross-references.

---

## A

### Autograd
PyTorch's automatic differentiation engine. When a tensor has `requires_grad=True`,
autograd records every operation applied to it in a computation graph. Calling
`.backward()` on a result traverses that graph in reverse, applying the chain rule at
each step, and deposits the gradient of the result with respect to every tracked tensor
in that tensor's `.grad` attribute. This is reverse-mode automatic differentiation — the
algorithm also known as backpropagation.

---

## B

### `.backward()`
The method that triggers the backward pass. Called on a scalar tensor (typically the
loss), it computes the gradient of that scalar with respect to every leaf tensor with
`requires_grad=True` and accumulates the results into each leaf's `.grad`. Called once
per training step.

---

## C

### Chain Rule
The calculus rule for differentiating composed functions: if z depends on u and u depends
on x, then dz/dx = (dz/du)·(du/dx). Autograd applies this rule numerically at every
operation in the computation graph, working backwards from the loss. A neural network is
a deep composition of operations, so the chain rule is applied many times per backward
pass.

### Computation Graph
The record autograd builds during the forward pass: a directed graph whose nodes are
tensors and whose edges are the operations that produced them (visible as each tensor's
`grad_fn`). The backward pass traverses this graph from the loss back to every leaf.
The graph is rebuilt fresh on every forward pass.

### Convergence
The point at which the loss stops meaningfully decreasing across epochs — the parameters
have (approximately) reached the bottom of the loss bowl. Visible as the loss curve
flattening out.

---

## E

### Epoch
One complete pass through the training data. In the full-batch gradient descent of
19_0_3, one epoch = one parameter update; with mini-batches (introduced in 19_1_2), one
epoch contains many updates.

---

## G

### Gradient
The derivative of the loss with respect to a parameter — the slope of the loss landscape
at the current parameter value. A positive gradient means the loss increases as the
parameter increases (so gradient descent moves the parameter down); a negative gradient
means the opposite. Stored in each parameter's `.grad` attribute after `.backward()`.

### Gradient Accumulation
PyTorch's default behavior: each `.backward()` call *adds* its result to `.grad` rather
than replacing it. Useful in advanced settings (accumulating over several mini-batches),
but in a normal training loop it is a silent bug — which is why every step begins by
zeroing the gradients.

### Gradient Descent
The iterative optimization algorithm behind all neural network training: compute the
gradient of the loss at the current parameters, take a small step in the downhill
direction (`w = w − lr × w.grad`), and repeat until convergence. Connects directly to the
RSS bowl from 17_0_5 — gradient descent is how you find the bottom of the bowl when no
closed-form formula exists.

### `.grad`
The attribute where autograd deposits a tensor's gradient after `.backward()`. `None`
until the first backward pass; afterwards a tensor with the same shape as the parameter.

### `grad_fn`
The attribute recording which operation produced a tensor (e.g. `PowBackward0` for
`x ** 2`). Leaf tensors you created yourself have `grad_fn = None`; computed tensors
carry the operation record that the backward pass follows.

---

## L

### Leaf Node / Leaf Tensor
A tensor created directly by the user (not computed from other tensors). Leaves with
`requires_grad=True` — model parameters — are where final gradients land during the
backward pass. Intermediate results are waypoints; their gradients are not retained by
default.

### Learning Rate
The step-size multiplier in the gradient descent update `w = w − lr × w.grad`. The most
consequential hyperparameter: too large and the loss diverges (the step overshoots the
minimum and bounces outward); too small and convergence takes impractically many steps.
The standard approach is to try values spanning several orders of magnitude (0.001, 0.01,
0.1) and pick the one that converges cleanly.

### Loss Landscape
The surface formed by plotting the loss at every possible combination of parameter
values. For one parameter it is a curve (the U-shaped bowl from 17_0_5); for millions of
parameters it is a high-dimensional surface that gradient descent navigates one local
slope at a time.

---

## M

### MSE (Mean Squared Error)
The average of squared prediction errors: RSS divided by n. The regression loss used in
19_0_3; see the RSS and TSS entries in the 17_0 glossary for the underlying ideas.

---

## R

### `requires_grad`
The flag that turns on gradient tracking for a tensor. Parameters are created with
`requires_grad=True`; input data is not (you optimize weights, not data). The one
capability tensors have that NumPy arrays do not.

---

## S

### SGD (Stochastic Gradient Descent)
In PyTorch, `torch.optim.SGD` — the optimizer implementing the plain gradient descent
update `w = w − lr × w.grad` across all parameters at once. ("Stochastic" refers to
computing gradients on random mini-batches rather than the full dataset; mini-batches
arrive in 19_1_2.) Compare with **Adam** in the 19_1 glossary.

### Standardization
Rescaling a variable to zero mean and unit variance — the same operation as sklearn's
`StandardScaler`. In 19_0_3 it keeps the gradients for the weight and bias on a similar
scale so a single learning rate works for both, and it makes the optimal bias exactly 0
for centered data.

---

## T

### Tensor
PyTorch's fundamental data structure: a multi-dimensional array of numbers, analogous to
a NumPy array, with two additional capabilities — device-agnostic computation
(`.to(device)`) and automatic gradient tracking (`requires_grad=True`). Defaults to
`float32` (NumPy defaults to `float64`).

### `torch.no_grad()`
A context manager that suspends computation-graph building for everything inside it.
Used for the validation loop, metric computation, and inference — any forward pass that
will never need a `.backward()` call. Saves memory and time.

### Training Loop
The five-step cycle repeated every epoch: (1) zero the gradients, (2) forward pass,
(3) compute the loss, (4) `.backward()`, (5) update the parameters. The same structure
trains a one-parameter regression and a hundred-layer network — only the forward pass
changes.

---

## Z

### `.zero_grad()` / `optimizer.zero_grad()`
Resets `.grad` to zero on a parameter (or, via the optimizer, on every parameter at
once). Always called at the start of each training step; forgetting it is the most common
silent bug in a first PyTorch training loop, because gradients accumulate by default.
