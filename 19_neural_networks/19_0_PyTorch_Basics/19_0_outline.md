# 19_0 PyTorch Basics — Topic Outline

This document provides a complete outline of all topics covered across the three
notebooks in the 19_0 PyTorch Basics series.

---

## 19_0_1: Tensors — PyTorch's Fundamental Data Structure

**Dataset:** None (toy arrays; Palmer Penguins columns as a conversion example)

### What Is a Tensor?
- A multi-dimensional array: scalar (0-D), vector (1-D), matrix (2-D), and beyond
- Creating tensors with `torch.tensor()` from lists and nested lists
- The four key attributes: `.shape`, `.dtype`, `.ndim`, `.device` (plus `.numel()`)
- The default-dtype difference: PyTorch `float32` vs. NumPy `float64`, and why
  (float32 is twice as fast on GPU hardware)

### Creating Tensors from NumPy and pandas
- `torch.from_numpy(arr)` shares memory; `torch.tensor(arr)` always copies
- `.numpy()` to go back (requires CPU tensor without gradient tracking)
- The pandas pattern: clean in pandas → `.values` → `torch.tensor(..., dtype=torch.float32)`
- Convenience constructors: `zeros`, `ones`, `rand`, `arange`, `linspace`, `eye`

### The NumPy Analogy
- Side-by-side operation table: creation, indexing, slicing, arithmetic, matrix multiply,
  reductions, reshape, transpose, stack/cat — identical syntax verified on shared data
- Four divergences that cause bugs:
  1. `.item()` to extract a Python scalar (reductions return 0-D tensors)
  2. In-place operations end with an underscore (`x.add_(10)`)
  3. Default dtype (`float32` vs. `float64`) — always pass `dtype=torch.float32`
  4. `.numpy()` preconditions — use `.detach().numpy()` on tracked tensors

### Device-Agnostic Code
- `device = 'cuda' if torch.cuda.is_available() else 'cpu'` boilerplate
- `.to(device)` on tensors and models; the training loop is identical on any hardware
- CPU is expected (and sufficient) for this course

### The One Thing Tensors Have That Arrays Don't
- `requires_grad=True` as a teaser: PyTorch silently records every operation
- `grad_fn` appears on computed tensors — the record the next notebook explains

---

## 19_0_2: Autograd — How PyTorch Computes Derivatives

**Dataset:** None (pure math on small tensors)

### Computing a Derivative Automatically
- The simplest case: `y = x**2`, `y.backward()`, `x.grad` — verified against calculus
  (dy/dx = 2x)
- Two-panel diagram: the forward pass records (`grad_fn`), the backward pass
  differentiates
- Leaf nodes (created with `requires_grad=True`) are where gradients land; intermediate
  tensors are waypoints

### The Chain Rule Made Automatic
- Hand calculation of dz/da and dz/db for z = (ax + b)², then autograd verification —
  the numbers match (42 and 14)
- Scaling up: a single neuron (`w1*x1 + w2*x2 + bias`) with squared-error loss; one
  `.backward()` call yields the gradient for every parameter simultaneously
- Data tensors (x) do not get `requires_grad` — you optimize weights, not data

### The Gradient Accumulation Trap
- `.grad` accumulates across `.backward()` calls by default (demonstrated: 6, 12, 18
  instead of 6, 6, 6)
- The fix: zero gradients at the start of every step (`x.grad.zero_()`; in a real loop,
  `optimizer.zero_grad()`)
- The non-negotiable step order: zero → forward → loss → backward → update

### Turning Off Gradient Tracking
- `torch.no_grad()` context manager: `grad_fn = None` inside the block
- When to use it: validation loop, metric computation, inference
- The trade: no backward capability in exchange for less memory and time

### What This Means for a Neural Network
- A network is a deep composition of operations; `loss.backward()` traverses the whole
  graph and fills every weight's `.grad` in one pass
- The annotated five-line training loop, with every line mapped back to a section of
  this notebook
- Reverse-mode automatic differentiation = backpropagation; cost is roughly one forward
  pass regardless of parameter count

---

## 19_0_3: Gradient Descent from Scratch

**Dataset:** Palmer Penguins — `flipper_length_mm` → `body_mass_g` (the familiar
regression from Unit 17), both variables standardized

### Setup and the Loss Bowl
- Why iterate at all: linear regression has a closed-form solution (17_0_5); neural
  networks do not — gradient descent is the only option
- Standardizing both variables (zero mean, unit variance) and why it helps: comparable
  gradient scales, loss starts near 1.0, optimal bias is exactly 0
- Reproducing the 17_0_5 RSS bowl in PyTorch: MSE vs. weight sweep with bias fixed at 0
- For standardized variables the optimal slope equals Pearson r

### One Gradient Descent Step by Hand
- Start at w = 0, b = 0; forward pass, `loss.backward()`, inspect `w.grad` and `b.grad`
- Manual update inside `torch.no_grad()`: `w -= lr * w.grad`
- Verify the loss decreased

### The Training Loop
- The five-step loop (zero → forward → loss → backward → update) run for 300 epochs
- The loss curve: rapid early descent, then convergence
- Converged slope matches Pearson r

### Gradient Descent vs. the Analytic Solution
- Convert the standardized-space parameters back to original units (g/mm)
- Overlay against sklearn's `LinearRegression` line — visually indistinguishable
- Lesson: for linear regression the analytic shortcut wins; for networks there is no
  shortcut

### Learning Rate Sensitivity
- Three-panel experiment: lr = 0.001 (too small — still converging after 300 epochs),
  lr = 0.1 (just right — converges within ~50), lr = 1.5 (too large — diverges within a
  few steps)
- No universal formula: try values spanning orders of magnitude and pick the one that
  converges cleanly

---

## Supporting Materials

| File | Description |
|------|-------------|
| `19_0_practice_quiz.ipynb` | Practice quiz (12 questions across the three notebooks) |
| `19_0_glossary.md` | Key terminology definitions |
| `19_0_outline.md` | This outline |

---

## Cross-Cutting Themes

| Theme | Notebooks |
|---|---|
| **NumPy → tensor bridge** | 19_0_1 (operation table, divergences) |
| **The computation graph** | 19_0_1 (teaser), 19_0_2 (built and traversed), 19_0_3 (used by the loop) |
| **Zeroing gradients** | 19_0_2 (the trap), 19_0_3 (in the loop) |
| **`torch.no_grad()`** | 19_0_2 (introduced), 19_0_3 (parameter updates, evaluation) |
| **Connection to Unit 17** | 19_0_3 (RSS bowl from 17_0_5; penguins regression; analytic vs. iterative) |
| **The five-step training loop** | 19_0_2 (annotated preview), 19_0_3 (implemented from scratch) |
