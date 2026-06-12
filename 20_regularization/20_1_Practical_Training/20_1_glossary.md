# 20_1 Regularization & Practical Training — Glossary

This document defines all technical and conceptual terms used across the five content
notebooks in the 20_1 Practical Training series. Terms already defined in the 19_0/19_1
glossaries are noted with cross-references.

---

## B

### Batch Normalization
Normalizing each layer's activations to roughly zero mean and unit variance *within a
mini-batch*, then applying a learned scale (gamma) and shift (beta) so the network can
undo the normalization if useful. Fights unstable training (not overfitting): it keeps a
usable signal flowing through deep networks and lets the optimizer tolerate much larger
learning rates.

### `nn.BatchNorm1d(num_features)`
The batch-normalization layer for fully connected activations. Placement:
`Linear → BatchNorm → ReLU` (normalize the linear output *before* the activation). A real
layer with learnable weights, plus non-learned `running_mean`/`running_var` buffers.

---

## C

### Checkpoint
A saved snapshot of model state during training. The light form is just the weights
(`torch.save(model.state_dict(), 'best_model.pt')`), overwritten whenever validation loss
hits a new low. The full form bundles
`{'model': …, 'optimizer': …, 'epoch': …, 'val_loss': …}` so an interrupted run can be
*resumed* — the optimizer's state matters because Adam keeps running averages per
parameter.

### `CosineAnnealingLR(optimizer, T_max)`
A learning-rate schedule that decays the rate smoothly along a cosine curve from the
starting value to near zero over `T_max` epochs — slow at first, faster through the
middle, flattening near the end. A sensible default schedule; widely used in modern deep
learning.

---

## D

### Dropout
The most widely used regularization technique in deep learning: during training, randomly
zero each activation with probability `p` on every forward pass. A network that cannot
rely on any single neuron must spread what it learns across redundant paths, which
generalizes better. An honest caveat from 20_1_1: on a network that is extremely
oversized relative to its data (~217 parameters per sample), dropout slows the late climb
of validation loss but cannot prevent memorization outright — there, it needs early
stopping (20_1_4) as a partner. Its reputation is earned on larger datasets and deeper
networks.

### `nn.Dropout(p)`
The dropout layer. `p` is the probability of zeroing each unit; survivors are scaled up
by `1 / (1 − p)` so the overall magnitude is preserved. Placement: *after* the
activation — `Linear → ReLU → Dropout`.

### Dropout Rate (`p`)
The tunable knob: typical values 0.2–0.5, with 0.3–0.4 a sensible starting point. Too
small and the late overfitting persists; too large and the network cannot fit the
training data at all (underfitting). Tune it on the *validation* set by reading the loss
curves.

---

## E

### Early Stopping
Halting training once validation loss has failed to improve for `patience` consecutive
epochs, then **restoring the best checkpoint**. The standard form of model selection for
neural networks. The restore step is essential: when training stops, the in-memory model
is from the *stopping* epoch — `patience` epochs past the best one — so early stopping
without restoring the checkpoint hands you the wrong model.

---

## I

### Internal Covariate Shift
The drift in the distribution of a layer's inputs as the parameters of earlier layers
change during training. Activations can shrink toward zero (vanishing signal and
gradients) or blow up, layer by layer. Batch normalization tames it by re-normalizing at
every block.

---

## L

### Learning Rate Schedule
A rule for changing the learning rate over the course of training — large early (fast
progress while parameters are far from good values), small late (settle into the minimum
instead of bouncing around it). What a schedule fixes is the *endpoint*: all strategies
reach about the same best loss on an easy problem, but a fixed rate leaves the final
epochs noisy and elevated. See also **Learning Rate** in the 19_0 glossary.

---

## M

### `model.train()` / `model.eval()`
The mode switch first introduced in 19_1_2 and fully explained in this module. For
**dropout**: train mode = random zeroing active (the regularization), eval mode =
disabled and deterministic (the inference requirement). For **BatchNorm**: train mode =
normalize with the current batch's statistics *and* update the running averages, eval
mode = use the frozen running statistics. Both layers are silently wrong — no error
raised — if you forget the switch.

---

## P

### Patience
The early-stopping knob: how many consecutive epochs without a validation-loss
improvement to tolerate before halting. Too small and ordinary epoch-to-epoch noise stops
you prematurely; too large and you waste compute. Values of 10–25 are common.

---

## R

### Running Statistics (`running_mean`, `running_var`)
Buffers each BatchNorm layer maintains during training: slow-moving averages of the batch
means and variances it has seen, serving as the layer's estimate of the population
statistics. Used (frozen) at evaluation time so predictions are deterministic and
independent of whatever batch a sample happens to arrive in. Evaluating while still in
train mode keeps *mutating* these buffers — evaluating the model literally changes the
model.

---

## S

### `scheduler.step()`
Advances a learning-rate schedule. Two rules: call it **after** `optimizer.step()` (the
optimizer uses the current rate; the scheduler sets the rate for the next round), and for
epoch-based schedulers like `StepLR`/`CosineAnnealingLR`, call it **once per epoch,
outside the batch loop** — inside the batch loop it advances once per *batch*, finishing
a 200-epoch schedule in a couple dozen epochs.

### `model.state_dict()`
See the 19_1 glossary. In this module it finally gets written to disk: it is the object
you pass to `torch.save`, and the file stores *only the numbers* — nothing about the
architecture, which is why loading requires a structurally identical model.

### `StepLR(optimizer, step_size, gamma)`
Step decay: hold the learning rate for `step_size` epochs, then multiply it by `gamma`
(e.g. ×0.1), producing a staircase. Simple and easy to reason about; its downside is the
abrupt order-of-magnitude jumps, which cosine annealing smooths away.

---

## T

### `torch.save(obj, path)` / `torch.load(path)`
Serialize an object (conventionally a `state_dict`, with a `.pt` extension) to disk and
read it back. `load_state_dict(torch.load(...))` copies the saved tensors into a fresh
model; a shape mismatch with the saved tensors raises a `RuntimeError` naming the
offending parameter — the signal that your loading architecture has drifted from the one
you trained.

---

## U

### Underfitting
The failure mode at the other end from overfitting: a model (for example, one with too
much dropout) that cannot even fit the training data well. Visible as a training loss
that never gets low. See also **Overfitting (in loss curves)** in the 19_1 glossary.

---

## W

### Warmup
Starting with a very small learning rate and ramping *up* over the first few epochs
(e.g. `LinearLR(start_factor=0.1, total_iters=10)`) before any decay begins — letting
randomly-initialized parameters find their footing before taking full-size steps.
Standard practice for training Transformers; usually composed with a decay schedule via
`SequentialLR`.

### `weights_only=True`
The safe-loading argument to `torch.load`. `torch.save` is built on Python's `pickle`,
and unpickling an untrusted file can execute arbitrary code; `weights_only=True` loads
only tensors and safe primitive types. Costs nothing when the file is your own, protects
you when it is not — use it on every load.
