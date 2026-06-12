# 19_1 Neural Networks — Glossary

This document defines all technical and conceptual terms used across the four notebooks
in the 19_1 Neural Networks series. Terms already defined in the 19_0 PyTorch Basics
glossary or earlier units are noted with cross-references.

---

## A

### Adam (Adaptive Moment Estimation)
An optimizer that improves on plain SGD in two ways: it maintains a separate, adaptive
learning rate for each parameter, and it uses momentum (a weighted average of past
gradients) to smooth noisy mini-batch gradient estimates. The practical default for
feedforward networks (`torch.optim.Adam`, `lr=1e-3` as a starting point). Caveat from
19_1_2: Adam fits faster, which can also mean overfitting faster — pair it with weight
decay and loss-curve monitoring.

---

## B

### Backpropagation
The algorithm that computes the gradient of the loss with respect to every weight in a
network by applying the chain rule backwards through the computation graph — what
`loss.backward()` runs. See **Autograd** in the 19_0 glossary; backpropagation is
reverse-mode automatic differentiation applied to a neural network.

### Backward Pass
The second half of a training step: traversing the computation graph from the loss back
to every parameter, filling each parameter's `.grad`. Triggered by `loss.backward()`.

### `nn.BCELoss`
Binary cross-entropy loss: the binary-classification loss used with a sigmoid output.
Penalizes confident wrong predictions much more heavily than uncertain ones. Expects
probabilities in (0, 1) and float targets of shape (n, 1). For the multiclass
equivalent, see **`nn.CrossEntropyLoss`**.

---

## C

### Class Weights (in the loss)
The `weight=` argument of `nn.CrossEntropyLoss`: a tensor of K per-class weights
(computed with sklearn's `compute_class_weight('balanced', …)`) that multiplies each
sample's loss contribution by the weight of its true class. Rare classes get more
gradient signal. The PyTorch analogue of the `sample_weight` mechanism from 18_5_3.

### `nn.CrossEntropyLoss`
The multiclass classification loss. **It expects raw logits, not softmax
probabilities** — it applies softmax and the log-loss in one numerically stable internal
step. Consequently the model's final layer is a plain `nn.Linear(hidden, K)` with no
activation, and targets are integer class labels (`torch.long`, shape (n,)), not one-hot
vectors. Applying softmax in the model *and* using CrossEntropyLoss double-applies
softmax and silently corrupts the gradients.

---

## D

### `DataLoader`
Wraps a dataset and yields mini-batches of a given `batch_size`; `shuffle=True`
randomizes sample order each epoch so the model cannot learn from the data's ordering.
The standard way to feed training data to a PyTorch loop.

---

## E

### Epoch
One complete pass through the training data. With mini-batches, the parameters are
updated once per batch — many times per epoch. See also the 19_0 glossary entry.

---

## F

### Forward Pass
The first half of a training step: running the input through the model
(`preds = model(X_batch)`) to produce predictions, building the computation graph along
the way. Defined by the model's `forward` method.

---

## H

### Hidden Layer
A layer between the input and the output whose neurons learn an intermediate
representation of the data. The transformation it learns can make a problem that is not
linearly separable (like XOR) separable in the new representation — which the output
layer, itself just a logistic regression, can then split with a straight line.

---

## L

### `nn.Linear(in_features, out_features)`
The fundamental building block: a fully connected layer storing a weight matrix of shape
(out, in) and a bias vector of shape (out,). Its forward computation is
`output = input @ weight.T + bias`. Logistic regression is one `nn.Linear(n, 1)` plus a
sigmoid.

### Logits
The raw, unbounded scores produced by the final `nn.Linear` layer before any activation.
For multiclass models, `torch.softmax(logits, dim=1)` converts them to probabilities;
`logits.argmax(dim=1)` gives the predicted class directly (softmax preserves order, so
the argmax is the same either way).

---

## M

### Mini-batch
A small subset of the training data (typically 32–256 samples) processed before each
parameter update. Mini-batch gradient descent is the practical middle ground between
full-batch gradient descent (one slow, exact update per epoch) and one-sample SGD (many
noisy updates).

### `model.train()` / `model.eval()`
Mode switches that change the behavior of layers like Dropout and BatchNorm (training
behavior vs. deterministic evaluation behavior). The 19_1 networks do not use those
layers yet, but the habit matters: call `model.train()` before the training phase and
`model.eval()` (together with `torch.no_grad()`) before validation and inference.

### `nn.Module`
The base class for every PyTorch model. The contract has two parts: `__init__` defines
and registers the layers (any module assigned as an attribute is automatically tracked),
and `forward(x)` defines the computation. You call the model like a function —
`model(x)` — and PyTorch invokes `forward` for you.

---

## O

### One-hot vs. Integer Targets
PyTorch's `CrossEntropyLoss` expects integer class labels (0, 1, 2, …) rather than
one-hot vectors — the opposite of the convention described in 18_5_1 for some other
frameworks. The integer form is more memory-efficient and lets the loss index the
correct class's probability directly.

### Overfitting (in loss curves)
Training loss continuing to fall while validation loss bottoms out and then rises — the
model has stopped learning generalizable patterns and started memorizing training noise.
The same bias-variance phenomenon seen with tree depth in 18_6, made directly visible by
plotting both curves per epoch.

---

## P

### `model.parameters()` / `model.named_parameters()`
Iterators over every trainable tensor registered in the module hierarchy (named variant
also yields each tensor's name). Passed to the optimizer at construction
(`torch.optim.Adam(model.parameters(), …)`); also used to count parameters:
`sum(p.numel() for p in model.parameters())`.

---

## R

### ReLU (Rectified Linear Unit)
The default hidden-layer activation: `max(0, z)`. Passes positive values unchanged,
clips negatives to zero. This non-linearity between linear layers is what lets a network
learn curved decision boundaries. See 18_2_3 for the comparison with sigmoid and tanh.

---

## S

### `nn.Sequential`
Shorthand for a straight-line stack of layers — `nn.Sequential(nn.Linear(2, 4),
nn.ReLU(), nn.Linear(4, 1), nn.Sigmoid())` — with no custom class needed. Use it when
the forward pass is just "apply these layers in order"; write an `nn.Module` class when
you need branching, multiple inputs, or any custom logic. Note: inside `Sequential` you
use the module versions of activations (`nn.ReLU()`), not the functional ones
(`torch.relu`).

### Sigmoid (output activation)
Maps any real number to (0, 1); the output activation for *binary* classification,
paired with `nn.BCELoss`. For multiclass, it is replaced by K logits +
`CrossEntropyLoss` (softmax applied internally).

### Softmax
Converts K logits into K probabilities that sum to 1:
`softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)`. The multiclass output function — the same one
inside XGBoost's `multi:softprob` (18_5_1). In PyTorch, applied explicitly only at
inference time (`torch.softmax(model(X), dim=1)`); during training, `CrossEntropyLoss`
applies it internally.

### `model.state_dict()`
An ordered dictionary mapping each named parameter to its tensor — a snapshot of all the
model's weights. The standard mechanism for saving and loading trained models.

---

## T

### `TensorDataset`
Pairs feature and target tensors so they are always sliced together; the dataset object
a `DataLoader` draws batches from.

### Test Set (vs. Validation Set)
The three-way split logic: the **training set** updates the weights; the **validation
set** guides decisions (architecture, epochs, learning rate) and is therefore slightly
optimistic; the **test set** is touched only once, at the very end, and gives the
unbiased performance estimate. Reporting validation performance as the final result is a
form of data leakage — the same principle behind the OOF/test-set discipline of 18_1.

---

## V

### Validation Loop
The per-epoch evaluation pass: `model.eval()`, `torch.no_grad()`, compute the loss (or a
metric like macro F1) on the held-out validation set. Plotting validation loss alongside
training loss is how overfitting is detected during training.

---

## W

### Weight Decay
L2 regularization applied by the optimizer (`weight_decay=` argument): a penalty on
large weights that slows the collapse of training loss and narrows the train/validation
gap. The neural-network analogue of Ridge regression's penalty from 17_2.
