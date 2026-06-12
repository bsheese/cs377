# 19_1 Neural Networks — Topic Outline

This document provides a complete outline of all topics covered across the four
notebooks in the 19_1 Neural Networks series.

---

## 19_1_1: Anatomy of a Network

**Dataset:** XOR (the same 4 points from 18_2_3)

### The `nn.Module` Contract
- Every PyTorch model is a class inheriting from `nn.Module`
- `__init__` defines and registers the layers; `forward(x)` defines the computation
- You call `model(x)`, never `forward` directly
- `MinimalNet`: one `nn.Linear(2, 1)` + sigmoid — the smallest possible network
- `nn.Linear(in, out)` stores a weight matrix (out × in) and bias vector;
  forward computes `input @ weight.T + bias`

### Logistic Regression Is a One-Layer Network
- `LogisticRegressionModule`: `nn.Linear(n, 1)` + sigmoid
- Numerical verification that the module output equals σ(Wx + b) computed by hand —
  the same check done in 18_2_3 Section 1

### Adding a Hidden Layer — XORNet
- Why XOR needs a hidden layer (not linearly separable; recall 18_2_3 Section 2)
- `XORNet`: Linear(2, 4) → ReLU → Linear(4, 1) → sigmoid; 17 parameters total
- Annotated architecture diagram with weight-matrix shapes
- Training with the 19_0_3 five-step loop plus two new pieces: `nn.BCELoss` and
  `torch.optim.SGD` (so all 17 parameters update in one `optimizer.step()`)
- Loss curve, 100% accuracy on all four points, and the learned decision boundary —
  the same result sklearn's MLPClassifier produced in 18_2_3 (where it needed
  `tanh` + `lbfgs` to succeed; here every knob is visible in our own code)

### `nn.Sequential` — Shorthand for Simple Stacks
- The same XORNet as `nn.Sequential(Linear, ReLU, Linear, Sigmoid)`
- When to use each style: Sequential for straight pipelines, an `nn.Module` class for
  custom logic
- Introspection: `named_parameters()` (names + shapes), parameter counting,
  `state_dict()` for saving/loading

---

## 19_1_2: The Training Loop

**Dataset:** Wisconsin Breast Cancer (569 samples, 30 features — same data as unit 18_6)

### Data Preparation
- Three-way split: 60% train / 20% validation / 20% test, stratified
- Why a validation set: monitoring overfitting and guiding decisions during training
- `StandardScaler` fit on train only; conversion to tensors; targets shaped (n, 1)
  float for `BCELoss`

### Batching with `DataLoader`
- Full-batch vs. mini-batch gradient descent; why mini-batches are the practical default
- `TensorDataset` + `DataLoader(batch_size=32, shuffle=True)`
- 341 training samples → 11 batches → 11 parameter updates per epoch

### The Complete Training Loop
- The canonical annotated template: per-epoch training phase (`model.train()`, batch
  loop, five steps) followed by validation phase (`model.eval()` + `torch.no_grad()`)
- Why `model.train()`/`model.eval()` matter (Dropout/BatchNorm behavior — habit formed
  now, layers arrive later)
- A (64, 32) network with Adam + `weight_decay` (L2 regularization); train and
  validation loss plotted together
- Reading the curves: a mild, gently-rising validation loss vs. a collapsing training
  loss; rises-vs-plateaus as the question to ask

### Overfitting Is Visible in the Loss Curves
- A deliberately oversized (256, 256) network (~74K parameters on 341 samples,
  ratio ≈ 217×) with no regularization
- Training loss collapses to ~0; validation loss bottoms out early (best epoch ≈ 12)
  and then climbs — the overfitting signal
- Same bias-variance pattern as the tree-depth experiment in unit 18_6

### Adam vs. SGD
- What Adam adds: per-parameter adaptive learning rates + momentum
- Same architecture, same seed, both optimizers: Adam reaches a far lower *training*
  loss; on this small dataset SGD ends with the slightly better *validation* loss —
  faster fitting can mean faster overfitting
- Adam remains the practical default (no learning-rate hunt), paired with weight decay
  and loss-curve monitoring

### Evaluating on the Test Set
- Retrain on train+val, evaluate once on the untouched test set
- `classification_report` (sklearn) on the network's thresholded outputs;
  malignant recall as the clinically important metric (per 18_6)
- Result lands in the same range as the 18_6 ensemble methods (~94–97% nested-CV
  accuracy)
- Three-way-split reasoning: validation performance is slightly optimistic; only the
  test set is unbiased

---

## 19_1_3: Multiclass Classification

**Dataset:** Cardiotocography (CTG) fetal health — 2,126 samples, 35 features, 3
imbalanced classes (same data as 18_5_3)

### From Sigmoid to Softmax
- Binary: one output neuron + sigmoid. Multiclass: K output neurons producing K logits
- Softmax converts K logits to K probabilities summing to 1 — the same function inside
  XGBoost's `multi:softprob` (18_5_1)
- **The `CrossEntropyLoss` convention:** the loss takes raw logits and applies softmax
  internally; the model's final layer is `nn.Linear(h, K)` with *no* activation
- Binary vs. multiclass comparison table (output layer, loss, target shape)
- `CTGNet`: 35 → 128 → 64 → 3 with ReLU between layers
- Targets must be `torch.long` integers of shape (n,), not one-hot vectors

### Handling Class Imbalance
- 78% Normal: the imbalance trap from 18_5_3 revisited
- `compute_class_weight('balanced')` → tensor → `nn.CrossEntropyLoss(weight=…)`
- Each sample's loss scaled by its true class's weight — same math as 18_5_3's sample
  weights, different API

### Training the Multiclass Network
- Identical loop to 19_1_2; only the loss function changed
- Tracking macro F1 (the headline metric for imbalanced multiclass, per 18_5_3) on
  train and validation each epoch

### The Probability Output
- `torch.softmax(model(X), dim=1)` produces the same N×K probability matrix as
  XGBoost's `predict_proba()`; rows sum to 1
- Predicted class = `argmax(dim=1)`

### Evaluation
- Final model trained on train+val (class weights recomputed), evaluated once on test
- `classification_report` + confusion matrix; test macro F1 ≈ 0.98 — effectively tying
  the XGBoost result from 18_5_3 (~0.98 stratified-CV macro F1)

---

## 19_1_9: Exercise — Wine Quality Prediction

**Dataset:** Red Wine Quality, three-class Low/Medium/High target (same binning as
exercise 18_5_9)

### Tasks
1. Load the data and recreate the three-class target (given pattern from 18_5_9)
2. Integer labels, 60/20/20 stratified split, `StandardScaler`, tensors
   (`torch.long` labels), `DataLoader`
3. Define `WineNet(nn.Module)` with two hidden layers; count parameters
4. Train 200 epochs with `CrossEntropyLoss(weight=class_weights)` + Adam, tracking
   training loss and validation macro F1
5. Plot the curves; diagnose overfitting from them
6. `classification_report` and macro F1 on the held-out test set
7. 5-fold stratified cross-validation of the full pipeline (scale inside each fold);
   mean ± std macro F1 vs. the single-split result
8. Reflection questions: NN vs. XGBoost on this data, the value of per-epoch monitoring,
   dataset size vs. model choice, capacity vs. regularization, when to choose a network
   despite no analytic solution

Every TODO cell is comment-only (Run-All safe); each task has an
"Execute to see solution" cell that prints reference code.

---

## Supporting Materials

| File | Description |
|------|-------------|
| `19_1_practice_quiz.ipynb` | Practice quiz (12 questions across the three content notebooks) |
| `19_1_glossary.md` | Key terminology definitions |
| `19_1_outline.md` | This outline |

---

## Cross-Cutting Themes

| Theme | Notebooks |
|---|---|
| **The five-step training loop** | 19_1_1 (XOR), 19_1_2 (canonical batched template), 19_1_3 (unchanged for multiclass), 19_1_9 (student-written) |
| **Connection to 18_2_3** | 19_1_1 (XOR, MLPClassifier comparison, σ(Wx+b) verification) |
| **Connection to 18_5 / 18_6** | 19_1_2 (breast cancer, ensemble baseline), 19_1_3 (CTG, macro F1, class weights), 19_1_9 (wine, XGBoost comparison) |
| **Overfitting in loss curves** | 19_1_2 (deliberate overtraining), 19_1_9 (diagnosis task) |
| **sklearn for evaluation, PyTorch for training** | 19_1_2, 19_1_3, 19_1_9 (`classification_report`, `f1_score`, `compute_class_weight`) |
| **Loss-function conventions** | 19_1_1/19_1_2 (`BCELoss` + sigmoid), 19_1_3/19_1_9 (`CrossEntropyLoss` + raw logits) |
