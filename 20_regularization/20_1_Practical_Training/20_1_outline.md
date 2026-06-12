# 20_1 Regularization & Practical Training — Topic Outline

This document provides a complete outline of all topics covered across the six notebooks
in the 20_1 Practical Training series.

**Running example:** all notebooks use the Wisconsin Breast Cancer dataset (569 samples,
30 features, the same 60/20/20 split as 19_1_2), with the deliberately oversized
(256, 256) network from 19_1_2 as the overfitting baseline. 20_1_5 adds the CTG fetal
health dataset (from 19_1_3) as the production-template counterpoint.

**The module's through-line:** none of these techniques lowers the *floor* of the
validation loss on this problem — what they change is the *late behavior* of training.
Dropout damps (but on this extreme network cannot stop) the late drift, a schedule calms
the endpoint, and checkpointing with early stopping captures the best epoch regardless.

---

## 20_1_1: Dropout

### The Problem Dropout Solves
- Reproducing the 19_1_2 overfit: training loss collapses toward zero while validation
  loss bottoms out early and then drifts upward
- The motivating idea: randomly remove neurons during training so no single neuron can be
  relied on — the network must learn redundant, generalizable paths

### `nn.Dropout(p)` in Practice
- Each forward pass independently zeros each activation with probability `p`; survivors
  scaled by `1 / (1 − p)` (demonstrated on an all-ones vector)
- Placement: after the activation — `Linear → ReLU → Dropout`
- Typical `p`: 0.2–0.5

### Why `model.train()` / `model.eval()` Matter (the promised explanation)
- Train mode: dropout active and stochastic — the same input gives a different output on
  every call (demonstrated 5× in each mode)
- Eval mode: dropout disabled and deterministic
- Forgetting `model.eval()` before predicting = predictions from a randomly crippled
  network, with no error raised

### Dropout in the Overfit Network
- The same (256, 256) architecture with `nn.Dropout(0.3)` after each ReLU; everything
  else held fixed
- The honest reading: at the textbook rate, best loss, end-of-training loss, *and* the
  collapsed training loss are all essentially unchanged — with ~217 parameters per
  training sample, the network memorizes *through* the dropout noise
- The lesson: dropout alone does not stop late-stage memorization on a network this
  oversized; it needs the checkpointing/early-stopping partner from 20_1_4

### The Dropout-Rate Trade-off
- p = 0.1 / 0.3 / 0.5 vs. the no-dropout baseline, judged by best validation loss and
  the mean over the last 50 epochs (a single endpoint value is too noisy on 114 samples)
- Mid-training, stronger dropout holds the loss visibly lower (p = 0.5 also posts the
  lowest best); by epoch 300 every configuration drifts into the same band
- No underfitting in sight even at p = 0.5 — the network is too oversized for that here
- 0.3–0.4 remains the standard starting point; these curves say "move up, and bring
  early stopping"

### Final Evaluation on the Test Set
- Retrain at the chosen rate on train+val; single evaluation on the untouched test set
- `model.eval()` before predicting, now mandatory

---

## 20_1_2: Batch Normalization

### The Problem BatchNorm Solves
- `StandardScaler` only normalizes the *inputs*; the guarantee is gone after one layer
- Demonstration: an 8-layer `Linear → ReLU` stack where the activation std decays from
  0.34 to 0.03 layer by layer — the signal (and its gradients) quietly vanishing
- "Internal covariate shift" as the name for the drift

### `nn.BatchNorm1d` in Practice
- Normalize per feature across the batch, then apply learned scale (gamma) and shift
  (beta) — a real layer with weights
- Placement: `Linear → BatchNorm → ReLU` (the safe default; the ReLU-first variant exists
  but is not recommended here)
- Demonstrated on deliberately mis-scaled data (mean ≈ 10, std ≈ 5 → mean 0, std 1), then
  inside the deep stack: activation std held steady at every depth

### Why BatchNorm Needs the Mode Switch — the Running-Statistics Story
- Training mode: normalize with the current batch's statistics *and* update
  `running_mean` / `running_var`
- Eval mode: use the frozen running statistics — deterministic, batch-independent
- The subtle bug demonstrated: evaluating in train mode keeps mutating the running stats,
  so evaluating the model changes the model
- The two halves of the train/eval rule: dropout (on/off) vs. BatchNorm (batch vs. frozen
  statistics)

### BatchNorm and Dropout Together
- The standard combined block: `Linear → BatchNorm → ReLU → Dropout`

### What BatchNorm Buys You (the honest version)
- At a normal learning rate (Adam, lr=1e-3) on this small problem: no visible convergence
  speedup — plain and BatchNorm networks are neck and neck (the speedup reputation comes
  from deep nets and plain SGD)
- At a 50× learning rate (0.05): the plain network thrashes (final loss ≈ 4.8); the
  BatchNorm network keeps descending smoothly (≈ 0.02) — stability is the payoff here

### Final Evaluation on the Test Set

---

## 20_1_3: Learning Rate Schedules

### Why a Fixed Learning Rate Is a Compromise
- Three strategies on the `L(w) = w²` bowl: fixed-too-large ping-pongs forever
  (5 → −5 → 5), fixed-small crawls, decaying converges fast *and* settles
- Deliberate optimizer choice: plain SGD, not Adam, so the learning rate's effect is
  visible (Adam's adaptive steps would mask it on a problem this small) — with the
  explicit note that schedules + Adam/AdamW are standard for real deep networks

### Step Decay with `StepLR`
- `StepLR(step_size=50, gamma=0.1)`: a staircase, read directly from the scheduler before
  any training
- Downside: abrupt 10× jumps

### Cosine Annealing with `CosineAnnealingLR`
- Smooth decay from the base rate to ~0 over `T_max` epochs; no abrupt jumps

### The `scheduler.step()` Convention
- Rule 1: after `optimizer.step()` (today's update uses today's rate)
- Rule 2: once per epoch, outside the batch loop — inside it, a 200-epoch cosine schedule
  finishes in ~18 epochs
- A reusable `train_with_schedule()` helper with an injectable `make_scheduler`

### Warmup
- `LinearLR(start_factor=0.1, total_iters=10)`: ramp up over the first 10 epochs
- Why: random initial parameters + large noisy gradients; standard for Transformers;
  composable with decay via `SequentialLR` (concept to recognize, not needed here)

### Schedules on the Real Network
- The (256, 256) network, SGD lr=0.5, 200 epochs: fixed vs. step vs. cosine
- The honest reading: all three reach the same *best* validation loss; the schedules win
  at the *endpoint* (less late-epoch drift); step-vs-cosine differences are within the
  noise of a single split — "schedule beats fixed," not "cosine beats step"

### Final Evaluation on the Test Set

---

## 20_1_4: Saving and Checkpointing

### `state_dict()` and `torch.save`
- The state dict: named parameter tensors (`net.0.weight`, …); ~296 KB for the
  74K-parameter network
- The file stores only the numbers — nothing about the architecture

### Loading a Checkpoint
- Fresh model + `load_state_dict(torch.load(path, weights_only=True))`; verified by
  matching predictions to the last decimal

### The Architecture Must Match
- Loading into a 128-wide model raises `RuntimeError: size mismatch for net.0.weight…` —
  the error to recognize when loading code drifts from training code

### Saving a Full Training State
- The resume idiom: `{'model', 'optimizer', 'epoch', 'val_loss'}` — the optimizer state
  matters because Adam keeps per-parameter running averages

### Early Stopping with Checkpoint Saving
- First, checkpointing alone over 300 epochs: best epoch lands early (~11), final
  validation loss ends ~2.6× the best — the wasted, harmful tail made visible
- Then the full pattern: save on every new best, stop after `patience=20` epochs without
  improvement, **restore the checkpoint** (the in-memory model at stopping time is
  `patience` epochs past the best — early stopping without the restore gives the wrong
  model)
- Patience trade-off: 10–25 common

### Why `weights_only=True`
- `torch.save` is pickle-based; unpickling untrusted files can execute arbitrary code;
  `weights_only=True` loads only tensors and safe primitives

### Final Evaluation on the Test Set
- The restored best checkpoint evaluated once; demo `.pt` files cleaned up

---

## 20_1_5: The Full Regularized Pipeline

### The Regularized Architecture
- `build_net(..., use_bn, use_dropout)`: stacks `Linear → BatchNorm → ReLU → Dropout`
  blocks with switches per technique — the switches make the ablation possible

### The Complete `train_model()` Function
- The module's deliverable: one reusable loop bundling checkpointing (tempfile), optional
  per-epoch scheduler, optional early stopping with restore; returns the best-epoch model
  plus history

### When Regularization Rescues a Model (breast cancer)
- Baseline (256, 256): best val 0.051 early, ends ~0.135 after 300 epochs — drifts to
  ~2.6× its best
- Full pipeline (BN + dropout + cosine + early stop): ends near its best (~0.06),
  early-stopped around epoch 40
- The subtle point: because `train_model` restores checkpoints, *both* models test well —
  the baseline needed checkpointing to rescue it; the regularized model is good all the
  way to its endpoint

### When the Baseline Is Already Excellent (CTG ablation)
- Same net four ways (baseline / +dropout / +batchnorm / +both), 150 epochs,
  checkpoint-restored: test macro F1 spread of **0.005** — within noise
- The lesson: regularization is a fix for overfitting, not a free upgrade; it is a safe
  default because it rescues you when overfitting and costs almost nothing when not

### The Final Training Report
- Full pipeline on CTG: best epoch, val loss, parameter count, test macro F1 ≈ 0.978,
  classification report

---

## 20_1_9: Exercise — The Regularization Toolkit on Wine Quality

Same dataset and three-class target as 19_1_9; the only thing changing is the model and
how it is trained.

### Tasks
1. Rebuild the wine data pipeline (60/20/20 split, scaler, tensors, loader, balanced
   class weights)
2. Write a reusable `train_model()` (loss + val macro F1 tracking, optional scheduler,
   optional early stopping with checkpoint restore); compare plain WineNet vs. +Dropout(0.3)
3. Add BatchNorm (`Linear → BatchNorm → ReLU`); compare convergence curves
4. Apply `CosineAnnealingLR`; compare best/final val macro F1
5. Early stopping with `patience=25`; report best epoch vs. stopping epoch
6. The fully regularized WineNet (BN + dropout + cosine + early stop);
   `classification_report` on the test set vs. the baseline
7. Reflection: which technique mattered on this small dataset and what that implies about
   whether the network was overfitting; the wasted-tail reading of early stopping; whether
   the toolkit changes the NN-vs-XGBoost verdict from 19_1_9

Every TODO cell is comment-only (Run-All safe); each task has an "Execute to see
solution" cell that prints reference code.

---

## Supporting Materials

| File | Description |
|------|-------------|
| `20_1_practice_quiz.ipynb` | Practice quiz (12 questions across the five content notebooks) |
| `20_1_glossary.md` | Key terminology definitions |
| `20_1_outline.md` | This outline |

---

## Cross-Cutting Themes

| Theme | Notebooks |
|---|---|
| **Late drift, not the floor** | 20_1_1 (dropout), 20_1_3 (schedules), 20_1_4/20_1_5 (checkpointing captures the best epoch) |
| **`model.train()` / `model.eval()`** | 20_1_1 (dropout on/off), 20_1_2 (batch vs. frozen statistics), everywhere afterwards |
| **Honest negative results** | 20_1_2 (no speedup with Adam at normal lr), 20_1_3 (step ≈ cosine within noise), 20_1_5 (CTG ablation spread 0.005), 20_1_9 (small-data verdict) |
| **The overfit (256, 256) showcase** | 20_1_1, 20_1_3, 20_1_4, 20_1_5 (breast-cancer section) |
| **Reusable training template** | 20_1_3 (`train_with_schedule`), 20_1_5 (`train_model`), 20_1_9 (student-written version) |
| **Connection to Module 19** | The overfit network and 0.0513 best val loss (19_1_2), CTG and macro F1 (19_1_3), WineNet (19_1_9), promises closed (train/eval, weight_decay context, state_dict, schedules) |
