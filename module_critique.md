# Module 20 (Regularization & Practical Training) — Critique and Revision Plan

> Modules 16–19 were critiqued and revised in previous sessions; all of their plan items
> were implemented (see git history). This document covers Module 20.

Scope: all 6 notebooks in `20_regularization/20_1_Practical_Training/` plus `TODO.md`,
the module's build blueprint. Method: full read of every cell, execution-state audit,
verification of saved outputs against prose claims, cross-checks against the Module 19
notebooks this module continues, and a comparison of the build against the TODO spec.

## Overall Assessment

Module 20 is the strongest first draft of any module reviewed so far. The narrative
through-line (each technique controls the *late drift* of validation loss rather than
lowering its floor; checkpointing then captures the best epoch) is carried consistently
across all five content notebooks and is honestly supported by the experiments — including
two places where the notebooks deliberately show a technique *not* helping (BatchNorm does
not speed convergence with Adam on this small problem; the CTG ablation spread is 0.005
macro F1). Five of the six notebooks are already fully executed with outputs that match
their prose. The remaining problems are small and specific: one notebook was never
executed, the exercise's solution code uses syntax that only parses on Python ≥ 3.12, one
cross-reference points to the wrong section, and the three support files the blueprint
specifies are unbuilt.

---

## A. Execution and format

**A1. `20_1_1_Dropout.ipynb` has never been executed** (zero outputs, zero execution
counts) and its cells are missing `id` fields (nbformat emits `MissingIDFieldWarning`; the
other five notebooks are fine). Because the notebook is unexecuted, its carefully-written
interpretive prose is unverified — in particular Section 4's three-bullet reading of the
dropout comparison table ("the *best* validation loss is about the same … the
*end-of-training* loss is where they diverge sharply") and Section 5's claim that "p = 0.3
tends to land in the sweet spot here" alongside a computed `best_p` printout. If the run
selects a different `best_p`, the prose and the printed result will contradict each other
on the page. Execute, then reconcile.

**A2. Re-execution of the other five.** Their saved outputs appear to come from this venv
(numbers are mutually consistent and consistent with 19_1_2's 0.0513 best validation
loss), so a fresh run should reproduce them. Re-execute all six for environment
consistency, then diff the key numbers; reconcile prose if anything shifts.

## B. Student-facing code defects

**B1. The 20_1_9 solution code only parses on Python ≥ 3.12.** Five of the six solution
cells contain f-strings that reuse double quotes inside double-quoted f-strings —
`f"... {f1_score(..., average="macro"):.3f}"` and `f"... {es_h["best_epoch"]}"`. That is
PEP 701 syntax, legal only on Python 3.12+; on any older interpreter (and in some
students' local environments) it is a `SyntaxError`. Every other module's solutions use
single quotes inside f-strings. Because solutions are stored as printed strings, the
notebook itself executes fine — the bug only bites the student who pastes the code, which
is the worst place for it. Fix the quoting in all five solutions (Tasks 2–6).

## C. Cross-reference and prose fixes

**C1. `20_1_4` points to the wrong section.** Section 2's note says `weights_only=True`
"will be explained in Section 5" — the explanation is Section 6 (Section 5 is early
stopping). The intro's learning-objectives list is fine; only the inline pointer is off.

**C2. Blueprint deviations worth recording, not fixing.** The TODO planned a
"convergence speed" win for BatchNorm in 20_1_2 Section 5; the notebook honestly found
(and teaches) that with Adam on this small problem BatchNorm buys stability at high
learning rates, *not* speed — a better outcome than the plan. Similarly the TODO's 20_1_3
plan ("cosine should reach the lowest final loss") became the more defensible "schedule
beats fixed at the endpoint; step-vs-cosine differences are within noise." The outline
must describe the as-built notebooks, not the blueprint.

## D. Missing support files (specified by the module's own blueprint)

**D1.** `TODO.md` specifies — and leaves unchecked — `20_1_glossary.md` (with an explicit
term list), `20_1_outline.md`, and `20_1_practice_quiz.ipynb` (10–12 questions: dropout
placement, train/eval behavior, BatchNorm order, scheduler convention, checkpointing
patterns). As with Module 19, building these implements the author's existing plan. Build
all three in the established formats and check the boxes in `TODO.md`.

---

## Revision Plan

**Phase 1 — code & reference fixes (before execution)**
1. Fix the PEP-701 f-string quoting in the five 20_1_9 solution cells (B1).
2. Fix the "Section 5" → "Section 6" pointer in 20_1_4 (C1).
3. Normalize cell IDs in 20_1_1 (A1).

**Phase 2 — execution & reconciliation**
4. Execute all six notebooks (`venv/bin/jupyter nbconvert --execute --inplace`); the
   heaviest is 20_1_1 (five 300-epoch runs of a small network) — minutes, not hours.
5. Reconcile 20_1_1's prose with its first-ever outputs: the Section 4 comparison table
   reading, the Section 5 `best_p` selection and sweet-spot claim, and the final test
   report. Diff the other five notebooks' key numbers against their previous outputs and
   reconcile if anything moved.
6. Confirm no `.pt` artifacts remain (20_1_4 cleans up after itself; 20_1_5/20_1_9 use
   tempfiles).

**Phase 3 — support files (per TODO blueprint)**
7. Write `20_1_glossary.md` with the blueprint's term list (18_5/19_x glossary format,
   cross-referencing where each concept was first taught).
8. Write `20_1_outline.md` mirroring the as-built notebooks, including the honest-negative
   findings (C2).
9. Build `20_1_practice_quiz.ipynb` (12 questions across the five content notebooks) with
   the standard inline QuizApp, shipped unexecuted, all keys verified.
10. Check off the three support-file boxes in `TODO.md`.

**Phase 4 — verification**
11. Final sweep: all 6 content notebooks execute cleanly with zero unexecuted code cells
    and zero errors; quiz unexecuted by design with valid keys; no nested-double-quote
    f-strings remain in solution strings; no stray `.pt` files.

---

## Implementation Notes (all plan items completed)

All Phase 1–4 items were implemented. The execution pass surfaced one substantive finding
beyond the plan:

- **20_1_1's central dropout demonstration does not work as written — the prose was wrong
  and has been rewritten around the true result.** On its first-ever execution, the
  Section 4 comparison showed dropout at the textbook rate (p = 0.3) changing essentially
  nothing: best validation loss 0.0537 vs 0.0513 without dropout, end-of-training loss
  0.1317 vs 0.1348, and the training loss still collapsing to 0.0000 (the prose had
  claimed it "does not collapse as close to zero"). A verification sweep across rates
  confirmed the pattern: p = 0.5 visibly slows the climb mid-training and posts the lowest
  best loss (0.0484), but by epoch 300 every configuration drifts into the same band
  (~0.125–0.135) — at ~217 parameters per training sample, the network memorizes the
  training set *through* any tested amount of dropout noise. Sections 4–6 were rewritten
  honestly (consistent with the module's own honest-negatives theme): dropout damps but
  cannot stop late memorization in this regime, which is exactly why it needs the
  checkpointing/early-stopping partner from 20_1_4. The rate sweep's judging metric was
  changed from a single noisy end-of-training value to best-val + mean-of-last-50-epochs,
  the auto-selected `best_p` now follows best validation loss (selecting 0.5, matching the
  printed output instead of contradicting it), the summary table row was corrected, and
  the related 20_1_5 paragraph ("the full pipeline stays flat") was reattributed to early
  stopping cutting the run short, with the regularizers damping the climb inside that
  window. Glossary and outline match the corrected framing.
- The original Section 4/5 prose ("dropout's job is to stop the model walking away from
  the floor"; "p = 0.3 lands in the sweet spot") was written before the notebook was ever
  run — a textbook case of why the execute-then-reconcile policy exists.
- The other five notebooks reproduced their saved outputs **bit-identically** on
  re-execution (key-number diff: zero changes), so no further reconciliation was needed.
- Phase 1 fixes as planned: 9 PEP-701-only f-strings re-quoted across the five 20_1_9
  solutions (verified zero remain), the 20_1_4 "Section 5" pointer corrected to Section 6,
  and 20_1_1's missing cell IDs normalized (no more `MissingIDFieldWarning`).
- Support files built per the TODO blueprint and the checklist boxes checked:
  `20_1_glossary.md` (23 terms), `20_1_outline.md` (describing the as-built notebooks,
  including the honest-negative findings), and `20_1_practice_quiz.ipynb` (20 questions
  across the five content notebooks, standard inline QuizApp, unexecuted, all keys
  verified verbatim).

Final state: all 6 content notebooks execute end-to-end with zero errors and full saved
outputs; the quiz ships unexecuted with 20 verified questions; no stray `.pt` artifacts
(20_1_4 cleans up after itself; 20_1_5/20_1_9 use tempfiles); the stale-string and
PEP-701 sweeps are clean; `TODO.md`'s build checklist is fully checked off.
