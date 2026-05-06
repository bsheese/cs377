# TODO — Notebook Improvements

Items below are the remaining criticisms from the initial review. Spelling fixes and the Part 1/Part 2 bridge (the `load_and_clean_ames()` transition) have already been addressed.

---

### 1. Refactor colossal functions into smaller, teachable pieces

`perform_feature_selection_and_evaluation` (Part 3) is ~140 lines with 5 internal code paths controlled by a string parameter. `run_modeling_workflow` (Part 2/3) is ~100 lines. Both mix scaling, feature selection, evaluation, and printing into monoliths. Undergraduates will never write code like this and learn little from reading it.

**Plan:** Break each into focused helper functions (e.g., `build_ols_pipeline()`, `build_ridge_pipeline()`, `evaluate_pipeline()`). Six 15-line functions teach more than one 140-line function.

---

### 2. Remove or explain hidden automation in Part 1's data cleaning

The notebook waves away several key decisions:
- "The code to automate the search for these features is not included in this notebook. You do not need to look at it or know how it works." (yolked variable detection)
- "To try to make this exercise shorter, I'm not going to explain these" (dropping ~15 columns)
- "These have probably been erroneously identified by our code" (about yolked pairs kept anyway)

**Plan:** Either include the detection code (or a simplified version) so students see the logic, or rephrase these lines to acknowledge the limitation honestly.

---

### 3. Reconsider Part 4's deliberate model-sabotage

Part 4 drops 18 of the best features with: "to make this a bit harder for the model." Test R² drops from ~0.93 to ~0.83. The stated goal is to demonstrate grid search and learning curves, but the same tools can be demonstrated on the original feature set. As-is, students see their hard-won model collapse for no clear reason.

**Plan:** Either run the section on the unchanged feature set, or add a clear explanation: "We are deliberately removing the strongest predictors to create a scenario where hyperparameter tuning makes a visible difference. If the model already scored 0.93, the tuning curves would be nearly flat."

---

### 4. Remove the "To Do" from published materials

Part 5 contains: *"To Do: This is using sqrt of features to specify the number of features to consider at each node. This is appropriate for standard classification tasks (regular trees), but for regression trees the more mathematically optimal number is 1/3 of the features... This will be corrected for future classes."*

Current students are knowingly receiving incorrect guidance while being told it will be fixed for a later class.

**Plan:** Fix the `max_features` choice now (change default to match regression best practices, e.g., use `max_features=1/3` or explain the tradeoff without a TODO), and remove the TODO note.

---

### 5. Restructure the practice quiz

The single cell contains ~1,500+ lines of nested Python dictionaries mixed with ipywidgets code. It only works in Jupyter/Colab. There are no answer explanations — the "correct" answer is just highlighted green.

**Plan:** Extract quiz data into a separate JSON or CSV file. Add explanation text for each answer. Consider a markdown-based format that works in any viewer.

---

### 6. Align evaluation strategy between Part 4 and Part 5

Part 4 spends significant time teaching nested cross-validation as the gold standard. Part 5 opens Decision Trees and Random Forest sections with a basic train-test split (no CV at all), only adding nested CV for XGBoost at the end.

**Plan:** Either apply CV consistently across all Part 5 models, or add a note explaining why the simpler evaluation is acceptable for the initial tree exploration.

---

### 7. Harmonize CV fold choices across notebooks

Part 2 uses `cv_folds=2` ("to keep computation time manageable"). Part 3 uses `cv_folds=5`. Part 4 uses 5. Part 5's nested CV uses 5 outer and 3 inner. There is no explanation of when to choose one over the other.

**Plan:** Add a consistent rule (e.g., "we use 5-fold CV throughout unless otherwise noted") and explain the computational tradeoff in one place rather than ad-hoc.

---

### 8. Break up dense markdown wall-of-text sections

Part 3's Ridge explanation is a 40-line single-paragraph cell. Lasso is 35 lines. Part 2's polynomial explanation is similar. Students will not read these.

**Plan:** Add sub-headings, bullet lists, bold key terms, summary callout boxes, or inline diagrams.

---

### 9. Add active learning / "you try it" sections

Every code cell in every notebook is pre-filled. Students run and watch. There are no blank cells, challenges, or exercises across all 5 parts + 4 supplementals.

**Plan:** Add 2-3 minimal exercises per notebook: e.g., "Try running forward selection with a different random_state" or "Add `Lot Area` squared to the polynomial list and see if it gets selected."

---

### 10. Integrate or cut the three "notes" notebooks

- `notes_on_trees` (237 lines): One analogy — "a decision tree is like 20 questions" — with hand-drawn matplotlib art.
- `notes_on_curves` (339 lines): Uses synthetic data, not the Ames dataset students have been working with.
- `nested_cv_notes` (167 lines): Restates Part 4's nested CV content with a studying-for-finals analogy.

Combined, a student would spend ~1 hour reading and running these for very little new information.

**Plan:** Either fold the useful parts into the main notebooks as sidebars, or remove them. The nested CV notes analogy could be a useful callout box inside Part 4 rather than a standalone notebook.

---

### 11. Remove duplicated cells in Part 2

The two cells at `execution_count` 10 and 11 are identical — same `calculate_vif` function, same output, same markdown header `"First, let's check VIF across all numeric features:"` appearing twice in a row.

---

### 12. Fix the Part 3 "full features" comparison table

Part 3 reloads raw data mid-notebook with minimal cleaning (277 features), then builds a comparison table that mixes:
- The OLS result from the earlier 38-feature section (19 features selected via forward selection)
- New results for Ridge/Lasso/ElasticNet on the 277-feature set

The comparison is misleading because the feature sets differ. Additionally, Ridge showing "272 features kept" is a display artifact (the `> 1e-5` threshold catches only 4 coefficients that happened to shrink below precision), not a real property of Ridge.

**Plan:** Run all models on the same feature set, or create two separate comparison tables.

---

### 13. Standardize Colab badge format

Part 1 uses the old-style `<a href="...">` HTML badge. Most other notebooks use the markdown-style `[![Open In Colab]...]` badge. `notes_on_curves` has two back-to-back.

**Plan:** Pick one format and apply it consistently across all notebooks. Remove duplicates.
