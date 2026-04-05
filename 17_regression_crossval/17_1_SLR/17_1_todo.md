# 17_1 SLR Revision TODO List

Based on the comprehensive review performed on April 4, 2026.

## Structural & Sequence Changes
- [ ] **Swap 17.1.3 and 17.1.4**: R-squared is used as a diagnostic in the Residuals notebook (17.1.3) but isn't explained until the Correlation notebook (17.1.4). Swap the order and re-index.
- [ ] **Rename files**:
    - `17.1.3_SLR_Residuals...` -> `17.1.4_SLR_Residuals...`
    - `17.1.4_SLR_Correlation...` -> `17.1.3_SLR_Correlation...`
- [ ] **Establish Transitions**: Add explicit "In the previous notebook..." and "In the next notebook..." markdown cells to every file.

## Technical Fixes
- [ ] **Remove Colab Forms**: Delete all cells related to `exercise_report_response`, `cs125_tools`, and the mocked local tools.
- [ ] **Fix Markdown Rendering**: Repair all merged headings (e.g., `## HeadingBody text`) by ensuring double newlines between headings and paragraphs.
- [ ] **Reset Exercises**: In notebooks 17.1.2, 17.1.8, and 17.1.9, replace pre-filled solutions with empty exercise cells or skeleton code with `# enter your code here` comments.
- [ ] **Standardize Data Loading**: Move to local CSV loading or robust data generation for all notebooks to avoid 404 errors from external URLs.
- [ ] **Replace Boston Housing**: Replace the ethically deprecated Boston Housing dataset in 17.1.2 and 17.1.5 with a modern alternative (e.g., Ames Housing or a built-in Seaborn dataset like `tips` or `mpg`).

## Content & Terminology Gaps
- [ ] **Integrate Key Terms**: Ensure every term from OpenIntro Ch7 §7.4.2 appears in context:
    - [ ] coefficient of determination
    - [ ] correlation
    - [ ] extrapolation
    - [ ] high leverage
    - [ ] indicator variable
    - [ ] influential point
    - [ ] least squares line
    - [ ] leverage point
    - [ ] outcome
    - [ ] outlier
    - [ ] predictor
    - [ ] R-squared
    - [ ] regression sum of squares (SSR)
    - [ ] residuals
    - [ ] sum of squared error (SSE)
    - [ ] total sum of squares (SST)
- [ ] **17.1.0 Preview**: Add brief mentions of residuals, R-squared, and extrapolation to the introduction.
- [ ] **17.1.5 Bridge**: Explicitly connect the visualization of grouped slopes (`hue`, `col`) to the algebraic interpretation of indicator variables.
- [ ] **17.1.6 Connection**: Ensure the term "correlation" is used in the context of outlier impact.

## Deliverables
- [ ] **Practice Quiz**: Create `17_1_practice_quiz.ipynb` modeled after the 17_2 and 18_5 versions (approx. 100+ questions).
- [ ] **Glossary & Outline**: Create `17_1_glossary.md` and `17_1_outline.md` (based on `openintro_ch7_outline.md`).
- [ ] **Discussion Questions**: Create `17_1_discussion_questions.md`.
- [ ] **Final Execution**: Ensure all notebooks run top-to-bottom without error in the project `venv`.
