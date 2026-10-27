# Supplemental activities

In-class activities that go with the course notebooks but are not part of a unit itself. Each notebook is self-contained and loads its data from this repo's raw GitHub URLs, like the unit notebooks.

| Notebook | Goes with | What it is |
|---|---|---|
| [Ridge_vs_Lasso_Pair](Ridge_vs_Lasso_Pair.ipynb) | [17_2_1_3 Regularization](../17_regression_crossval/17_2_1_MLR/17_2_1_3_MLR_Ames_Part3_Regularization.ipynb) | Pair activity on the Ames data: how Ridge and Lasso split the credit between near-identical "twin" features, why Lasso reaches exactly zero, and which penalty fits which situation. One partner writes short scikit-learn snippets in the notebook (hints included); the other records answers in the Canvas quiz. |
| [Nested_CV_Pair](Nested_CV_Pair.ipynb) | [17_2_1_4 Tuning](../17_regression_crossval/17_2_1_MLR/17_2_1_4_MLR_Ames_Part4_HyperparameterTuning.ipynb), [17_2_1_5 Nested CV](../17_regression_crossval/17_2_1_MLR/17_2_1_5_MLR_Ames_Part5_NestedCV.ipynb) | Pair activity on the Ames data: a recap of why Lasso reaches exactly zero, the two loops of nested cross-validation, why one test split can mislead (ten random splits against the nested estimate) and which model to deploy. One partner makes small edits to given scikit-learn code; the other records answers in the Canvas quiz. |
| [Nested_CV_Review](Nested_CV_Review.ipynb) | [Nested_CV_Pair](Nested_CV_Pair.ipynb) | Follow-up written after the Nested CV pair activity, for reading before Exam 2: works through the mistakes that came up in more than one pair (why Lasso stops at exactly zero, inner-loop sizes, pipeline grid keys, reading the fold alphas, why one test split is luck and not leakage, and what to deploy and report), with code that reproduces each number. Saved with outputs. |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsheese/cs377/blob/main/supplemental/Ridge_vs_Lasso_Pair.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsheese/cs377/blob/main/supplemental/Nested_CV_Pair.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsheese/cs377/blob/main/supplemental/Nested_CV_Review.ipynb)
