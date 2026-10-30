# Supplemental activities

In-class activities that go with the course notebooks but are not part of a unit itself. Each notebook is self-contained and loads its data from this repo's raw GitHub URLs, like the unit notebooks.

| Notebook | Goes with | What it is |
|---|---|---|
| [Ridge_vs_Lasso_Pair](Ridge_vs_Lasso_Pair.ipynb) | [17_2_1_3 Regularization](../17_regression_crossval/17_2_1_MLR/17_2_1_3_MLR_Ames_Part3_Regularization.ipynb) | Pair activity on the Ames data: how Ridge and Lasso split the credit between near-identical "twin" features, why Lasso reaches exactly zero, and which penalty fits which situation. One partner writes short scikit-learn snippets in the notebook (hints included); the other records answers in the Canvas quiz. |
| [Nested_CV_Pair](Nested_CV_Pair.ipynb) | [17_2_1_4 Tuning](../17_regression_crossval/17_2_1_MLR/17_2_1_4_MLR_Ames_Part4_HyperparameterTuning.ipynb), [17_2_1_5 Nested CV](../17_regression_crossval/17_2_1_MLR/17_2_1_5_MLR_Ames_Part5_NestedCV.ipynb) | Pair activity on the Ames data: a recap of why Lasso reaches exactly zero, the two loops of nested cross-validation, why one test split can mislead (ten random splits against the nested estimate) and which model to deploy. One partner makes small edits to given scikit-learn code; the other records answers in the Canvas quiz. |
| [Nested_CV_Review](Nested_CV_Review.ipynb) | [Nested_CV_Pair](Nested_CV_Pair.ipynb) | Follow-up written after the Nested CV pair activity, for reading before Exam 2: works through the mistakes that came up in more than one pair (why Lasso stops at exactly zero, inner-loop sizes, pipeline grid keys, reading the fold alphas, why one test split is luck and not leakage, and what to deploy and report), with code that reproduces each number. Saved with outputs. |
| [Confusion_Matrix_Pair](Confusion_Matrix_Pair.ipynb) | [18_1_1 Foundations](../18_classification/18_1_Classification_Basics/18_1_1_Classification_Foundations.ipynb), [18_1_2 Confusion Matrix](../18_classification/18_1_Classification_Basics/18_1_2_Confusion_Matrix_Basic_Metrics.ipynb) | Pair activity on the German Credit model: the accuracy paradox and the two denominators on paper, then counting the four cells of the confusion matrix with masks, precision, recall and F1 by hand, a comparison in which the more accurate model (no `scale_pos_weight`) misses more defaulters and costs the bank more, and reading the classification report's macro and weighted averages. One partner makes small edits to given code; the other records answers in the Canvas quiz. |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsheese/cs377/blob/main/supplemental/Ridge_vs_Lasso_Pair.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsheese/cs377/blob/main/supplemental/Nested_CV_Pair.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsheese/cs377/blob/main/supplemental/Nested_CV_Review.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsheese/cs377/blob/main/supplemental/Confusion_Matrix_Pair.ipynb)
