# Discussion Questions: Data Cleaning for Multiple Linear Regression

## Initial Data Inspection

1. The notebook begins by using `.info()` to examine the DataFrame. What information does this provide that is critical before starting any modeling work? Why is the "Non-Null Count" column especially important?

2. The notebook removes rows where `Gr Liv Area` >= 4000, citing the original dataset author's recommendation. What kind of data problem can extreme outliers like these create for a linear regression model?

## Removing Uninformative Features

3. `Order` and `PID` are dropped because they are unique identifiers. If you accidentally left a unique identifier in a regression model, what would happen? What would the model "learn" from it?

4. The notebook checks for columns with only one unique value (`len(df[x].unique()) <= 1`). Why are monotonic (constant) features useless for modeling?

5. Duplicate rows and rows where all values are NaN are removed. What could cause duplicate rows to appear in a real dataset? Why might rows with all NaN values exist?

## Yolked Variables

6. The notebook describes "yolked features" where a categorical feature being "None" always corresponds to a numeric feature being 0. For example, when `Garage Type` is "None," `Garage Area` is always 0. Why does this create a problem for linear regression?

7. The notebook resolves yolked variables by dropping some features entirely (e.g., Pool QC, Pool Area, Garage Yr Blt) and combining others (e.g., collapsing `Garage Type` into a binary `garage_attached`). What is the reasoning behind these different approaches?

8. The notebook notes that `Electrical` is "yolked" with `Garage Area` and `Fence` is "yolked" with `Wood Deck SF`, but says these were "probably erroneously identified." Why might an automated detection algorithm produce false positives for yolked variables?

## Cleaning Categorical Features

9. The notebook drops categorical features where one category accounts for more than 70% of values. Why are highly unbalanced categorical features problematic for regression models?

10. For categorical features where one category is above 50%, the notebook creates a binary column (the top category vs. all others) rather than one-hot encoding all categories. What are the tradeoffs of this approach compared to full one-hot encoding?

11. The `Foundation` feature is collapsed from multiple categories into three: `PConc`, `CBlock`, and `Other`. Why might grouping rare categories into an "Other" category be preferable to keeping them separate?

12. `Exterior 1st` and `Exterior 2nd` are dropped because "I have no good plan for the exteriors, and I don't want to explode the one_hots." What does "explode the one_hots" mean, and why is having too many one-hot encoded columns a concern?

## Cleaning Numeric Features

13. Numeric features where one value accounts for more than 90% of observations are dropped entirely. Features where one value accounts for more than 80% are converted to boolean (0/1). What is the logic behind treating these two thresholds differently?

14. `Mas Vnr Area` is 60% zero and is dropped, but the notebook notes it "would be worth holding onto for non-basic models." Why might a feature that is mostly zero still be valuable in a more sophisticated model?

15. Missing numeric values are filled with the column median. Why is median imputation often preferred over mean imputation? Under what circumstances might median imputation introduce bias?

## Broader Data Cleaning Decisions

16. The notebook makes many subjective decisions: which features to drop, which categories to collapse, which thresholds to use. Two analysts cleaning the same dataset might make different choices. How could different cleaning decisions affect the final model's performance and interpretability?

17. The conclusion states: "If we planned on using other techniques (like what we will see in a few weeks), we'd leave in the more and clean a less." Why would different modeling techniques require different levels of data cleaning?

18. Throughout the notebook, the author drops features to keep the exercise shorter and simpler. In a real-world project, what are the risks of dropping features too aggressively during cleaning versus keeping too many?

## Feature Engineering Decisions

19. `Garage Type` is collapsed into `garage_attached` (1 if "Attchd", 0 otherwise), and `Garage Finish` is collapsed into `garage_unfinished` (1 if not "Unf", 0 otherwise). What information is lost when converting a multi-category feature into a binary one?

20. The `safe_drop` helper function checks whether columns exist before trying to drop them. Why is this kind of defensive coding important when writing data cleaning pipelines that might be run on different versions of a dataset?

## Preparation for Modeling

21. At the end of this notebook, the DataFrame has been reduced from 82 columns to approximately 38. The remaining features are a mix of numeric and boolean types. Why is having a clean, consistent set of feature types important before feeding data into a scikit-learn pipeline?

22. This notebook is labeled "Part 1: Data Cleaning" and precedes notebooks on feature selection, regularization, and cross-validation. Why is it critical to separate data cleaning from model building rather than doing both simultaneously?

23. The notebook's cleaning choices are explicitly made with "multiple linear regression techniques" in mind. Identify at least two decisions that would likely be different if the goal were to build a tree-based model (like Random Forest) instead.
