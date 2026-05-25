# 06 — Pandas Intro: Discussion Questions

## Series and the Index
1. A Python list retrieves elements by position (`my_list[2]`). A pandas Series also lets you retrieve by label. When is label-based access more useful than position-based access? Give a concrete example from the Titanic dataset.
2. What happens when you perform arithmetic between two Series that have *different* indexes? Try it and describe what you observe. Why might this behavior be useful? When could it cause a bug?

## DataFrames
3. A DataFrame can be thought of as "a dictionary of Series." In what sense is this true? What does it clarify about how column selection (`df["col"]`) works?
4. When you select a single column with `df["col"]`, you get a `Series`. When you select multiple columns with `df[["col1", "col2"]]`, you get a `DataFrame`. Why does the double-bracket syntax matter? What would happen if you used single brackets with a list?

## Selecting and Filtering
5. What is the difference between `.loc[]` and `.iloc[]`? Construct an example where they return different results on the same DataFrame.
6. Why must you use `&` instead of `and` when combining boolean conditions on a DataFrame? What error or unexpected behavior do you get if you use `and`?
7. What is `SettingWithCopyWarning`, and why does pandas raise it? When should you use `.copy()`, and when is it unnecessary?

## Data Cleaning
8. The notebook introduces artificial missing ages to demonstrate cleaning techniques on an otherwise complete dataset. Why might the real Titanic data have had missing ages in the first place? Does the reason data is missing affect how you should handle it?
9. Why should you always compute fill values (e.g., median age) from the *training* set, not the full dataset? What problem does computing from the full dataset cause in a machine learning context?

## GroupBy
10. Explain split–apply–combine in your own words without using pandas terminology. Give a non-data-science analogy.
11. After a `.groupby()`, the grouping columns become the index of the result. What does `.reset_index()` do, and why is it often the next step?

## Pivot Tables and Crosstabs
12. `pd.crosstab()` and `pd.pivot_table()` can answer some of the same questions. When would you choose one over the other?
13. What does `normalize="index"` mean in `pd.crosstab()`? How does it differ from `normalize="all"`? Give a question that each form is best suited to answer.
14. The correlation between `pclass` and `survived` is negative: third-class passengers survived less often than first-class. Does this mean *being in third class caused lower survival*? What alternative explanations exist? What additional data would help distinguish between them?
