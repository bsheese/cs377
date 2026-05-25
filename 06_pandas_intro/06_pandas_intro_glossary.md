# 06 — Pandas Intro: Glossary

**aggregation** — A function that reduces many values to one (e.g., mean, sum, count).

**axis** — The direction of an operation. `axis=0` means along rows (default for most operations); `axis=1` means along columns.

**boolean mask** — A Series of `True`/`False` values used to filter rows or columns.

**category dtype** — A pandas data type optimized for columns with a small number of distinct string or integer values. More memory-efficient than `object`.

**copy** — An independent DataFrame not linked to the original; modifying it does not affect the source.

**contingency table** — Another name for a cross-tabulation: a table showing the joint frequency distribution of two categorical variables.

**crosstab** — Short for cross-tabulation; `pd.crosstab()` produces frequency (or proportion) counts for combinations of two categorical variables.

**DataFrame** — A two-dimensional tabular data structure; a collection of Series sharing the same index.

**dtype** — The data type of a column (e.g., `int64`, `float64`, `object`, `bool`, `category`).

**GroupBy** — The pandas implementation of the split–apply–combine pattern.

**index** — The labeled axis of a Series or the row labels of a DataFrame. Every value in a Series has an index label.

**.iloc[]** — Integer-location based indexing; always uses integer positions (0-based), like Python list indexing.

**.loc[]** — Label-based indexing; uses index labels (not positions).

**missing value (NaN)** — A placeholder for absent data. `NaN` stands for "Not a Number."

**pivot table** — A summary table that aggregates a numeric column across two categorical dimensions.

**Pearson correlation** — A measure of linear association between two numeric variables, ranging from -1 (perfect negative) to +1 (perfect positive).

**Series** — A one-dimensional labeled array; the fundamental building block of pandas.

**split–apply–combine** — The conceptual pattern behind GroupBy: split data into groups, apply a function to each group, combine the results.

**vectorized operation** — An operation applied element-by-element to an entire Series or column without an explicit loop.

**view** — A reference into an existing DataFrame. Modifying a view may modify the original, which is why pandas raises `SettingWithCopyWarning`.
