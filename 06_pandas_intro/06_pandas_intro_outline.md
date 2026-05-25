# 06 — Pandas Intro: Module Outline

## Audience
Undergrad CS and data science majors with Python experience; no prior pandas exposure.

## Dataset
Titanic passenger data (887 rows, 8 columns: survived, pclass, name, sex, age, sibsp, parch, fare).
Fetched at runtime from the course GitHub data repository.

## Notebooks

| Notebook | Topic | Key Concepts |
|---|---|---|
| 06.1 | Series | Index, label vs position access, vectorized ops, boolean indexing, `.str` |
| 06.2 | DataFrame Basics | Loading CSV, inspection, column selection, adding/dropping columns |
| 06.3 | Selecting & Filtering | `.loc[]`, `.iloc[]`, boolean masks, `.isin()`, `~`, `.query()`, copy vs view |
| 06.4 | Data Cleaning | Missing values, `.dropna()`, `.fillna()`, duplicates, rename, type conversion, `.str` |
| 06.5 | GroupBy | Split–apply–combine, `.groupby()`, `.agg()`, multi-key grouping, reset index |
| 06.6 | Pivot Tables & Descriptive | `pd.crosstab()`, `pd.pivot_table()`, `.corr()`, `.nlargest()`, `.nsmallest()` |

## What Is Intentionally Excluded
- `.merge()` and `.concat()` (follow-on topic)
- `melt()` and advanced reshaping
- Time-series indexing (`DatetimeIndex`)
- Visualization (introduced in later units)

## Learning Sequence
Series (06.1) → DataFrame (06.2) → Filtering (06.3) → Cleaning (06.4) → GroupBy (06.5) → Pivot (06.6)
