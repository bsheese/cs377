# Discussion Questions: 16 — Introduction to ML, NumPy, and Pandas

---

## 16_1: Introduction to Machine Learning

### ML vs. Traditional Programming vs. Statistics

1. The notebook distinguishes ML from traditional programming by saying ML "learns rules from data" instead of being given explicit rules. Give a concrete example where writing explicit rules would be infeasible, and explain why learning from data is the better approach.

2. Statistics is described as inference-focused while ML is described as prediction-focused. What does this mean in practice? Can a model be both statistically sound and predictively powerful?

3. A spam filter built with explicit rules (e.g., "flag all emails with 'FREE MONEY'") would fail against novel spam. How does an ML-based spam filter handle this differently?

4. "Garbage in, garbage out" is stated as a fundamental principle. What does this mean? Give two examples of poor-quality training data and explain how each would corrupt a model.

### Supervised, Unsupervised, and Reinforcement Learning

5. Supervised learning requires labeled training data. What does "labeled" mean, and why is labeling data often expensive or difficult in the real world?

6. Unsupervised learning discovers patterns without labels. If you handed an unsupervised clustering algorithm a dataset of grocery store purchases, what kinds of patterns might it find? What couldn't it tell you?

7. Reinforcement learning learns through reward signals. Why can't reinforcement learning be used for most business prediction problems (e.g., predicting customer churn)?

8. A student argues: "Any classification problem could be done with regression by predicting a number and rounding." What breaks down with this approach? (Hint: think about predicted values outside [0,1] and what "distance" means for categories.)

### Classification vs. Regression

9. Song danceability is predicted as a binary classification (danceable/not). Why is this a simplification, and when might treating it as a regression problem be more appropriate?

10. K-Nearest Neighbors classifies a new point based on its k closest neighbors. Why does the number of neighbors (k) matter? What goes wrong with k=1 vs. k=100?

11. The decision boundary separates predicted classes visually. What happens to the decision boundary when you add a third feature (random noise) to the KNN classifier?

12. A linear regression model predicts house price from square footage. The slope is 0.15 (thousands per sqft). A 500 sqft increase is predicted to add $75,000. If you fit this model on houses between 1,000 and 3,000 sqft, should you trust its prediction for a 10,000 sqft mansion? Why or why not?

### Tools and Workflow

13. `StandardScaler` is applied to features before KNN. Why does KNN specifically require feature scaling, while a decision tree might not?

14. `knn.predict_proba()` returns probabilities for each class. How does this differ from `knn.predict()`? When would probabilities be more useful than hard class predictions?

15. The notebook notes that data cleaning and preparation is "a major portion of this course." Does this surprise you? Why might data preparation take more time than model fitting?

---

## 16_3: Introduction to NumPy Arrays

### Arrays vs. Lists

1. A Python list can store `[1, "hello", 3.14, True]` in a single list. A NumPy array cannot. Why does this type homogeneity make arrays better for numerical computation?

2. If you assign `9.99` to an integer array, it gets truncated to `9`. Why doesn't NumPy raise an error? Under what circumstances could this silent truncation cause a hard-to-find bug?

3. `np.append()` is described as creating a new array each time. Why is this O(n)? What is the correct pattern when you need to build an array incrementally?

4. The notebook shows that `mylist + 4` raises a `TypeError` but `myarray + 4` works. Why does this same `+` operator behave differently for lists vs. arrays?

### Vectorization and Performance

5. "Vectorized operations run in compiled C, not Python loops." Why is C code faster than Python code for this kind of computation? What's being skipped?

6. A student writes a loop: `for i in range(len(arr)): arr[i] = arr[i] * 2`. This works but is slow. What is the vectorized equivalent, and why is it faster?

7. Give a real-world scenario where the performance difference between vectorized operations and Python loops would actually matter.

### Boolean Masking

8. `tempos > 120` returns a boolean array instead of a single True/False. What makes this useful that a simple `if` statement cannot provide?

9. You want songs where tempo > 120 AND energy > 0.6. Why must you use `(tempos > 120) & (energies > 0.6)` with `&` instead of Python's `and`?

10. `mask.sum()` counts True values because True=1 and False=0. This works in Python, but is it good practice? What would make the intent clearer?

### Broadcasting and Reshaping

11. Broadcasting allows `(3,1)` and `(3,)` arrays to be added together, producing a `(3,3)` result. Draw out what this means for a specific example with numbers. When could broadcasting produce unexpected results?

12. `.reshape(-1, 1)` converts a 1D array to a 2D column. Why does scikit-learn's `LinearRegression` require 2D input? What does the `-1` mean?

13. If an array has shape `(100,)` and you reshape it to `(-1, 5)`, what is the resulting shape? What would happen if the array had 103 elements instead?

### Random Numbers and Statistics

14. `np.random.seed(42)` is called before generating data. What would happen if you didn't set the seed? Why is reproducibility important in machine learning?

15. Why is pre-allocating a `np.zeros()` array and filling it in a loop faster than building an array with repeated `np.append()` calls?

---

## 16_4: Pandas Series

### Series vs. Arrays vs. Lists

1. Series is described as "a NumPy array with a labeled index." What new capabilities does the label-based index provide that positional indexing alone cannot?

2. A NumPy array has no built-in mechanism for missing values (NaN). Why is NaN handling particularly important in real-world data?

3. When you do `alias = series` and `copy = series.copy()`, only the alias reflects changes to the original. Why? When would accidentally using an alias instead of a copy cause a subtle bug?

### Indexing: `.loc[]` vs. `.iloc[]`

4. `series.loc['apple':'cherry']` includes 'cherry' in the result. `series.iloc[1:3]` excludes position 3. Why are these two slicing behaviors different? Which is more consistent with Python conventions?

5. If a Series has repeated index labels (e.g., two entries both labeled 'apple'), what does `series.loc['apple']` return? Is this a feature or a potential bug in your code?

6. In what situation would `.iloc[]` be the only correct choice, even if the Series has meaningful string labels?

### Boolean Masking

7. You want state populations between 5 million and 15 million. Write the masking logic and explain why each part is necessary. What would happen if you used Python's `and` instead of `&`?

8. Masking returns a filtered Series, not a list. What advantage does returning a Series have over returning a Python list of values?

### String Operations

9. `.str.contains()` works on a whole Series at once. Why is this better than writing `for title in series: if substring in title: ...`?

10. `.str.split().str[0]` extracts the first element from each split result. What type does `.str.split()` return, and why must you chain `.str[0]` rather than `[0]`?

11. String methods like `.str.upper()` return new Series rather than modifying in place. Why is this the safer behavior?

### Sorting

12. `.sort_values()` returns a new Series. Why does the notebook warn against `inplace=True`? What specific bug can it cause with chained operations?

13. In the state population example, the notebook asks you to find states whose cumulative population exceeds 50% of the US total. Why does sorting first matter here?

---

## 16_5: Pandas DataFrames

### Structure and Indexing

1. A DataFrame is described as "a collection of Series sharing a common index." What does this mean, and what constraint does it place on adding a new column?

2. `.set_index(column_name)` moves a column to the index. When is this useful, and when might it cause problems?

3. `.loc[row_selector, column_selector]` has two dimensions; `.iloc[]` also takes two. Why is specifying both dimensions important when selecting from DataFrames?

4. `df.loc[1973, 'artist']` returns a single value, while `df.loc[[1973], 'artist']` returns a Series. Why does wrapping the row label in a list change the return type?

### Boolean Masking and Filtering

5. Why must you use `&`, `|`, and `~` instead of `and`, `or`, and `not` when combining boolean conditions on DataFrames?

6. In the Titanic dataset, finding a "Sagesser" passenger when searching for the Sage family is called a "false positive." What type of filter would prevent this false positive?

7. You want passengers under 10 OR over 60. Write the boolean mask and explain why this requires `|` rather than `&`.

8. The `~` operator negates a boolean Series. Give an example of a filtering task where negation is more natural than writing the affirmative condition.

### Modifying DataFrames

9. Creating a new column with `df.loc[:, 'new_col'] = value` broadcasts a scalar across all rows. When might broadcasting cause unexpected behavior?

10. `df.drop(columns=['col1'])` returns a new DataFrame unless `inplace=True`. When is the non-destructive default helpful, and when might you prefer `inplace=True`?

11. The notebook uses `.values` when copying multiple columns: `df[['new1', 'new2']] = df[['old1', 'old2']].values`. Why is `.values` needed here?

### Real Data Analysis

12. In the Grammy Winners dataset, New York leads in both absolute and per-capita Grammy winners. Does this surprise you? What confounding variables might explain this result?

13. In the Titanic dataset, combining `(df['Age'] < 10) | (df['Age'] > 60)` to find the very young and very old — what real-world questions could you answer with this filter?

14. The `.info()` method shows non-null counts. Before doing any analysis, why should this always be one of the first things you check?

15. You find the Palsson family in the Titanic data (all perished). Does this constitute a meaningful finding or just an anecdote? What would you need to show statistically to make a valid claim about family survival rates?

