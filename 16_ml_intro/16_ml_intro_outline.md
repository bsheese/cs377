# 16 ML Intro — Topic Outline

This document provides a complete outline of all topics covered across the notebooks in the 16 ML Intro series: 16_1, 16_3, 16_4, 16_5, and the 16_9 exercises. (There is no 16_2 — notebook basics are covered inside 16_1.)

---

## 16_1: Introduction to Machine Learning

**Topics:** Conceptual overview of ML, its relationship to traditional CS and statistics, types of ML, and tools for the course.

### What is Machine Learning?
- ML as a subfield of artificial intelligence
- Contrast with traditional software engineering (deterministic vs. probabilistic)
- Contrast with statistics (inference vs. prediction)
- ML prioritizes predictive accuracy and computational efficiency over interpretability

### Examples of ML Problems
- Spam filtering: rule-based vs. statistical vs. ML approaches
- Computer vision: why traditional programming can't handle images
- Recommendation systems: navigating massive user behavior matrices

### The Central Role of Data
- ML as empirical: data is the raw material
- "Garbage in, garbage out" — data quality determines model quality
- Bias, noise, and representativeness of training data
- Why most course time goes into data preparation, not algorithm coding

### Three Categories of Machine Learning
- **Supervised Learning:** labeled data, learning input→output mapping
  - Classification: discrete categories (danceable/not, spam/not, cancerous/not)
  - Regression: continuous values (prices, GPA, crop yield)
- **Unsupervised Learning:** unlabeled data, discovering hidden patterns
  - Clustering (customer segments), dimensionality reduction
- **Reinforcement Learning:** agent interacts with environment, maximizes reward
  - Games (Chess, Go), robotics, self-driving cars
  - Semi-supervised and self-supervised as hybrid approaches

### Supervised Learning Deep Dive
- Features ($x$), target ($y$), training data ($x_i, y_i$)
- Learning a mapping function $f$ such that $f(x) \approx y$
- Training/fitting the model, then predicting on new data ($\hat{y}$)
- Formal definitions with DJ/danceability example
- Tempo and energy as features, danceable/not as target

### Jupyter Notebook Basics
- Mix of text (Markdown) and code cells
- Execution order: cell numbers track run sequence; cells share state
- Run cells top-to-bottom; "Run All" to execute everything in order
- Restart the kernel if the notebook seems broken

### Supervised Learning Worked Examples
- **KNN Classifier** — song danceability: features (tempo, energy) → danceable/not
  - Decision boundary: the region where the model changes its prediction
  - Prediction confidence / probability (`.predict_proba()`)
  - Feature scaling with `StandardScaler` before fitting KNN
- **Linear Regression** — house price from square footage
  - Model fit, prediction (`\hat{y}`), residual standard deviation as uncertainty
- **K-Means Clustering** — the same songs without labels
  - Discovering groups from features alone vs. naming the groups
  - Side-by-side contrast with the supervised (labeled) view

### Tools for the Course
- Jupyter Notebooks / Google Colab
- Python libraries: Pandas, Matplotlib, NumPy, scikit-learn (Seaborn appears in the 16_9 exercises)

---

## 16_3: Introducing Numpy Arrays

**Topics:** List review, array creation, array operations, broadcasting, reshaping, random number generation, statistical methods, multi-dimensional arrays.

### Python List Review
- Creating lists, appending, extending
- Indexing, slicing, stepping
- Iteration with for loops
- Copying vs. aliasing

### From Lists to Numpy Arrays
- Comparison of lists, arrays, and Series: features, performance, use cases
- `np.asarray()` to convert lists
- Arrays enforce a single data type (unlike lists)
- Indexing, slicing, and boolean masking work similarly to lists
- Type coercion: assigning a float to a string array converts it to string

### Array Arithmetic and Broadcasting
- Adding, subtracting, multiplying, dividing — applied element-wise
- Lists cannot do this: `list + 4` produces an error
- **Broadcasting:** applying an operation between arrays of different shapes (e.g., scalar to array, row to 2D matrix); NumPy stretches the smaller shape to match

### Creating Arrays
- `np.array()` from Python lists
- `np.arange()` with start, stop, step
- `np.zeros()`, `np.ones()`, `np.full()`
- `np.linspace()` for regularly spaced intervals

### Random Number Generation
- `np.random.seed()` for reproducibility
- `np.random.randint()`, `np.random.random()`, `np.random.normal()`
- `np.random.shuffle()`, `np.random.choice()`

### Reshaping Arrays
- `.reshape()` to change array dimensions
- `.shape` attribute for inspecting dimensions
- Arrays from nested lists (multi-dimensional)

### Statistical Methods
- Array methods: `.min()`, `.max()`, `.mean()`, `.std()`, `.sum()`
- Z-score normalization: subtracting mean and dividing by standard deviation

### Boolean Masking on Arrays
- Comparison operators produce boolean arrays
- Using a boolean array directly to filter: `array[condition]`
- Computing statistics on filtered subsets

### Best Practices
- Don't create empty arrays and append — pre-allocate with `np.zeros()`

---

## 16_4: Pandas Series

**Topics:** Comparing lists/arrays/Series, Series creation, copying vs. aliasing, examining data, selecting data, updating, sorting, operations, string methods.

### Lists, Arrays, and Series — A Comparison
- Side-by-side comparison of features, performance, and use cases
- Arrays: homogeneous, fast math, no labels
- Series: labeled index, heterogeneous-friendly, built-in NaN handling, alignment by label

### Installing and Importing Pandas
- Standard import: `import pandas as pd`

### Creating Series
- Empty series with `pd.Series(dtype=...)`
- From lists: with or without specifying dtype
- With a custom index
- dtype inference vs. explicit specification

### Copying Series vs. Aliasing
- Alias vs. copy (`series.copy()`)
- Checking object identity with `is`
- Selection returns a new object; the original is unchanged unless reassigned

### Examining Series
- Large series display truncation
- `.head()` and `.tail()` with optional argument

### Selecting Data Using the Index
- `.loc[]`: label-based indexing and slicing
  - Includes stop value in slices
  - Returns all values with a given label (duplicates allowed)
- `.iloc[]`: position-based indexing
  - Does not include stop value
  - Supports negative indexing
- Index and values attributes (`.index`, `.values`)

### Selection by Condition (Boolean Masks)
- Comparison operators produce boolean series
- Using a boolean mask with `.loc[]`
- Combining masks with `&` (and) and `|` (or)
- Single-step vs. multi-step masking

### Updating Values and Labels
- Updating by label (`.loc`) or position (`.iloc`)
- Updating multiple values with slice assignment
- Index reassignment (not mutable in place)

### Basic Math Operations
- Element-wise arithmetic: `series + 5`, `series - 5`, `series * 5`, `series / 5`
- Operations return a new Series; the original is unchanged
- Reassign to persist changes: `series = series + 5`
- `.cumsum()` — cumulative sum across the series

### Basic String Operations
- String concatenation across a series
- `.str` accessor for vectorized string methods
  - `.lower()`, `.upper()`, `.len()`
  - `.startswith()`, `.endswith()`, `.contains()`
  - `.find()`, `.replace()`, `.split()`
- Avoid for-loops when vectorized operations exist

### Sorting Series
- `.sort_index()` — sort by index labels
- `.sort_values()` — sort by values
- Sorting returns a new Series; reassign to persist
- `ascending=False` for descending order
- In-place sorting exists but not recommended for this course

### Examples
- Fruit names and weights (index operations)
- Best-selling books (dtype, string operations)
- US state populations: computing percentages, finding states accounting for >50% of total with `.cumsum()`

---

## 16_5: Pandas Dataframes

**Topics:** Creating, examining, selecting, modifying, boolean operations, grouping.

### Creating Dataframes
- Empty Dataframe: `pd.DataFrame()`
- From CSV: `pd.read_csv()`
- Albums sales dataset from Wikipedia

### Copying Dataframes vs. Creating Views
- Alias vs. copy (`df.copy()`)
- Object identity checking with `is`

### Examining Dataframes
- `.info()` — column types, non-null counts, memory usage
- `.head()` and `.tail()` — first/last rows
- `.index` and `.columns` attributes

### Setting and Resetting the Index
- `.set_index()` — promote a column to the index
- `.reset_index()` — move index back to a column
- `.sort_index()` — sort by index values

### Updating Index Labels and Column Labels
- Assigning a new list or series to `.index`
- `.columns.str.capitalize()`, `.columns.str.lower()`

### Selecting Data with `.loc`
- Row and column selection: `df.loc[row, column]`
- Slicing rows and columns by label
- Lists of labels for rows and columns
- Includes stop value in slices

### Selecting Data with `.iloc`
- Position-based selection
- Does not include stop value
- Supports negative indexing

### Boolean Masks for Selection
- Creating masks with comparison operators
- Single masks and combined masks (`&`, `|`)
- `~` for negation

### Modifying Rows and Columns
- Creating a new row: assign to a non-existent index label
- Creating a new column: assign to a non-existent column label
- Copying a series within a dataframe
- Deleting rows and columns with `.drop()`

### Updating Values
- By label with `.loc[]`
- By condition with masks
- `.isin()` for membership tests

### Basic Math Operations
- Element-wise arithmetic on selected columns
- Updating all values without a loop

### Basic String Operations
- `.str.lower()` on dataframe columns

### Examples
- Grammy winners by state
- Top 10 states by total grammy winners
- Most populous states with no grammy winners
- Winners per million population

### Boolean Combinations and Negations
- Combining booleans with `&` (and) and `|` (or)
- Negating booleans with `~`
- Order of operations: parentheses around each condition
- `.str.startswith()` and `.str.contains()` for string-based masks

### Titanic Dataset Examples
- Loading Titanic data
- Finding passengers by age, sex, fare conditions
- Finding families (Palsson, Sage) with string matching
- Excluding false matches with negation
- `regex=False` to treat `.str.contains()` arguments as plain text

---

## 16_9: EDA Workout — Auto MPG Exercises

**Topics:** Integrative fill-in-the-blank exercises applying the 16_3–16_5 skills to the seaborn Auto MPG dataset (398 cars, 1970–1982).

### Task 1 — NumPy
- `.to_numpy()`, summary statistics (`.mean()`, `.std()`, `.min()`, `.max()`)
- Boolean masking and statistics on masked subsets

### Task 2 — Pandas Series
- `.loc[]` label slicing (inclusive stop)
- `.sort_values(ascending=False)` with `.head()`
- Boolean masks on a Series

### Task 3 — Pandas DataFrame
- Creating a derived column from two existing columns
- Combined boolean masks with `&`
- Comparing group means with `.loc[]` selection and `.mean()`

---
