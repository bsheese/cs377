# 16 ML Intro — Topic Outline

This document provides a complete outline of all topics covered across the five notebooks in the 16 ML Intro series.

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

### Tools for the Course
- Jupyter Notebooks / Google Colab
- Python libraries: Pandas, Matplotlib, Seaborn, NumPy, scikit-learn, statsmodels

---

## 16_2: Introduction to Jupyter Notebooks

**Topics:** How to use notebooks, code cells, text cells, execution order, and troubleshooting.

### What is a Notebook?
- Mix of text and code cells
- Text cells use Markdown formatting
- Code cells run Python and display output

### Code Blocks
- Writing and executing code
- Ctrl+Enter and Shift+Enter shortcuts
- Execution order: cell numbers track run sequence
- Cells share state: order of execution matters

### Imports and Dependencies
- Standard imports: `pandas as pd`, `numpy as np`, `seaborn as sns`, `matplotlib.pyplot as plt`
- Earlier cells must be executed before later cells that depend on them

### Making New Cells
- Hover to create code or text cells
- Experiment freely — nothing can be broken permanently

### Text Cells and Markdown
- Formatting options available in Colab toolbar
- External Markdown guide reference

### Troubleshooting
- Run cells sequentially from the top
- "Run All" to execute everything in order
- Restart the virtual machine if the notebook seems broken

### Magic Commands
- `%` prefix for command-line access
- Not required for this course

---

## 16_3: Introducing Numpy Arrays

**Topics:** List review, array creation, array operations, statistical methods, multi-dimensional arrays.

### Python List Review
- Creating lists, appending, extending
- Indexing, slicing, stepping
- Iteration with for loops
- Copying vs. aliasing

### From Lists to Numpy Arrays
- `np.asarray()` to convert lists
- Arrays enforce a single data type (unlike lists)
- Indexing, slicing, and appending work similarly
- Type coercion: assigning a float to a string array converts it to string

### Array Arithmetic
- Adding, subtracting, multiplying, dividing — applied element-wise
- Lists cannot do this: `list + 4` produces an error
- Contrast with list multiplication (repetition)

### Creating Arrays
- `np.array()` from Python lists
- `np.arange()` with start, stop, step
- `np.zeros()`, `np.ones()`, `np.empty()`
- `np.linspace()` for regularly spaced intervals

### Statistical Methods
- Array methods: `.min()`, `.max()`, `.mean()`, `.std()`, `.sum()`

### Multi-Dimensional Arrays
- Creating with `shape` parameter
- `.shape` attribute for inspecting dimensions
- Arrays from nested lists

### Best Practices
- Don't create empty arrays and append — pre-allocate with `np.zeros()`
- `np.empty()` returns uninitialized memory (garbage values are expected)

---

## 16_4: Pandas Series

**Topics:** Series creation, copying vs. aliasing, examining data, selecting data, updating, sorting, operations, string methods.

### Installing and Importing Pandas
- Standard import: `import pandas as pd`

### Creating Series
- Empty series with `pd.Series(dtype=...)`
- From lists: with or without specifying dtype
- With a custom index
- dtype inference vs. explicit specification

### Copying Series vs. Creating Views
- Alias vs. copy (`series.copy()`)
- Checking object identity with `is`
- Views: modifying how we look at data without copying

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
- Combining masks with `|` (or) operator
- Single-step vs. multi-step masking

### Updating Values and Labels
- Updating by label (`.loc`) or position (`.iloc`)
- Updating multiple values with slice assignment
- Index reassignment (not mutable in place)

### Basic Math Operations
- Element-wise arithmetic: `series + 5`, `series - 5`, `series * 5`, `series / 5`
- Operations create views, not modifications
- Reassign to persist changes: `series = series + 5`

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
- Sorting creates a view; reassign to persist
- `ascending=False` for descending order
- In-place sorting exists but not recommended for this course

### Examples (US State Populations)
- Creating a state population series
- Computing percentages of total
- Iterating to find states accounting for >50% of population

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

---

## Cross-Cutting Themes

| Theme | Notebooks |
|---|---|
| **ML vs. traditional CS vs. statistics** | 16_1 |
| **Data quality and preparation** | 16_1, 16_5 |
| **Supervised learning (classification/regression)** | 16_1 |
| **Pandas fundamentals** | 16_4, 16_5 |
| **Index-based selection (.loc vs. .iloc)** | 16_4, 16_5 |
| **Boolean masking** | 16_4, 16_5 |
| **Vectorized operations** | 16_3, 16_4, 16_5 |
| **Copying vs. aliasing** | 16_3, 16_4, 16_5 |
