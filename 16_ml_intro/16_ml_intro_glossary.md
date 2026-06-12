# 16 ML Intro — Glossary

This document defines the technical and conceptual terms used across the notebooks in the 16 ML Intro series, plus a few forward-looking terms (marked as such) that become important in later units.

---

## A

### Accuracy
The fraction of predictions a classifier gets right. In scikit-learn, `model.score(X_test, y_test)` reports accuracy for classifiers. Only meaningful when measured on data the model did not train on; see **Test Set**.

### Array (Numpy Array)
A grid of values, all of the same data type, indexed by non-negative integers. Unlike Python lists, arrays support element-wise arithmetic operations. Created with `np.array()`, `np.arange()`, `np.zeros()`, `np.ones()`, `np.full()`, etc.

### Artificial Intelligence
The broader field of computer science concerned with creating systems that can perform tasks that typically require human intelligence. Machine learning is a subfield of AI.

---

## B

### Boolean Mask
An array or Series of `True`/`False` values used to select a subset of data. Created by applying a comparison operator (e.g., `>`, `==`, `<=`) to an array or Series. With NumPy arrays, the mask is passed directly inside `array[mask]`. With Pandas, it is passed inside `.loc[]` to filter rows. Multiple masks can be combined with `&` (and), `|` (or), and `~` (not).

### Broadcasting
NumPy's mechanism for applying operations between arrays of different shapes. When an operation involves a scalar and an array, or two arrays whose shapes are compatible, NumPy automatically "stretches" the smaller shape to match the larger one without copying data. Example: adding a single number to every element of a 2D matrix.

---

## C

### Classification
A type of supervised learning where the target variable is a discrete category. Examples: spam vs. not spam, cancerous vs. not cancerous, danceable vs. not danceable. Contrast with **Regression**.

### Clustering
An unsupervised learning technique where data points are grouped into clusters based on similarity, without using pre-existing labels. Example: grouping customers into marketing segments based on purchasing history.

### Code Cell
A block in a Jupyter notebook that contains executable code. When run, the code is executed by a kernel (e.g., Python) and output is displayed below the cell.

### Colab (Google Colaboratory)
Google's hosted Jupyter notebook service. Allows notebooks to be run in the cloud without local installation. Includes free GPU/TPU access and easy sharing.

### Copy vs. Alias
A copy is an entirely new object with the same values; modifying it does not affect the original. An alias is an additional name referencing the same object; modifications through the alias affect the original. In Pandas, use `.copy()` to create a true copy; assignment (`=`) creates an alias.

---

## D

### Data Leakage
*(Forward-looking: introduced in depth in unit 17.)* When information from outside the training set influences the model during training or feature selection, leading to overly optimistic performance estimates. Pipelines in scikit-learn prevent this by fitting transformations (e.g., scaling) only on training folds.

### DataFrame
A two-dimensional Pandas data structure with labeled rows and columns. Acts like a spreadsheet. Can be thought of as a collection of Series that share the same index.

### Decision Boundary
The region in feature space where a classifier changes its prediction from one class to another. For a KNN model trained on two features (e.g., tempo and energy), the decision boundary is the line (or curve) that separates the "danceable" region from the "not danceable" region.

### dtype (Data Type)
Specifies the type of data stored in a Series or array. Common dtypes: `int64`, `float64`, `string`, `object` (mixed types), `bool`. Explicit dtype specification can be done with the `dtype=` parameter.

---

## F

### Feature
An input variable used by a model to make predictions. Represented mathematically as a vector $x$. In the DJ example, tempo and energy are features. Also called a predictor or independent variable.

### Feature Scaling
Transforming features so they share a common numerical range, typically mean 0 and standard deviation 1 (standardization). Necessary for distance-based algorithms like **K-Nearest Neighbors (KNN)**, which treat all features as equally weighted. In scikit-learn, `StandardScaler` computes $(x - \mu) / \sigma$ for each feature.

### Fitting
The process of training a model on data. During fitting, the model learns the mapping from features to target variable. Also called **Training**.

---

## G

### "Garbage In, Garbage Out" (GIGO)
A principle stating that the quality of a model's output is directly limited by the quality of its input data. If training data is noisy, biased, or unrepresentative, the resulting model will reflect those flaws.

---

## H

### Head / Tail
Pandas methods — `.head(n)` returns the first `n` rows (default 5), `.tail(n)` returns the last `n` rows (default 5). Used to inspect data quickly without printing the entire structure.

---

## I

### Index
A labeled axis for identifying rows in a Series or DataFrame. Can be integers, strings, or any hashable type. Unlike Python lists, the index can have non-sequential or duplicated values. Access elements by index label with `.loc[]`.

### `.iloc[]`
Position-based (implicit) indexing for Series and DataFrames. Selects data by integer position, like a Python list. Does not include the stop value in slices. Supports negative indexing.

### `import pandas as pd`
The standard convention for importing the Pandas library. Must be executed before any Pandas code can run.

---

## J

### Jupyter Notebook
An interactive computing environment that integrates text, code, and code output in a single document. Formerly known as IPython Notebook. Supports over 40 programming languages.

---

## K

### K-Means
An unsupervised clustering algorithm that groups data points into $k$ clusters by repeatedly assigning each point to the nearest cluster center and re-computing the centers. Used in 16_1 to show that the song groups can be discovered from features alone — without the labels supervised learning requires. The discovered clusters are numbered, not named; interpreting them is up to us.

### K-Nearest Neighbors (KNN)
A supervised learning algorithm that classifies a new data point by looking at the $k$ closest training examples in feature space and taking a majority vote. Used in 16_1 to predict song danceability from tempo and energy. Sensitive to the scale of features, so **Feature Scaling** is applied before fitting. Predictions can include a probability estimate via `.predict_proba()`.

---

## L

### `.loc[]`
Label-based (explicit) indexing for Series and DataFrames. Selects data by index label. Unlike Python slicing, includes the stop value in slices. For DataFrames, takes two arguments: row selector and column selector separated by a comma.

---

## M

### Machine Learning
A subfield of AI in which algorithms learn patterns from data to make predictions or decisions without being explicitly programmed for every scenario. Prioritizes predictive accuracy, often trading off interpretability.

### Markdown
A lightweight markup language used to format text cells in Jupyter notebooks. Supports headings, lists, links, bold, italic, code blocks, LaTeX math, and more.

### Multi-Dimensional Array
A Numpy array with more than one dimension (e.g., 2D matrix, 3D tensor). Created by passing a nested list to `np.array()` or using the `shape` parameter with `np.zeros()` or `np.ones()`. Shape is inspected with the `.shape` attribute.

---

## N

### Numpy
A Python library for numerical computing. Provides the `ndarray` (n-dimensional array) data structure, mathematical functions, random number generation, and linear algebra routines. Standard import: `import numpy as np`.

---

## O

### Overfitting
When a model learns the noise and specific patterns of the training data rather than the underlying generalizable relationships. Results in high training performance but poor performance on new, unseen data. Contrast with **Underfitting**.

---

## P

### Pandas
A Python library for data manipulation and analysis. Provides the Series (1D) and DataFrame (2D) data structures, along with tools for reading, cleaning, transforming, and analyzing data. Standard import: `import pandas as pd`.

### Pipeline (scikit-learn)
*(Forward-looking: introduced in unit 17 and used throughout the course.)* A tool that chains together multiple processing steps (e.g., scaling, feature selection, modeling) into a single estimator. Ensures each step is applied correctly within cross-validation, preventing data leakage.

---

## R

### Regression
A type of supervised learning where the target variable is a continuous numerical value. Examples: predicting home prices, GPA, temperature. Contrast with **Classification**.

### Reinforcement Learning
A type of machine learning where an agent learns by interacting with an environment, receiving rewards or penalties for its actions. The agent's goal is to learn a policy that maximizes cumulative reward over time. Used in game-playing AI, robotics, and self-driving cars.

---

## S

### Series
A one-dimensional Pandas data structure with labeled index. Similar to a Python list but with additional functionality: custom index, vectorized operations, alignment by label. A DataFrame can be thought of as a collection of Series.

### Scikit-learn
A Python library for machine learning. Provides tools for model building, evaluation, feature selection, preprocessing, and more. Works with NumPy arrays and Pandas DataFrames.

### `.str` Accessor
A Pandas accessor that provides vectorized string methods for Series of strings. Allows applying Python string methods (`.lower()`, `.upper()`, `.split()`, `.replace()`, etc.) to every element in a Series without writing a loop.

### Supervised Learning
A type of machine learning where the model is trained on labeled data (input-output pairs). The goal is to learn a mapping from inputs to outputs to predict labels for new, unseen data. Includes **Classification** (discrete categories) and **Regression** (continuous values).

---

## T

### Target Variable
The output variable that a supervised learning model aims to predict. Represented mathematically as $y$. Also called the dependent variable, label, or response variable.

### Test Set
A portion of the labeled data deliberately held back and *not* used during training, so the model can be scored on examples it has never seen. Measuring performance on the training data itself is misleading — a model can memorize its training examples yet fail on new data. Contrast with the **Training** data. See **Train/Test Split**.

### Train/Test Split
Dividing labeled data into a training set (used to fit the model) and a **Test Set** (held back to evaluate it on unseen examples). scikit-learn's `train_test_split` performs the division. Used in nearly every modeling notebook in the course; it guards against overstating a model's quality by testing it on data it was trained on. Related: **Overfitting**, **Data Leakage**.

### Training
The process of exposing an algorithm to data so that it learns patterns and relationships. In supervised learning, training uses labeled data to find the mapping from features to target. Also called **Fitting**.

---

## U

### Underfitting
*(Forward-looking: explored in depth in later units.)* When a model is too simple to capture the underlying patterns in the data. Results in poor performance on both training data and new data. Contrast with **Overfitting**.

### Unsupervised Learning
A type of machine learning where the model is given unlabeled data and must find hidden patterns or structures on its own. Common tasks include **Clustering** (grouping similar items) and dimensionality reduction.

---

## V

### `.value_counts()`
A Pandas method that tallies how many times each unique value appears in a column (or Series), returning the counts sorted from most to least common. The standard tool for inspecting a categorical column — and, in classification, for checking **class balance** (how many examples exist of each category the model must predict).

### Vectorization
Performing an operation on an entire array or Series at once (e.g., `myarray + 4`, `series.str.lower()`) instead of looping element-by-element in Python. Vectorized operations run in compiled C code under the hood, making them dramatically faster than the equivalent Python loop — the main reason to prefer NumPy/Pandas operations over `for` loops.

### View (informal usage)
In this module, "viewing" data means displaying the result of a selection (e.g., `.head()`, `.tail()`, boolean masking) without changing the original object. These operations return **new objects** — the original Series or DataFrame is left unchanged, and the result is discarded unless assigned to a variable. (Pandas also has a stricter internal notion of memory views vs. copies, which we will meet in later units.)

---

## Y

### y-hat ($\hat{y}$)
The predicted value of the target variable output by a trained model. Distinguished from the actual true value $y$. Pronounced "y-hat."
