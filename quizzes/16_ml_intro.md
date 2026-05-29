# 16 · Intro to Machine Learning

## 16_1: Intro to ML

### What is the fundamental difference between traditional programming and machine learning?
- [ ] Traditional programming learns rules from data; ML executes predefined rules
- [x] ML learns rules from data; traditional programming executes predefined rules
- [ ] ML is faster; traditional programming is more accurate
- [ ] Traditional programming uses Python; ML requires specialized hardware

### In supervised learning, what are 'features' and what is the 'target'?
- [ ] Features are the labels we predict; the target is the input data
- [x] Features are input variables used for prediction; the target is the output we predict
- [ ] Features are training examples; the target is the test set
- [ ] Features are model parameters; the target is the loss function

### How does a classification problem differ from a regression problem?
- [ ] Classification predicts continuous values; regression predicts discrete categories
- [x] Classification predicts discrete categories; regression predicts continuous values
- [ ] Classification requires more data than regression
- [ ] Regression always outperforms classification on structured data

### Why does a KNN classifier require StandardScaler before training?
- [ ] StandardScaler improves model accuracy by removing outliers
- [x] KNN measures distance between points; unscaled features with large ranges dominate the distance
- [ ] KNN cannot process negative numbers without scaling
- [ ] StandardScaler converts categorical features to numeric values

### You add a random noise feature to a KNN model. What happens to the decision boundary?
- [ ] The boundary becomes smoother and more accurate
- [x] The boundary becomes noisier and less accurate because meaningless distance is added
- [ ] The boundary is unaffected because KNN ignores irrelevant features
- [ ] The model automatically detects and removes the noise feature

### knn.predict_proba() returns probabilities for each class. When is this more useful than knn.predict()?
- [ ] When the model needs to run faster than predict()
- [x] When you need confidence levels or want to adjust the decision threshold
- [ ] When the training data has missing values
- [ ] When features are categorical rather than continuous

### A regression model trained on houses between 1,000–3,000 sqft predicts a 10,000 sqft house. Why is this risky?
- [ ] The model will refuse to make predictions outside its training range
- [x] Extrapolating beyond the training range assumes the linear trend continues, which may be wrong
- [ ] Regression models cannot process numbers above 3,000
- [ ] The prediction will be identical to the mean of the training data

### Which of the following is an example of unsupervised learning?
- [ ] Predicting whether an email is spam using labeled spam/not-spam examples
- [x] Grouping customers by purchasing patterns without any predefined categories
- [ ] Training an agent to play chess by rewarding winning moves
- [ ] Predicting tomorrow's temperature from historical weather data

### The 'garbage in, garbage out' principle refers to what?
- [ ] Models should be deleted if they produce incorrect predictions
- [x] Poor quality training data produces unreliable models regardless of algorithm sophistication
- [ ] Code errors in data processing always cause model crashes
- [ ] Outliers must be removed before any model can be trained

## 16_3: NumPy Arrays

### You assign 9.99 to position 0 of an integer NumPy array. The stored value becomes 9. Why?
- [ ] NumPy rounds all floats to the nearest integer
- [x] NumPy truncates the decimal because the array enforces integer dtype
- [ ] 9.99 is converted to its binary representation and rounded up
- [ ] NumPy raises a TypeError for mixed-type assignments

### Why is `mylist + 4` a TypeError but `myarray + 4` works correctly?
- [x] Arrays support element-wise arithmetic; lists interpret + as concatenation only
- [ ] Lists cannot store the number 4 without type conversion
- [ ] NumPy redefines + to mean append for all Python objects
- [ ] Lists require a for-loop; arrays do not use Python operators

### Why should you avoid building an array with repeated np.append() calls inside a loop?
- [ ] np.append() is deprecated in modern NumPy versions
- [x] Each call copies the entire array, making the loop O(n²) overall
- [ ] np.append() modifies the original array and causes index errors
- [ ] Loops cannot call NumPy functions more than once

### You want songs with tempo > 120 AND energy > 0.6. Why must you use & instead of Python's 'and'?
- [x] 'and' is reserved for boolean scalars; & applies element-wise to boolean arrays
- [ ] & is faster than 'and' for large arrays
- [ ] 'and' returns an integer array instead of a boolean array
- [ ] NumPy does not recognize Python keywords inside array expressions

### np.linspace(0, 1, 5) produces [0.0, 0.25, 0.5, 0.75, 1.0]. What does np.arange(0, 1, 0.25) produce?
- [ ] [0.0, 0.25, 0.5, 0.75, 1.0] — five elements including endpoint
- [x] [0.0, 0.25, 0.5, 0.75] — four elements excluding endpoint
- [ ] [0.25, 0.5, 0.75, 1.0] — excludes start but includes endpoint
- [ ] [0.0, 0.25, 0.5, 0.75, 1.0, 1.25] — continues past 1.0 by one step

### arr.reshape(-1, 1) converts a 1D array to 2D. Why does scikit-learn require this shape?
- [x] Scikit-learn uses 2D arrays to represent a feature matrix with rows=samples, cols=features
- [ ] Scikit-learn cannot process arrays with fewer than 2 dimensions
- [ ] 2D arrays store more decimal precision than 1D arrays
- [ ] reshape(-1, 1) normalizes the data to unit variance

### np.random.seed(42) is called before generating training data. What would happen without it?
- [ ] The array would contain only zeros until explicitly populated
- [x] The random data would change each run, making results non-reproducible
- [ ] NumPy would raise an error when generating random numbers
- [ ] The generated distribution would no longer be normal

### An array has shape (100,). You reshape it to (-1, 5). What is the resulting shape?
- [ ] (5, 100) — the -1 is replaced by 100
- [x] (20, 5) — NumPy infers 20 rows to make 100 elements fit
- [ ] (100, 5) — the original size is preserved as rows
- [ ] (-1, 5) — NumPy cannot infer the -1 dimension here

## 16_4: Pandas Series

### series.loc['apple':'cherry'] includes 'cherry' in the result. series.iloc[1:3] excludes position 3. Why?
- [x] .loc uses inclusive slicing for label-based access; .iloc uses exclusive slicing like Python lists
- [ ] .loc and .iloc behave identically; the difference is a bug in this example
- [ ] .iloc includes the endpoint if the index is a string
- [ ] .loc excludes the endpoint when labels are alphabetically ordered

### A Series has two entries with the label 'apple'. What does series.loc['apple'] return?
- [ ] Only the first 'apple' entry by default
- [x] A Series containing both 'apple' entries
- [ ] A KeyError because duplicate labels are invalid
- [ ] The last 'apple' entry since it overwrites the first

### You write series_copy = series. Later, series_copy['apple'] = 999 also changes series. Why?
- [x] Assignment creates an alias pointing to the same object, not an independent copy
- [ ] Pandas Series are immutable, so changes propagate automatically
- [ ] The value 999 overwrites the original Series because it's out of range
- [ ] Labels in Series always share memory across assignments

### series.sort_values() returns a new Series. Why does the notebook warn against inplace=True?
- [ ] inplace=True is deprecated and will be removed in future Pandas versions
- [x] inplace=True can create subtle bugs when mixed with chained operations or views
- [ ] inplace=True is slower than returning a new Series
- [ ] inplace=True resets the index, which destroys the original label information

### You want states with populations between 5 million and 15 million. Which code is correct?
- [ ] series.loc[series > 5e6 and series < 15e6]
- [x] series.loc[(series > 5e6) & (series < 15e6)]
- [ ] series.loc[series > 5e6 | series < 15e6]
- [ ] series[(5e6 < series < 15e6)]

### series.str.split().str[0] extracts the first word from each string. Why is .str[0] needed instead of [0]?
- [x] .str.split() returns a Series of lists; .str[0] extracts element 0 from each list element-wise
- [ ] [0] would return the entire first row of the Series
- [ ] .str[0] is faster because it uses vectorized operations
- [ ] [0] raises a TypeError when applied to string Series

### The cumulative sum of sorted state populations is used to find which states account for 50% of the US population. Why must you sort first?
- [ ] Unsorted data cannot be summed with .cumsum()
- [x] Sorting by descending population lets you find the minimum number of largest states that reach the threshold
- [ ] Boolean masking requires sorted data to function correctly
- [ ] .cumsum() resets to zero if it encounters an unsorted value

## 16_5: Pandas DataFrames

### Why must you use &, |, and ~ instead of Python's 'and', 'or', 'not' with DataFrame boolean masks?
- [ ] 'and'/'or' are keywords reserved for if-statements and cannot appear in expressions
- [x] 'and'/'or' operate on the whole Series as a single truth value; &/| operate element-wise
- [ ] Pandas requires uppercase operators AND, OR, NOT for boolean operations
- [ ] 'and'/'or' produce integers instead of boolean arrays

### df.loc[1973, 'artist'] returns a single value. df.loc[[1973], 'artist'] returns a Series. Why?
- [x] Wrapping the row label in a list signals 'return a DataFrame/Series' rather than a scalar
- [ ] Double brackets always return two copies of the same value
- [ ] This is a Pandas bug that was fixed in newer versions
- [ ] The list syntax applies the selection to the column axis instead

### When copying multiple columns, the notebook uses .values: df[['new1','new2']] = df[['old1','old2']].values. Why is .values needed?
- [x] .values prevents Pandas from trying to align on index labels, avoiding NaN results
- [ ] .values converts the DataFrame to a faster integer format
- [ ] .values is required when the columns have different data types
- [ ] .values drops duplicate index entries before assignment

### Searching for the Sage family with df['Name'].str.contains('Sage') also returns 'Sagesser'. How would you fix this?
- [x] Use str.startswith('Sage ') or str.contains(' Sage') to require a word boundary
- [ ] Use str.find('Sage') == 0 to match only at the start of the string
- [ ] Append a wildcard: str.contains('Sage*')
- [ ] Filter by name length after applying contains

### df.drop(columns=['col1']) returns a new DataFrame. When would using inplace=True be preferable?
- [ ] When the DataFrame is too large to copy into memory
- [x] When you want to permanently modify the DataFrame without creating an extra variable
- [ ] inplace=True is always preferable because it is faster
- [ ] Only when dropping rows, not columns

### df.info() shows Non-Null Count for each column. Why should this be the first thing you check on a new dataset?
- [ ] Non-Null Count shows which columns have the most unique values
- [x] Missing values can cause silent errors in operations and biased results if not handled
- [ ] Columns with missing values must be removed before any analysis
- [ ] Non-Null Count determines the correct dtypes for each column

### You want to add an 'age_group' column: 'child' if Age < 18, 'adult' otherwise. What is the correct approach?
- [ ] df['age_group'] = 'child' if df['Age'] < 18 else 'adult'
- [ ] Use two boolean masks: assign 'child' where mask is True, 'adult' elsewhere
- [ ] df['age_group'] = df['Age'].apply(lambda x: 'child' if x < 18 else 'adult')
- [x] Both the second and third options are correct; the first option raises a ValueError
