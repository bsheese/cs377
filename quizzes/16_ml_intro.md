# 16 · Intro to Machine Learning

## 16_1: Intro to ML

### What is the fundamental difference between traditional programming and machine learning?
- [ ] Traditional programming learns rules from data; ML executes predefined rules
- [x] ML learns rules from data; traditional programming executes predefined rules
- [ ] ML produces faster output; traditional programming produces more accurate predictions
- [ ] Traditional programming requires Python scripting; ML requires specialized GPU hardware

### In supervised learning, what are 'features' and what is the 'target'?
- [ ] Features are the labels we predict; the target is the input data
- [x] Features are input variables; the target is the output variable to predict
- [ ] Features represent individual training examples; the target is the withheld test set
- [ ] Features are the model's learned parameters; the target is the loss function

### How does a classification problem differ from a regression problem?
- [ ] Classification predicts continuous values; regression predicts discrete categories
- [x] Classification predicts discrete categories; regression predicts continuous values
- [ ] Classification requires more labeled training data than regression problems do
- [ ] Regression outperforms classification on any type of structured tabular data

### Why does a KNN classifier require StandardScaler before training?
- [ ] StandardScaler boosts accuracy by detecting and removing outliers from training data
- [x] KNN uses distances; features with large ranges will dominate without scaling
- [ ] KNN distance calculations break when feature values include negative numbers
- [ ] StandardScaler converts categorical string features into numeric codes before fitting

### You add a random noise feature to a KNN model. What happens to the decision boundary?
- [ ] The boundary becomes cleaner and more accurate due to the added variation
- [x] The boundary degrades because irrelevant noise inflates the distance metric
- [ ] The boundary is unaffected because KNN ignores irrelevant features
- [ ] The model automatically detects and removes the noise feature

### knn.predict_proba() returns probabilities for each class. When is this more useful than knn.predict()?
- [ ] When the model needs to run faster, skipping the final classification step
- [x] When you need confidence scores or a tunable decision threshold
- [ ] When the training data contains missing values that the model must interpolate
- [ ] When the input features are categorical variables rather than continuous numeric values

### A regression model trained on houses between 1,000–3,000 sqft predicts a 10,000 sqft house. Why is this risky?
- [ ] The model will refuse to make predictions outside its training range
- [x] Extrapolating outside the training range assumes the linear trend holds
- [ ] Regression models cannot process input values that exceed the training set maximum
- [ ] The prediction will be identical to the mean of the training data

### Which of the following is an example of unsupervised learning?
- [ ] Predicting whether an email is spam using labeled spam/not-spam examples
- [x] Grouping customers by purchasing patterns without any predefined categories
- [ ] Training an agent to play chess by rewarding winning moves
- [ ] Predicting tomorrow's temperature from labeled historical weather station data

### The 'garbage in, garbage out' principle refers to what?
- [ ] Models should be deleted if they produce incorrect predictions
- [x] Poor-quality training data produces unreliable models no matter the algorithm
- [ ] Bugs in data processing code cause model training to crash
- [ ] Outliers must be removed before any model can be trained

## 16_3: NumPy Arrays

### You assign 9.99 to position 0 of an integer NumPy array. The stored value becomes 9. Why?
- [ ] NumPy rounds all float values down to the nearest integer in the array
- [x] NumPy truncates the decimal because the array enforces integer dtype
- [ ] 9.99 is converted to its binary representation and rounded up
- [ ] NumPy raises a TypeError when it encounters any float in an integer array

### Why is `mylist + 4` a TypeError but `myarray + 4` works correctly?
- [x] Arrays support element-wise arithmetic; lists interpret + as concatenation only
- [ ] Lists cannot store the number 4 without type conversion
- [ ] NumPy redefines + to mean append for all Python objects
- [ ] Lists require a for-loop; arrays do not use Python operators

### Why should you avoid building an array with repeated np.append() calls inside a loop?
- [ ] np.append() was deprecated in NumPy 2.0 and now raises a warning
- [x] Each call copies the whole array, creating O(n²) total time
- [ ] np.append() modifies the original array and causes index errors
- [ ] Python loops are not permitted to call NumPy functions more than once

### You want songs with tempo > 120 AND energy > 0.6. Why must you use & instead of Python's 'and'?
- [x] 'and' operates on scalar booleans; & applies the condition element-wise to arrays
- [ ] & is the faster operator than 'and' for any large NumPy array
- [ ] 'and' returns an integer array instead of a boolean array
- [ ] NumPy does not recognize Python keywords inside array expressions

### np.linspace(0, 1, 5) produces [0.0, 0.25, 0.5, 0.75, 1.0]. What does np.arange(0, 1, 0.25) produce?
- [ ] [0.0, 0.25, 0.5, 0.75, 1.0] — five elements including endpoint
- [x] [0.0, 0.25, 0.5, 0.75] — four elements excluding endpoint
- [ ] [0.25, 0.5, 0.75, 1.0] — excludes start but includes endpoint
- [ ] [0.0, 0.25, 0.5, 0.75, 1.0, 1.25] — extends one step past 1.0

### arr.reshape(-1, 1) converts a 1D array to 2D. Why does scikit-learn require this shape?
- [x] Scikit-learn expects a 2D feature matrix: rows are samples, columns are features
- [ ] Scikit-learn requires 2D input to reduce memory overhead when processing features
- [ ] 2D arrays store more decimal precision per element than equivalent 1D arrays
- [ ] reshape(-1, 1) centers and normalizes all values in the array to unit variance

### np.random.seed(42) is called before generating training data. What would happen without it?
- [ ] The array would contain only zeros until explicitly populated
- [x] Each run generates different data, making results non-reproducible
- [ ] NumPy would raise an error when generating random numbers
- [ ] The generated distribution would no longer follow a normal distribution shape

### An array has shape (100,). You reshape it to (-1, 5). What is the resulting shape?
- [ ] (5, 100) — the -1 dimension is replaced by the total element count
- [x] (20, 5) — NumPy infers 20 rows to accommodate 100 elements
- [ ] (100, 5) — the original array's size of 100 becomes the row count
- [ ] (-1, 5) — NumPy cannot resolve the -1 and raises an error

## 16_4: Pandas Series

### series.loc['apple':'cherry'] includes 'cherry' in the result. series.iloc[1:3] excludes position 3. Why?
- [x] .loc includes the endpoint in label slices; .iloc excludes it like Python lists
- [ ] .loc and .iloc behave identically; the difference is a bug in this example
- [ ] .iloc includes the endpoint when the index uses string labels instead of integers
- [ ] .loc excludes the endpoint when string labels happen to be alphabetically ordered

### A Series has two entries with the label 'apple'. What does series.loc['apple'] return?
- [ ] Only the first 'apple' entry by default
- [x] A Series containing both 'apple' entries
- [ ] A KeyError because duplicate labels are invalid
- [ ] The last 'apple' entry since it overwrites the first

### You write series_copy = series. Later, series_copy['apple'] = 999 also changes series. Why?
- [x] Assignment creates an alias to the same object, not a copy
- [ ] Pandas Series objects are immutable, so any modification propagates to all references
- [ ] The value 999 overwrites the original Series because it's out of range
- [ ] All labels in a Series share the same memory block across variable assignments

### series.sort_values() returns a new Series. Why does the notebook warn against inplace=True?
- [ ] inplace=True is deprecated and will be removed in future Pandas versions
- [x] inplace=True creates subtle bugs with chained operations or views
- [ ] inplace=True runs slower than returning a new Series in all Pandas versions
- [ ] inplace=True resets the index, which destroys the original label information

### You want states with populations between 5 million and 15 million. Which code is correct?
- [ ] series.loc[series > 5e6 and series < 15e6]
- [x] series.loc[(series > 5e6) & (series < 15e6)]
- [ ] series.loc[series > 5e6 | series < 15e6]
- [ ] series[(5e6 < series < 15e6)]

### series.str.split().str[0] extracts the first word from each string. Why is .str[0] needed instead of [0]?
- [x] .str.split() returns a Series of lists; .str[0] extracts element 0 element-wise
- [ ] [0] would return only the first element of the entire outer Series
- [ ] .str[0] is the faster option because it applies vectorized operations to each element
- [ ] [0] raises a TypeError whenever it is applied to a string-typed Series

### The cumulative sum of sorted state populations is used to find which states account for 50% of the US population. Why must you sort first?
- [ ] Unsorted population data produces incorrect results when passed to the .cumsum() method
- [x] Descending sort lets you find the fewest states needed to reach the 50% threshold
- [ ] Boolean masking on population data requires the values to be sorted to work
- [ ] .cumsum() resets its running total to zero when it encounters an out-of-order value

## 16_5: Pandas DataFrames

### Why must you use &, |, and ~ instead of Python's 'and', 'or', 'not' with DataFrame boolean masks?
- [ ] 'and'/'or' are keywords reserved for if-statements and cannot appear in expressions
- [x] 'and'/'or' treat the Series as one truth value; &/| work element-wise
- [ ] Pandas requires the uppercase operators AND, OR, and NOT for all boolean operations
- [ ] 'and'/'or' produce a single integer value rather than a boolean array

### df.loc[1973, 'artist'] returns a single value. df.loc[[1973], 'artist'] returns a Series. Why?
- [x] A list argument signals Pandas to return a Series rather than a scalar
- [ ] Using a list inside brackets returns two independent copies of the matched value
- [ ] This is a known Pandas bug that was patched in recent stable versions
- [ ] The list syntax redirects the label selection to operate on the column axis

### When copying multiple columns, the notebook uses .values: df[['new1','new2']] = df[['old1','old2']].values. Why is .values needed?
- [x] .values strips the index, preventing Pandas from aligning labels and creating NaN
- [ ] .values converts the DataFrame columns into a faster integer-based storage format
- [ ] .values is required when the source and target columns have different data types
- [ ] .values scans for and drops any duplicate index entries before the assignment runs

### Searching for the Sage family with df['Name'].str.contains('Sage') also returns 'Sagesser'. How would you fix this?
- [x] Use str.startswith('Sage ') or str.contains(' Sage') to require a word boundary
- [ ] Use str.find('Sage') == 0 to match only at the start of the string
- [ ] Apply a case-insensitive flag like str.contains('sage', case=False) to fix the match
- [ ] Filter the results by checking name length after the str.contains() method runs

### df.drop(columns=['col1']) returns a new DataFrame. When would using inplace=True be preferable?
- [ ] When the DataFrame is too large to hold an extra copy in memory
- [x] When you want to permanently modify the DataFrame without creating an extra variable
- [ ] inplace=True is preferable in all cases because it runs measurably faster
- [ ] inplace=True only takes effect when dropping rows rather than individual columns

### df.info() shows Non-Null Count for each column. Why should this be the first thing you check on a new dataset?
- [ ] Non-Null Count shows which columns contain the most unique distinct values
- [x] Missing values cause silent errors and biased results if not handled
- [ ] Columns with missing values must be removed before any analysis
- [ ] Non-Null Count is used to determine the appropriate dtypes for each column

### You want to add an 'age_group' column: 'child' if Age < 18, 'adult' otherwise. What is the correct approach?
- [ ] df['age_group'] = 'child' if df['Age'] < 18 else 'adult'
- [ ] Use two boolean masks: assign 'child' where mask is True, 'adult' elsewhere
- [ ] df['age_group'] = df['Age'].apply(lambda x: 'child' if x < 18 else 'adult')
- [x] Both the second and third options work; the first raises a ValueError
