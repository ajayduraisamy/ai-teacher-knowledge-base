# Concept: Pandas Series

## Concept ID

PYT-076

## Difficulty

INTERMEDIATE

## Domain

Python

## Module

Pandas

## Learning Objectives

- Create and initialize a Pandas Series from various data sources
- Access and manipulate the index, values, name, and dtype attributes
- Perform vectorized operations between Series objects
- Understand automatic index alignment behavior
- Handle missing data (NaN) within Series operations
- Apply Series as a representation of single-feature data in ML pipelines

## Prerequisites

- Python fundamentals: lists, dictionaries, NumPy arrays
- Basic understanding of NumPy (`np.array`, `np.nan`)
- Familiarity with Python indexing and slicing syntax
- Installing and importing Pandas (`pip install pandas`)

## Definition

A **Pandas Series** is a one-dimensional labeled array capable of holding any data type (integers, strings, floats, Python objects, etc.). The axis labels are collectively called the **index**. A Series is built on top of NumPy and provides labeled access, alignment, and missing-data handling as first-class features.

```python
import pandas as pd
import numpy as np

s = pd.Series(data, index=index, dtype=dtype, name=name)
```

Each element in a Series has an associated label (its index entry). If no index is provided, Pandas auto-generates a numeric `RangeIndex` from 0 to len(data)-1.

## Intuition

Think of a Series as a hybrid between a Python dictionary and a NumPy array.

- Like a **dictionary**, each value has a key (the index label), and you can look up values by label.
- Like a **NumPy array**, operations are vectorized and fast, and you can slice by integer position.

A Series enforces a consistent structure: every value is of the same dtype, and labels are stored in an `Index` object that supports alignment, set operations, and fancy indexing.

## Why This Concept Matters

The Series is the fundamental building block of Pandas. Every column in a DataFrame is a Series. Mastering Series operations gives you:

- Ability to manipulate individual columns in isolation
- Understanding of the label-based alignment that powers merges, joins, and arithmetic across DataFrames
- Capacity to handle missing data efficiently before feeding into ML models
- Foundation for time-series analysis where each observation is a labeled point

Without a solid grasp of Series, working with DataFrames will feel fragile and error prone. Most indexing bugs in Pandas stem from misunderstanding how Series alignment works.

## Real World Examples

1. **Stock Prices**: A Series indexed by `DatetimeIndex` with daily closing prices for a single ticker.
2. **Customer Ages**: A single column extracted from a customer database, indexed by customer ID.
3. **Feature Vectors**: A single feature column (e.g., "square footage") for all rows in a dataset.
4. **Time Series Anomalies**: Residual errors from a forecasting model, indexed by timestamp.
5. **Survey Responses**: One column of Likert-scale responses indexed by respondent ID.

## AI/ML Relevance

In machine learning, a Series often represents a **single feature** or a **target variable** extracted from a larger DataFrame.

- **Target vector**: `y = df['price']` — a Series passed to `model.fit(X, y)`.
- **Feature column**: After feature engineering, each new column is a Series.
- **Prediction output**: `model.predict(X)` returns an array that you wrap in a Series for alignment with original indices.
- **Residual analysis**: `y_true - y_pred` produces a Series with automatic index alignment.
- **Cross-validation splits**: Stratified splits often use a Series to maintain label integrity.

## Code Examples

### Example 1: Creating a Series from a list

```python
import pandas as pd
import numpy as np

s = pd.Series([10, 20, 30, 40, 50])
print(s)
# Output:
# 0    10
# 1    20
# 2    30
# 3    40
# 4    50
# dtype: int64
```

### Example 2: Creating a Series with a custom index and name

```python
s = pd.Series([85, 92, 78, 95],
              index=['Alice', 'Bob', 'Charlie', 'Diana'],
              name='Exam Score',
              dtype='float64')
print(s)
# Output:
# Alice      85.0
# Bob        92.0
# Charlie    78.0
# Diana      95.0
# Name: Exam Score, dtype: float64
```

### Example 3: Accessing attributes

```python
print("Index   :", s.index)
# Output: Index   : Index(['Alice', 'Bob', 'Charlie', 'Diana'], dtype='object')
print("Values  :", s.values)
# Output: Values  : [85. 92. 78. 95.]
print("Name    :", s.name)
# Output: Name    : Exam Score
print("dtype   :", s.dtype)
# Output: dtype   : float64
print("Shape   :", s.shape)
# Output: Shape   : (4,)
```

### Example 4: Vectorized operations

```python
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([4, 5, 6], index=['a', 'b', 'c'])

print(s1 + s2)
# Output:
# a    5
# b    7
# c    9
# dtype: int64

print(s1 * 10)
# Output:
# a    10
# b    20
# c    30
# dtype: int64

print(s1 > 2)
# Output:
# a    False
# b    False
# c     True
# dtype: bool
```

### Example 5: Automatic index alignment

```python
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([4, 5, 6], index=['b', 'c', 'd'])

result = s1 + s2
print(result)
# Output:
# a    NaN
# b    6.0
# c    8.0
# d    NaN
# dtype: float64
```

Pandas aligns on the index label automatically. Labels that don't appear in both produce NaN.

### Example 6: NaN handling

```python
s = pd.Series([1.0, 2.0, np.nan, 4.0, np.nan])
print(s.isna())
# Output:
# 0    False
# 1    False
# 2     True
# 3    False
# 4     True
# dtype: bool

print(s.fillna(s.mean()))
# Output:
# 0    1.000000
# 1    2.000000
# 2    2.333333
# 3    4.000000
# 4    2.333333
# dtype: float64
```

### Example 7: Boolean indexing on a Series

```python
s = pd.Series([10, 25, 8, 42, 17], index=['a', 'b', 'c', 'd', 'e'])
high = s[s > 15]
print(high)
# Output:
# b    25
# d    42
# e    17
# dtype: int64
```

## Common Mistakes

1. **Assuming alignment by position instead of label.** Two Series with the same values but different indices will produce NaN on arithmetic rather than element-wise positional math.
2. **Mutating a Series inside a DataFrame.** Modifying a Series extracted via `df['col']` can trigger a `SettingWithCopyWarning` if the extraction was a view.
3. **Forgetting that `s.values` returns a NumPy array.** Subsequent operations on `.values` lose all index information.
4. **Using `in` incorrectly.** `value in s` checks the index, not the values. Use `value in s.values` to check values.
5. **Confusing `s.iloc[0]` with `s.loc[0]`.** If the index contains the integer 0, `s.loc[0]` retrieves by label while `s.iloc[0]` retrieves by position — they may differ.

## Interview Questions

### Beginner

1. What is a Pandas Series and how does it differ from a Python list?
2. How do you create a Series from a dictionary?
3. What happens when you add two Series with overlapping and non-overlapping index labels?
4. How can you check for missing values in a Series?
5. What attributes does a Series expose?

### Intermediate

1. Explain automatic index alignment with a code example.
2. How would you remove NaN values from a Series?
3. What is the difference between `.loc` and `.iloc` when indexing a Series?
4. How do you rename the index labels of a Series?
5. How can you apply a custom function to every element of a Series?

### Advanced

1. Explain the memory layout of a Series and how it differs from a NumPy array. When is a Series not backed by a contiguous NumPy array?
2. How does the `name` attribute propagate during arithmetic operations between Series objects?
3. Implement a custom class that behaves like a Series but only accepts integer values, raising on float assignment.

## Practice Problems

### Easy

1. Create a Series from `[100, 200, 300, 400, 500]` with index labels `'a'` through `'e'`, then select the element at label `'c'`.
2. Given `pd.Series([1, 2, 3, 4], index=['w', 'x', 'y', 'z'])`, extract only the values greater than 2.
3. Create two Series of length 3 with overlapping indices and compute their sum.
4. Check how many missing values are in `pd.Series([1, np.nan, 3, np.nan, 5])`.
5. Fill NaN values in a Series with the mean of the non-NaN values.

### Medium

1. Create a Series with a `DatetimeIndex` and extract only entries from a specific month.
2. Given a Series of 100 random integers, find the index location where the cumulative sum exceeds 500.
3. Implement a function that normalizes a Series to range [0, 1] using only vectorized operations.
4. Merge two Series with partly overlapping indices, keeping all values but preferring values from the first Series on overlap.
5. Compute a rolling z-score (mean and std over a 5-element window) for a Series of 50 random values.

### Hard

1. Implement a class `OrderedSeries` that extends Series behavior and guarantees that arithmetic operations only work if both operands have identical index ordering (raises `ValueError` otherwise).
2. Given a Series of timestamps with irregular spacing, resample it to regular 1-minute intervals using forward-fill, but only for gaps of 5 minutes or less (leave larger gaps as NaN).
3. Build a streaming Series accumulator that reads chunks from a generator and maintains running statistics (mean, std, min, max) without storing the entire Series in memory.

## Solutions

```python
# E1
s = pd.Series([100, 200, 300, 400, 500], index=['a', 'b', 'c', 'd', 'e'])
print(s['c'])  # 300

# E2
s = pd.Series([1, 2, 3, 4], index=['w', 'x', 'y', 'z'])
print(s[s > 2])

# E3
a = pd.Series([1, 2, 3], index=['x', 'y', 'z'])
b = pd.Series([4, 5, 6], index=['y', 'z', 'w'])
print(a + b)

# E4
s = pd.Series([1, np.nan, 3, np.nan, 5])
print(s.isna().sum())  # 2

# E5
s = pd.Series([1, np.nan, 3, np.nan, 5])
print(s.fillna(s.mean()))

# M1
dates = pd.date_range('2024-01-01', periods=90, freq='D')
s = pd.Series(np.random.randn(90), index=dates)
print(s[s.index.month == 1])

# M2
np.random.seed(42)
s = pd.Series(np.random.randint(1, 20, 100))
cumsum = s.cumsum()
print(cumsum[cumsum > 500].index[0])

# M3
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# M4
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([4, 5, 6], index=['b', 'c', 'd'])
combined = s1.combine_first(s2)
print(combined)

# M5
s = pd.Series(np.random.randn(50))
rolling_mean = s.rolling(5).mean()
rolling_std = s.rolling(5).std()
z_score = (s - rolling_mean) / rolling_std
```

## Related Concepts

- NumPy arrays (underlying data store)
- Python dictionaries (analogous key-value behavior)
- DataFrame columns (each column is a Series)
- Index object (shared across Series and DataFrames)

## Next Concepts

- DataFrame construction and manipulation
- Indexing and selection with `.loc` and `.iloc`
- Handling missing data with `dropna` and `fillna`
- GroupBy operations on Series within DataFrames

## Summary

A **Pandas Series** is a one-dimensional labeled array that combines the speed of NumPy with dictionary-like label access. Its key features — automatic index alignment, NaN support, vectorized operations, and rich attribute access — make it the cornerstone of data manipulation in Pandas. Every column in a DataFrame is a Series, and nearly every Pandas operation reduces to Series-level computation.

## Key Takeaways

- Series = labeled 1D array with index, values, name, and dtype.
- Arithmetic aligns on index labels, not position — mismatched labels produce NaN.
- Vectorized operations are fast and avoid Python loops.
- Use `.loc` for label-based access, `.iloc` for integer position.
- In ML, a Series serves as a feature column or target vector.
- NaN handling (`isna`, `fillna`, `dropna`) is built-in and essential for real-world data.
- The `name` attribute propagates through operations and appears in display output.
