# Concept: Handling Missing Data

## Concept ID

PYT-080

## Difficulty

INTERMEDIATE

## Domain

Python

## Module

Pandas

## Learning Objectives

- Detect missing values with `isna()` and `notna()`
- Remove missing values using `dropna()`
- Fill missing values with `fillna()` using forward fill, backward fill, and interpolation
- Replace specific values with `replace()`
- Understand the different types of null values in Pandas (None, NaN, NaT, NA)
- Apply missing data strategies to real-world ML preprocessing pipelines

## Prerequisites

- DataFrame and Series fundamentals (PYT-076, PYT-077)
- Indexing and selection (PYT-079)
- Understanding of `np.nan` and Python `None`

## Definition

Missing data in Pandas refers to values that are absent, unknown, or not applicable. Pandas represents missing data using:

- **`NaN`** (Not a Number): `float` type, from NumPy. The default null marker.
- **`None`**: Python singleton, automatically converted to `NaN` in numeric contexts.
- **`NaT`** (Not a Time): For datetime and timedelta data.
- **`pd.NA`**: Pandas' experimental nullable integer/boolean type marker.

```python
import pandas as pd
import numpy as np

# Detecting
df.isna()
df.notna()

# Removing
df.dropna()
df.dropna(axis=1, how='any')

# Filling
df.fillna(value)
df.fillna(method='ffill')
df.fillna(method='bfill')
df.interpolate()

# Replacing
df.replace(to_replace, value)
```

## Intuition

Think of missing data as **holes** in your dataset. The three strategies are:

1. **Detect**: Find where the holes are.
2. **Remove**: Drop rows or columns with holes (simple but may lose information).
3. **Fill**: Patch the holes with reasonable estimates (preserves data size).

The choice between removal and filling depends on:
- How much data is missing
- Why it is missing (MCAR, MAR, MNAR)
- The downstream task (ML models generally cannot handle NaN directly)

## Why This Concept Matters

Real-world datasets almost always have missing values. Surveys have non-responses, sensors fail, databases have nulls, and data integration produces gaps. How you handle missing data directly impacts:

- **Model quality**: Poor imputation introduces bias; aggressive dropping reduces sample size.
- **Data leakage**: Fill methods using future data (e.g., bfill) can leak information in time series.
- **Pipeline correctness**: Missing values must be handled before ML algorithms that don't support NaN (e.g., scikit-learn, XGBoost, neural networks).

## Real World Examples

1. **Medical Records**: Patient blood pressure measurements missing for some visits.
2. **Customer Surveys**: Income field left blank by 30% of respondents.
3. **IoT Sensors**: Temperature readings lost during network outages (gaps in time series).
4. **Financial Data**: Stock prices missing on non-trading days (weekends, holidays).
5. **Web Analytics**: Pageview count missing for pages that weren't tracked.

## AI/ML Relevance

Missing data handling is a critical preprocessing step in ML pipelines:

- **scikit-learn**: Estimators do not support NaN. Use `SimpleImputer` or custom `fillna` before training.
- **XGBoost/LightGBM**: Some implementations handle NaN natively by learning default split directions.
- **Neural Networks**: Require imputation or use masking layers.
- **Time Series Forecasting**: `ffill` is common but can be misleading if gaps are large.
- **Feature Engineering**: Missing indicators (`isna()`) can be powerful features.

## Code Examples

### Example 1: Detecting missing values

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, np.nan, 5],
    'C': ['x', 'y', None, 'z', 'w'],
    'D': [1.0, 2.0, 3.0, 4.0, 5.0]
})

print(df.isna())
# Output:
#        A      B      C      D
# 0  False   True  False  False
# 1  False  False  False  False
# 2   True  False   True  False
# 3  False   True  False  False
# 4  False  False  False  False

print(df.isna().sum())
# Output:
# A    1
# B    2
# C    1
# D    0
# dtype: int64

print(df.notna())
# Output:
#        A      B      C     D
# 0   True  False   True  True
# 1   True   True   True  True
# 2  False   True  False  True
# 3   True  False   True  True
# 4   True   True   True  True
```

### Example 2: Dropping missing values

```python
# Drop rows with ANY missing value
print(df.dropna())
# Output:
#      A    B  C    D
# 1  2.0  2.0  y  2.0
# 4  5.0  5.0  w  5.0

# Drop rows where ALL values are missing
df_all_na = pd.DataFrame({'X': [np.nan, np.nan], 'Y': [np.nan, np.nan]})
print(df_all_na.dropna(how='all'))
# Output:
# Empty DataFrame
# Columns: [X, Y]
# Index: []

# Drop columns with any missing values
print(df.dropna(axis=1))
# Output:
#      D
# 0  1.0
# 1  2.0
# 2  3.0
# 3  4.0
# 4  5.0

# Keep rows with at least 3 non-NA values
print(df.dropna(thresh=3))
# Output:
#      A    B    C    D
# 0  1.0  NaN    x  1.0
# 1  2.0  2.0    y  2.0
# 3  4.0  NaN    z  4.0
# 4  5.0  5.0    w  5.0
```

### Example 3: Filling missing values with a constant

```python
# Fill all NaN with 0
print(df.fillna(0))
# Output:
#      A    B  C    D
# 0  1.0  0.0  x  1.0
# 1  2.0  2.0  y  2.0
# 2  0.0  3.0  0  3.0
# 3  4.0  0.0  z  4.0
# 4  5.0  5.0  w  5.0

# Fill with column mean
df_filled = df.copy()
df_filled['A'] = df['A'].fillna(df['A'].mean())
print(df_filled)
# Output:
#      A    B    C    D
# 0  1.0  NaN    x  1.0
# 1  2.0  2.0    y  2.0
# 2  3.0  3.0  NaN  3.0
# 3  4.0  NaN    z  4.0
# 4  5.0  5.0    w  5.0

# Different fill values per column
df.fillna({'A': 0, 'B': df['B'].median()})
```

### Example 4: Forward fill and backward fill

```python
s = pd.Series([1, np.nan, np.nan, 4, np.nan, 6])

# Forward fill
print(s.fillna(method='ffill'))
# Output:
# 0    1.0
# 1    1.0
# 2    1.0
# 3    4.0
# 4    4.0
# 5    6.0
# dtype: float64

# Backward fill
print(s.fillna(method='bfill'))
# Output:
# 0    1.0
# 1    4.0
# 2    4.0
# 3    4.0
# 4    6.0
# 5    6.0
# dtype: float64

# Limit the number of consecutive fills
print(s.fillna(method='ffill', limit=1))
# Output:
# 0    1.0
# 1    1.0
# 2    NaN
# 3    4.0
# 4    4.0
# 5    6.0
# dtype: float64
```

### Example 5: Interpolation

```python
# Linear interpolation
s = pd.Series([1, np.nan, np.nan, 4, np.nan, 6])
print(s.interpolate(method='linear'))
# Output:
# 0    1.000000
# 1    2.000000
# 2    3.000000
# 3    4.000000
# 4    5.000000
# 5    6.000000
# dtype: float64

# Time-based interpolation
dates = pd.date_range('2024-01-01', periods=6, freq='D')
ts = pd.Series([10, np.nan, np.nan, 40, np.nan, 60], index=dates)
print(ts.interpolate(method='time'))
# Output:
# 2024-01-01    10.0
# 2024-01-02    20.0
# 2024-01-03    30.0
# 2024-01-04    40.0
# 2024-01-05    50.0
# 2024-01-06    60.0
# dtype: float64

# Polynomial interpolation
s = pd.Series([1, np.nan, np.nan, 27, np.nan, 125])
print(s.interpolate(method='polynomial', order=3))
# Output:
# 0      1.0
# 1      8.0
# 2      NaN  (may fail if order too high)
# 3     27.0
# 4     64.0
# 5    125.0
# dtype: float64
```

### Example 6: Using `replace()`

```python
df = pd.DataFrame({
    'A': [1, -999, 3, -999, 5],
    'B': ['NA', 'x', 'y', 'NA', 'z']
})

# Replace specific values with NaN
df_replaced = df.replace(-999, np.nan)
print(df_replaced)
# Output:
#      A  B
# 0  1.0 NA
# 1  NaN  x
# 2  3.0  y
# 3  NaN NA
# 4  5.0  z

# Replace multiple values at once
df_replaced = df.replace({-999: np.nan, 'NA': np.nan})
print(df_replaced)
# Output:
#      A    B
# 0  1.0  NaN
# 1  NaN    x
# 2  3.0    y
# 3  NaN  NaN
# 4  5.0    z

# Replace using regex
df['B'] = df['B'].replace(r'^N.*', np.nan, regex=True)
```

### Example 7: Missing value indicator features

```python
df = pd.DataFrame({'value': [1, np.nan, 3, np.nan, 5]})
df['value_missing'] = df['value'].isna().astype(int)
df['value'] = df['value'].fillna(df['value'].mean())
print(df)
# Output:
#    value  value_missing
# 0    1.0              0
# 1    3.0              1
# 2    3.0              0
# 3    3.0              1
# 4    5.0              0
```

## Common Mistakes

1. **Assuming `dropna()` drops rows by default.** It drops rows (axis=0). Use `axis=1` for columns.
2. **Filling with mean/median before train/test split.** This leaks test set information. Always fit imputer on training set only.
3. **Using `ffill` without checking gap length.** In time series, forward-filling across large gaps introduces misleading constant values.
4. **Confusing `None` with `NaN` in non-numeric columns.** `df['col'].fillna('missing')` works; `df.fillna(0)` only fills numeric columns.
5. **Not using `inplace` correctly.** Many users expect `df.fillna(0)` to modify `df` in place, but it returns a new DataFrame unless `inplace=True`.

## Interview Questions

### Beginner

1. How do you check for missing values in a DataFrame?
2. What is the difference between `isna()` and `isnull()`?
3. How do you drop all rows with any missing values?
4. How do you fill NaN with a constant value?
5. What does `fillna(method='ffill')` do?

### Intermediate

1. Explain the difference between MCAR, MAR, and MNAR missing data mechanisms.
2. How would you fill missing values differently for numeric vs categorical columns?
3. What is the danger of using `fillna` with the column mean before a train/test split?
4. How do you use `interpolate` for time-series missing data?
5. How can you create a missing value indicator feature?

### Advanced

1. Implement a custom imputer that uses K-Nearest Neighbors to estimate missing values.
2. Compare the bias-variance trade-off of mean imputation vs multiple imputation (MICE) in a regression setting.
3. Under what conditions is listwise deletion (dropping rows) preferred over imputation? Provide mathematical justification.

## Practice Problems

### Easy

1. Create a DataFrame with 5 columns and 10 rows, introduce 10 random NaN values, then count them.
2. Drop any column that has more than 50% missing values.
3. Fill all missing numeric values with the column median.
4. Forward-fill missing values in a Series, limiting to 2 consecutive fills.
5. Replace the value `-1` in a DataFrame with `NaN`.

### Medium

1. On a dataset with 30% missing values in one column, compare mean imputation vs median imputation by computing the distribution difference.
2. Implement a function that imputes missing categorical values using the mode within each group defined by another column.
3. For a time series with 1000 points and 50 missing values scattered randomly, compare linear interpolation, ffill, and bfill using RMSE against the true values.
4. Build a pipeline that first drops columns with > 80% missing, then imputes remaining numeric missing with median and categorical with mode, then adds missing indicators.
5. Detect and visualize the pattern of missingness in a DataFrame using a missingno-style heatmap (binary matrix of isna).

### Hard

1. Implement the MICE (Multiple Imputation by Chained Equations) algorithm from scratch for a dataset with 3 columns.
2. Design an adaptive imputation strategy that chooses between mean, median, or KNN imputation based on the distribution characteristics of each column (skewness, kurtosis, missing rate).
3. Build a class `MissingDataFrame` that extends DataFrame and automatically tracks which values were originally missing, maintaining a parallel boolean mask through all transformations.

## Solutions

```python
# E1
# No sample code needed — straightforward.

# E2
df.dropna(thresh=len(df) * 0.5, axis=1, inplace=True)

# E3
df.fillna(df.median(), inplace=True)

# E4
s = pd.Series([1, np.nan, np.nan, np.nan, 5])
s.fillna(method='ffill', limit=2)

# E5
df.replace(-1, np.nan, inplace=True)

# M1
original = df['col'].copy()
mean_filled = df['col'].fillna(df['col'].mean())
median_filled = df['col'].fillna(df['col'].median())
print(f"Mean diff: {(mean_filled - original).sum()}")
print(f"Median diff: {(median_filled - original).sum()}")

# M2
def impute_mode_by_group(df, value_col, group_col):
    mode_per_group = df.groupby(group_col)[value_col].transform(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    )
    df[value_col] = df[value_col].fillna(mode_per_group)
    return df

# M3 (conceptual)
from sklearn.metrics import mean_squared_error
# ... create series with known missing, compare methods

# M4
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# M5
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.imshow(df.isna(), aspect='auto', cmap='gray')
```

## Related Concepts

- Data cleaning and preprocessing
- Imputation techniques (mean, median, mode, KNN, MICE)
- scikit-learn `SimpleImputer` and `IterativeImputer`
- Missing data mechanisms (MCAR, MAR, MNAR)

## Next Concepts

- GroupBy operations for grouped imputation
- Merging datasets where missing data may result from join mismatches
- Time series-specific missing data handling

## Summary

Handling missing data is a fundamental data preprocessing task. Pandas provides `isna()`/`notna()` for detection, `dropna()` for removal, `fillna()` for filling (with ffill, bfill, interpolation), and `replace()` for value substitution. The choice of strategy depends on the missing data mechanism, data type, and downstream ML requirements. Adding missing-indicator features can improve model performance.

## Key Takeaways

- `isna()`/`notna()` detect missing values; sum them for a quick count.
- `dropna()` removes rows (axis=0) or columns (axis=1) with `how='any'` or `how='all'`.
- `fillna()` with `ffill`/`bfill` is useful for time series; limit prevents over-fill.
- `interpolate()` fills by estimating between known points (linear, time, polynomial).
- Always fit imputation parameters on training data only to avoid data leakage.
- `replace()` converts sentinel values (like -999 or 'NA') to NaN.
- Missing indicators (`isna().astype(int)`) preserve information about which values were originally missing.
