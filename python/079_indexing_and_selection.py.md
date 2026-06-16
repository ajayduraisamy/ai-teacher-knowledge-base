# Concept: Indexing and Selection

## Concept ID

PYT-079

## Difficulty

INTERMEDIATE

## Domain

Python

## Module

Pandas

## Learning Objectives

- Distinguish between label-based (`.loc`) and integer-based (`.iloc`) indexing
- Use `.at` and `.iat` for fast scalar access
- Apply boolean indexing with multiple conditions
- Use `.query()` for SQL-like filtering
- Filter columns with `.filter()`
- Check membership with `.isin()` and range membership with `.between()`

## Prerequisites

- DataFrame and Series fundamentals (PYT-076, PYT-077)
- Boolean logic with `&`, `|`, `~`
- Python slicing syntax

## Definition

Indexing and selection in Pandas refers to the set of methods used to extract subsets of data from Series and DataFrames. Pandas provides multiple indexing paradigms:

- **Label-based**: `.loc[row_label, col_label]`
- **Integer-based**: `.iloc[row_position, col_position]`
- **Fast scalar**: `.at[label]`, `.iat[position]`
- **Boolean mask**: `df[mask]`
- **Query string**: `df.query('expression')`
- **Column filter**: `df.filter(items=...)`

```python
df.loc[rows, columns]      # Label-based
df.iloc[rows, columns]     # Position-based
df.at[row_label, col_label]    # Fast single value (label)
df.iat[row_pos, col_pos]       # Fast single value (position)
df[df['col'] > 0]              # Boolean indexing
df.query('col > 0')            # Query string
df.isin(['a', 'b'])            # Membership check
df['col'].between(0, 10)       # Range check
```

## Intuition

Think of a DataFrame as a grid with two coordinate systems:

1. **Labels** (names/strings): The row index labels and column names. Use `.loc` when you know the names.
2. **Positions** (integers): The numerical 0-based positions. Use `.iloc` when you know the position.

The confusion between these two is the single most common source of Pandas bugs. A row at integer position 0 might have a label of `5` if the index is [5, 10, 15]. `df.loc[0]` would raise a KeyError, while `df.iloc[0]` would return the first row.

## Why This Concept Matters

Efficient and correct indexing is essential for:

- Data exploration: selecting subsets for visualization and analysis
- Feature engineering: creating new columns from selected data
- Training ML models: splitting features and targets, creating train/test splits
- Debugging: inspecting specific rows and columns
- Performance: using `.at`/`.iat` for scalar access is 10-100x faster than `.loc`/`.iloc`

Without proper indexing, you will encounter cryptic `KeyError`, `IndexingError`, and `SettingWithCopyWarning` messages.

## Real World Examples

1. **Train/Test Split**: Use `.iloc[:8000]` for training rows and `.iloc[8000:]` for test rows.
2. **Customer Segmentation**: Filter rows where `segment == 'Premium'` and `tenure > 12`.
3. **Feature Selection**: Use `.filter(regex='price')` to select all price-related columns.
4. **Outlier Detection**: Mask rows where a z-score column exceeds 3 standard deviations.
5. **Batch Processing**: Use `.iloc[start:start+batch_size]` to feed mini-batches to a model.

## AI/ML Relevance

- **Train/Validation/Test Splits**: Indexing by position (`iloc`) splits ordered data; indexing by label (`loc`) splits by index values for time-series.
- **Feature Selection**: `df.filter()` and boolean indexing extract relevant features.
- **Batch Inference**: `.iloc[batch_start:batch_end]` produces mini-batches for prediction.
- **Data Cleaning**: Boolean indexing identifies and isolates outliers or missing values.
- **Stratified Sampling**: Boolean masks create balanced class subsets.

## Code Examples

### Example 1: `.loc` — label-based indexing

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'Salary': [70000, 85000, 95000, 72000]
}, index=['a', 'b', 'c', 'd'])

# Select row by label
print(df.loc['b'])
# Output:
# Name         Bob
# Age           30
# Salary     85000
# Name: b, dtype: object

# Select multiple rows and a specific column
print(df.loc[['a', 'c'], 'Name'])
# Output:
# a      Alice
# c    Charlie
# Name: Name, dtype: object

# Slice rows by label (inclusive of endpoint)
print(df.loc['b':'d', 'Name':'Salary'])
# Output:
#       Name  Age  Salary
# b      Bob   30   85000
# c  Charlie   35   95000
# d    Diana   28   72000
```

### Example 2: `.iloc` — integer position indexing

```python
# Select first row
print(df.iloc[0])
# Output:
# Name      Alice
# Age          25
# Salary    70000
# Name: a, dtype: object

# Select first 3 rows, first 2 columns
print(df.iloc[:3, :2])
# Output:
#       Name  Age
# a    Alice   25
# b      Bob   30
# c  Charlie   35

# Last row
print(df.iloc[-1])
# Output:
# Name      Diana
# Age          28
# Salary    72000
# Name: d, dtype: object
```

### Example 3: `.at` and `.iat` — fast scalar access

```python
# Fast label-based scalar
print(df.at['c', 'Salary'])
# Output: 95000

# Fast position-based scalar
print(df.iat[1, 2])
# Output: 85000

# Performance comparison (conceptual)
# %timeit df.at['c', 'Salary']   ~ 2 µs
# %timeit df.loc['c', 'Salary']  ~ 20 µs
```

### Example 4: Boolean indexing

```python
# Single condition
over_30 = df[df['Age'] > 30]
print(over_30)
# Output:
#       Name  Age  Salary
# c  Charlie   35   95000

# Multiple conditions (use &, |, ~)
result = df[(df['Age'] > 25) & (df['Salary'] > 75000)]
print(result)
# Output:
#       Name  Age  Salary
# b      Bob   30   85000
# c  Charlie   35   95000

# Negation
not_it = df[~(df['Department'].isin(['HR'])]
# (assuming Department column exists)
```

### Example 5: `.query()` — SQL-like filtering

```python
# Using query with string expression
df['Department'] = ['IT', 'HR', 'Finance', 'IT']
result = df.query('Age > 25 and Salary > 75000')
print(result)
# Output:
#       Name  Age  Salary Department
# b      Bob   30   85000         HR
# c  Charlie   35   95000    Finance

# Query with variable reference
min_age = 28
result = df.query('Age >= @min_age')
print(result)
# Output:
#       Name  Age  Salary Department
# b      Bob   30   85000         HR
# c  Charlie   35   95000    Finance
# d    Diana   28   72000         IT
```

### Example 6: `.filter()` — column selection

```python
# Filter by column names
print(df.filter(items=['Name', 'Salary']))
# Output:
#       Name  Salary
# a    Alice   70000
# b      Bob   85000
# c  Charlie   95000
# d    Diana   72000

# Filter by regex (columns containing 'ame')
print(df.filter(regex='ame'))
# Output:
#       Name
# a    Alice
# b      Bob
# c  Charlie
# d    Diana

# Filter by column type
print(df.select_dtypes(include=['int64']))
# Output:
#    Age  Salary
# a   25   70000
# b   30   85000
# c   35   95000
# d   28   72000
```

### Example 7: `.isin()` — membership check

```python
# Filter rows where Department is in a list
result = df[df['Department'].isin(['IT', 'Finance'])]
print(result)
# Output:
#       Name  Age  Salary Department
# a    Alice   25   70000         IT
# c  Charlie   35   95000    Finance
# d    Diana   28   72000         IT

# Filter rows where index labels are in a set
result = df[df.index.isin(['a', 'c'])]
print(result)
# Output:
#       Name  Age  Salary Department
# a    Alice   25   70000         IT
# c  Charlie   35   95000    Finance
```

### Example 8: `.between()` — range check

```python
# Filter rows where Age is between 25 and 30 (inclusive)
result = df[df['Age'].between(25, 30)]
print(result)
# Output:
#      Name  Age  Salary Department
# a   Alice   25   70000         IT
# b     Bob   30   85000         HR
# d   Diana   28   72000         IT

# .between() on a Series
ages = df['Age']
mask = ages.between(28, 35)
print(mask)
# Output:
# a    False
# b     True
# c     True
# d     True
# Name: Age, dtype: bool
```

## Common Mistakes

1. **Chained indexing with `SettingWithCopyWarning`.** `df[df['a'] > 0]['b'] = 5` modifies a copy, not the original. Use `df.loc[df['a'] > 0, 'b'] = 5`.
2. **Using `.iloc` with labels or `.loc` with positions.** Always verify whether your selection is by label or by integer position.
3. **Forgetting parentheses in boolean conditions.** `df['a'] > 0 & df['b'] < 1` fails due to operator precedence. Use `(df['a'] > 0) & (df['b'] < 1)`.
4. **Assuming `.loc` slicing is exclusive on the end.** `.loc['a':'c']` includes the endpoint; `.iloc[0:3]` excludes position 3.
5. **Using `filter` for row filtering.** `df.filter()` filters column names, not rows. Use boolean indexing or `query` for rows.

## Interview Questions

### Beginner

1. What is the difference between `.loc` and `.iloc`?
2. How do you select a single cell by row label and column name?
3. What does `df[df['col'] > 5]` do?
4. How do you select the first 10 rows and the last 5 columns?
5. What does `df.filter(regex='test')` return?

### Intermediate

1. Explain the `SettingWithCopyWarning` and how to avoid it.
2. How would you use `.query()` to filter rows where a column value is within a list?
3. What is the difference between `df['col']` and `df.loc[:, 'col']`?
4. How can you select all rows where the index is in a given list?
5. When would you use `.at` instead of `.loc`?

### Advanced

1. Explain how `.loc` handles slicing with step, e.g., `df.loc['a':'d':2]`, and compare it to `.iloc` with step.
2. How does Pandas resolve partial string indexing with `.loc` on a DatetimeIndex?
3. Under the hood, compare the indexing paths for `.loc`, `.iloc`, `.at`, and direct `__getitem__` — which are views vs copies?

## Practice Problems

### Easy

1. Select the third row of a DataFrame using `.iloc`.
2. Select the `'Name'` column from a DataFrame using `.loc`.
3. Filter rows where a numeric column is greater than 50.
4. Check which elements of a Series are in the list `[10, 20, 30]`.
5. Filter rows where a column value is between 0 and 100.

### Medium

1. Given a DataFrame with 100 rows, select rows 20 to 40 (inclusive) and columns `'A'`, `'C'`, `'E'` only.
2. Use `.query()` to filter a DataFrame where two columns satisfy `a > b` and `c` is not null.
3. Implement a function that takes a DataFrame and a list of column names and returns only rows where any of those columns contain a specific value.
4. Select all rows where the index label contains the substring `'2024'`.
5. Using boolean indexing, set all negative values in a numeric column to 0 without chained indexing.

### Hard

1. Implement a class that wraps a DataFrame and enforces that all indexing operations go through a logging mechanism that records every access pattern.
2. Given a DataFrame with 50 columns, write a single expression that selects all rows where at least 3 of the columns 'A' through 'J' have values exceeding their respective medians.
3. Build a nested indexing system that allows selection like `df.loc_chain['a':'f', 'col1':'col4', ::2]` — selecting rows 'a' to 'f', columns 'col1' to 'col4', and every other row within that range.

## Solutions

```python
# E1
df.iloc[2]

# E2
df.loc[:, 'Name']

# E3
df[df['col'] > 50]

# E4
s = pd.Series([5, 15, 25, 35])
s.isin([10, 20, 30])

# E5
df[df['col'].between(0, 100)]

# M1
df.iloc[20:41, [df.columns.get_loc('A'),
                df.columns.get_loc('C'),
                df.columns.get_loc('E')]]

# M2
df.query('a > b and c == c')  # NaN != NaN, so c == c filters nulls

# M3
def filter_rows(df, cols, val):
    mask = df[cols].isin([val]).any(axis=1)
    return df[mask]

# M4
mask = df.index.astype(str).str.contains('2024')
df[mask]

# M5
df.loc[df['col'] < 0, 'col'] = 0
```

## Related Concepts

- Python list slicing (`list[start:stop:step]`)
- NumPy array indexing
- SQL `WHERE` clause (analogous to boolean indexing)
- Dictionary key access (analogous to `.loc` with label)

## Next Concepts

- Handling missing data in selected subsets
- GroupBy operations on indexed selections
- Merging and joining DataFrames on index

## Summary

Indexing and selection in Pandas provides multiple powerful paradigms: `.loc` for label-based access, `.iloc` for position-based access, `.at`/`.iat` for fast scalar retrieval, boolean indexing for conditional filtering, `.query()` for SQL-like expressions, `.filter()` for column selection, `.isin()` for membership checks, and `.between()` for range checks. Mastering these tools is essential for correct and efficient data manipulation.

## Key Takeaways

- `.loc` uses labels (inclusive end); `.iloc` uses integer positions (exclusive end).
- `.at`/`.iat` are 10-100x faster than `.loc`/`.iloc` for single scalar access.
- Boolean indexing requires parentheses for multiple conditions: `(cond1) & (cond2)`.
- `.query()` provides readable SQL-like filtering with `@` for variable references.
- Avoid chained indexing; use `.loc` with a boolean mask for assignment.
- `.isin()` checks membership; `.between()` checks range inclusion.
- `.filter()` selects columns by name, regex, or dtype — not rows.
