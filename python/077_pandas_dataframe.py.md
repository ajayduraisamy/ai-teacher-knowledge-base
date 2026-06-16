# Concept: Pandas DataFrame

## Concept ID

PYT-077

## Difficulty

INTERMEDIATE

## Domain

Python

## Module

Pandas

## Learning Objectives

- Construct a DataFrame from dictionaries, lists, NumPy arrays, and CSV files
- Inspect DataFrame structure using `shape`, `info()`, `describe()`, `head()`, `tail()`
- Access and modify column data types (dtypes)
- Convert between data types using `astype()` and `pd.to_numeric()`
- Understand the tabular data model and how it maps to ML datasets

## Prerequisites

- Series fundamentals (PYT-076)
- NumPy basics: arrays, shapes, dtypes
- Python dictionaries and lists
- Basic file I/O understanding

## Definition

A **DataFrame** is a two-dimensional, labeled data structure with columns of potentially different types. It is the primary Pandas data structure for tabular data. Think of it as a spreadsheet or SQL table where:

- **Rows** represent observations (samples, records)
- **Columns** represent variables (features, fields)
- Each column is a **Series** sharing a common index

```python
import pandas as pd
import numpy as np

df = pd.DataFrame(data, index=index, columns=columns, dtype=dtype)
```

## Intuition

A DataFrame is a **dictionary of Series** sharing the same index. Each column name maps to a Series. This design makes it natural to:

- Add or drop columns as if modifying a dictionary
- Apply column-wise operations using vectorized expressions
- Filter rows using boolean masks
- Align operations across rows and columns via the index

Conceptually, a DataFrame has two axes: `axis=0` (rows, index) and `axis=1` (columns). Most methods accept an `axis` parameter to choose which dimension to operate along.

## Why This Concept Matters

DataFrames are the workhorse of data science in Python. Every real-world dataset — CSV exports, SQL query results, JSON APIs, Excel spreadsheets — is loaded into a DataFrame for analysis. The DataFrame's ability to:

- Hold heterogeneous types (int, float, string, datetime) in the same table
- Handle missing values gracefully
- Provide a rich set of descriptive and aggregating methods
- Interface directly with machine learning libraries (scikit-learn, TensorFlow, PyTorch)

makes it indispensable. Without DataFrames, Python data science would require stitching together NumPy arrays, dicts, and custom loops.

## Real World Examples

1. **Customer Churn Table**: 100k rows with columns: `customer_id`, `age`, `tenure`, `monthly_charges`, `churn_label`.
2. **Housing Dataset**: `price`, `sqft`, `bedrooms`, `bathrooms`, `year_built`, `zipcode`.
3. **Sensor Readings**: `timestamp`, `temperature`, `humidity`, `pressure` from IoT devices.
4. **Financial Transactions**: `txn_id`, `amount`, `merchant`, `timestamp`, `fraud_flag`.
5. **Medical Records**: `patient_id`, `age`, `blood_pressure`, `cholesterol`, `diagnosis`.

## AI/ML Relevance

DataFrames are the primary container for tabular ML datasets.

- **Feature Matrix (X)**: A DataFrame of shape `(n_samples, n_features)` passed to model training.
- **Target Vector (y)**: A single column (Series) extracted from the DataFrame.
- **Preprocessing**: Pipelines for scaling, encoding, and imputation operate on DataFrames.
- **Feature Engineering**: New columns are added as Series to the DataFrame.
- **EDA**: `describe()`, `info()`, and `corr()` provide rapid dataset understanding.
- **Model Output**: Predictions are often concatenated as a new column back into the DataFrame for evaluation.

## Code Examples

### Example 1: Creating a DataFrame from a dictionary

```python
import pandas as pd
import numpy as np

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'Salary': [70000, 85000, 95000, 72000],
    'Department': ['IT', 'HR', 'Finance', 'IT']
}
df = pd.DataFrame(data)
print(df)
# Output:
#       Name  Age  Salary Department
# 0    Alice   25   70000         IT
# 1      Bob   30   85000         HR
# 2  Charlie   35   95000    Finance
# 3    Diana   28   72000         IT
```

### Example 2: Creating from a NumPy array with column names

```python
arr = np.random.randn(5, 3)
df = pd.DataFrame(arr, columns=['Feature_A', 'Feature_B', 'Feature_C'])
print(df)
# Output:
#    Feature_A  Feature_B  Feature_C
# 0   0.123456  -0.789012   1.234567
# 1  -0.345678   0.901234  -0.567890
# 2   1.234567  -0.123456   0.789012
# 3  -0.890123   0.456789  -1.234567
# 4   0.567890  -0.901234   0.345678
```

### Example 3: Inspecting a DataFrame

```python
print("Shape:", df.shape)
# Output: Shape: (5, 3)

print("\nInfo:")
df.info()
# Output:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 5 entries, 0 to 4
# Data columns (total 3 columns):
#  #   Column     Non-Null Count  Dtype
# ---  ------     --------------  -----
#  0   Feature_A  5 non-null      float64
#  1   Feature_B  5 non-null      float64
#  2   Feature_C  5 non-null      float64
# dtypes: float64(3)
# memory usage: 248.0 bytes

print("\nDescribe:")
print(df.describe())
# Output:
#        Feature_A  Feature_B  Feature_C
# count   5.000000   5.000000   5.000000
# mean    0.137958  -0.091345   0.115357
# std     0.854271   0.742101   0.950123
# min    -0.890123  -0.901234  -1.234567
# 25%    -0.345678  -0.789012  -0.567890
# 50%     0.123456  -0.123456   0.345678
# 75%     0.567890   0.456789   0.789012
# max     1.234567   0.901234   1.234567

print("\nHead (2 rows):")
print(df.head(2))
# Output:
#    Feature_A  Feature_B  Feature_C
# 0   0.123456  -0.789012   1.234567
# 1  -0.345678   0.901234  -0.567890

print("\nTail (2 rows):")
print(df.tail(2))
# Output:
#    Feature_A  Feature_B  Feature_C
# 3  -0.890123   0.456789  -1.234567
# 4   0.567890  -0.901234   0.345678
```

### Example 4: Column data types

```python
data = {
    'Name': ['Alice', 'Bob'],
    'Age': [25, 30],
    'Salary': [70000.0, 85000.0],
    'Is_Hired': [True, False]
}
df = pd.DataFrame(data)
print(df.dtypes)
# Output:
# Name         object
# Age           int64
# Salary      float64
# Is_Hired       bool
# dtype: object
```

### Example 5: Converting data types

```python
df = pd.DataFrame({'A': ['1', '2', '3'], 'B': [10.5, 20.3, 30.7]})
print(df.dtypes)
# Output:
# A    object
# B    float64
# dtype: object

# Convert column A to int
df['A'] = df['A'].astype(int)
# Convert column B to int (floor)
df['B'] = df['B'].astype(int)
print(df.dtypes)
# Output:
# A    int32
# B    int32
# dtype: object

# Robust numeric conversion
df = pd.DataFrame({'val': ['100', '200', 'N/A', '300']})
df['val_clean'] = pd.to_numeric(df['val'], errors='coerce')
print(df)
# Output:
#    val  val_clean
# 0  100      100.0
# 1  200      200.0
# 2  N/A        NaN
# 3  300      300.0
```

### Example 6: Adding and removing columns

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df['C'] = df['A'] + df['B']  # Add new column
print(df)
# Output:
#    A  B  C
# 0  1  4  5
# 1  2  5  7
# 2  3  6  9

df.drop('B', axis=1, inplace=True)  # Remove column
print(df)
# Output:
#    A  C
# 0  1  5
# 1  2  7
# 2  3  9
```

### Example 7: Renaming columns

```python
df = pd.DataFrame({'old_name': [1, 2, 3]})
df.rename(columns={'old_name': 'new_name'}, inplace=True)
print(df)
# Output:
#    new_name
# 0         1
# 1         2
# 2         3
```

## Common Mistakes

1. **Using `inplace=True` everywhere.** Many methods return a new object; chaining is cleaner and avoids confusion.
2. **Assuming `df.corr()` works on non-numeric columns.** It silently drops non-numeric columns — always check dtypes first.
3. **Confusing `axis=0` (rows) with `axis=1` (columns).** `df.drop('col', axis=1)` vs `df.drop(0, axis=0)`.
4. **Modifying a slice of a DataFrame.** Indexing with chained brackets (`df[df['a'] > 0]['b'] = 5`) triggers `SettingWithCopyWarning`.
5. **Forgetting that `df['col']` returns a Series, not a DataFrame.** Use `df[['col']]` for a single-column DataFrame.

## Interview Questions

### Beginner

1. What is a Pandas DataFrame? How is it different from a NumPy 2D array?
2. How do you create a DataFrame from a dictionary of lists?
3. What does `df.info()` tell you?
4. How do you check the shape of a DataFrame?
5. How can you rename a column in a DataFrame?

### Intermediate

1. Explain the difference between `df['col']` and `df[['col']]`.
2. How do you convert all columns of a DataFrame to optimal types?
3. What is `SettingWithCopyWarning` and how do you avoid it?
4. How would you add a new column that is a function of two existing columns?
5. How can you detect and handle duplicate column names?

### Advanced

1. Explain how Pandas handles mixed dtypes in a DataFrame internally (BlockManager vs ArrayManager).
2. How does memory usage scale with DataFrame size? How would you estimate memory for a 10M row DataFrame with 50 columns?
3. Implement a custom DataFrame subclass that automatically validates column types on assignment.

## Practice Problems

### Easy

1. Create a DataFrame with columns `'Product'`, `'Price'`, `'Quantity'` and 5 rows of sample data.
2. Use `df.describe()` on a numeric DataFrame and interpret the output.
3. Add a new column `'Total'` = `Price * Quantity` to the above DataFrame.
4. Rename all columns to uppercase.
5. Select only the numeric columns from a mixed-type DataFrame.

### Medium

1. Load a CSV with 10 columns, inspect dtypes, and convert any "object" column that contains numbers to float.
2. Given a DataFrame with 100 rows and 20 columns, find and drop columns where more than 50% of values are missing.
3. Create a DataFrame with a MultiIndex column (hierarchical columns) and slice a specific sub-column.
4. Implement a function that takes two DataFrames and returns only columns present in both.
5. For a DataFrame of 1M rows, measure memory usage per column and identify the most expensive column.

### Hard

1. Implement a memory-optimized DataFrame reader that reads a CSV in chunks, downcasts dtypes, and returns a single memory-efficient DataFrame.
2. Build a validation decorator that ensures all columns of a DataFrame match a predefined schema (name + dtype) before a function executes.
3. Write a function that converts a deeply nested JSON (list of dicts of dicts) into a flat DataFrame automatically.

## Solutions

```python
# E1
df = pd.DataFrame({
    'Product': ['A', 'B', 'C', 'D', 'E'],
    'Price': [10, 20, 15, 25, 30],
    'Quantity': [5, 3, 8, 2, 4]
})

# E2
print(df.describe())

# E3
df['Total'] = df['Price'] * df['Quantity']

# E4
df.rename(columns=str.upper, inplace=True)

# E5
numeric_df = df.select_dtypes(include=[np.number])

# M1
df = pd.read_csv('data.csv')
for col in df.select_dtypes(include=['object']).columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')

# M2
threshold = 0.5 * len(df)
df.dropna(thresh=threshold, axis=1, inplace=True)

# M3
arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples)
df = pd.DataFrame(np.random.randn(4, 4), columns=index)
print(df['A'])  # sub-column

# M4
def common_columns(df1, df2):
    common = df1.columns.intersection(df2.columns)
    return df1[common], df2[common]

# M5
import sys
for col in df.columns:
    mem = df[col].memory_usage(deep=True)
    print(f"{col}: {mem / 1024**2:.2f} MB")
```

## Related Concepts

- Series (each column is a Series)
- NumPy ndarray (underlying storage)
- SQL tables (analogous relational model)
- Excel spreadsheets (familiar tabular metaphor)

## Next Concepts

- Reading and writing data (CSV, Excel, SQL, JSON)
- Indexing and selection with `.loc`, `.iloc`, boolean masks
- Missing data handling on DataFrames
- GroupBy operations for aggregation

## Summary

A **Pandas DataFrame** is a two-dimensional labeled data structure for tabular data. It stores heterogeneous column types, supports row and column operations, and provides rich inspection methods (`info()`, `describe()`, `head()`, `tail()`). Each column is a Series sharing a common index. DataFrames are the primary container for tabular ML datasets and are the foundation for data wrangling, preprocessing, and analysis in Python.

## Key Takeaways

- DataFrame = 2D labeled structure with rows (index) and columns (names).
- Construct from dict, list, array, or external files.
- Use `shape` for dimensions, `dtypes` for column types, `info()` for summary.
- `describe()` provides statistical summary for numeric columns.
- `astype()` and `pd.to_numeric()` handle type conversion.
- In ML, `X = df.drop('target', axis=1)` and `y = df['target']`.
- Each column is a Series; select columns with `df['col']` or `df[['col1', 'col2']]`.
- Avoid `SettingWithCopyWarning` by using `.loc` with explicit copies.
