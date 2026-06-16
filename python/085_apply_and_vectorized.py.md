# Concept: Apply and Vectorized Operations

## Concept ID

PYT-085

## Difficulty

INTERMEDIATE

## Domain

Python

## Module

Pandas

## Learning Objectives

- Distinguish between `apply()`, `applymap()`, and `map()` methods
- Apply custom functions along an axis with `apply()`
- Use vectorized string operations via the `.str` accessor
- Chain operations with `pipe()`
- Identify when `apply()` is necessary vs when vectorized operations are faster
- Apply best practices for performance in Pandas transformations

## Prerequisites

- DataFrame and Series fundamentals (PYT-076, PYT-077)
- Function definitions in Python (`def`, `lambda`)
- Understanding of vectorization concepts
- Basic string manipulation in Python

## Definition

Pandas provides several methods for applying functions to data:

- **`apply()`**: Apply a function along an axis of a DataFrame or to a Series. Works row-wise (`axis=1`) or column-wise (`axis=0`).
- **`applymap()`**: Apply a function element-wise to an entire DataFrame (deprecated in newer versions, use `map()` on DataFrames instead).
- **`map()`**: Apply a function element-wise to a Series (substitution/replacement). Not available on DataFrames.
- **`.str` accessor**: Provides vectorized string operations on Series (e.g., `str.lower()`, `str.contains()`).
- **`pipe()`**: Chain functions that take a DataFrame as the first argument.

```python
# Apply
df.apply(func, axis=0)       # Column-wise
df.apply(func, axis=1)       # Row-wise
series.apply(func)            # Element-wise

# Map
series.map(dict_or_func)      # Element-wise substitution/transformation

# Applymap (Use DataFrame.map in newer versions)
df.map(func)                  # Element-wise on whole DataFrame

# Vectorized string ops
series.str.lower()
series.str.contains('pattern')
series.str.extract(r'(pattern)')

# Pipe
df.pipe(func, arg1, arg2)
```

## Intuition

Think of `apply()` as a **loop disguised as a function call**. It iterates over rows or columns and applies your function to each one. It is flexible but slow.

Vectorized operations (like `df['a'] + df['b']`, or `series.str.lower()`) operate on the entire column at once using optimized C/Cython code. They are **fast** but limited to operations that NumPy/Pandas already support natively.

The rule of thumb: **use vectorized operations whenever possible, and fall back to `apply()` only when no vectorized alternative exists.**

## Why This Concept Matters

Performance in Pandas is not just about writing correct code — it's about writing code that scales. The difference between a vectorized operation and `apply()` on a 10-million-row dataset can be:

- Vectorized: 0.05 seconds
- Apply with built-in function: 2 seconds
- Apply with custom Python function: 30 seconds
- For loop with `iterrows()`: 5+ minutes

Understanding when and how to use each tool determines whether your data pipeline runs in seconds or hours. This is especially critical in ML pipelines where preprocessing runs repeatedly.

## Real World Examples

1. **Feature Engineering**: Creating a `price_per_sqft` column using vectorized division.
2. **Text Cleaning**: `.str.lower().str.replace(r'[^a-z]', '')` on a text column.
3. **Categorizing**: Using `apply()` to map a continuous age value to an age bucket label.
4. **Date Parsing**: Vectorized `pd.to_datetime()` on a date column.
5. **Row-wise Logic**: Using `apply(axis=1)` to compute a composite score from multiple numeric columns.

## AI/ML Relevance

- **Feature Engineering**: Most feature transformations should be vectorized.
- **Text Preprocessing**: `.str` accessor for cleaning text before TF-IDF or embeddings.
- **Custom Transformers**: `apply()` in scikit-learn's `FunctionTransformer` wraps Pandas apply.
- **Pipeline Integration**: `pipe()` enables clean integration of custom preprocessing steps.
- **Performance Critical**: For large datasets, avoiding `apply()` in favor of vectorized ops can reduce training pipeline time by 10-100x.

## Code Examples

### Example 1: `apply()` on a Series

```python
import pandas as pd
import numpy as np

s = pd.Series([1, 2, 3, 4, 5])

# Square each value using lambda
result = s.apply(lambda x: x ** 2)
print(result)
# Output:
# 0     1
# 1     4
# 2     9
# 3    16
# 4    25
# dtype: int64

# Using a named function
def categorize(value):
    if value < 3:
        return 'Low'
    elif value < 5:
        return 'Medium'
    else:
        return 'High'

print(s.apply(categorize))
# Output:
# 0      Low
# 1      Low
# 2   Medium
# 3   Medium
# 4     High
# dtype: object
```

### Example 2: `apply()` on a DataFrame

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})

# Column-wise: sum of each column
print(df.apply(np.sum, axis=0))
# Output:
# A     6
# B    15
# C    24
# dtype: int64

# Row-wise: sum of each row
print(df.apply(np.sum, axis=1))
# Output:
# 0    12
# 1    15
# 2    18
# dtype: int64

# Row-wise: range (max - min) for each row
print(df.apply(lambda row: row.max() - row.min(), axis=1))
# Output:
# 0    6
# 1    6
# 2    6
# dtype: int64

# Column-wise: count of values > threshold
print(df.apply(lambda col: (col > 5).sum(), axis=0))
# Output:
# A    0
# B    0
# C    3
# dtype: int64
```

### Example 3: `map()` — element-wise transformation on a Series

```python
s = pd.Series(['cat', 'dog', 'rabbit', 'cat', 'dog'])

# Map using a dictionary
animal_map = {'cat': 'feline', 'dog': 'canine', 'rabbit': 'lagomorph'}
print(s.map(animal_map))
# Output:
# 0      feline
# 1      canine
# 2    lagomorph
# 3      feline
# 4      canine
# dtype: object

# Map using a function
print(s.map(str.upper))
# Output:
# 0       CAT
# 1       DOG
# 2    RABBIT
# 3       CAT
# 4       DOG
# dtype: object

# Map returns NaN for unmapped values
print(s.map({'cat': 'feline', 'dog': 'canine'}))
# Output:
# 0      feline
# 1      canine
# 2         NaN
# 3      feline
# 4      canine
# dtype: object
```

### Example 4: Vectorized string operations with `.str`

```python
s = pd.Series(['  Hello World!  ', 'Python is great', 'DATA SCIENCE'])

# Cleaning
print(s.str.strip().str.lower())
# Output:
# 0     hello world!
# 1    python is great
# 2       data science
# dtype: object

# String contains
print(s.str.strip().str.contains('Hello', case=False))
# Output:
# 0     True
# 1    False
# 2    False
# dtype: bool

# Regex extraction
emails = pd.Series(['alice@example.com', 'bob@gmail.com', 'noemail'])
print(emails.str.extract(r'([a-z]+)@'))
# Output:
#        0
# 0  alice
# 1    bob
# 2    NaN

# Replace
cleaned = s.str.strip().str.replace(r'\s+', ' ', regex=True)
print(cleaned)
# Output:
# 0      Hello World!
# 1    Python is great
# 2       DATA SCIENCE
# dtype: object

# Length
print(s.str.len())
# Output:
# 0    16
# 1    17
# 2    12
# dtype: int64

# Split
print(s.str.strip().str.split(' '))
# Output:
# 0     [Hello, World!]
# 1    [Python, is, great]
# 2       [DATA, SCIENCE]
# dtype: object
```

### Example 5: Vectorized vs `apply()` performance comparison

```python
import time

df = pd.DataFrame({'a': np.random.randn(100000), 'b': np.random.randn(100000)})

# Vectorized
start = time.time()
df['c_vector'] = df['a'] * 2 + df['b'] ** 2
vector_time = time.time() - start
print(f"Vectorized: {vector_time:.4f} sec")
# Output: Vectorized: 0.0020 sec

# Apply row-wise
start = time.time()
df['c_apply'] = df.apply(lambda row: row['a'] * 2 + row['b'] ** 2, axis=1)
apply_time = time.time() - start
print(f"Apply: {apply_time:.4f} sec")
# Output: Apply: 0.8500 sec

print(f"Speedup: {apply_time / vector_time:.0f}x")
# Output: Speedup: 425x
```

### Example 6: `applymap()` / `map()` on DataFrames

```python
df = pd.DataFrame(np.random.randn(3, 3), columns=['A', 'B', 'C'])
print(df)
# Output:
#           A         B         C
# 0  0.123456 -0.789012  1.234567
# 1 -0.345678  0.901234 -0.567890
# 2  1.234567 -0.123456  0.789012

# Element-wise: format each value to 2 decimal places
formatted = df.map(lambda x: f"{x:.2f}")
print(formatted)
# Output:
#       A     B     C
# 0  0.12 -0.79  1.23
# 1 -0.35  0.90 -0.57
# 2  1.23 -0.12  0.79

# Element-wise: sign function
print(df.map(np.sign))
# Output:
#      A    B    C
# 0  1.0 -1.0  1.0
# 1 -1.0  1.0 -1.0
# 2  1.0 -1.0  1.0
```

### Example 7: `pipe()` — method chaining

```python
def add_column(df, col_name, values):
    df[col_name] = values
    return df

def filter_rows(df, col, threshold):
    return df[df[col] > threshold]

def normalize(df, col):
    df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

result = (df
    .pipe(add_column, 'squared', df['value'] ** 2)
    .pipe(filter_rows, 'value', 3)
    .pipe(normalize, 'value')
)
print(result)
# Output:
#      value    squared
# 3 -1.26491         16
# 4 -0.63246         25
# 5  0.00000         36
# 6  0.63246         49
# 7  1.26491         64
# 8  1.89737         81
# 9  2.52982        100
```

## Common Mistakes

1. **Using `apply(axis=1)` when a vectorized alternative exists.** It is 100-1000x slower.
2. **Modifying the original DataFrame inside `apply()` functions.** This can lead to unexpected side effects and `SettingWithCopyWarning`.
3. **Confusing `map()` with `apply()` on a Series.** `map()` is for element-wise substitution (dictionary or function); `apply()` is more general.
4. **Using `apply()` on string operations when `.str` accessor is available.** Always prefer vectorized `.str` methods.
5. **Forgetting that `apply()` along `axis=1` passes each row as a Series, not as individual values.** Column access in the function must use `row['col_name']`.

## Interview Questions

### Beginner

1. What is the difference between `apply()` and `map()` on a Series?
2. How do you use `apply()` to compute the row-wise sum of a DataFrame?
3. What are vectorized string operations in Pandas?
4. How do you use `str.contains()` to filter rows?
5. What does `pipe()` do in Pandas?

### Intermediate

1. Why is `apply(axis=1)` slower than vectorized operations?
2. How would you apply different functions to different columns using a single `apply()` call?
3. What is the difference between `applymap()` and `apply()` on a DataFrame?
4. How can you use `str.extract()` with a regex group to create a new column?
5. When would you choose `pipe()` over a series of separate assignment statements?

### Advanced

1. Explain the internal dispatch mechanism of `apply()` — how does Pandas decide between cythonized and pure-Python paths?
2. Compare the performance of `apply()` + custom Cython functions vs numba-accelerated ufuncs vs vectorized NumPy operations for a custom element-wise transformation.
3. Implement an `apply` equivalent that uses multiprocessing to parallelize row-wise operations across multiple cores.

## Practice Problems

### Easy

1. Use `apply()` to convert a Series of Celsius temperatures to Fahrenheit.
2. Use `str.upper()` on a Series of strings.
3. Compute the row-wise mean of a DataFrame using `apply(axis=1)`.
4. Use `map()` with a dictionary to recode a Series of categorical values.
5. Use `str.contains()` to filter rows containing a specific substring.

### Medium

1. Use `apply()` to compute a weighted score from three columns: `score = (a*0.5 + b*0.3 + c*0.2)`.
2. Clean a text column by removing punctuation, converting to lowercase, and stripping whitespace — all vectorized.
3. Use `str.extract()` to pull phone numbers (regex: `\d{3}-\d{3}-\d{4}`) from a text column.
4. Build a pipeline using `pipe()` that normalizes numeric columns, encodes categorical columns, and drops missing values.
5. Compare the execution time of `apply()`, vectorized operations, and `np.where()` for a conditional column creation task on 1M rows.

### Hard

1. Implement a function `smart_apply` that analyzes the operation and automatically chooses between vectorized execution and `apply()` based on the function's characteristics.
2. Build a feature engineering class that registers custom transformations and applies them to a DataFrame using `pipe()` chaining, with parallel execution support.
3. Design a query compiler that translates simple Pandas `apply` operations (e.g., `df.apply(lambda row: row[col1] + row[col2], axis=1)`) into SQL WHERE clauses for out-of-core execution.

## Solutions

```python
# E1
celsius = pd.Series([0, 10, 20, 30, 100])
fahrenheit = celsius.apply(lambda c: c * 9/5 + 32)

# E2
s.str.upper()

# E3
df.apply(np.mean, axis=1)

# E4
s.map({'yes': 1, 'no': 0})

# E5
df[df['text'].str.contains('error', case=False, na=False)]

# M1
df['weighted'] = df.apply(lambda row: row['a']*0.5 + row['b']*0.3 + row['c']*0.2, axis=1)

# M2
cleaned = df['text'].str.replace(r'[^\w\s]', '', regex=True)
cleaned = cleaned.str.lower()
cleaned = cleaned.str.strip()

# M3
df['phone'] = df['text'].str.extract(r'(\d{3}-\d{3}-\d{4})')

# M4
def pipeline(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    df = pd.get_dummies(df, columns=cat_cols)
    df = df.dropna()
    return df

result = df.pipe(pipeline)

# M5
import time
df = pd.DataFrame({'a': np.random.randn(1000000), 'b': np.random.randn(1000000)})

# Vectorized
t1 = time.time()
df['c1'] = np.where(df['a'] > 0, df['a'] * 2, df['b'] * 3)
v_time = time.time() - t1

# Apply
t1 = time.time()
df['c2'] = df.apply(lambda r: r['a'] * 2 if r['a'] > 0 else r['b'] * 3, axis=1)
a_time = time.time() - t1

print(f"Vectorized: {v_time:.3f}s, Apply: {a_time:.3f}s, Speedup: {a_time/v_time:.0f}x")
```

## Related Concepts

- NumPy vectorization and ufuncs
- Python `map()` built-in function
- List comprehensions (faster than `apply()` for simple operations on small data)
- Numba JIT compilation for accelerating custom functions

## Next Concepts

- Advanced feature engineering pipelines
- Integrating Pandas with scikit-learn pipelines
- Performance optimization with Dask/Modin for larger-than-memory datasets

## Summary

Pandas provides multiple ways to apply functions: `apply()` for axis-wise operations, `map()` for element-wise Series transformations, `applymap()` (or `df.map()`) for element-wise DataFrame operations, `.str` accessor for vectorized string operations, and `pipe()` for clean function chaining. The key principle is to prefer vectorized operations (including `.str` methods) whenever possible, as they are 100-1000x faster than `apply()` with custom Python functions. Use `apply()` only when no vectorized alternative exists.

## Key Takeaways

- Vectorized ops (column arithmetic, `.str`, `np.where`) are 100-1000x faster than `apply()`.
- `apply(axis=0)` applies function column-wise; `apply(axis=1)` applies row-wise.
- `map()` is for element-wise Series transformation (dictionary or function).
- `.str` accessor provides vectorized string operations: `lower()`, `contains()`, `extract()`, `replace()`, `split()`.
- `pipe()` enables clean, readable method chaining for multi-step transformations.
- Avoid `apply(axis=1)` on large DataFrames — use vectorized column operations instead.
- For simple conditional columns, `np.where()` is faster than `apply()`.
- Always benchmark: what is fast on 1000 rows may be painfully slow on 10M rows.
