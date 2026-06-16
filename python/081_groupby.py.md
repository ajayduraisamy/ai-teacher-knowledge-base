# Concept: GroupBy Operations

## Concept ID

PYT-081

## Difficulty

INTERMEDIATE

## Domain

Python

## Module

Pandas

## Learning Objectives

- Understand the split-apply-combine paradigm
- Use `groupby()` to group DataFrames by one or more columns
- Apply aggregation functions with `agg()`
- Transform data within groups using `transform()`
- Filter groups using `filter()`
- Use named aggregation for multiple output columns
- Apply custom functions with `apply()` on groups

## Prerequisites

- DataFrame fundamentals (PYT-077)
- Aggregation functions: `sum()`, `mean()`, `count()`, `std()`, etc.
- Indexing and selection (PYT-079)
- Basic understanding of map-reduce patterns

## Definition

The **GroupBy** operation in Pandas implements the **split-apply-combine** strategy:

- **Split**: Break the data into groups based on one or more keys (columns, index levels, or functions).
- **Apply**: Apply a function (aggregation, transformation, or filtration) to each group independently.
- **Combine**: Merge the results into a single data structure.

```python
df.groupby(by=key, axis=0).agg(func)
df.groupby(by=key).transform(func)
df.groupby(by=key).filter(func)
```

## Intuition

Think of GroupBy as the Pandas equivalent of SQL's `GROUP BY` clause. It takes a flat table and produces grouped statistics:

- "What is the average salary by department?"
- "How many customers churned in each region?"
- "What is the total revenue per month?"

The `GroupBy` object is lazy — it doesn't compute anything until you call an aggregation, transformation, or filtration method.

## Why This Concept Matters

GroupBy is one of the most frequently used operations in data analysis. It enables:

- **Aggregation**: Computing summary statistics per group (mean, sum, count).
- **Comparison**: Comparing subgroups within a dataset (e.g., test scores by gender).
- **Normalization**: Scaling data within groups (z-scores, percentages).
- **Feature Engineering**: Creating group-level features (e.g., average purchase amount per customer).
- **Data Quality**: Detecting anomalies within groups.

Without GroupBy, you'd need nested loops and manual aggregation code — error-prone and slow.

## Real World Examples

1. **E-commerce**: Total sales per product category per month.
2. **HR Analytics**: Average tenure and salary by department and job level.
3. **Healthcare**: Average patient recovery time by treatment group.
4. **Finance**: Daily portfolio return by asset class.
5. **Marketing**: Click-through rate by campaign and channel.

## AI/ML Relevance

- **Aggregating Predictions**: Average model predictions by group for ensemble methods.
- **Feature Engineering**: Create group-level features (mean encoding, target encoding).
- **Cross-Validation**: GroupKFold ensures all samples from one group are in the same fold.
- **Bias Analysis**: Compare model performance across demographic groups.
- **Time Series**: Aggregate forecasts by week/month for reporting.

## Code Examples

### Example 1: Basic GroupBy — split-apply-combine

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'Department': ['IT', 'HR', 'IT', 'Finance', 'HR', 'IT'],
    'Employee': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
    'Salary': [70000, 85000, 95000, 72000, 65000, 88000],
    'Experience': [2, 5, 8, 3, 4, 6]
})

print(df)
# Output:
#   Department Employee  Salary  Experience
# 0         IT    Alice   70000           2
# 1         HR      Bob   85000           5
# 2         IT  Charlie   95000           8
# 3    Finance    Diana   72000           3
# 4         HR      Eve   65000           4
# 5         IT    Frank   88000           6

grouped = df.groupby('Department')
print(grouped)
# Output:
# <pandas.core.groupby.generic.DataFrameGroupBy object at 0x...>

# Mean salary per department
print(grouped['Salary'].mean())
# Output:
# Department
# Finance    72000.0
# HR         75000.0
# IT         84333.3
# Name: Salary, dtype: float64
```

### Example 2: Multiple aggregation functions

```python
result = df.groupby('Department')['Salary'].agg(['mean', 'std', 'min', 'max', 'count'])
print(result)
# Output:
#               mean           std    min    max  count
# Department
# Finance    72000.0          NaN  72000  72000      1
# HR         75000.0  14142.13562  65000  85000      2
# IT         84333.3  11789.37121  70000  95000      3

# Aggregating multiple columns
result = df.groupby('Department').agg({
    'Salary': ['mean', 'max'],
    'Experience': ['mean', 'min']
})
print(result)
# Output:
#             Salary         Experience
#               mean    max      mean min
# Department
# Finance    72000.0  72000  3.000000   3
# HR         75000.0  85000  4.500000   4
# IT         84333.3  95000  5.333333   2
```

### Example 3: Named aggregation

```python
result = df.groupby('Department').agg(
    avg_salary=('Salary', 'mean'),
    max_salary=('Salary', 'max'),
    avg_experience=('Experience', 'mean'),
    employee_count=('Employee', 'count')
)
print(result)
# Output:
#            avg_salary  max_salary  avg_experience  employee_count
# Department
# Finance        72000       72000        3.000000               1
# HR             75000       85000        4.500000               2
# IT             84333       95000        5.333333               3
```

### Example 4: `transform()` — broadcasting group statistics

```python
# Add a column with the mean salary per department
df['Dept_Avg_Salary'] = df.groupby('Department')['Salary'].transform('mean')
print(df)
# Output:
#   Department Employee  Salary  Experience  Dept_Avg_Salary
# 0         IT    Alice   70000           2     84333.333333
# 1         HR      Bob   85000           5     75000.000000
# 2         IT  Charlie   95000           8     84333.333333
# 3    Finance    Diana   72000           3     72000.000000
# 4         HR      Eve   65000           4     75000.000000
# 5         IT    Frank   88000           6     84333.333333

# Z-score within each group
df['Salary_Z_Group'] = df.groupby('Department')['Salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)
print(df[['Department', 'Employee', 'Salary', 'Salary_Z_Group']])
# Output:
#   Department Employee  Salary  Salary_Z_Group
# 0         IT    Alice   70000       -1.215951
# 1         HR      Bob   85000        0.707107
# 2         IT  Charlie   95000        0.904144
# 3    Finance    Diana   72000             NaN
# 4         HR      Eve   65000       -0.707107
# 5         IT    Frank   88000        0.311806
```

### Example 5: `filter()` — selecting groups

```python
# Keep only departments with average salary > 75000
result = df.groupby('Department').filter(lambda g: g['Salary'].mean() > 75000)
print(result)
# Output:
#   Department Employee  Salary  Experience
# 0         IT    Alice   70000           2
# 2         IT  Charlie   95000           8
# 5         IT    Frank   88000           6

# Keep groups with more than 1 employee
result = df.groupby('Department').filter(lambda g: len(g) > 1)
print(result['Department'].unique())
# Output:
# ['IT', 'HR']
```

### Example 6: Grouping by multiple columns

```python
df['Office'] = ['NY', 'NY', 'SF', 'SF', 'NY', 'SF']
result = df.groupby(['Department', 'Office'])['Salary'].mean()
print(result)
# Output:
# Department  Office
# Finance     SF        72000.0
# HR          NY        75000.0
# IT          NY        77500.0
#             SF        91500.0
# Name: Salary, dtype: float64

# Unstack for a pivot-like view
print(result.unstack())
# Output:
# Office          NY      SF
# Department
# Finance        NaN  72000.0
# HR          75000.0      NaN
# IT          77500.0  91500.0
```

### Example 7: `apply()` on groups

```python
# Top employee per department by salary
def top_employee(group):
    return group.nlargest(1, 'Salary')

result = df.groupby('Department').apply(top_employee)
print(result)
# Output:
#                    Department Employee  Salary  Experience Office
# Department
# Finance    3       Finance    Diana   72000           3     SF
# HR         1            HR      Bob   85000           5     NY
# IT         2            IT  Charlie   95000           8     SF
```

### Example 8: GroupBy with index levels

```python
arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
index = pd.MultiIndex.from_arrays(arrays, names=['Category', 'Sub'])
df_multi = pd.DataFrame({'Value': [10, 20, 30, 40]}, index=index)

# Group by the first level of the index
print(df_multi.groupby(level='Category')['Value'].sum())
# Output:
# Category
# A    30
# B    70
# Name: Value, dtype: int64
```

## Common Mistakes

1. **Forgetting that `groupby` is lazy.** No computation happens until `.agg()`, `.transform()`, `.filter()`, or `.apply()` is called.
2. **Using `.apply()` when `.transform()` or `.agg()` would be faster and more appropriate.** `.apply()` is flexible but slow.
3. **Assuming `groupby` preserves the original index.** By default, group keys become the index of the result. Use `as_index=False` to keep them as columns.
4. **Modifying the original DataFrame inside a `transform` or `apply`.** These may operate on copies, leading to `SettingWithCopyWarning`.
5. **Grouping by a column with many unique values.** The resulting groups can be tiny and cause division-by-zero or NaN issues.

## Interview Questions

### Beginner

1. What is the split-apply-combine paradigm?
2. How do you compute the mean of a column for each group?
3. What is the difference between `.agg()` and `.transform()`?
4. How do you group by multiple columns?
5. What does `as_index=False` do in `groupby`?

### Intermediate

1. How would you compute the percentage of total within each group?
2. What is the difference between `.filter()` and boolean indexing after `groupby`?
3. How do you use named aggregation to rename output columns?
4. How can you apply different aggregation functions to different columns?
5. When would you use `.apply()` on a GroupBy object vs `.agg()`?

### Advanced

1. Compare the performance of `.transform()` vs `groupby().apply()` for computing group-wise z-scores on a 10M-row dataset.
2. How does Pandas handle categorical groupers internally (e.g., `by=pd.Categorical(...)`) — does it observe all categories or only observed ones?
3. Implement a custom GroupBy engine that uses a hash-based partition approach for distributed (Dask-style) grouping.

## Practice Problems

### Easy

1. Group a DataFrame by `'Category'` and compute the mean of `'Value'`.
2. Count the number of rows in each group using `size()`.
3. Add a column to a DataFrame showing the group-wise maximum of another column.
4. Filter groups that have more than 10 rows.
5. Group by two columns and compute the sum of a third column.

### Medium

1. Compute the rank of values within each group and add it as a new column.
2. For each department, find the employee with the highest salary using `idxmax()` and `loc`.
3. Compute the cumulative sum of sales within each product group, ordered by date.
4. Implement a function that computes the deciles of a numeric column within each group.
5. Group a time series by month and year, then compute the month-over-month change in the sum.

### Hard

1. Implement a streaming groupby that processes a large CSV in chunks and computes rolling group statistics without loading all data into memory.
2. Build a class `GroupByEncoder` that performs target encoding (replacing a categorical column with the mean of the target within each group) with smoothing, handling new categories in test data.
3. Given a DataFrame with hierarchical groups (e.g., Company > Department > Team), compute aggregate statistics at each level and return a MultiIndex DataFrame with cumulative aggregation along the hierarchy.

## Solutions

```python
# E1
df.groupby('Category')['Value'].mean()

# E2
df.groupby('Category').size()

# E3
df['Group_Max'] = df.groupby('Category')['Value'].transform('max')

# E4
df.groupby('Category').filter(lambda g: len(g) > 10)

# E5
df.groupby(['Cat1', 'Cat2'])['Value'].sum()

# M1
df['Group_Rank'] = df.groupby('Category')['Value'].rank()

# M2
idx = df.groupby('Department')['Salary'].idxmax()
df.loc[idx]

# M3
df['Cumulative_Sales'] = df.groupby('Product')['Sales'].cumsum()

# M4
df['Decile'] = df.groupby('Category')['Value'].transform(
    lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')
)

# M5
monthly = df.set_index('Date').groupby('Product').resample('M')['Sales'].sum()
monthly.groupby('Product').pct_change()
```

## Related Concepts

- SQL `GROUP BY` clause
- MapReduce paradigm
- Pivot tables (a special case of grouping and aggregating)
- Window functions in SQL (analogous to `.transform()`)

## Next Concepts

- Pivot tables and cross-tabulation
- Merging and joining grouped results
- Time series resampling (a time-based groupby)

## Summary

The GroupBy operation in Pandas implements split-apply-combine: split data by keys, apply aggregation/transformation/filtration functions to each group, and combine the results. Key methods are `agg()` for summary statistics, `transform()` for broadcasting group values back to the original index, `filter()` for selecting groups based on conditions, and `apply()` for arbitrary custom logic. Named aggregation provides clean named output columns.

## Key Takeaways

- `groupby()` splits data by column values, index levels, or custom functions.
- `.agg()` returns reduced rows (one per group); `.transform()` returns same-length results.
- `.filter()` keeps or discards entire groups based on a condition.
- Named aggregation `.agg(name=(col, func))` is clean and readable.
- `.apply()` is flexible but slow — prefer `.agg()` or `.transform()` when possible.
- Multiple grouping keys produce a MultiIndex; use `unstack()` for pivot-like views.
- GroupBy is lazy — nothing computes until an aggregation method is called.
