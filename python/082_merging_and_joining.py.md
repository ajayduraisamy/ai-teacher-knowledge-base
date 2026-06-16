# Concept: Merging and Joining DataFrames

## Concept ID

PYT-082

## Difficulty

INTERMEDIATE

## Domain

Python

## Module

Pandas

## Learning Objectives

- Merge DataFrames using `pd.merge()` with inner, outer, left, and right joins
- Specify join keys using `on`, `left_on`, `right_on`
- Handle overlapping column names with `suffixes`
- Use `join()` for index-based merging
- Concatenate DataFrames vertically and horizontally with `pd.concat()`
- Perform index-based joins between DataFrames

## Prerequisites

- DataFrame fundamentals (PYT-077)
- Indexing concepts (PYT-079)
- SQL join types (INNER, LEFT, RIGHT, OUTER)
- Understanding of database normalization concepts

## Definition

**Merging** and **joining** refer to combining two or more DataFrames into a single DataFrame based on common columns or indices. Pandas provides three primary interfaces:

- **`pd.merge()`**: SQL-style merging with explicit key columns (most flexible).
- **`df.join()`**: Convenience method for index-based merging.
- **`pd.concat()`**: Stacking DataFrames along rows or columns (no explicit key matching).

```python
# SQL-style merge
pd.merge(left_df, right_df, how='inner', on='key_column')
pd.merge(left_df, right_df, how='left', left_on='key1', right_on='key2')

# Index-based join
df1.join(df2, how='inner')

# Concatenation
pd.concat([df1, df2], axis=0)  # Row-wise
pd.concat([df1, df2], axis=1)  # Column-wise
```

## Intuition

Think of merging as **looking up values from one table using keys from another table** — exactly like SQL joins:

- **Inner join**: Keep only rows where keys match in both DataFrames.
- **Left join**: Keep all rows from the left DataFrame, fill missing right-side values with NaN.
- **Right join**: Keep all rows from the right DataFrame, fill missing left-side values with NaN.
- **Outer join**: Keep all rows from both DataFrames, fill missing values with NaN.

Concatenation is simpler — it stacks DataFrames either vertically (adding rows) or horizontally (adding columns) without key-based matching.

## Why This Concept Matters

Real-world data is rarely in a single table. You'll often need to:

- Enrich a main dataset with supplementary information (e.g., adding customer names to a transactions table).
- Combine data from multiple sources (e.g., merging CSV exports from different systems).
- Build feature matrices by joining multiple tables on a common ID.
- Reconstruct normalized database schemas into flat tables for ML.

Understanding merge types and their behavior with duplicate keys, missing values, and index alignment is critical for producing correct results.

## Real World Examples

1. **Customer-Order Join**: Orders table + Customers table joined on `customer_id`.
2. **Feature Engineering**: User features from table A + behavioral features from table B + demographic features from table C, all merged on `user_id`.
3. **Data Enrichment**: Product sales + product metadata (category, price tier) merged on `product_id`.
4. **Time Series Alignment**: Trades table + market data table merged on nearest timestamp.
5. **Database Restoration**: Reconstructing a denormalized table from normalized tables via a series of joins.

## AI/ML Relevance

- **Feature Engineering**: Merging is the primary mechanism for combining features from multiple tables.
- **Data Pipelines**: ETL pipelines often involve a series of merges to build the final training dataset.
- **Time Series Modeling**: Joining lagged features back to the original DataFrame requires careful index-based merging.
- **Model Inference**: Merging predictions back to original records for evaluation.
- **Data Validation**: Merging ground truth labels with predictions on a common ID.

## Code Examples

### Example 1: Inner join

```python
import pandas as pd
import numpy as np

customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana']
})

orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104, 105],
    'customer_id': [1, 2, 2, 3, 5],
    'amount': [100, 200, 150, 300, 250]
})

# Inner join — only customers with orders
result = pd.merge(customers, orders, on='customer_id', how='inner')
print(result)
# Output:
#    customer_id     name  order_id  amount
# 0            1    Alice       101     100
# 1            2      Bob       102     200
# 2            2      Bob       103     150
# 3            3  Charlie       104     300
```

### Example 2: Left, right, and outer joins

```python
# Left join — all customers, whether they have orders or not
left = pd.merge(customers, orders, on='customer_id', how='left')
print(left)
# Output:
#    customer_id     name  order_id  amount
# 0            1    Alice     101.0   100.0
# 1            2      Bob     102.0   200.0
# 2            2      Bob     103.0   150.0
# 3            3  Charlie     104.0   300.0
# 4            4    Diana       NaN     NaN

# Right join — all orders, even if customer info missing
right = pd.merge(customers, orders, on='customer_id', how='right')
print(right)
# Output:
#    customer_id     name  order_id  amount
# 0          1.0    Alice       101     100
# 1          2.0      Bob       102     200
# 2          2.0      Bob       103     150
# 3          3.0  Charlie       104     300
# 4          5.0      NaN       105     250

# Outer join — everything from both
outer = pd.merge(customers, orders, on='customer_id', how='outer')
print(outer)
# Output:
#    customer_id     name  order_id  amount
# 0            1    Alice     101.0   100.0
# 1            2      Bob     102.0   200.0
# 2            2      Bob     103.0   150.0
# 3            3  Charlie     104.0   300.0
# 4            4    Diana       NaN     NaN
# 5            5      NaN     105.0   250.0
```

### Example 3: Merging on different key names

```python
customers = pd.DataFrame({
    'cust_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})
orders = pd.DataFrame({
    'order_id': [101, 102, 103],
    'customer_id': [1, 2, 4],
    'amount': [100, 200, 300]
})

result = pd.merge(customers, orders, left_on='cust_id', right_on='customer_id', how='left')
print(result)
# Output:
#    cust_id     name  order_id  customer_id  amount
# 0        1    Alice     101.0          1.0   100.0
# 1        2      Bob     102.0          2.0   200.0
# 2        3  Charlie       NaN          NaN     NaN
```

### Example 4: Handling overlapping column names with suffixes

```python
df1 = pd.DataFrame({'key': [1, 2, 3], 'value': ['A', 'B', 'C'], 'extra': [10, 20, 30]})
df2 = pd.DataFrame({'key': [1, 2, 4], 'value': ['X', 'Y', 'Z'], 'extra': [40, 50, 60]})

result = pd.merge(df1, df2, on='key', suffixes=('_left', '_right'))
print(result)
# Output:
#    key value_left  extra_left value_right  extra_right
# 0    1          A          10           X           40
# 1    2          B          20           Y           50
```

### Example 5: Index-based join with `.join()`

```python
df1 = pd.DataFrame({'value': [10, 20, 30]}, index=['a', 'b', 'c'])
df2 = pd.DataFrame({'other': [100, 200]}, index=['b', 'c'])

result = df1.join(df2, how='inner')
print(result)
# Output:
#    value  other
# b     20    100
# c     30    200

# Left join on index
result = df1.join(df2, how='left')
print(result)
# Output:
#    value  other
# a     10    NaN
# b     20  100.0
# c     30  200.0
```

### Example 6: Concatenation — row-wise and column-wise

```python
df_a = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df_b = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
df_c = pd.DataFrame({'C': [9, 10], 'D': [11, 12]})

# Row-wise (vertical)
result = pd.concat([df_a, df_b], axis=0)
print(result)
# Output:
#    A  B
# 0  1  3
# 1  2  4
# 0  5  7
# 1  6  8

# Column-wise (horizontal)
result = pd.concat([df_a, df_c], axis=1)
print(result)
# Output:
#    A  B   C   D
# 0  1  3   9  11
# 1  2  4  10  12

# Ignore original index
result = pd.concat([df_a, df_b], axis=0, ignore_index=True)
print(result)
# Output:
#    A  B
# 0  1  3
# 1  2  4
# 2  5  7
# 3  6  8
```

### Example 7: Merging with multiple keys

```python
df1 = pd.DataFrame({
    'year': [2023, 2023, 2024, 2024],
    'quarter': ['Q1', 'Q2', 'Q1', 'Q2'],
    'revenue': [100, 200, 150, 250]
})
df2 = pd.DataFrame({
    'year': [2023, 2023, 2024, 2024],
    'quarter': ['Q1', 'Q2', 'Q1', 'Q2'],
    'target': [110, 190, 140, 260]
})

result = pd.merge(df1, df2, on=['year', 'quarter'])
print(result)
# Output:
#    year quarter  revenue  target
# 0  2023      Q1      100     110
# 1  2023      Q2      200     190
# 2  2024      Q1      150     140
# 3  2024      Q2      250     260
```

### Example 8: Indicating source with `indicator`

```python
df1 = pd.DataFrame({'id': [1, 2, 3]})
df2 = pd.DataFrame({'id': [2, 3, 4]})

result = pd.merge(df1, df2, on='id', how='outer', indicator=True)
print(result)
# Output:
#    id      _merge
# 0   1   left_only
# 1   2        both
# 2   3        both
# 3   4  right_only
```

## Common Mistakes

1. **Forgetting that `pd.merge` defaults to an inner join.** If you want all rows from the left, specify `how='left'`.
2. **Merging on columns with different names without using `left_on`/`right_on`.** The `on` parameter only works when the key column has the same name in both DataFrames.
3. **Not handling duplicate key values.** If keys are not unique in one or both DataFrames, the merge produces a Cartesian product within each key group.
4. **Confusing `pd.concat(axis=1)` with a merge.** `concat` stacks columns by position, not by key matching.
5. **Ignoring index alignment in `.join()`.** The join is on the index, not on any column — ensure the indices align correctly.

## Interview Questions

### Beginner

1. What is the difference between an inner join and a left join?
2. How do you merge two DataFrames on a common column?
3. What does `how='outer'` do in a merge?
4. How is `pd.concat` different from `pd.merge`?
5. What does the `suffixes` parameter do?

### Intermediate

1. How would you merge two DataFrames with different column names for the key?
2. What happens when the merge key has duplicate values in both DataFrames?
3. How do you use the `indicator` parameter to debug a merge?
4. When would you use `.join()` instead of `pd.merge()`?
5. How do you concatenate multiple DataFrames while preserving their source identity?

### Advanced

1. Compare the performance of `pd.merge` vs `df.join` vs manual index alignment for 10M-row DataFrames.
2. Explain how Pandas handles a merge where both DataFrames have non-unique indexes — what is the computational complexity?
3. Implement a merge-like operation that performs fuzzy matching on string keys (e.g., Levenshtein distance) without producing a Cartesian product.

## Practice Problems

### Easy

1. Merge two DataFrames on `'id'` using an inner join.
2. Left-join a small lookup table to a main DataFrame.
3. Concatenate three DataFrames vertically.
4. Use `indicator=True` on a merge to see which rows came from which source.
5. Merge two DataFrames on index using `.join()`.

### Medium

1. Merge two DataFrames on different key column names, then rename the result columns to avoid suffixes.
2. Given a sales table and a targets table, merge them on `year` and `month`, then compute the percentage difference between actual and target.
3. Perform a self-join on an employee table to find manager names for each employee (using `left_on` and `right_on`).
4. Merge a DataFrame with itself on a time-based key to create lagged features (e.g., previous month's value).
5. Use `pd.concat` with `keys` parameter to create a MultiIndex on the result.

### Hard

1. Implement a function that performs a rolling merge: for each row in DataFrame A, find the nearest matching row in DataFrame B by a timestamp within a tolerance window.
2. Given a list of DataFrames with partially overlapping columns and rows, implement a recursive merge that produces the union of all columns and rows, filling missing values with NaN.
3. Build a custom `ConditionalMerge` class that supports merging on predicates like `A.value BETWEEN B.low AND B.high` without exploding to a full Cartesian product.

## Solutions

```python
# E1
pd.merge(df1, df2, on='id')

# E2
pd.merge(main_df, lookup_df, on='key', how='left')

# E3
pd.concat([df1, df2, df3], axis=0, ignore_index=True)

# E4
pd.merge(df1, df2, on='id', how='outer', indicator=True)

# E5
df1.join(df2, how='left')

# M1
result = pd.merge(
    df1, df2,
    left_on='customer_id',
    right_on='cust_id',
    suffixes=('', '_drop')
)
result.drop(columns=['cust_id'], inplace=True)

# M2
merged = pd.merge(sales, targets, on=['year', 'month'])
merged['pct_diff'] = (merged['actual'] - merged['target']) / merged['target'] * 100

# M3
pd.merge(employees, employees, left_on='manager_id', right_on='emp_id',
         suffixes=('', '_manager'))

# M4
df['prev_month'] = df['month'] - 1
merged = pd.merge(df, df, left_on=['id', 'prev_month'],
                  right_on=['id', 'month'], suffixes=('', '_lag'))

# M5
pd.concat([df1, df2], keys=['source1', 'source2'])
```

## Related Concepts

- SQL JOIN operations (INNER, LEFT, RIGHT, OUTER)
- Database normalization and denormalization
- Relational database design
- Set theory (union, intersection)

## Next Concepts

- Pivot tables for reshaping merged data
- Time series operations on merged datasets
- Advanced indexing with MultiIndex from concatenation

## Summary

Merging and joining are fundamental data combination operations in Pandas. `pd.merge()` provides SQL-style joins (inner, left, right, outer) on common columns with support for different key names, overlapping column suffixes, and merge indicators. `df.join()` offers a convenient interface for index-based merging. `pd.concat()` stacks DataFrames vertically or horizontally. Choosing the correct join type and handling duplicate keys are essential for correct data integration.

## Key Takeaways

- `pd.merge()` with `how` parameter (inner/left/right/outer) — primarily for column-based joins.
- Use `on` for same-named keys, `left_on`/`right_on` for different names.
- `suffixes` resolves overlapping column names from both DataFrames.
- `indicator=True` shows the source of each row in the merge.
- `df.join()` is syntactic sugar for index-based joins.
- `pd.concat(axis=0)` stacks rows; `pd.concat(axis=1)` stacks columns.
- Duplicate keys cause Cartesian products — check uniqueness before merging.
- Merge defaults to `how='inner'`, which discards non-matching rows.
