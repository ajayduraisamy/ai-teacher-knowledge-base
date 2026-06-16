# Concept: Pivot Tables

## Concept ID

PYT-083

## Difficulty

INTERMEDIATE

## Domain

Python

## Module

Pandas

## Learning Objectives

- Create pivot tables using `pivot_table()` with index, columns, values, and aggfunc
- Add subtotals with the `margins` parameter
- Unpivot data using `melt()`
- Reshape DataFrames with `stack()` and `unstack()`
- Distinguish between `pivot()` and `pivot_table()` and choose the right one

## Prerequisites

- DataFrame fundamentals (PYT-077)
- GroupBy operations (PYT-081)
- MultiIndex concepts
- Aggregation functions (mean, sum, count)

## Definition

A **pivot table** is a data summarization tool that aggregates data and displays it in a matrix format. Given a DataFrame, it:

- Groups data by unique values in **index** rows
- Creates columns from unique values in **columns**
- Aggregates **values** using a function (default: mean)

```python
pd.pivot_table(data, values=None, index=None, columns=None,
               aggfunc='mean', fill_value=None, margins=False, margins_name='All')

# Unpivot
pd.melt(frame, id_vars=None, value_vars=None, var_name=None, value_name='value')

# Reshape
df.stack()
df.unstack()

# Simple pivot (no aggregation)
df.pivot(index='row', columns='col', values='val')
```

## Intuition

Imagine you have a long-format table with columns: `Product`, `Month`, `Sales`. A pivot table reshapes this into a matrix where:

- Rows = unique `Product` values
- Columns = unique `Month` values
- Cells = aggregated `Sales` (e.g., sum or mean)

This is the "pivot" — rotating the data so that categorical values in one column become column headers.

The inverse operation, **melting** (or unpivoting), takes a wide matrix and collapses it into a long table, making it suitable for analysis and plotting.

## Why This Concept Matters

Pivot tables are essential for:

- **Exploratory Analysis**: Quickly summarize relationships between categorical variables.
- **Reporting**: Create cross-tabulations for business dashboards.
- **Data Reshaping**: Convert between long and wide formats — a frequent need in data cleaning.
- **Feature Engineering**: Create interaction features between categorical variables.
- **Visualization**: Prepare data for heatmaps, bar charts, and grouped plots.

Nearly every data analysis task benefits from the ability to reshape data flexibly.

## Real World Examples

1. **Sales Dashboard**: Monthly sales by product category (rows = product, columns = month, values = revenue).
2. **Survey Analysis**: Average satisfaction score by customer segment and region.
3. **A/B Testing**: Conversion rate by variant and device type.
4. **Educational Data**: Average test score by student grade level and subject.
5. **HR Analytics**: Headcount by department and job level.

## AI/ML Relevance

- **Feature Encoding**: Creating pivot tables of aggregated statistics (mean target encoding).
- **Interaction Features**: Product of categorical features as new columns.
- **Time Series**: Reshaping from long to wide format for sequence models.
- **Data Transformation**: Melting wide sensor data into long format for analysis.
- **Evaluation**: Building confusion matrices (pivot of actual vs predicted).

## Code Examples

### Example 1: Basic pivot table

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'Product': ['A', 'A', 'B', 'B', 'A', 'B'],
    'Month': ['Jan', 'Feb', 'Jan', 'Feb', 'Mar', 'Mar'],
    'Sales': [100, 200, 150, 250, 300, 350],
    'Region': ['North', 'North', 'South', 'South', 'North', 'South']
})

# Default: mean aggregation
pivot = pd.pivot_table(df, values='Sales', index='Product', columns='Month')
print(pivot)
# Output:
# Month     Feb    Jan    Mar
# Product
# A        200.0  100.0  300.0
# B        250.0  150.0  350.0
```

### Example 2: Pivot table with sum aggregation and margins

```python
pivot = pd.pivot_table(df, values='Sales', index='Product', columns='Month',
                       aggfunc='sum', margins=True, margins_name='Total')
print(pivot)
# Output:
# Month     Feb  Jan  Mar  Total
# Product
# A         200  100  300    600
# B         250  150  350    750
# Total     450  250  650   1350
```

### Example 3: Multiple values and multiple aggregation functions

```python
df['Quantity'] = [10, 15, 8, 12, 20, 18]

pivot = pd.pivot_table(df, values=['Sales', 'Quantity'], index='Product',
                       columns='Month', aggfunc={'Sales': 'sum', 'Quantity': 'mean'})
print(pivot)
# Output:
#         Quantity             Sales
# Month        Feb Jan Mar   Feb Jan Mar
# Product
# A            15.0  10  20   200 100 300
# B            12.0   8  18   250 150 350
```

### Example 4: MultiIndex rows and columns

```python
pivot = pd.pivot_table(df, values='Sales', index=['Region', 'Product'],
                       columns='Month', aggfunc='sum', fill_value=0)
print(pivot)
# Output:
# Month             Feb  Jan  Mar
# Region Product
# North  A          200  100  300
# South  B          250  150  350
```

### Example 5: `melt()` — unpivoting (wide to long)

```python
wide = pd.DataFrame({
    'Product': ['A', 'B', 'C'],
    'Jan': [100, 150, 200],
    'Feb': [200, 250, 300],
    'Mar': [300, 350, 400]
})
print(wide)
# Output:
#   Product  Jan  Feb  Mar
# 0       A  100  200  300
# 1       B  150  250  350
# 2       C  200  300  400

long = pd.melt(wide, id_vars=['Product'], var_name='Month', value_name='Sales')
print(long)
# Output:
#   Product Month  Sales
# 0       A   Jan    100
# 1       B   Jan    150
# 2       C   Jan    200
# 3       A   Feb    200
# 4       B   Feb    250
# 5       C   Feb    300
# 6       A   Mar    300
# 7       B   Mar    350
# 8       C   Mar    400
```

### Example 6: `stack()` and `unstack()`

```python
df = pd.DataFrame({
    'Product': ['A', 'A', 'B', 'B'],
    'Month': ['Jan', 'Feb', 'Jan', 'Feb'],
    'Sales': [100, 200, 150, 250]
})

# Set MultiIndex
df_mi = df.set_index(['Product', 'Month'])
print(df_mi)
# Output:
#              Sales
# Product Month
# A       Jan     100
#         Feb     200
# B       Jan     150
#         Feb     250

# Unstack: move inner index level to columns
unstacked = df_mi.unstack(level='Month')
print(unstacked)
# Output:
#         Sales
# Month     Feb  Jan
# Product
# A         200  100
# B         250  150

# Stack: collapse columns back to rows
stacked = unstacked.stack(level='Month')
print(stacked)
# Output:
#              Sales
# Product Month
# A       Feb     200
#         Jan     100
# B       Feb     250
#         Jan     150
```

### Example 7: `pivot()` vs `pivot_table()`

```python
# pivot() requires unique index-column pairs (no aggregation)
df_unique = pd.DataFrame({
    'Product': ['A', 'A', 'B', 'B'],
    'Month': ['Jan', 'Feb', 'Jan', 'Feb'],
    'Sales': [100, 200, 150, 250]
})

pivot_simple = df_unique.pivot(index='Product', columns='Month', values='Sales')
print(pivot_simple)
# Output:
# Month     Feb  Jan
# Product
# A         200  100
# B         250  150

# pivot_table() handles duplicates with aggregation
df_dup = pd.DataFrame({
    'Product': ['A', 'A', 'A', 'B'],
    'Month': ['Jan', 'Jan', 'Feb', 'Jan'],
    'Sales': [100, 150, 200, 250]
})

# pivot() would raise ValueError
pivot_agg = pd.pivot_table(df_dup, values='Sales', index='Product',
                           columns='Month', aggfunc='mean')
print(pivot_agg)
# Output:
# Month     Feb    Jan
# Product
# A        200.0  125.0
# B          NaN  250.0
```

### Example 8: Pivot table with custom aggfunc

```python
pivot = pd.pivot_table(df, values='Sales', index='Product', columns='Month',
                       aggfunc=lambda x: x.max() - x.min() if len(x) > 1 else 0)
print(pivot)
# Output:
# Month     Feb  Jan  Mar
# Product
# A           0    0    0
# B           0    0    0

# More practical: count
pivot_count = pd.pivot_table(df, values='Sales', index='Region',
                             columns='Month', aggfunc='count')
print(pivot_count)
# Output:
# Month   Feb  Jan  Mar
# Region
# North   1.0  1.0  1.0
# South   1.0  1.0  1.0
```

## Common Mistakes

1. **Using `pivot()` when data has duplicate index-column pairs.** `pivot()` requires unique pairs; use `pivot_table()` with an aggregation function for duplicates.
2. **Forgetting `fill_value`.** Missing combinations appear as NaN. Use `fill_value=0` for sparse data.
3. **Confusing `stack()` and `melt()`.** `stack()` works on MultiIndex columns; `melt()` works on flat column lists.
4. **Not setting `id_vars` in `melt()`.** Without it, all columns are treated as value columns, and the result may not be what you intended.
5. **Using `margins=True` on large data.** It adds a "Total" row/column that can be expensive to compute and may not be meaningful for all aggregation functions.

## Interview Questions

### Beginner

1. What is a pivot table and what problem does it solve?
2. How do you create a pivot table with `pivot_table()`?
3. What does the `margins` parameter do?
4. What is the difference between `pivot()` and `pivot_table()`?
5. How does `melt()` change the shape of a DataFrame?

### Intermediate

1. How would you create a pivot table that shows both sum and count in the same table?
2. What is the difference between `stack()`/`unstack()` and `pivot_table()`?
3. How do you handle missing values in a pivot table?
4. How would you unpivot a DataFrame that has multiple value columns?
5. How can you use multiple aggregation functions in a single pivot table?

### Advanced

1. Compare `stack()`/`unstack()` with `pivot_table()` — when would you choose one over the other for a given reshaping task?
2. Implement a pivot table with hierarchical row and column indices where the value column itself is computed (e.g., ratio of two pivoted values).
3. Design a function that automatically detects whether a DataFrame is in long or wide format and converts it to the opposite format based on column name heuristics.

## Practice Problems

### Easy

1. Create a pivot table of mean sales by `'Product'` (rows) and `'Quarter'` (columns).
2. Add margins (totals) to the above pivot table.
3. Use `melt()` to convert a wide DataFrame with columns `'A'`, `'B'`, `'C'` into long format.
4. Use `pivot()` on a DataFrame with unique index-column combinations.
5. Fill NaN values in a pivot table with 0.

### Medium

1. Create a pivot table with multiple aggregation functions (sum, mean, count) and multiple value columns.
2. Use `stack()` and `unstack()` to rotate a MultiIndex DataFrame and explain the transformation.
3. Take a long-format DataFrame, convert it to wide format using `pivot_table()`, then convert it back to long using `melt()`, verifying the data is preserved.
4. Build a pivot table with two levels in both the index and columns.
5. Create a custom aggregation function for `pivot_table()` that computes the 90th percentile.

### Hard

1. Implement a function that generates a "difference" pivot table: starting from two DataFrames (e.g., current month vs previous month), produce a single pivot table showing the absolute and percentage difference side by side.
2. Build a dynamic pivot engine that takes a DataFrame and a configuration dict (specifying which columns are index, columns, values, and aggfunc) and returns the pivot table, with automatic detection of numeric vs categorical columns.
3. Design a class `PivotCache` that memoizes pivot table computations and supports incremental updates when new rows are added to the source DataFrame.

## Solutions

```python
# E1
pd.pivot_table(df, values='Sales', index='Product', columns='Quarter')

# E2
pd.pivot_table(df, values='Sales', index='Product', columns='Quarter', margins=True)

# E3
pd.melt(df, id_vars=['id'], var_name='variable', value_name='value')

# E4
df.pivot(index='row', columns='col', values='val')

# E5
pd.pivot_table(df, ..., fill_value=0)

# M1
pd.pivot_table(df, values=['Sales', 'Qty'], index='Product', columns='Month',
               aggfunc={'Sales': 'sum', 'Qty': ['mean', 'count']})

# M2
multi = df.set_index(['Product', 'Month'])
wide = multi.unstack(level='Month')
long = wide.stack(level='Month')

# M3
wide = pd.pivot_table(long_df, values='val', index='id', columns='var')
long_again = pd.melt(wide.reset_index(), id_vars='id', var_name='var', value_name='val')

# M4
pd.pivot_table(df, values='Sales', index=['Region', 'Product'],
               columns=['Month', 'Year'])

# M5
def p90(x):
    return x.quantile(0.9)

pd.pivot_table(df, values='Sales', index='Product', columns='Month', aggfunc=p90)
```

## Related Concepts

- SQL `PIVOT` and `UNPIVOT` operations
- GroupBy (pivot tables are a specialized GroupBy)
- MultiIndex DataFrames
- Data normalization (long vs wide format debate)

## Next Concepts

- Time series resampling (a time-based pivot)
- Advanced data transformation pipelines
- Visualization of pivot table results

## Summary

Pivot tables in Pandas (`pivot_table()`) reshape and aggregate data from long to wide format. The inverse operation, melting (`melt()`), reshapes wide to long. `stack()` and `unstack()` provide alternative reshaping for MultiIndex DataFrames. `pivot()` is a simpler variant for unique data without aggregation. Pivot tables are essential for EDA, reporting, and preparing data for visualization and ML.

## Key Takeaways

- `pivot_table()` groups by index, creates columns, and aggregates values — handles duplicates.
- `pivot()` requires unique index-column pairs (no aggregation).
- `margins=True` adds row/column totals.
- `melt()` unpivots wide tables to long format for analysis.
- `stack()` moves column levels to row index; `unstack()` does the reverse.
- Use `fill_value` to replace NaN after pivoting.
- Multiple aggregation functions and value columns are supported.
- Pivot tables are the Pandas equivalent of Excel pivot tables and SQL PIVOT.
