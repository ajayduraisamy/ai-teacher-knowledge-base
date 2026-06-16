# Concept: Array Indexing and Slicing

## Concept ID

PYT-067

## Difficulty

Intermediate

## Domain

Python

## Module

NumPy

## Learning Objectives

- Index individual elements in 1D and 2D arrays
- Slice subarrays using start:stop:step notation
- Use fancy indexing with integer arrays
- Apply boolean masks to filter data
- Use `np.where()` for conditional selection
- Locate non-zero elements with `np.nonzero()`

## Prerequisites

- NumPy array creation (PYT-066)
- Python slicing syntax `[start:stop:step]`
- Boolean logic and comparison operators

## Definition

Array indexing and slicing refers to the techniques used to access, extract, and modify subsets of NumPy array data. NumPy supports basic indexing (integers), slicing (colon notation), fancy indexing (integer arrays), and boolean indexing (masking). These methods enable fast, memory-efficient, and expressive data selection without Python loops.

## Intuition

Think of an array as a grid where each cell has a coordinate. Basic indexing picks a single cell by its coordinates. Slicing extracts a rectangular block. Fancy indexing picks arbitrary rows/columns by listing their indices. Boolean masking selects cells that satisfy a condition, like using a stencil to paint only certain areas. All these operations return views (not copies) when using basic slicing, making them extremely fast.

## Why This Concept Matters

Indexing and slicing are the most frequent operations in data analysis. Every time you select a feature column, filter rows by a condition, or extract a region of interest from an image, you use these techniques. Mastery of NumPy indexing translates directly to proficiency in pandas, where similar patterns apply.

## Real World Examples

1. **Image Cropping:** Extract a region of interest from an image array with `img[50:200, 100:300]`.
2. **Time Series Filtering:** Select all measurements above a threshold: `data[data > threshold]`.
3. **Feature Selection:** In a dataset `X` of shape `(n_samples, n_features)`, select specific feature columns: `X[:, [0, 3, 5]]`.
4. **Outlier Removal:** Remove rows where any feature exceeds 3 standard deviations: `data[np.all(np.abs(z_scores) < 3, axis=1)]`.
5. **Data Augmentation:** Randomly select and flip image patches for training: `batch[indices, :, :, :]`.

## AI/ML Relevance

Indexing is used at every stage of ML pipelines: train/test splitting (row selection), feature engineering (column selection), batch generation (random row selection), data cleaning (boolean masking for outliers), and evaluation (selecting predictions for specific classes). Understanding views vs copies is critical to avoid memory issues in large datasets.

## Code Examples

### Example 1: Basic Indexing in 1D and 2D

```python
import numpy as np

# 1D array
arr1d = np.array([10, 20, 30, 40, 50])
print("arr1d:", arr1d)
print("arr1d[0]:", arr1d[0])
print("arr1d[-1]:", arr1d[-1])
print("arr1d[2]:", arr1d[2])

# 2D array
arr2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])
print("\narr2d:\n", arr2d)
print("arr2d[0, 0]:", arr2d[0, 0])
print("arr2d[1, 2]:", arr2d[1, 2])
print("arr2d[-1, -1]:", arr2d[-1, -1])
```
```
# Output:
# arr1d: [10 20 30 40 50]
# arr1d[0]: 10
# arr1d[-1]: 50
# arr1d[2]: 30
# 
# arr2d:
#  [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
# arr2d[0, 0]: 1
# arr2d[1, 2]: 7
# arr2d[-1, -1]: 12
```

### Example 2: Slicing

```python
import numpy as np

arr = np.array([[0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19]])

# First two rows, all columns
print("arr[:2, :]:\n", arr[:2, :])

# All rows, columns 1 to 3
print("\narr[:, 1:4]:\n", arr[:, 1:4])

# Rows 1 to 3, columns 2 to 4
print("\narr[1:3, 2:5]:\n", arr[1:3, 2:5])

# Step slicing — every other row
print("\narr[::2, :]:\n", arr[::2, :])

# Reverse rows
print("\narr[::-1, :]:\n", arr[::-1, :])
```
```
# Output:
# arr[:2, :]:
#  [[0 1 2 3 4]
#  [5 6 7 8 9]]
# 
# arr[:, 1:4]:
#  [[ 1  2  3]
#  [ 6  7  8]
#  [11 12 13]
#  [16 17 18]]
# 
# arr[1:3, 2:5]:
#  [[ 7  8  9]
#  [12 13 14]]
# 
# arr[::2, :]:
#  [[ 0  1  2  3  4]
#  [10 11 12 13 14]]
# 
# arr[::-1, :]:
#  [[15 16 17 18 19]
#  [10 11 12 13 14]
#  [ 5  6  7  8  9]
#  [ 0  1  2  3  4]]
```

### Example 3: Fancy Indexing

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50, 60, 70, 80])

# Select specific indices
indices = [0, 3, 5]
print("arr[indices]:", arr[indices])

# 2D fancy indexing
arr2d = np.arange(25).reshape(5, 5)
print("\narr2d:\n", arr2d)

# Select specific rows
print("\narr2d[[0, 2, 4]]:\n", arr2d[[0, 2, 4]])

# Select specific rows and columns (mesh)
rows = [0, 1, 2]
cols = [0, 2, 4]
print("\narr2d[rows, cols]:", arr2d[rows, cols])

# Select using index grid
print("\narr2d[np.ix_(rows, cols)]:\n", arr2d[np.ix_(rows, cols)])
```
```
# Output:
# arr[indices]: [10 40 60]
# 
# arr2d:
#  [[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
#  [10 11 12 13 14]
#  [15 16 17 18 19]
#  [20 21 22 23 24]]
# 
# arr2d[[0, 2, 4]]:
#  [[ 0  1  2  3  4]
#  [10 11 12 13 14]
#  [20 21 22 23 24]]
# 
# arr2d[rows, cols]: [0 6 12]
# 
# arr2d[np.ix_(rows, cols)]:
#  [[ 0  2  4]
#  [ 5  7  9]
#  [10 12 14]]
```

### Example 4: Boolean Masking

```python
import numpy as np

arr = np.array([12, 25, 8, 40, 3, 17, 30])

# Mask for values > 15
mask = arr > 15
print("Mask:", mask)
print("arr[mask]:", arr[mask])

# Multiple conditions with &, |, ~
arr2d = np.arange(12).reshape(3, 4)
print("\narr2d:\n", arr2d)

mask = (arr2d % 2 == 0) & (arr2d > 5)
print("\nEven and >5:\n", arr2d[mask])

# Replace values conditionally
arr_copy = arr.copy()
arr_copy[arr_copy < 10] = -1
print("\nReplace <10 with -1:", arr_copy)

# Boolean indexing on specific axis
arr2d_bool = arr2d[arr2d[:, 0] > 2]  # rows where first col > 2
print("\nRows where first col > 2:\n", arr2d_bool)
```
```
# Output:
# Mask: [False  True False  True False  True  True]
# arr[mask]: [25 40 17 30]
# 
# arr2d:
#  [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
# 
# Even and >5:
#  [ 6  8 10]
# 
# Replace <10 with -1: [-1 25 -1 40 -1 17 30]
# 
# Rows where first col > 2:
#  [[ 4  5  6  7]
#  [ 8  9 10 11]]
```

### Example 5: np.where and np.nonzero

```python
import numpy as np

arr = np.array([10, 0, 25, 0, 30, 45, 0])

# np.where returns indices where condition is True
indices = np.where(arr > 0)
print("Indices where arr > 0:", indices)
print("Values:", arr[indices])

# np.where with x and y (ternary)
result = np.where(arr > 0, arr, -1)
print("Where positive keep, else -1:", result)

# np.nonzero (same as where for non-zero)
nonzero_indices = np.nonzero(arr)
print("Non-zero indices:", nonzero_indices)
print("Non-zero values:", arr[nonzero_indices])

# 2D example
arr2d = np.array([[0, 1, 0],
                   [3, 0, 4],
                   [0, 5, 0]])
rows, cols = np.nonzero(arr2d)
print("\n2D non-zero positions:")
for r, c in zip(rows, cols):
    print(f"  ({r}, {c}) -> {arr2d[r, c]}")
```
```
# Output:
# Indices where arr > 0: (array([0, 2, 4, 5]),)
# Values: [10 25 30 45]
# Where positive keep, else -1: [10 -1 25 -1 30 45 -1]
# Non-zero indices: (array([0, 2, 4, 5]),)
# Non-zero values: [10 25 30 45]
# 
# 2D non-zero positions:
#   (0, 1) -> 1
#   (1, 0) -> 3
#   (1, 2) -> 4
#   (2, 1) -> 5
```

### Example 6: Views vs Copies

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Basic slicing returns a VIEW
view = arr[1:4]
view[0] = 99
print("Original after view modification:", arr)

# Fancy indexing returns a COPY
copy = arr[[0, 2, 4]]
copy[0] = -100
print("Original after copy modification:", arr)

# Use .copy() to force a copy from a slice
safe_copy = arr[1:4].copy()
safe_copy[0] = 77
print("Original after safe_copy modification:", arr)
```
```
# Output:
# Original after view modification: [ 1 99  3  4  5]
# Original after copy modification: [ 1 99  3  4  5]
# Original after safe_copy modification: [ 1 99  3  4  5]
```

## Common Mistakes

1. **Confusing Views with Copies:** Slicing returns a view, not a copy. Modifying a slice modifies the original array. Use `.copy()` to create an independent copy.

2. **Chained Indexing:** `arr[1:3][0]` works but is harder to read and can cause unexpected behavior with assignments. Use `arr[1:3, 0]` instead.

3. **Forgetting Parentheses in Boolean Conditions:** `arr[arr > 5 & arr < 10]` is invalid due to operator precedence. Use `arr[(arr > 5) & (arr < 10)]`.

4. **Using `and`/`or` Instead of `&`/`|`:** Python's `and` and `or` don't work element-wise on arrays. Use `&`, `|`, and `~` for element-wise boolean operations.

5. **Indexing Out of Bounds:** Unlike lists, NumPy raises an `IndexError` for out-of-bounds access. Always check shapes before indexing.

6. **Fancy Indexing Returns a Copy:** Fancy indexing (using integer arrays) always returns a copy. Assigning to a fancy index modifies the original but reads back as a copy.

7. **Ellipsis Confusion:** `arr[...]` is valid and selects all dimensions, but forgetting it in higher-dimensional slicing leads to errors.

## Interview Questions

### Beginner

1. **Q:** How do you select the third element of a 1D array?
   **A:** `arr[2]` using zero-based indexing.

2. **Q:** How do you extract the first two rows and first three columns of a 2D array?
   **A:** `arr[:2, :3]`.

3. **Q:** What does `arr[::-1]` do?
   **A:** It reverses the array along the first axis.

4. **Q:** How do you select all elements greater than 5?
   **A:** `arr[arr > 5]`.

5. **Q:** What is the difference between `arr[1]` and `arr[1, :]` for a 2D array?
   **A:** Both select row 1. `arr[1]` simplifies to a 1D array while `arr[1, :]` is explicit. They produce the same result.

### Intermediate

1. **Q:** What is the difference between basic slicing and fancy indexing in terms of memory?
   **A:** Basic slicing returns a view (shares memory with the original). Fancy indexing returns a copy (allocates new memory).

2. **Q:** How do you use `np.where` to replace values conditionally?
   **A:** `np.where(condition, x, y)` returns `x` where condition is True, `y` where False. For example: `np.where(arr > 0, arr, 0)` replaces negatives with zero.

3. **Q:** What does `np.nonzero` return and how is it different from `np.where`?
   **A:** `np.nonzero(a)` returns a tuple of arrays containing the indices of non-zero elements. `np.where(condition)` is equivalent to `np.nonzero(condition)`, but `np.where` also supports conditional replacement with three arguments.

4. **Q:** Explain what `arr[np.ix_([0, 2], [1, 3])]` does.
   **A:** `np.ix_` creates an open mesh from index arrays, selecting a rectangular submatrix. It selects rows 0 and 2 and columns 1 and 3.

5. **Q:** How do you select all rows of a 2D array where the first column is greater than the mean?
   **A:** `arr[arr[:, 0] > arr[:, 0].mean()]`.

### Advanced

1. **Q:** How does NumPy handle indexing with a tuple containing both slices and arrays (mixed indexing)?
   **A:** NumPy combines slice and array indexing using advanced indexing rules. The array indices are broadcast against each other. The slice dimensions appear in the output in the same position as the original, while array indices produce dimensions at the front or back depending on whether they are interspersed.

2. **Q:** What is the difference between `arr[arr > 5] = 0` and `arr[arr > 5].copy() = 0`?
   **A:** The first modifies elements in-place (assigns 0 to elements > 5). The second creates a copy, assigns 0 to the copy, and discards it — the original is unchanged.

3. **Q:** Explain the concept of "fancy indexing" in the context of integer array indexing and boolean indexing. When does each return a view vs a copy?
   **A:** Fancy indexing (integer arrays or boolean arrays) always returns a copy, never a view. Basic slicing (integers and slices) returns a view. Boolean indexing is always a copy because the selected elements are not necessarily contiguous in memory.

## Practice Problems

### Easy

1. Create a 1D array `[10, 20, 30, 40, 50, 60, 70, 80]` and extract elements at positions 1, 3, and 5.

2. From a 5x5 identity matrix, extract the first 3 rows and first 3 columns.

3. Create an array from 0 to 9 and reverse it using slicing.

4. From `arr = np.array([[1,2],[3,4],[5,6]])`, extract the second column.

5. Replace all negative values in `np.array([1, -2, 3, -4, 5])` with 0 using boolean indexing.

### Medium

1. Create a 6x6 array of integers 0 to 35. Extract a 2x2 block from the center.

2. From a 10x10 array, select rows where the sum of the row is greater than 50.

3. Using `np.where`, create an array where positive values are squared and negative values are set to 0.

4. Given `arr = np.random.randn(5, 5)`, replace all values more than 2 standard deviations from the mean with the mean.

5. Use fancy indexing to extract the diagonal elements of a 4x4 array in a single line.

### Hard

1. Implement a function that uses boolean indexing to create a mask for all local maxima (elements greater than both neighbors) in a 1D array.

2. Given a 100x5 dataset, remove all rows that have any NaN values without using loops.

3. Create a 3D array (4, 4, 3) and use mixed indexing to select: all rows of the first two "layers", only columns 1 and 3, and only the second channel (index 1 of the last dimension).

## Solutions

### Easy Solutions

```python
# 1
arr = np.array([10, 20, 30, 40, 50, 60, 70, 80])
result = arr[[1, 3, 5]]
print(result)  # [20 40 60]

# 2
I = np.eye(5)
result = I[:3, :3]
print(result)

# 3
arr = np.arange(10)
reversed_arr = arr[::-1]
print(reversed_arr)

# 4
arr = np.array([[1,2],[3,4],[5,6]])
col2 = arr[:, 1]
print(col2)

# 5
arr = np.array([1, -2, 3, -4, 5])
arr[arr < 0] = 0
print(arr)
```

### Medium Solutions

```python
# 1
arr = np.arange(36).reshape(6, 6)
center = arr[2:4, 2:4]
print(center)

# 2
arr = np.random.randint(0, 10, size=(10, 10))
row_sums = arr.sum(axis=1)
selected = arr[row_sums > 50]
print(selected.shape)

# 3
arr = np.array([-3, -1, 0, 2, 4])
result = np.where(arr > 0, arr**2, 0)
print(result)

# 4
arr = np.random.randn(5, 5)
mean, std = arr.mean(), arr.std()
mask = np.abs(arr - mean) > 2 * std
arr[mask] = mean
print(arr)

# 5
arr = np.arange(16).reshape(4, 4)
diag = arr[range(4), range(4)]
print(diag)
```

### Hard Solutions

```python
# 1 Local maxima
def local_maxima(arr):
    return (arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:])

arr = np.array([1, 3, 2, 5, 4, 6, 1])
mask = np.zeros_like(arr, dtype=bool)
mask[1:-1] = local_maxima(arr)
print("Local maxima at:", np.where(mask)[0], "values:", arr[mask])

# 2 Remove rows with NaN
data = np.random.rand(100, 5)
data[np.random.choice(100, 5), np.random.choice(5, 5)] = np.nan
clean = data[~np.any(np.isnan(data), axis=1)]
print(f"Original: {data.shape}, Clean: {clean.shape}")

# 3 Mixed indexing
arr3d = np.arange(4*4*3).reshape(4, 4, 3)
result = arr3d[:2, [1, 3], 1]
print(result.shape)
print(result)
```

## Related Concepts

- Python list slicing syntax
- NumPy array creation (PYT-066)
- pandas `.loc` and `.iloc`

## Next Concepts

- Array operations and broadcasting (PYT-068)
- Reshaping arrays (PYT-072)
- Concatenation (PYT-073)

## Summary

NumPy provides four main indexing methods: basic indexing with integers, slicing with colon notation, fancy indexing with integer arrays, and boolean masking with conditional arrays. Basic slicing returns views (memory-efficient), while fancy and boolean indexing return copies. `np.where()` enables conditional selection and replacement, and `np.nonzero()` locates non-zero elements. Understanding these techniques is fundamental to efficient data manipulation in Python.

## Key Takeaways

- Basic slicing (`arr[1:3]`) returns a view — modifying it changes the original
- Fancy indexing (`arr[[0, 2]]`) returns a copy — independent from the original
- Boolean masking (`arr[arr > 5]`) is the most expressive filtering method
- Use `np.where(condition, x, y)` for conditional replacement
- Use `&`, `|`, `~` instead of `and`, `or`, `not` for element-wise boolean ops
- Always parenthesize boolean conditions: `(arr > 5) & (arr < 10)`
