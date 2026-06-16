# Concept: Concatenation and Splitting

## Concept ID

PYT-073

## Difficulty

Intermediate

## Domain

Python

## Module

NumPy

## Learning Objectives

- Join arrays along different axes with `np.concatenate`
- Stack arrays vertically, horizontally, and depth-wise with `vstack`, `hstack`, `dstack`
- Stack 1D arrays as columns or rows with `column_stack` and `row_stack`
- Split arrays with `np.split`, `np.array_split`, `hsplit`, and `vsplit`
- Understand the differences between concatenation functions
- Apply these operations for dataset construction and augmentation

## Prerequisites

- NumPy array creation (PYT-066)
- Array shape and dimensions (PYT-066)
- Reshaping (PYT-072)

## Definition

Concatenation is the process of joining two or more arrays along an existing axis. Stacking is a specialized form of concatenation that creates new axes. Splitting is the inverse operation — dividing an array into multiple sub-arrays along an axis. NumPy provides a family of functions for these operations, each optimized for common use cases.

## Intuition

Think of concatenation as gluing arrays together along a specific edge. `vstack` stacks arrays vertically (like stacking blocks on top of each other — rows increase). `hstack` stacks horizontally (like putting tables side by side — columns increase). `dstack` stacks depth-wise (like stacking layers in a 3D volume). Splitting is the reverse: cutting an array into pieces along a given axis.

## Why This Concept Matters

Real-world data comes from multiple sources and must be combined. Training datasets are built by stacking features and labels. Data augmentation creates new samples by transforming and concatenating. Batch processing requires splitting data into mini-batches. Understanding concatenation and splitting is essential for data pipeline construction.

## Real World Examples

1. **Adding Bias Term:** `np.hstack([X, np.ones((n, 1))])` to prepend a column of ones for the bias term in linear regression.
2. **Train/Test Splitting:** Split a dataset into training and testing subsets along the row axis.
3. **Batch Processing:** Split a large dataset into mini-batches for stochastic gradient descent.
4. **Feature Engineering:** Concatenate original features with polynomial or interaction features.
5. **Image Augmentation:** Create a batch of augmented images by stacking original and transformed versions.

## AI/ML Relevance

Data loading pipelines use splitting to create mini-batches. Model ensembles concatenate predictions from multiple models. Feature engineering stacks new features alongside existing ones. CV pipelines stack channels (RGB + depth). Sequence models split time series into windows. Cross-validation requires splitting data into folds.

## Code Examples

### Example 1: np.concatenate

```python
import numpy as np

a = np.array([[1, 2],
              [3, 4]])
b = np.array([[5, 6],
              [7, 8]])

# Concatenate along axis=0 (rows)
c0 = np.concatenate([a, b], axis=0)
print("concat axis=0:\n", c0)
print("Shape:", c0.shape)

# Concatenate along axis=1 (columns)
c1 = np.concatenate([a, b], axis=1)
print("\nconcat axis=1:\n", c1)
print("Shape:", c1.shape)

# Multiple arrays
c_multi = np.concatenate([a, b, a], axis=0)
print("\nConcat 3 arrays:\n", c_multi)
```
```
# Output:
# concat axis=0:
#  [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]
# Shape: (4, 2)
#
# concat axis=1:
#  [[1 2 5 6]
#  [3 4 7 8]]
# Shape: (2, 4)
#
# Concat 3 arrays:
#  [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]
#  [1 2]
#  [3 4]]
```

### Example 2: vstack, hstack, dstack

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# vstack (vertical): stacks along axis 0
print("vstack:\n", np.vstack([a, b]))
print("Shape:", np.vstack([a, b]).shape)

# hstack (horizontal): stacks along axis 1 (or flattens for 1D)
print("\nhstack:", np.hstack([a, b]))
print("Shape:", np.hstack([a, b]).shape)

# 2D hstack
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6]])
print("\nhstack 2D:\n", np.hstack([A, B.T]))

# dstack (depth): stacks along axis 2
print("\ndstack:\n", np.dstack([a, b]))
print("Shape:", np.dstack([a, b]).shape)
```
```
# Output:
# vstack:
#  [[1 2 3]
#  [4 5 6]]
# Shape: (2, 3)
#
# hstack: [1 2 3 4 5 6]
# Shape: (6,)
#
# hstack 2D:
#  [[1 2 5]
#  [3 4 6]]
#
# dstack:
#  [[[1 4]
#   [2 5]
#   [3 6]]]
# Shape: (1, 3, 2)
```

### Example 3: column_stack and row_stack

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# column_stack: stacks 1D arrays as columns
col = np.column_stack([a, b])
print("column_stack:\n", col)
print("Shape:", col.shape)

# row_stack (same as vstack)
row = np.row_stack([a, b])
print("\nrow_stack:\n", row)
print("Shape:", row.shape)

# Practical: combine features
features = np.array([[0.5, 0.2],
                     [0.8, 0.3],
                     [0.1, 0.9]])
labels = np.array([1, 0, 1])

# Add labels as a column
dataset = np.column_stack([features, labels])
print("\nFeatures with label column:\n", dataset)
```
```
# Output:
# column_stack:
#  [[1 4]
#  [2 5]
#  [3 6]]
# Shape: (3, 2)
#
# row_stack:
#  [[1 2 3]
#  [4 5 6]]
# Shape: (2, 3)
#
# Features with label column:
#  [[0.5 0.2 1. ]
#  [0.8 0.3 0. ]
#  [0.1 0.9 1. ]]
```

### Example 4: split, array_split

```python
import numpy as np

arr = np.arange(10)
print("Original:", arr)

# split into 5 equal parts
parts = np.split(arr, 5)
print("\nsplit into 5:")
for i, p in enumerate(parts):
    print(f"  Part {i}: {p}")

# array_split handles uneven splits
arr2 = np.arange(10)
parts_uneven = np.array_split(arr2, 3)
print("\narray_split into 3:")
for i, p in enumerate(parts_uneven):
    print(f"  Part {i}: {p}")

# Split at specific indices
parts_at = np.split(arr, [3, 7])
print("\nsplit at [3, 7]:")
for i, p in enumerate(parts_at):
    print(f"  Part {i}: {p}")
```
```
# Output:
# Original: [0 1 2 3 4 5 6 7 8 9]
#
# split into 5:
#   Part 0: [0 1]
#   Part 1: [2 3]
#   Part 2: [4 5]
#   Part 3: [6 7]
#   Part 4: [8 9]
#
# array_split into 3:
#   Part 0: [0 1 2 3]
#   Part 1: [4 5 6 7]
#   Part 2: [8 9]
#
# split at [3, 7]:
#   Part 0: [0 1 2]
#   Part 1: [3 4 5 6]
#   Part 2: [7 8 9]
```

### Example 5: hsplit and vsplit

```python
import numpy as np

arr = np.arange(24).reshape(4, 6)
print("Original:\n", arr)

# vsplit: split along axis 0 (horizontal split = vertical stacking)
v_parts = np.vsplit(arr, 2)
print("\nvsplit into 2:")
for i, p in enumerate(v_parts):
    print(f"  Part {i}:\n    {p}")

# hsplit: split along axis 1 (vertical split = horizontal stacking)
h_parts = np.hsplit(arr, 3)
print("\nhsplit into 3:")
for i, p in enumerate(h_parts):
    print(f"  Part {i}:\n    {p}")

# Split at specific column indices
h_parts_at = np.hsplit(arr, [2, 4])
print("\nhsplit at columns [2, 4]:")
for i, p in enumerate(h_parts_at):
    print(f"  Part {i}:\n    {p}")
```
```
# Output:
# Original:
#  [[ 0  1  2  3  4  5]
#  [ 6  7  8  9 10 11]
#  [12 13 14 15 16 17]
#  [18 19 20 21 22 23]]
#
# vsplit into 2:
#   Part 0:
#     [[ 0  1  2  3  4  5]
#      [ 6  7  8  9 10 11]]
#   Part 1:
#     [[12 13 14 15 16 17]
#      [18 19 20 21 22 23]]
#
# hsplit into 3:
#   Part 0:
#     [[ 0  1]
#      [ 6  7]
#      [12 13]
#      [18 19]]
#   Part 1:
#     [[ 2  3]
#      [ 8  9]
#      [14 15]
#      [20 21]]
#   Part 2:
#     [[ 4  5]
#      [10 11]
#      [16 17]
#      [22 23]]
#
# hsplit at columns [2, 4]:
#   Part 0:
#     [[ 0  1]
#      [ 6  7]
#      [12 13]
#      [18 19]]
#   Part 1:
#     [[ 2  3]
#      [ 8  9]
#      [14 15]
#      [20 21]]
#   Part 2:
#     [[ 4  5]
#      [10 11]
#      [16 17]
#      [22 23]]
```

### Example 6: Practical Data Pipeline

```python
import numpy as np

np.random.seed(42)

# Generate synthetic dataset
n = 1000
X = np.random.randn(n, 3)
y = np.random.randint(0, 2, n).reshape(-1, 1)

# Add bias term (column of ones)
X_with_bias = np.hstack([np.ones((n, 1)), X])
print("X with bias shape:", X_with_bias.shape)

# Combine features and labels for shuffling together
dataset = np.hstack([X, y])
np.random.shuffle(dataset)

# Split back into features and labels
X_shuffled = dataset[:, :-1]
y_shuffled = dataset[:, -1]
print("X shuffled shape:", X_shuffled.shape)
print("y shuffled shape:", y_shuffled.shape)

# Create mini-batches
batch_size = 100
batches = np.array_split(dataset, n // batch_size)
print(f"\nCreated {len(batches)} batches of ~{batch_size} samples each")
for i, batch in enumerate(batches[:3]):
    print(f"  Batch {i}: {batch.shape}")
```
```
# Output:
# X with bias shape: (1000, 4)
# X shuffled shape: (1000, 3)
# y shuffled shape: (1000,)
#
# Created 10 batches of ~100 samples each
#   Batch 0: (100, 4)
#   Batch 1: (100, 4)
#   Batch 2: (100, 4)
```

### Example 7: Adding New Samples to Existing Data

```python
import numpy as np

# Existing training data
X_train = np.array([[0.5, 0.2],
                    [0.8, 0.3]])
y_train = np.array([0, 1])

# New data collected
X_new = np.array([[0.1, 0.9],
                  [0.4, 0.6],
                  [0.7, 0.2]])
y_new = np.array([1, 0, 1])

# Append new data
X_train = np.vstack([X_train, X_new])
y_train = np.concatenate([y_train, y_new])

print("Combined X:\n", X_train)
print("\nCombined y:", y_train)
print("Final shapes: X", X_train.shape, "y", y_train.shape)
```
```
# Output:
# Combined X:
#  [[0.5 0.2]
#  [0.8 0.3]
#  [0.1 0.9]
#  [0.4 0.6]
#  [0.7 0.2]]
#
# Combined y: [0 1 1 0 1]
# Final shapes: X (5, 2) y (5,)
```

## Common Mistakes

1. **Shape Mismatch on Concatenation Axis:** All arrays must have the same shape except along the concatenation axis. `np.concatenate([a(3,4), b(2,4)], axis=0)` works, but `a(3,4)` and `b(3,5)` along axis=0 fails.

2. **Confusing `vstack` and `hstack` for 1D Arrays:** For 1D arrays, `vstack` creates a 2D array (stacking rows), `hstack` remains 1D (concatenating elements), and `column_stack` creates a 2D array with columns.

3. **Using `split` Instead of `array_split` for Uneven Splits:** `split` requires the array to divide evenly into the specified number of parts. `array_split` handles uneven splits gracefully.

4. **Forgetting to Reshape Before Concatenation:** When combining scalars or 1D arrays with 2D arrays, reshape first: `np.hstack([X, y.reshape(-1, 1)])`.

5. **Creating Unnecessary Copies:** `np.concatenate` always returns a new array (copies data). For repeated concatenation in loops, pre-allocate and assign to avoid O(n^2) behavior.

6. **Split Indices Confusion:** `np.split(arr, [3, 7])` splits at indices 3 and 7, producing 3 parts. The indices are cut points, not sizes.

7. **Assuming `column_stack` Works Like `hstack` for 2D:** `column_stack` converts 1D arrays to columns first. For 2D arrays, `column_stack` is equivalent to `hstack`.

## Interview Questions

### Beginner

1. **Q:** How do you join two arrays along rows?
   **A:** `np.concatenate([a, b], axis=0)` or `np.vstack([a, b])`.

2. **Q:** What is the difference between `vstack` and `hstack`?
   **A:** `vstack` stacks vertically (increases rows, axis=0). `hstack` stacks horizontally (increases columns, axis=1). For 1D arrays, `vstack` creates 2D while `hstack` stays 1D.

3. **Q:** How do you add a column of ones to a feature matrix?
   **A:** `np.hstack([np.ones((n, 1)), X])` or `np.column_stack([np.ones(n), X])`.

4. **Q:** What does `np.array_split` do differently from `np.split`?
   **A:** `np.array_split` allows unequal splits when the array can't be divided evenly. `np.split` raises an error for uneven splits.

5. **Q:** How do you split a matrix vertically into 3 equal parts?
   **A:** `np.vsplit(arr, 3)` or `np.split(arr, 3, axis=0)`.

### Intermediate

1. **Q:** What is the difference between `np.hstack` and `np.column_stack` for 1D arrays?
   **A:** `hstack` concatenates 1D arrays into a longer 1D array. `column_stack` converts each 1D array to a column (shape (n, 1)) and then stacks them horizontally, producing a 2D array.

2. **Q:** How would you split a dataset into training and testing sets using NumPy?
   **A:** Shuffle indices with `np.random.permutation(n)`, compute split point `int(n * 0.8)`, then use `np.split(data, [split_point])` or index with the shuffled indices.

3. **Q:** What is the time complexity of repeatedly concatenating arrays in a loop? How can it be improved?
   **A:** Repeated concatenation creates a new array each iteration, resulting in O(n^2) time. Pre-allocate a list of arrays and call `np.concatenate` once, or pre-allocate the final array and assign to slices.

4. **Q:** How does `np.dstack` work, and when would you use it?
   **A:** `np.dstack` stacks arrays along the third axis (depth). It's useful for combining image channels or stacking 2D arrays into a volume. For 1D arrays, it first converts each to shape (1, n, 1) and then stacks.

5. **Q:** What happens if you pass `axis=None` to `np.concatenate`?
   **A:** With `axis=None`, all arrays are flattened before concatenation, producing a 1D result. This is equivalent to `np.hstack` on flattened versions.

### Advanced

1. **Q:** Compare the memory behavior of `np.concatenate` vs pre-allocating an output array and filling slices. When is each approach preferred?
   **A:** `np.concatenate` allocates new memory and copies all data. Pre-allocation (`out = np.zeros(total_shape)`, then assign slices) allows filling incrementally, useful for streaming or out-of-core processing. `concatenate` is more convenient for one-time joins; pre-allocation is better for incremental assembly or when intermediate arrays must be kept.

2. **Q:** How would you implement a "merge" operation that interleaves two arrays along an axis (like a zip)?
   **A:** For 1D: `np.empty(len(a) + len(b), dtype=a.dtype)` and assign with strides: `result[0::2], result[1::2] = a, b`. For ND: create output with doubled size along the merge axis and use slicing with steps.

3. **Q:** Explain the concept of "split points" in `np.split` and how `np.array_split` determines split sizes for uneven divisions.
   **A:** `np.split` uses indices_or_sections: either a number N (divides into N equal parts) or a list of split indices. `array_split` with N parts distributes the remainder across the first splits: if `n % N != 0`, the first `n % N` splits get `ceil(n/N)` elements, the rest get `floor(n/N)`.

## Practice Problems

### Easy

1. Create two 2x3 arrays and concatenate them along axis 0.

2. Use `vstack` to combine `[1, 2, 3]` and `[4, 5, 6]` into a 2x3 matrix.

3. Add a column of zeros to the right of a 4x2 matrix.

4. Split a 1D array of 12 elements into 4 equal parts.

5. Use `hsplit` to split a 3x9 matrix into 3 equal blocks of 3 columns each.

### Medium

1. Create a dataset of 100 samples with 5 features. Split into train (80%) and test (20%) sets.

2. Concatenate 5 arrays of shape (100, 10) into a single array of shape (500, 10). Then split it back into 5 arrays.

3. Add a bias column to a 200x3 feature matrix. Then column_stack the labels. Then split the result back into X and y.

4. Create 10 arrays of shape (50, 4) and combine them into a single (500, 4) array efficiently (without a loop of concatenates).

5. Given a 4x4 matrix, extract the diagonal elements as a column vector, then hstack it back to the original matrix.

### Hard

1. Implement a function `interleave(a, b, axis=0)` that interleaves two arrays along the given axis.

2. Implement a function `split_into_windows(data, window_size, step)` that splits a 2D time series into overlapping windows (rows) without loops.

3. Implement a function `merge_datasets(features_list, labels_list)` that takes a list of feature arrays and label arrays (potentially with different sample counts per class) and creates a balanced dataset by upsampling minority classes using `np.concatenate` and `np.random.choice`.

## Solutions

### Easy Solutions

```python
# 1
a = np.ones((2, 3))
b = np.zeros((2, 3))
print(np.concatenate([a, b], axis=0))

# 2
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.vstack([a, b]))

# 3
mat = np.ones((4, 2))
zeros_col = np.zeros((4, 1))
print(np.hstack([mat, zeros_col]))

# 4
arr = np.arange(12)
print(np.split(arr, 4))

# 5
mat = np.arange(27).reshape(3, 9)
for part in np.hsplit(mat, 3):
    print(part)
```

### Medium Solutions

```python
# 1 Train/test split
X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)
indices = np.random.permutation(100)
split = 80
X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 2 Concat 5 arrays
arrays = [np.random.randn(100, 10) for _ in range(5)]
combined = np.concatenate(arrays, axis=0)
back = np.split(combined, 5)
print(f"Combined: {combined.shape}, Parts: {[a.shape for a in back]}")

# 3 Bias + labels
X = np.random.randn(200, 3)
y = np.random.randint(0, 2, 200).reshape(-1, 1)
X_bias = np.hstack([np.ones((200, 1)), X])
dataset = np.hstack([X_bias, y])
X_back, y_back = dataset[:, :-1], dataset[:, -1]
print(f"X: {X_back.shape}, y: {y_back.shape}")

# 4 Efficient combine
arrays = [np.random.randn(50, 4) for _ in range(10)]
combined = np.vstack(arrays)  # or np.concatenate(arrays, axis=0)
print(f"Combined: {combined.shape}")

# 5 Diagonal as column
mat = np.arange(16).reshape(4, 4)
diag_col = np.diag(mat).reshape(-1, 1)
extended = np.hstack([mat, diag_col])
print("Extended:\n", extended)
```

### Hard Solutions

```python
# 1 Interleave arrays
def interleave(a, b, axis=0):
    a_shape = list(a.shape)
    b_shape = list(b.shape)
    assert a_shape == b_shape, "Shapes must match"
    result_shape = a_shape.copy()
    result_shape[axis] = a_shape[axis] + b_shape[axis]
    result = np.empty(result_shape, dtype=a.dtype)
    slices_a = [slice(None)] * result.ndim
    slices_b = [slice(None)] * result.ndim
    slices_a[axis] = slice(0, None, 2)
    slices_b[axis] = slice(1, None, 2)
    result[tuple(slices_a)] = a
    result[tuple(slices_b)] = b
    return result

a, b = np.array([1, 3, 5]), np.array([2, 4, 6])
print("Interleaved:", interleave(a, b, axis=0))

# 2 Overlapping windows
def split_into_windows(data, window_size, step=1):
    n = data.shape[0]
    n_windows = (n - window_size) // step + 1
    indices = np.arange(window_size)[None, :] + np.arange(0, n_windows * step, step)[:, None]
    return data[indices]

ts = np.arange(100).reshape(-1, 1)
windows = split_into_windows(ts, window_size=10, step=5)
print(f"Windows shape: {windows.shape}")
print(f"Window 0: {windows[0].ravel()}, Window 1: {windows[1].ravel()}")

# 3 Balanced dataset via upsampling
def merge_datasets(features_list, labels_list):
    # Find the largest class
    sizes = [len(l) for l in labels_list]
    max_size = max(sizes)
    balanced_features, balanced_labels = [], []
    for i in range(len(features_list)):
        n = len(labels_list[i])
        if n < max_size:
            indices = np.random.choice(n, max_size - n, replace=True)
            upsample_features = features_list[i][indices]
            upsample_labels = labels_list[i][indices]
            feat = np.vstack([features_list[i], upsample_features])
            lab = np.concatenate([labels_list[i], upsample_labels])
        else:
            feat = features_list[i]
            lab = labels_list[i]
        balanced_features.append(feat)
        balanced_labels.append(lab)
    X_balanced = np.vstack(balanced_features)
    y_balanced = np.concatenate(balanced_labels)
    return X_balanced, y_balanced

class0_X = np.random.randn(50, 4)
class0_y = np.zeros(50)
class1_X = np.random.randn(30, 4)
class1_y = np.ones(30)
X_bal, y_bal = merge_datasets([class0_X, class1_X], [class0_y, class1_y])
print(f"Balanced: X={X_bal.shape}, y={y_bal.shape}")
print(f"Class counts: {np.bincount(y_bal.astype(int))}")
```

## Related Concepts

- Python list concatenation with `+` and `extend`
- Pandas `concat`, `merge`, and `join`
- TensorFlow `tf.concat` and `tf.split`
- PyTorch `torch.cat` and `torch.split`

## Next Concepts

- Universal functions / ufuncs (PYT-074)
- Structured arrays (PYT-075)

## Summary

NumPy provides a rich set of functions for joining and splitting arrays. `concatenate` is the general-purpose join function. `vstack`, `hstack`, and `dstack` are convenience wrappers for common stacking scenarios. `column_stack` and `row_stack` handle 1D arrays specifically. Splitting functions (`split`, `array_split`, `hsplit`, `vsplit`) divide arrays along axes. These operations are essential for data pipeline construction in ML workflows.

## Key Takeaways

- `np.concatenate([a, b], axis=n)` is the general joining function
- `vstack` = rows increase; `hstack` = columns increase; `dstack` = depth increases
- `column_stack` makes columns from 1D arrays; `row_stack` = `vstack`
- `array_split` handles uneven splits; `split` requires even division
- `hsplit`/`vsplit` are convenient for axis=1/axis=0 splitting
- Split indices are cut points, not sizes
- Repeated concatenation in loops is O(n^2) — pre-allocate for efficiency
- Always check shapes match except on the concatenation axis
