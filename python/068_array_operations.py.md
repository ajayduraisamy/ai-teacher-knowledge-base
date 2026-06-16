# Concept: Array Operations

## Concept ID

PYT-068

## Difficulty

Intermediate

## Domain

Python

## Module

NumPy

## Learning Objectives

- Perform vectorized arithmetic on NumPy arrays
- Understand and apply broadcasting rules
- Compute dot products with `np.dot`
- Use aggregate functions: `np.sum`, `np.mean`, `np.max`, `np.min`
- Find indices of extremes with `np.argmax` and `np.argmin`
- Clip values with `np.clip`

## Prerequisites

- NumPy array creation (PYT-066)
- Array indexing and slicing (PYT-067)
- Basic understanding of matrix multiplication

## Definition

Array operations in NumPy include element-wise arithmetic, broadcasting (automatic dimension alignment), linear algebra operations, reduction/aggregation functions, and utility operations like clipping. These operations are vectorized, meaning they execute in compiled C code without Python loops, providing massive performance gains.

## Intuition

Vectorized operations treat the entire array as a single unit rather than a collection of elements. Adding two arrays together corresponds to adding each pair of elements. Broadcasting extends this by automatically aligning arrays of different shapes — think of it as stretching a smaller array to match a larger one before performing the operation. Aggregation functions collapse dimensions, reducing an array to summary statistics.

## Why This Concept Matters

Vectorization is the key performance advantage of NumPy. A single `arr1 + arr2` is not just cleaner code than nested loops — it's 10-100x faster. Broadcasting enables elegant, concise code for operations like centering data (subtracting the mean from every element) or scaling features. Aggregation functions are essential for data analysis and form the basis of descriptive statistics.

## Real World Examples

1. **Feature Scaling:** `(X - X.mean(axis=0)) / X.std(axis=0)` standardizes a dataset in one line.
2. **Euclidean Distance:** `np.sqrt(((a - b)**2).sum(axis=1))` computes distances between points.
3. **Gradient Descent:** `w -= learning_rate * X.T.dot(X.dot(w) - y)` updates weights in linear regression.
4. **Image Brightness:** `img = np.clip(img + 50, 0, 255)` adjusts brightness with clipping.
5. **Normalization:** `X / np.linalg.norm(X, axis=1, keepdims=True)` normalizes rows to unit length.

## AI/ML Relevance

Almost every ML algorithm relies on vectorized operations. Forward passes in neural networks are sequences of matrix multiplications and element-wise activations. Loss functions use aggregation operations. Broadcasting handles batch dimensions automatically. Gradient computations rely on dot products. `np.clip` is used for gradient clipping to prevent exploding gradients.

## Code Examples

### Example 1: Vectorized Arithmetic

```python
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print("a + b:", a + b)
print("a - b:", a - b)
print("a * b:", a * b)
print("a / b:", a / b)
print("a ** 2:", a ** 2)
print("b % 3:", b % 3)

# Scalar operations
print("\na + 5:", a + 5)
print("a * 2.5:", a * 2.5)

# Comparison operators
print("\na > 2:", a > 2)
print("b == 30:", b == 30)
```
```
# Output:
# a + b: [11 22 33 44]
# a - b: [-9 -18 -27 -36]
# a * b: [10 40 90 160]
# a / b: [0.1 0.1 0.1 0.1]
# a ** 2: [ 1  4  9 16]
# b % 3: [1 2 0 1]
# 
# a + 5: [ 6  7  8  9]
# a * 2.5: [ 2.5  5.   7.5 10. ]
# 
# a > 2: [False False  True  True]
# b == 30: [False False  True False]
```

### Example 2: Broadcasting

```python
import numpy as np

# Scalar broadcasting
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print("Original:\n", arr)
print("arr * 10:\n", arr * 10)

# 1D broadcast to 2D
row = np.array([10, 20, 30])
print("\narr + row:\n", arr + row)

col = np.array([[100], [200]])
print("\narr + col:\n", arr + col)

# Centering data
data = np.array([[10, 20, 30],
                 [15, 25, 35],
                 [12, 22, 32]])
mean = data.mean(axis=0)
centered = data - mean
print("\nOriginal mean:", mean)
print("Centered:\n", centered)
print("Centered mean:", centered.mean(axis=0))
```
```
# Output:
# Original:
#  [[1 2 3]
#  [4 5 6]]
# arr * 10:
#  [[10 20 30]
#  [40 50 60]]
# 
# arr + row:
#  [[11 22 33]
#  [14 25 36]]
# 
# arr + col:
#  [[101 102 103]
#  [204 205 206]]
# 
# Original mean: [12.33333333 22.33333333 32.33333333]
# Centered:
#  [[-2.33333333 -2.33333333 -2.33333333]
#  [ 2.66666667  2.66666667  2.66666667]
#  [-0.33333333 -0.33333333 -0.33333333]]
# Centered mean: [0. 0. 0.]
```

### Example 3: Dot Products and Matrix Multiplication

```python
import numpy as np

# 1D dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print("1D dot:", np.dot(a, b))       # 1*4 + 2*5 + 3*6

# 2D matrix multiplication
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])
print("\nA @ B:\n", A @ B)            # Python 3.5+ operator
print("np.dot(A, B):\n", np.dot(A, B))

# Matrix-vector product
v = np.array([1, 0])
print("\nA @ v:", A @ v)

# Dot product with axis specification
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
w = np.array([0.5, 1.0, 1.5])
result = arr @ w
print("\narr @ w:", result)
```
```
# Output:
# 1D dot: 32
# 
# A @ B:
#  [[19 22]
#  [43 50]]
# np.dot(A, B):
#  [[19 22]
#  [43 50]]
# 
# A @ v: [1 3]
# 
# arr @ w: [ 7.  16.5]
```

### Example 4: Aggregation Operations

```python
import numpy as np

arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

print("Array:\n", arr)
print("sum:", np.sum(arr))
print("mean:", np.mean(arr))
print("max:", np.max(arr))
print("min:", np.min(arr))
print("std:", np.std(arr))
print("var:", np.var(arr))
print("prod:", np.prod(arr))

# Axis-wise aggregation
print("\nSum along axis=0 (columns):", np.sum(arr, axis=0))
print("Sum along axis=1 (rows):", np.sum(arr, axis=1))

# Keep dimensions with keepdims
col_sum = np.sum(arr, axis=0, keepdims=True)
print("\nSum with keepdims:\n", col_sum)
print("Shape:", col_sum.shape)
```
```
# Output:
# Array:
#  [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
# sum: 78
# mean: 6.5
# max: 12
# min: 1
# std: 3.452052529534663
# var: 11.916666666666666
# prod: 479001600
# 
# Sum along axis=0 (columns): [15 18 21 24]
# Sum along axis=1 (rows): [10 26 42]
# 
# Sum with keepdims:
#  [[15 18 21 24]]
# Shape: (1, 4)
```

### Example 5: argmax, argmin, and clip

```python
import numpy as np

arr = np.array([[3, 7, 1],
                [8, 2, 5],
                [4, 9, 6]])

print("Array:\n", arr)
print("argmax (flattened):", np.argmax(arr))
print("argmin (flattened):", np.argmin(arr))

# Unravel index to 2D
flat_idx = np.argmax(arr)
row, col = np.unravel_index(flat_idx, arr.shape)
print(f"Max value {arr[row, col]} at position ({row}, {col})")

# Axis-wise argmax
print("\nargmax axis=0 (per column):", np.argmax(arr, axis=0))
print("argmax axis=1 (per row):", np.argmax(arr, axis=1))

# Clip values
data = np.array([1, 5, 10, 15, 20, 25, 30])
clipped = np.clip(data, 5, 20)
print("\nOriginal:", data)
print("Clipped (5, 20):", clipped)

# Clip with percentile-based bounds
arr = np.random.randn(100)
lower, upper = np.percentile(arr, [5, 95])
clipped_arr = np.clip(arr, lower, upper)
print(f"\nBefore clip: min={arr.min():.3f}, max={arr.max():.3f}")
print(f"After clip:  min={clipped_arr.min():.3f}, max={clipped_arr.max():.3f}")
```
```
# Output:
# Array:
#  [[3 7 1]
#  [8 2 5]
#  [4 9 6]]
# argmax (flattened): 7
# argmin (flattened): 2
# Max value 9.0 at position (2, 1)
# 
# argmax axis=0 (per column): [1 2 0]
# argmax axis=1 (per row): [1 0 1]
# 
# Original: [ 1  5 10 15 20 25 30]
# Clipped (5, 20): [ 5  5 10 15 20 20 20]
# 
# Before clip: min=-2.871, max=2.724
# After clip:  min=-1.573, max=1.865
```

### Example 6: Cumulative Operations

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

print("cumsum:", np.cumsum(arr))
print("cumprod:", np.cumprod(arr))

# 2D cumulative sum
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6]])
print("\n2D cumsum (axis=0):\n", np.cumsum(arr2d, axis=0))
print("2D cumsum (axis=1):\n", np.cumsum(arr2d, axis=1))

# diff for discrete differences
print("\ndiff:", np.diff(arr))
print("diff with n=2:", np.diff(arr, n=2))
```
```
# Output:
# cumsum: [ 1  3  6 10 15]
# cumprod: [  1   2   6  24 120]
# 
# 2D cumsum (axis=0):
#  [[1 2 3]
#  [5 7 9]]
# 2D cumsum (axis=1):
#  [[ 1  3  6]
#  [ 4  9 15]]
# 
# diff: [1 1 1 1]
# diff with n=2: [0 0 0]
```

## Common Mistakes

1. **Misunderstanding Broadcasting Rules:** Broadcasting works from the last dimension backwards. `arr(3, 4) + arr(3,)` fails because `3 != 4`. Use `arr(4,)` or reshape to `(3, 1)`.

2. **Forgetting that `np.dot` Changes Behavior by Dimensionality:** For 1D arrays `np.dot` computes the inner product. For 2D arrays it computes matrix multiplication. For higher dimensions it computes sum products over the last axis of a and second-to-last of b.

3. **Confusing Axis Parameter:** `axis=0` operates along rows (vertically, collapsing columns). `axis=1` operates along columns (horizontally, collapsing rows). For a 2D array, `sum(axis=0)` sums each column.

4. **In-Place vs Out-of-Place:** `arr + 1` returns a new array. `arr += 1` modifies in-place. Use in-place operations for memory efficiency with large arrays.

5. **Assuming `np.clip` Modifies In-Place:** `np.clip(arr, 0, 1)` returns a new array by default. Use `np.clip(arr, 0, 1, out=arr)` for in-place clipping.

6. **Overlooking Integer Overflow:** `np.array([100, 200, 300], dtype=np.uint8) + 100` overflows because `uint8` max is 255. Use `dtype=np.int16` or `np.float64` for intermediate results.

7. **Misapplying `np.argmax` with Ties:** `np.argmax` returns the index of the first maximum. If multiple equal maxima exist, only the first is returned.

## Interview Questions

### Beginner

1. **Q:** How do you add a scalar to every element of a NumPy array?
   **A:** `arr + 5` — NumPy broadcasts the scalar across all elements.

2. **Q:** What does `np.sum(arr, axis=0)` compute?
   **A:** It sums each column, collapsing rows. For a 2D array of shape `(m, n)`, the result is shape `(n,)`.

3. **Q:** How do you find the maximum value and its position in an array?
   **A:** Use `arr.max()` for the maximum value and `arr.argmax()` for its flattened index.

4. **Q:** What is the difference between `arr * arr` and `arr @ arr`?
   **A:** `arr * arr` is element-wise multiplication. `arr @ arr` is matrix multiplication (dot product).

5. **Q:** What does `np.clip(arr, 0, 1)` do?
   **A:** It limits all values in `arr` to the range `[0, 1]` — values below 0 become 0, values above 1 become 1.

### Intermediate

1. **Q:** Explain broadcasting rules with an example where shape `(3, 1)` and `(4,)` are added.
   **A:** `(3, 1)` broadcasts against `(4,)` → `(3, 4)`. The column is stretched across 4 columns, and the row is stretched across 3 rows. The result is `(3, 4)`.

2. **Q:** When would you use `keepdims=True` in aggregation?
   **A:** To preserve the number of dimensions for broadcasting. For example, `arr / arr.sum(axis=1, keepdims=True)` correctly normalizes each row because shapes `(m, n)` and `(m, 1)` broadcast.

3. **Q:** How is `np.dot(a, b)` different from `np.multiply(a, b)` for 2D matrices?
   **A:** `np.dot(a, b)` computes matrix multiplication (sum of products over inner dimension). `np.multiply(a, b)` computes element-wise multiplication (Hadamard product), requiring identical shapes.

4. **Q:** What is the time complexity difference between a Python loop sum and `np.sum`?
   **A:** Python loop is `O(n)` in interpreted code with overhead per iteration. `np.sum` is `O(n)` in compiled C code with no Python overhead — typically 10-50x faster for large arrays.

5. **Q:** How can `np.clip` be used for gradient clipping in neural networks?
   **A:** `np.clip(grad, -threshold, threshold)` limits gradients to a range, preventing exploding gradients during backpropagation.

### Advanced

1. **Q:** Explain how broadcasting works in the general case. What shapes are compatible?
   **A:** Two dimensions are compatible if they are equal or one of them is 1. NumPy compares dimensions from the trailing dimension forward. Missing dimensions are treated as 1. If all dimension pairs are compatible, the broadcast shape takes the maximum size in each dimension.

2. **Q:** What is the difference between `np.einsum` and `np.dot`, and when would you use `einsum`?
   **A:** `np.einsum` uses Einstein summation notation for arbitrary tensor contractions. It can express `np.dot`, `np.sum`, `np.outer`, and more in a unified way. Use it for complex multi-dimensional operations or to avoid multiple reshapes.

3. **Q:** How does NumPy handle the reduction operation for arrays with strides that are not contiguous, and what is the performance implication?
   **A:** For non-contiguous arrays (e.g., transposed views), NumPy may need to create a temporary contiguous copy before performing reduction operations, incurring memory and time overhead. Use `np.ascontiguousarray()` to ensure contiguity when doing many operations.

## Practice Problems

### Easy

1. Create two 1D arrays of length 5 and compute their element-wise sum, difference, and product.

2. Given `arr = np.array([[1, 2], [3, 4], [5, 6]])`, compute the sum of all elements, sum of each column, and sum of each row.

3. Create a 4x4 array of random floats and find its minimum, maximum, mean, and standard deviation.

4. Use `np.clip` to clamp the values of `np.array([-5, -2, 0, 3, 7, 10])` to the range `[-1, 5]`.

5. Find the indices of the 3 largest values in `np.array([10, 3, 7, 1, 9, 4])`.

### Medium

1. Given a 100x5 dataset, standardize each feature (subtract column mean, divide by column std) using broadcasting.

2. Compute the pairwise Euclidean distance matrix for 10 points in 4D space without loops.

3. Implement a function that computes the softmax of a 2D array along axis=1 using broadcasting.

4. Given `A` (3x4) and `B` (4x2), compute `C = A @ B`. Then compute the row-wise L2 norm of C.

5. Create a 10x10 matrix and set all values in the border (first/last row, first/last column) to 0 using slicing and broadcasting.

### Hard

1. Implement matrix multiplication manually using broadcasting and `np.sum`, then compare speed against `A @ B` for a 1000x1000 matrix.

2. Implement a function that computes the rolling mean of a 1D array with window size `k` using only NumPy operations (no loops).

3. Implement the forward pass of a 2-layer neural network (ReLU activation) using only NumPy operations: `h = relu(X @ W1 + b1)`, then `y = h @ W2 + b2`. Use random shapes: X(100, 20), W1(20, 64), b1(64,), W2(64, 10), b2(10,).

## Solutions

### Easy Solutions

```python
# 1
a, b = np.array([1, 2, 3, 4, 5]), np.array([10, 20, 30, 40, 50])
print(a + b, a - b, a * b)

# 2
arr = np.array([[1, 2], [3, 4], [5, 6]])
print("Total:", arr.sum())
print("Columns:", arr.sum(axis=0))
print("Rows:", arr.sum(axis=1))

# 3
arr = np.random.rand(4, 4)
print("Min:", arr.min(), "Max:", arr.max())
print("Mean:", arr.mean(), "Std:", arr.std())

# 4
arr = np.array([-5, -2, 0, 3, 7, 10])
print(np.clip(arr, -1, 5))

# 5
arr = np.array([10, 3, 7, 1, 9, 4])
indices = np.argsort(arr)[-3:][::-1]
print(indices, arr[indices])
```

### Medium Solutions

```python
# 1 Standardize dataset
data = np.random.randn(100, 5)
mean = data.mean(axis=0)
std = data.std(axis=0)
standardized = (data - mean) / std
print("Mean after:", standardized.mean(axis=0).round(6))
print("Std after:", standardized.std(axis=0))

# 2 Pairwise Euclidean distance
points = np.random.randn(10, 4)
diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
dist = np.sqrt((diff**2).sum(axis=-1))
print(dist.shape, dist[0, :3])

# 3 Softmax
def softmax(x):
    e_x = np.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

scores = np.random.randn(5, 3)
print(softmax(scores).sum(axis=1))

# 4 Matrix multiplication + row norms
A, B = np.random.randn(3, 4), np.random.randn(4, 2)
C = A @ B
row_norms = np.sqrt((C**2).sum(axis=1))
print("Row norms:", row_norms)

# 5 Border zero
M = np.ones((10, 10))
M[0, :] = M[-1, :] = M[:, 0] = M[:, -1] = 0
print(M[:3, :3])
```

### Hard Solutions

```python
# 1 Manual matmul
A = np.random.randn(1000, 1000)
B = np.random.randn(1000, 1000)

# Broadcasting-based manual matmul
def manual_matmul(A, B):
    return np.sum(A[:, :, np.newaxis] * B[np.newaxis, :, :], axis=1)

import time
start = time.time()
C1 = A @ B
t1 = time.time() - start

start = time.time()
C2 = manual_matmul(A, B)
t2 = time.time() - start

print(f"Built-in: {t1:.3f}s, Manual: {t2:.3f}s, Correct: {np.allclose(C1, C2)}")

# 2 Rolling mean
def rolling_mean(arr, k):
    kernel = np.ones(k) / k
    return np.convolve(arr, kernel, mode='valid')

arr = np.arange(20, dtype=float)
print(rolling_mean(arr, 5))

# 3 Two-layer NN forward pass
X = np.random.randn(100, 20)
W1, b1 = np.random.randn(20, 64), np.random.randn(64)
W2, b2 = np.random.randn(64, 10), np.random.randn(10)

h = np.maximum(0, X @ W1 + b1)  # ReLU
y = h @ W2 + b2
print(f"Output shape: {y.shape}")
```

## Related Concepts

- Linear algebra operations (PYT-069)
- Universal functions / ufuncs (PYT-074)
- Basic Python arithmetic operators

## Next Concepts

- NumPy linear algebra (PYT-069)
- NumPy random module (PYT-070)
- NumPy statistics (PYT-071)

## Summary

NumPy array operations provide fast, vectorized computation on arrays. Arithmetic operators work element-wise with automatic broadcasting. `np.dot` and `@` handle matrix multiplication. Aggregation functions (`sum`, `mean`, `max`, `min`) with axis parameter enable flexible reduction. `argmax`/`argmin` locate extremes, and `clip` bounds values. Broadcasting rules allow operations between arrays of different shapes by aligning from the trailing dimensions.

## Key Takeaways

- Vectorized operations are 10-50x faster than Python loops
- Broadcasting aligns arrays from the last dimension back
- Use `axis` parameter for row-wise vs column-wise aggregation
- `@` is the matrix multiplication operator (Python 3.5+)
- `np.clip` is essential for bounding values and gradient clipping
- `keepdims=True` preserves dimensions for broadcasting compatibility
- In-place operations (`+=`, `*=`) save memory for large arrays
