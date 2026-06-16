# Concept: NumPy Arrays

## Concept ID

PYT-066

## Difficulty

Intermediate

## Domain

Python

## Module

NumPy

## Learning Objectives

- Create NumPy arrays from Python lists using `np.array()`
- Generate arrays with `np.zeros()`, `np.ones()`, `np.empty()`, `np.arange()`, and `np.linspace()`
- Inspect array properties: `ndim`, `shape`, `dtype`, `size`
- Compare memory efficiency of NumPy arrays vs Python lists
- Convert between data types using `astype()`

## Prerequisites

- Basic Python syntax (variables, lists, loops)
- Understanding of data types (`int`, `float`)
- Familiarity with importing modules

## Definition

A NumPy array (`ndarray`) is a homogeneous, multi-dimensional container of elements of the same data type, optimized for numerical computation. It is the core data structure of the NumPy library and provides vectorized operations, broadcasting, and efficient memory usage.

## Intuition

Think of a NumPy array as a grid of values, all of the same type, indexed by a tuple of non-negative integers. A 1D array is like a column of data. A 2D array is like a table (rows and columns). A 3D array is like a stack of tables. Unlike Python lists, NumPy arrays enforce a single data type across all elements, which allows the library to store data in contiguous memory blocks and apply fast, vectorized operations.

## Why This Concept Matters

NumPy arrays are the foundation of nearly every data science and machine learning library in Python, including pandas, scikit-learn, TensorFlow, and PyTorch. Understanding how to create and manipulate arrays is essential for efficient numerical computing. The performance gains from vectorized operations over Python loops can be 10-100x or more.

## Real World Examples

1. **Image Processing:** An RGB image is a 3D NumPy array of shape `(height, width, 3)`.
2. **Sensor Data:** A weather station records temperature, humidity, and pressure every minute, stored as a 2D array of shape `(1440, 3)` for one day.
3. **Financial Portfolios:** Stock prices over time are stored in 2D arrays where rows are time steps and columns are different stocks.
4. **Physics Simulations:** Particle positions in 3D space stored as a 2D array of shape `(n_particles, 3)`.
5. **Neural Network Inputs:** A batch of 32 images of size 64x64 with 3 color channels has shape `(32, 64, 64, 3)`.

## AI/ML Relevance

NumPy arrays are the universal data container in machine learning pipelines. Features are stored as 2D arrays `(n_samples, n_features)`, labels as 1D arrays `(n_samples,)`. Weight matrices in neural networks are NumPy arrays. Gradient computations, loss functions, and evaluation metrics all operate on NumPy arrays before being converted to tensors.

## Code Examples

### Example 1: Creating Arrays from Lists

```python
import numpy as np

# 1D array from a list
arr1d = np.array([1, 2, 3, 4, 5])
print("1D array:", arr1d)
print("Type:", type(arr1d))

# 2D array from nested lists
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print("\n2D array:\n", arr2d)

# 3D array
arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("\n3D array:\n", arr3d)
```
```
# Output:
# 1D array: [1 2 3 4 5]
# Type: <class 'numpy.ndarray'>
# 
# 2D array:
#  [[1 2 3]
#  [4 5 6]]
# 
# 3D array:
#  [[[1 2]
#   [3 4]]
# 
#  [[5 6]
#   [7 8]]]
```

### Example 2: Specialized Array Creation Functions

```python
import numpy as np

zeros = np.zeros((3, 4))
print("Zeros (3x4):\n", zeros)

ones = np.ones((2, 3))
print("\nOnes (2x3):\n", ones)

empty = np.empty((2, 2))
print("\nEmpty (2x2) — uninitialized values:\n", empty)

arange = np.arange(0, 10, 2)
print("\narange 0 to 10 step 2:", arange)

linspace = np.linspace(0, 1, 5)
print("\nlinspace 0 to 1, 5 points:", linspace)
```
```
# Output:
# Zeros (3x4):
#  [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]
# 
# Ones (2x3):
#  [[1. 1. 1.]
#  [1. 1. 1.]]
# 
# Empty (2x2) — uninitialized values:
#  [[4.9e-324 4.9e-324]
#  [4.9e-324 4.9e-324]]
# 
# arange 0 to 10 step 2: [0 2 4 6 8]
# 
# linspace 0 to 1, 5 points: [0.   0.25 0.5  0.75 1.  ]
```

### Example 3: Array Properties

```python
import numpy as np

arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

print("Array:\n", arr)
print("ndim (dimensions):", arr.ndim)
print("shape:", arr.shape)
print("dtype:", arr.dtype)
print("size (total elements):", arr.size)
print("itemsize (bytes per element):", arr.itemsize)
print("nbytes (total bytes):", arr.nbytes)
```
```
# Output:
# Array:
#  [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
# ndim (dimensions): 2
# shape: (3, 4)
# dtype: int32
# size (total elements): 12
# itemsize (bytes per element): 4
# nbytes (total bytes): 48
```

### Example 4: Specifying and Changing Data Types

```python
import numpy as np

# Explicit dtype
arr_int = np.array([1, 2, 3], dtype=np.int64)
print("int64 array:", arr_int.dtype)

arr_float = np.array([1, 2, 3], dtype=np.float32)
print("float32 array:", arr_float.dtype)

# Casting with astype
arr = np.array([1.7, 2.3, 3.9])
arr_int32 = arr.astype(np.int32)
print("Float to int32:", arr_int32)  # Truncation!

# Boolean array
arr_bool = np.array([0, 1, 0, 1], dtype=np.bool_)
print("Boolean array:", arr_bool)
```
```
# Output:
# int64 array: int64
# float32 array: float32
# Float to int32: [1 2 3]
# Boolean array: [False  True False  True]
```

### Example 5: Memory Efficiency — NumPy vs Python List

```python
import numpy as np
import sys

# Python list of 1000 integers
py_list = list(range(1000))
list_size = sys.getsizeof(py_list) + sum(sys.getsizeof(i) for i in py_list)

# NumPy array of 1000 int32
np_arr = np.arange(1000, dtype=np.int32)
np_size = np_arr.nbytes

print(f"Python list size: {list_size:,} bytes")
print(f"NumPy array size: {np_size:,} bytes")
print(f"Ratio: {list_size / np_size:.1f}x")

# Performance: vectorized vs loop
import time
large_list = list(range(10_000_000))
large_arr = np.arange(10_000_000, dtype=np.int64)

start = time.time()
squared_list = [x**2 for x in large_list]
list_time = time.time() - start

start = time.time()
squared_arr = large_arr ** 2
arr_time = time.time() - start

print(f"\nList comprehension: {list_time:.3f}s")
print(f"NumPy vectorized:   {arr_time:.3f}s")
print(f"Speedup: {list_time / arr_time:.1f}x")
```
```
# Output:
# Python list size: 46,012 bytes
# NumPy array size: 4,000 bytes
# Ratio: 11.5x
# 
# List comprehension: 1.247s
# NumPy vectorized:   0.032s
# Speedup: 39.0x
```

### Example 6: Creating Identity and Diagonal Arrays

```python
import numpy as np

# Identity matrix
I = np.eye(4)
print("Identity (4x4):\n", I)

# Diagonal array
D = np.diag([10, 20, 30])
print("\nDiagonal:\n", D)

# Extract diagonal from existing array
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
diag = np.diag(arr)
print("\nDiagonal of existing array:", diag)
```
```
# Output:
# Identity (4x4):
#  [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]]
# 
# Diagonal:
#  [[10  0  0]
#  [ 0 20  0]
#  [ 0  0 30]]
# 
# Diagonal of existing array: [1 5 9]
```

## Common Mistakes

1. **Mixing Data Types:** Passing a list like `[1, 2, 3.5]` creates a float64 array because all elements must be the same type. NumPy will upcast to accommodate all values.

2. **Confusing `np.arange` with `np.linspace`:** `np.arange(0, 5, 0.5)` creates `[0.0, 0.5, 1.0, ...]` while `np.linspace(0, 5, 11)` creates 11 evenly spaced points between 0 and 5 inclusive.

3. **Copy vs View:** Assigning an array slice to a new variable creates a view, not a copy. Modifying the view modifies the original. Use `.copy()` to create an independent copy.

4. **Forgetting Parentheses in Shape:** `np.zeros(3, 4)` is an error because shape must be a tuple: `np.zeros((3, 4))`.

5. **Integer Division in `np.arange`:** `np.arange(0, 1, 0.1)` may not end exactly at 1 due to floating-point precision. Use `np.linspace` for guaranteed endpoint inclusion.

6. **Assuming Default Dtype:** `np.zeros((3, 3))` defaults to `float64`, not integer. Use `np.zeros((3, 3), dtype=int)` for integers.

7. **Confusing `size` and `shape[0]`:** `size` is the total number of elements across all dimensions. `shape[0]` is just the first dimension size.

## Interview Questions

### Beginner

1. **Q:** How do you create a NumPy array from a Python list?
   **A:** Use `np.array([1, 2, 3])`. The list is passed as the first argument.

2. **Q:** What is the difference between `np.zeros((3, 4))` and `np.empty((3, 4))`?
   **A:** `np.zeros` fills the array with zeros. `np.empty` allocates memory without initializing values, so it contains whatever data was in that memory location (faster but unpredictable).

3. **Q:** What attributes does a NumPy array have?
   **A:** `ndim` (number of dimensions), `shape` (size in each dimension), `dtype` (data type of elements), `size` (total number of elements).

4. **Q:** How do you create a sequence from 0 to 9 in NumPy?
   **A:** `np.arange(10)` creates `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`.

5. **Q:** What does `np.linspace(0, 1, 5)` produce?
   **A:** It produces 5 evenly spaced numbers between 0 and 1 inclusive: `[0., 0.25, 0.5, 0.75, 1.]`.

### Intermediate

1. **Q:** Why are NumPy arrays more memory efficient than Python lists?
   **A:** NumPy arrays store elements in contiguous memory with a fixed dtype, avoiding Python object overhead. A Python list stores pointers to Python objects (each with refcount, type, etc.), while NumPy stores raw bytes.

2. **Q:** How does NumPy handle mixed data types when creating an array?
   **A:** NumPy upcasts all elements to the highest common data type. For example, `np.array([1, 2, 3.0])` creates a float64 array.

3. **Q:** What is the difference between `np.arange(1, 10, 2)` and `np.linspace(1, 10, 5)`?
   **A:** `np.arange` uses a step size (produces `[1, 3, 5, 7, 9]`). `np.linspace` uses a count (produces 5 evenly spaced points from 1 to 10 inclusive: `[1., 3.25, 5.5, 7.75, 10.]`).

4. **Q:** What happens if you pass a single integer to `np.array(5)` vs `np.array([5])`?
   **A:** `np.array(5)` creates a 0-dimensional array (shape `()`), while `np.array([5])` creates a 1-dimensional array (shape `(1,)`).

5. **Q:** How do you check the memory usage of a NumPy array?
   **A:** Use `arr.nbytes` for the total memory consumed by the array data. Multiply by 8 for bits or compare with `sys.getsizeof(arr)` which includes array object overhead.

### Advanced

1. **Q:** How does NumPy handle memory alignment and what is its effect on performance?
   **A:** NumPy ensures proper memory alignment (e.g., 4-byte alignment for float32, 8-byte for float64) which allows CPU SIMD instructions to process multiple elements simultaneously. Misaligned arrays can cause significant performance degradation.

2. **Q:** Explain the internal memory layout of a NumPy array (ndarray object).
   **A:** The ndarray consists of a Python object header, a pointer to a block of contiguous memory (data buffer), shape tuple, strides tuple, dtype object, and flags. Strides define how many bytes to skip to move to the next element along each dimension.

3. **Q:** What is the difference between C-order (row-major) and Fortran-order (column-major) arrays, and how does it affect performance?
   **A:** C-order stores rows contiguously; Fortran-order stores columns contiguously. Iterating in the order of memory layout is faster due to cache locality. Use `order='F'` in creation functions for Fortran order. Transposing can change which operations are fast.

## Practice Problems

### Easy

1. Create a 1D array of 10 zeros and set the 5th element to 1.

2. Create a 3x3 identity matrix using `np.eye`.

3. Generate an array of 20 evenly spaced numbers between 0 and 100.

4. Create a 2D array of shape (4, 5) filled with the value 7.

5. Create an array `[10, 20, 30, 40, 50]` and check its dtype, shape, and ndim.

### Medium

1. Create an array of 100 evenly spaced points from -pi to pi. Then compute the sine of each point.

2. Generate a 5x5 array of random integers between 1 and 100. Find its min, max, mean, and standard deviation.

3. Create a 1D array from 0 to 99, then reshape it into a 10x10 array. Extract the diagonal.

4. Given `arr = np.array([1, 2, 3, 4, 5])`, create a new array of squares using broadcasting (not a loop). Time it against a list comprehension.

5. Create a 3x4 array of float32 zeros, then change its dtype to int32 without changing the underlying data.

### Hard

1. Implement a function that creates a checkerboard pattern of shape (n, n) with alternating 0s and 1s without using loops.

2. Compare the memory and speed of creating a Python list of 1 million integers vs a NumPy int32 array. Measure the time to compute the sum of squares for both.

3. Create a 4D array representing a batch of 10 RGB images of size 32x32. Extract the red channel of the 3rd image. Explain the strides of this array.

## Solutions

### Easy Solutions

```python
# 1
arr = np.zeros(10)
arr[4] = 1
print(arr)

# 2
I = np.eye(3)
print(I)

# 3
arr = np.linspace(0, 100, 20)
print(arr)

# 4
arr = np.full((4, 5), 7)
print(arr)

# 5
arr = np.array([10, 20, 30, 40, 50])
print(arr.dtype, arr.shape, arr.ndim)
```

### Medium Solutions

```python
# 1
x = np.linspace(-np.pi, np.pi, 100)
y = np.sin(x)
print(y[:5])

# 2
rand_arr = np.random.randint(1, 101, size=(5, 5))
print("Min:", rand_arr.min(), "Max:", rand_arr.max())
print("Mean:", rand_arr.mean(), "Std:", rand_arr.std())

# 3
arr = np.arange(100).reshape(10, 10)
diag = np.diag(arr)
print(diag)

# 4
arr = np.array([1, 2, 3, 4, 5])
squared = arr ** 2
print(squared)

# 5
arr = np.zeros((3, 4), dtype=np.float32)
arr_int = arr.view(np.int32)
print(arr_int)
```

### Hard Solutions

```python
# 1 Checkerboard
def checkerboard(n):
    return np.indices((n, n)).sum(axis=0) % 2

print(checkerboard(6))

# 2 Memory comparison
py_list = list(range(1_000_000))
np_arr = np.arange(1_000_000, dtype=np.int32)
import sys
list_mem = sys.getsizeof(py_list) + sum(sys.getsizeof(i) for i in py_list[:100]) * 10000
print(f"List: ~{list_mem:,} bytes, NumPy: {np_arr.nbytes:,} bytes")

# 3 Batch of images
batch = np.random.rand(10, 32, 32, 3) * 255
batch = batch.astype(np.uint8)
image3_red = batch[2, :, :, 0]
print("Shape:", image3_red.shape)
print("Strides:", batch.strides)
```

## Related Concepts

- Python list data structure
- Python `array` module
- `memoryview` and buffer protocol

## Next Concepts

- Array indexing and slicing (PYT-067)
- Array operations and broadcasting (PYT-068)
- Linear algebra with NumPy (PYT-069)

## Summary

NumPy arrays provide a powerful, efficient foundation for numerical computing in Python. Key creation functions include `np.array`, `np.zeros`, `np.ones`, `np.empty`, `np.arange`, and `np.linspace`. Every array has attributes `ndim`, `shape`, `dtype`, and `size` that describe its structure. NumPy arrays are significantly more memory-efficient and faster for numerical operations than Python lists, making them essential for data science and machine learning workloads.

## Key Takeaways

- Use `np.array()` to convert Python lists to NumPy arrays
- `np.zeros`, `np.ones`, `np.arange`, and `np.linspace` are essential creation functions
- `shape`, `dtype`, `ndim`, and `size` describe array structure
- NumPy arrays are homogeneous, contiguous, and memory-efficient
- Vectorized operations on arrays are 10-100x faster than Python loops
- Always use `.copy()` explicitly to avoid unintended view mutations
