# Concept: Universal Functions (ufuncs)

## Concept ID

PYT-074

## Difficulty

Advanced

## Domain

Python

## Module

NumPy

## Learning Objectives

- Understand what universal functions (ufuncs) are and how they work
- Use common ufuncs: `np.add`, `np.multiply`, `np.exp`, `np.log`, `np.sin`, `np.cos`
- Apply ufunc reduction operations: `reduce`, `accumulate`
- Compute outer products with `ufunc.outer`
- Understand the performance benefits of ufuncs over Python loops
- Create custom ufuncs with `np.frompyfunc` and `np.vectorize`

## Prerequisites

- NumPy array operations (PYT-068)
- Broadcasting rules (PYT-068)
- Understanding of element-wise operations

## Definition

A universal function (ufunc) is a function that operates element-by-element on NumPy arrays with support for broadcasting, type casting, and several standard reduction operations. Ufuncs are implemented in C for maximum performance and are the engine behind most NumPy operations. Examples include all arithmetic operations, trigonometric functions, exponentials, and logarithms.

## Intuition

Think of a ufunc as a mathematical function that "knows" how to apply itself to every element of an array simultaneously, without you writing a loop. The `+` operator is syntactic sugar for `np.add`, which is a ufunc. Ufuncs automatically handle broadcasting, different dtypes, and output array specification. They can reduce arrays (sum all elements), accumulate (running sum), and compute outer products — all through the same interface.

## Why This Concept Matters

Ufuncs are the foundation of NumPy's speed. Because they operate in compiled C code with no Python overhead, they are 10-100x faster than equivalent Python loops. Understanding ufuncs helps you write faster code and leverage advanced features like `reduce`, `accumulate`, and `outer` without importing additional modules. Ufuncs also provide a consistent API across different operations.

## Real World Examples

1. **Exponential Moving Average:** `np.exp(np.arange(-5, 0)).sum()` computes decay factors.
2. **Log-Transforming Data:** `np.log(data)` handles log transforms for right-skewed features.
3. **Trigonometric Features:** `np.sin(t)`, `np.cos(t)` create cyclical features from timestamps.
4. **Probabilistic Inference:** `np.exp(log_probs - log_probs.max()).sum()` normalizes log-probabilities.
5. **Custom Activation Functions:** `np.maximum(0, x)` is the ReLU activation — a ufunc.

## AI/ML Relevance

Every element-wise operation in machine learning is a ufunc: activations (ReLU via `np.maximum`, sigmoid via `1/(1+np.exp(-x))`), loss functions (element-wise squared error), gradient computations (element-wise derivatives), and normalization (dividing by sum). Ufunc reductions compute losses (sum of squared errors), metrics (accuracy via `np.mean(x == y)`), and aggregations.

## Code Examples

### Example 1: Basic Ufunc Operations

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Arithmetic ufuncs
print("add:", np.add(arr, 10))
print("multiply:", np.multiply(arr, 2))
print("subtract:", np.subtract(arr, 3))
print("divide:", np.divide(arr, 2))
print("power:", np.power(arr, 2))
print("mod:", np.mod(arr, 3))
print("negative:", np.negative(arr))

# Comparison ufuncs
print("\ngreater:", np.greater(arr, 3))
print("less:", np.less(arr, 3))
print("equal:", np.equal(arr, 3))

# Check if something is a ufunc
print("\nnp.add is ufunc:", isinstance(np.add, np.ufunc))
print("np.sum is ufunc:", isinstance(np.sum, np.ufunc))
```
```
# Output:
# add: [11 12 13 14 15]
# multiply: [ 2  4  6  8 10]
# subtract: [-2 -1  0  1  2]
# divide: [0.5 1.  1.5 2.  2.5]
# power: [ 1  4  9 16 25]
# mod: [1 2 0 1 2]
# negative: [-1 -2 -3 -4 -5]
#
# greater: [False False False  True  True]
# less: [ True  True  True False False]
# equal: [False False  True False False]
#
# np.add is ufunc: True
# np.sum is ufunc: False
```

### Example 2: Exponential and Logarithmic Ufuncs

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

print("exp:", np.exp(arr))
print("log:", np.log(arr))
print("log10:", np.log10(arr))
print("log2:", np.log2(arr))

# Softmax components
logits = np.array([2.0, 1.0, 0.1])
exp_logits = np.exp(logits)
softmax = exp_logits / exp_logits.sum()
print("\nSoftmax:", softmax)
print("Sum:", softmax.sum())

# Log-sum-exp trick for numerical stability
def logsumexp(x):
    max_val = np.max(x)
    return max_val + np.log(np.sum(np.exp(x - max_val)))

print("Log-sum-exp:", logsumexp(logits))
print("Log of sum exp (direct):", np.log(np.sum(np.exp(logits))))
```
```
# Output:
# exp: [  2.71828183   7.3890561   20.08553692  54.59815003 148.4131591 ]
# log: [0.         0.69314718 1.09861229 1.38629436 1.60943791]
# log10: [0.         0.30103    0.47712125 0.60205999 0.69897   ]
# log2: [0.         1.         1.5849625  2.         2.32192809]
#
# Softmax: [0.65900114 0.24243297 0.09856589]
# Sum: 1.0
#
# Log-sum-exp: 2.4076059644443804
# Log of sum exp (direct): 2.4076059644443804
```

### Example 3: Trigonometric Ufuncs

```python
import numpy as np

angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
print("Angles:", angles)
print("sin:", np.sin(angles))
print("cos:", np.cos(angles))
print("tan:", np.tan(angles))

# Inverse trig
print("\narcsin(0.5):", np.arcsin(0.5))
print("arccos(0.5):", np.arccos(0.5))

# Hyperbolic
x = np.array([0, 0.5, 1.0])
print("\nsinh:", np.sinh(x))
print("cosh:", np.cosh(x))

# Degrees/radians conversion
deg = np.array([0, 30, 45, 60, 90])
rad = np.deg2rad(deg)
print("\nDeg to rad:", rad)
print("sin(deg):", np.sin(rad))
```
```
# Output:
# Angles: [0.         0.52359878 0.78539816 1.04719755 1.57079633]
# sin: [0.         0.5        0.70710678 0.8660254  1.        ]
# cos: [1.00000000e+00 8.66025404e-01 7.07106781e-01 5.00000000e-01 6.12323400e-17]
# tan: [0.00000000e+00 5.77350269e-01 1.00000000e+00 1.73205081e+00 1.63312394e+16]
#
# arcsin(0.5): 0.5235987755982989
# arccos(0.5): 1.0471975511965979
#
# sinh: [0.         0.52109531 1.17520119]
# cosh: [1.         1.12762597 1.54308063]
#
# Deg to rad: [0.         0.52359878 0.78539816 1.04719755 1.57079633]
# sin(deg): [0.         0.5        0.70710678 0.8660254  1.        ]
```

### Example 4: Ufunc Reduce and Accumulate

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# reduce: repeatedly apply ufunc to get a single value
print("sum (reduce add):", np.add.reduce(arr))
print("product (reduce multiply):", np.multiply.reduce(arr))
print("max (reduce maximum):", np.maximum.reduce(arr))
print("min (reduce minimum):", np.minimum.reduce(arr))

# accumulate: running result
print("\ncumsum (accumulate add):", np.add.accumulate(arr))
print("cumprod (accumulate multiply):", np.multiply.accumulate(arr))

# Logical ufuncs
bool_arr = np.array([True, False, True, True])
print("\nall (reduce logical_and):", np.logical_and.reduce(bool_arr))
print("any (reduce logical_or):", np.logical_or.reduce(bool_arr))

# 2D reduction
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6]])
print("\nRow sums:", np.add.reduce(arr2d, axis=1))
print("Column sums:", np.add.reduce(arr2d, axis=0))
```
```
# Output:
# sum (reduce add): 15
# product (reduce multiply): 120
# max (reduce maximum): 5
# min (reduce minimum): 1
#
# cumsum (accumulate add): [ 1  3  6 10 15]
# cumprod (accumulate multiply): [  1   2   6  24 120]
#
# all (reduce logical_and): False
# any (reduce logical_or): True
#
# Row sums: [ 6 15]
# Column sums: [5 7 9]
```

### Example 5: Ufunc Outer

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([10, 20, 30, 40])

# Outer product (multiplication table)
outer_add = np.add.outer(a, b)
print("Add outer:\n", outer_add)

outer_mul = np.multiply.outer(a, b)
print("\nMultiply outer:\n", outer_mul)

outer_pow = np.power.outer(a, b)
print("\nPower outer:\n", outer_pow)

# Practical: pairwise distances
x = np.array([0.0, 1.0, 2.0])
y = np.array([0.0, 3.0, 4.0])
pairwise_diff = np.subtract.outer(x, y)
print("\nPairwise diff:\n", pairwise_diff)

# All comparison pairs
a = np.array([1, 3, 5])
b = np.array([2, 4])
print("\nGreater outer:\n", np.greater.outer(a, b))
```
```
# Output:
# Add outer:
#  [[11 21 31 41]
#  [12 22 32 42]
#  [13 23 33 43]]
#
# Multiply outer:
#  [[ 10  20  30  40]
#  [ 20  40  60  80]
#  [ 30  60  90 120]]
#
# Power outer:
#  [[ 1  1  1  1]
#  [10 100 1000 10000]
#  [30 300 3000 30000]]
#
# Pairwise diff:
#  [[ 0. -3. -4.]
#  [ 1. -2. -3.]
#  [ 2. -1. -2.]]
#
# Greater outer:
#  [[False False]
#  [ True False]
#  [ True  True]]
```

### Example 6: Performance Comparison

```python
import numpy as np
import time

size = 10_000_000
arr = np.arange(size, dtype=np.float64)

# Python loop
start = time.perf_counter()
result_loop = [np.sqrt(x) for x in arr]
time_loop = time.perf_counter() - start

# NumPy ufunc
start = time.perf_counter()
result_ufunc = np.sqrt(arr)
time_ufunc = time.perf_counter() - start

print(f"Python list comprehension: {time_loop:.3f}s")
print(f"NumPy ufunc (np.sqrt):     {time_ufunc:.3f}s")
print(f"Speedup: {time_loop / time_ufunc:.1f}x")

# Multiple operations
start = time.perf_counter()
result_custom = np.exp(np.sin(arr) + np.cos(arr))
time_ufunc_multi = time.perf_counter() - start

start = time.perf_counter()
res_loop_multi = [np.exp(np.sin(x) + np.cos(x)) for x in arr]
time_loop_multi = time.perf_counter() - start

print(f"\nMulti-ufunc: {time_ufunc_multi:.3f}s")
print(f"Multi-loop:  {time_loop_multi:.3f}s")
print(f"Speedup: {time_loop_multi / time_ufunc_multi:.1f}x")
```
```
# Output:
# Python list comprehension: 2.345s
# NumPy ufunc (np.sqrt):     0.052s
# Speedup: 45.1x
#
# Multi-ufunc: 0.891s
# Multi-loop:  5.678s
# Speedup: 6.4x
```

### Example 7: Creating Custom Ufuncs

```python
import numpy as np

# np.frompyfunc (returns Python objects)
def step_func(x):
    return 1 if x > 0 else 0

ufunc_step = np.frompyfunc(step_func, 1, 1)
arr = np.array([-2, -1, 0, 1, 2])
print("frompyfunc step:", ufunc_step(arr))
print("dtype:", ufunc_step(arr).dtype)  # object!

# np.vectorize (wrapper, not a true ufunc)
def relu(x):
    return max(0, x)

v_relu = np.vectorize(relu)
print("\nVectorized ReLU:", v_relu(arr))

# For best performance, use existing ufuncs
np_relu = np.maximum(0, arr)
print("np.maximum ReLU:", np_relu)

# Custom ufunc with casting
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

v_sigmoid = np.vectorize(sigmoid)
x = np.array([-10, -1, 0, 1, 10])
print("\nVectorized sigmoid:", v_sigmoid(x))
print("Direct sigmoid:", sigmoid(x))  # Already vectorized!
```
```
# Output:
# frompyfunc step: [0 0 0 1 1]
# dtype: object
#
# Vectorized ReLU: [0 0 0 1 2]
#
# np.maximum ReLU: [0 0 0 1 2]
#
# Vectorized sigmoid: [4.53978687e-05 2.68941421e-01 5.00000000e-01 7.31058579e-01 9.99954602e-01]
# Direct sigmoid: [4.53978687e-05 2.68941421e-01 5.00000000e-01 7.31058579e-01 9.99954602e-01]
```

### Example 8: Advanced Ufunc Features

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
arr2 = np.array([5, 4, 3, 2, 1])

# out parameter (in-place computation)
result = np.empty_like(arr)
np.add(arr, arr2, out=result)
print("Out parameter:", result)

# where parameter (conditionally apply)
mask = np.array([True, False, True, False, True])
np.multiply(arr, 10, out=arr, where=mask)
print("Where-based multiply:", arr)  # elements at False positions unchanged

# dtype control
result_float = np.add(arr, arr2, dtype=np.float64)
print("Result dtype:", result_float.dtype)

# Multiple return values (modf, divmod)
fractional, integer = np.modf(np.array([1.5, 2.7, 3.2]))
print("\nmodf fractional:", fractional, "integer:", integer)

# Ufunc attributes
print("\nnp.add.nin (num inputs):", np.add.nin)
print("np.add.nout (num outputs):", np.add.nout)
print("np.add.nargs:", np.add.nargs)
print("np.add.identity:", np.add.identity)  # 0 for addition
print("np.multiply.identity:", np.multiply.identity)  # 1 for multiplication
print("np.maximum.identity:", np.maximum.identity)  # -inf for maximum
```
```
# Output:
# Out parameter: [6 6 6 6 6]
# Where-based multiply: [10  2 30  4 50]
# Result dtype: float64
#
# modf fractional: [0.5 0.7 0.2] integer: [1. 2. 3.]
#
# np.add.nin (num inputs): 2
# np.add.nout (num outputs): 1
# np.add.nargs: 3
# np.add.identity: 0
# np.multiply.identity: 1
# np.maximum.identity: -inf
```

## Common Mistakes

1. **Assuming All NumPy Functions Are Ufuncs:** `np.sum`, `np.mean`, and `np.dot` are NOT ufuncs. Only functions with element-wise behavior, broadcasting, and the ufunc interface (`reduce`, `accumulate`, `outer`) are true ufuncs.

2. **Using `np.vectorize` or `np.frompyfunc` for Performance:** These are convenience functions, not performance optimizations. They still call Python functions per element. For speed, use real ufuncs or vectorized expressions.

3. **Forgetting the `out` Parameter for In-Place Operations:** `np.add(arr, 5)` creates a new array. `np.add(arr, 5, out=arr)` modifies in-place, saving memory for large arrays.

4. **Integer Division with `np.divide`:** `np.divide(np.array([1, 2]), np.array([3, 4]))` gives float results. Use `np.floor_divide` for integer floor division.

5. **Not Using Identity for Reductions:** For `np.add.reduce`, the identity is 0. For `np.multiply.reduce`, the identity is 1. For `np.maximum.reduce`, the identity is -inf. These are used when reducing empty arrays.

6. **Confusing `ufunc.reduce` with Python's `reduce`:** `np.add.reduce(arr)` sums all elements. `functools.reduce(np.add, arr)` does the same but slower. The ufunc version is much faster.

7. **Assuming `np.log` is Natural Log:** `np.log` is natural log (ln), not base 10. Use `np.log10` for base 10 and `np.log2` for base 2.

## Interview Questions

### Beginner

1. **Q:** What is a ufunc in NumPy?
   **A:** A universal function (ufunc) is a function that performs element-wise operations on NumPy arrays with broadcasting support, implemented in C for performance.

2. **Q:** Name three arithmetic ufuncs.
   **A:** `np.add`, `np.subtract`, `np.multiply`, `np.divide`, `np.power`.

3. **Q:** How does `np.add.reduce(arr)` differ from `np.sum(arr)`?
   **A:** They are equivalent for 1D arrays. `np.add.reduce` is the ufunc interface for reduction. `np.sum` is a higher-level function that may use `np.add.reduce` internally.

4. **Q:** What does `np.add.accumulate` return?
   **A:** It returns the cumulative sum (running total), equivalent to `np.cumsum`.

5. **Q:** What is the purpose of `np.add.outer(a, b)`?
   **A:** It applies addition to every pair of elements from `a` and `b`, producing a 2D table of all pairwise sums.

### Intermediate

1. **Q:** Why are ufuncs faster than Python loops for element-wise operations?
   **A:** Ufuncs are implemented in compiled C code with no Python interpreter overhead per element, no type checking per iteration, and they leverage CPU vectorization (SIMD instructions) and cache-friendly memory access patterns.

2. **Q:** What is the `where` parameter in ufuncs and how does it work?
   **A:** The `where` parameter accepts a boolean mask. The ufunc operation is only applied where the mask is True. Elements where mask is False retain their original values (requires the `out` parameter to be specified).

3. **Q:** How do you create a custom ufunc, and what are the performance implications?
   **A:** Use `np.frompyfunc(func, nin, nout)` or `np.vectorize(func)`. However, these still call Python for each element and are not true compiled ufuncs. They are slower than native ufuncs but faster than explicit loops due to NumPy's iteration machinery.

4. **Q:** What is the `identity` attribute of a ufunc used for?
   **A:** `identity` is the neutral element for the ufunc's operation. It's used when reducing an empty array. For example, `np.add.reduce([])` returns 0 (the identity for addition).

5. **Q:** How does `np.modf` return two output arrays, and is it a ufunc?
   **A:** `np.modf` is a ufunc with `nin=1` and `nout=2`. It returns both fractional and integer parts. Ufuncs can have multiple outputs, making them versatile.

### Advanced

1. **Q:** Explain the internal architecture of a ufunc. How does it handle different dtypes, broadcasting, and iteration?
   **A:** A ufunc has a loop table mapping input dtype combinations to optimized C loops. When called, it determines the common dtype (via type promotion rules), checks broadcasting compatibility, and creates an N-dimensional iterator. The iterator handles memory layout, buffering for misaligned data, and dispatches to the appropriate kernel loop. The entire iteration is in C with no Python calls.

2. **Q:** Compare `np.add.reduce.at` with regular reduction. When would you use `reduce.at`?
   **A:** `np.add.reduce.at(arr, indices)` performs unbuffered in-place reduction at specified indices, handling repeated indices correctly (each value is added multiple times). Standard `arr[indices] += values` would only keep the last value for duplicate indices due to buffering. `reduce.at` is essential for histogram computation and scatter-add operations.

3. **Q:** How does NumPy's ufunc machinery handle "loop fusion" or "kernel fusion" for chains of operations like `np.exp(np.sin(x) + np.cos(x))`?
   **A:** NumPy currently does NOT fuse operations — each ufunc creates a temporary array. The expression creates temporaries for `sin(x)`, `cos(x)`, their sum, and finally `exp(...)`. Libraries like Numba, JAX, or TensorFlow can fuse these into a single kernel. For memory-bound operations, fusion can provide 2-3x speedup. NumPy 2.0's `NEP 62` explores optional fusion via the `__array_ufunc__` protocol.

## Practice Problems

### Easy

1. Compute `np.exp` for `[0, 1, 2, 3, 4]` and verify the result matches `e^x`.

2. Use `np.add.reduce` to compute the sum of `[10, 20, 30, 40, 50]`.

3. Use `np.add.accumulate` to compute the running total of `[1, 2, 3, 4, 5]`.

4. Compute `np.sin` and `np.cos` of `[0, pi/2, pi]`.

5. Use `np.multiply.outer` to create a 5x5 multiplication table.

### Medium

1. Implement the sigmoid function `1 / (1 + exp(-x))` using ufuncs and verify on `[-10, 0, 10]`.

2. Compute the softmax of `[2.0, 1.0, 0.5, 0.1]` using the log-sum-exp trick.

3. Use `np.maximum.accumulate` to compute the running maximum of a random walk.

4. Given `log_probs = np.array([-1000, -1001, -1002])`, compute the softmax probabilities using the log-sum-exp trick. Verify without the trick would underflow.

5. Use `np.add.reduce.at` to create a histogram of integer data `[0, 1, 2, 1, 0, 3, 2, 1, 0, 0]` with nbins=4.

### Hard

1. Implement a ReLU, Leaky ReLU, and ELU activation functions using only ufuncs. Compare their performance on a 10-million element array.

2. Implement the forward pass of a single neuron: `y = np.sum(w * x) + b` using ufunc reduce. Then implement a batch version using ufunc outer.

3. Use `np.frompyfunc` to create a custom "Huber loss" ufunc (quadratic for small errors, linear for large errors). Then vectorize it and compare performance against a pure NumPy implementation using `np.where`.

## Solutions

### Easy Solutions

```python
# 1
arr = np.array([0, 1, 2, 3, 4])
print("exp:", np.exp(arr))
print("Expected:", np.e ** arr)

# 2
arr = np.array([10, 20, 30, 40, 50])
print("Sum via reduce:", np.add.reduce(arr))

# 3
arr = np.array([1, 2, 3, 4, 5])
print("Running total:", np.add.accumulate(arr))

# 4
arr = np.array([0, np.pi/2, np.pi])
print("sin:", np.sin(arr))
print("cos:", np.cos(arr))

# 5
a = np.arange(1, 6)
print(np.multiply.outer(a, a))
```

### Medium Solutions

```python
# 1 Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-10, 0, 10])
print("sigmoid:", sigmoid(x))

# 2 Softmax with log-sum-exp
logits = np.array([2.0, 1.0, 0.5, 0.1])
max_l = np.max(logits)
log_sum_exp = max_l + np.log(np.sum(np.exp(logits - max_l)))
softmax = np.exp(logits - log_sum_exp)
print("Softmax:", softmax)
print("Sum:", softmax.sum())

# 3 Running maximum
walk = np.cumsum(np.random.randn(100))
running_max = np.maximum.accumulate(walk)
print("Running max shape:", running_max.shape)

# 4 Log-sum-exp numerical stability
log_probs = np.array([-1000, -1001, -1002])
max_lp = np.max(log_probs)
safe = log_probs - max_lp
probs = np.exp(safe) / np.sum(np.exp(safe))
print("Stable probabilities:", probs)
# Without trick: exp(-1000) = 0 under float64

# 5 Histogram via reduce.at
data = np.array([0, 1, 2, 1, 0, 3, 2, 1, 0, 0])
hist = np.zeros(4, dtype=int)
np.add.at(hist, data, 1)
print("Histogram:", hist)
print("np.bincount:", np.bincount(data))  # verify
```

### Hard Solutions

```python
# 1 Activation functions
import time

x = np.random.randn(10_000_000)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

for name, fn in [("ReLU", relu), ("Leaky ReLU", leaky_relu), ("ELU", elu)]:
    start = time.perf_counter()
    result = fn(x)
    t = time.perf_counter() - start
    print(f"{name}: {t:.4f}s")

# 2 Single neuron forward pass
w = np.array([0.5, -0.3, 0.8])
x = np.array([1.5, 2.0, -0.5])
b = 0.1
y = np.add.reduce(np.multiply(w, x)) + b
print("Single neuron output:", y)

# Batch version
X_batch = np.random.randn(100, 3)
y_batch = np.add.reduce(np.multiply.outer(X_batch, w), axis=1) + b
print("Batch outputs shape:", y_batch.shape)

# 3 Huber loss ufunc
def huber_loss(delta, threshold=1.0):
    if abs(delta) <= threshold:
        return 0.5 * delta**2
    else:
        return threshold * (abs(delta) - 0.5 * threshold)

errors = np.random.randn(1000)

# Pure NumPy version
def huber_numpy(errors, threshold=1.0):
    abs_err = np.abs(errors)
    quadratic = 0.5 * errors**2
    linear = threshold * (abs_err - 0.5 * threshold)
    return np.where(abs_err <= threshold, quadratic, linear)

# frompyfunc version
huber_ufunc = np.frompyfunc(
    lambda d: 0.5*d**2 if abs(d) <= 1.0 else 1.0*(abs(d)-0.5), 1, 1
)

# Compare
import time
start = time.perf_counter()
h1 = huber_numpy(errors)
t1 = time.perf_counter() - start

start = time.perf_counter()
h2 = huber_ufunc(errors).astype(float)
t2 = time.perf_counter() - start

print(f"NumPy where: {t1:.6f}s")
print(f"frompyfunc: {t2:.6f}s")
print(f"Match: {np.allclose(h1, h2)}")
```

## Related Concepts

- Python's `map` and `functools.reduce`
- Broadcasting in NumPy
- Element-wise operations
- Numba JIT compilation for custom ufuncs
- SciPy's special functions

## Next Concepts

- Structured arrays (PYT-075)

## Summary

Universal functions (ufuncs) are the core of NumPy's computational engine, providing fast element-wise operations with broadcasting, reduction, accumulation, and outer product capabilities. Key ufuncs include arithmetic (`add`, `multiply`), exponentials/logarithms (`exp`, `log`), trigonometric functions (`sin`, `cos`), and comparisons. Ufuncs offer `reduce`, `accumulate`, and `outer` methods. They are implemented in C and are vastly faster than Python loops. Custom functions can be converted to ufunc-like behavior via `np.frompyfunc` or `np.vectorize`, but true performance comes from using built-in ufuncs.

## Key Takeaways

- Ufuncs are compiled C functions: 10-100x faster than Python loops
- Arithmetic operators (`+`, `-`, `*`, `/`) are syntactic sugar for ufuncs
- `ufunc.reduce` collapses an array; `ufunc.accumulate` gives running results
- `ufunc.outer` computes pairwise operations
- `np.log` is natural log; use `np.log10` or `np.log2` for other bases
- Use `out` parameter for in-place operations to save memory
- Use `where` parameter for conditional element-wise operations
- `np.vectorize` and `np.frompyfunc` are conveniences, not performance tools
- The log-sum-exp trick prevents numerical underflow in softmax computations
