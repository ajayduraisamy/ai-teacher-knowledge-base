# Concept: Performance Optimization

## Concept ID

PYT-055

## Difficulty

Advanced

## Domain

Python

## Module

Concurrency and Performance

## Learning Objectives

- Profile Python code using `cProfile` and `timeit` to identify bottlenecks
- Use `__slots__` to reduce memory overhead of classes
- Understand the speed benefits of `PyPy` as a JIT-compiled Python interpreter
- Accelerate numerical code with `Numba` JIT compilation
- Use `array` and `memoryview` for memory-efficient data handling
- Optimize string operations with `join` instead of concatenation
- Apply profiling-driven optimization workflows

## Prerequisites

- Solid Python programming experience
- Understanding of built-in data structures (list, dict, str)
- Familiarity with NumPy basics (helpful but not required)
- Basic knowledge of CPU vs memory tradeoffs

## Definition

Performance optimization in Python is the systematic process of measuring, identifying, and eliminating bottlenecks in code. It involves using profiling tools like `cProfile` and `timeit` to locate slow sections, then applying optimization techniques such as using `__slots__`, compiling with Numba or Cython, choosing efficient data structures, and reducing memory allocations.

## Intuition

Optimizing Python without profiling is like trying to fix a leaky boat in the dark — you might patch a spot that is not leaking while ignoring the real hole. Always measure first. Once you know where the time is spent, you can apply targeted optimizations: use the right data structure, avoid unnecessary allocations, leverage just-in-time compilation, or rewrite hot paths in C.

## Why This Concept Matters

Python prioritizes developer productivity over raw speed, but real-world applications often need to process large data or meet latency requirements. Knowing how to profile and optimize ensures you can get the performance you need without abandoning Python. A well-optimized Python program can be 10-100x faster than a naive one, often approaching C-level speed for numerical code with Numba.

## Real World Examples

1. **Data Pipelines:** Optimizing ETL jobs that process terabytes of data daily.
2. **Web APIs:** Reducing p99 latency from 200ms to 20ms through profiling.
3. **Scientific Computing:** Accelerating simulation code 100x with Numba JIT.
4. **Game Development:** Optimizing Python game loops for 60 FPS.
5. **ML Training Pipelines:** Reducing data loading bottlenecks to maximize GPU utilization.

## AI/ML Relevance

- **Optimizing Data Pipelines:** Using `__slots__` in dataset classes, pre-fetching with threads.
- **Numba for ML Kernels:** JIT-compiling custom activation functions, loss functions, and metrics.
- **Memory View for Tensors:** Zero-copy slicing of large arrays with `memoryview`.
- **Profile Training Loops:** Identifying whether CPU data loading or GPU compute is the bottleneck.
- **Efficient Tokenization:** Pre-allocating arrays and using `join` for string tokenization.

## Code Examples

### Example 1: Profiling with cProfile

```python
import cProfile
import pstats

def slow_function():
    total = 0
    for i in range(1000000):
        total += i ** 2
    return total

def fast_function():
    return sum(i * i for i in range(1000000))

profiler = cProfile.Profile()
profiler.enable()
slow_function()
fast_function()
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats("cumtime").print_stats(10)
# Output: Profile showing slow_function taking ~0.1s and fast_function ~0.08s
```

### Example 2: Timing with timeit

```python
import timeit

setup = "data = list(range(1000))"

stmt_squares = "[x * x for x in data]"
stmt_map = "list(map(lambda x: x * x, data))"

t1 = timeit.timeit(stmt_squares, setup, number=10000)
t2 = timeit.timeit(stmt_map, setup, number=10000)

print(f"List comprehension: {t1:.3f}s")
print(f"map + list:         {t2:.3f}s")
# Output (varies):
# List comprehension: 0.512s
# map + list:         0.618s
```

### Example 3: __slots__ Memory Optimization

```python
import sys

class PointNoSlots:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class PointWithSlots:
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x = x
        self.y = y

p1 = PointNoSlots(1, 2)
p2 = PointWithSlots(1, 2)

print(f"No __slots__: {sys.getsizeof(p1)} bytes (dict: {sys.getsizeof(p1.__dict__)})")
print(f"With __slots__: {sys.getsizeof(p2)} bytes")
# Output:
# No __slots__: 48 bytes (dict: 120 bytes)
# With __slots__: 40 bytes
```

### Example 4: String Concatenation vs join

```python
import timeit

n = 10000
strings = ["hello"] * n

def concat():
    result = ""
    for s in strings:
        result += s
    return result

def join_method():
    return "".join(strings)

t1 = timeit.timeit(concat, number=100)
t2 = timeit.timeit(join_method, number=100)

print(f"Concatenation: {t1:.3f}s")
print(f"join:          {t2:.3f}s")
# Output:
# Concatenation: 0.145s
# join:          0.003s
```

### Example 5: Numba JIT Acceleration

```python
import numba
import time

def sum_squares(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

@numba.jit(nopython=True)
def sum_squares_jit(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

# Warm up JIT
sum_squares_jit(10)

n = 10_000_000
start = time.time()
result1 = sum_squares(n)
print(f"Pure Python: {time.time() - start:.3f}s")

start = time.time()
result2 = sum_squares_jit(n)
print(f"Numba JIT:   {time.time() - start:.3f}s")
# Output:
# Pure Python: 1.250s
# Numba JIT:   0.035s
```

### Example 6: array vs list

```python
import array
import sys
import timeit

n = 100000
py_list = list(range(n))
arr = array.array("i", range(n))

print(f"List size: {sys.getsizeof(py_list) + n * 28} bytes (approx)")
print(f"Array size: {sys.getsizeof(arr) + n * 4} bytes (approx)")

def sum_list():
    return sum(py_list)

def sum_array():
    return sum(arr)

t1 = timeit.timeit(sum_list, number=1000)
t2 = timeit.timeit(sum_array, number=1000)
print(f"Sum list:  {t1:.3f}s")
print(f"Sum array: {t2:.3f}s")
# Output:
# List size: 2800048 bytes (approx)
# Array size: 400056 bytes (approx)
# Sum list:  0.385s
# Sum array: 0.298s
```

### Example 7: memoryview for Zero-Copy Slicing

```python
import array

data = array.array("i", range(1000000))

# Without memoryview — creates a copy
slice_copy = data[100000:200000]

# With memoryview — zero-copy view
mv = memoryview(data)
slice_view = mv[100000:200000]

print(f"Copy type: {type(slice_copy)}")
print(f"View type: {type(slice_view)}")
print(f"First element: {slice_view[0]}")

# Cast to bytes for efficient I/O
bytes_view = mv.cast("b")
print(f"Bytes view length: {len(bytes_view)}")
# Output:
# Copy type: <class 'array.array'>
# View type: <class 'memoryview'>
# First element: 100000
# Bytes view length: 4000000
```

## Common Mistakes

1. **Optimizing Without Profiling:** "Pre-optimizing" code that is not the bottleneck wastes time and may reduce readability.
2. **Using `+` for String Concatenation in Loops:** Each `+` creates a new string object — O(n²) time.
3. **Not Using Local Variable References:** Accessing local variables is faster than global or attribute lookups.
4. **Overusing `__slots__`:** Using `__slots__` in classes with few instances adds complexity for minimal gain.
5. **Ignoring Memory Allocations:** Frequent allocations in hot loops cause GC pressure and slow down code.
6. **Assuming PyPy Works Everywhere:** PyPy does not support all C extensions (e.g., NumPy is limited).
7. **Misunderstanding Timeit:** Running `timeit` with too few repetitions or without isolating setup code.

## Interview Questions

### Beginner

1. What is `cProfile` used for?
2. How do you use `timeit` to measure a small code snippet?
3. What does `__slots__` do in a class?
4. Why is `"".join(list_of_strings)` faster than string concatenation in a loop?
5. What is the difference between `array.array` and a list?

### Intermediate

1. How does PyPy achieve faster execution than CPython?
2. Explain how Numba's `@jit` decorator works.
3. What is a memoryview and when would you use it?
4. How would you profile a multi-threaded Python application?
5. What is the trade-off between using `__slots__` and dynamic attribute assignment?

### Advanced

1. Explain how CPython's reference counting and garbage collection affect performance.
2. How would you use Cython to optimize a hot loop? Show example.
3. Design a profiling system that can identify memory leaks in long-running Python services.

## Practice Problems

### Easy

1. **Profile Squares:** Profile two ways to compute squares of 1-1000: list comprehension vs `map`.
2. **Time It:** Use `timeit` to compare `time.sleep(0.1)` vs `pass` to see timing overhead.
3. **Join vs Concat:** Benchmark concatenating 10000 strings with `+` vs `"".join()`.
4. **Local vs Global:** Compare access speed of a global variable vs passing it as a parameter.
5. **Slots Experiment:** Create 10000 instances of a class with and without `__slots__`, compare memory.

### Medium

1. **Profile a Web Request:** Use `cProfile` on a simple HTTP request and identify the top 3 bottlenecks.
2. **Numba Mandelbrot:** Implement the Mandelbrot set with and without Numba, compare speed.
3. **Array vs List Filter:** Compare filtering a list of 1M ints using list comprehension vs `array.array`.
4. **String Builder:** Implement an efficient string builder class that pre-allocates a buffer.
5. **Profile a Data Pipeline:** Profile a CSV reading and aggregation pipeline, optimize the slowest step.

### Hard

1. **Cython Extension:** Write a Cython extension for a Fibonacci function and compare to pure Python.
2. **Custom Allocator:** Implement a simple memory pool allocator for small objects.
3. **Profile and Optimize a Real Project:** Take an existing open-source Python tool, profile it, and submit a performance improvement PR.

## Solutions

### Solution to Easy 1: Profile Squares

```python
import cProfile

def with_list_comp():
    return [x * x for x in range(1000)]

def with_map():
    return list(map(lambda x: x * x, range(1000)))

profiler = cProfile.Profile()
profiler.enable()
for _ in range(10000):
    with_list_comp()
    with_map()
profiler.disable()

import pstats
pstats.Stats(profiler).sort_stats("cumtime").print_stats(5)
# Output: Profile showing cumulative times for both approaches
```

### Solution to Medium 1: Profile a Web Request

```python
import cProfile
import urllib.request

def fetch_url():
    with urllib.request.urlopen("http://example.com") as resp:
        return resp.read()

profiler = cProfile.Profile()
profiler.enable()
fetch_url()
profiler.disable()

import pstats
pstats.Stats(profiler).sort_stats("time").print_stats(10)
# Output: Top bottlenecks shown (likely SSL/TLS handshake, socket operations)
```

### Solution to Hard 1: Cython Fibonacci

```cython
# fibonacci.pyx
def fib(int n):
    cdef int a = 0, b = 1, i
    for i in range(n):
        a, b = b, a + b
    return a

# setup.py
# from distutils.core import setup
# from Cython.Build import cythonize
# setup(ext_modules=cythonize("fibonacci.pyx"))
```

## Related Concepts

- **Big O Notation (PYT-056):** Analyzing algorithmic complexity before optimizing.
- **Data Structures (PYT-058 through PYT-062):** Choosing the right data structure for performance.
- **Concurrency Patterns (PYT-054):** Parallelism for performance gains.

## Next Concepts

- **056 — Big O Notation:** Formal analysis of algorithm efficiency.
- **057 — Recursion:** Understanding recursion depth and optimization.

## Summary

Performance optimization is a systematic process: profile to find bottlenecks, then apply targeted optimizations. Key techniques include using `__slots__` for memory efficiency, Numba JIT for numerical acceleration, `array` and `memoryview` for compact data representation, and `str.join` for efficient string building. Always measure before and after optimizing.

## Key Takeaways

- Profile first with `cProfile` or `timeit` — never guess where bottlenecks are.
- Use `__slots__` to reduce memory in classes with many instances.
- Use Numba's `@jit` decorator for numerical hot loops (10-100x speedup).
- Prefer `"".join(strings)` over `+=` in loops.
- Use `array.array` for homogeneous numeric data — smaller memory, faster access.
- Use `memoryview` for zero-copy slicing and efficient I/O.
- Consider PyPy for pure Python programs without C extensions.
