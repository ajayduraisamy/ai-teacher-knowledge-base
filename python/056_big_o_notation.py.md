# Concept: Big O Notation

## Concept ID

PYT-056

## Difficulty

Intermediate

## Domain

Python

## Module

Data Structures and Algorithms

## Learning Objectives

- Define Big O notation and its role in algorithm analysis
- Identify common time complexities: O(1), O(log n), O(n), O(n log n), O(n²), O(2^n)
- Analyze the time and space complexity of Python code
- Determine the complexity of built-in Python operations on lists, dicts, sets
- Compare different algorithms by their asymptotic efficiency
- Understand worst-case vs average-case complexity

## Prerequisites

- Basic Python programming (loops, functions, lists, dicts)
- High school mathematics (logarithms, exponents, sequences)

## Definition

Big O notation is a mathematical notation that describes the limiting behavior of a function as its argument approaches infinity. In computer science, it characterizes the time or space complexity of an algorithm in terms of the input size `n`. It provides a high-level understanding of how an algorithm scales, ignoring constant factors and lower-order terms.

## Intuition

Big O answers the question: "What happens when the input gets really large?" If your algorithm takes 1 second for 1000 items and 4 seconds for 2000 items, it might be O(n²). If it takes 2 seconds for both, it might be O(1). Big O helps you choose algorithms that will still work when your data grows from thousands to millions of items.

## Why This Concept Matters

Without understanding Big O, you might write code that works fine on small test data but grinds to a halt in production. Choosing an O(n log n) sort over O(n²) can mean the difference between 2 seconds and 200 hours for a million items. Big O is the universal language for discussing algorithm efficiency — it is essential for system design interviews, database query optimization, and building scalable systems.

## Real World Examples

1. **Database Indexing:** B-tree lookups are O(log n), enabling fast queries on billions of rows.
2. **Search Engines:** Inverted index queries are O(1) per term, making Google instant.
3. **Graph Algorithms:** Dijkstra's shortest path (O(E log V)) powers GPS navigation.
4. **Caching:** Hash-map lookups (O(1)) make Redis lightning fast.
5. **Compression:** Huffman coding uses O(n log n) priority queues.

## AI/ML Relevance

- **Training Complexity:** Gradient descent is typically O(n * epochs * features) per iteration.
- **Inference Latency:** Tree-based models are O(depth) for prediction, neural nets are O(weights).
- **k-NN Search:** Brute force is O(n * d); KD-trees reduce to O(log n) on average.
- **Matrix Operations:** Matrix multiplication is O(n³) naively, ~O(n^2.8) with Strassen.
- **Transformer Attention:** Self-attention is O(n² * d), motivating sparse attention research.

## Code Examples

### Example 1: O(1) — Constant Time

```python
def get_first(items):
    return items[0]

def lookup(d, key):
    return d.get(key)

data = list(range(1000000))
get_first(data)  # O(1) — always 1 step
lookup({"a": 1, "b": 2}, "a")  # O(1) — hash table
# Output: 0, 1
```

### Example 2: O(log n) — Logarithmic Time

```python
import bisect

def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

arr = list(range(1000000))
# bisect_left also does O(log n)
print(binary_search(arr, 777777))
print(bisect.bisect_left(arr, 777777))
# Output: 777777, 777777
```

### Example 3: O(n) — Linear Time

```python
def find_max(items):
    max_val = items[0]
    for item in items:
        if item > max_val:
            max_val = item
    return max_val

def contains(items, target):
    return target in items  # O(n) for list

data = list(range(1000000))
print(find_max(data))
print(contains(data, 999999))
# Output: 999999, True
```

### Example 4: O(n log n) — Log-Linear Time

```python
import time

# Timsort: Python's built-in sort is O(n log n)
data = list(range(1000000, 0, -1))
sorted_data = sorted(data)  # O(n log n)
print(sorted_data[:5])
print(sorted_data[-5:])
# Output: [1, 2, 3, 4, 5] [999996, 999997, 999998, 999999, 1000000]
```

### Example 5: O(n²) — Quadratic Time

```python
def has_duplicates_naive(items):
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j]:
                return True
    return False

def has_duplicates_fast(items):
    return len(items) != len(set(items))  # O(n)

data = list(range(10000))
data.append(42)
print(has_duplicates_naive(data))
print(has_duplicates_fast(data))
# Output: True, True
```

### Example 6: O(2^n) — Exponential Time

```python
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

def fibonacci_linear(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

import time
start = time.time()
print(f"fib(35) recursive: {fibonacci_recursive(35)} ({time.time() - start:.2f}s)")

start = time.time()
print(f"fib(35) linear: {fibonacci_linear(35)} ({time.time() - start:.2f}s)")
# Output:
# fib(35) recursive: 9227465 (2.85s)
# fib(35) linear: 9227465 (0.00s)
```

### Example 7: Python Data Structure Complexities

```python
# List operations
arr = [1, 2, 3, 4, 5]
# arr[i] → O(1)
# arr.append(x) → O(1) amortized
# arr.pop() → O(1)
# arr.pop(0) → O(n)
# arr.insert(0, x) → O(n)
# x in arr → O(n)
# len(arr) → O(1)

# Set/Dict operations
s = {1, 2, 3, 4, 5}
# x in s → O(1) average
# s.add(x) → O(1) average
# s.remove(x) → O(1) average

d = {"a": 1, "b": 2}
# d[key] → O(1) average
# key in d → O(1) average

print("All operations documented above — no runtime errors")
# Output: All operations documented above — no runtime errors
```

## Common Mistakes

1. **Confusing O(2n) with O(n):** Big O drops constants — O(2n) = O(n), O(100n) = O(n).
2. **Ignoring Space Complexity:** Optimizing time while using exponentially more memory can be worse.
3. **Assuming Worst Case Is Always Relevant:** For hash tables, worst case is O(n) but average is O(1).
4. **Forgetting the Input Size:** Saying an algorithm is "fast" without reference to input size.
5. **Not Considering Amortized Cost:** `list.append` is O(1) amortized but occasionally O(n) when resizing.
6. **Loop Nested Loop = Always O(n²):** Two loops over different inputs produce O(n*m), not O(n²).
7. **Assuming Built-in Operations Are Free:** `sorted()` in a loop creates hidden O(n log n) costs.

## Interview Questions

### Beginner

1. What does O(1) mean?
2. What is the time complexity of accessing a list element by index?
3. What is the time complexity of `x in list` vs `x in set`?
4. What is the time complexity of Python's `sorted()`?
5. What does "dropping constants" mean in Big O?

### Intermediate

1. Why is dictionary lookup O(1) on average?
2. Explain the time complexity of `list.append` — why is it amortized O(1)?
3. Compare O(n log n) and O(n²). At n=1,000,000, how much slower is O(n²)?
4. What is the space complexity of recursive Fibonacci?
5. How would you determine the complexity of `collections.Counter`?

### Advanced

1. Explain the Master Theorem for divide-and-conquer recurrences.
2. Prove that comparison-based sorting is Ω(n log n).
3. Given an algorithm with recurrence T(n) = 2T(n/2) + O(n), what is its complexity?

## Practice Problems

### Easy

1. **Complexity Classifier:** Given 6 code snippets, label each as O(1), O(n), O(n²).
2. **Find Maximum:** Write an O(n) function and an O(n log n) function to find max.
3. **Duplicate Checker:** Compare O(n²) vs O(n) duplicate detection.
4. **Binary vs Linear Search:** Benchmark both on a sorted list of 100000 elements.
5. **List vs Set Membership:** Compare `in` operator on list vs set of 100000 elements.

### Medium

1. **Nested Loops Analyzer:** Analyze and fix a triple-nested loop to O(n²).
2. **Matrix Diagonal:** Write a function to sum the diagonal of an n×n matrix in O(n).
3. **Anagram Checker:** Compare O(n log n) sort-based vs O(n) counter-based anagram detection.
4. **Prefix Sum:** Compute prefix sums in O(n) vs O(n²).
5. **Space vs Time:** Implement a memoized Fibonacci and compare time/space complexity.

### Hard

1. **Recurrence Solver:** Derive the complexity of `T(n) = 3T(n/4) + O(n)`.
2. **Amortized Analysis:** Design a dynamic array class and prove amortized O(1) append.
3. **Lower Bound Proof:** Prove that finding the minimum in an unsorted array is Ω(n).

## Solutions

### Solution to Easy 1: Complexity Classifier

```python
# O(1)
def first(items):
    return items[0] if items else None

# O(n)
def sum_all(items):
    return sum(items)

# O(n²)
def pairs(items):
    return [(a, b) for a in items for b in items]
```

### Solution to Medium 1: Nested Loops Analyzer

```python
# Original: O(n*m) where m is the number of unique items
def count_occurrences_naive(items):
    counts = {}
    for item in items:
        count = 0
        for x in items:
            if x == item:
                count += 1
        counts[item] = count
    return counts  # O(n²)

# Optimized: O(n)
def count_occurrences_fast(items):
    from collections import Counter
    return dict(Counter(items))
```

### Solution to Hard 1: Recurrence Solver

```python
# T(n) = 3T(n/4) + O(n)
# Using Master Theorem: a = 3, b = 4, f(n) = n
# log_b(a) = log_4(3) ≈ 0.792
# f(n) = n = n^1, and 1 > 0.792
# Case 3: O(n)
# Answer: O(n)
```

## Related Concepts

- **Recursion (PYT-057):** Analyzing recursive algorithm complexity.
- **Sorting (PYT-064):** Understanding why sorting is O(n log n).
- **Searching (PYT-065):** Binary search O(log n) vs linear O(n).
- **Data Structures (PYT-058-062):** Complexity of operations on each structure.

## Next Concepts

- **057 — Recursion:** Recursive functions and their complexity analysis.
- **058 — Linked Lists:** Linear data structure with O(1) insert/delete at ends.

## Summary

Big O notation describes how algorithm runtime or memory grows with input size. Common complexities: O(1) (constant), O(log n) (logarithmic), O(n) (linear), O(n log n) (log-linear), O(n²) (quadratic), O(2^n) (exponential). Python's built-in data structures offer specific guarantees: list indexing O(1), membership O(n), dict/set operations O(1) average. Analyzing complexity helps choose the right algorithm for large-scale problems.

## Key Takeaways

- Big O describes asymptotic growth, not exact runtime.
- Drop constants and lower-order terms: O(2n + 100) = O(n).
- Python list: index/append O(1), insert/pop(0)/in O(n).
- Python dict/set: get/set/in O(1) average, O(n) worst case.
- Sorting is O(n log n) — Python uses Timsort.
- Always consider space complexity alongside time complexity.
- Profile to confirm — theoretical complexity does not always predict real-world performance perfectly.
