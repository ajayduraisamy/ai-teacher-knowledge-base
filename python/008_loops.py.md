# Concept: Loops

## Concept ID

PYT-008

## Difficulty

BEGINNER

## Prerequisites

- PYT-002: Variables and Data Types
- PYT-005: Operators and Expressions
- PYT-007: Conditionals

## Domain

Python

## Module

Python Basics

## Learning Objectives

- Write `for` loops over `range()`, strings, lists, and other iterables
- Write `while` loops with correct termination conditions
- Use `break` to exit a loop early, `continue` to skip to the next iteration, and `pass` as a no-op placeholder
- Understand the `else` clause on loops (executes when no `break` occurs)
- Traverse multiple sequences in parallel with `zip()` and access both index and value with `enumerate()`
- Control iteration order with `reversed()` and `sorted()`
- Construct nested loops for multidimensional data

## Definition

Loops are control structures that repeatedly execute a block of code as long as (or for each time) a condition is met. Python provides two primary loop constructs: the `for` loop (iteration over a sequence or iterable) and the `while` loop (repetition until a condition becomes falsy). Loop control statements (`break`, `continue`, `pass`) modify flow within a loop, and the optional `else` clause runs when the loop terminates normally (without `break`).

## Intuition

Imagine you have a stack of 100 exam papers to grade. You could grade them one by one: pick a paper, grade it, set it aside, repeat until the stack is empty. That is a loop. In programming, loops automate repetitive tasks. Instead of writing 100 identical lines of code, you write one loop that repeats 100 times. Loops are the engine that powers data processing, waiting for events, and batch operations.

## Why This Concept Matters

Loops are the backbone of data processing. Every time you need to process a collection — sum numbers in a list, train a model over epochs, read lines from a file, send requests to multiple URLs — you need a loop. Without loops, code would be impossibly repetitive and rigid. Understanding how to control loops with `break`, `continue`, and `else` is essential for writing efficient, correct iteration logic.

## Real World Examples

1. **Summing numbers**: Loop through a list of sales figures to compute total revenue.
2. **Training an ML model**: A `for` loop runs over epochs; `break` stops early if loss plateaus.
3. **Reading a file**: `for line in file:` processes each line without loading the entire file.
4. **Web scraping**: Loop over a list of URLs, fetch each page, extract data.
5. **Progress monitoring**: A `while` loop checks a condition (e.g., GPU temperature below threshold) and logs metrics until the condition is met.

## AI/ML Relevance

- **Epoch loop**: The outer training loop is almost always a `for epoch in range(num_epochs)`.
- **Batch iteration**: Data loaders yield batches; a `for` loop iterates over them.
- **Grid search**: Nested `for` loops over hyperparameter values.
- **Early stopping**: A `while` loop or `break` inside an epoch loop halts training when validation loss stops improving.
- **Data augmentation**: Loop over each sample in a dataset, apply random transformations.

## Code Examples

### Example 1: `for` loop with `range()`

`range(stop)` generates integers from 0 up to (but not including) `stop`. `range(start, stop, step)` supports custom start and step values.

```python
for i in range(5):
    print(i, end=" ")
# Output: 0 1 2 3 4

print()
for i in range(2, 10, 2):
    print(i, end=" ")
# Output: 2 4 6 8

print()
for i in range(10, 0, -1):
    print(i, end=" ")
# Output: 10 9 8 7 6 5 4 3 2 1
```

### Example 2: `while` loop

A `while` loop continues as long as its condition evaluates to `True`. Be careful to update the condition inside the loop to avoid infinite loops.

```python
count = 0
while count < 3:
    print(f"Count: {count}")
    count += 1
# Output: Count: 0
# Output: Count: 1
# Output: Count: 2

# Infinite loop (avoid!)
# while True:
#     print("Press Ctrl+C to stop")
```

### Example 3: `break`, `continue`, and `pass`

- `break` exits the loop immediately.
- `continue` skips the rest of the current iteration and moves to the next.
- `pass` does nothing — used as a placeholder when syntax requires a statement.

```python
for i in range(10):
    if i == 5:
        break
    print(i, end=" ")
# Output: 0 1 2 3 4

print()
for i in range(5):
    if i == 2:
        continue
    print(i, end=" ")
# Output: 0 1 3 4

for i in range(3):
    pass  # Placeholder — does nothing
```

### Example 4: `else` clause on loops

The `else` block after a `for` or `while` loop runs **only if** the loop terminated normally (i.e., no `break` was encountered). This is useful for search loops.

```python
numbers = [1, 3, 5, 7, 9]
target = 6

for n in numbers:
    if n == target:
        print(f"Found {target}")
        break
else:
    print(f"{target} not found")
# Output: 6 not found

# Another example: checking primality
num = 17
for i in range(2, int(num ** 0.5) + 1):
    if num % i == 0:
        print(f"{num} is composite")
        break
else:
    print(f"{num} is prime")
# Output: 17 is prime
```

### Example 5: `enumerate()` and `zip()`

`enumerate(iterable)` yields `(index, value)` pairs. `zip(*iterables)` aggregates elements from multiple iterables in parallel.

```python
fruits = ["apple", "banana", "cherry"]
for idx, fruit in enumerate(fruits):
    print(f"{idx}: {fruit}")
# Output: 0: apple
# Output: 1: banana
# Output: 2: cherry

names = ["Alice", "Bob", "Charlie"]
scores = [85, 92, 78]
for name, score in zip(names, scores):
    print(f"{name}: {score}")
# Output: Alice: 85
# Output: Bob: 92
# Output: Charlie: 78

# zip stops at the shortest iterable
a = [1, 2, 3]
b = ["x", "y"]
for pair in zip(a, b):
    print(pair, end=" ")
# Output: (1, 'x') (2, 'y')
```

### Example 6: `reversed()` and `sorted()`

`reversed(seq)` returns elements in reverse order. `sorted(iterable)` returns a new sorted list.

```python
items = [3, 1, 4, 1, 5, 9]
for val in reversed(items):
    print(val, end=" ")
# Output: 9 5 1 4 1 3

print()
for val in sorted(items):
    print(val, end=" ")
# Output: 1 1 3 4 5 9

print()
for val in sorted(items, reverse=True):
    print(val, end=" ")
# Output: 9 5 4 3 1 1
```

### Example 7: Nested loops

A loop inside another loop is a nested loop. The inner loop completes all its iterations for each iteration of the outer loop.

```python
for i in range(3):
    for j in range(2):
        print(f"({i},{j})", end=" ")
    print()
# Output: (0,0) (0,1)
# Output: (1,0) (1,1)
# Output: (2,0) (2,1)

# Multiplication table (5x5)
for row in range(1, 6):
    for col in range(1, 6):
        print(f"{row * col:4}", end="")
    print()
# Output:    1   2   3   4   5
# Output:    2   4   6   8  10
# Output:    3   6   9  12  15
# Output:    4   8  12  16  20
# Output:    5  10  15  20  25
```

## Common Mistakes

1. **Infinite loops** — Forgetting to update the condition variable in a `while` loop: `while x < 10:` with no `x += 1`.
2. **Modifying a list while iterating** — Adding or removing elements during a `for` loop can skip items or cause `IndexError`. Iterate over a copy instead.
3. **Off-by-one errors** — `for i in range(len(lst))` and using `lst[i]` when `i` goes out of bounds. Remember `range(n)` goes from `0` to `n-1`.
4. **Forgetting that `range()` is exclusive** — `range(5)` yields `0,1,2,3,4`, not `5`.
5. **Using `else` on a loop without understanding its semantics** — Beginners expect `else` to always run; it runs only when no `break` occurs.
6. **Misusing `continue`** — Placing `continue` before updating a `while` loop counter, causing an infinite loop.
7. **Unnecessary manual indexing** — Using `for i in range(len(lst))` when `for item in lst:` is simpler, or forgetting `enumerate()` when both index and value are needed.

## Interview Questions

### Beginner

1. **Q:** What is the difference between a `for` loop and a `while` loop?  
   **A:** A `for` loop iterates over a sequence or iterable; a `while` loop repeats as long as a condition is `True`.

2. **Q:** What does `range(5)` generate?  
   **A:** The sequence `0, 1, 2, 3, 4`.

3. **Q:** How do you exit a loop immediately?  
   **A:** Use the `break` statement.

4. **Q:** What does `continue` do?  
   **A:** It skips the rest of the current iteration and proceeds to the next one.

5. **Q:** What is the purpose of `pass`?  
   **A:** It is a no-op placeholder that does nothing; used where syntax requires a statement but no action is needed.

### Intermediate

1. **Q:** When does the `else` clause on a loop execute?  
   **A:** When the loop terminates normally (no `break` was encountered).

2. **Q:** How does `zip()` handle iterables of different lengths?  
   **A:** It stops when the shortest iterable is exhausted.

3. **Q:** What is the output of `list(enumerate(["a", "b"], start=1))`?  
   **A:** `[(1, 'a'), (2, 'b')]`.

4. **Q:** How do you iterate over a dictionary's key-value pairs?  
   **A:** `for key, value in my_dict.items():`.

5. **Q:** What is the difference between `sorted()` and `.sort()`?  
   **A:** `sorted()` returns a new list; `.sort()` sorts the list in place and returns `None`.

### Advanced

1. **Q:** How would you flatten a nested list using a loop?  
   **A:** Use nested `for` loops or a list comprehension: `[item for sublist in nested for item in sublist]`.

2. **Q:** Explain the danger of modifying a list while iterating over it. Provide a safer alternative.  
   **A:** Inserting/removing elements shifts indices and can skip items or cause `IndexError`. Safer: iterate over a copy (`for item in lst[:]`), or collect indices and modify after the loop.

3. **Q:** Implement a generator that yields Fibonacci numbers indefinitely using a `while` loop.  
   **A:**  
   ```python
   def fibonacci():
       a, b = 0, 1
       while True:
           yield a
           a, b = b, a + b
   ```

## Practice Problems

### Easy

1. **Sum 1 to N** — Accept an integer `n` and compute the sum from 1 to `n` using a loop.
2. **Factorial** — Compute the factorial of a non-negative integer using a loop.
3. **Count Vowels** — Accept a string and count the number of vowels (a, e, i, o, u).
4. **Print Pattern** — Print a right-angled triangle of asterisks with `n` rows.
5. **Even Numbers** — Print all even numbers from 1 to 20 using a `for` loop.

### Medium

1. **Palindrome Checker** — Accept a string and determine if it reads the same forwards and backwards using a loop (without `[::-1]`).
2. **Prime Numbers up to N** — Print all prime numbers less than or equal to a given `n`.
3. **List Intersection** — Given two lists, print their common elements using loops.
4. **Matrix Addition** — Given two 2D matrices (lists of lists), compute their sum.
5. **Word Frequency** — Accept a sentence and count the frequency of each word using a loop and a dictionary.

### Hard

1. **Flatten Nested List** — Write a function that recursively flattens a deeply nested list into a single-level list using loops.
2. **Custom Enumerate** — Implement your own version of `enumerate()` as a generator function.
3. **Spiral Matrix** — Given an `n x m` matrix, print its elements in spiral order (clockwise) using loops and direction control.

## Solutions

### Easy

```python
# 1. Sum 1 to N
n = int(input())
total = 0
for i in range(1, n + 1):
    total += i
print(total)
# Output (n=5): 15

# 2. Factorial
n = int(input())
fact = 1
for i in range(2, n + 1):
    fact *= i
print(fact)
# Output (n=5): 120

# 3. Count Vowels
text = input()
vowels = "aeiou"
count = 0
for ch in text.lower():
    if ch in vowels:
        count += 1
print(count)
# Output (text="Hello"): 2

# 4. Print Pattern
n = 5
for i in range(1, n + 1):
    print("*" * i)

# 5. Even Numbers
for i in range(2, 21, 2):
    print(i, end=" ")
# Output: 2 4 6 8 10 12 14 16 18 20
```

### Medium

```python
# 1. Palindrome Checker
s = input().lower()
is_pal = True
for i in range(len(s) // 2):
    if s[i] != s[-(i + 1)]:
        is_pal = False
        break
print(is_pal)

# 2. Prime Numbers up to N
n = int(input())
for num in range(2, n + 1):
    for divisor in range(2, int(num ** 0.5) + 1):
        if num % divisor == 0:
            break
    else:
        print(num, end=" ")

# 3. List Intersection
a = [1, 2, 2, 3, 4]
b = [2, 4, 6]
result = []
for x in a:
    if x in b and x not in result:
        result.append(x)
print(result)
# Output: [2, 4]

# 4. Matrix Addition
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
result = [[0, 0], [0, 0]]
for i in range(len(A)):
    for j in range(len(A[0])):
        result[i][j] = A[i][j] + B[i][j]
print(result)
# Output: [[6, 8], [10, 12]]

# 5. Word Frequency
sentence = "the cat and the dog"
words = sentence.split()
freq = {}
for word in words:
    freq[word] = freq.get(word, 0) + 1
print(freq)
# Output: {'the': 2, 'cat': 1, 'and': 1, 'dog': 1}
```

### Hard

```python
# 1. Flatten Nested List
def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

nested = [1, [2, [3, 4]], 5]
print(flatten(nested))
# Output: [1, 2, 3, 4, 5]

# 2. Custom Enumerate
def my_enumerate(iterable, start=0):
    i = start
    for item in iterable:
        yield i, item
        i += 1

for idx, val in my_enumerate(["a", "b", "c"], start=1):
    print(idx, val)
# Output: (1, 'a') (2, 'b') (3, 'c')

# 3. Spiral Matrix
def spiral_order(matrix):
    result = []
    while matrix:
        result += matrix.pop(0)
        if matrix and matrix[0]:
            for row in matrix:
                result.append(row.pop())
        if matrix:
            result += matrix.pop()[::-1]
        if matrix and matrix[0]:
            for row in matrix[::-1]:
                result.append(row.pop(0))
    return result

mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(spiral_order(mat))
# Output: [1, 2, 3, 6, 9, 8, 7, 4, 5]
```

## Related Concepts

- **Conditionals** — Loops often contain conditional statements to control behaviour.
- **Lists** — The most common iterable used with `for` loops.
- **Strings** — Iterable character by character.
- **Functions** — Encapsulate loop logic into reusable blocks.
- **Generators** — Functions that `yield` values, often powered by `while` loops.

## Next Concepts

- **List Comprehensions** — Concise looping to create lists.
- **Dictionaries** — Iterating over keys, values, and items.
- **Exception Handling** — Using `try`/`except` inside loops for robust iteration.

## Summary

Loops (`for` and `while`) enable repetitive execution. Use `break` to exit early, `continue` to skip iterations, and `pass` as a placeholder. The loop `else` clause is a distinctive Python feature that runs only when no `break` occurs. `enumerate()` and `zip()` simplify common iteration patterns. Nested loops handle multidimensional data. Mastering loops is essential for processing collections, building algorithms, and controlling program flow.

## Key Takeaways

- `for` loops iterate over iterables; `while` loops run until a condition is falsy.
- `range(start, stop, step)` is the standard way to generate numeric sequences.
- `break` exits the loop; `continue` jumps to the next iteration.
- The `else` clause executes only if the loop ended without a `break`.
- `enumerate()` yields `(index, value)` pairs; `zip()` aggregates parallel iterables.
- Avoid modifying a list while iterating over it; work on a copy instead.
- Nested loops are necessary for 2D data but can become performance bottlenecks.
