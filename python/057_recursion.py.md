# Concept: Recursion

## Concept ID

PYT-057

## Difficulty

Intermediate

## Domain

Python

## Module

Data Structures and Algorithms

## Learning Objectives

- Define recursion and identify its components: base case and recursive case
- Write recursive functions for problems with self-similar structure
- Understand the call stack and its role in recursion
- Recognize and avoid stack overflow errors
- Use memoization to optimize recursive functions
- Compare recursive vs iterative approaches
- Understand why Python does not optimize tail recursion

## Prerequisites

- Understanding of function calls and the call stack
- Familiarity with list and dict operations
- Basic understanding of Big O notation (PYT-056)

## Definition

Recursion is a programming technique where a function calls itself to solve a problem by breaking it down into smaller subproblems of the same type. A recursive function has two parts: a base case that terminates the recursion, and a recursive case that reduces the problem toward the base case.

## Intuition

Imagine standing between two mirrors facing each other. Each mirror shows a smaller version of the same scene inside itself. That is recursion — a function calling a smaller version of itself. Each call is a "mirror" that reflects the same problem but with smaller input, until it reaches the smallest possible version (the base case), which is solved directly.

## Why This Concept Matters

Recursion is the natural way to describe many problems: tree traversal, divide-and-conquer algorithms, backtracking, and mathematical definitions (Fibonacci, factorial). It often produces cleaner, more readable code than iterative equivalents. More importantly, many algorithms and data structures are inherently recursive — understanding recursion is essential for working with trees, graphs, and dynamic programming.

## Real World Examples

1. **File System Navigation:** Recursively listing all files in a directory tree.
2. **Web Crawling:** Crawling links on a page and following them recursively.
3. **Parsing:** Recursive descent parsers for JSON, XML, and programming languages.
4. **Fractal Generation:** Drawing fractals like the Mandelbrot set or Sierpinski triangle.
5. **Game Algorithms:** Minimax algorithm for chess/game AI with recursive evaluation.

## AI/ML Relevance

- **Recursive Neural Networks (RvNN):** Tree-structured neural networks for NLP parsing.
- **Decision Trees:** Recursive partitioning of feature space during training.
- **Reinforcement Learning:** Recursive value function estimation in dynamic programming.
- **Backtracking in Optimization:** Recursive search for optimal solutions in combinatorial problems.
- **Recursive Feature Elimination:** Recursively removing features to find the best subset.

## Code Examples

### Example 1: Factorial

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(0))
print(factorial(5))
print(factorial(10))
# Output:
# 1
# 120
# 3628800
```

### Example 2: Fibonacci Sequence

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

for i in range(10):
    print(fibonacci(i), end=" ")
# Output: 0 1 1 2 3 5 8 13 21 34
```

### Example 3: Fibonacci with Memoization

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_memo(n):
    if n <= 1:
        return n
    return fib_memo(n - 1) + fib_memo(n - 2)

import time
start = time.time()
print(f"fib(35) = {fib_memo(35)}")
print(f"Time: {time.time() - start:.4f}s")
# Output:
# fib(35) = 9227465
# Time: 0.0001s
```

### Example 4: Recursive Directory Traversal

```python
import os

def list_files(path, indent=0):
    for entry in os.scandir(path):
        prefix = "  " * indent
        if entry.is_dir():
            print(f"{prefix}[{entry.name}]")
            list_files(entry.path, indent + 1)
        else:
            print(f"{prefix}{entry.name}")

# list_files(".")  # Uncomment to see output
print("Recursive directory listing ready")
# Output: Recursive directory listing ready
```

### Example 5: Tower of Hanoi

```python
def hanoi(n, source, target, auxiliary):
    if n == 1:
        print(f"Move disk 1 from {source} to {target}")
        return
    hanoi(n - 1, source, auxiliary, target)
    print(f"Move disk {n} from {source} to {target}")
    hanoi(n - 1, auxiliary, target, source)

hanoi(3, "A", "C", "B")
# Output:
# Move disk 1 from A to C
# Move disk 2 from A to B
# Move disk 1 from C to B
# Move disk 3 from A to C
# Move disk 1 from B to A
# Move disk 2 from B to C
# Move disk 1 from A to C
```

### Example 6: Recursive vs Iterative (Reversing a String)

```python
def reverse_recursive(s):
    if len(s) <= 1:
        return s
    return reverse_recursive(s[1:]) + s[0]

def reverse_iterative(s):
    return s[::-1]

print(reverse_recursive("python"))
print(reverse_iterative("python"))
# Output:
# nohtyp
# nohtyp
```

### Example 7: Stack Overflow Demonstration

```python
import sys

def infinite_recursion(n):
    return infinite_recursion(n + 1)

try:
    infinite_recursion(0)
except RecursionError as e:
    print(f"Recursion limit reached: {e}")
    print(f"System recursion limit: {sys.getrecursionlimit()}")
# Output:
# Recursion limit reached: maximum recursion depth exceeded
# System recursion limit: 1000
```

## Common Mistakes

1. **Missing Base Case:** Without a base case, recursion never terminates, causing stack overflow.
2. **Base Case Never Reached:** The recursive case does not reduce the problem toward the base case.
3. **Too Much Recursion Depth:** Python's default recursion limit is 1000 — deep recursion crashes.
4. **Not Using Memoization:** Naive recursive Fibonacci has O(2^n) complexity — memoization makes it O(n).
5. **Assuming Tail Call Optimization:** Python does not optimize tail recursion — deep tail-recursive functions still overflow.
6. **Mutating Shared State:** Passing mutable objects as defaults or globals across recursive calls causes bugs.
7. **Overcomplicating Iterative Problems:** Recursion is elegant for trees but wasteful for simple loops.

## Interview Questions

### Beginner

1. What is a base case in recursion?
2. Write a recursive function to compute the sum of digits of a number.
3. What is the difference between recursion and iteration?
4. How does the call stack relate to recursion?
5. What happens if you call a recursive function without a base case?

### Intermediate

1. Explain why naive recursive Fibonacci is O(2^n). How does memoization improve it?
2. What is tail recursion? Why doesn't Python optimize it?
3. Write a recursive function to check if a string is a palindrome.
4. How would you convert a recursive function to an iterative one?
5. What is the maximum recursion depth in Python and how can you change it?

### Advanced

1. Implement a recursive descent parser for simple arithmetic expressions.
2. Explain how mutual recursion works and give an example.
3. Design a recursive algorithm for generating all permutations of a list. Analyze its complexity.

## Practice Problems

### Easy

1. **Sum to N:** Write a recursive function that sums numbers from 1 to n.
2. **Power:** Write a recursive function to compute x^n.
3. **Count Digits:** Count the number of digits in an integer recursively.
4. **GCD:** Implement Euclid's algorithm recursively for GCD.
5. **List Sum:** Sum all elements in a list recursively.

### Medium

1. **Palindrome:** Check if a string is a palindrome using recursion.
2. **Flatten List:** Recursively flatten a nested list `[1, [2, [3, 4]], 5]`.
3. **Binary Search:** Implement binary search recursively.
4. **Permutations:** Generate all permutations of a string recursively.
5. **Subset Sum:** Determine if a subset of a list sums to a target value.

### Hard

1. **N-Queens:** Solve the N-Queens problem using recursive backtracking.
2. **Tower of Hanoi Visualizer:** Animate the Tower of Hanoi solution step by step.
3. **Recursive Descent Parser:** Parse and evaluate arithmetic expressions with +, -, *, / and parentheses.

## Solutions

### Solution to Easy 1: Sum to N

```python
def sum_to_n(n):
    if n <= 0:
        return 0
    return n + sum_to_n(n - 1)

print(sum_to_n(5))
print(sum_to_n(100))
# Output:
# 15
# 5050
```

### Solution to Medium 1: Palindrome

```python
def is_palindrome(s):
    if len(s) <= 1:
        return True
    if s[0] != s[-1]:
        return False
    return is_palindrome(s[1:-1])

print(is_palindrome("racecar"))
print(is_palindrome("hello"))
print(is_palindrome("a"))
print(is_palindrome(""))
# Output:
# True
# False
# True
# True
```

### Solution to Hard 1: N-Queens

```python
def solve_n_queens(n):
    def is_safe(board, row, col):
        for i in range(row):
            if board[i] == col or abs(board[i] - col) == row - i:
                return False
        return True

    def backtrack(row):
        if row == n:
            solutions.append(board[:])
            return
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                backtrack(row + 1)

    solutions = []
    board = [-1] * n
    backtrack(0)
    return solutions

solutions = solve_n_queens(4)
print(f"Number of solutions for 4-Queens: {len(solutions)}")
for sol in solutions:
    print(sol)
# Output:
# Number of solutions for 4-Queens: 2
# [1, 3, 0, 2]
# [2, 0, 3, 1]
```

## Related Concepts

- **Big O Notation (PYT-056):** Analyzing recursive algorithm complexity.
- **Stacks (PYT-059):** The call stack is the underlying mechanism for recursion.
- **Trees (PYT-060):** Tree traversal is naturally recursive.
- **Heaps (PYT-061):** Heap operations can be implemented recursively.

## Next Concepts

- **058 — Linked Lists:** Linear data structure with recursive operations.
- **059 — Stacks and Queues:** Using stacks to model recursion iteratively.

## Summary

Recursion is a technique where a function calls itself to solve subproblems. Every recursive function needs a base case (to stop) and a recursive case (to progress). Python's call stack tracks pending calls, and exceeding the recursion limit (default 1000) causes `RecursionError`. Memoization with `functools.lru_cache` optimizes recursive functions that recompute the same values. Python does not optimize tail recursion.

## Key Takeaways

- Always define a base case that terminates recursion.
- Ensure each recursive call moves toward the base case.
- Use `@lru_cache` to memoize expensive recursive computations.
- Be aware of Python's recursion depth limit (~1000).
- Tail recursion is not optimized in Python — prefer iteration for very deep recursion.
- Recursion is ideal for tree-shaped problems (directory trees, expression trees, game trees).
- Any recursive function can be converted to an iterative one using an explicit stack.
