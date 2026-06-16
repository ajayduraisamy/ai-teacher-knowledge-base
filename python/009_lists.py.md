# Concept: Lists

## Concept ID

PYT-009

## Difficulty

BEGINNER

## Prerequisites

- PYT-002: Variables and Data Types
- PYT-005: Operators and Expressions
- PYT-008: Loops

## Domain

Python

## Module

Python Basics

## Learning Objectives

- Create lists using square brackets `[]` and the `list()` constructor
- Access elements with positive and negative indexing
- Extract sublists with slicing `[start:stop:step]`
- Modify lists using methods: `append`, `extend`, `insert`, `remove`, `pop`, `clear`, `index`, `count`, `sort`, `reverse`
- Understand shallow vs. deep copying and use `copy()` or `copy.deepcopy()`
- Work with nested lists for multidimensional data

## Definition

A list is an ordered, mutable (changeable) collection of elements enclosed in square brackets `[]`, with elements separated by commas. Lists can contain items of any data type — integers, strings, floats, other lists, or a mix. They are one of Python's most versatile and commonly used data structures, providing a rich set of methods for modification, searching, sorting, and copying.

## Intuition

Think of a list as a resizable, ordered container. Imagine a shopping list: you write items one after another, you can cross items off, add new ones in between, check how many there are, or take a picture of just the dairy section. A Python list works the same way — it holds items in order, can grow or shrink, and you can access, insert, remove, or rearrange elements at will.

## Why This Concept Matters

Lists are the backbone of data storage in Python. Whether you are storing sensor readings, filenames, user IDs, or word tokens, you will almost certainly use a list. Understanding indexing, slicing, and list methods is essential for manipulating data effectively. Knowing the difference between shallow and deep copy prevents subtle, hard-to-find bugs when working with nested data.

## Real World Examples

1. **To-do app**: Tasks are stored as a list of strings; users can add, remove, reorder, and mark tasks as done.
2. **Data pipeline**: A list of file paths is processed sequentially — read, transform, write.
3. **Leaderboard**: Player scores stored in a list, sorted descending to show the top players.
4. **Image processing**: A 2D list (list of lists) represents a grayscale image, where each inner list is a row of pixel intensities.
5. **ML batch**: A mini-batch of training samples is often a list of individual data points (each point may be a tuple or list).

## AI/ML Relevance

- **Dataset storage**: A list of feature vectors is a common in-memory representation before converting to NumPy arrays.
- **Hyperparameter search**: Lists hold the values to try — e.g., `learning_rates = [0.001, 0.01, 0.1]`.
- **Metrics tracking**: Training and validation loss per epoch are appended to a list for plotting.
- **Ensemble methods**: A list of trained model objects is used for voting or averaging predictions.
- **Data augmentation**: A list of transformed images is built by iterating over originals and applying random transformations.

## Code Examples

### Example 1: Creating lists and indexing

Python supports both positive indexing (from the start, 0-based) and negative indexing (from the end, -1 is the last element).

```python
fruits = ["apple", "banana", "cherry", "date", "elderberry"]

print(fruits[0])      # First element
# Output: apple
print(fruits[-1])     # Last element
# Output: elderberry
print(fruits[-2])     # Second last
# Output: date

# Using the list() constructor
squares = list(range(1, 6))
print(squares)
# Output: [1, 2, 3, 4, 5]

# Mixed types
mixed = [1, "hello", 3.14, True, None]
print(mixed)
# Output: [1, 'hello', 3.14, True, None]
```

### Example 2: Slicing `[start:stop:step]`

Slicing extracts a contiguous sublist. `start` is inclusive, `stop` is exclusive, `step` controls direction and stride.

```python
nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(nums[2:6])      # Indices 2 through 5
# Output: [2, 3, 4, 5]
print(nums[:4])       # First 4 elements
# Output: [0, 1, 2, 3]
print(nums[6:])       # From index 6 to the end
# Output: [6, 7, 8, 9]
print(nums[::2])      # Every second element
# Output: [0, 2, 4, 6, 8]
print(nums[::-1])     # Reversed list
# Output: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
print(nums[8:3:-1])   # From 8 down to 4 (exclusive)
# Output: [8, 7, 6, 5, 4]
```

### Example 3: List methods — adding elements

```python
# append — add single element to the end
cart = []
cart.append("milk")
cart.append("bread")
cart.append("eggs")
print(cart)
# Output: ['milk', 'bread', 'eggs']

# extend — add all elements from another iterable
cart.extend(["butter", "cheese"])
print(cart)
# Output: ['milk', 'bread', 'eggs', 'butter', 'cheese']

# insert — add at a specific index
cart.insert(1, "yogurt")
print(cart)
# Output: ['milk', 'yogurt', 'bread', 'eggs', 'butter', 'cheese']
```

### Example 4: List methods — removing elements

```python
items = ["a", "b", "c", "b", "d"]

# remove — removes the FIRST occurrence of a value
items.remove("b")
print(items)
# Output: ['a', 'c', 'b', 'd']

# pop — removes and returns element at index (default last)
last = items.pop()
print(last, items)
# Output: d ['a', 'c', 'b']

first = items.pop(0)
print(first, items)
# Output: a ['c', 'b']

# clear — removes all elements
items.clear()
print(items)
# Output: []
```

### Example 5: Searching, counting, sorting, reversing

```python
nums = [3, 1, 4, 1, 5, 9, 2, 6]

# index — returns index of first occurrence (raises ValueError if not found)
print(nums.index(4))
# Output: 2
print(nums.index(1))
# Output: 1

# count — how many times a value appears
print(nums.count(1))
# Output: 2

# sort — in-place ascending sort
nums.sort()
print(nums)
# Output: [1, 1, 2, 3, 4, 5, 6, 9]

# sort descending
nums.sort(reverse=True)
print(nums)
# Output: [9, 6, 5, 4, 3, 2, 1, 1]

# reverse — in-place reversal
nums.reverse()
print(nums)
# Output: [1, 1, 2, 3, 4, 5, 6, 9]
```

### Example 6: Shallow vs. deep copy

A shallow copy creates a new list but does **not** copy nested objects — they remain references to the same objects. A deep copy recursively copies everything.

```python
import copy

original = [[1, 2], [3, 4]]

# Shallow copy using .copy() or slicing
shallow = original.copy()
shallow[0][0] = 99
print(original)   # Original is also modified!
# Output: [[99, 2], [3, 4]]

# Deep copy using copy.deepcopy()
deep = copy.deepcopy(original)
deep[0][0] = 100
print(original)   # Original is unchanged
# Output: [[99, 2], [3, 4]]
print(deep)
# Output: [[100, 2], [3, 4]]
```

### Example 7: Nested lists (matrix)

A 2D list (list of lists) is commonly used to represent matrices or tabular data.

```python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]

print(matrix[1][2])   # Row 1, Column 2
# Output: 6

# Transpose using nested list comprehension
transpose = [[row[i] for row in matrix] for i in range(3)]
print(transpose)
# Output: [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
```

## Common Mistakes

1. **Using `=` to copy lists** — `list2 = list1` does not create a new list; both variables point to the same object. Use `list1.copy()` or `list1[:]`.
2. **Modifying a list while iterating** — Adding or removing elements during iteration causes skipped items or `IndexError`. Iterate over a slice copy `for item in lst[:]`.
3. **Confusing `append` and `extend`** — `append([4, 5])` adds one element (the list), while `extend([4, 5])` adds 4 and 5 as individual elements.
4. **Using `sort()` and forgetting it returns `None`** — `sorted_list = my_list.sort()` sets `sorted_list` to `None`. Use `sorted(my_list)` to get a new sorted list.
5. **Index out of range** — Accessing `lst[len(lst)]` or beyond raises `IndexError`. Valid indices are `0` to `len(lst)-1`.
6. **Assuming `remove()` removes all occurrences** — It only removes the first one. Use a loop or list comprehension to remove all.
7. **Forgetting that slicing creates a new list** — `sub = lst[1:3]` creates a copy of the slice. Modifying `sub` does not affect `lst` (for top-level elements).

## Interview Questions

### Beginner

1. **Q:** How do you create an empty list?  
   **A:** `[]` or `list()`.

2. **Q:** What does `lst[-1]` return?  
   **A:** The last element of the list.

3. **Q:** How do you add an element to the end of a list?  
   **A:** Use `lst.append(element)`.

4. **Q:** What is the difference between `lst.pop()` and `lst.pop(0)`?  
   **A:** `pop()` removes and returns the last element; `pop(0)` removes and returns the first element.

5. **Q:** How do you find the length of a list?  
   **A:** `len(lst)`.

### Intermediate

1. **Q:** Explain the difference between shallow copy and deep copy.  
   **A:** A shallow copy creates a new list containing references to the original elements; modifications to nested objects affect both. A deep copy recursively creates copies of all nested objects.

2. **Q:** What does `lst.sort()` return?  
   **A:** `None`. It sorts the list in place.

3. **Q:** How do you remove all occurrences of a value from a list?  
   **A:** Use a list comprehension: `lst = [x for x in lst if x != value]`.

4. **Q:** What is the time complexity of `lst.insert(0, x)`?  
   **A:** O(n) — inserting at the beginning requires shifting all elements right.

5. **Q:** How can you find the index of the last occurrence of a value?  
   **A:** `len(lst) - 1 - lst[::-1].index(value)`.

### Advanced

1. **Q:** Implement a function that flattens a nested list one level deep using a list comprehension.  
   **A:** `[item for sublist in nested for item in sublist]`.

2. **Q:** How does Python's `list.sort()` differ from `sorted()` in terms of stability?  
   **A:** Both are stable sorts (equal elements maintain their original relative order). `list.sort()` is in-place; `sorted()` returns a new list.

3. **Q:** Given `a = [[0]] * 3`, what is `a[0][0] = 1`? Explain.  
   **A:** `a` becomes `[[1], [1], [1]]`. `[[0]] * 3` creates a list of three references to the **same** inner list. Modifying one modifies all.

## Practice Problems

### Easy

1. **Sum of List** — Given a list of numbers, compute the sum using a loop.
2. **Largest Element** — Find the largest number in a list without using `max()`.
3. **Count Occurrences** — Count how many times a given value appears in a list.
4. **Reverse a List** — Create a reversed copy of a list without using `reverse()` or `[::-1]`.
5. **First and Last** — Given a list, return a new list containing only the first and last elements.

### Medium

1. **Remove Duplicates** — Remove duplicate elements from a list while preserving order.
2. **Rotate List** — Rotate a list by `k` positions to the right (e.g., `[1,2,3,4,5]` with `k=2` → `[4,5,1,2,3]`).
3. **Merge Sorted Lists** — Merge two sorted lists into one sorted list without using `sort()`.
4. **List Intersection** — Given two lists, return a list of elements that appear in both.
5. **Second Largest** — Find the second largest element in a list.

### Hard

1. **Subarray with Max Sum** — Given a list of integers (positive and negative), find the contiguous subarray with the maximum sum (Kadane's algorithm).
2. **Matrix Rotation** — Rotate an `n x n` matrix (list of lists) 90 degrees clockwise in place.
3. **Custom Flatten** — Write a function that flattens a deeply nested list of arbitrary depth, handling both lists and non-lists.

## Solutions

### Easy

```python
# 1. Sum of List
nums = [1, 2, 3, 4, 5]
total = 0
for n in nums:
    total += n
print(total)
# Output: 15

# 2. Largest Element
nums = [3, 7, 2, 9, 5]
largest = nums[0]
for n in nums:
    if n > largest:
        largest = n
print(largest)
# Output: 9

# 3. Count Occurrences
nums = [1, 2, 2, 3, 2, 4]
target = 2
count = 0
for n in nums:
    if n == target:
        count += 1
print(count)
# Output: 3

# 4. Reverse a List
nums = [1, 2, 3, 4]
reversed_list = []
for i in range(len(nums) - 1, -1, -1):
    reversed_list.append(nums[i])
print(reversed_list)
# Output: [4, 3, 2, 1]

# 5. First and Last
nums = [10, 20, 30, 40, 50]
print([nums[0], nums[-1]])
# Output: [10, 50]
```

### Medium

```python
# 1. Remove Duplicates
items = [1, 2, 2, 3, 3, 3, 4]
seen = []
for item in items:
    if item not in seen:
        seen.append(item)
print(seen)
# Output: [1, 2, 3, 4]

# 2. Rotate List
def rotate(lst, k):
    k = k % len(lst)
    return lst[-k:] + lst[:-k]

print(rotate([1, 2, 3, 4, 5], 2))
# Output: [4, 5, 1, 2, 3]

# 3. Merge Sorted Lists
a = [1, 3, 5, 7]
b = [2, 4, 6, 8]
merged = []
i = j = 0
while i < len(a) and j < len(b):
    if a[i] < b[j]:
        merged.append(a[i])
        i += 1
    else:
        merged.append(b[j])
        j += 1
merged.extend(a[i:])
merged.extend(b[j:])
print(merged)
# Output: [1, 2, 3, 4, 5, 6, 7, 8]

# 4. List Intersection
a = [1, 2, 2, 3, 4]
b = [2, 4, 6, 8]
result = [x for x in set(a) if x in b]
print(result)
# Output: [2, 4]

# 5. Second Largest
nums = [3, 7, 1, 7, 5, 9, 9]
unique = sorted(set(nums))
print(unique[-2] if len(unique) >= 2 else None)
# Output: 7
```

### Hard

```python
# 1. Subarray with Max Sum (Kadane's algorithm)
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
max_sum = cur_sum = nums[0]
for n in nums[1:]:
    cur_sum = max(n, cur_sum + n)
    max_sum = max(max_sum, cur_sum)
print(max_sum)
# Output: 6

# 2. Matrix Rotation 90° clockwise
def rotate(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for row in matrix:
        row.reverse()

mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
rotate(mat)
print(mat)
# Output: [[7, 4, 1], [8, 5, 2], [9, 6, 3]]

# 3. Custom Flatten (arbitrary depth)
def deep_flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(deep_flatten(item))
        else:
            result.append(item)
    return result

nested = [1, [2, [3, [4, 5]]], 6]
print(deep_flatten(nested))
# Output: [1, 2, 3, 4, 5, 6]
```

## Related Concepts

- **Strings** — Strings share some list-like behaviour (indexing, slicing, iteration) but are immutable.
- **Tuples** — Ordered, immutable collections; similar to lists but cannot be changed.
- **Loops** — `for` loops are the primary way to traverse lists.
- **List Comprehensions** — A concise syntax for creating new lists from existing ones.

## Next Concepts

- **List Comprehensions** — Transform and filter lists in one readable line.
- **Dictionaries** — Key-value mapping for structured data.
- **Sets** — Unordered collections of unique elements for membership testing.
- **Sorting (advanced)** — Custom sort keys with `key=` and `functools.cmp_to_key`.

## Summary

Lists are ordered, mutable, and versatile. They support positive and negative indexing, slicing, and a rich set of built-in methods. Understanding the difference between shallow and deep copy is essential for avoiding mutation bugs with nested lists. Lists are the foundation for storing and manipulating collections of data in Python and are used extensively in data processing, ML pipelines, and algorithmic problem-solving.

## Key Takeaways

- Create lists with `[]` or `list(iterable)`.
- Indexing: `lst[0]` (first), `lst[-1]` (last).
- Slicing: `lst[start:stop:step]` produces a new list.
- Key methods: `append`, `extend`, `insert`, `remove`, `pop`, `sort`, `reverse`, `index`, `count`, `clear`.
- `lst.sort()` sorts in place and returns `None`; `sorted(lst)` returns a new list.
- Shallow copy: `lst.copy()` or `lst[:]` — nested objects are shared.
- Deep copy: `copy.deepcopy(lst)` — fully independent copy.
- Nested lists can represent matrices and multidimensional data.
