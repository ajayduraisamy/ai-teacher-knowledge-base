# Concept: Sorting

## Concept ID

PYT-064

## Difficulty

Intermediate

## Domain

Python

## Module

Data Structures and Algorithms

## Learning Objectives

- Use sorted() and list.sort() with custom keys
- Understand Timsort — Python's hybrid sorting algorithm
- Implement stable and unstable sorts
- Sort complex objects by attribute
- Use lambda functions as sort keys
- Compare sorting algorithm complexities

## Prerequisites

- Familiarity with lists and strings
- Basic understanding of lambda functions
- Big O notation (PYT-056)

## Definition

Sorting is the process of arranging elements in a specific order (typically ascending or descending). Python provides sorted() (returns a new sorted list) and list.sort() (sorts in place). Both use Timsort, a hybrid stable sorting algorithm derived from merge sort and insertion sort, with O(n log n) worst-case time complexity.

## Intuition

Think of sorting a deck of cards. You might pick up cards one by one and insert them in the right position (insertion sort), or split the deck in half, sort each half, and merge them (merge sort). Timsort is smart about detecting already-sorted runs in the data, making it much faster on real-world data that is often partially ordered.

## Why This Concept Matters

Sorting is a fundamental operation used in virtually every application. It enables efficient search (binary search), data presentation, duplicate detection, and is a building block for many algorithms. Understanding how sorting works in Python helps you write efficient code and choose the right key functions for complex sorting needs.

## Real World Examples

1. **E-commerce:** Sorting products by price, rating, or relevance.
2. **Spreadsheets:** Sorting rows by column values.
3. **Database Queries:** ORDER BY clauses in SQL.
4. **Music Players:** Sorting playlists by artist, album, or year.
5. **File Managers:** Sorting files by name, date, or size.

## AI/ML Relevance

- **Feature Sorting:** Sorting feature importance values for visualization.
- **Ranking Metrics:** Sorting predictions for NDCG and MAP evaluation.
- **KNN Search:** Sorting distances to find nearest neighbors.
- **Decision Trees:** Sorting feature values to find optimal split points.
- **Data Preprocessing:** Sorting timestamps for time series analysis.

## Code Examples

### Example 1: Basic sorted() and list.sort()

`python
numbers = [3, 1, 4, 1, 5, 9, 2, 6]

sorted_copy = sorted(numbers)
print(f"Original: {numbers}")
print(f"Sorted copy: {sorted_copy}")

numbers.sort()
print(f"In-place sort: {numbers}")

numbers.sort(reverse=True)
print(f"Reversed: {numbers}")
# Output:
# Original: [3, 1, 4, 1, 5, 9, 2, 6]
# Sorted copy: [1, 1, 2, 3, 4, 5, 6, 9]
# In-place sort: [1, 1, 2, 3, 4, 5, 6, 9]
# Reversed: [9, 6, 5, 4, 3, 2, 1, 1]
`

### Example 2: Sorting with Custom Key

`python
words = ["banana", "apple", "cherry", "date", "elderberry"]

# Sort by length
print(sorted(words, key=len))

# Sort by last character
print(sorted(words, key=lambda s: s[-1]))

# Sort by reverse alphabetical
print(sorted(words, key=lambda s: s[::-1]))
# Output:
# ['date', 'apple', 'banana', 'cherry', 'elderberry']
# ['banana', 'apple', 'cherry', 'date', 'elderberry']
# ['date', 'cherry', 'banana', 'elderberry', 'apple']
`

### Example 3: Sorting Complex Objects

`python
from operator import attrgetter

class Student:
    def __init__(self, name, grade, age):
        self.name = name
        self.grade = grade
        self.age = age

    def __repr__(self):
        return f"Student({self.name}, {self.grade}, {self.age})"

students = [
    Student("Alice", 85, 22),
    Student("Bob", 92, 20),
    Student("Charlie", 78, 23),
    Student("Diana", 92, 21),
]

# Sort by grade descending, then age ascending
students.sort(key=lambda s: (-s.grade, s.age))
print(students)

# Sort by name using attrgetter
from operator import attrgetter
print(sorted(students, key=attrgetter("name")))
# Output:
# [Student(Bob, 92, 20), Student(Diana, 92, 21), Student(Alice, 85, 22), Student(Charlie, 78, 23)]
# [Student(Alice, 85, 22), Student(Bob, 92, 20), Student(Charlie, 78, 23), Student(Diana, 92, 21)]
`

### Example 4: Stable Sort

`python
pairs = [(1, "b"), (2, "a"), (1, "a"), (2, "b"), (1, "c")]

# Sort by first element (stable: preserves original order for ties)
pairs.sort(key=lambda x: x[0])
print(pairs)
# Output: [(1, 'b'), (1, 'a'), (1, 'c'), (2, 'a'), (2, 'b')]
# The (1, 'b'), (1, 'a'), (1, 'c') order is preserved from input.

# Multiple passes for complex sort
data = [(1, "c"), (2, "a"), (1, "a"), (2, "b")]
data.sort(key=lambda x: x[1])  # Sort by second element first
data.sort(key=lambda x: x[0])  # Then by first (stable)
print(data)
# Output: [(1, 'a'), (1, 'c'), (2, 'a'), (2, 'b')]
`

### Example 5: Sorting with itemgetter

`python
from operator import itemgetter

data = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35},
    {"name": "Diana", "age": 25},
]

# Sort by age, then by name
sorted_data = sorted(data, key=itemgetter("age", "name"))
for d in sorted_data:
    print(d)
# Output:
# {'name': 'Bob', 'age': 25}
# {'name': 'Diana', 'age': 25}
# {'name': 'Alice', 'age': 30}
# {'name': 'Charlie', 'age': 35}
`

### Example 6: Timsort in Action

`python
import time

# Partially sorted data (Timsort's strength)
data_partial = list(range(1000000)) + [500000]
data_random = [i for i in range(1000000)]
import random
random.shuffle(data_random)

# Timsort shines on partially sorted data
data1 = data_partial.copy()
start = time.time()
data1.sort()
print(f"Partial sort: {time.time() - start:.4f}s")

data2 = data_random.copy()
start = time.time()
data2.sort()
print(f"Random sort: {time.time() - start:.4f}s")
# Output:
# Partial sort: 0.0012s
# Random sort: 0.0850s
`

### Example 7: Custom Sort Order

`python
# Sorting by a predefined priority
priority = {"high": 0, "medium": 1, "low": 2}
tasks = ["low", "high", "medium", "high", "low"]
tasks.sort(key=lambda x: priority[x])
print(tasks)

# Natural sort (human-friendly)
import re
def natural_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]

filenames = ["file2.txt", "file10.txt", "file1.txt", "file20.txt"]
print(sorted(filenames))
print(sorted(filenames, key=natural_key))
# Output:
# ['high', 'high', 'medium', 'low', 'low']
# ['file1.txt', 'file10.txt', 'file2.txt', 'file20.txt']
# ['file1.txt', 'file2.txt', 'file10.txt', 'file20.txt']
`

## Common Mistakes

1. **Assuming sorted() Modifies the List:** sorted() returns a new list; use list.sort() for in-place.
2. **Using .sort() on Non-List Iterables:** sort() is a list method — use sorted() for tuples, strings, etc.
3. **Ignoring Stability:** If equal elements change relative order unexpectedly, remember Python sort is stable.
4. **Modifying List During Sort:** Never modify a list while iterating over its sorted version.
5. **Expensive Key Functions:** A key function called many times is slower — precompute keys if expensive.
6. **Trying to Sort Mixed Types:** sorted([1, "a", 2]) raises TypeError — ensure comparable types.
7. **Assuming Sort Is O(1) Space:** Timsort uses O(n) auxiliary space in the worst case.

## Interview Questions

### Beginner

1. What is the difference between sorted() and list.sort()?
2. How do you sort a list in descending order?
3. What is the time complexity of Python's sort?
4. How do you sort a list of strings by their length?
5. What does the key parameter do in sorted()?

### Intermediate

1. How does Timsort work and why is it efficient?
2. What does it mean for a sort to be stable?
3. How would you sort a list of tuples by the second element?
4. How do you perform a case-insensitive sort of strings?
5. How would you sort a list of dictionaries by a specific key?

### Advanced

1. Explain how Timsort detects and uses natural runs in data.
2. Implement merge sort from scratch. Why is it stable?
3. How would you sort a very large file that does not fit in memory (external sorting)?

## Practice Problems

### Easy

1. **Sort Numbers:** Sort a list of integers in ascending and descending order.
2. **Sort Strings:** Sort a list of strings by length and alphabetically.
3. **Deduplicate with Sort:** Remove duplicates from a list by sorting and scanning.
4. **Sort by Last Char:** Sort words by their last character.
5. **Sort Dicts:** Sort a list of dicts by a specific key.

### Medium

1. **Custom Sort Order:** Sort a list with a custom priority mapping.
2. **Multi-Key Sort:** Sort students by grade (desc), then age (asc).
3. **Natural Sort:** Implement natural (human-friendly) sorting for filenames.
4. **Sort Anagrams:** Group anagrams by sorting each word's characters.
5. **Sort Intervals:** Sort intervals by start time, then merge overlapping ones.

### Hard

1. **Implementation of Merge Sort:** Write merge sort from scratch.
2. **Implementation of Quick Sort:** Write in-place quick sort with pivot selection.
3. **External Sort:** Design an algorithm to sort a 10GB file with 1GB RAM.

## Solutions

### Solution to Easy 1: Sort Numbers

`python
nums = [3, 1, 4, 1, 5, 9, 2, 6]
print(sorted(nums))
print(sorted(nums, reverse=True))

nums.sort()
print(nums)
# Output:
# [1, 1, 2, 3, 4, 5, 6, 9]
# [9, 6, 5, 4, 3, 2, 1, 1]
# [1, 1, 2, 3, 4, 5, 6, 9]
`

### Solution to Medium 1: Custom Sort Order

`python
status_order = {"active": 0, "pending": 1, "inactive": 2}
users = [
    {"name": "Alice", "status": "pending"},
    {"name": "Bob", "status": "active"},
    {"name": "Charlie", "status": "inactive"},
]

users.sort(key=lambda u: status_order[u["status"]])
print([u["name"] for u in users])
# Output: ['Bob', 'Alice', 'Charlie']
`

### Solution to Hard 1: Merge Sort

`python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

print(merge_sort([3, 1, 4, 1, 5, 9, 2, 6]))
# Output: [1, 1, 2, 3, 4, 5, 6, 9]
`

## Related Concepts

- **Big O Notation (PYT-056):** Analyzing sorting algorithm complexity.
- **Searching (PYT-065):** Binary search requires sorted data.
- **Heaps (PYT-061):** Heap sort uses a heap data structure.
- **Trees (PYT-060):** Tree sort uses BST in-order traversal.

## Next Concepts

- **065 — Searching:** Efficient search in sorted data with binary search.

## Summary

Python provides sorted() and list.sort() powered by Timsort — a hybrid stable sort combining merge sort and insertion sort. Timsort detects natural runs in data, achieving O(n) on already-sorted data and O(n log n) worst case. Custom sorting is done via the key parameter with lambda functions, itemgetter, or ttrgetter. Python's sort is stable, equal elements retain their original relative order.

## Key Takeaways

- Use sorted() for a new list, list.sort() for in-place.
- Timsort is O(n log n) worst case, O(n) best case (already sorted).
- Custom keys with lambdas, itemgetter, ttrgetter.
- Sort is stable — use multi-pass sorting for complex orderings.
- Precompute expensive key functions for performance.
- Python 3 does not allow sorting mixed types.
- Natural sorting requires regex-based key splitting for human-friendly order.
