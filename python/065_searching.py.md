# Concept: Searching

## Concept ID

PYT-065

## Difficulty

Intermediate

## Domain

Python

## Module

Data Structures and Algorithms

## Learning Objectives

- Implement linear search on unsorted data
- Implement binary search on sorted data
- Use the isect module for efficient binary search in Python
- Understand interpolation search and its advantages
- Choose the appropriate search algorithm based on data characteristics
- Analyze search algorithm time complexities

## Prerequisites

- Basic understanding of lists and indexing
- Familiarity with Big O notation (PYT-056)
- Understanding of sorting (PYT-064)

## Definition

Searching is the process of finding the position of a target value within a collection of elements. Linear search checks every element sequentially (O(n)). Binary search repeatedly divides the search interval in half (O(log n)), requiring sorted data. Python's isect module provides binary search functions for sorted sequences.

## Intuition

Looking for a word in a dictionary: linear search would start at page 1 and flip through every page. Binary search would open the dictionary in the middle, determine if the word comes before or after, and discard half the dictionary. Each step halves the remaining search space, making binary search exponentially faster on large sorted datasets.

## Why This Concept Matters

Search is one of the most frequent operations in computing. Every database query, autocomplete suggestion, and file lookup involves search. Choosing the right search algorithm — linear for small/unsorted data, binary for sorted data — can mean the difference between milliseconds and seconds. Understanding search also teaches divide-and-conquer thinking fundamental to algorithm design.

## Real World Examples

1. **Database Indexes:** B-trees enable O(log n) search in billions of records.
2. **Autocomplete:** Binary search on sorted word lists.
3. **Version Control:** git bisect uses binary search to find the commit that introduced a bug.
4. **Phone Contacts:** Searching by name in a sorted contact list.
5. **Spell Check:** Binary search in a dictionary of valid words.

## AI/ML Relevance

- **Hyperparameter Search:** Grid search, random search, Bayesian optimization.
- **Nearest Neighbor Search:** KD-trees, ball trees for O(log n) nearest neighbor search.
- **Binary Search for Optimal Thresholds:** Finding optimal classification thresholds.
- **Search in Large Model Spaces:** Architecture search with pruning.
- **Caching Lookup:** O(1) hash-based search in ML pipeline caches.

## Code Examples

### Example 1: Linear Search

`python
def linear_search(arr, target):
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1

data = [4, 2, 7, 1, 9, 3, 6]
print(linear_search(data, 7))
print(linear_search(data, 5))
print(linear_search(data, 6))
# Output:
# 2
# -1
# 6
`

### Example 2: Binary Search (Manual)

`python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

sorted_data = [1, 3, 5, 7, 9, 11, 13, 15]
print(binary_search(sorted_data, 7))
print(binary_search(sorted_data, 2))
print(binary_search(sorted_data, 15))
# Output:
# 3
# -1
# 7
`

### Example 3: Using bisect Module

`python
import bisect

sorted_data = [1, 3, 5, 7, 9, 11, 13, 15]

# bisect_left returns insertion point to maintain sorted order
print(bisect.bisect_left(sorted_data, 7))
print(bisect.bisect_left(sorted_data, 6))

# bisect_right returns insertion point after any existing matches
print(bisect.bisect_right(sorted_data, 7))

# insort inserts while maintaining sorted order
bisect.insort(sorted_data, 8)
print(sorted_data)
# Output:
# 3
# 3
# 4
# [1, 3, 5, 7, 8, 9, 11, 13, 15]
`

### Example 4: Searching with Custom Key

`python
import bisect

class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade

    def __repr__(self):
        return f"{self.name}({self.grade})"

students = [
    Student("Alice", 85),
    Student("Bob", 72),
    Student("Charlie", 91),
    Student("Diana", 67),
]

# Search by grade using a parallel list
grades = [s.grade for s in students]
grades.sort()

# Find where a grade would be inserted
pos = bisect.bisect_left(grades, 72)
print(f"Grade 72 would be at position: {pos}")

# Find if a student with grade 85 exists
pos = bisect.bisect_left(grades, 85)
if pos < len(grades) and grades[pos] == 85:
    print(f"Found grade 85 at position {pos}")
# Output:
# Grade 72 would be at position: 1
# Found grade 85 at position 2
`

### Example 5: Interpolation Search

`python
def interpolation_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right and arr[left] <= target <= arr[right]:
        if left == right:
            return left if arr[left] == target else -1

        # Estimate position (works best for uniformly distributed data)
        pos = left + ((target - arr[left]) * (right - left)) // (arr[right] - arr[left])

        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            left = pos + 1
        else:
            right = pos - 1
    return -1

uniform = list(range(0, 1000, 2))  # Even numbers, uniform distribution
print(interpolation_search(uniform, 42))
print(interpolation_search(uniform, 43))
print(interpolation_search(uniform, 998))
# Output:
# 21
# -1
# 499
`

### Example 6: Searching in Rotated Sorted Array

`python
def search_rotated(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid

        if nums[left] <= nums[mid]:  # Left half is sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # Right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1

rotated = [4, 5, 6, 7, 0, 1, 2]
print(search_rotated(rotated, 0))
print(search_rotated(rotated, 3))
print(search_rotated(rotated, 4))
# Output:
# 4
# -1
# 0
`

### Example 7: Performance Comparison

`python
import time
import bisect

size = 10_000_000
sorted_data = list(range(size))
target = size - 1  # Worst case for linear

start = time.time()
result = -1
for i, v in enumerate(sorted_data):
    if v == target:
        result = i
        break
print(f"Linear search: {time.time() - start:.4f}s (found at {result})")

start = time.time()
pos = bisect.bisect_left(sorted_data, target)
found = pos < len(sorted_data) and sorted_data[pos] == target
print(f"Binary search: {time.time() - start:.4f}s (found={found})")
# Output:
# Linear search: 0.3251s (found at 9999999)
# Binary search: 0.0002s (found=True)
`

## Common Mistakes

1. **Binary Search on Unsorted Data:** Binary search only works correctly on sorted arrays.
2. **Off-by-One Errors:** Incorrect midpoint calculation or boundary conditions.
3. **Integer Overflow in Midpoint:** (left + right) // 2 can overflow in some languages — Python is safe.
4. **Assuming Uniform Distribution for Interpolation Search:** Interpolation search degrades to O(n) on non-uniform data.
5. **Not Handling Duplicates:** Standard binary search finds any occurrence; use isect_left/isect_right for boundaries.
6. **Modifying Data During Search:** Inserting/deleting elements while searching causes incorrect results.
7. **Using Linear Search on Sorted Data:** Always use binary search on sorted data for efficiency.

## Interview Questions

### Beginner

1. What is linear search and what is its time complexity?
2. What is binary search and what are its requirements?
3. How does binary search work step by step?
4. What is the time complexity of binary search?
5. What does the isect module do?

### Intermediate

1. How do you find the first and last occurrence of a value in a sorted array?
2. What is the difference between isect_left and isect_right?
3. How would you search in a rotated sorted array?
4. What is interpolation search and when is it better than binary search?
5. How do you perform binary search on a list of objects by a specific attribute?

### Advanced

1. Implement an exponential search algorithm and explain when it is useful.
2. How would you implement search in a ternary search tree?
3. Design a search algorithm for a list that is too large to fit in memory.

## Practice Problems

### Easy

1. **Linear Search:** Find all occurrences of a target in a list.
2. **Binary Search:** Implement binary search recursively.
3. **Find Insert Position:** Find the index where a target should be inserted.
4. **Count Occurrences:** Count how many times a target appears in a sorted list.
5. **Search in Sorted Matrix:** Search for a value in a row and column sorted matrix.

### Medium

1. **First and Last Position:** Find the first and last position of a target in a sorted array.
2. **Search in Rotated Array:** Search for a target in a rotated sorted array.
3. **Peak Element:** Find a peak element in an array (greater than neighbors).
4. **Search in Nearly Sorted Array:** Search in an array where each element can be off by at most one position.
5. **Find Minimum in Rotated Array:** Find the minimum element in a rotated sorted array.

### Hard

1. **Median of Two Sorted Arrays:** Find the median of two sorted arrays in O(log(min(n,m))).
2. **Search in Sorted Unknown-Sized Array:** Search in an array with unknown size (no len()).
3. **Kth Smallest in Two Sorted Arrays:** Find the Kth smallest element across two sorted arrays.

## Solutions

### Solution to Easy 1: Linear Search All Occurrences

`python
def linear_search_all(arr, target):
    return [i for i, v in enumerate(arr) if v == target]

print(linear_search_all([1, 3, 2, 3, 4, 3, 5], 3))
# Output: [1, 3, 5]
`

### Solution to Medium 1: First and Last Position

`python
import bisect

def search_range(nums, target):
    left = bisect.bisect_left(nums, target)
    if left == len(nums) or nums[left] != target:
        return [-1, -1]
    right = bisect.bisect_right(nums, target) - 1
    return [left, right]

print(search_range([5, 7, 7, 8, 8, 10], 8))
print(search_range([5, 7, 7, 8, 8, 10], 6))
# Output:
# [3, 4]
# [-1, -1]
`

### Solution to Hard 1: Median of Two Sorted Arrays

`python
def find_median_sorted_arrays(nums1, nums2):
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    low, high = 0, m

    while low <= high:
        partition1 = (low + high) // 2
        partition2 = (m + n + 1) // 2 - partition1

        max_left1 = float("-inf") if partition1 == 0 else nums1[partition1 - 1]
        min_right1 = float("inf") if partition1 == m else nums1[partition1]
        max_left2 = float("-inf") if partition2 == 0 else nums2[partition2 - 1]
        min_right2 = float("inf") if partition2 == n else nums2[partition2]

        if max_left1 <= min_right2 and max_left2 <= min_right1:
            if (m + n) % 2 == 0:
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2.0
            else:
                return max(max_left1, max_left2)
        elif max_left1 > min_right2:
            high = partition1 - 1
        else:
            low = partition1 + 1

    raise ValueError("Input arrays are not sorted")

print(find_median_sorted_arrays([1, 3], [2]))
print(find_median_sorted_arrays([1, 2], [3, 4]))
# Output:
# 2.0
# 2.5
`

## Related Concepts

- **Sorting (PYT-064):** Binary search requires sorted data.
- **Big O Notation (PYT-056):** Analyzing search complexity.
- **Trees (PYT-060):** BST search is binary search on a tree structure.
- **Graph Algorithms (PYT-063):** BFS/DFS are graph search algorithms.

## Next Concepts

- Continuing practice with data structure and algorithm fundamentals.

## Summary

Searching finds the position of a target in a collection. Linear search (O(n)) works on any data but is slow for large collections. Binary search (O(log n)) requires sorted data and is exponentially faster. Python's isect module provides isect_left, isect_right, and insort for efficient binary search and insertion. Interpolation search improves on binary search for uniformly distributed data.

## Key Takeaways

- Linear search: O(n), works on any data, simple.
- Binary search: O(log n), requires sorted data, use isect module.
- isect_left finds the first occurrence; isect_right finds the last+1.
- Binary search can adapt to rotated arrays and unknown-size arrays.
- Interpolation search is O(log log n) on uniformly distributed data but O(n) worst case.
- Always sort before binary search — the sorting cost may be amortized over many searches.
- Searching is the foundation for many other algorithms (dictionaries, databases, AI).
