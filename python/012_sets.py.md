# Concept: Sets

## Concept ID

PYT-012

## Difficulty

BEGINNER

## Domain

Python

## Module

Python Basics

## Learning Objectives

- Understand what sets are and the concept of unordered uniqueness
- Create sets using literal syntax and the `set()` constructor
- Perform set operations: union, intersection, difference, symmetric difference
- Use set methods: add, remove, discard, pop, clear, update, intersection_update
- Work with frozenset (immutable sets)
- Write set comprehensions
- Apply sets to real-world problems like deduplication and membership testing

## Prerequisites

- Basic Python syntax (variables, data types)
- Familiarity with lists and tuples
- Understanding of mutable vs immutable objects
- Knowledge of dictionary basics (sets share `{}` notation but differ)

## Definition

A set is a built-in Python data structure that stores an unordered collection of unique, hashable elements. Sets are mutable (you can add or remove elements), but the elements themselves must be immutable. Sets implement mathematical set operations like union, intersection, difference, and symmetric difference. Membership testing in a set is O(1) average time complexity, making sets extremely efficient for uniqueness checks and deduplication.

## Intuition

Think of a set as a bag of unique items where order doesn't matter. If you toss identical items into the bag, duplicates vanish — only one of each remains. This is exactly how a physical set of poker chips works: you have one chip of each color, and you can quickly check whether a particular color is in your collection. Unlike a list where you count positions, a set only cares about membership: "Is this element present or not?" The Venn diagrams you learned in school — overlapping circles showing common elements — are the mathematical foundation of sets.

## Why This Concept Matters

Sets solve two fundamental problems elegantly: removing duplicates and testing membership. They are indispensable for data cleaning (deduplication), finding common elements between datasets, tracking visited states (e.g., in graph traversal algorithms like BFS/DFS), and efficiently checking whether an item exists in a collection. Sets also enable powerful mathematical operations that would require verbose loops with lists. In data science pipelines, sets are used to extract unique labels, find intersecting features between datasets, and compute similarity metrics.

## Real World Examples

- **Removing duplicate emails**: Deduplicating a mailing list by converting to a set
- **Finding mutual friends**: Intersection of two users' friend sets on social media
- **Tracking visited URLs**: Web crawlers use a set of visited URLs to avoid revisiting pages
- **Unique tags/categories**: Extracting all unique tags from a collection of blog posts
- **Shopping cart items**: Ensuring a product isn't added twice
- **Spam detection**: Maintaining a set of known spam email addresses or domains

## AI/ML Relevance

- **Deduplication**: Removing duplicate rows from training datasets, or duplicate samples in data augmentation
- **Unique label extraction**: Getting all unique class labels from a dataset using `set(labels)`
- **Jaccard similarity**: Computing set-based similarity between documents or users: `|A ∩ B| / |A ∪ B|`
- **Feature selection**: Finding the intersection of important features across multiple models
- **Vocabulary building**: Collecting unique tokens/words from a text corpus for NLP tokenizers
- **Set-based metrics**: Precision and recall in information retrieval can be expressed as set operations on relevant vs retrieved documents
- **Anomaly detection**: Tracking which data points have been seen during training (set membership)

## Code Examples

### Example 1: Creating Sets

```python
# Literal syntax
fruits = {"apple", "banana", "cherry", "apple"}
print(fruits)
# Output: {'banana', 'apple', 'cherry'}

# Using set() constructor from any iterable
unique_chars = set("mississippi")
print(unique_chars)
# Output: {'m', 'i', 'p', 's'}

# From a list (deduplication)
numbers = set([1, 2, 2, 3, 3, 3, 4])
print(numbers)
# Output: {1, 2, 3, 4}

# Empty set (must use set(), not {})
empty_set = set()
empty_dict = {}
print(type(empty_set), type(empty_dict))
# Output: <class 'set'> <class 'dict'>
```

### Example 2: Adding and Removing Elements

```python
colors = {"red", "blue"}

# add
colors.add("green")
print(colors)
# Output: {'red', 'green', 'blue'}

# remove — raises KeyError if missing
colors.remove("red")
print(colors)
# Output: {'green', 'blue'}

# discard — does NOT raise error if missing
colors.discard("yellow")
print(colors)
# Output: {'green', 'blue'}

# pop — remove and return an arbitrary element
popped = colors.pop()
print(f"Removed: {popped}", colors)
# Output: Removed: green {'blue'}

# clear — empty the set
colors.add("red")
colors.add("yellow")
colors.clear()
print(colors)
# Output: set()
```

### Example 3: Membership Testing

```python
valid_statuses = {"active", "pending", "inactive"}

print("active" in valid_statuses)
# Output: True
print("deleted" in valid_statuses)
# Output: False
print("suspended" not in valid_statuses)
# Output: True

# Membership checking is MUCH faster on sets vs lists
import time
big_list = list(range(100000))
big_set = set(big_list)
# The set lookup is O(1), list is O(n)
```

### Example 4: Set Operations

```python
a = {1, 2, 3, 4, 5}
b = {4, 5, 6, 7, 8}

# Union — elements in either set
print(a | b)
# Output: {1, 2, 3, 4, 5, 6, 7, 8}
print(a.union(b))
# Output: {1, 2, 3, 4, 5, 6, 7, 8}

# Intersection — elements in both sets
print(a & b)
# Output: {4, 5}
print(a.intersection(b))
# Output: {4, 5}

# Difference — elements in a but not in b
print(a - b)
# Output: {1, 2, 3}
print(a.difference(b))
# Output: {1, 2, 3}

# Symmetric difference — elements in either but not both
print(a ^ b)
# Output: {1, 2, 3, 6, 7, 8}
print(a.symmetric_difference(b))
# Output: {1, 2, 3, 6, 7, 8}
```

### Example 5: Subset, Superset, and Disjoint Checks

```python
x = {1, 2, 3}
y = {1, 2, 3, 4, 5}
z = {6, 7}

# issubset
print(x <= y, x.issubset(y))
# Output: True True

# issuperset
print(y >= x, y.issuperset(x))
# Output: True True

# Proper subset (strict)
print(x < y)
# Output: True
print(x < x)
# Output: False

# isdisjoint — no common elements
print(x.isdisjoint(z))
# Output: True
print(x.isdisjoint(y))
# Output: False
```

### Example 6: Set Update Methods (In-Place)

```python
a = {1, 2, 3}
b = {3, 4, 5}

# update — adds all elements from another iterable (union in-place)
a.update(b)
print(a)
# Output: {1, 2, 3, 4, 5}

# intersection_update
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}
a.intersection_update(b)
print(a)
# Output: {3, 4}

# difference_update
a = {1, 2, 3, 4, 5}
b = {3, 4}
a.difference_update(b)
print(a)
# Output: {1, 2, 5}

# symmetric_difference_update
a = {1, 2, 3}
b = {3, 4, 5}
a.symmetric_difference_update(b)
print(a)
# Output: {1, 2, 4, 5}
```

### Example 7: Set Comprehensions

```python
# Square roots of numbers 0-9
squares = {x**2 for x in range(10)}
print(squares)
# Output: {0, 1, 64, 4, 36, 9, 16, 49, 81, 25}

# Unique vowel lengths
words = ["hello", "world", "python", "sets"]
vowel_counts = {sum(1 for c in w if c in "aeiou") for w in words}
print(vowel_counts)
# Output: {1, 2}

# Filtering with conditions
evens_squared = {x**2 for x in range(20) if x % 2 == 0}
print(evens_squared)
# Output: {0, 64, 4, 36, 100, 16, 256, 196, 144, 324, 64}

# Unique characters from multiple strings
strings = ["abc", "bcd", "cde"]
unique_chars = {c for s in strings for c in s}
print(sorted(unique_chars))
# Output: ['a', 'b', 'c', 'd', 'e']
```

### Example 8: Frozenset

```python
# frozenset is immutable and hashable — can be used as dict key
fs = frozenset([1, 2, 3])
print(fs)
# Output: frozenset({1, 2, 3})

# Can be used as a dictionary key
lookup = {frozenset(["alice", "bob"]): "friends"}
lookup[frozenset(["charlie", "diana"])] = "coworkers"
print(lookup)
# Output: {frozenset({'bob', 'alice'}): 'friends', frozenset({'diana', 'charlie'}): 'coworkers'}

# frozenset supports all read-only set operations
a = frozenset([1, 2, 3, 4])
b = frozenset([3, 4, 5, 6])
print(a & b)
# Output: frozenset({3, 4})

# Cannot modify a frozenset
try:
    a.add(5)
except AttributeError as e:
    print(f"Cannot modify: {e}")
# Output: Cannot modify: 'frozenset' object has no attribute 'add'
```

## Common Mistakes

1. **Using `{}` for an empty set**: `{}` creates an empty dictionary, not an empty set. Use `set()` for empty sets.
2. **Assuming sets maintain insertion order**: Sets are unordered. Even though Python 3.7+ sometimes preserves insertion order as an implementation detail, you must never rely on it — order is not guaranteed across implementations.
3. **Adding mutable elements**: Lists, dictionaries, or other sets cannot be added to a set — they raise `TypeError: unhashable type: 'list'`
4. **Confusing `remove()` and `discard()`**: `remove()` raises `KeyError` if the element is missing; `discard()` silently does nothing. Using `remove()` without checking membership first can crash your program.
5. **Using `pop()` expecting a specific element**: `set.pop()` removes and returns an arbitrary element (the "first" only by implementation detail). Never rely on which element is popped.
6. **Forgetting that sets are mutable**: If you need a hashable set (e.g., for use as a dict key), use `frozenset` instead.
7. **Assuming `set()` deduplicates in a stable order**: The order of elements after deduplication is unpredictable.
8. **Using `|`, `&`, `-`, `^` with non-set iterables**: These operators only work between two sets. For mixed types, use methods like `set.union(other_iterable)` which accepts any iterable.

## Interview Questions

### Beginner

1. **Q**: What is the difference between a list and a set?
   **A**: Lists are ordered, allow duplicates, and support indexing. Sets are unordered, contain only unique elements, and do not support indexing. Sets provide O(1) membership testing; lists are O(n).

2. **Q**: How do you remove duplicates from a list while preserving order?
   **A**: Use a loop with a seen set: `seen = set(); [x for x in list if not (x in seen or seen.add(x))]`. In Python 3.7+, you can also use `dict.fromkeys(list)` which preserves insertion order.

3. **Q**: How do you create an empty set?
   **A**: Using `set()`. Using `{}` creates an empty dictionary, not a set.

4. **Q**: What is the time complexity of checking if an element is in a set?
   **A**: O(1) average case (constant time) due to hash table implementation.

5. **Q**: Can a set contain another set? Why or why not?
   **A**: No, because sets are mutable and therefore not hashable. A `frozenset` can contain other frozensets since frozensets are immutable and hashable.

### Intermediate

1. **Q**: Explain the difference between `remove()`, `discard()`, and `pop()` on a set.
   **A**: `remove(x)` removes x from the set, raising KeyError if x is not present. `discard(x)` also removes x but does nothing if x is missing. `pop()` removes and returns an arbitrary element, raising KeyError on an empty set.

2. **Q**: How would you find common elements between two lists efficiently?
   **A**: Convert both lists to sets and use intersection: `list(set(list1) & set(list2))`. This is O(n+m) average versus O(n*m) for nested loops.

3. **Q**: What is a frozenset and when would you use it?
   **A**: `frozenset` is an immutable, hashable version of a set. Use it when you need a set as a dictionary key, as an element of another set, or when you want to guarantee the collection cannot be modified.

4. **Q**: How do you compute the Jaccard similarity between two sets?
   **A**: Jaccard similarity = `len(set_a & set_b) / len(set_a | set_b)`. It measures the proportion of shared elements to total unique elements, ranging from 0 (no overlap) to 1 (identical sets).

5. **Q**: Can you use the `+` operator on sets? Why or why not?
   **A**: No, `+` is not defined for sets. Sets use `|` for union, `&` for intersection, `-` for difference, and `^` for symmetric difference. Using `+` raises a TypeError.

### Advanced

1. **Q**: How does Python implement sets internally, and what determines iteration order?
   **A**: Python sets are implemented as hash tables (similar to dictionaries but without values). The iteration order depends on the hash values of the elements and the insertion order at the time of insertion, but it is explicitly documented as arbitrary. Python 3.7+ preserves insertion order for some cases as a CPython implementation detail, but this should never be relied upon.

2. **Q**: How would you implement a set-like data structure with O(1) operations if you couldn't use Python's built-in set?
   **A**: Implement a hash table with a backing array, a hash function, and open addressing or chaining for collision resolution. The table stores elements at indices computed by `hash(element) % table_size`. For duplicate detection, check if the element already exists at the probed position before insertion.

3. **Q**: What are the memory trade-offs between sets and lists for large collections?
   **A**: Sets use significantly more memory than lists (typically 4-8x more) because they maintain a sparse hash table to keep load factors low for O(1) performance. A list with 1 million integers uses ~8 MB, while a set with the same elements uses ~32-64 MB. However, sets provide O(1) membership testing versus O(n) for lists. The memory premium is the cost of speed.

## Practice Problems

### Easy

1. **Remove Duplicates**: Write a function that takes a list and returns a list with duplicates removed (order doesn't matter).

2. **Common Elements**: Given two lists, return a list of elements that appear in both.

3. **Unique Characters**: Write a function that checks if a string contains all unique characters.

4. **Set Union**: Given two lists, return all unique elements from both combined.

5. **Element Exists**: Write a function that checks if a given element exists in a set.

### Medium

1. **Jaccard Similarity**: Write a function that computes the Jaccard similarity between two sets.

2. **Symmetric Difference**: Given two lists, return elements that are in either list but not both.

3. **Subset Check**: Given two sets, determine if the first is a subset of the second.

4. **Word Overlap**: Given two sentences, find the set of words that appear in both.

5. **Unique Pairs**: Given a list of numbers, find all unique pairs that sum to a target value.

### Hard

1. **Anagram Groups**: Given a list of strings, group them into anagrams using sets/frozensets.

2. **Set Cover Problem**: Given a universal set and a collection of subsets, find the minimum number of subsets whose union equals the universal set (greedy approximation).

3. **Bloom Filter**: Implement a simple Bloom filter using a bit array and multiple hash functions for probabilistic membership testing (set-like but space-efficient).

## Solutions

### Easy Solutions

**1. Remove Duplicates**
```python
def remove_duplicates(items):
    return list(set(items))

print(remove_duplicates([1, 2, 2, 3, 3, 3, 4]))
# Output: [1, 2, 3, 4]
```

**2. Common Elements**
```python
def common_elements(list1, list2):
    return list(set(list1) & set(list2))

print(common_elements([1, 2, 3, 4], [3, 4, 5, 6]))
# Output: [3, 4]
```

**3. Unique Characters**
```python
def all_unique(s):
    return len(s) == len(set(s))

print(all_unique("abcde"), all_unique("hello"))
# Output: True False
```

**4. Set Union**
```python
def union_lists(list1, list2):
    return list(set(list1) | set(list2))

print(union_lists([1, 2, 3], [3, 4, 5]))
# Output: [1, 2, 3, 4, 5]
```

**5. Element Exists**
```python
def element_exists(s, elem):
    return elem in s

my_set = {10, 20, 30}
print(element_exists(my_set, 20), element_exists(my_set, 99))
# Output: True False
```

### Medium Solutions

**1. Jaccard Similarity**
```python
def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)

s1 = {1, 2, 3, 4}
s2 = {3, 4, 5, 6}
print(f"{jaccard(s1, s2):.2f}")
# Output: 0.33
```

**2. Symmetric Difference**
```python
def symmetric_diff(list1, list2):
    return list(set(list1) ^ set(list2))

print(symmetric_diff([1, 2, 3, 4], [3, 4, 5, 6]))
# Output: [1, 2, 5, 6]
```

**3. Subset Check**
```python
def is_subset(a, b):
    return a <= b

print(is_subset({1, 2}, {1, 2, 3}), is_subset({1, 4}, {1, 2, 3}))
# Output: True False
```

**4. Word Overlap**
```python
def word_overlap(sentence1, sentence2):
    words1 = set(sentence1.lower().split())
    words2 = set(sentence2.lower().split())
    return words1 & words2

print(word_overlap("the quick brown fox", "the lazy brown dog"))
# Output: {'the', 'brown'}
```

**5. Unique Pairs**
```python
def unique_pairs(nums, target):
    seen = set()
    pairs = set()
    for num in nums:
        complement = target - num
        if complement in seen:
            pair = tuple(sorted((num, complement)))
            pairs.add(pair)
        seen.add(num)
    return list(pairs)

print(unique_pairs([1, 2, 3, 4, 5, 6], 7))
# Output: [(3, 4), (2, 5), (1, 6)]
```

### Hard Solutions

**1. Anagram Groups**
```python
from collections import defaultdict

def group_anagrams(words):
    groups = defaultdict(list)
    for w in words:
        key = frozenset(w)  # Note: this doesn't handle duplicate letters well
        groups[key].append(w)
    return list(groups.values())

# Better approach uses sorted string as key
def group_anagrams_improved(words):
    groups = defaultdict(list)
    for w in words:
        key = "".join(sorted(w))
        groups[key].append(w)
    return list(groups.values())

print(group_anagrams_improved(["eat", "tea", "tan", "ate", "nat", "bat"]))
# Output: [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

**2. Set Cover (Greedy Approximation)**
```python
def greedy_set_cover(universe, subsets):
    uncovered = set(universe)
    covered_sets = []
    subsets = [set(s) for s in subsets]

    while uncovered:
        best_subset = max(subsets, key=lambda s: len(uncovered & s))
        covered_sets.append(best_subset)
        uncovered -= best_subset

    return covered_sets

universe = {1, 2, 3, 4, 5}
subsets = [{1, 2}, {2, 3, 4}, {4, 5}, {1, 5}]
result = greedy_set_cover(universe, subsets)
print([sorted(s) for s in result])
# Output: [[2, 3, 4], [1, 5]]
```

**3. Simple Bloom Filter**
```python
import hashlib

class BloomFilter:
    def __init__(self, size, num_hashes):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [False] * size

    def _hashes(self, item):
        item = str(item).encode()
        return [int(hashlib.md5(item + str(i).encode()).hexdigest(), 16) % self.size
                for i in range(self.num_hashes)]

    def add(self, item):
        for h in self._hashes(item):
            self.bit_array[h] = True

    def __contains__(self, item):
        return all(self.bit_array[h] for h in self._hashes(item))

bf = BloomFilter(1000, 3)
bf.add("apple")
bf.add("banana")
print("apple" in bf, "grape" in bf)
# Output: True False
```

## Related Concepts

- Dictionaries (similar hash table-based implementation)
- Lists (ordered vs unordered, duplicates allowed vs not)
- Tuples (immutable ordered collection)
- Frozenset (immutable variant of set)
- Hash tables (underlying implementation)
- Boolean algebra (set operations map to logical operations)
- Venn diagrams (visual representation of set operations)

## Next Concepts

- List comprehensions — concise list creation with filtering
- Dictionary comprehensions — building dictionaries from iterables
- Sorting techniques — sorted(), .sort(), and custom sort keys
- Collections module (Counter, defaultdict)
- Iterators and generators — lazy evaluation patterns

## Summary

Sets are unordered collections of unique, hashable elements optimized for membership testing, deduplication, and mathematical set operations. They provide O(1) average-time membership checks and support union (`|`), intersection (`&`), difference (`-`), and symmetric difference (`^`). Sets can be created with literal syntax (`{1, 2, 3}`) or the `set()` constructor. The `frozenset` variant is immutable and hashable, suitable for use as dictionary keys. Set comprehensions provide a concise syntax for building sets from iterables. Sets are essential for data cleaning, deduplication, and efficient membership queries across many domains, including AI/ML workflows.

## Key Takeaways

- Create with `{elem1, elem2}` or `set(iterable)`; use `set()` for empty sets
- Elements are unique, unordered, and must be hashable
- O(1) membership testing — use sets when you need fast `in` checks
- Set operations: `|` (union), `&` (intersection), `-` (difference), `^` (symmetric difference)
- Add/remove with `add()`, `remove()`, `discard()`, `pop()`
- `frozenset` is the immutable, hashable version
- Set comprehensions: `{x**2 for x in range(10)}`
- Use `issubset()`, `issuperset()`, `isdisjoint()` for relationship checks
- Convert to sorted list if order is needed: `sorted(my_set)`
- Sets are more memory-intensive than lists but much faster for membership tests
