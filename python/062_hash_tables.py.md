# Concept: Hash Tables

## Concept ID

PYT-062

## Difficulty

Intermediate

## Domain

Python

## Module

Data Structures and Algorithms

## Learning Objectives

- Understand how hash tables work: hash function, buckets, collision resolution
- Use Python's hash() function and understand its properties
- Explain CPython 3.10+ compact dict internals
- Implement custom __hash__ and __eq__ methods
- Understand load factor and rehashing
- Compare open addressing vs chaining collision resolution

## Prerequisites

- Familiarity with Python dict and set operations
- Understanding of Big O notation (PYT-056)
- Basic knowledge of object equality and identity

## Definition

A hash table (hash map) is a data structure that maps keys to values by computing an index (hash) from the key. It provides O(1) average-time insertion, deletion, and lookup. Python's dict and set are hash-table-based. The hash function converts a key into an integer index, and collision resolution handles cases where different keys produce the same index.

## Intuition

Imagine a library with numbered shelves. A librarian uses a formula (hash function) based on the book title to compute the shelf number. Most books get their own shelf, but sometimes two books map to the same shelf (collision). The librarian has a system for that — either placing them side by side (chaining) or finding the next empty shelf (open addressing). Looking up a book using this system is near-instant.

## Why This Concept Matters

Hash tables are the most widely used data structure in computing. Python dicts power the language's object model (attribute lookup, namespace resolution), JSON parsing, caching, and countless algorithms. Understanding hash table internals helps you write correct __hash__ and __eq__ methods, choose appropriate key types, and debug performance issues caused by poor hash distribution.

## Real World Examples

1. **Database Indexing:** Hash indexes for equality lookups.
2. **Caching:** Memcached and Redis are distributed hash tables.
3. **Symbol Tables:** Compilers use hash tables for variable lookup.
4. **URL Shorteners:** Mapping short codes to long URLs.
5. **Blockchain:** Each block's hash links to the previous block.

## AI/ML Relevance

- **Feature Hashing:** Mapping categorical features to fixed-size vectors.
- **Embedding Lookup:** Token-to-vector mappings are hash tables.
- **Count-Min Sketch:** Probabilistic hash-based frequency estimation.
- **Bloom Filters:** Space-efficient set membership using multiple hashes.
- **LRU Cache:** Combining hash table with doubly linked list.

## Code Examples

### Example 1: Basic Dict Operations

`python
d = {}
d["name"] = "Alice"
d["age"] = 30
d["city"] = "New York"

print(d["name"])
print(d.get("country", "Unknown"))
print(len(d))
print("age" in d)
# Output:
# Alice
# Unknown
# 3
# True
`

### Example 2: hash() Function

`python
print(hash(42))
print(hash("hello"))
print(hash((1, 2, 3)))

try:
    hash([1, 2, 3])
except TypeError as e:
    print(f"Cannot hash list: {e}")
# Output:
# 42
# 8654326037129430268
# 977326415226631523
# Cannot hash list: unhashable type: 'list'
`

### Example 3: Custom __hash__ and __eq__

`python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __hash__(self):
        return hash((self.name, self.age))

    def __eq__(self, other):
        if not isinstance(other, Person):
            return NotImplemented
        return self.name == other.name and self.age == other.age

    def __repr__(self):
        return f"Person({self.name}, {self.age})"

p1 = Person("Alice", 30)
p2 = Person("Alice", 30)
p3 = Person("Bob", 25)

d = {p1: "engineer"}
print(d[p2])
print(d.get(p3, "not found"))
# Output:
# engineer
# not found
`

### Example 4: Load Factor and Performance

`python
import time

# Good hash distribution
good = {i: i for i in range(1000000)}
start = time.time()
for i in range(10000):
    _ = good[i]
print(f"Good hash lookups: {time.time() - start:.4f}s")

# Poor hash — all keys have same hash (worst case)
class BadKey:
    def __hash__(self):
        return 42
    def __init__(self, val):
        self.val = val
    def __eq__(self, other):
        return self.val == other.val

bad = {BadKey(i): i for i in range(10000)}
start = time.time()
for i in range(1000):
    _ = bad[BadKey(i)]
print(f"Bad hash lookups: {time.time() - start:.4f}s")
# Output:
# Good hash lookups: 0.0012s
# Bad hash lookups: 0.0854s
`

### Example 5: Dict as Cache (Memoization)

`python
def fib_memo(n, cache={}):
    if n in cache:
        return cache[n]
    if n <= 1:
        return n
    cache[n] = fib_memo(n - 1) + fib_memo(n - 2)
    return cache[n]

import time
start = time.time()
print(f"fib(100) = {fib_memo(100)}")
print(f"Time: {time.time() - start:.4f}s")

print(f"Cache size: {len(fib_memo.__defaults__[0])}")
# Output:
# fib(100) = 354224848179261915075
# Time: 0.0003s
# Cache size: 101
`

### Example 6: Hash Table Collisions Visualization

`python
def simple_hash(key, table_size):
    return sum(ord(c) for c in str(key)) % table_size

keys = ["apple", "banana", "cherry", "date", "elderberry"]
table_size = 5
table = [[] for _ in range(table_size)]

for k in keys:
    h = simple_hash(k, table_size)
    table[h].append(k)
    print(f"'{k}' -> bucket {h}")

print(f"\nHash table: {table}")
# Output:
# 'apple' -> bucket 3
# 'banana' -> bucket 2
# 'cherry' -> bucket 1
# 'date' -> bucket 4
# 'elderberry' -> bucket 1
# Hash table: [[], ['cherry', 'elderberry'], ['banana'], ['apple'], ['date']]
`

### Example 7: Counter with DefaultDict

`python
from collections import defaultdict

text = "the quick brown fox jumps over the lazy dog"
counter = defaultdict(int)
for word in text.split():
    counter[word] += 1

print(dict(counter))
# Output: {'the': 2, 'quick': 1, 'brown': 1, 'fox': 1, 'jumps': 1, 'over': 1, 'lazy': 1, 'dog': 1}
`

## Common Mistakes

1. **Using Mutable Keys:** Lists, dicts, and sets cannot be used as dict keys because they are unhashable.
2. **Inconsistent __hash__ and __eq__:** Objects that are equal must have the same hash; violating this breaks dicts/sets.
3. **Assuming Dict Order Before 3.7:** Python 3.7+ guarantees insertion order, but earlier versions do not.
4. **Poor Hash Distribution:** Custom __hash__ that returns a constant makes dict degrade to O(n).
5. **Modifying Keys After Insertion:** Modifying a key after using it in a dict changes its hash and breaks lookup.
6. **Overriding __eq__ Without __hash__:** This sets __hash__ to None, making the object unhashable.
7. **Ignoring Collision Attacks:** An attacker can craft keys with the same hash, causing O(n) performance (CVE-2012-1150 fixed this by randomizing hash seed).

## Interview Questions

### Beginner

1. What is a hash table?
2. What is the time complexity of dict lookup?
3. What types can be used as dict keys?
4. What is the hash() function used for?
5. How do you check if a key exists in a dict?

### Intermediate

1. Explain how Python dicts handle collisions.
2. What are the requirements for implementing __hash__ and __eq__?
3. How does Python 3.6+ compact dict save memory?
4. What is load factor and why does it matter?
5. How does Python randomize hash values for security?

### Advanced

1. Explain CPython 3.10+ dict implementation in detail (combined table, indices array).
2. How would you implement a hash table from scratch in Python?
3. Describe how Python's set intersection uses hash tables for O(min(n,m)) performance.

## Practice Problems

### Easy

1. **Two Sum:** Find two numbers in a list that sum to a target.
2. **First Repeating Character:** Find the first character that repeats in a string.
3. **Word Count:** Count word frequencies in a sentence.
4. **Intersection:** Find common elements between two lists.
5. **Anagram Check:** Check if two strings are anagrams.

### Medium

1. **Group Anagrams:** Group a list of strings by anagram.
2. **Subarray Sum:** Find subarrays with a given sum.
3. **LRU Cache:** Implement LRU cache using dict + doubly linked list.
4. **Longest Consecutive Sequence:** Find longest consecutive element sequence.
5. **Top K Frequent:** Find K most frequent elements.

### Hard

1. **Hash Table Implementation:** Build a hash table from scratch with open addressing.
2. **Design a URL Shortener:** Design a system that maps short URLs to long URLs.
3. **Consistent Hashing:** Implement consistent hashing for distributed caching.

## Solutions

### Solution to Easy 1: Two Sum

`python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

print(two_sum([2, 7, 11, 15], 9))
print(two_sum([3, 2, 4], 6))
# Output:
# [0, 1]
# [1, 2]
`

### Solution to Medium 1: Group Anagrams

`python
from collections import defaultdict

def group_anagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = "".join(sorted(s))
        groups[key].append(s)
    return list(groups.values())

print(group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
# Output: [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
`

### Solution to Hard 1: Hash Table from Scratch

`python
class HashTable:
    def __init__(self, capacity=8):
        self.capacity = capacity
        self.size = 0
        self.table = [None] * capacity
        self.DELETED = object()

    def _hash(self, key):
        return hash(key) % self.capacity

    def _probe(self, key):
        i = self._hash(key)
        while self.table[i] is not None:
            if self.table[i] is self.DELETED:
                return i
            if self.table[i][0] == key:
                return i
            i = (i + 1) % self.capacity
        return i

    def _resize(self):
        old_table = self.table
        self.capacity *= 2
        self.size = 0
        self.table = [None] * self.capacity
        for entry in old_table:
            if entry and entry is not self.DELETED:
                self[entry[0]] = entry[1]

    def __setitem__(self, key, value):
        if self.size >= self.capacity // 2:
            self._resize()
        i = self._probe(key)
        if self.table[i] is None or self.table[i] is self.DELETED:
            self.size += 1
        self.table[i] = (key, value)

    def __getitem__(self, key):
        i = self._probe(key)
        if self.table[i] is None or self.table[i] is self.DELETED:
            raise KeyError(key)
        return self.table[i][1]

    def __delitem__(self, key):
        i = self._probe(key)
        if self.table[i] is None or self.table[i] is self.DELETED:
            raise KeyError(key)
        self.table[i] = self.DELETED
        self.size -= 1

ht = HashTable()
ht["name"] = "Alice"
ht["age"] = 30
print(ht["name"])
del ht["age"]
# Output: Alice
`

## Related Concepts

- **Sets:** Python sets are hash tables without values.
- **Dict Internals:** Understanding how CPython implements dicts.
- **Hashing Algorithms:** SHA, MD5 for cryptographic vs non-cryptographic hashing.
- **Load Factor:** Tradeoff between memory and performance.

## Next Concepts

- **063 — Graph Algorithms:** Adjacency lists use hash tables for O(1) neighbor lookup.
- **064 — Sorting:** Using dicts for counting sort / bucket sort.

## Summary

Hash tables map keys to values using a hash function to compute an index. Python's dict and set are hash-table-based with O(1) average operations. Custom objects need __hash__ and __eq__ to work as keys. CPython 3.6+ uses a compact dict implementation with separate indices and entries arrays, reducing memory usage by ~30%.

## Key Takeaways

- Dict/set lookups are O(1) average, O(n) worst case (hash collisions).
- Only immutable, hashable objects can be dict keys.
- Custom __hash__ must be consistent with __eq__.
- CPython 3.6+ dict preserves insertion order.
- Load factor triggers rehashing — Python resizes when ~2/3 full.
- Poor hash functions degrade dict performance to O(n).
- Use defaultdict and Counter from collections for common patterns.
