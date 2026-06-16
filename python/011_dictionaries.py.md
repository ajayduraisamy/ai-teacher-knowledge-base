# Concept: Dictionaries

## Concept ID

PYT-011

## Difficulty

BEGINNER

## Domain

Python

## Module

Python Basics

## Learning Objectives

- Understand what dictionaries are and how they store key-value pairs
- Create dictionaries using literal syntax and the `dict()` constructor
- Access, add, modify, and remove dictionary elements
- Use dictionary methods effectively (keys, values, items, get, update, pop, etc.)
- Work with dictionary comprehensions and merging operators
- Apply specialized dict types: defaultdict, OrderedDict, Counter

## Prerequisites

- Basic Python syntax (variables, data types)
- Familiarity with lists and tuples
- Understanding of immutable vs mutable objects

## Definition

A dictionary is a built-in Python data structure that stores an unordered collection of key-value pairs. Each key must be unique and hashable (immutable), while values can be any Python object. Dictionaries are also known as "hash maps" or "associative arrays" in other languages. They provide O(1) average-time complexity for lookups, insertions, and deletions.

## Intuition

Think of a dictionary as a real-world phonebook. You look up a person's name (the key) to find their phone number (the value). You wouldn't flip through every page — you go directly to the name. That's exactly how Python dictionaries work: they use a hash of the key to jump directly to the value's storage location. Keys are like labels on filing cabinet drawers — each label is unique, and you open the drawer to get the contents inside.

## Why This Concept Matters

Dictionaries are one of the most versatile and frequently used data structures in Python. They appear in virtually every Python codebase — from configuration files and API responses to database records and AI model parameters. JSON, the universal data interchange format, maps directly to Python dictionaries. Understanding dictionaries is essential for data manipulation, building lookup tables, counting frequencies, grouping data, and representing structured information efficiently.

## Real World Examples

- **Configuration settings**: Storing application configuration as key-value pairs (e.g., `{"host": "localhost", "port": 8080}`)
- **HTTP headers**: Request and response headers are key-value mappings
- **Database records**: A row from a database query represented as a dictionary of column names to values
- **Counting word frequencies**: Mapping each unique word to how many times it appears
- **Memoization/caching**: Storing computed results keyed by their function arguments
- **API responses**: JSON payloads from REST APIs deserialize directly into Python dictionaries

## AI/ML Relevance

- **Hyperparameter dictionaries**: ML models use dictionaries to store hyperparameters (learning rate, batch size, epochs) passed to training functions
- **Feature maps**: Representing feature names mapped to their values for a data point
- **One-hot encoding**: Representing categorical variables where each category maps to a binary vector, often stored as a dictionary mapping category → index
- **Token-to-index mappings**: NLP pipelines convert words to integer IDs via lookup dictionaries
- **Model configuration**: Saving and loading model architectures and weights using dictionary-based configs
- **Evaluation metrics**: Tracking metrics like accuracy, precision, recall in a dictionary during training loops

## Code Examples

### Example 1: Creating Dictionaries

```python
# Literal syntax
student = {"name": "Alice", "age": 22, "major": "CS"}
print(student)
# Output: {'name': 'Alice', 'age': 22, 'major': 'CS'}

# Using dict() constructor with keyword arguments
instructor = dict(name="Bob", department="Math", years=10)
print(instructor)
# Output: {'name': 'Bob', 'department': 'Math', 'years': 10}

# Using dict() from a list of tuples
pairs = [("a", 1), ("b", 2), ("c", 3)]
converted = dict(pairs)
print(converted)
# Output: {'a': 1, 'b': 2, 'c': 3}

# Empty dictionary
empty = {}
print(type(empty), len(empty))
# Output: <class 'dict'> 0
```

### Example 2: Accessing and Modifying

```python
data = {"name": "Charlie", "score": 95, "grade": "A"}

# Access via key
print(data["name"])
# Output: Charlie

# get() with default if key missing
print(data.get("age", "N/A"))
# Output: N/A

# Adding a new key
data["age"] = 21
print(data)
# Output: {'name': 'Charlie', 'score': 95, 'grade': 'A', 'age': 21}

# Modifying an existing key
data["score"] = 98
print(data["score"])
# Output: 98

# setdefault — sets value only if key doesn't exist
data.setdefault("level", "senior")
data.setdefault("level", "junior")  # won't overwrite
print(data["level"])
# Output: senior
```

### Example 3: Dictionary Methods

```python
inventory = {"apples": 10, "bananas": 5, "oranges": 8}

# keys(), values(), items()
print(list(inventory.keys()))
# Output: ['apples', 'bananas', 'oranges']
print(list(inventory.values()))
# Output: [10, 5, 8]
print(list(inventory.items()))
# Output: [('apples', 10), ('bananas', 5), ('oranges', 8)]

# update — merge another dict
inventory.update({"apples": 15, "grapes": 12})
print(inventory)
# Output: {'apples': 15, 'bananas': 5, 'oranges': 8, 'grapes': 12}

# pop — remove and return value
removed = inventory.pop("bananas")
print(removed, inventory)
# Output: 5 {'apples': 15, 'oranges': 8, 'grapes': 12}

# popitem — remove and return last inserted (Python 3.7+)
key, val = inventory.popitem()
print(f"Removed: {key}={val}", inventory)
# Output: Removed: grapes=12 {'apples': 15, 'oranges': 8}

# clear — empty the dict
copy_inv = inventory.copy()
copy_inv.clear()
print(copy_inv)
# Output: {}
```

### Example 4: Dictionary Comprehension

```python
# Basic comprehension: squares
squares = {x: x**2 for x in range(1, 6)}
print(squares)
# Output: {1: 1, 2: 4, 3: 9, 4: 16, 5: 16}

# With condition: only even numbers
even_squares = {x: x**2 for x in range(1, 11) if x % 2 == 0}
print(even_squares)
# Output: {2: 4, 4: 16, 6: 36, 8: 64, 10: 100}

# Swapping keys and values
original = {"a": 1, "b": 2, "c": 3}
swapped = {v: k for k, v in original.items()}
print(swapped)
# Output: {1: 'a', 2: 'b', 3: 'c'}

# Transforming values
words = {"hello": 5, "world": 5, "python": 6}
caps = {k.upper(): v for k, v in words.items()}
print(caps)
# Output: {'HELLO': 5, 'WORLD': 5, 'PYTHON': 6}
```

### Example 5: Merging with | Operator (Python 3.9+)

```python
defaults = {"theme": "dark", "lang": "en", "notifications": True}
user_prefs = {"lang": "fr", "font_size": 14}

# Merging with | — user_prefs overrides defaults
config = defaults | user_prefs
print(config)
# Output: {'theme': 'dark', 'lang': 'fr', 'notifications': True, 'font_size': 14}

# |= in-place update
base = {"x": 10, "y": 20}
override = {"y": 99, "z": 30}
base |= override
print(base)
# Output: {'x': 10, 'y': 99, 'z': 30}

# Multiple merges
a = {"a": 1}
b = {"b": 2}
c = {"c": 3}
merged = a | b | c
print(merged)
# Output: {'a': 1, 'b': 2, 'c': 3}
```

### Example 6: defaultdict

```python
from collections import defaultdict

# Grouping items
words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
freq = defaultdict(int)
for w in words:
    freq[w] += 1
print(dict(freq))
# Output: {'apple': 3, 'banana': 2, 'cherry': 1}

# Group by first letter
cities = ["Boston", "Berlin", "Barcelona", "Paris", "Prague"]
by_letter = defaultdict(list)
for city in cities:
    by_letter[city[0]].append(city)
print(dict(by_letter))
# Output: {'B': ['Boston', 'Berlin', 'Barcelona'], 'P': ['Paris', 'Prague']}

# Nested defaultdict
nested = defaultdict(lambda: defaultdict(int))
nested["alice"]["math"] = 95
nested["alice"]["physics"] = 88
print(nested["alice"]["math"])
# Output: 95
```

### Example 7: OrderedDict and Counter

```python
from collections import OrderedDict, Counter

# OrderedDict — remembers insertion order
od = OrderedDict()
od["z"] = 1
od["a"] = 2
od["m"] = 3
print(list(od.keys()))
# Output: ['z', 'a', 'm']

# move_to_end
od.move_to_end("z")
print(list(od.keys()))
# Output: ['a', 'm', 'z']

# Counter — count hashable objects
nums = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
counter = Counter(nums)
print(counter)
# Output: Counter({4: 4, 3: 3, 2: 2, 1: 1})

# most_common
print(counter.most_common(2))
# Output: [(4, 4), (3, 3)]

# Counter arithmetic
c1 = Counter(a=3, b=1)
c2 = Counter(a=1, b=2)
print(c1 + c2)
# Output: Counter({'a': 4, 'b': 3})
print(c1 - c2)
# Output: Counter({'a': 2})
```

## Common Mistakes

1. **Using a mutable object as a key**: Lists, dictionaries, or sets as keys raise `TypeError: unhashable type`
2. **Assuming dicts are ordered before Python 3.7**: Prior to 3.7, insertion order was not guaranteed; relying on order in older versions causes bugs
3. **Accessing missing keys with bracket notation**: `d["missing"]` raises `KeyError` instead of returning `None` — use `d.get("missing")` instead
4. **Modifying a dict while iterating**: Adding or removing keys during iteration raises `RuntimeError: dictionary changed size during iteration`
5. **Confusing `copy()` with deep copy**: `d.copy()` is a shallow copy; nested mutable objects are shared, not duplicated
6. **Using `=` to merge dicts**: `d1 = d1 | d2` creates a new dict; `d1 |= d2` updates in place. Forgetting the difference wastes memory
7. **Overusing `defaultdict(int)` without realizing it never raises KeyError**: This silently returns 0 for missing keys, which can mask bugs
8. **Forgetting that `popitem()` removes the last inserted item**: It's LIFO (last-in, first-out), not FIFO

## Interview Questions

### Beginner

1. **Q**: How do you check if a key exists in a dictionary without raising an error?
   **A**: Use `key in my_dict` which returns `True` or `False`, or use `my_dict.get(key, default)` which returns the default if key is missing.

2. **Q**: What is the difference between `d[key]` and `d.get(key)`?
   **A**: `d[key]` raises a `KeyError` if the key doesn't exist. `d.get(key)` returns `None` (or a provided default) if the key is missing, without raising an error.

3. **Q**: How do you iterate over both keys and values in a dictionary?
   **A**: Use `for key, value in my_dict.items():` to loop over all key-value pairs.

4. **Q**: What types can be used as dictionary keys?
   **A**: Any immutable (hashable) type: strings, integers, floats, tuples (containing only hashable elements), frozensets. Lists, dictionaries, and sets cannot be keys.

5. **Q**: How do you create an empty dictionary?
   **A**: With `{}` or `dict()`. Note that `set()` creates an empty set, not an empty dictionary.

### Intermediate

1. **Q**: Explain the difference between shallow copy and deep copy for dictionaries.
   **A**: A shallow copy (`d.copy()` or `dict(d)`) creates a new dictionary but the values reference the same objects. A deep copy (`copy.deepcopy(d)`) recursively copies all nested objects so they are completely independent.

2. **Q**: How does `defaultdict` differ from a regular dictionary?
   **A**: `defaultdict` from `collections` takes a default factory function. When a missing key is accessed, it calls the factory to create and return a default value instead of raising `KeyError`.

3. **Q**: What is the time complexity of dictionary operations?
   **A**: Average O(1) for get, set, delete, and membership check. Worst-case O(n) if there are many hash collisions (rare due to Python's hash randomization).

4. **Q**: How do you merge two dictionaries in Python 3.9+?
   **A**: Use the `|` operator: `merged = d1 | d2`. For in-place update: `d1 |= d2`. In older versions, use `{**d1, **d2}` or `d1.update(d2)`.

5. **Q**: Why can't you modify a dictionary while iterating over it?
   **A**: Modifying the dictionary's size during iteration changes the underlying hash table structure, causing unpredictable behavior. Python raises `RuntimeError` to prevent this. Use `list(d.items())` to create a snapshot first.

### Advanced

1. **Q**: How does Python handle hash collisions in dictionaries?
   **A**: Python uses open addressing with pseudo-random probing. When two keys hash to the same slot, Python probes subsequent slots (using a perturbed sequence derived from the hash) until it finds an empty slot or the matching key. This is implemented in C for performance.

2. **Q**: What is the memory overhead of a dictionary compared to a list?
   **A**: Dictionaries have significantly higher memory overhead because they store hash values, key pointers, value pointers, and maintain a sparse hash table (typically 1/3 to 2/3 full). A dict with n entries allocates roughly 4-8x more memory than a list of n items, depending on the load factor and resizing history.

3. **Q**: How would you implement an LRU cache using OrderedDict?
   **A**: Use `OrderedDict` with `move_to_end()` on access and `popitem(last=False)` to evict the oldest entry when the cache exceeds capacity. This provides O(1) operations. Python's `functools.lru_cache` decorator uses this exact approach internally.

## Practice Problems

### Easy

1. **Word Counter**: Given a sentence string, return a dictionary with word counts.

2. **Grade Lookup**: Given a dict of student names to grades, write a function that returns the grade for a given student, or "Not found" if missing.

3. **Invert Dictionary**: Write a function that swaps keys and values (assume all values are unique).

4. **Filter by Value**: Given a dict, return a new dict with only entries whose values are greater than a threshold.

5. **Key Exists**: Write a function that checks if a list of keys all exist in a given dictionary.

### Medium

1. **Group By Length**: Given a list of strings, group them into a dictionary keyed by string length.

2. **Merge with Sum**: Given two dictionaries with numeric values, merge them by summing values for common keys.

3. **Nested Access**: Write a function that safely accesses a value in a nested dictionary given a dot-separated path like "a.b.c".

4. **Top N Frequent**: Given a list of items, return the top N most frequent items as a list of (item, count) tuples.

5. **Dict Diff**: Write a function that compares two dictionaries and returns keys that are added, removed, or changed.

### Hard

1. **Deep Merge**: Write a recursive function that deep-merges two nested dictionaries. Nested dict values with the same key should be merged recursively, not overwritten.

2. **JSON Flattener**: Write a function that flattens a nested dictionary into a single-level dictionary with dot-separated keys (e.g., `{"a": {"b": 1}}` becomes `{"a.b": 1}`).

3. **LRU Cache Implementation**: Implement an LRU (Least Recently Used) cache using `OrderedDict` with a fixed maximum size. The cache should support `get(key)` and `put(key, value)` in O(1) time.

## Solutions

### Easy Solutions

**1. Word Counter**
```python
def word_count(sentence):
    words = sentence.split()
    counts = {}
    for w in words:
        w = w.lower().strip(".,!?;:")
        counts[w] = counts.get(w, 0) + 1
    return counts

print(word_count("apple banana apple cherry banana apple"))
# Output: {'apple': 3, 'banana': 2, 'cherry': 1}
```

**2. Grade Lookup**
```python
def lookup_grade(roster, student):
    return roster.get(student, "Not found")

grades = {"Alice": "A", "Bob": "B", "Charlie": "A"}
print(lookup_grade(grades, "Bob"), lookup_grade(grades, "Diana"))
# Output: B Not found
```

**3. Invert Dictionary**
```python
def invert_dict(d):
    return {v: k for k, v in d.items()}

original = {"a": 1, "b": 2, "c": 3}
print(invert_dict(original))
# Output: {1: 'a', 2: 'b', 3: 'c'}
```

**4. Filter by Value**
```python
def filter_by_value(d, threshold):
    return {k: v for k, v in d.items() if v > threshold}

scores = {"Alice": 85, "Bob": 72, "Charlie": 90, "Diana": 65}
print(filter_by_value(scores, 80))
# Output: {'Alice': 85, 'Charlie': 90}
```

**5. Key Exists**
```python
def keys_exist(d, keys):
    return all(k in d for k in keys)

d = {"a": 1, "b": 2, "c": 3}
print(keys_exist(d, ["a", "c"]), keys_exist(d, ["a", "z"]))
# Output: True False
```

### Medium Solutions

**1. Group By Length**
```python
def group_by_length(words):
    result = {}
    for w in words:
        result.setdefault(len(w), []).append(w)
    return result

print(group_by_length(["cat", "dog", "apple", "bat", "cherry"]))
# Output: {3: ['cat', 'dog', 'bat'], 5: ['apple'], 6: ['cherry']}
```

**2. Merge with Sum**
```python
def merge_with_sum(d1, d2):
    result = d1.copy()
    for k, v in d2.items():
        result[k] = result.get(k, 0) + v
    return result

a = {"x": 10, "y": 20}
b = {"y": 5, "z": 15}
print(merge_with_sum(a, b))
# Output: {'x': 10, 'y': 25, 'z': 15}
```

**3. Nested Access**
```python
def deep_get(d, path):
    keys = path.split(".")
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key)
            if d is None:
                return None
        else:
            return None
    return d

data = {"a": {"b": {"c": 42}}}
print(deep_get(data, "a.b.c"), deep_get(data, "a.x.y"))
# Output: 42 None
```

**4. Top N Frequent**
```python
from collections import Counter

def top_n(items, n):
    return Counter(items).most_common(n)

items = [1, 1, 2, 2, 2, 3, 3, 3, 3, 4]
print(top_n(items, 2))
# Output: [(3, 4), (2, 3)]
```

**5. Dict Diff**
```python
def dict_diff(d1, d2):
    added = {k: d2[k] for k in d2 if k not in d1}
    removed = {k: d1[k] for k in d1 if k not in d2}
    changed = {k: (d1[k], d2[k]) for k in d1 if k in d2 and d1[k] != d2[k]}
    return {"added": added, "removed": removed, "changed": changed}

a = {"x": 1, "y": 2, "z": 3}
b = {"y": 99, "z": 3, "w": 10}
print(dict_diff(a, b))
# Output: {'added': {'w': 10}, 'removed': {'x': 1}, 'changed': {'y': (2, 99)}}
```

### Hard Solutions

**1. Deep Merge**
```python
def deep_merge(d1, d2):
    result = d1.copy()
    for k, v in d2.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result

a = {"a": 1, "b": {"c": 2, "d": 3}}
b = {"b": {"d": 99, "e": 4}, "f": 5}
print(deep_merge(a, b))
# Output: {'a': 1, 'b': {'c': 2, 'd': 99, 'e': 4}, 'f': 5}
```

**2. JSON Flattener**
```python
def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

nested = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
print(flatten_dict(nested))
# Output: {'a.b': 1, 'a.c.d': 2, 'e': 3}
```

**3. LRU Cache**
```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

cache = LRUCache(2)
cache.put("a", 1)
cache.put("b", 2)
print(cache.get("a"))
# Output: 1
cache.put("c", 3)
print(cache.get("b"))
# Output: -1
```

## Related Concepts

- Tuples (as immutable dictionary keys)
- Lists (ordered sequence vs keyed mapping)
- JSON module (serialize dicts to/from JSON)
- Collections module (defaultdict, OrderedDict, Counter, ChainMap)
- Hash tables (underlying implementation concept)
- Set (unordered collection of unique hashable elements)

## Next Concepts

- Sets — unordered collections of unique elements
- List comprehensions — concise way to create lists
- Dictionary comprehensions — building dictionaries with concise syntax
- Named tuples — lightweight immutable data containers
- Dataclasses — structured data containers with type hints

## Summary

Dictionaries are Python's built-in mapping type, storing key-value pairs with O(1) average access time. Keys must be immutable and unique. Dictionaries support a rich set of methods for manipulation, comprehensions for concise construction, and the `|` operator for merging (Python 3.9+). Specialized variants from the `collections` module — `defaultdict`, `OrderedDict`, and `Counter` — extend dictionary functionality for specific use cases. Dictionaries are fundamental to Python programming and are especially critical in data-intensive fields like AI/ML for representing configurations, feature mappings, and model parameters.

## Key Takeaways

- Create with `{}` or `dict()`, access with `d[key]` or `d.get(key)`
- Keys must be hashable (immutable); values can be anything
- Average O(1) lookup, insert, and delete time complexity
- Use `.keys()`, `.values()`, `.items()` for iteration
- `defaultdict` provides automatic default values for missing keys
- `Counter` simplifies counting and frequency analysis
- `OrderedDict` guarantees insertion order (though regular dicts also do in Python 3.7+)
- Dictionary comprehensions: `{k: v for k, v in iterable}`
- Merge with `|` (Python 3.9+) or `{**d1, **d2}` (older versions)
- Never modify a dictionary during iteration over its keys
