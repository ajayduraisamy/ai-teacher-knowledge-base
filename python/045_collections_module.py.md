# Concept: collections Module

## Concept ID

PYT-045

## Difficulty

Intermediate

## Domain

Python

## Module

Intermediate Python

## Learning Objectives

- Understand the `collections` module and its specialized container types
- Use `Counter` for counting and frequency analysis
- Leverage `defaultdict` for automatic missing-key handling
- Maintain insertion order with `OrderedDict`
- Use `deque` for efficient appends and pops from both ends
- Combine multiple dictionaries with `ChainMap`
- Extend dict and list functionality with `UserDict` and `UserList`
- Choose the right collection type for each use case

## Prerequisites

- Python dictionaries and their methods
- Lists, tuples, and basic data structures
- Object-oriented basics (inheritance)
- Understanding of hashable types

## Definition

The **`collections`** module is a standard Python library that provides specialized container datatypes extending the built-in types (`dict`, `list`, `tuple`, `set`). These containers offer optimized implementations for specific use cases: counting, grouping, ordered storage, double-ended queues, and dictionary composition.

## Intuition

Think of the `collections` module as a toolbox of specialized data containers. While Python's built-in `dict` and `list` are general-purpose, `collections` offers purpose-built variants: `Counter` for tallying, `defaultdict` for handling missing keys, `deque` for fast queue operations, and more. Each container solves a specific problem more elegantly than the general-purpose alternatives.

## Why This Concept Matters

The `collections` module reduces boilerplate code and improves performance. Without `Counter`, you write manual counting loops. Without `defaultdict`, you write `if key in dict` checks everywhere. Without `deque`, list-based queue operations are O(n). These specialized containers make code cleaner, faster, and more aligned with the problem domain.

## Real World Examples

- `Counter`: word frequency analysis, voting tallies, inventory counts
- `defaultdict`: grouping items by category, building adjacency lists
- `OrderedDict`: maintaining insertion order in LRU caches, ordered configurations
- `deque`: undo/redo operations, task queues, sliding windows
- `ChainMap`: layered configuration (defaults + user settings + environment)
- `UserDict`/`UserList`: creating custom dict/list subclasses with validation

## AI/ML Relevance

The `collections` module in AI/ML:
- `Counter` for class frequency analysis in imbalanced datasets
- `defaultdict` for building feature dictionaries and sparse representations
- `OrderedDict` for maintaining ordered model state dictionaries
- `deque` for sliding window data collection in time series
- `ChainMap` for layered model configurations (base + experiment overrides)
- `UserDict` for creating parameter dictionaries with validation

## Code Examples

### Example 1: Counter for frequency analysis

```python
from collections import Counter

words = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']
counter = Counter(words)

print(counter)
print(counter.most_common(2))

counter.update(['banana', 'date'])
print(counter['banana'])

c1 = Counter(a=3, b=1)
c2 = Counter(a=1, b=2)
print(c1 + c2)
print(c1 - c2)

# Output:
# Counter({'apple': 3, 'banana': 2, 'cherry': 1})
# [('apple', 3), ('banana', 2)]
# 3
# Counter({'a': 4, 'b': 3})
# Counter({'a': 2})
```

### Example 2: defaultdict for automatic missing keys

```python
from collections import defaultdict

data = [('fruit', 'apple'), ('fruit', 'banana'), ('veg', 'carrot'), ('fruit', 'cherry')]
groups = defaultdict(list)

for category, item in data:
    groups[category].append(item)

print(dict(groups))

word_counts = defaultdict(int)
for word in ['a', 'b', 'a', 'c', 'b', 'a']:
    word_counts[word] += 1

print(dict(word_counts))

def default_value():
    return 'N/A'

d = defaultdict(default_value)
print(d['missing'])

# Output:
# {'fruit': ['apple', 'banana', 'cherry'], 'veg': ['carrot']}
# {'a': 3, 'b': 2, 'c': 1}
# N/A
```

### Example 3: OrderedDict

```python
from collections import OrderedDict

od = OrderedDict()
od['z'] = 1
od['a'] = 2
od['c'] = 3

for k, v in od.items():
    print(f'{k}: {v}')

od.move_to_end('z')
print(list(od.keys()))

od.move_to_end('z', last=False)
print(list(od.keys()))

print(od.popitem())
print(od.popitem(last=False))

# Output:
# z: 1
# a: 2
# c: 3
# ['a', 'c', 'z']
# ['z', 'a', 'c']
# ('c', 3)
# ('z', 1)
```

### Example 4: deque

```python
from collections import deque

d = deque([1, 2, 3])
print(d)

d.append(4)
d.appendleft(0)
print(d)

print(d.pop())
print(d.popleft())
print(d)

d.rotate(1)
print(d)
d.rotate(-2)
print(d)

d = deque(maxlen=3)
for i in range(5):
    d.append(i)
    print(d)

# Output:
# deque([1, 2, 3])
# deque([0, 1, 2, 3, 4])
# 4
# 0
# deque([1, 2, 3])
# deque([3, 1, 2])
# deque([1, 2, 3])
# deque([0], maxlen=3)
# deque([0, 1], maxlen=3)
# deque([0, 1, 2], maxlen=3)
# deque([1, 2, 3], maxlen=3)
# deque([2, 3, 4], maxlen=3)
```

### Example 5: ChainMap

```python
from collections import ChainMap

defaults = {'theme': 'light', 'language': 'en', 'show_notifications': True}
user_prefs = {'theme': 'dark', 'language': 'fr'}
runtime_overrides = {'show_notifications': False}

config = ChainMap(runtime_overrides, user_prefs, defaults)
print(config['theme'])
print(config['language'])
print(config['show_notifications'])

user_prefs['theme'] = 'blue'
print(config['theme'])

debug_config = config.new_child({'debug': True})
print(debug_config['debug'])

# Output:
# dark
# fr
# False
# blue
# True
```

### Example 6: UserDict

```python
from collections import UserDict

class ValidatedDict(UserDict):
    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError('Keys must be strings')
        if not isinstance(value, (int, float)):
            raise TypeError('Values must be numbers')
        super().__setitem__(key, value)

d = ValidatedDict()
d['age'] = 30
d['score'] = 95.5
print(d['age'])
print(d)

try:
    d[42] = 'invalid'
except TypeError as e:
    print(e)

# Output:
# 30
# {'age': 30, 'score': 95.5}
# Keys must be strings
```

### Example 7: deque for sliding window

```python
from collections import deque

def sliding_average(data, window_size):
    window = deque(maxlen=window_size)
    averages = []
    for value in data:
        window.append(value)
        if len(window) == window_size:
            avg = sum(window) / window_size
            averages.append(avg)
    return averages

data = [10, 20, 30, 40, 50, 60, 70]
print(sliding_average(data, 3))

# Output:
# [20.0, 30.0, 40.0, 50.0, 60.0]
```

## Common Mistakes

1. Forgetting that `Counter` returns 0 (not KeyError) for missing keys, which can mask bugs
2. Passing a callable (not a type) to `defaultdict` — `defaultdict(list)` is correct
3. Assuming Python 3.7+ dict makes `OrderedDict` obsolete — `OrderedDict` still has `move_to_end()`
4. Using `deque` when a simple list suffices — `deque` is optimized for ends, not random access
5. Forgetting that `ChainMap` looks up keys in order and raises KeyError if not found

## Interview Questions

### Beginner - 5

1. What is `Counter` and how do you create one?
2. How does `defaultdict` differ from a regular dictionary?
3. What is the difference between `deque` and a regular list?
4. When would you use `OrderedDict`?
5. What is `ChainMap` useful for?

### Intermediate - 5

1. How does `Counter.most_common(n)` work internally?
2. When would you use `UserDict` instead of subclassing `dict` directly?
3. What is the time complexity of deque.appendleft() and deque.popleft()?
4. How do `OrderedDict.move_to_end()` and `popitem()` work for an LRU cache?
5. How does `ChainMap` handle key conflicts across multiple maps?

### Advanced - 3

1. Implement an LRU cache using `OrderedDict`.
2. How would you implement a `DefaultDict` that passes the missing key to the factory function?
3. Explain the internal implementation differences between `deque` and `list`.

## Practice Problems

### Easy - 5

1. Use `Counter` to find the most common character in a string.
2. Use `defaultdict` to group a list of numbers by even/odd.
3. Create a `deque` and add elements to both ends.
4. Use `OrderedDict` to create a dictionary sorted by insertion order.
5. Use `ChainMap` to merge two simple dictionaries.

### Medium - 5

1. Implement an LRU cache with `OrderedDict`.
2. Use `Counter` to implement a simple spell checker (word frequency with edit distance).
3. Build a nested `defaultdict` for a multi-level category tree.
4. Use `deque` for a round-robin task scheduler.
5. Create a `UserList` subclass that only accepts numeric values.

### Hard - 3

1. Implement a `DefaultOrderedDict` combining `defaultdict` and `OrderedDict`.
2. Build a multi-level config system using nested `ChainMap` with environment variable overrides.
3. Design a frequency-based caching system using `Counter` and `OrderedDict`.

## Solutions

### Easy 1

```python
from collections import Counter
text = 'hello world'
counter = Counter(text)
print(counter.most_common(3))

# Output:
# [('l', 3), ('o', 2), ('h', 1)]
```

### Medium 1

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

cache = LRUCache(3)
cache.put('a', 1)
cache.put('b', 2)
cache.put('c', 3)
print(cache.get('a'))
cache.put('d', 4)
print(list(cache.cache.keys()))

# Output:
# 1
# ['b', 'c', 'd']
```

### Hard 1

```python
from collections import OrderedDict, defaultdict

class DefaultOrderedDict(OrderedDict):
    def __init__(self, default_factory=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_factory = default_factory

    def __missing__(self, key):
        if self.default_factory is not None:
            self[key] = self.default_factory()
            return self[key]
        raise KeyError(key)

dod = DefaultOrderedDict(list)
dod['group1'].append('item1')
dod['group1'].append('item2')
dod['group2'].append('item3')
print(dict(dod))
print(list(dod.keys()))

# Output:
# {'group1': ['item1', 'item2'], 'group2': ['item3']}
# ['group1', 'group2']
```

## Related Concepts

- Built-in dict, list, tuple, set
- Dictionary comprehensions
- Data structures and algorithms
- `__missing__` hook for custom dict subclasses
- The `abc` module (abstract base classes for collections)
- Hashable and unhashable types

## Next Concepts

- Regular expressions (PYT-046)
- datetime module (PYT-047)
- JSON serialization (PYT-048)
- Custom data structures in Python

## Summary

The `collections` module provides specialized container datatypes that solve common programming problems more elegantly and efficiently than built-in types. From `Counter` for frequency analysis and `defaultdict` for automatic key handling, to `deque` for efficient queue operations and `OrderedDict` for ordered mappings, these tools are essential for writing clean, performant Python code.

## Key Takeaways

- `Counter` simplifies counting and frequency analysis with `most_common()`
- `defaultdict` eliminates manual missing-key checks with a factory function
- `OrderedDict` maintains insertion order and provides `move_to_end()` / `popitem()`
- `deque` offers O(1) appends and pops from both ends
- `ChainMap` combines multiple dicts for layered lookups
- `UserDict`/`UserList` are safer to subclass than built-in types
- AI/ML: use `Counter` for class distribution analysis, `defaultdict` for feature maps, and `deque` for sliding windows
