# Concept: Standard Library

## Concept ID

PYT-021

## Difficulty

Intermediate

## Domain

Python

## Module

Functions and Modules

## Learning Objectives

- Navigate Python's standard library effectively
- Use `os` for operating system interactions (paths, environment variables, directory walking)
- Use `sys` for interpreter-level operations (argv, path, exit, platform info)
- Perform mathematical and statistical computations with `math`, `statistics`, and `random`
- Handle dates and times with `datetime`
- Serialize data with `json`
- Write regular expressions with `re`
- Leverage `collections` for specialized data structures
- Use `itertools` for efficient iteration patterns

## Prerequisites

- PYT-020: Modules and Imports (basic import syntax)
- Basic Python data types (strings, lists, dicts)
- Understanding of iterators and generators

## Definition

The **Python Standard Library** is a vast collection of modules included with every Python installation. It provides ready-to-use solutions for file I/O, system calls, data serialization, text processing, mathematics, networking, concurrency, and more. The standard library follows the "batteries included" philosophy — common tasks can be accomplished without installing third-party packages.

## Intuition

Think of the standard library as a giant toolbox that comes free with your Python house. Instead of building every tool from scratch, you open the drawer and grab `os` for file operations, `json` for data exchange, `re` for pattern matching, and `collections` for advanced data structures. You don't need to go to the hardware store (PyPI) for everyday tasks — the tools are already in your kitchen.

## Why This Concept Matters

The standard library dramatically accelerates development. Before searching for third-party packages, you should check whether Python already includes what you need. The standard library is well-tested, well-documented, and guaranteed to be available. In AI/ML, you use `os` for data path management, `json` for model configs, `re` for text preprocessing, `collections` for counting and grouping, `itertools` for data batching, and `random` for data shuffling and sampling.

## Real World Examples

1. **Path management**: `os.path.join()` creates platform-independent file paths.
2. **Configuration files**: `json.load()` reads model hyperparameters from JSON configs.
3. **Text preprocessing**: `re.sub()` removes special characters from text data.
4. **Data batching**: `itertools.islice()` creates mini-batches from large datasets.
5. **Sampling**: `random.choice()` selects random items from a dataset.

## AI/ML Relevance

The standard library is the foundation of all ML workflows. `os` and `sys` manage file paths and environment variables. `json` and `csv` handle data serialization. `re` preprocesses text data. `collections.Counter` builds frequency distributions. `itertools` chains, batches, and cycles through data. `random` shuffles training data. `statistics` computes basic metrics. Understanding these modules reduces dependencies and deepens understanding of Python's capabilities.

## Code Examples

### Example 1: `os` — operating system interface

```python
import os

# Path manipulation
path = os.path.join("data", "train", "images")
print(path)
# Output: data\train\images (Windows) or data/train/images (Unix)

# Environment variables
print(os.environ.get("HOME", "Not set"))
# Output: C:\Users\Username (or /home/username on Unix)

# Directory walking
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".py"):
            print(os.path.join(root, file))
# Output: (recursively lists all .py files)

# File info
print(os.path.getsize(__file__))
# Output: (file size in bytes)

# Current working directory
print(os.getcwd())
# Output: D:\Ajay\ai-teacher-knowledge-base\python
```

### Example 2: `sys` — system-specific parameters

```python
import sys

# Command line arguments
print(sys.argv)
# Output (if run as: python script.py --lr 0.001 --epochs 10):
# ['script.py', '--lr', '0.001', '--epochs', '10']

# Python path
print(sys.path[:3])
# Output: ['D:\\Ajay\\ai-teacher-knowledge-base\\python', ...]

# Exit program
# sys.exit(0)  # Successful exit
# sys.exit(1)  # Error exit

# Platform info
print(sys.platform)
# Output: win32 (or linux, darwin, etc.)

# Version info
print(sys.version_info)
# Output: sys.version_info(major=3, minor=12, micro=0, ...)
```

### Example 3: `math` — mathematical functions

```python
import math

print(math.pi)
# Output: 3.141592653589793

print(math.e)
# Output: 2.718281828459045

print(math.sqrt(144))
# Output: 12.0

print(math.sin(math.pi / 2))
# Output: 1.0

print(math.log(100, 10))
# Output: 2.0

print(math.ceil(4.2))
# Output: 5

print(math.floor(4.8))
# Output: 4

print(math.factorial(5))
# Output: 120

print(math.gcd(48, 18))
# Output: 6
```

### Example 4: `statistics` — statistical functions

```python
import statistics

data = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]

print(statistics.mean(data))
# Output: 4.0

print(statistics.median(data))
# Output: 4.0

print(statistics.mode([1, 1, 2, 3, 3, 3, 4]))
# Output: 3

print(statistics.stdev(data))
# Output: 1.0801234497346435

print(statistics.variance(data))
# Output: 1.1666666666666667

print(statistics.quantiles(data, n=4))
# Output: [3.0, 4.0, 5.0]
```

### Example 5: `random` — pseudo-random number generation

```python
import random

# Float between 0 and 1
print(random.random())
# Output: 0.374540 (varies)

# Integer in range
print(random.randint(1, 10))
# Output: 7 (varies)

# Choice from sequence
colors = ["red", "green", "blue", "yellow"]
print(random.choice(colors))
# Output: blue (varies)

# Sample without replacement
print(random.sample(range(100), 5))
# Output: [42, 17, 83, 5, 91] (varies)

# Shuffle in place
cards = list(range(52))
random.shuffle(cards)
print(cards[:5])
# Output: [31, 7, 44, 19, 2] (varies)

# Reproducible results
random.seed(42)
print([random.randint(1, 6) for _ in range(3)])
# Output: [2, 1, 5] (deterministic with seed=42)
```

### Example 6: `datetime` — date and time handling

```python
import datetime

# Current datetime
now = datetime.datetime.now()
print(now)
# Output: 2026-06-16 15:30:00.123456 (example)

# Create specific date
date = datetime.date(2026, 12, 25)
print(date)
# Output: 2026-12-25

# Date arithmetic
today = datetime.date.today()
delta = datetime.timedelta(days=7)
next_week = today + delta
print(next_week)
# Output: 2026-06-23

# Formatting
print(now.strftime("%Y-%m-%d %H:%M:%S"))
# Output: 2026-06-16 15:30:00

# Parsing
parsed = datetime.datetime.strptime("2026-01-01", "%Y-%m-%d")
print(parsed)
# Output: 2026-01-01 00:00:00

# Timestamps
print(datetime.datetime.now().timestamp())
# Output: 1781741400.123456 (Unix timestamp)
```

### Example 7: `json` — JSON serialization

```python
import json

# Python dict to JSON string
config = {
    "learning_rate": 0.001,
    "epochs": 50,
    "optimizer": "adam",
    "layers": [784, 256, 64, 10],
    "dropout": 0.2,
    "use_cuda": True
}

json_str = json.dumps(config, indent=2)
print(json_str)
# Output:
# {
#   "learning_rate": 0.001,
#   "epochs": 50,
#   "optimizer": "adam",
#   "layers": [784, 256, 64, 10],
#   "dropout": 0.2,
#   "use_cuda": true
# }

# JSON string to Python dict
loaded = json.loads(json_str)
print(loaded["learning_rate"])
# Output: 0.001

# Read from file
# with open("config.json", "w") as f:
#     json.dump(config, f, indent=2)

# with open("config.json") as f:
#     config = json.load(f)
```

### Example 8: `re` — regular expressions

```python
import re

# Pattern matching
text = "Contact us at support@example.com or sales@company.org"
emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
print(emails)
# Output: ['support@example.com', 'sales@company.org']

# Substitution
cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", "Hello, World! #Python")
print(cleaned)
# Output: Hello World Python

# Validation
pattern = r"^\d{3}-\d{2}-\d{4}$"
print(bool(re.match(pattern, "123-45-6789")))
# Output: True
print(bool(re.match(pattern, "1234-56-789")))
# Output: False

# Splitting
parts = re.split(r"[,;]\s*", "apple, banana; cherry, date")
print(parts)
# Output: ['apple', 'banana', 'cherry', 'date']

# Named groups
match = re.search(r"(?P<name>\w+)@(?P<domain>\w+\.\w+)", "user@example.com")
print(match.group("name"))
# Output: user
print(match.group("domain"))
# Output: example.com
```

### Example 9: `collections` — specialized data structures

```python
from collections import Counter, defaultdict, deque, namedtuple, OrderedDict

# Counter: count hashable objects
fruits = ["apple", "banana", "apple", "orange", "banana", "apple"]
counter = Counter(fruits)
print(counter)
# Output: Counter({'apple': 3, 'banana': 2, 'orange': 1})
print(counter.most_common(2))
# Output: [('apple', 3), ('banana', 2)]

# defaultdict: dict with default factory
groups = defaultdict(list)
groups["a"].append(1)
groups["b"].append(2)
groups["a"].append(3)
print(dict(groups))
# Output: {'a': [1, 3], 'b': [2]}

# deque: double-ended queue with fast appends/pops
queue = deque(["a", "b", "c"])
queue.append("d")
queue.appendleft("z")
print(queue)
# Output: deque(['z', 'a', 'b', 'c', 'd'])
print(queue.popleft())
# Output: z

# namedtuple: lightweight immutable data class
Point = namedtuple("Point", ["x", "y"])
p = Point(10, 20)
print(p.x, p.y)
# Output: 10 20
print(p[0], p[1])
# Output: 10 20

# OrderedDict: dict that remembers insertion order (default in Python 3.7+)
ordered = OrderedDict()
ordered["z"] = 1
ordered["a"] = 2
ordered["m"] = 3
print(list(ordered.keys()))
# Output: ['z', 'a', 'm']
```

### Example 10: `itertools` — iterator building blocks

```python
import itertools

# count: infinite counter
counter = itertools.count(start=10, step=2)
print([next(counter) for _ in range(5)])
# Output: [10, 12, 14, 16, 18]

# cycle: infinite cycle through iterable
cycler = itertools.cycle(["train", "val", "test"])
print([next(cycler) for _ in range(5)])
# Output: ['train', 'val', 'test', 'train', 'val']

# chain: chain iterables together
combined = list(itertools.chain([1, 2, 3], [4, 5], [6]))
print(combined)
# Output: [1, 2, 3, 4, 5, 6]

# product: Cartesian product
for combo in itertools.product([0, 1], repeat=2):
    print(combo, end=" ")
# Output: (0, 0) (0, 1) (1, 0) (1, 1)

# permutations: all orderings
print(list(itertools.permutations([1, 2, 3], 2)))
# Output: [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

# combinations: all subsets
print(list(itertools.combinations([1, 2, 3], 2)))
# Output: [(1, 2), (1, 3), (2, 3)]

# islice: slice an iterator
print(list(itertools.islice(range(100), 10, 20)))
# Output: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

# groupby: group consecutive elements
data = [("A", 1), ("A", 2), ("B", 3), ("B", 4)]
for key, group in itertools.groupby(data, key=lambda x: x[0]):
    print(key, list(group))
# Output: A [('A', 1), ('A', 2)]
#         B [('B', 3), ('B', 4)]
```

## Common Mistakes

1. **Importing deeply nested modules incorrectly**: Use `from os.path import join` not `from os import path.join`.
2. **Using `os.system()` instead of `subprocess`**: `os.system()` is deprecated for running shell commands; use `subprocess.run()`.
3. **Forgetting `re.escape()`**: When using user input in regex patterns, escape special characters with `re.escape()`.
4. **Using mutable default arguments with `collections.defaultdict`**: `defaultdict(lambda: [])` avoids shared mutable state.
5. **Modifying a list while iterating**: Use `itertools.filterfalse` or list comprehensions instead of removing items during iteration.

## Interview Questions

### Beginner

1. What is the difference between `os.path.join()` and string concatenation for paths?
2. How do you parse a JSON string in Python?
3. What is `sys.argv` and how is it used?
4. How do you generate a random integer between 1 and 100?
5. What is the difference between `math.floor()` and `math.trunc()`?

### Intermediate

1. Write a one-liner using `collections.Counter` to find the most common word in a string.
2. How do you use `itertools.groupby` to group a list of dictionaries by a key?
3. Write a regex to extract all URLs from a text.
4. What is the difference between `datetime.date` and `datetime.datetime`?
5. How do you create a deque with a maximum length?

### Advanced

1. Implement a lazy batch generator for training data using `itertools.islice`.
2. How would you implement a rolling window over a sequence using `collections.deque`?
3. Write a function using `os.walk` that finds the largest file in a directory tree.

## Practice Problems

### Easy

1. Use `os.path.join` to create a path to "data/processed/train.csv".
2. Write a script that prints the current date in "YYYY-MM-DD" format.
3. Use `random.choice` to pick a random winner from a list of names.
4. Parse the JSON string `'{"name": "Alice", "age": 30}'`.
5. Use `math.sqrt` to compute the Euclidean distance between (1, 2) and (4, 6).

### Medium

1. Use `collections.Counter` to count word frequencies in a sentence.
2. Write a regex that validates a phone number in the format (123) 456-7890.
3. Use `itertools.product` to generate all possible binary strings of length 4.
4. Create a `namedtuple` `Student` with fields `name`, `grade`, `age`.
5. Use `os.walk` to count the total number of `.py` files in a project.

### Hard

1. Implement a sliding window of size k over a list using `collections.deque`.
2. Write a function that uses `itertools.tee` to create three independent iterators from a single iterable.
3. Implement a simple configuration loader that reads from JSON, supports environment variable overrides, and provides dot-notation access.

## Solutions

### Easy Solutions

```python
# 1
import os.path
path = os.path.join("data", "processed", "train.csv")

# 2
import datetime
print(datetime.date.today().strftime("%Y-%m-%d"))

# 3
import random
names = ["Alice", "Bob", "Charlie"]
print(random.choice(names))

# 4
import json
data = json.loads('{"name": "Alice", "age": 30}')

# 5
import math
dist = math.sqrt((4-1)**2 + (6-2)**2)
```

### Medium Solutions

```python
# 1
from collections import Counter
sentence = "the cat and the dog and the bird"
word_counts = Counter(sentence.split())
print(word_counts.most_common(1))

# 2
import re
pattern = r"^\(\d{3}\) \d{3}-\d{4}$"
valid = bool(re.match(pattern, "(123) 456-7890"))

# 3
from itertools import product
list(product([0, 1], repeat=4))

# 4
from collections import namedtuple
Student = namedtuple("Student", ["name", "grade", "age"])

# 5
count = 0
for root, dirs, files in os.walk("."):
    count += sum(1 for f in files if f.endswith(".py"))
```

### Hard Solutions

```python
# 1 — Sliding window
from collections import deque
def sliding_window(iterable, k):
    it = iter(iterable)
    window = deque(itertools.islice(it, k), maxlen=k)
    if len(window) == k:
        yield tuple(window)
    for item in it:
        window.append(item)
        yield tuple(window)

# 2 — Tee
from itertools import tee
def tee_example(iterable):
    a, b, c = tee(iterable, 3)
    return list(a), list(b), list(c)

# 3 — Config loader
import json, os
class Config:
    def __init__(self, path):
        with open(path) as f:
            self._data = json.load(f)
        for key in list(self._data.keys()):
            env_val = os.environ.get(key.upper())
            if env_val is not None:
                self._data[key] = env_val
    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(name)
```

## Related Concepts

- PYT-020: Modules and Imports
- PYT-022: Packages and __init__.py
- PYT-023: Virtual Environments
- PYT-030: File I/O

## Next Concepts

- PYT-050: Third-Party Libraries (NumPy, Pandas)
- PYT-060: Data Serialization
- PYT-070: Logging and Debugging

## Summary

Python's standard library provides essential modules for everyday programming. `os` handles operating system interactions, `sys` provides interpreter access, `math` and `statistics` cover numerical operations, `random` generates pseudo-random numbers, `datetime` manages dates and times, `json` serializes data, `re` enables pattern matching, `collections` offers advanced data structures, and `itertools` provides efficient iteration patterns. Mastering the standard library reduces external dependencies and accelerates development.

## Key Takeaways

- Check the standard library before installing third-party packages.
- Use `os.path` for cross-platform path manipulation; never hardcode path separators.
- `json.dumps()`/`json.loads()` for serialization; `json.dump()`/`json.load()` for files.
- `re` module is powerful but use raw strings (`r"pattern"`) to avoid escape issues.
- `collections.Counter`, `defaultdict`, `deque`, and `namedtuple` solve common data structure needs.
- `itertools` provides memory-efficient iteration patterns: `chain`, `product`, `islice`, `groupby`.
- Always seed `random` with `random.seed()` for reproducible results in ML.
