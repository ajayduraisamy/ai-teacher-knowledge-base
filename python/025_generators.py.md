# Concept: Generators

## Concept ID

PYT-025

## Difficulty

Advanced

## Domain

Python

## Module

Functions and Modules

## Learning Objectives

- Understand generators as functions that produce sequences lazily
- Differentiate `yield` from `return`
- Write generator functions using `yield`
- Use generator expressions for memory-efficient iteration
- Drive generators with `next()` and handle `StopIteration`
- Implement two-way communication using `generator.send()`
- Delegate to sub-generators with `yield from`
- Build AI/ML data pipelines with generators

## Prerequisites

- PYT-016: Functions (return values, function execution)
- PYT-018: Lambda Functions
- Understanding of iterables and iterators
- Basic knowledge of list comprehensions

## Definition

A **generator** is a function that produces a sequence of values lazily using the `yield` keyword. Unlike a regular function that computes all values at once and returns them, a generator yields one value at a time, pausing its state between each yield. When the generator is next iterated, it resumes execution from where it paused. Generators are memory-efficient because they don't store the entire sequence in memory.

```python
def count_up_to(n):
    i = 0
    while i < n:
        yield i
        i += 1
```

## Intuition

Think of a generator as a vending machine that restocks one item at a time. When you press the button (call `next()`), it dispenses one item and waits. It doesn't load all items into the tray at once — it only retrieves the next one when asked. A regular function is like a catering service that brings all the food at once. A generator is like a tapas bar that brings one small plate at a time, exactly when you're ready for it.

## Why This Concept Matters

Generators are fundamental for processing large datasets that don't fit in memory. In AI/ML, training data often consists of millions of examples — loading everything at once would exhaust RAM. Generators enable lazy data loading, on-the-fly preprocessing, infinite data streams for data augmentation, and efficient batch generation. Understanding generators is essential for implementing custom data loaders in PyTorch, TensorFlow, and other ML frameworks.

## Real World Examples

1. **Data loading**: PyTorch's `DataLoader` uses generators behind the scenes to yield batches.
2. **Large file processing**: Reading a 100GB CSV file line by line using a generator.
3. **Streaming API responses**: Processing tweets from the Twitter API as they arrive.
4. **Infinite sequences**: Generating Fibonacci numbers, prime numbers, or random noise indefinitely.
5. **Pipelines**: Chaining multiple generators to build data preprocessing pipelines.

## AI/ML Relevance

Generators are essential in AI/ML for efficient data handling. PyTorch's `DataLoader` yields batches of data lazily during training. TensorFlow's `tf.data.Dataset` uses generator-like semantics for pipeline construction. Custom training loops often use generators for:
- Loading data from disk on-the-fly
- Applying data augmentation stochastically
- Yielding infinite streams of augmented data
- Batching variable-length sequences
- Shuffling large datasets without loading into RAM

The `yield from` syntax is particularly useful for composing preprocessing pipelines.

## Code Examples

### Example 1: Basic generator function

```python
def count_up_to(n):
    """Generator that counts from 0 to n-1."""
    i = 0
    while i < n:
        yield i
        i += 1

# Using the generator
counter = count_up_to(5)
print(next(counter))
# Output: 0
print(next(counter))
# Output: 1
print(next(counter))
# Output: 2
print(next(counter))
# Output: 3
print(next(counter))
# Output: 4
# print(next(counter))  # Raises StopIteration

# Iterating with for loop (handles StopIteration automatically)
for num in count_up_to(3):
    print(num, end=" ")
# Output: 0 1 2
```

### Example 2: Generator for memory-efficient file reading

```python
def read_large_file(file_path):
    """Read a large file line by line without loading it all into memory."""
    with open(file_path, "r") as f:
        for line in f:
            yield line.strip()

# Usage:
# for line in read_large_file("huge_dataset.csv"):
#     process(line)
```

### Example 3: Generator expressions

```python
# Generator expression vs list comprehension
import sys

# List comprehension (creates full list in memory)
squares_list = [x ** 2 for x in range(1000)]
print(f"List size: {sys.getsizeof(squares_list)} bytes")
# Output: List size: 8856 bytes

# Generator expression (lazy, produces values on demand)
squares_gen = (x ** 2 for x in range(1000))
print(f"Generator size: {sys.getsizeof(squares_gen)} bytes")
# Output: Generator size: 208 bytes

# Both produce the same values
print(sum(squares_gen))
# Output: 332833500

# Generator expressions work with any iterable protocol
gen = (x for x in range(10) if x % 2 == 0)
print(list(gen))
# Output: [0, 2, 4, 6, 8]
```

### Example 4: Infinite generator

```python
def fibonacci():
    """Generate Fibonacci numbers indefinitely."""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

fib = fibonacci()
print([next(fib) for _ in range(10)])
# Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# Infinite random noise generator (for ML data augmentation)
import random

def noise_generator(mean=0.0, std=1.0):
    """Generate infinite Gaussian noise."""
    while True:
        yield random.gauss(mean, std)

noise = noise_generator()
print([round(next(noise), 2) for _ in range(5)])
# Output: [0.12, -0.45, 1.33, -0.78, 0.56] (varies)
```

### Example 5: `generator.send()` — two-way communication

```python
def echo():
    """Generator that echoes what it receives."""
    while True:
        received = yield
        print(f"Received: {received}")

gen = echo()
next(gen)  # Prime the generator (advance to first yield)
gen.send("Hello")
# Output: Received: Hello
gen.send(42)
# Output: Received: 42
gen.close()

# Practical example: accumulator with send
def accumulator():
    total = 0
    while True:
        value = yield total
        if value is not None:
            total += value

acc = accumulator()
next(acc)  # Prime
print(acc.send(10))
# Output: 10
print(acc.send(5))
# Output: 15
print(acc.send(20))
# Output: 35
```

### Example 6: `yield from` — delegating to sub-generators

```python
def sub_generator(values):
    for v in values:
        yield v

def main_generator():
    yield "start"
    yield from sub_generator([1, 2, 3])
    yield "middle"
    yield from sub_generator([4, 5])
    yield "end"

print(list(main_generator()))
# Output: ['start', 1, 2, 3, 'middle', 4, 5, 'end']

# Flatten nested iterables with yield from
def flatten(nested):
    for item in nested:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield item

nested = [1, [2, [3, 4], 5], 6]
print(list(flatten(nested)))
# Output: [1, 2, 3, 4, 5, 6]
```

### Example 7: AI/ML batch generator for training

```python
import random
import math

def batch_generator(data, batch_size=32, shuffle=True):
    """Yield batches of data for training.

    Args:
        data: List of training examples
        batch_size: Number of examples per batch
        shuffle: Whether to shuffle data each epoch

    Yields:
        Batches of data as lists
    """
    data = list(data)
    n_samples = len(data)

    while True:  # Infinite generator for multi-epoch training
        indices = list(range(n_samples))
        if shuffle:
            random.shuffle(indices)

        for start in range(0, n_samples, batch_size):
            batch_indices = indices[start:start + batch_size]
            batch = [data[i] for i in batch_indices]
            yield batch

# Simulate training data
train_data = [(i, i * 2) for i in range(100)]

# Create batch generator
batcher = batch_generator(train_data, batch_size=10)

# Get first epoch
print("First 3 batches of epoch 1:")
for _ in range(3):
    batch = next(batcher)
    print(f"  Batch size: {len(batch)}, first item: {batch[0]}")

# Get first batch of epoch 2
print("First batch of epoch 2:")
batch = next(batcher)
print(f"  Batch size: {len(batch)}, first item: {batch[0]}")
```

### Example 8: Data augmentation pipeline using generators

```python
import random

def identity(x):
    return x

def add_noise(x, noise_level=0.01):
    return x + random.gauss(0, noise_level)

def rotate(x, angle=10):
    """Simulated rotation."""
    return f"rotate({x}, {angle})"

def augment_pipeline(data, augmentations=None):
    """Generator that applies a sequence of augmentations to data."""
    if augmentations is None:
        augmentations = [identity]

    for item in data:
        augmented = item
        for aug in augmentations:
            augmented = aug(augmented)
        yield augmented

# Usage
data = [1, 2, 3, 4, 5]
augmentations = [lambda x: x * 1.1]  # Simple scaling
augmented = augment_pipeline(data, augmentations)

print(list(augmented))
# Output: [1.1, 2.2, 3.3000000000000003, 4.4, 5.5]
```

### Example 9: Pipeline of generators for ML preprocessing

```python
def load_data(paths):
    """Generator that yields raw data paths."""
    for path in paths:
        yield path

def read_files(paths):
    """Generator that reads file contents."""
    for path in paths:
        # Simulating reading a file
        yield f"content_of_{path}"

def tokenize(texts):
    """Generator that tokenizes text."""
    for text in texts:
        yield text.split("_")

def filter_short(tokens_list, min_len=2):
    """Generator that filters short sequences."""
    for tokens in tokens_list:
        if len(tokens) >= min_len:
            yield tokens

# Compose the pipeline
paths = ["doc1.txt", "doc2.txt", "a.txt"]
pipeline = filter_short(
    tokenize(
        read_files(
            load_data(paths)
        )
    ),
    min_len=2
)

print(list(pipeline))
# Output: [['content', 'of', 'doc1', 'txt'], ['content', 'of', 'doc2', 'txt']]
```

### Example 10: Generator for cross-validation splits

```python
def kfold_split(data, k=5):
    """Generator that yields train/val indices for k-fold cross-validation."""
    n = len(data)
    indices = list(range(n))
    random.shuffle(indices)
    fold_size = math.ceil(n / k)

    for fold in range(k):
        val_start = fold * fold_size
        val_end = min(val_start + fold_size, n)
        val_indices = indices[val_start:val_end]
        train_indices = indices[:val_start] + indices[val_end:]
        yield train_indices, val_indices

# Example
data = list(range(20))
for fold, (train_idx, val_idx) in enumerate(kfold_split(data, k=5)):
    print(f"Fold {fold}: train={len(train_idx)}, val={len(val_idx)}")
# Output:
# Fold 0: train=16, val=4
# Fold 1: train=16, val=4
# Fold 2: train=16, val=4
# Fold 3: train=16, val=4
# Fold 4: train=16, val=4
```

## Common Mistakes

1. **Confusing `yield` with `return`**: A function with `yield` is a generator — it returns a generator object, not the yielded value. Calling it does not execute the body.
2. **Exhausting a generator**: Generators can only be iterated once. After `StopIteration`, the generator is exhausted and cannot be reused.
3. **Forgetting to prime a generator before `send()`**: You must call `next()` or `send(None)` first to advance the generator to the first `yield` before sending actual values.
4. **Assuming generators are always faster**: Generators are memory-efficient, not necessarily faster. The overhead of `yield`/`next()` can make them slower than lists for small datasets.
5. **Not closing generators**: Generators can hold resources (file handles, network connections). Use `.close()` or `with` statements to ensure cleanup.

## Interview Questions

### Beginner

1. What is a generator? How is it different from a regular function?
2. What keyword is used to create a generator function?
3. What is the difference between `yield` and `return`?
4. How do you get values from a generator?
5. What happens when a generator reaches the end of its function?

### Intermediate

1. What is a generator expression? How is it different from a list comprehension?
2. Explain how `generator.send()` works and when you would use it.
3. What does `yield from` do?
4. Why are generators memory-efficient compared to lists?
5. Write a generator that yields the first n prime numbers.

### Advanced

1. Implement a generator that simulates a sliding window over a sequence.
2. How would you implement coroutines using `yield` and `send()`?
3. Explain the difference between subgenerators and `yield from` in terms of delegation.

## Practice Problems

### Easy

1. Write a generator `even_numbers(n)` that yields even numbers up to n.
2. Write a generator `countdown(n)` that counts down from n to 0.
3. Write a generator expression that yields squares of odd numbers less than 20.
4. Write a generator that yields characters of a string one at a time.
5. Write a generator that yields the current timestamp each time it's called.

### Medium

1. Write a generator that yields the first n Fibonacci numbers.
2. Write a generator that reads a CSV file line by line and splits each line into columns.
3. Write a generator that yields overlapping n-grams from a sequence.
4. Write a generator `cyclic_counter(max=10)` that cycles from 0 to max-1 indefinitely.
5. Write a generator pipeline that chains a `filter`, `transform`, and `limit` generator.

### Hard

1. Implement a generator-based data loader that supports shuffling, batching, and prefetching.
2. Write a coroutine (two-way generator) that implements a simple state machine.
3. Implement a generator that performs online mean and variance computation over a data stream.

## Solutions

### Easy Solutions

```python
# 1
def even_numbers(n):
    for i in range(0, n, 2):
        yield i

# 2
def countdown(n):
    while n >= 0:
        yield n
        n -= 1

# 3
gen = (x**2 for x in range(20) if x % 2 == 1)

# 4
def char_generator(s):
    for ch in s:
        yield ch

# 5
import time
def timestamp_gen():
    while True:
        yield time.time()
```

### Medium Solutions

```python
# 1
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# 2
import csv
def csv_reader(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            yield row

# 3
def ngrams(seq, n=2):
    seq = list(seq)
    for i in range(len(seq) - n + 1):
        yield seq[i:i + n]

# 4
def cyclic_counter(max=10):
    while True:
        for i in range(max):
            yield i

# 5
def limit(gen, n):
    for i, val in enumerate(gen):
        if i >= n:
            break
        yield val

chain = limit((x**2 for x in range(100)), 5)
print(list(chain))  # [0, 1, 4, 9, 16]
```

### Hard Solutions

```python
# 1 — Data loader with shuffle, batch, prefetch
import random
from collections import deque

class DataLoader:
    def __init__(self, data, batch_size=32, shuffle=True, prefetch=2):
        self.data = list(data)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.prefetch = prefetch
        self._indices = list(range(len(self.data)))

    def __iter__(self):
        return self._generator()

    def _generator(self):
        buffer = deque()
        while True:
            if self.shuffle:
                random.shuffle(self._indices)
            for start in range(0, len(self.data), self.batch_size):
                batch_indices = self._indices[start:start + self.batch_size]
                batch = [self.data[i] for i in batch_indices]
                buffer.append(batch)
                if len(buffer) > self.prefetch:
                    yield buffer.popleft()
            if not buffer:
                break
        while buffer:
            yield buffer.popleft()

# 2 — Coroutine state machine
def state_machine():
    state = "idle"
    while True:
        event = yield state
        if state == "idle" and event == "start":
            state = "running"
        elif state == "running" and event == "pause":
            state = "paused"
        elif state == "running" and event == "stop":
            state = "idle"
        elif state == "paused" and event == "resume":
            state = "running"
        elif state == "paused" and event == "stop":
            state = "idle"

# sm = state_machine()
# next(sm)  # Prime
# print(sm.send("start"))  # running
# print(sm.send("pause"))  # paused
# print(sm.send("resume")) # running
# print(sm.send("stop"))   # idle

# 3 — Online mean and variance
def online_stats():
    """Generator that computes running mean and variance."""
    n = 0
    mean = 0.0
    m2 = 0.0  # Sum of squares of differences from current mean
    while True:
        value = yield (mean, m2 / n if n > 0 else 0.0)
        n += 1
        delta = value - mean
        mean += delta / n
        delta2 = value - mean
        m2 += delta * delta2

# stats = online_stats()
# next(stats)  # Prime
# print(stats.send(10))  # (10.0, 0.0)
# print(stats.send(20))  # (15.0, 25.0)
# print(stats.send(30))  # (20.0, 66.666...)
```

## Related Concepts

- PYT-016: Functions (return vs yield)
- PYT-024: Decorators
- PYT-018: Lambda Functions
- PYT-030: Iterators and Iterables

## Next Concepts

- PYT-050: Async/Await (coroutines evolved from generators)
- PYT-060: Context Managers
- PYT-070: Data Pipelines

## Summary

Generators are functions that use `yield` to produce sequences lazily. They maintain state between calls, making them memory-efficient for large datasets. Generator expressions are the generator equivalent of list comprehensions. The `send()` method enables two-way communication with generators. `yield from` delegates to sub-generators, enabling composable pipelines. In AI/ML, generators are the foundation of data loading, batch generation, and preprocessing pipelines.

## Key Takeaways

- Use `yield` instead of `return` to create a generator function.
- Generators produce values lazily — they compute on demand, not all at once.
- Generator expressions use `()` instead of `[]` and are memory-efficient.
- One-time use: a generator is exhausted after iteration.
- `gen.send(value)` sends a value back into the generator after `yield`.
- `yield from` delegates to another generator or iterable.
- Essential for ML data pipelines: batch generators, augmentation pipelines, cross-validation.
- Always prime a generator with `next()` before using `send()`.
