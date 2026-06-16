# Concept: Iterators

## Concept ID

PYT-038

## Difficulty

Intermediate

## Domain

Python

## Module

Intermediate Python

## Learning Objectives

- Understand the iterator protocol: `__iter__` and `__next__`
- Use `iter()` and `next()` built-in functions
- Handle `StopIteration` correctly
- Distinguish between iterables and iterators
- Build custom iterator classes
- Understand how `for` loops work internally
- Recognize common iterator patterns and use cases

## Prerequisites

- Python loops and the `for` statement
- Class definition and methods
- Understanding of `__init__` and other magic methods

## Definition

An **iterator** is an object that implements the iterator protocol, consisting of two methods: `__iter__()` (returns the iterator object itself) and `__next__()` (returns the next value or raises `StopIteration`). An **iterable** is any object that can be passed to `iter()` to produce an iterator.

Iterators represent a stream of data — they produce values one at a time and are exhausted once consumed. This lazy evaluation model is fundamental to Python's memory efficiency and is used throughout the language.

## Intuition

Think of an iterator like a book reader. The book itself is the iterable — you can start reading it. The reader is the iterator — it remembers where you are and can give you the next page when asked. Once you reach the last page (`StopIteration`), the reader is done. You cannot re-read from the same exhausted iterator, but you can start a new reader on the same book.

## Why This Concept Matters

Iterators are the foundation of Python's iteration model. Every `for` loop, every list comprehension, every generator expression relies on the iterator protocol. Understanding iterators means understanding how Python processes sequences, how to create memory-efficient data pipelines, and how to build custom objects that work seamlessly with Python's iteration idioms.

## Real World Examples

- Reading large files line by line without loading the entire file into memory
- Database cursors that fetch rows one at a time
- Paginated API responses consumed incrementally
- Infinite sequences like Fibonacci numbers or prime generators
- Streaming data processing (log files, sensor data, network streams)

## AI/ML Relevance

Iterators are essential in AI/ML for:
- Dataset loading and batching — yielding batches of training data on demand
- Infinite data generators for simulation and augmentation
- Custom data pipelines that transform data lazily
- Streaming model predictions on large datasets
- Implementing custom sampling strategies

## Code Examples

### Example 1: Basic iterator protocol

```python
class CountUpTo:
    def __init__(self, max_val):
        self.max_val = max_val
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.max_val:
            raise StopIteration
        self.current += 1
        return self.current

counter = CountUpTo(5)
for num in counter:
    print(num, end=' ')
print()

# Output:
# 1 2 3 4 5
```

### Example 2: Iterable vs iterator — separate classes

```python
class Squares:
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return SquaresIterator(self.n)

class SquaresIterator:
    def __init__(self, n):
        self.n = n
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.n:
            raise StopIteration
        self.i += 1
        return (self.i - 1) ** 2

sq = Squares(4)
for val in sq:
    print(val, end=' ')
print()

# Multiple loops are possible because each creates a new iterator
for val in sq:
    print(val, end=' ')
print()

# Output:
# 0 1 4 9
# 0 1 4 9
```

### Example 3: iter() and next() built-in functions

```python
nums = [10, 20, 30]
it = iter(nums)

print(next(it))
print(next(it))
print(next(it))

try:
    next(it)
except StopIteration:
    print('Iterator exhausted')

# Output:
# 10
# 20
# 30
# Iterator exhausted
```

### Example 4: Custom iterator for Fibonacci sequence

```python
class Fibonacci:
    def __init__(self, max_count):
        self.max_count = max_count
        self.count = 0
        self.a, self.b = 0, 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.max_count:
            raise StopIteration
        self.count += 1
        self.a, self.b = self.b, self.a + self.b
        return self.a

fib = Fibonacci(8)
print(list(fib))

# Output:
# [1, 1, 2, 3, 5, 8, 13, 21]
```

### Example 5: Infinite iterator (with break)

```python
class InfiniteCounter:
    def __init__(self, start=0):
        self.current = start

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        return self.current

counter = InfiniteCounter()
for i, val in enumerate(counter):
    print(val, end=' ')
    if i >= 5:
        break
print()

# Output:
# 1 2 3 4 5 6
```

### Example 6: Iterator for a custom container

```python
class Playlist:
    def __init__(self):
        self.songs = []

    def add_song(self, song):
        self.songs.append(song)

    def __iter__(self):
        return PlaylistIterator(self.songs)

class PlaylistIterator:
    def __init__(self, songs):
        self.songs = songs
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.songs):
            raise StopIteration
        song = self.songs[self.index]
        self.index += 1
        return f'Now playing: {song}'

my_playlist = Playlist()
my_playlist.add_song('Song A')
my_playlist.add_song('Song B')
my_playlist.add_song('Song C')

for s in my_playlist:
    print(s)

# Output:
# Now playing: Song A
# Now playing: Song B
# Now playing: Song C
```

### Example 7: for loop internals — what Python actually does

```python
# This for loop:
# for x in [1, 2, 3]:
#     print(x)
#
# Python translates to:

_iterable = [1, 2, 3]
_iterator = iter(_iterable)
while True:
    try:
        x = next(_iterator)
    except StopIteration:
        break
    print(x)

# Output:
# 1
# 2
# 3
```

## Common Mistakes

1. Confusing iterables and iterators — not every iterable is an iterator (lists are iterable but not iterators)
2. Reusing an exhausted iterator — iterators can be consumed only once; create a new one for each loop
3. Forgetting to raise `StopIteration` in custom `__next__` methods
4. Implementing `__iter__` but not `__next__` on a single class when it should be its own iterator
5. Returning `None` instead of raising `StopIteration` to signal the end

## Interview Questions

### Beginner - 5

1. What is the difference between an iterable and an iterator?
2. How do the `iter()` and `next()` built-in functions work?
3. What does `StopIteration` mean?
4. How does a `for` loop use iterators internally?
5. Are all iterables also iterators? Give an example.

### Intermediate - 5

1. Implement a custom iterator class that iterates over even numbers up to a given limit.
2. What happens if you call `next()` on an already consumed iterator?
3. How can you make a custom class work with `for` loops?
4. What is the purpose of `__iter__` returning `self` versus returning a separate iterator object?
5. How do generator functions relate to iterators?

### Advanced - 3

1. How would you implement a bidirectional iterator that supports both forward and backward traversal?
2. Explain how Python's `map`, `filter`, and `zip` functions use the iterator protocol.
3. How does the `itertools` module leverage iterators for memory-efficient composition?

## Practice Problems

### Easy - 5

1. Create a simple iterator that counts from 1 to 10.
2. Use `iter()` and `next()` to manually iterate over a string.
3. Write a custom iterator for a list of words that returns them in uppercase.
4. Create an iterator that returns squares of numbers from 1 to N.
5. Use a `for` loop to iterate over a custom object and print each value.

### Medium - 5

1. Implement a `Range` class that mimics the built-in `range()` using the iterator protocol.
2. Create an iterator that yields only the odd numbers from a given iterable (lazy filter).
3. Build a `CyclicIterator` that cycles through a list indefinitely.
4. Implement a `PaginatedData` iterator that simulates fetching data from an API page by page.
5. Create an iterator chain — an iterator that takes multiple iterables and yields from each in sequence.

### Hard - 3

1. Implement a `PeekableIterator` that wraps any iterator and allows peeking at the next value without consuming it.
2. Build a tree traversal iterator (in-order, pre-order, post-order) for a binary tree without recursion.
3. Design and implement a lazy `CSVParser` that reads a CSV file row by row using a custom iterator with type conversion.

## Solutions

### Easy 1

```python
class CountToTen:
    def __init__(self):
        self.n = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.n >= 10:
            raise StopIteration
        self.n += 1
        return self.n

for num in CountToTen():
    print(num, end=' ')
print()

# Output:
# 1 2 3 4 5 6 7 8 9 10
```

### Medium 1

```python
class Range:
    def __init__(self, start, stop=None, step=1):
        if stop is None:
            self.start, self.stop = 0, start
        else:
            self.start, self.stop = start, stop
        self.step = step

    def __iter__(self):
        return RangeIterator(self.start, self.stop, self.step)

class RangeIterator:
    def __init__(self, start, stop, step):
        self.current = start
        self.stop = stop
        self.step = step

    def __iter__(self):
        return self

    def __next__(self):
        if (self.step > 0 and self.current >= self.stop) or \
           (self.step < 0 and self.current <= self.stop):
            raise StopIteration
        val = self.current
        self.current += self.step
        return val

print(list(Range(5)))
print(list(Range(2, 8)))

# Output:
# [0, 1, 2, 3, 4]
# [2, 3, 4, 5, 6, 7]
```

### Hard 1

```python
class PeekableIterator:
    def __init__(self, iterable):
        self._iterator = iter(iterable)
        self._next_item = None
        self._has_peeked = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._has_peeked:
            self._has_peeked = False
            return self._next_item
        return next(self._iterator)

    def peek(self):
        if not self._has_peeked:
            try:
                self._next_item = next(self._iterator)
                self._has_peeked = True
            except StopIteration:
                return None
        return self._next_item

it = PeekableIterator([1, 2, 3])
print(it.peek())
print(next(it))
print(next(it))

# Output:
# 1
# 1
# 2
```

## Related Concepts

- Generator functions (`yield`)
- Generator expressions
- The `itertools` module (PYT-044)
- The `for` loop implementation
- Lazy evaluation
- `map()`, `filter()`, `zip()` built-in functions

## Next Concepts

- Context managers (PYT-039)
- Closures (PYT-041)
- Generator-based coroutines
- Asynchronous iterators (`__aiter__`, `__anext__`)

## Summary

Iterators are the mechanism by which Python implements lazy, memory-efficient iteration. The iterator protocol — `__iter__` and `__next__` — is simple yet powerful, enabling custom objects to participate in `for` loops, comprehensions, and all Python iteration idioms. Understanding iterators deepens your grasp of Python's design philosophy and opens the door to creating efficient data pipelines.

## Key Takeaways

- Every iterator is an iterable, but not every iterable is an iterator
- The iterator protocol consists of `__iter__()` and `__next__()`
- `StopIteration` signals the end of iteration
- `for` loops work by calling `iter()` on an iterable and then `next()` in a loop
- Iterators are single-use and stateful — they are exhausted once consumed
- Generators are the easiest way to create iterators in practice
- Custom iterators enable lazy, memory-efficient processing of large or infinite sequences
