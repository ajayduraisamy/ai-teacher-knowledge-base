# Concept: itertools Module

## Concept ID

PYT-044

## Difficulty

Advanced

## Domain

Python

## Module

Advanced Python

## Learning Objectives

- Understand the `itertools` module as a toolkit for iterator-based operations
- Use infinite iterators: `count`, `cycle`, `repeat`
- Filter iterators: `compress`, `dropwhile`, `takewhile`, `filterfalse`
- Combine iterators: `chain`, `zip_longest`, `product`
- Partition iterators: `groupby`, `islice`
- Generate combinatorics: `permutations`, `combinations`, `combinations_with_replacement`
- Compute accumulated values with `accumulate`
- Apply itertools to AI/ML hyperparameter grid generation

## Prerequisites

- Iterators and the iterator protocol (PYT-038)
- Generator functions and expressions
- Lambda functions
- Basic understanding of combinatorics (permutations, combinations)

## Definition

The **`itertools`** module is a standard Python library that provides a collection of fast, memory-efficient tools for working with iterators. These functions implement various iterator building blocks inspired by constructs from functional programming languages (like APL, Haskell, and SML). All itertools functions produce iterators that compute lazily, consuming elements only as needed.

## Intuition

Think of `itertools` as a Swiss Army knife for iterators. Each tool performs one specific operation on an iterator stream — chaining streams, filtering elements, grouping consecutive elements, generating combinations. Because they are lazy and compose well, you can chain multiple itertools operations to build complex data processing pipelines with minimal memory usage.

## Why This Concept Matters

The `itertools` module is essential for writing memory-efficient Python code. Instead of creating intermediate lists that consume memory, itertools functions operate lazily, producing elements on demand. This is crucial for processing large datasets, infinite sequences, or streaming data. The combinatorial generators (`product`, `permutations`, `combinations`) are indispensable for testing, search, and optimization.

## Real World Examples

- Generating test cases with `product` for all combinations of parameters
- Processing log files with `islice` to skip headers
- Grouping sorted data with `groupby` for aggregate analysis
- Creating infinite data streams with `cycle` for simulation
- Paginating data with `islice` for API endpoints
- Running totals with `accumulate` for financial calculations

## AI/ML Relevance

The `itertools` module in AI/ML:
- `product` for generating hyperparameter grid search configurations
- `permutations` for enumerating sequence orders in RNN/transformer experiments
- `combinations` for feature selection experiments
- `cycle` for infinite data loading and batch cycling
- `accumulate` for computing running loss and metrics during training
- `chain` for concatenating multiple datasets or data sources

## Code Examples

### Example 1: Infinite iterators — count, cycle, repeat

```python
from itertools import count, cycle, repeat

# count: infinite arithmetic progression
for i, val in enumerate(count(10, 2)):
    print(val, end=' ')
    if i >= 4:
        break
print()

# cycle: infinitely repeat an iterable
counter = 0
for val in cycle('ABC'):
    print(val, end=' ')
    counter += 1
    if counter >= 7:
        break
print()

# repeat: repeat a single value
print(list(repeat('hello', 3)))

# Output:
# 10 12 14 16 18
# A B C A B C A
# ['hello', 'hello', 'hello']
```

### Example 2: Combining iterators — chain, zip_longest

```python
from itertools import chain, zip_longest

# chain: concatenate iterables
list1 = [1, 2, 3]
list2 = [4, 5, 6]
list3 = [7, 8]
print(list(chain(list1, list2, list3)))

# zip_longest: zip with fill value for uneven iterables
a = [1, 2, 3]
b = ['a', 'b']
print(list(zip_longest(a, b, fillvalue='*')))

# Output:
# [1, 2, 3, 4, 5, 6, 7, 8]
# [(1, 'a'), (2, 'b'), (3, '*')]
```

### Example 3: Filtering iterators — compress, dropwhile, takewhile, filterfalse

```python
from itertools import compress, dropwhile, takewhile, filterfalse

data = [1, 2, 3, 4, 5, 6]
selectors = [True, False, True, False, True, False]
print(list(compress(data, selectors)))

# dropwhile: drop elements while condition is true, then yield all
print(list(dropwhile(lambda x: x < 3, [1, 2, 3, 4, 1, 2])))

# takewhile: yield elements while condition is true, then stop
print(list(takewhile(lambda x: x < 3, [1, 2, 3, 4, 1, 2])))

# filterfalse: opposite of filter (keeps elements where predicate is False)
print(list(filterfalse(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6])))

# Output:
# [1, 3, 5]
# [3, 4, 1, 2]
# [1, 2]
# [1, 3, 5]
```

### Example 4: groupby for grouping consecutive elements

```python
from itertools import groupby

data = [('A', 1), ('A', 2), ('B', 3), ('B', 4), ('A', 5)]

# IMPORTANT: groupby only groups CONSECUTIVE elements
for key, group in groupby(data, key=lambda x: x[0]):
    items = list(group)
    print(f'{key}: {items}')

print('---')
# Sort first for proper grouping
sorted_data = sorted(data, key=lambda x: x[0])
for key, group in groupby(sorted_data, key=lambda x: x[0]):
    items = list(group)
    print(f'{key}: {items}')

# Output:
# A: [('A', 1), ('A', 2)]
# B: [('B', 3), ('B', 4)]
# A: [('A', 5)]
# ---
# A: [('A', 1), ('A', 2), ('A', 5)]
# B: [('B', 3), ('B', 4)]
```

### Example 5: islice — slicing iterators

```python
from itertools import islice

def generate_numbers():
    n = 0
    while True:
        yield n
        n += 1

gen = generate_numbers()
print(list(islice(gen, 5)))
print(list(islice(gen, 3, 8)))
print(list(islice(gen, 2, 10, 2)))

# Output:
# [0, 1, 2, 3, 4]
# [8, 9, 10, 11, 12]
# [16, 18, 20, 22, 24]
```

### Example 6: Combinatorics — product, permutations, combinations

```python
from itertools import product, permutations, combinations, combinations_with_replacement

# product: Cartesian product
print(list(product('AB', [1, 2])))

# permutations: r-length tuples, no repeated elements, order matters
print(list(permutations('ABC', 2)))

# combinations: r-length tuples, no repeated elements, order does NOT matter
print(list(combinations('ABC', 2)))

# combinations_with_replacement: with repeated elements
print(list(combinations_with_replacement('ABC', 2)))

# Output:
# [('A', 1), ('A', 2), ('B', 1), ('B', 2)]
# [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B')]
# [('A', 'B'), ('A', 'C'), ('B', 'C')]
# [('A', 'A'), ('A', 'B'), ('A', 'C'), ('B', 'B'), ('B', 'C'), ('C', 'C')]
```

### Example 7: accumulate for running totals

```python
from itertools import accumulate
import operator

numbers = [1, 2, 3, 4, 5]

# Default: running sum
print(list(accumulate(numbers)))

# Running product
print(list(accumulate(numbers, operator.mul)))

# Running maximum
print(list(accumulate(numbers, max)))

# Custom function
print(list(accumulate(numbers, lambda a, b: a * 2 + b)))

# Output:
# [1, 3, 6, 10, 15]
# [1, 2, 6, 24, 120]
# [1, 2, 3, 4, 5]
# [1, 4, 11, 26, 57]
```

### Example 8: AI/ML hyperparameter grid search

```python
from itertools import product

def grid_search(param_grid):
    """Generate all combinations of hyperparameters."""
    keys = param_grid.keys()
    values = param_grid.values()
    for combination in product(*values):
        yield dict(zip(keys, combination))

param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'batch_size': [16, 32, 64],
    'optimizer': ['sgd', 'adam'],
}

for i, params in enumerate(grid_search(param_grid)):
    print(f'Config {i+1}: {params}')
    if i >= 5:
        print('...')
        break

# Output:
# Config 1: {'learning_rate': 0.1, 'batch_size': 16, 'optimizer': 'sgd'}
# Config 2: {'learning_rate': 0.1, 'batch_size': 16, 'optimizer': 'adam'}
# Config 3: {'learning_rate': 0.1, 'batch_size': 32, 'optimizer': 'sgd'}
# Config 4: {'learning_rate': 0.1, 'batch_size': 32, 'optimizer': 'adam'}
# Config 5: {'learning_rate': 0.1, 'batch_size': 64, 'optimizer': 'sgd'}
# Config 6: {'learning_rate': 0.1, 'batch_size': 64, 'optimizer': 'adam'}
# ...
```

## Common Mistakes

1. Forgetting that `groupby` only groups consecutive elements — always sort first for semantic grouping
2. Exhausting an iterator passed to multiple itertools functions (iterators are single-use)
3. Converting large itertools results to lists unnecessarily — keeping them as iterators preserves memory
4. Confusing `permutations` (order matters) with `combinations` (order does not matter)
5. Using `product` with large input sets — results grow exponentially

## Interview Questions

### Beginner - 5

1. What is the `itertools` module used for?
2. What is the difference between `permutations` and `combinations`?
3. How does `chain` work?
4. What does `count(10, 2)` produce?
5. How is `takewhile` different from `filter`?

### Intermediate - 5

1. How does `groupby` work and what is its most important constraint?
2. How would you use `islice` to paginate an iterator?
3. What does `accumulate` do and how can you customize its behavior?
4. Compare `product` with nested `for` loops — when is each appropriate?
5. How would you implement a sliding window over an iterator using itertools?

### Advanced - 3

1. How would you implement `tee` (copy an iterator) without the itertools module?
2. Explain how `zip_longest` handles uneven iterables compared to `zip`.
3. Design an iterator pipeline that reads a large file, filters lines, transforms them, and groups results — all lazily using itertools.

## Practice Problems

### Easy - 5

1. Use `count` to generate the first 10 even numbers.
2. Use `cycle` to create a traffic light simulator (green, yellow, red).
3. Use `chain` to concatenate three lists into one iterator.
4. Use `product` to generate all pairs from two lists.
5. Use `accumulate` to compute running interest on a principal.

### Medium - 5

1. Use `groupby` to group a list of words by their first letter.
2. Use `combinations` to find all unique pairs from a team of 5 players.
3. Use `islice` to skip the first 10 lines of a file and read the next 20.
4. Use `compress` to filter a list based on a boolean mask.
5. Use `product` to generate all possible RGB color tuples (0-255).

### Hard - 3

1. Implement a sliding window iterator using `tee` and `islice`.
2. Build a lazy `batched` function (Python 3.12+ `itertools.batched`) using `islice`.
3. Design a memory-efficient `read_csv_chunks` generator using itertools that reads a CSV file in chunks.

## Solutions

### Easy 1

```python
from itertools import count, islice

evens = (n * 2 for n in count(1))
print(list(islice(evens, 10)))

# Output:
# [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
```

### Medium 1

```python
from itertools import groupby

words = ['apple', 'banana', 'avocado', 'blueberry', 'cherry', 'cranberry']
sorted_words = sorted(words, key=lambda w: w[0])

for letter, group in groupby(sorted_words, key=lambda w: w[0]):
    print(f'{letter}: {list(group)}')

# Output:
# a: ['apple', 'avocado']
# b: ['banana', 'blueberry']
# c: ['cherry', 'cranberry']
```

### Hard 1

```python
from itertools import tee, islice

def sliding_window(iterable, n):
    iterators = tee(iterable, n)
    for i, it in enumerate(iterators):
        for _ in range(i):
            next(it, None)
    return zip(*iterators)

data = [1, 2, 3, 4, 5]
for window in sliding_window(data, 3):
    print(window)

# Output:
# (1, 2, 3)
# (2, 3, 4)
# (3, 4, 5)
```

## Related Concepts

- Iterators and iterator protocol (PYT-038)
- Generator functions and expressions
- `functools` module (PYT-043)
- `collections` module (PYT-045)
- Lazy evaluation
- Functional programming patterns
- Memory-efficient data processing

## Next Concepts

- `collections` module (PYT-045)
- Regular expressions (PYT-046)
- Data processing pipelines
- Advanced combinatorics

## Summary

The `itertools` module is a powerful collection of iterator-based building blocks for functional-style programming in Python. It provides infinite iterators, filtering tools, combinators, grouping utilities, and combinatorial generators — all implemented lazily for maximum memory efficiency. Mastering `itertools` enables elegant solutions to complex data processing problems.

## Key Takeaways

- All itertools functions produce lazy iterators — they compute on demand
- `count`, `cycle`, `repeat` create infinite iterators (use with `islice` to limit)
- `chain` concatenates iterables; `zip_longest` handles uneven lengths
- `groupby` groups consecutive elements — sort first for proper grouping
- `product`, `permutations`, `combinations` generate combinatorial arrangements
- `accumulate` computes running totals with customizable binary functions
- `islice` provides lazy slicing for any iterator
- AI/ML: `product` is essential for hyperparameter grid search; `accumulate` for running metrics
