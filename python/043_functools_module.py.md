# Concept: functools Module

## Concept ID

PYT-043

## Difficulty

Advanced

## Domain

Python

## Module

Advanced Python

## Learning Objectives

- Understand the `functools` module and its role in functional programming
- Use `lru_cache` and `cache` to memoize expensive function calls
- Apply `wraps` to preserve metadata in decorated functions
- Use `reduce` for cumulative operations
- Implement single-dispatch generic functions with `singledispatch`
- Leverage `total_ordering` to complete comparison operators
- Convert comparison functions to key functions with `cmp_to_key`
- Understand `update_wrapper` for manual wrapper updates

## Prerequisites

- Functions and decorators
- Recursion and memoization concepts
- Comparison and ordering in Python (`__lt__`, `__gt__`, etc.)
- Type hints and dispatch (helpful but not required)

## Definition

The **`functools`** module is a standard Python library that provides higher-order functions and utilities for working with callable objects. It supports functional programming patterns by offering tools for memoization, function composition, method adaptation, and generic function dispatch. The module helps write cleaner, more efficient, and more maintainable code.

## Intuition

Think of `functools` as a toolbox for "function modification and enhancement." If functions are the building blocks of your program, `functools` gives you the tools to wrap them with caching (`lru_cache`), preserve their identity through decorators (`wraps`), create generic versions that work on different types (`singledispatch`), and process sequences cumulatively (`reduce`).

## Why This Concept Matters

The `functools` module solves common problems that arise in virtually every Python codebase. Memoization with `lru_cache` can dramatically speed up recursive or repetitive computations. `wraps` is essential for writing proper decorators that don't break introspection. `singledispatch` enables clean type-based dispatch without complex `if/elif` chains. These tools make code faster, cleaner, and more Pythonic.

## Real World Examples

- Caching database query results with `lru_cache`
- Writing proper decorators with `wraps` for API rate limiters
- Implementing type-dispatch for serialization functions
- Reducing a list of values with cumulative operations
- Creating comparable enum-like classes with `total_ordering`
- Handling legacy comparison functions with `cmp_to_key`

## AI/ML Relevance

The `functools` module in AI/ML applications:
- `lru_cache` for caching expensive feature computations and model inference results
- `cache` for lightweight memoization in data preprocessing
- `singledispatch` for type-specific data transformations in pipelines
- `reduce` for cumulative metrics computation across batches
- `wraps` for creating decorators that log or time model training functions

## Code Examples

### Example 1: lru_cache for memoization

```python
from functools import lru_cache

@lru_cache(maxsize=32)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(10))
print(fibonacci(35))

print(fibonacci.cache_info())

# Output:
# 55
# 9227465
# CacheInfo(hits=32, misses=36, maxsize=32, currsize=32)
```

### Example 2: cache (Python 3.9+)

```python
from functools import cache

@cache
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))
print(factorial(10))
print(factorial.cache_info())

# Output:
# 120
# 3628800
# CacheInfo(hits=10, misses=11, maxsize=None, currsize=11)
```

### Example 3: wraps for preserving metadata

```python
from functools import wraps

def log_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f'Calling {func.__name__} with {args}')
        return func(*args, **kwargs)
    return wrapper

@log_calls
def add(a, b):
    """Add two numbers together."""
    return a + b

print(add.__name__)
print(add.__doc__)
print(add(3, 4))

# Output:
# add
# Add two numbers together.
# Calling add with (3, 4)
# 7
```

### Example 4: reduce for cumulative operations

```python
from functools import reduce

numbers = [1, 2, 3, 4, 5]

product = reduce(lambda x, y: x * y, numbers)
print(f'Product: {product}')

# Custom reduce: find maximum
maximum = reduce(lambda a, b: a if a > b else b, numbers)
print(f'Maximum: {maximum}')

# Reduce with initial value
sum_from_10 = reduce(lambda x, y: x + y, numbers, 10)
print(f'Sum + 10: {sum_from_10}')

# Output:
# Product: 120
# Maximum: 5
# Sum + 10: 25
```

### Example 5: singledispatch for generic functions

```python
from functools import singledispatch

@singledispatch
def serialize(obj):
    return str(obj)

@serialize.register(int)
def _(obj):
    return f'int:{obj}'

@serialize.register(float)
def _(obj):
    return f'float:{obj:.2f}'

@serialize.register(list)
def _(obj):
    items = ', '.join(serialize(item) for item in obj)
    return f'list:[{items}]'

@serialize.register(dict)
def _(obj):
    items = ', '.join(f'{k}={serialize(v)}' for k, v in obj.items())
    return f'dict:{{{items}}}'

print(serialize(42))
print(serialize(3.14159))
print(serialize([1, 2, 3]))
print(serialize({'a': 1, 'b': 2}))

# Output:
# int:42
# float:3.14
# list:[int:1, int:2, int:3]
# dict:{a=int:1, b=int:2}
```

### Example 6: total_ordering for comparison methods

```python
from functools import total_ordering

@total_ordering
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __lt__(self, other):
        return self.age < other.age

    def __eq__(self, other):
        return self.age == other.age

    def __repr__(self):
        return f'{self.name}({self.age})'

people = [
    Person('Alice', 30),
    Person('Bob', 25),
    Person('Charlie', 35),
]

print(sorted(people))
print(Person('Alice', 30) > Person('Bob', 25))
print(Person('Alice', 30) >= Person('Charlie', 35))

# Output:
# [Bob(25), Alice(30), Charlie(35)]
# True
# False
```

### Example 7: cmp_to_key for legacy comparators

```python
from functools import cmp_to_key

def compare_by_length(a, b):
    if len(a) < len(b):
        return -1
    elif len(a) > len(b):
        return 1
    return 0

words = ['python', 'java', 'c', 'rust', 'javascript']
sorted_words = sorted(words, key=cmp_to_key(compare_by_length))
print(sorted_words)

# Output:
# ['c', 'java', 'rust', 'python', 'javascript']
```

### Example 8: update_wrapper

```python
from functools import update_wrapper

def make_decorator(extra_info):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f'Extra: {extra_info}')
            return func(*args, **kwargs)
        update_wrapper(wrapper, func)
        return wrapper
    return decorator

@make_decorator('important')
def greet(name):
    """Greet a person."""
    return f'Hello, {name}!'

print(greet.__name__)
print(greet.__doc__)
print(greet('Alice'))

# Output:
# greet
# Greet a person.
# Extra: important
# Hello, Alice!
```

## Common Mistakes

1. Using `lru_cache` on functions with mutable arguments (lists, dicts) — they are unhashable
2. Forgetting `wraps` in decorators, which breaks function metadata
3. Using `reduce` when a simple loop or built-in (`sum`, `any`, `all`) would be clearer
4. Not considering cache size limits with `lru_cache` for memory-heavy results
5. Using `singledispatch` when a simple `isinstance` check would suffice (over-engineering)
6. Forgetting that `total_ordering` adds performance overhead from extra comparison calls

## Interview Questions

### Beginner - 5

1. What is `functools.lru_cache` used for?
2. How does `functools.wraps` help when writing decorators?
3. What does `functools.reduce` do?
4. What is the purpose of `functools.total_ordering`?
5. How do you inspect the cache statistics of an `lru_cache`-decorated function?

### Intermediate - 5

1. What is the difference between `lru_cache` and `cache` (Python 3.9+)?
2. How does `singledispatch` work and when would you use it?
3. Compare `reduce` with `accumulate` from the `itertools` module.
4. What happens with `lru_cache` when the decorated function has keyword arguments?
5. How does `cmp_to_key` adapt legacy comparison functions to Python 3's key-based sorting?

### Advanced - 3

1. How would you implement a custom `lru_cache` with TTL (time-to-live) expiration?
2. Explain how `singledispatch` uses the type annotation system and how to register implementations for generic types.
3. How does `partial` from `functools` differ from "currying" in functional languages, and how would you implement true currying?

## Practice Problems

### Easy - 5

1. Use `lru_cache` to memoize a recursive factorial function.
2. Write a decorator with `wraps` that logs execution time.
3. Use `reduce` to concatenate a list of strings.
4. Create a `@total_ordering` class with only `__lt__` and `__eq__`.
5. Use `cmp_to_key` to sort strings by their last character.

### Medium - 5

1. Implement a `@singledispatch` function for pretty-printing different data types.
2. Use `lru_cache` with `maxsize=None` to cache a function that computes prime numbers.
3. Create a decorator factory using `wraps` that accepts a log level parameter.
4. Use `reduce` to implement a pipeline of functions (function composition).
5. Build a cache with both `lru_cache` and a timeout mechanism.

### Hard - 3

1. Implement a generic memoization decorator that works with unhashable arguments by converting them to hashable types.
2. Build a `singledispatch`-based serialization system that handles nested data structures with recursive dispatch.
3. Design and implement a `TTLCache` decorator using `functools` and `time` that evicts entries after a configurable timeout.

## Solutions

### Easy 1

```python
from functools import lru_cache

@lru_cache(maxsize=32)
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(10))
print(factorial.cache_info())

# Output:
# 3628800
# CacheInfo(hits=9, misses=11, maxsize=32, currsize=11)
```

### Medium 1

```python
from functools import singledispatch

@singledispatch
def pretty_print(value):
    return str(value)

@pretty_print.register(int)
def _(value):
    return f'INT: {value} (binary: {value:b})'

@pretty_print.register(float)
def _(value):
    return f'FLOAT: {value:.4f}'

@pretty_print.register(str)
def _(value):
    return f'STR: "{value}" (len={len(value)})'

@pretty_print.register(list)
def _(value):
    items = ', '.join(pretty_print(v) for v in value)
    return f'LIST: [{items}]'

print(pretty_print(42))
print(pretty_print(3.14159))
print(pretty_print('hello'))
print(pretty_print([1, 2.5, 'three']))

# Output:
# INT: 42 (binary: 101010)
# FLOAT: 3.1416
# STR: "hello" (len=5)
# LIST: [INT: 1 (binary: 1), FLOAT: 2.5000, STR: "three" (len=5)]
```

### Hard 1

```python
from functools import lru_cache

def memoize_with_hashable(func):
    cache = {}

    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper

@memoize_with_hashable
def expensive_op(data, factor=1):
    # Simulate expensive operation
    print(f'Computing for {data}...')
    return [x * factor for x in data]

result1 = expensive_op([1, 2, 3], 2)
result2 = expensive_op([1, 2, 3], 2)
print(result1)
print(result1 is result2)

# Output:
# Computing for [1, 2, 3]...
# [2, 4, 6]
# True
```

## Related Concepts

- Partial functions (PYT-042)
- `itertools` module (PYT-044)
- Decorators and closures
- Functional programming patterns
- Memoization and caching strategies
- Type dispatch and polymorphism

## Next Concepts

- `itertools` module (PYT-044)
- `collections` module (PYT-045)
- Advanced decorator patterns
- Performance optimization techniques

## Summary

The `functools` module is an essential part of Python's functional programming toolkit. It provides memoization (`lru_cache`, `cache`), function metadata preservation (`wraps`, `update_wrapper`), cumulative operations (`reduce`), generic dispatch (`singledispatch`), ordering utilities (`total_ordering`, `cmp_to_key`), and more. Mastering `functools` leads to code that is faster, cleaner, and more maintainable.

## Key Takeaways

- `lru_cache` and `cache` provide easy memoization for expensive functions
- `wraps` is essential for writing proper decorators that preserve function metadata
- `reduce` performs cumulative left-to-right operations on iterables
- `singledispatch` enables clean, type-based function dispatch
- `total_ordering` auto-completes comparison operators from just `__lt__` and `__eq__`
- `cmp_to_key` bridges the gap between legacy comparison functions and key-based sorting
- AI/ML: use `lru_cache` for caching feature computations and `singledispatch` for type-specific pipeline transformations
