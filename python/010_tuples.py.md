# Concept: Tuples

## Concept ID

PYT-010

## Difficulty

BEGINNER

## Prerequisites

- PYT-002: Variables and Data Types
- PYT-005: Operators and Expressions
- PYT-009: Lists

## Domain

Python

## Module

Python Basics

## Learning Objectives

- Create tuples using parentheses `()` and the `tuple()` constructor
- Understand and explain tuple immutability and its implications
- Use tuple packing and unpacking, including with `*` (star expressions)
- Access tuple elements via indexing and slicing
- Apply the two tuple methods: `count()` and `index()`
- Work with `namedtuple` from the `collections` module for self-documenting code
- Decide when to use a tuple instead of a list
- Write the correct syntax for single-element tuples

## Definition

A tuple is an ordered, immutable (unchangeable) collection of elements enclosed in parentheses `()`, with elements separated by commas. Unlike lists, tuples cannot be modified after creation — no appending, removing, or reassigning elements. Tuples can hold elements of any data type and support indexing, slicing, and iteration just like lists. They are often used for fixed collections of related values, such as coordinates, database records, or function return values.

## Intuition

Think of a tuple as a sealed envelope containing several items. You can look inside, count the items, find something by its position, or even take a picture (copy) — but you cannot change what is inside the envelope. This immutability is a promise: the data will not change, which makes tuples safe to use as dictionary keys, set elements, or return values from functions that should not be altered accidentally.

## Why This Concept Matters

Tuples are lightweight, memory-efficient, and hashable (if all elements are hashable). They are the default for returning multiple values from a function. They signal to other developers that the data is constant — a strong hint that the collection should not be modified. Named tuples add readability by giving field names to tuple elements. Understanding the trade-offs between lists and tuples is a mark of a mature Python developer.

## Real World Examples

1. **GPS coordinates**: A latitude/longitude pair is naturally a tuple `(40.7128, -74.0060)` — it should not change.
2. **Database records**: A single row from a relational database is often represented as a tuple of column values.
3. **Function return values**: `divmod(a, b)` returns `(quotient, remainder)` as a tuple.
4. **Dictionary keys**: Tuples of immutable objects can serve as compound dictionary keys (e.g., mapping `(city, year)` to population).
5. **Configuration constants**: Fixed settings like colour RGB values `(255, 0, 0)` for red are stored as tuples.

## AI/ML Relevance

- **Data points**: Individual rows in a dataset are frequently represented as tuples before conversion to NumPy arrays.
- **Hyperparameter tuples**: A combination of hyperparameters like `(learning_rate, batch_size, num_layers)` is passed as a tuple to grid search.
- **Shape tuples**: Tensor shapes in frameworks like PyTorch and TensorFlow are tuples, e.g., `(batch_size, channels, height, width)`.
- **Token-label pairs**: In NLP, a token and its label can be a tuple `("Paris", "LOC")`.
- **Evaluation metrics**: Multiple metrics returned from `evaluate()` are often packed into a tuple: `(loss, accuracy, f1_score)`.

## Code Examples

### Example 1: Creating tuples

Tuples can be created with or without parentheses, though parentheses are preferred for clarity. The comma is the essential element.

```python
# With parentheses
point = (3, 4)
print(point)
# Output: (3, 4)

# Without parentheses (tuple packing)
colors = "red", "green", "blue"
print(colors)
# Output: ('red', 'green', 'blue')

# Using the tuple() constructor
numbers = tuple([1, 2, 3, 4])
print(numbers)
# Output: (1, 2, 3, 4)

# Empty tuple
empty = ()
print(type(empty))
# Output: <class 'tuple'>

# Single-element tuple (REQUIRES trailing comma!)
single = (42,)
print(type(single), single)
# Output: <class 'tuple'> (42,)

# Without comma — it is NOT a tuple
not_a_tuple = (42)
print(type(not_a_tuple))
# Output: <class 'int'>
```

### Example 2: Indexing and slicing

Tuples support the same indexing and slicing syntax as lists.

```python
t = (10, 20, 30, 40, 50)

print(t[0])       # First element
# Output: 10
print(t[-1])      # Last element
# Output: 50
print(t[1:4])     # Slice from index 1 to 3
# Output: (20, 30, 40)
print(t[::-1])    # Reversed tuple
# Output: (50, 40, 30, 20, 10)

# Iteration
for val in t:
    print(val, end=" ")
# Output: 10 20 30 40 50
```

### Example 3: Tuple immutability

Once created, a tuple cannot be changed. Attempting to modify one raises a `TypeError`.

```python
t = (1, 2, 3)

# t[0] = 99  # TypeError: 'tuple' object does not support item assignment

# However, if a tuple contains a mutable object (like a list),
# the mutable object itself CAN be modified.
t_with_list = (1, [2, 3], 4)
t_with_list[1].append(99)
print(t_with_list)
# Output: (1, [2, 3, 99], 4)

# You still cannot replace the list with another list though:
# t_with_list[1] = [5, 6]  # TypeError
```

### Example 4: Tuple unpacking

Unpacking assigns each element of a tuple to a variable in one statement. Star expressions (`*`) capture multiple elements into a list.

```python
# Basic unpacking
point = (3, 4)
x, y = point
print(f"x={x}, y={y}")
# Output: x=3, y=4

# Swapping variables using tuple packing/unpacking
a, b = 5, 10
a, b = b, a
print(f"a={a}, b={b}")
# Output: a=10, b=5

# Star expression (Python 3+)
first, *middle, last = (1, 2, 3, 4, 5)
print(first, middle, last)
# Output: 1 [2, 3, 4] 5

# Ignoring values with underscore
name, _, score = ("Alice", "Engineering", 95)
print(name, score)
# Output: Alice 95
```

### Example 5: Tuple methods — `count()` and `index()`

Tuples have only two built-in methods (versus lists which have many). This is a consequence of immutability — there is no need for mutation methods.

```python
t = (1, 2, 2, 3, 2, 4, 5, 2)

# count — number of occurrences
print(t.count(2))
# Output: 4
print(t.count(99))
# Output: 0

# index — first index of a value (raises ValueError if not found)
print(t.index(3))
# Output: 3
print(t.index(2))
# Output: 1
```

### Example 6: Named tuples

`collections.namedtuple` creates tuple subclasses with named fields, combining the immutability of tuples with the readability of objects.

```python
from collections import namedtuple

# Define a named tuple type
Point = namedtuple("Point", ["x", "y"])

# Create instances
p1 = Point(3, 4)
p2 = Point(x=10, y=20)

# Access by name or by index
print(p1.x, p1[0])
# Output: 3 3
print(p2.y, p2[1])
# Output: 20 20

# Unpacking works
x, y = p1
print(x, y)
# Output: 3 4

# Immutability
# p1.x = 99  # AttributeError: can't set attribute

# Useful for database-like records
Student = namedtuple("Student", ["name", "age", "grade"])
s = Student("Alice", 20, "A")
print(f"{s.name} (age {s.age}): {s.grade}")
# Output: Alice (age 20): A
```

### Example 7: When to use tuples vs. lists

Tuples are preferable when the data:
- Should not change (immutability as a contract).
- Will be used as dictionary keys or set elements (requires hashability).
- Represents a fixed structure (e.g., coordinates, RGB values, database rows).
- Benefits from memory efficiency (tuples are slightly smaller than lists).

```python
# Good for tuples
rgb = (255, 128, 0)           # Colour — fixed structure
location = (51.5, -0.12)      # GPS — should not change
key = ("Smith", "John", 1990) # Dictionary key — needs to be hashable

# Good for lists
scores = [85, 92, 78]         # Will be updated
names = []                    # Will be populated
grades = [1, 2, 3, 4, 5]      # Might be sorted or filtered
```

## Common Mistakes

1. **Forgetting the trailing comma for a single-element tuple** — `(5)` is an integer, not a tuple. Write `(5,)`.
2. **Trying to modify a tuple after creation** — This causes a `TypeError`. Convert to a list if modification is needed, then convert back.
3. **Assuming parentheses are required** — `a = 1, 2, 3` creates a tuple, but readability suffers without parentheses.
4. **Using a mutable object inside a tuple and being surprised it can change** — The tuple holds a reference to the mutable object; the reference cannot change, but the object itself can.
5. **Confusing `namedtuple` with regular classes** — `namedtuple` fields are immutable and there is no `__dict__`. Use `dataclasses` or regular classes if you need mutable fields.
6. **Thinking tuples are completely immutable** — If a tuple contains a list, the list can be modified (though the reference in the tuple stays the same).
7. **Forgetting that `sort()` does not exist on tuples** — Use `sorted(t)` which returns a new list, or convert to a list first.

## Interview Questions

### Beginner

1. **Q:** How do you create a tuple?  
   **A:** `(1, 2, 3)` or `tuple([1, 2, 3])`.

2. **Q:** What happens if you try to change a tuple element?  
   **A:** A `TypeError` is raised because tuples are immutable.

3. **Q:** What is tuple unpacking?  
   **A:** Assigning each element of a tuple to a separate variable: `x, y = (3, 4)`.

4. **Q:** How do you create a tuple with one element?  
   **A:** `(42,)` — the trailing comma is essential.

5. **Q:** What two methods does a tuple have?  
   **A:** `count()` and `index()`.

### Intermediate

1. **Q:** What is the advantage of `namedtuple` over a regular tuple?  
   **A:** Named fields make the code self-documenting and allow access by name, improving readability while retaining tuple immutability.

2. **Q:** Can a tuple be used as a dictionary key?  
   **A:** Yes, if and only if all elements of the tuple are hashable (immutable). A tuple containing a list cannot be a key.

3. **Q:** Explain the difference in memory usage between a tuple and a list.  
   **A:** Tuples are smaller because they pre-allocate exactly enough space and are stored in one contiguous block. Lists overallocate to allow efficient appends.

4. **Q:** How does the `*` operator work with tuples?  
   **A:** `*` repeats the tuple: `(1, 2) * 3` → `(1, 2, 1, 2, 1, 2)`.

5. **Q:** What is the output of `divmod(17, 5)`?  
   **A:** `(3, 2)` — a tuple containing the quotient and remainder.

### Advanced

1. **Q:** How would you write a function that returns multiple values without using a tuple?  
   **A:** You cannot — Python's multiple return values are always packed into a tuple internally: `return a, b` is syntactic sugar for `return (a, b)`.

2. **Q:** Explain how `*args` in function definitions relates to tuple packing/unpacking.  
   **A:** `*args` collects extra positional arguments into a tuple. When calling a function, `*iterable` unpacks an iterable into positional arguments.

3. **Q:** Implement a simple `namedtuple`-like class without using `collections.namedtuple`.  
   **A:**  
   ```python
   class Point:
       __slots__ = ("x", "y")
       def __init__(self, x, y):
           self.x = x
           self.y = y
       def __repr__(self):
           return f"Point(x={self.x}, y={self.y})"
       def __iter__(self):
           yield self.x
           yield self.y
   ```
   This is immutable if `__slots__` is used, but the fields can still be set after construction. For full immutability, use `namedtuple` directly.

## Practice Problems

### Easy

1. **Swap Variables** — Swap the values of two variables using tuple packing/unpacking (no temporary variable).
2. **First and Last** — Given a tuple, return a new tuple with only the first and last elements.
3. **Count Element** — Count how many times a specific value appears in a tuple.
4. **Sum of Tuple** — Compute the sum of all numbers in a tuple.
5. **Reverse Tuple** — Reverse a tuple and return the reversed tuple.

### Medium

1. **Find Index** — Given a value, find its first and last index in a tuple. Return `None` for any that are not found.
2. **Tuple to Dictionary** — Given a tuple of `(key, value)` pairs, convert it to a dictionary.
3. **Rotate Tuple** — Rotate a tuple by `k` positions to the right (return a new tuple).
4. **Named Tuple for Student** — Define a `Student` namedtuple with fields `name`, `age`, `gpa`. Create a few instances and sort them by GPA.
5. **Unique Elements** — Given a tuple, return a tuple of unique elements in the order they first appear.

### Hard

1. **Nested Tuple Flatten** — Flatten a nested tuple of arbitrary depth into a single-level tuple (e.g., `(1, (2, (3, 4)))` → `(1, 2, 3, 4)`).
2. **Matrix Transpose** — Given a tuple of tuples representing a matrix, return its transpose as a tuple of tuples.
3. **Memoised Fibonacci** — Use a tuple as a cache key for a memoised Fibonacci function that takes `(n, modulus)` as arguments.

## Solutions

### Easy

```python
# 1. Swap Variables
a, b = 5, 10
a, b = b, a
print(a, b)
# Output: 10 5

# 2. First and Last
t = (10, 20, 30, 40, 50)
result = (t[0], t[-1])
print(result)
# Output: (10, 50)

# 3. Count Element
t = (1, 2, 2, 3, 2, 4)
print(t.count(2))
# Output: 3

# 4. Sum of Tuple
t = (1, 2, 3, 4, 5)
print(sum(t))
# Output: 15

# 5. Reverse Tuple
t = (1, 2, 3, 4)
print(t[::-1])
# Output: (4, 3, 2, 1)
```

### Medium

```python
# 1. Find Index
t = (1, 2, 3, 2, 4, 2)
val = 2
first = t.index(val) if val in t else None
last = len(t) - 1 - t[::-1].index(val) if val in t else None
print(first, last)
# Output: 1 5

# 2. Tuple to Dictionary
pairs = (("a", 1), ("b", 2), ("c", 3))
d = dict(pairs)
print(d)
# Output: {'a': 1, 'b': 2, 'c': 3}

# 3. Rotate Tuple
def rotate(t, k):
    if not t:
        return t
    k = k % len(t)
    return t[-k:] + t[:-k]

print(rotate((1, 2, 3, 4, 5), 2))
# Output: (4, 5, 1, 2, 3)

# 4. Named Tuple for Student
from collections import namedtuple
Student = namedtuple("Student", ["name", "age", "gpa"])
students = [
    Student("Alice", 20, 3.8),
    Student("Bob", 19, 3.5),
    Student("Charlie", 22, 3.9),
]
sorted_students = sorted(students, key=lambda s: s.gpa, reverse=True)
print([s.name for s in sorted_students])
# Output: ['Charlie', 'Alice', 'Bob']

# 5. Unique Elements
def unique_tuple(t):
    seen = set()
    result = []
    for item in t:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return tuple(result)

print(unique_tuple((1, 2, 2, 3, 1, 4)))
# Output: (1, 2, 3, 4)
```

### Hard

```python
# 1. Nested Tuple Flatten
def flatten_tuple(t):
    result = []
    for item in t:
        if isinstance(item, tuple):
            result.extend(flatten_tuple(item))
        else:
            result.append(item)
    return tuple(result)

nested = (1, (2, (3, 4)), 5)
print(flatten_tuple(nested))
# Output: (1, 2, 3, 4, 5)

# 2. Matrix Transpose
matrix = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
transposed = tuple(zip(*matrix))
print(transposed)
# Output: ((1, 4, 7), (2, 5, 8), (3, 6, 9))

# 3. Memoised Fibonacci with tuple cache
cache = {}
def fib_mod(n, mod):
    key = (n, mod)
    if key in cache:
        return cache[key]
    if n <= 1:
        return n % mod
    result = (fib_mod(n - 1, mod) + fib_mod(n - 2, mod)) % mod
    cache[key] = result
    return result

print(fib_mod(10, 100))
# Output: 55
print(fib_mod(100, 1000))
# Output: 875
```

## Related Concepts

- **Lists** — Mutable counterpart to tuples; same indexing/slicing but supports modification.
- **Strings** — Immutable sequences of characters; share indexing/slicing with tuples.
- **Dictionary Keys** — Tuples are often used as compound keys because they are hashable.
- **Packing/Unpacking** — Extends beyond tuples to lists, generators, and function arguments.

## Next Concepts

- **Sets** — Unordered, mutable, unique-element collections.
- **Dictionaries** — Key-value mappings; tuples frequently serve as keys.
- **Dataclasses** — Modern alternative to `namedtuple` with mutable fields and default values.
- **Generators** — Functions that `yield` multiple values, often unpacked like tuples.

## Summary

Tuples are immutable, ordered sequences ideal for fixed collections of related data. They support indexing, slicing, unpacking, and two methods (`count` and `index`). `namedtuple` adds field names for readability. Tuples are hashable (when contents are hashable), making them valid dictionary keys. Choose tuples over lists when immutability is desired, when the data represents a fixed structure, or when memory efficiency matters.

## Key Takeaways

- Tuples are immutable — once created, they cannot be changed.
- Single-element tuples require a trailing comma: `(42,)`.
- Tuples support indexing, slicing, and iteration just like lists.
- Two methods only: `count()` and `index()`.
- Tuple unpacking enables elegant multi-assignment, variable swapping, and handling of multiple return values.
- `namedtuple` from `collections` creates readable, self-documenting tuples with named fields.
- Use tuples for fixed data, dictionary keys, and multiple return values from functions.
- Tuples are more memory-efficient than lists for fixed-size collections.
