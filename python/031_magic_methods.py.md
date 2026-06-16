# Concept: Magic Methods

## Concept ID

PYT-031

## Difficulty

Intermediate

## Domain

Python

## Module

Object-Oriented Programming

## Learning Objectives

- Understand what magic methods are and how they integrate with Python's data model
- Implement `__str__`, `__repr__`, `__len__`, `__getitem__`, `__setitem__`
- Create iterable classes with `__iter__` and `__next__`
- Make objects callable with `__call__`
- Use context managers via `__enter__` and `__exit__`
- Implement equality, hashing, and arithmetic with `__eq__`, `__hash__`, `__add__`, `__mul__`
- Control truthiness with `__bool__`

## Prerequisites

- Classes and objects (PYT-026)
- Encapsulation (PYT-030)
- Python data types and operators

## Definition

Magic methods (also called **dunder methods** because of the **d**ouble **under**score prefix and suffix) are special methods that begin and end with `__`. They allow custom classes to integrate seamlessly with Python's built-in operations, functions, and syntax. When you write `len(obj)`, `str(obj)`, `obj[key]`, `obj + other`, or `for x in obj`, Python internally calls the corresponding magic method.

```python
class Showcase:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Showcase({self.value!r})"

s = Showcase(42)
print(repr(s))
```

```
# Output:
Showcase(42)
```

## Intuition

Magic methods are like plugging your custom object into Python's electrical sockets. When you define `__len__`, your object becomes compatible with `len()` — it fits the "length" socket. When you define `__getitem__`, it fits the "indexing" socket. Each magic method is a standard interface that Python recognizes, allowing your custom class to behave like a built-in type.

## Why This Concept Matters

Magic methods are what make Python's "duck typing" and "everything is an object" philosophy work seamlessly. They let your custom classes participate in Python's rich ecosystem of built-in functions, operators, and language constructs. Libraries like NumPy, pandas, and PyTorch use magic methods extensively to create objects that feel like native data types.

## Real World Examples

1. **NumPy arrays:** support `+`, `*`, `[]`, `len()` through magic methods.
2. **Django QuerySets:** support iteration (`__iter__`), indexing (`__getitem__`), length (`__len__`), and boolean evaluation (`__bool__`).
3. **Pathlib paths:** `__truediv__` overloads `/` for path joining (`Path("a") / "b"`).
4. **Context managers:** `open()` returns a file object usable with `with` (`__enter__`, `__exit__`).
5. **Callable classes:** Keras layers use `__call__` so `layer(x)` runs the forward pass.

## AI/ML Relevance

- PyTorch `nn.Module` uses `__call__` to run `forward()` — `model(x)` not `model.forward(x)`.
- Custom datasets use `__len__` and `__getitem__` to work with `DataLoader`.
- Loss functions use `__call__` to be invoked as `criterion(pred, target)`.
- Parameter groups and configs use `__eq__` and `__hash__` for deduplication.
- `__enter__`/`__exit__` enable clean resource management for GPU training sessions.

## Code Examples

### Example 1: `__str__` and `__repr__`

```python
class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages

    def __repr__(self):
        return f"Book({self.title!r}, {self.author!r}, {self.pages})"

    def __str__(self):
        return f"'{self.title}' — {self.author}"

b = Book("1984", "George Orwell", 328)
print(repr(b))
print(str(b))
print(b)
```

```
# Output:
Book('1984', 'George Orwell', 328)
'1984' — George Orwell
'1984' — George Orwell
```

### Example 2: `__len__` and `__getitem__`

```python
class Playlist:
    def __init__(self, songs):
        self._songs = songs

    def __len__(self):
        return len(self._songs)

    def __getitem__(self, index):
        if isinstance(index, str):
            # Find by song name
            for song in self._songs:
                if song.lower() == index.lower():
                    return song
            raise KeyError(f"Song '{index}' not found")
        return self._songs[index]

    def __setitem__(self, index, song):
        self._songs[index] = song

pl = Playlist(["Bohemian Rhapsody", "Stairway to Heaven", "Hotel California"])
print(len(pl))
print(pl[1])
print(pl["hotel california"])
pl[0] = "Imagine"
print(pl[0])
```

```
# Output:
3
Stairway to Heaven
Hotel California
Imagine
```

### Example 3: `__iter__` and `__next__`

```python
class Countdown:
    def __init__(self, start):
        self.start = start

    def __iter__(self):
        self.n = self.start
        return self

    def __next__(self):
        if self.n <= 0:
            raise StopIteration
        self.n -= 1
        return self.n + 1

for num in Countdown(5):
    print(num, end=" ")
print()
```

```
# Output:
5 4 3 2 1
```

### Example 4: `__call__`

```python
class Multiplier:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return x * self.factor

double = Multiplier(2)
triple = Multiplier(3)

print(double(5))
print(triple(5))
print(double(double(5)))
```

```
# Output:
10
15
20
```

### Example 5: `__enter__` and `__exit__`

```python
class ManagedFile:
    def __init__(self, filename, mode="r"):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        print(f"Opening {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Closing {self.filename}")
        self.file.close()
        # Return False to propagate exceptions, True to suppress
        return False

with ManagedFile("test.txt", "w") as f:
    f.write("Hello, world!")

import os
os.remove("test.txt")
```

```
# Output:
Opening test.txt
Closing test.txt
```

### Example 6: `__eq__`, `__hash__`, `__add__`, `__mul__`

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

    def __eq__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __add__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Vector(self.x * scalar, self.y * scalar)

    def __bool__(self):
        return self.x != 0 or self.y != 0

v1 = Vector(2, 3)
v2 = Vector(2, 3)
v3 = Vector(4, 5)

print(v1 == v2)
print(v1 == v3)
print(hash(v1))
print(v1 + v3)
print(v1 * 3)
print(bool(Vector(0, 0)))
print(bool(v1))
```

```
# Output:
True
False
-1685978611
Vector(6, 8)
Vector(6, 9)
False
True
```

## Common Mistakes

1. **Returning the wrong type from magic methods** — `__add__` should return a new instance of the class, not mutate `self`.
2. **Forgetting `NotImplemented`** — when an operation isn't supported for the other type, return `NotImplemented` (not raise `TypeError`), so Python can try the reflected operation.
3. **Defining `__eq__` without `__hash__`** — this makes the class unhashable (can't be used in sets or as dict keys). Either set `__hash__ = None` or define both.
4. **Modifying `self` in `__getitem__`** — `__getitem__` should be read-only; don't mutate internal state.
5. **Infinite recursion in `__setattr__`** — using `self.x = value` inside `__setattr__` calls `__setattr__` again. Use `self.__dict__[key] = value`.
6. **Ignoring `exc_type` in `__exit__`** — if you want to handle exceptions, check `exc_type`; otherwise return `False` to propagate them.
7. **Making `__next__` reset the iteration** — if you call `next()` after `StopIteration`, it should raise `StopIteration` again, not restart.
8. **Comparing unrelated types in `__eq__`** — always check `isinstance` and return `NotImplemented` for incompatible types.

## Interview Questions

### Beginner

1. What are magic methods in Python?
2. What is the difference between `__str__` and `__repr__`?
3. Which magic method makes an object callable?
4. Which magic method is called by `len(obj)`?
5. What magic methods are needed to make an object iterable?

### Intermediate

1. How do `__enter__` and `__exit__` implement the context manager protocol?
2. What should `__add__` return? Should it modify `self`?
3. Why must `__eq__` and `__hash__` be defined together?
4. What is the difference between `__getitem__` and `__setitem__`?
5. How does `__bool__` affect `if obj:` and `not obj`?

### Advanced

1. Explain the reflected operators (e.g., `__radd__`) and when Python calls them.
2. How do `__slots__`, `__getattr__`, and `__setattr__` interact with magic methods?
3. Implement a class that delegates `__getattr__` to a wrapped object.

## Practice Problems

### Easy

1. Create a `Person` class with `__str__` and `__repr__` that show name and age.
2. Create a `Deck` class with `__len__` (52) and `__getitem__` returning card names.
3. Implement `__bool__` for a `Battery` class that returns True if charge > 0.
4. Create a `Multiplier` class that uses `__call__` to multiply by a stored factor.
5. Add `__eq__` to a `Point` class with x, y.

### Medium

1. Create a `Range` class that works like `range()` using `__iter__` and `__next__`.
2. Implement a `Matrix` class with `__add__`, `__mul__` (scalar and matrix).
3. Build a `ConfigManager` context manager with `__enter__`/`__exit__` that loads and saves config.
4. Create a `FrozenDict` class using `__getitem__`, `__len__`, `__iter__`, but raising on `__setitem__`.
5. Implement a `Logger` class using `__call__` that prepends timestamps to messages.

### Hard

1. Implement a `SparseVector` class where most elements are zero, using `__getitem__`, `__setitem__`, `__add__`, `__mul__`.
2. Build a `Memoize` descriptor using `__call__` that caches function return values.
3. Implement a full `Polynomial` class with `__add__`, `__sub__`, `__mul__`, `__call__`, `__eq__`, and derivative.

## Solutions

### Easy — Solution 1

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __repr__(self):
        return f"Person({self.name!r}, {self.age})"

    def __str__(self):
        return f"{self.name} ({self.age} years old)"

p = Person("Alice", 30)
print(repr(p))
print(str(p))
print(p)
```

```
# Output:
Person('Alice', 30)
Alice (30 years old)
Alice (30 years old)
```

### Medium — Solution 1

```python
class Range:
    def __init__(self, start, stop=None, step=1):
        if stop is None:
            self.start = 0
            self.stop = start
        else:
            self.start = start
            self.stop = stop
        self.step = step

    def __iter__(self):
        self.current = self.start
        return self

    def __next__(self):
        if (self.step > 0 and self.current >= self.stop) or \
           (self.step < 0 and self.current <= self.stop):
            raise StopIteration
        result = self.current
        self.current += self.step
        return result

    def __len__(self):
        if self.step > 0 and self.stop > self.start:
            return max(0, (self.stop - self.start - 1) // self.step + 1)
        return 0

for i in Range(1, 6):
    print(i, end=" ")
print()
print(len(Range(1, 6)))
print(list(Range(0, 10, 2)))
```

```
# Output:
1 2 3 4 5
5
[0, 2, 4, 6, 8]
```

### Hard — Solution 1

```python
class SparseVector:
    def __init__(self, length, data=None):
        self.length = length
        self._data = {} if data is None else {k: v for k, v in data.items() if v != 0}

    def __getitem__(self, idx):
        if not 0 <= idx < self.length:
            raise IndexError("Index out of range")
        return self._data.get(idx, 0)

    def __setitem__(self, idx, value):
        if not 0 <= idx < self.length:
            raise IndexError("Index out of range")
        if value == 0:
            self._data.pop(idx, None)
        else:
            self._data[idx] = value

    def __add__(self, other):
        if not isinstance(other, SparseVector) or self.length != other.length:
            return NotImplemented
        result = SparseVector(self.length, dict(self._data))
        for idx, val in other._data.items():
            result[idx] = result[idx] + val
        return result

    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        new_data = {k: v * scalar for k, v in self._data.items()}
        return SparseVector(self.length, new_data)

    def __repr__(self):
        items = sorted(self._data.items())
        return f"SparseVector({self.length}, {items})"

    def __eq__(self, other):
        if not isinstance(other, SparseVector) or self.length != other.length:
            return NotImplemented
        return self._data == other._data

v1 = SparseVector(10, {0: 1, 3: 5, 7: 2})
v2 = SparseVector(10, {3: 3, 7: 2, 9: 4})
print(v1[3])
print(v1[1])
v3 = v1 + v2
print(v3)
v4 = v1 * 2
print(v4)
```

```
# Output:
5
0
SparseVector(10, [(0, 1), (3, 8), (7, 4), (9, 4)])
SparseVector(10, [(0, 2), (3, 10), (7, 4)])
```

## Related Concepts

- Classes and Objects
- Operators and Expressions
- Iterators and Generators

## Next Concepts

- Abstract Classes
- Data Classes
- Composition vs Inheritance

## Summary

Magic methods (dunder methods) allow custom classes to integrate with Python's built-in operations and syntax. `__str__` and `__repr__` control string representation. `__len__` and `__getitem__` make objects collection-like. `__iter__`/`__next__` enable iteration. `__call__` makes objects callable. `__enter__`/`__exit__` support context managers. `__eq__`/`__hash__` handle equality and hashing. `__add__`/`__mul__` overload arithmetic. `__bool__` controls truthiness. Properly implementing magic methods makes custom classes feel like first-class Python citizens.

## Key Takeaways

- Magic methods let custom objects use Python's built-in operations
- `__repr__` should be unambiguous; `__str__` should be readable
- `__eq__` and `__hash__` must be consistent — define both or neither
- `__call__` makes objects callable like functions
- `__enter__`/`__exit__` enable the `with` statement
- Return `NotImplemented` for unsupported types in arithmetic magic methods
- Magic methods are the key to writing Pythonic, idiomatic classes
