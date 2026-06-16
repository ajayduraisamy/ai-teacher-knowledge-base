# Concept: Polymorphism

## Concept ID

PYT-029

## Difficulty

Intermediate

## Domain

Python

## Module

Object-Oriented Programming

## Learning Objectives

- Understand method overriding as a form of polymorphism
- Apply duck typing to write flexible, interface-agnostic code
- Use `Protocol` classes for structural subtyping
- Implement operator overloading via magic methods
- Distinguish between compile-time and run-time polymorphism

## Prerequisites

- Classes and objects (PYT-026)
- Inheritance (PYT-028)
- Magic methods basics

## Definition

Polymorphism means "many forms." In Python, it is the ability of different object types to respond to the same method call in their own way. Python achieves polymorphism through:

1. **Method overriding** — subclasses provide specific implementations of parent methods.
2. **Duck typing** — "if it walks like a duck and quacks like a duck, it's a duck." Objects are judged by what methods they have, not their type.
3. **Protocol classes** — formal structural typing with `typing.Protocol`.
4. **Operator overloading** — magic methods like `__add__`, `__mul__` define how operators work on custom classes.

## Intuition

Imagine a universal remote with a "play" button. When you point it at a DVD player, it plays a movie. Point it at a music streamer, it plays a song. Point it at a game console, it starts a game. The remote doesn't care what device it's talking to — it just calls "play," and each device responds appropriately. That's polymorphism: the same interface, different implementations.

## Why This Concept Matters

Polymorphism enables you to write generic, reusable code that works with any object that supports the required interface. This is the foundation of dependency injection, plugin architectures, strategy patterns, and frameworks. In Python's dynamic nature, polymorphism is everywhere — `len()` works on strings, lists, tuples, dicts, and custom objects that define `__len__`.

## Real World Examples

1. **File-like objects:** `open()` returns a file object; `io.StringIO` and `io.BytesIO` provide the same interface. Code that reads from a file works with any of them.
2. **Streaming APIs:** Different video codecs implement the same `encode()` / `decode()` interface.
3. **Database backends:** Django's ORM works with PostgreSQL, MySQL, SQLite using the same query interface.
4. **Logging handlers:** `FileHandler`, `StreamHandler`, `SMTPHandler` all implement `emit()`.
5. **Plugin systems:** Each plugin implements a standard interface (e.g., `run(data)`), and the main program calls them uniformly.

## AI/ML Relevance

- Scikit-learn estimators share a common API: `fit(X, y)`, `predict(X)`, `score(X, y)`. You can swap models without changing the surrounding code.
- PyTorch's `nn.Module` subclasses all implement `forward()`. The trainer calls `model(x)` uniformly.
- Feature extractors, transformers, and vectorizers implement `fit_transform()` / `transform()`.
- Custom loss functions implement `__call__(y_pred, y_true)`.
- Data loaders all support iteration via `__iter__` and `__next__` or `__getitem__`.

## Code Examples

### Example 1: Method Overriding (Classic Polymorphism)

```python
class Animal:
    def make_sound(self):
        return "..."

class Dog(Animal):
    def make_sound(self):
        return "Woof!"

class Cat(Animal):
    def make_sound(self):
        return "Meow!"

class Cow(Animal):
    def make_sound(self):
        return "Moo!"

def animal_chorus(animals):
    for animal in animals:
        print(animal.make_sound())

animals = [Dog(), Cat(), Cow(), Animal()]
animal_chorus(animals)
```

```
# Output:
Woof!
Meow!
Moo!
...
```

### Example 2: Duck Typing

```python
class Duck:
    def quack(self):
        return "Quack!"
    def swim(self):
        return "Swimming"

class Person:
    def quack(self):
        return "I'm pretending to be a duck!"
    def swim(self):
        return "Doing laps"

def make_it_quack(thing):
    print(thing.quack())

make_it_quack(Duck())
make_it_quack(Person())

# No inheritance needed — just the method
```

```
# Output:
Quack!
I'm pretending to be a duck!
```

### Example 3: Protocol Classes (Structural Subtyping)

```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> str:
        ...

class Circle:
    def draw(self) -> str:
        return "Drawing a circle"

class Square:
    def draw(self) -> str:
        return "Drawing a square"

class Triangle:
    def render(self) -> str:
        return "Rendering a triangle"

def render_all(objects):
    for obj in objects:
        print(obj.draw())

render_all([Circle(), Square()])
# Triangle doesn't implement draw(), so type checker would flag it
```

```
# Output:
Drawing a circle
Drawing a square
```

### Example 4: Operator Overloading with `__add__` and `__mul__`

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Vector(self.x * scalar, self.y * scalar)

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(2, 3)
v2 = Vector(4, 5)
print(v1 + v2)
print(v1 * 3)
```

```
# Output:
Vector(6, 8)
Vector(6, 9)
```

### Example 5: AI/ML — Unified Model API

```python
class LinearModel:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def fit(self, X, y):
        # Simplified training
        self.w = 1.5
        self.b = 0.5
        return self

    def predict(self, X):
        return [self.w * x + self.b for x in X]

class TreeModel:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth

    def fit(self, X, y):
        # Simplified training
        self.rules = ["if x > 0.5 then 1 else 0"]
        return self

    def predict(self, X):
        return [1 if x > 0.5 else 0 for x in X]

def evaluate_model(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    return model.predict(X_test)

X_train = [1, 2, 3, 4]
y_train = [2, 3.5, 5, 6.5]
X_test = [3, 5]

linear = LinearModel(1.0, 0.0)
tree = TreeModel()

print(evaluate_model(linear, X_train, y_train, X_test))
print(evaluate_model(tree, X_train, y_train, X_test))
```

```
# Output:
[5.0, 8.0]
[1, 1]
```

### Example 6: Polymorphism with `__len__` and `__getitem__`

```python
class MyList:
    def __init__(self, items):
        self._items = items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]

class MyDict:
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return self._data[key]

def show_length_and_first(obj):
    print(f"Length: {len(obj)}")
    print(f"First: {obj[0] if len(obj) > 0 else 'N/A'}")

show_length_and_first(MyList([10, 20, 30]))
show_length_and_first(MyDict({"a": 1, "b": 2}))
```

```
# Output:
Length: 3
First: 10
Length: 2
First: a
```

## Common Mistakes

1. **Confusing polymorphism with inheritance** — polymorphism is about interface, not necessarily inheritance. Duck typing doesn't require a shared parent.
2. **Forgetting `NotImplemented` return** — when a magic method doesn't support the other type, return `NotImplemented`, not `TypeError` (Python will try the reflected operation).
3. **Breaking the Liskov substitution principle** — a subclass that weakens preconditions or strengthens postconditions breaks polymorphism.
4. **Overloading by type checking within a method** — instead of `if isinstance(...)`, use multiple dispatch or accept any object that responds to the method.
5. **Assuming Protocol classes enforce at runtime** — `Protocol` is for static type checking only; it does not enforce anything at runtime.
6. **Mutating objects during `__add__`** — `__add__` should return a new object, not modify `self`.
7. **Not implementing `__hash__` when `__eq__` is overridden** — this breaks dictionary key and set usage.

## Interview Questions

### Beginner

1. What is polymorphism?
2. What is duck typing in Python?
3. How does method overriding achieve polymorphism?
4. What operator does `__add__` overload?
5. Can two unrelated classes be used polymorphically in Python?

### Intermediate

1. Explain the Liskov substitution principle.
2. What is the difference between method overloading and method overriding?
3. How does `typing.Protocol` enable structural subtyping?
4. What is the difference between `NotImplemented` and `TypeError` in magic methods?
5. How does Python decide which implementation of a method to call?

### Advanced

1. How does multiple dispatch (functools.singledispatch) relate to polymorphism?
2. Implement a visitor pattern that leverages polymorphism.
3. How does the MRO affect polymorphic method resolution in diamond inheritance?

## Practice Problems

### Easy

1. Create `Square`, `Circle`, `Triangle` classes each with `area()` method. Write a function `total_area(shapes)` that sums them.
2. Implement duck typing with `start()`, `stop()` methods in `Car`, `Bike`, `Plane` classes.
3. Create two unrelated classes that both implement `play()` and pass them to a `start_playing()` function.
4. Overload `__sub__` for a `Point` class to subtract two points.
5. Write a function that accepts any object with a `save(filename)` method.

### Medium

1. Implement a `Shape` Protocol and create `Circle`, `Square`, `Rectangle` that satisfy it.
2. Create a `FlexibleEncoder` that can encode any object with a `to_dict()` method (duck typing).
3. Implement `__radd__` and `__iadd__` for a custom `Counter` class.
4. Build a plugin system using Protocol: each plugin has `process(data)` and `name`.
5. Use `functools.singledispatch` to implement a `serialize()` function for different types.

### Hard

1. Implement a full numeric tower: `Complex`, `Real`, `Rational`, `Integer` classes with operator overloading.
2. Build a simple dependency injection container that resolves interfaces to implementations using Protocol.
3. Implement a generic `Pipeline` class where each stage is any callable object with `__call__`.

## Solutions

### Easy — Solution 1

```python
import math

class Square:
    def __init__(self, side):
        self.side = side
    def area(self):
        return self.side ** 2

class Circle:
    def __init__(self, radius):
        self.radius = radius
    def area(self):
        return math.pi * self.radius ** 2

class Triangle:
    def __init__(self, base, height):
        self.base = base
        self.height = height
    def area(self):
        return 0.5 * self.base * self.height

def total_area(shapes):
    return sum(s.area() for s in shapes)

shapes = [Square(4), Circle(3), Triangle(6, 8)]
print(total_area(shapes))
```

```
# Output:
70.27433388230814
```

### Medium — Solution 1

```python
from typing import Protocol

class Shape(Protocol):
    def area(self) -> float:
        ...
    def perimeter(self) -> float:
        ...

class Circle:
    def __init__(self, radius):
        self.radius = radius
    def area(self):
        return 3.14159 * self.radius ** 2
    def perimeter(self):
        return 2 * 3.14159 * self.radius

class Square:
    def __init__(self, side):
        self.side = side
    def area(self):
        return self.side ** 2
    def perimeter(self):
        return 4 * self.side

def print_shape_info(s: Shape):
    print(f"Area: {s.area():.2f}, Perimeter: {s.perimeter():.2f}")

print_shape_info(Circle(5))
print_shape_info(Square(4))
```

```
# Output:
Area: 78.54, Perimeter: 31.42
Area: 16.00, Perimeter: 16.00
```

### Hard — Solution 1

```python
from functools import singledispatch

@singledispatch
def serialize(obj):
    raise TypeError(f"Unsupported type: {type(obj)}")

@serialize.register(int)
def _(obj):
    return {"type": "int", "value": obj}

@serialize.register(str)
def _(obj):
    return {"type": "str", "value": obj, "length": len(obj)}

@serialize.register(list)
def _(obj):
    return {"type": "list", "items": [serialize(item) for item in obj], "count": len(obj)}

@serialize.register(dict)
def _(obj):
    return {"type": "dict", "keys": list(obj.keys()), "values": [serialize(v) for v in obj.values()]}

print(serialize(42))
print(serialize("hello"))
print(serialize([1, "a", [2, 3]]))
```

```
# Output:
{'type': 'int', 'value': 42}
{'type': 'str', 'value': 'hello', 'length': 5}
{'type': 'list', 'items': [{'type': 'int', 'value': 1}, {'type': 'str', 'value': 'a', 'length': 1}, {'type': 'list', 'items': [{'type': 'int', 'value': 2}, {'type': 'int', 'value': 3}], 'count': 2}], 'count': 3}
```

## Related Concepts

- Inheritance
- Magic Methods
- Abstract Classes

## Next Concepts

- Encapsulation
- Abstract Classes
- Multiple Inheritance

## Summary

Polymorphism means "many forms" — the ability of different object types to respond to the same method call appropriately. Python achieves it through method overriding, duck typing, Protocol classes, and operator overloading. Duck typing judges objects by their behavior (what methods they have) rather than their type. Protocol classes add static type safety to duck typing. Operator overloading via magic methods allows custom classes to work with Python's operators.

## Key Takeaways

- Polymorphism lets you write code that works with objects of different types through a common interface
- Duck typing: "if it walks like a duck and quacks like a duck, it's a duck"
- `typing.Protocol` provides formal structural subtyping for static type checkers
- Magic methods like `__add__` overload operators for custom classes
- Returning `NotImplemented` lets Python try the reflected operator
- Scikit-learn's unified `fit/predict` API is a classic real-world example
