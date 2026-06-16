# Concept: Encapsulation

## Concept ID

PYT-030

## Difficulty

Intermediate

## Domain

Python

## Module

Object-Oriented Programming

## Learning Objectives

- Understand the concept of encapsulation and data hiding
- Use `_var` as a protected convention
- Use `__var` for name mangling
- Implement `@property` for controlled attribute access
- Write getter and setter methods
- Create computed properties

## Prerequisites

- Classes and objects (PYT-026)
- Method types (PYT-027)
- Understanding of `self`

## Definition

Encapsulation is the bundling of data (attributes) and methods that operate on that data within a single unit (a class), while restricting direct access to some of the object's internal state. In Python, encapsulation is achieved through:

1. **Convention:** `_single_leading_underscore` means "protected" — internal use, not part of the public API.
2. **Name mangling:** `__double_leading_underscore` triggers name mangling to `_ClassName__var`, discouraging accidental access from outside.
3. **Properties:** `@property` provides controlled access to attributes with getters, setters, and deleters.

## Intuition

Think of a vending machine. You interact with it through a clear interface — insert coins, press buttons, collect your snack. You don't directly access the internal coin mechanism, the product chute, or the cooling system. The machine encapsulates its internal complexity behind a simple interface. Encapsulation in classes works the same way: you expose a clean public API and hide internal implementation details.

## Why This Concept Matters

Encapsulation is a fundamental OOP principle that protects the integrity of an object's state. It prevents external code from putting the object into an invalid state, makes refactoring safer (internal changes don't break external code), and reduces coupling between components. In large codebases and libraries, encapsulation is essential for maintaining backward compatibility.

## Real World Examples

1. **Bank accounts:** `balance` is read-only via a property; `deposit()` and `withdraw()` validate before modifying.
2. **Temperature sensors:** internal raw sensor data is stored as `_raw_voltage`; the public `temperature` property converts to Celsius.
3. **Configuration classes:** internal `_config` dict is private; public properties validate keys and types.
4. **Database models:** internal `_connection` is private; `save()`, `query()` are public.
5. **ML model training:** internal `_weights`, `_gradients` are private; `fit()`, `predict()` are public.

## AI/ML Relevance

- Model weights are often stored as `_parameters` with controlled access via properties
- Hyperparameter objects use properties to validate ranges (e.g., `learning_rate` must be > 0)
- Dataset classes keep internal `_data` and `_labels` protected, exposing only `__getitem__` and `__len__`
- Training state (epoch, optimizer state) is encapsulated in `_state` dict
- Gradient norms and intermediate activations are often computed properties

## Code Examples

### Example 1: Protected Convention with `_var`

```python
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self._salary = salary  # Protected — internal use

    def get_annual_bonus(self):
        return self._salary * 0.1

e = Employee("Alice", 80000)
print(e.name)
print(e.get_annual_bonus())
# Accessing _salary works but is discouraged
print(e._salary)
```

```
# Output:
Alice
8000.0
80000
```

### Example 2: Name Mangling with `__var`

```python
class Counter:
    def __init__(self):
        self.__count = 0

    def increment(self):
        self.__count += 1

    def get_count(self):
        return self.__count

c = Counter()
c.increment()
c.increment()
print(c.get_count())
# print(c.__count)  # AttributeError
print(c._Counter__count)  # Name mangling — accessible but discouraged
```

```
# Output:
2
2
```

### Example 3: Properties for Controlled Access

```python
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius

    @property
    def celsius(self):
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero")
        self._celsius = value

    @property
    def fahrenheit(self):
        return (self._celsius * 9 / 5) + 32

    @fahrenheit.setter
    def fahrenheit(self, value):
        self._celsius = (value - 32) * 5 / 9

t = Temperature(100)
print(t.celsius)
print(t.fahrenheit)

t.celsius = 0
print(t.fahrenheit)

t.fahrenheit = 212
print(t.celsius)

try:
    t.celsius = -300
except ValueError as e:
    print(e)
```

```
# Output:
100
212.0
32.0
100.0
Temperature below absolute zero
```

### Example 4: Read-Only and Computed Properties

```python
import math

class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value <= 0:
            raise ValueError("Radius must be positive")
        self._radius = value

    @property
    def diameter(self):
        return self._radius * 2

    @property
    def area(self):
        return math.pi * self._radius ** 2

    @property
    def circumference(self):
        return 2 * math.pi * self._radius

c = Circle(5)
print(c.radius, c.diameter, c.area, c.circumference)
c.radius = 10
print(c.radius, c.diameter, c.area, c.circumference)
```

```
# Output:
5 10 78.53981633974483 31.41592653589793
10 20 314.1592653589793 62.83185307179586
```

### Example 5: AI/ML — Model with Encapsulated Weights

```python
import math

class LinearRegression:
    def __init__(self, learning_rate=0.01):
        self._learning_rate = learning_rate
        self._weight = 0.0
        self._bias = 0.0
        self._fitted = False

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        if value <= 0:
            raise ValueError("Learning rate must be positive")
        self._learning_rate = value

    @property
    def weight(self):
        if not self._fitted:
            raise RuntimeError("Model not fitted yet")
        return self._weight

    @property
    def bias(self):
        if not self._fitted:
            raise RuntimeError("Model not fitted yet")
        return self._bias

    def fit(self, X, y):
        n = len(X)
        mean_x = sum(X) / n
        mean_y = sum(y) / n
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(X, y))
        den = sum((xi - mean_x) ** 2 for xi in X)
        self._weight = num / den
        self._bias = mean_y - self._weight * mean_x
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("Model not fitted yet")
        return [self._weight * x + self._bias for x in X]

model = LinearRegression(0.01)
model.fit([1, 2, 3, 4], [2, 4, 6, 8])
print(f"Weight: {model.weight:.1f}, Bias: {model.bias:.1f}")
print(model.predict([5, 6]))
```

```
# Output:
Weight: 2.0, Bias: 0.0
[10.0, 12.0]
```

### Example 6: Property Deleter and Caching

```python
class Processor:
    def __init__(self):
        self._data = None
        self._cache = {}

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self._cache = {}  # Invalidate cache

    @property
    def processed_data(self):
        if "processed" not in self._cache:
            print("Computing processed_data...")
            if self._data is None:
                raise ValueError("No data set")
            self._cache["processed"] = [x * 2 for x in self._data]
        return self._cache["processed"]

    @processed_data.deleter
    def processed_data(self):
        self._cache.pop("processed", None)

p = Processor()
p.data = [1, 2, 3]
print(p.processed_data)
print(p.processed_data)
del p.processed_data
print(p.processed_data)
```

```
# Output:
Computing processed_data...
[2, 4, 6]
[2, 4, 6]
Computing processed_data...
[2, 4, 6]
```

## Common Mistakes

1. **Thinking `__var` makes attributes truly private** — Python has no true private. Name mangling just renames to `_ClassName__var`; it's still accessible.
2. **Using getter/setter methods when properties are more Pythonic** — Java-style `get_name()` / `set_name()` is un-Pythonic; use `@property`.
3. **Defining a property setter before the getter** — the `@property` decorator must be defined first, then `@name.setter`.
4. **Creating infinite recursion in a property** — accessing `self.x` inside the `x` property's getter calls the property again. Use `self._x` internally.
5. **Forgetting to invalidate caches in setters** — if you cache computed values, the setter must clear the cache.
6. **Over-encapsulating simple data** — not every attribute needs a property. Use properties only when you need validation or computation.
7. **Exposing mutable internal objects** — returning `self._list` directly lets callers modify it. Return a copy or use a read-only view.
8. **Using double underscore for everything** — `__` is for avoiding name conflicts in subclasses, not general privacy. Use `_` for internal attributes.

## Interview Questions

### Beginner

1. What is encapsulation?
2. What is the difference between `_var` and `__var`?
3. How do you define a read-only property in Python?
4. What does name mangling do?
5. How is `@property.setter` different from a regular method?

### Intermediate

1. Why doesn't Python have true private attributes?
2. When should you use `@property` vs a regular getter method?
3. How does name mangling help with inheritance?
4. What is the property deleter used for?
5. How would you implement a lazy (cached) property?

### Advanced

1. Explain the descriptor protocol and how `@property` is implemented.
2. How can you prevent attribute deletion on an object?
3. Implement a read-only descriptor that works on both classes and instances.

## Practice Problems

### Easy

1. Create a `Person` class with a private `__age` attribute, a getter `age`, and a setter that validates 0-150.
2. Create a `Password` class that stores `__password` and only exposes a `check(guess)` method.
3. Add a read-only `full_name` property to a `Person` class with `first_name` and `last_name`.
4. Create a `BankAccount` class with a read-only `balance` property and `deposit()`/`withdraw()` methods.
5. Implement a `Counter` class with a private `__count` that can only be incremented via `increment()`.

### Medium

1. Create a `Rectangle` class where `width` and `height` are properties that validate positivity, and `area` is a computed property.
2. Implement a `Config` class that stores settings in `__config` dict with typed property accessors.
3. Create a `LazyProperty` descriptor that computes a value once then caches it.
4. Build a `Temperature` class with `celsius` and `fahrenheit` properties that keep each other in sync.
5. Implement a `User` class with a password setter that hashes the password and a getter that raises `AttributeError`.

### Hard

1. Implement a `ValidatedProperty` descriptor that supports type checking, range validation, and custom validators.
2. Create a proxy object that encapsulates another object and logs all attribute access.
3. Implement an `Immutable` class decorator that prevents attribute modification after `__init__`.

## Solutions

### Easy — Solution 1

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.__age = age

    @property
    def age(self):
        return self.__age

    @age.setter
    def age(self, value):
        if not (0 <= value <= 150):
            raise ValueError("Age must be between 0 and 150")
        self.__age = value

p = Person("Alice", 30)
print(p.age)
p.age = 35
print(p.age)
try:
    p.age = 200
except ValueError as e:
    print(e)
```

```
# Output:
30
35
Age must be between 0 and 150
```

### Medium — Solution 1

```python
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        if value <= 0:
            raise ValueError("Width must be positive")
        self._width = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        if value <= 0:
            raise ValueError("Height must be positive")
        self._height = value

    @property
    def area(self):
        return self._width * self._height

    @property
    def perimeter(self):
        return 2 * (self._width + self._height)

r = Rectangle(5, 3)
print(r.area, r.perimeter)
r.width = 10
print(r.area, r.perimeter)
```

```
# Output:
15 16
30 26
```

### Hard — Solution 1

```python
class ValidatedProperty:
    def __init__(self, type_check=None, min_val=None, max_val=None, validator=None):
        self.type_check = type_check
        self.min_val = min_val
        self.max_val = max_val
        self.validator = validator
        self.data = {}

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.private_name)

    def __set__(self, obj, value):
        if self.type_check and not isinstance(value, self.type_check):
            raise TypeError(f"Expected {self.type_check}, got {type(value)}")
        if self.min_val is not None and value < self.min_val:
            raise ValueError(f"Value must be >= {self.min_val}")
        if self.max_val is not None and value > self.max_val:
            raise ValueError(f"Value must be <= {self.max_val}")
        if self.validator and not self.validator(value):
            raise ValueError("Validation failed")
        obj.__dict__[self.private_name] = value

class Product:
    name = ValidatedProperty(type_check=str)
    price = ValidatedProperty(type_check=(int, float), min_val=0)
    quantity = ValidatedProperty(type_check=int, min_val=0, max_val=10000)

    def __init__(self, name, price, quantity):
        self.name = name
        self.price = price
        self.quantity = quantity

p = Product("Laptop", 999.99, 50)
print(p.name, p.price, p.quantity)
try:
    p.price = -5  # Raises ValueError
except ValueError as e:
    print(e)
```

```
# Output:
Laptop 999.99 50
Value must be >= 0
```

## Related Concepts

- Classes and Objects
- Properties and Descriptors
- Method Types

## Next Concepts

- Magic Methods
- Abstract Classes
- Data Classes

## Summary

Encapsulation bundles data and behavior while hiding internal state. Python uses `_var` (protected convention), `__var` (name mangling for subclass safety), and `@property` (controlled getter/setter access). Properties can be read-only, read-write, or computed. While Python doesn't enforce true privacy, conventions and properties encourage disciplined data access that makes code safer, more maintainable, and easier to refactor.

## Key Takeaways

- `_var` is a convention for internal/protected attributes
- `__var` triggers name mangling to avoid subclass name conflicts
- `@property` creates controlled, computed, or read-only attributes
- Properties support getter, setter, and deleter
- Encapsulation protects object invariants and allows safe refactoring
- Use simplicity: not every attribute needs a property; add one when you need validation or computation
