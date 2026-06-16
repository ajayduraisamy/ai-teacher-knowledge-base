# Concept: Descriptors

## Concept ID

PYT-040

## Difficulty

Advanced

## Domain

Python

## Module

Advanced Python

## Learning Objectives

- Understand the descriptor protocol: `__get__`, `__set__`, `__delete__`
- Distinguish between data and non-data descriptors
- Implement custom descriptors for validation and computed attributes
- Understand how `property()` works as a descriptor
- Explain how `@classmethod` and `@staticmethod` are implemented via descriptors
- Use `__set_name__` for automatic attribute naming
- Build reusable descriptor classes

## Prerequisites

- Class definition and the `__init__` method
- The `property()` built-in and `@property` decorator
- Understanding of method resolution order (MRO)
- Basic metaclass concepts (helpful but not required)

## Definition

A **descriptor** is any Python object that implements at least one of `__get__()`, `__set__()`, or `__delete__()` methods. When a descriptor's attribute is accessed on an instance, Python intercepts the lookup and delegates to the descriptor methods. Descriptors are the underlying mechanism behind properties, class methods, static methods, and `__slots__`.

Descriptors control attribute access at a fundamental level, enabling reusable attribute behavior across classes.

## Intuition

Think of a descriptor as a "smart attribute" that runs code every time you get, set, or delete it. Normal attributes just store and return a value. A descriptor can validate inputs, transform outputs, log access, compute values lazily, or enforce constraints — all while looking like a regular attribute access.

Properties are a special case of descriptors: they control access to a single attribute. Custom descriptors generalize this to create reusable behaviors that can be applied to many attributes across many classes.

## Why This Concept Matters

Descriptors are one of Python's most powerful metaprogramming features. Understanding descriptors demystifies how Python works internally — how `@property` works, how methods receive `self`, how `@classmethod` and `@staticmethod` behave. Descriptors enable the creation of reusable attribute behaviors like type validation, range checking, lazy loading, and ORM field definitions. Frameworks like Django, SQLAlchemy, and Pydantic rely heavily on descriptors.

## Real World Examples

- `@property` for computed or validated attributes
- Django model fields (CharField, IntegerField) as descriptors
- SQLAlchemy column descriptors
- Pydantic field validators
- Lazy loading of expensive resources
- Type-checked attributes in dataclasses
- Logged or traced attribute access for debugging

## AI/ML Relevance

Descriptors in AI/ML contexts:
- Implementing hyperparameter constraints (positive, bounded ranges)
- Lazy loading of large model weights or datasets
- Logging model attribute changes during training
- Creating typed layer descriptors in neural network frameworks
- Implementing observation spaces in reinforcement learning environments

## Code Examples

### Example 1: Basic descriptor protocol

```python
class VerboseAttribute:
    def __get__(self, obj, objtype=None):
        print(f'Getting {self.name} from {obj}')
        return obj.__dict__.get(self.name, 0)

    def __set__(self, obj, value):
        print(f'Setting {self.name} on {obj} to {value}')
        obj.__dict__[self.name] = value

    def __set_name__(self, owner, name):
        self.name = name

class MyClass:
    x = VerboseAttribute()
    y = VerboseAttribute()

obj = MyClass()
obj.x = 42
print(obj.x)
print(obj.y)

# Output:
# Setting x on <__main__.MyClass object at 0x...> to 42
# Getting x from <__main__.MyClass object at 0x...>
# 42
# Getting y from <__main__.MyClass object at 0x...>
# 0
```

### Example 2: Validation descriptor

```python
class PositiveNumber:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        return obj.__dict__.get(self.name, 0)

    def __set__(self, obj, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f'{self.name} must be a number')
        if value <= 0:
            raise ValueError(f'{self.name} must be positive')
        obj.__dict__[self.name] = value

class Rectangle:
    width = PositiveNumber()
    height = PositiveNumber()

    def __init__(self, width, height):
        self.width = width
        self.height = height

    @property
    def area(self):
        return self.width * self.height

r = Rectangle(10, 20)
print(r.area)

try:
    r.width = -5
except ValueError as e:
    print(e)

try:
    r.height = 'invalid'
except TypeError as e:
    print(e)

# Output:
# 200
# width must be positive
# height must be number
```

### Example 3: property() as a descriptor

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError('Radius cannot be negative')
        self._radius = value

    @property
    def area(self):
        import math
        return math.pi * self._radius ** 2

c = Circle(5)
print(c.radius)
print(f'{c.area:.2f}')

c.radius = 10
print(f'{c.area:.2f}')

# Output:
# 5
# 78.54
# 314.16
```

### Example 4: @classmethod as a descriptor

```python
class MyClass:
    @classmethod
    def factory(cls, value):
        return cls(value)

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f'MyClass({self.value})'

obj = MyClass.factory(42)
print(obj)

# @classmethod works because it's a descriptor whose
# __get__ binds the method to the class, not the instance

# Output:
# MyClass(42)
```

### Example 5: @staticmethod as a descriptor

```python
class MathUtils:
    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def multiply(a, b):
        return a * b

print(MathUtils.add(3, 4))
print(MathUtils.multiply(3, 4))

# @staticmethod is also a descriptor — it returns the
# raw function without binding to class or instance

# Output:
# 7
# 12
```

### Example 6: Lazy loading descriptor

```python
class LazyProperty:
    def __init__(self, func):
        self.func = func
        self.name = func.__name__

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        value = self.func(obj)
        obj.__dict__[self.name] = value
        return value

class DataProcessor:
    def __init__(self, data):
        self.data = data

    @LazyProperty
    def processed(self):
        print('Computing expensive transformation...')
        return [x * 2 for x in self.data]

dp = DataProcessor([1, 2, 3, 4, 5])
print('Object created')
print(dp.processed)
print(dp.processed)

# Output:
# Object created
# Computing expensive transformation...
# [2, 4, 6, 8, 10]
# [2, 4, 6, 8, 10]
```

### Example 7: Data vs non-data descriptors

```python
class NonDataDescriptor:
    def __get__(self, obj, objtype=None):
        return 'non-data descriptor'

class DataDescriptor:
    def __get__(self, obj, objtype=None):
        return 'data descriptor'

    def __set__(self, obj, value):
        print(f'Setting to {value}')

class Demo:
    non_data = NonDataDescriptor()
    data = DataDescriptor()

    def __init__(self):
        self.instance_attr = 'instance'

d = Demo()

# Instance dict shadows non-data descriptor:
d.non_data = 'shadow'
print(d.non_data)

# Instance dict does NOT shadow data descriptor:
d.data = 'try shadow'
print(d.data)

# Output:
# shadow
# Setting to try shadow
# data descriptor
```

## Common Mistakes

1. Forgetting to implement `__set__` when creating read-only data descriptors
2. Storing descriptor state on the descriptor object itself instead of the instance's `__dict__`
3. Confusing data descriptors (with `__set__` or `__delete__`) and non-data descriptors (only `__get__`)
4. Not handling the case when `obj` is `None` in `__get__` (accessing from the class)
5. Overusing descriptors when a simple `@property` would suffice

## Interview Questions

### Beginner - 5

1. What is a descriptor in Python?
2. What methods form the descriptor protocol?
3. What is the difference between a data descriptor and a non-data descriptor?
4. How does `@property` relate to descriptors?
5. What is `__set_name__` used for?

### Intermediate - 5

1. Implement a descriptor that validates that a value is a string.
2. How does `@classmethod` use the descriptor protocol to bind methods to the class?
3. What is the priority order when Python resolves attribute access?
4. How would you implement a read-only descriptor?
5. What happens when you access a descriptor attribute on the class itself (not an instance)?

### Advanced - 3

1. Explain the full attribute lookup chain: data descriptors, instance `__dict__`, non-data descriptors, class `__dict__`, MRO.
2. How would you implement `cached_property` (Python 3.8+) using descriptors?
3. How do descriptors interact with `__slots__` and what happens when both are defined?

## Practice Problems

### Easy - 5

1. Create a descriptor that always returns 42 regardless of what is set.
2. Implement a `LoggedAttribute` descriptor that prints every get and set.
3. Create a `PositiveInteger` descriptor that only allows positive integers.
4. Implement a read-only descriptor using `__set__` that raises `AttributeError`.
5. Create a `StringAttribute` descriptor that only accepts string values.

### Medium - 5

1. Build a `Range` descriptor that validates numeric values are within a given range.
2. Implement a `CachedProperty` descriptor that computes once and caches the result.
3. Create a `TypedAttribute` descriptor that accepts a type and validates assignments.
4. Build a `DefaultValue` descriptor that returns a default if no value has been set.
5. Implement an `ObservedAttribute` descriptor that notifies a callback on value changes.

### Hard - 3

1. Implement a descriptor system that mimics Django model fields with validation, defaults, and null handling.
2. Build a `ValidatedClass` metaclass/descriptor combo that automatically applies validators from class annotations.
3. Design and implement a lazy-loading ORM field system where database values are loaded on first access.

## Solutions

### Easy 1

```python
class ConstDescriptor:
    def __get__(self, obj, objtype=None):
        return 42

    def __set__(self, obj, value):
        raise AttributeError('Cannot modify constant')

class Demo:
    value = ConstDescriptor()

d = Demo()
print(d.value)

# Output:
# 42
```

### Medium 1

```python
class Range:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        return obj.__dict__.get(self.name)

    def __set__(self, obj, value):
        if not (self.min_val <= value <= self.max_val):
            raise ValueError(
                f'{self.name} must be between {self.min_val} and {self.max_val}'
            )
        obj.__dict__[self.name] = value

class Score:
    value = Range(0, 100)
    def __init__(self, value):
        self.value = value

s = Score(85)
print(s.value)

try:
    s = Score(150)
except ValueError as e:
    print(e)

# Output:
# 85
# value must be between 0 and 100
```

### Hard 1 (simplified)

```python
class Field:
    def __init__(self, field_type, default=None):
        self.field_type = field_type
        self.default = default

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name, self.default)

    def __set__(self, obj, value):
        if not isinstance(value, self.field_type):
            raise TypeError(
                f'{self.name} must be {self.field_type.__name__}'
            )
        obj.__dict__[self.name] = value

class User:
    name = Field(str, default='')
    age = Field(int, default=0)

u = User()
u.name = 'Alice'
u.age = 30
print(u.name, u.age)

# Output:
# Alice 30
```

## Related Concepts

- Properties (`@property`)
- Class methods (`@classmethod`) and static methods (`@staticmethod`)
- `__slots__` for attribute restriction
- Metaclasses and attribute access
- `__getattribute__` vs `__getattr__`
- The attribute lookup chain (MRO)

## Next Concepts

- Closures (PYT-041)
- Decorators and function wrapping
- Metaclasses
- Advanced OOP patterns

## Summary

Descriptors are the engine behind Python's attribute access system. By implementing `__get__`, `__set__`, and `__delete__`, custom descriptors can intercept, validate, transform, and compute attribute access. Properties, class methods, and static methods are all implemented using descriptors. Understanding descriptors unlocks deep insight into Python's object model and enables powerful metaprogramming patterns.

## Key Takeaways

- The descriptor protocol: `__get__`, `__set__`, `__delete__`
- Data descriptors (with `__set__`/`__delete__`) take priority over instance `__dict__`
- Non-data descriptors (only `__get__`) are shadowed by instance attributes
- `property()`, `@classmethod`, `@staticmethod` are all descriptor-based
- `__set_name__` provides automatic attribute naming
- Descriptors enable reusable, composable attribute behavior
- Use descriptors for validation, lazy loading, typing, and ORM-style field definitions
