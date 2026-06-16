# Concept: Method Types

## Concept ID

PYT-027

## Difficulty

Intermediate

## Domain

Python

## Module

Object-Oriented Programming

## Learning Objectives

- Distinguish between instance methods, class methods, and static methods
- Use `@classmethod` with `cls` parameter
- Use `@staticmethod` without `self` or `cls`
- Use `@property` for computed attributes
- Decide when each method type is appropriate

## Prerequisites

- Classes and objects (PYT-026)
- Understanding of `self` and `__init__`
- Familiarity with decorators

## Definition

Python classes support three types of methods:

1. **Instance methods** — receive `self`, operate on instance data.
2. **Class methods** — receive `cls` (the class itself), decorated with `@classmethod`.
3. **Static methods** — receive neither `self` nor `cls`, decorated with `@staticmethod`.

Additionally, the `@property` decorator turns a method into a computed, read-only attribute.

## Intuition

Think of a class as a factory. **Instance methods** are like tools given to each product the factory makes — they work with that product's specific features. **Class methods** are like instructions posted on the factory wall — they affect or inform the factory as a whole. **Static methods** are like a generic reference manual sitting inside the factory — they perform a task related to the domain but don't need access to the factory or its products.

## Why This Concept Matters

Choosing the right method type makes your code clearer, safer, and more expressive. Instance methods are the default and most common. Class methods provide alternative constructors and factory patterns. Static methods organize utility functions that belong to the class's namespace. Properties let you present computed data as attributes, not method calls, improving API ergonomics.

## Real World Examples

1. **Alternative constructors:** `datetime.fromtimestamp(ts)` is a class method that creates a datetime from a timestamp.
2. **Factory patterns:** `torch.nn.Sequential` class methods that build models from configuration.
3. **Utility functions:** A `MathHelper` class with static methods like `factorial(n)`, `is_prime(n)`.
4. **Computed properties:** A `Circle` class with a `@property` for `area` computed from `radius`.
5. **Configuration validation:** A `Config` class that validates its own class-level defaults.

## AI/ML Relevance

- **Class methods** serve as factory methods for model architectures (e.g., `ResNet18()`, `ResNet50()`).
- **Static methods** organize data preprocessing utilities (normalization, tokenization).
- **Properties** are used for computed metrics like loss, accuracy, or gradient norms.
- Scikit-learn transformers use class methods for `get_feature_names_out()`.

## Code Examples

### Example 1: Instance Methods

```python
class Counter:
    def __init__(self, start=0):
        self.count = start

    def increment(self, step=1):
        self.count += step
        return self.count

    def reset(self):
        self.count = 0

c = Counter(10)
print(c.increment())
print(c.increment(5))
c.reset()
print(c.count)
```

```
# Output:
11
16
0
```

### Example 2: Class Methods

```python
class Employee:
    company = "TechCorp"

    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    @classmethod
    def from_string(cls, data_string):
        name, salary = data_string.split(",")
        return cls(name.strip(), int(salary.strip()))

    @classmethod
    def change_company(cls, new_name):
        cls.company = new_name

e = Employee.from_string("Alice, 75000")
print(e.name, e.salary)
print(e.company)

Employee.change_company("NewTech")
print(Employee.company)
```

```
# Output:
Alice 75000
TechCorp
NewTech
```

### Example 3: Static Methods

```python
class MathUtils:
    @staticmethod
    def is_even(n):
        return n % 2 == 0

    @staticmethod
    def factorial(n):
        if n < 0:
            raise ValueError("Negative input")
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    @staticmethod
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

print(MathUtils.is_even(10))
print(MathUtils.factorial(5))
print(MathUtils.gcd(48, 18))
```

```
# Output:
True
120
6
```

### Example 4: `@property` for Computed Attributes

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
            raise ValueError("Radius cannot be negative")
        self._radius = value

    @property
    def area(self):
        import math
        return math.pi * self._radius ** 2

    @property
    def circumference(self):
        import math
        return 2 * math.pi * self._radius

c = Circle(5)
print(c.radius)
print(c.area)
print(c.circumference)

c.radius = 10
print(c.area)
```

```
# Output:
5
78.53981633974483
31.41592653589793
314.1592653589793
```

### Example 5: AI/ML — Factory Method for Models

```python
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        print(f"Created network with layers: {layers}")

    @classmethod
    def small(cls):
        return cls([784, 128, 10])

    @classmethod
    def medium(cls):
        return cls([784, 256, 128, 10])

    @classmethod
    def large(cls):
        return cls([784, 512, 256, 128, 10])

    @staticmethod
    def activation_summary():
        return ["ReLU", "Sigmoid", "Tanh", "Softmax"]

model_small = NeuralNetwork.small()
model_large = NeuralNetwork.large()
print(NeuralNetwork.activation_summary())
```

```
# Output:
Created network with layers: [784, 128, 10]
Created network with layers: [784, 512, 256, 128, 10]
['ReLU', 'Sigmoid', 'Tanh', 'Softmax']
```

### Example 6: Combining All Method Types

```python
class TemperatureConverter:
    _scale = "metric"

    def __init__(self, celsius=0):
        self.celsius = celsius

    @property
    def fahrenheit(self):
        return (self.celsius * 9 / 5) + 32

    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5 / 9

    @classmethod
    def from_fahrenheit(cls, f):
        celsius = (f - 32) * 5 / 9
        return cls(celsius)

    @classmethod
    def set_scale(cls, scale):
        cls._scale = scale

    @staticmethod
    def is_freezing(celsius):
        return celsius <= 0

t = TemperatureConverter(100)
print(t.fahrenheit)

t.fahrenheit = 32
print(t.celsius)

t2 = TemperatureConverter.from_fahrenheit(212)
print(t2.celsius)

print(TemperatureConverter.is_freezing(-5))
```

```
# Output:
212.0
0.0
100.0
True
```

## Common Mistakes

1. **Using `self` in a static method** — causes a TypeError; static methods receive no automatic arguments.
2. **Forgetting `@staticmethod` or `@classmethod`** — the method behaves as an instance method, expecting `self` as the first argument.
3. **Using `self` in a class method** — convention is `cls`; using `self` is misleading even though it works.
4. **Calling a property as a method** — `obj.property()` raises TypeError if the property is read-only; properties are accessed without parentheses.
5. **Modifying a property's value without a setter** — properties without setter are read-only; assigning raises `AttributeError`.
6. **Overusing static methods** — if a method operates on instance data, it should be an instance method, not static.
7. **Using class methods when regular functions suffice** — if the method doesn't need the class at all, consider a module-level function instead.
8. **Defining a property setter before the getter** — the `@property` decorator must come first, then `@name.setter`, `@name.deleter`.

## Interview Questions

### Beginner

1. What is the difference between an instance method and a static method?
2. What decorator is used for class methods?
3. How do you access a class method? Do you need an instance?
4. What is `cls` in a class method?
5. How do you define a static method in Python?

### Intermediate

1. When would you use a class method over a static method?
2. How does `@property` change the way you access an attribute?
3. Can you define a property setter without a getter? Why or why not?
4. What happens if you don't decorate a method that should be static?
5. Give an example of an alternative constructor using `@classmethod`.

### Advanced

1. Explain the method resolution order for `@classmethod` vs `@staticmethod` with inheritance.
2. How can you use `__init_subclass__` combined with class methods for plugin registration?
3. Implement a custom descriptor that behaves like `@property`.

## Practice Problems

### Easy

1. Define a `Person` class with an instance method `greet()` that prints "Hello, my name is {name}".
2. Add a static method `is_adult(age)` that returns `True` if age >= 18.
3. Add a class method `from_birth_year(cls, name, year)` that calculates age automatically.
4. Create a `Rectangle` class with a `@property` for `area`.
5. Add a setter to the `Rectangle` class that recomputes dimensions based on a ratio.

### Medium

1. Create a `BankAccount` class with a property `balance` (read-only) and methods `deposit()`, `withdraw()`.
2. Implement a `Student` class with a class attribute `school_name` and a class method to change it.
3. Build a `Product` class with a static method `calculate_discount(price, percent)`.
4. Create a `Timer` class with properties `elapsed_seconds`, `elapsed_minutes`, and `elapsed_hours`.
5. Implement a `Color` class with class methods `from_hex(cls, hex_str)` and `from_rgb(cls, r, g, b)`.

### Hard

1. Build a `ValidatedProperty` descriptor that validates type and range on assignment.
2. Create a `Registry` metaclass that automatically collects all subclasses using `__init_subclass__` and class methods.
3. Implement a `CachedProperty` descriptor that computes once then caches the result.

## Solutions

### Easy — Solution 1

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        print(f"Hello, my name is {self.name}")

    @staticmethod
    def is_adult(age):
        return age >= 18

    @classmethod
    def from_birth_year(cls, name, year):
        age = 2026 - year
        return cls(name, age)

p = Person("Alice", 25)
p.greet()
print(Person.is_adult(20))
print(Person.is_adult(16))
p2 = Person.from_birth_year("Bob", 2000)
print(p2.name, p2.age)
```

```
# Output:
Hello, my name is Alice
True
False
Bob 26
```

### Medium — Solution 1

```python
class BankAccount:
    def __init__(self, owner, initial_balance=0):
        self.owner = owner
        self._balance = initial_balance

    @property
    def balance(self):
        return self._balance

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Amount must be positive")
        self._balance += amount

    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("Amount must be positive")
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount

acc = BankAccount("Alice", 1000)
acc.deposit(500)
print(acc.balance)
acc.withdraw(200)
print(acc.balance)
```

```
# Output:
1500
1300
```

### Hard — Solution 1

```python
class ValidatedProperty:
    def __init__(self, attr_name, expected_type, min_val=None, max_val=None):
        self.attr_name = attr_name
        self.expected_type = expected_type
        self.min_val = min_val
        self.max_val = max_val

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.attr_name, None)

    def __set__(self, obj, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(f"Expected {self.expected_type}")
        if self.min_val is not None and value < self.min_val:
            raise ValueError(f"Value must be >= {self.min_val}")
        if self.max_val is not None and value > self.max_val:
            raise ValueError(f"Value must be <= {self.max_val}")
        setattr(obj, self.attr_name, value)

class Product:
    price = ValidatedProperty("_price", (int, float), min_val=0)
    quantity = ValidatedProperty("_quantity", int, min_val=0)

    def __init__(self, name, price, quantity):
        self.name = name
        self.price = price
        self.quantity = quantity

p = Product("Laptop", 999.99, 10)
print(p.price, p.quantity)
try:
    p.price = -5
except ValueError as e:
    print(e)
```

```
# Output:
999.99 10
Value must be >= 0
```

## Related Concepts

- Classes and Objects
- Decorators
- Properties and Descriptors

## Next Concepts

- Inheritance
- Polymorphism
- Encapsulation

## Summary

Python provides three method types: instance methods (operate on `self`), class methods (operate on `cls`, decorated with `@classmethod`), and static methods (no automatic arguments, decorated with `@staticmethod`). The `@property` decorator turns a method into a computed attribute, optionally with a setter. Choosing the right type improves code clarity and follows the principle of least surprise.

## Key Takeaways

- Instance methods are the default; use them when you need to access instance data
- `@classmethod` receives the class; use it for alternative constructors and factory methods
- `@staticmethod` receives nothing; use it for utility functions related to the class
- `@property` makes computed values look like attributes
- Properties with setters enable controlled attribute assignment
- Class methods modify class-level state that affects all instances
