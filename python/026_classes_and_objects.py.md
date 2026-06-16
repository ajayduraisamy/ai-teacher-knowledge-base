# Concept: Classes and Objects

## Concept ID

PYT-026

## Difficulty

Intermediate

## Domain

Python

## Module

Object-Oriented Programming

## Learning Objectives

- Define a Python class using the `class` keyword
- Instantiate objects from a class
- Understand the role of `self` and `__init__`
- Distinguish between instance attributes and class attributes
- Implement `__str__` and `__repr__` for readable object representation
- Compare objects using `__eq__`

## Prerequisites

- Basic Python syntax (variables, functions, conditionals, loops)
- Understanding of built-in data types (`str`, `int`, `list`, `dict`)
- Familiarity with functions and return values

## Definition

A **class** is a blueprint for creating objects. An **object** (instance) is a concrete entity created from that blueprint. Classes bundle data (attributes) and behavior (methods) together. In Python, everything is an object — integers, strings, lists, and functions all have types that are classes.

```python
class Dog:
    pass

print(type(Dog))
print(type(Dog()))
```

```
# Output:
<class 'type'>
<class '__main__.Dog'>
```

## Intuition

Think of a class as a cookie cutter and objects as the cookies you stamp out. The cookie cutter defines the shape (attributes like `flavor`, `size`) and the process (methods like `bake()`, `decorate()`). Each cookie you make is a separate object — you can have many chocolate chip cookies, each of a slightly different size, but they all share the same fundamental structure defined by the cutter.

## Why This Concept Matters

Classes and objects are the foundation of object-oriented programming (OOP), one of the most widely used programming paradigms. OOP helps you model real-world entities, organize code into reusable components, and manage complexity in large codebases. Nearly every major Python framework and library — Django, PyTorch, scikit-learn, Flask — relies heavily on classes.

## Real World Examples

1. **Web Frameworks (Django):** A `User` class models site users. Each user is an object with attributes like `username`, `email`, `password`.
2. **Game Development:** A `Player` class with attributes `health`, `score`, `position`, and methods `move()`, `jump()`, `attack()`.
3. **GUI Applications:** A `Button` class in Tkinter/PyQt that stores `text`, `color`, `size` and handles click events.
4. **Data Science Pipelines:** A `DataLoader` class that loads, cleans, and transforms data.
5. **Banking Systems:** An `Account` class with attributes `balance`, `owner` and methods `deposit()`, `withdraw()`.

## AI/ML Relevance

In machine learning, classes are used to define models, datasets, layers, and training loops.

- PyTorch models inherit from `torch.nn.Module`. Each custom model is a class with `__init__` (defining layers) and `forward` (defining the computation).
- Dataset classes like `torch.utils.data.Dataset` require `__len__` and `__getitem__` methods.
- Scikit-learn estimators are classes with a `fit` and `predict` interface.
- Custom loss functions, metrics, and callbacks are all defined as classes.

## Code Examples

### Example 1: Defining a Simple Class

```python
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

s1 = Student("Alice", 20)
s2 = Student("Bob", 22)

print(s1.name, s1.age)
print(s2.name, s2.age)
```

```
# Output:
Alice 20
Bob 22
```

### Example 2: Instance Methods

```python
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

r = Rectangle(5, 3)
print(f"Area: {r.area()}")
print(f"Perimeter: {r.perimeter()}")
```

```
# Output:
Area: 15
Perimeter: 16
```

### Example 3: Class Attributes vs Instance Attributes

```python
class Employee:
    company = "TechCorp"  # Class attribute

    def __init__(self, name, salary):
        self.name = name    # Instance attribute
        self.salary = salary

e1 = Employee("Alice", 70000)
e2 = Employee("Bob", 80000)

print(e1.company, e1.name, e1.salary)
print(e2.company, e2.name, e2.salary)

Employee.company = "NewTech"
print(e1.company)
print(e2.company)
```

```
# Output:
TechCorp Alice 70000
TechCorp Bob 80000
NewTech
NewTech
```

### Example 4: `__str__` and `__repr__`

```python
class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages

    def __repr__(self):
        return f"Book('{self.title}', '{self.author}', {self.pages})"

    def __str__(self):
        return f"'{self.title}' by {self.author} ({self.pages} pages)"

b = Book("1984", "George Orwell", 328)
print(repr(b))
print(str(b))
print(b)  # implicitly calls __str__
```

```
# Output:
Book('1984', 'George Orwell', 328)
'1984' by George Orwell (328 pages)
'1984' by George Orwell (328 pages)
```

### Example 5: Comparing Objects with `__eq__`

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

v1 = Vector(3, 4)
v2 = Vector(3, 4)
v3 = Vector(1, 2)

print(v1 == v2)
print(v1 == v3)
print(v1 is v2)
```

```
# Output:
True
False
False
```

### Example 6: AI/ML — Simple PyTorch-like Model Class

```python
class LinearModel:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        return self.weight * x + self.bias

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"LinearModel(weight={self.weight}, bias={self.bias})"

model = LinearModel(2.0, 1.0)
print(model)
print(model(3.0))
print(model(5.0))
```

```
# Output:
LinearModel(weight=2.0, bias=1.0)
7.0
11.0
```

## Common Mistakes

1. **Forgetting `self` as the first parameter** of every instance method leads to a `TypeError` when calling the method.
2. **Naming the instance parameter something other than `self`** — while technically allowed, it breaks convention and confuses readers.
3. **Modifying a mutable class attribute via an instance** creates an instance attribute that shadows the class attribute, causing subtle bugs.
4. **Omitting `__init__`** — Python provides a default no-arg `__init__`, but forgetting to define one when you need initialization logic is a common oversight.
5. **Using `__str__` when `__repr__` is more appropriate** — `__repr__` should ideally return a string that can recreate the object, while `__str__` is for user-friendly display.
6. **Not calling `super().__init__()` in subclasses** — this breaks parent class initialization.
7. **Assuming `==` compares object content by default** — it compares identity unless `__eq__` is defined.

## Interview Questions

### Beginner

1. What is the difference between a class and an object?
2. What is `self` in Python?
3. How do you define a constructor in Python?
4. What is the difference between `__str__` and `__repr__`?
5. How do you create an instance of a class?

### Intermediate

1. Explain the difference between class attributes and instance attributes.
2. What happens when you modify a class attribute through an instance?
3. How would you make a class that supports comparison with `==`?
4. Can a class have more than one `__init__` method? Why or why not?
5. What does `__slots__` do and when would you use it?

### Advanced

1. Explain the Python object model — what methods does every object have?
2. How does Python's attribute lookup work (the descriptor protocol)?
3. Implement a singleton pattern using a class.

## Practice Problems

### Easy

1. Define a `Car` class with `make`, `model`, `year` attributes. Create two instances and print their details.
2. Add a method `age(self, current_year)` to the `Car` class that returns the car's age.
3. Define a `BankAccount` class with `owner` and `balance` attributes and methods `deposit()` and `withdraw()`.
4. Add `__str__` to `BankAccount` so it prints "Account owned by {owner}: ${balance}".
5. Create a `Point` class with `x`, `y` coordinates and a method `distance_from_origin()`.

### Medium

1. Add `__eq__` and `__repr__` to the `Point` class.
2. Create a `Library` class that holds a list of `Book` objects with methods `add_book()`, `remove_book()`, and `search_by_author()`.
3. Implement a `Stack` class using a list with `push()`, `pop()`, `peek()`, and `is_empty()` methods.
4. Create a `Temperature` class that stores Celsius internally but provides properties to get/set Fahrenheit.
5. Implement a `Fraction` class with `numerator`, `denominator` and methods `__add__`, `__mul__`, `simplify()`.

### Hard

1. Build a simple ORM-like class `Model` that auto-generates `__init__` from class-level field definitions.
2. Implement a `Descriptor` class that validates an attribute is always a positive integer.
3. Create a `Logger` class that counts the number of times each method is called across all instances.

## Solutions

### Easy — Solution 1

```python
class Car:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year

car1 = Car("Toyota", "Camry", 2020)
car2 = Car("Honda", "Civic", 2021)
print(f"{car1.make} {car1.model} {car1.year}")
print(f"{car2.make} {car2.model} {car2.year}")
```

```
# Output:
Toyota Camry 2020
Honda Civic 2021
```

### Medium — Solution 1

```python
import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def distance_from_origin(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

p1 = Point(3, 4)
p2 = Point(3, 4)
p3 = Point(1, 2)
print(p1)
print(p1 == p2)
print(p1 == p3)
print(p1.distance_from_origin())
```

```
# Output:
Point(3, 4)
True
False
5.0
```

### Hard — Solution 1

```python
class Model:
    _fields = []

    @classmethod
    def _init_fields(cls):
        cls._fields = []
        for name in dir(cls):
            if not name.startswith('_'):
                val = getattr(cls, name)
                if not callable(val):
                    cls._fields.append(name)

    @classmethod
    def create(cls, **kwargs):
        cls._init_fields()
        obj = cls.__new__(cls)
        for field in cls._fields:
            setattr(obj, field, kwargs.get(field))
        return obj

    def __repr__(self):
        fields = ', '.join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({fields})"

class User(Model):
    name = ""
    email = ""

user = User.create(name="Alice", email="alice@example.com")
print(user)
```

```
# Output:
User(name=Alice, email=alice@example.com)
```

## Related Concepts

- Functions and Scope
- Built-in Data Types
- Modules and Packages

## Next Concepts

- Method Types (instance, class, static methods)
- Inheritance
- Polymorphism
- Encapsulation

## Summary

Classes are blueprints for creating objects. The `__init__` method initializes new instances. `self` refers to the current instance. Class attributes are shared across all instances, while instance attributes are unique to each object. Special methods like `__str__`, `__repr__`, and `__eq__` customize string representation and comparison behavior. Classes are fundamental to organizing code and are heavily used in AI/ML frameworks for defining models, datasets, and training components.

## Key Takeaways

- Use `class ClassName:` to define a class
- `__init__` is the constructor, called when you create an instance
- Always use `self` as the first parameter of instance methods
- Class attributes are shared; instance attributes are per-object
- `__str__` is for readable output, `__repr__` is for unambiguous representation
- Define `__eq__` to enable meaningful object comparison with `==`
- Classes enable code reuse, organization, and real-world modeling
