# Concept: Inheritance

## Concept ID

PYT-028

## Difficulty

Intermediate

## Domain

Python

## Module

Object-Oriented Programming

## Learning Objectives

- Create parent and child classes
- Use `super().__init__()` to call the parent constructor
- Override methods in child classes
- Use `isinstance()` and `issubclass()` for type checking
- Distinguish between extending and overriding

## Prerequisites

- Classes and objects (PYT-026)
- Method types (PYT-027)
- Basic understanding of `self`

## Definition

Inheritance is a mechanism where a **child class** (subclass) derives properties and behaviors from a **parent class** (superclass). The child automatically inherits all attributes and methods of the parent and can add new ones or override existing ones. This promotes code reuse and establishes an "is-a" relationship.

```python
class Animal:
    pass

class Dog(Animal):
    pass

print(issubclass(Dog, Animal))
```

```
# Output:
True
```

## Intuition

Think of inheritance as a family tree. A parent passes down traits (attributes, methods) to their children. Children can have their own unique traits or modify inherited ones. A `Dog` **is an** `Animal` — it has everything an animal has (breathes, eats), but it also has specific behaviors (barks). A `Poodle` **is a** `Dog` — it inherits dog traits and adds poodle-specific features.

## Why This Concept Matters

Inheritance is a cornerstone of OOP. It enables you to write generic code in a base class and specialize it in subclasses without duplication. Frameworks rely on inheritance extensively — Django's `Model` classes, PyTorch's `nn.Module`, and scikit-learn's estimators all expect you to subclass and override specific methods.

## Real World Examples

1. **Django Models:** `class Product(models.Model):` — the base `Model` provides save, delete, query methods; you add fields.
2. **PyTorch Modules:** `class MyModel(nn.Module):` — base provides parameter management, device transfer; you define `forward`.
3. **Exception Hierarchy:** `class ValueError(Exception):` — all exceptions inherit from `BaseException`.
4. **GUI Widgets:** `class MyButton(tk.Button):` — inherits layout, event handling; you customize appearance.
5. **Testing:** `class TestUserAPI(unittest.TestCase):` — inherits `assertEqual`, `setUp`, `tearDown`.

## AI/ML Relevance

- All custom PyTorch models inherit from `torch.nn.Module`, inheriting parameter registration, `to()`, `train()`, `eval()`.
- Custom datasets inherit from `torch.utils.data.Dataset` and must override `__len__` and `__getitem__`.
- Scikit-learn requires estimators to inherit from `BaseEstimator` and `ClassifierMixin` / `RegressorMixin`.
- Keras layers inherit from `tf.keras.layers.Layer`.
- Custom callbacks inherit from `keras.callbacks.Callback`.

## Code Examples

### Example 1: Basic Inheritance

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return "..."

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

animals = [Dog("Rex"), Cat("Whiskers"), Animal("Generic")]
for a in animals:
    print(f"{a.name} says {a.speak()}")
```

```
# Output:
Rex says Woof!
Whiskers says Meow!
Generic says ...
```

### Example 2: Using `super().__init__()`

```python
class Vehicle:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year

    def info(self):
        return f"{self.year} {self.make} {self.model}"

class Car(Vehicle):
    def __init__(self, make, model, year, doors):
        super().__init__(make, model, year)
        self.doors = doors

    def info(self):
        return f"{super().info()}, {self.doors} doors"

class Motorcycle(Vehicle):
    def __init__(self, make, model, year, has_sidecar):
        super().__init__(make, model, year)
        self.has_sidecar = has_sidecar

    def info(self):
        sidecar = "with sidecar" if self.has_sidecar else "no sidecar"
        return f"{super().info()}, {sidecar}"

c = Car("Toyota", "Camry", 2021, 4)
m = Motorcycle("Harley", "Street", 2020, False)
print(c.info())
print(m.info())
```

```
# Output:
2021 Toyota Camry, 4 doors
2020 Harley Street, no sidecar
```

### Example 3: Overriding vs Extending

```python
class Base:
    def greet(self):
        return "Hello"

    def farewell(self):
        return "Goodbye"

class Child(Base):
    # Override — completely replaces parent behavior
    def greet(self):
        return "Hey there!"

    # Extend — adds to parent behavior
    def farewell(self):
        original = super().farewell()
        return f"{original}, my friend!"

b = Base()
c = Child()
print(b.greet())
print(c.greet())
print(b.farewell())
print(c.farewell())
```

```
# Output:
Hello
Hey there!
Goodbye
Goodbye, my friend!
```

### Example 4: `isinstance()` and `issubclass()`

```python
class Media:
    pass

class Book(Media):
    pass

class Video(Media):
    pass

class EBook(Book):
    pass

b = Book()
v = Video()
e = EBook()

print(isinstance(b, Media))
print(isinstance(b, Book))
print(isinstance(b, Video))
print(isinstance(e, Book))
print(issubclass(EBook, Book))
print(issubclass(EBook, Media))
print(issubclass(Book, EBook))
```

```
# Output:
True
True
False
True
True
True
False
```

### Example 5: AI/ML — Custom Dataset

```python
class Dataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class LabeledDataset(Dataset):
    def __init__(self, data, labels):
        super().__init__(data)
        self.labels = labels

    def __getitem__(self, idx):
        features = super().__getitem__(idx)
        label = self.labels[idx]
        return features, label

data = ["image1", "image2", "image3"]
labels = [0, 1, 0]
ds = LabeledDataset(data, labels)
print(len(ds))
print(ds[0])
print(ds[1])
```

```
# Output:
3
('image1', 0)
('image2', 1)
```

### Example 6: Multi-level Inheritance

```python
class Person:
    def __init__(self, name):
        self.name = name

class Employee(Person):
    def __init__(self, name, employee_id):
        super().__init__(name)
        self.employee_id = employee_id

    def work(self):
        return f"{self.name} is working"

class Manager(Employee):
    def __init__(self, name, employee_id, team_size):
        super().__init__(name, employee_id)
        self.team_size = team_size

    def work(self):
        return f"{self.name} is managing a team of {self.team_size}"

m = Manager("Alice", "E001", 5)
print(m.work())
print(m.name, m.employee_id)
print(isinstance(m, Person))
print(isinstance(m, Employee))
print(isinstance(m, Manager))
```

```
# Output:
Alice is managing a team of 5
Alice E001
True
True
True
```

## Common Mistakes

1. **Forgetting to call `super().__init__()`** — the parent's initialization code never runs, leaving attributes unset.
2. **Wrong `super()` arguments** — in Python 3, `super()` with no arguments works correctly; do not pass the class and instance manually.
3. **Circular inheritance** — creating loops that cause `TypeError: Cannot create a consistent method resolution order (MRO)`.
4. **Deep inheritance chains** — more than 3-4 levels makes code hard to understand and debug.
5. **Overriding a method but accidentally breaking the expected contract** — overriding `__getitem__` with a different signature violates the Liskov substitution principle.
6. **Using inheritance when composition is more appropriate** — not everything that "uses" something should inherit from it.
7. **Accessing private name-mangled attributes** — `__attr` is mangled to `_ClassName__attr` and is not directly accessible in subclasses without the mangled name.
8. **Assuming `super()` always calls the direct parent** — with multiple inheritance, `super()` follows MRO, which may not be the immediate parent.

## Interview Questions

### Beginner

1. What is inheritance in Python?
2. How do you define a class that inherits from another class?
3. What is `super()` and why would you use it?
4. What is the difference between overriding and overloading?
5. How do you check if an object is an instance of a class?

### Intermediate

1. Explain the Liskov substitution principle in the context of inheritance.
2. What happens if a child class does not call `super().__init__()`?
3. Can you inherit from multiple classes in Python? What are the risks?
4. How does `super()` work in a multi-level inheritance chain?
5. What is the difference between `isinstance()` and `issubclass()`?

### Advanced

1. Explain Python's C3 linearization algorithm for MRO.
2. How does `super()` behave in a diamond inheritance scenario?
3. How can you prevent a class from being subclassed?

## Practice Problems

### Easy

1. Create a base class `Shape` with a method `area()` returning 0. Create `Circle` and `Rectangle` subclasses.
2. Add a `__str__` method to `Shape` and override it in `Circle` and `Rectangle`.
3. Create a base `Logger` class with `log(message)` and a `FileLogger` subclass that writes to a file.
4. Create a `Person` base class and `Student` subclass with an extra `grade` attribute.
5. Use `isinstance()` to check object types in a list of mixed shapes.

### Medium

1. Implement a class hierarchy: `ElectronicDevice` -> `Phone` -> `Smartphone`. Each adds features.
2. Create a `Product` base class and `DiscountedProduct` subclass that overrides the price property.
3. Build a `Validator` base class with `validate(value)` and subclasses `EmailValidator`, `PhoneValidator`.
4. Create a `Notification` base with `send()` and subclasses `EmailNotification`, `SMSNotification`.
5. Implement the template method pattern: a `DataProcessor` class with a fixed `process()` flow and abstract steps.

### Hard

1. Implement a mixin chain for serialization: `JSONMixin`, `XMLMixin`, `PickleMixin` that add `to_json()`, `to_xml()`, `to_pickle()` to any class.
2. Create a class hierarchy for a simple ORM with `Model`, `TableModel`, `ViewModel` using metaclasses.
3. Implement a proxy class that inherits from a target class dynamically and intercepts all method calls.

## Solutions

### Easy — Solution 1

```python
import math

class Shape:
    def area(self):
        return 0

    def __str__(self):
        return f"{self.__class__.__name__}"

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2

    def __str__(self):
        return f"Circle(r={self.radius})"

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def __str__(self):
        return f"Rectangle({self.width}x{self.height})"

shapes = [Circle(5), Rectangle(3, 4), Shape()]
for s in shapes:
    print(f"{s}: area = {s.area():.2f}")
```

```
# Output:
Circle(r=5): area = 78.54
Rectangle(3x4): area = 12.00
Shape: area = 0.00
```

### Medium — Solution 1

```python
class ElectronicDevice:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def power_on(self):
        return f"{self.brand} {self.model} is powering on"

class Phone(ElectronicDevice):
    def __init__(self, brand, model, number):
        super().__init__(brand, model)
        self.number = number

    def call(self, contact):
        return f"Calling {contact} from {self.number}"

class Smartphone(Phone):
    def __init__(self, brand, model, number, os):
        super().__init__(brand, model, number)
        self.os = os

    def install_app(self, app_name):
        return f"Installing {app_name} on {self.brand} {self.model} ({self.os})"

s = Smartphone("Apple", "iPhone 15", "555-0100", "iOS 18")
print(s.power_on())
print(s.call("Alice"))
print(s.install_app("Calculator"))
```

```
# Output:
Apple iPhone 15 is powering on
Calling Alice from 555-0100
Installing Calculator on Apple iPhone 15 (iOS 18)
```

### Hard — Solution 1

```python
import json
import pickle
import xml.etree.ElementTree as ET
from xml.dom import minidom

class JSONMixin:
    def to_json(self):
        return json.dumps(self.__dict__, indent=2)

class XMLMixin:
    def to_xml(self):
        root = ET.Element(self.__class__.__name__)
        for key, value in self.__dict__.items():
            child = ET.SubElement(root, key)
            child.text = str(value)
        rough = ET.tostring(root, encoding='unicode')
        return minidom.parseString(rough).toprettyxml(indent="  ")

class PickleMixin:
    def to_pickle(self):
        return pickle.dumps(self.__dict__)

    def from_pickle(cls, data):
        obj = cls.__new__(cls)
        obj.__dict__ = pickle.loads(data)
        return obj

class SerializablePerson(JSONMixin, XMLMixin, PickleMixin):
    def __init__(self, name, age):
        self.name = name
        self.age = age

p = SerializablePerson("Alice", 30)
print(p.to_json())
print("---")
print(p.to_xml()[:100] + "...")
```

```
# Output:
{
  "name": "Alice",
  "age": 30
}
---
<?xml version="1.0" ?>
<SerializablePerson>
  <name>Alice</name>
  <age>30</age>
...
```

## Related Concepts

- Classes and Objects
- Method Types
- Polymorphism

## Next Concepts

- Polymorphism
- Encapsulation
- Abstract Classes

## Summary

Inheritance allows a child class to reuse and extend the behavior of a parent class. Use `super().__init__()` to call the parent constructor. Child classes can override methods (replace entirely) or extend them (add behavior). Use `isinstance()` to check object types and `issubclass()` to check class relationships. Inheritance is central to Python frameworks and AI/ML libraries.

## Key Takeaways

- Child classes inherit all attributes and methods from parent classes
- Always call `super().__init__()` in the child's `__init__`
- Override = replace, Extend = override + call `super().method()`
- `isinstance(obj, Class)` checks type; `issubclass(Child, Parent)` checks hierarchy
- Favor shallow inheritance trees (2-3 levels max)
- Deep inheritance chains are hard to maintain and debug
