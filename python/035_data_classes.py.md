# Concept: Data Classes

## Concept ID

PYT-035

## Difficulty

Intermediate

## Domain

Python

## Module

Object-Oriented Programming

## Learning Objectives

- Create classes using `@dataclass` decorator
- Use `field()` for fine-grained control over attributes
- Create immutable data classes with `frozen=True`
- Enable ordering with `order=True`
- Use `__post_init__` for validation and derived values
- Optimize memory with `__slots__` in data classes
- Convert data classes to dicts and tuples with `asdict()` and `astuple()`

## Prerequisites

- Classes and objects (PYT-026)
- Magic methods (PYT-031)
- Basic understanding of type hints

## Definition

A **data class** is a class that primarily exists to hold data. Introduced in Python 3.7 via PEP 557, the `@dataclass` decorator automatically generates `__init__`, `__repr__`, `__eq__`, and optionally `__hash__`, `__lt__`, `__le__`, `__gt__`, `__ge__`, and `__ FrozenInstanceError` for frozen classes. It dramatically reduces boilerplate code for simple data containers.

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

p = Point(3.0, 4.0)
print(p)
print(p.x, p.y)
```

```
# Output:
Point(x=3.0, y=4.0)
3.0 4.0
```

## Intuition

Writing a class that just holds data is tedious — you write `__init__`, `__repr__`, `__eq__`, and maybe `__hash__` every time. A data class says "I'm a container for data" and Python writes all that boilerplate for you automatically. It's like giving Python a CSV header row and having it build the entire class for you.

## Why This Concept Matters

Data classes eliminate repetitive boilerplate code, reduce bugs (no more mismatched `__init__` parameters), and make your intent clear — "this class is just data." They integrate with type hints, support immutability, and provide useful utilities like `asdict()` and `astuple()`. They are the go-to choice for configuration, API responses, database records, and ML experiment parameters.

## Real World Examples

1. **Configuration classes:** Storing hyperparameters, database config, API keys.
2. **API response models:** Mapping JSON responses to typed Python objects.
3. **Data transfer objects (DTOs):** Passing data between layers of an application.
4. **Database row models:** Representing rows from SQL queries.
5. **ML experiment tracking:** Storing experiment parameters, metrics, and results.

## AI/ML Relevance

- Experiment configuration classes (learning rate, batch size, epochs) as `@dataclass` with defaults and validation.
- Data point representations (features, labels, metadata) as data classes.
- Training state objects (epoch, loss, metrics) for checkpointing.
- Hyperparameter search spaces with typed fields and ranges.
- Dataset metadata (paths, transforms, splits) stored in data classes.
- `asdict()` converts experiment configs to JSON for logging.

## Code Examples

### Example 1: Basic Data Class

```python
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
    email: str

p = Person("Alice", 30, "alice@example.com")
print(p)
print(p.name)
print(p.age)
```

```
# Output:
Person(name='Alice', age=30, email='alice@example.com')
Alice
30
```

### Example 2: Default Values and field()

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Student:
    name: str
    grades: List[float] = field(default_factory=list)
    active: bool = True
    student_id: int = field(default=0, compare=False)

s1 = Student("Alice")
s1.grades.append(95.0)
s1.grades.append(87.0)

s2 = Student("Bob")
s2.grades.append(92.0)

print(s1)
print(s2)
print(s1 == s2)  # student_id is excluded from comparison
```

```
# Output:
Student(name='Alice', grades=[95.0, 87.0], active=True, student_id=0)
Student(name='Bob', grades=[92.0], active=True, student_id=0)
True
```

### Example 3: Frozen (Immutable) Data Class

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    learning_rate: float
    batch_size: int
    epochs: int
    model_name: str

config = Config(0.001, 64, 10, "resnet18")
print(config)
print(config.learning_rate)

# config.learning_rate = 0.01  # FrozenInstanceError!
```

```
# Output:
Config(learning_rate=0.001, batch_size=64, epochs=10, model_name='resnet18')
0.001
```

### Example 4: Ordering Support

```python
from dataclasses import dataclass

@dataclass(order=True)
class Product:
    price: float
    name: str

p1 = Product(10.99, "Widget")
p2 = Product(5.99, "Gadget")
p3 = Product(10.99, "Thingy")

print(p1 < p2)
print(p2 < p1)
print(p1 <= p3)
print(p3 >= p2)
print(sorted([p1, p2, p3]))
```

```
# Output:
False
True
True
True
[Product(price=5.99, name='Gadget'), Product(price=10.99, name='Thingy'), Product(price=10.99, name='Widget')]
```

### Example 5: `__post_init__` for Validation

```python
from dataclasses import dataclass

@dataclass
class Temperature:
    celsius: float

    def __post_init__(self):
        if self.celsius < -273.15:
            raise ValueError(f"Temperature {self.celsius}C is below absolute zero")
        # Derived attribute
        object.__setattr__(self, 'kelvin', self.celsius + 273.15)

t = Temperature(100)
print(t.celsius, t.kelvin)

t2 = Temperature(-40)
print(t2.celsius, t2.kelvin)

try:
    t3 = Temperature(-300)
except ValueError as e:
    print(e)
```

```
# Output:
100 373.15
-40 233.15
Temperature -300C is below absolute zero
```

### Example 6: `asdict()` and `astuple()`

```python
from dataclasses import dataclass, asdict, astuple
from typing import List

@dataclass
class Address:
    street: str
    city: str
    zip_code: str

@dataclass
class Employee:
    name: str
    salary: float
    address: Address
    skills: List[str]

emp = Employee(
    name="Alice",
    salary=85000.0,
    address=Address("123 Main St", "Portland", "97201"),
    skills=["Python", "ML", "SQL"]
)

print(asdict(emp))
print("---")
print(astuple(emp))
```

```
# Output:
{'name': 'Alice', 'salary': 85000.0, 'address': {'street': '123 Main St', 'city': 'Portland', 'zip_code': '97201'}, 'skills': ['Python', 'ML', 'SQL']}
---
('Alice', 85000.0, Address(street='123 Main St', city='Portland', zip_code='97201'), ['Python', 'ML', 'SQL'])
```

### Example 7: AI/ML — Experiment Config

```python
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json

@dataclass
class DatasetConfig:
    name: str
    path: str
    validation_split: float = 0.2
    shuffle: bool = True
    augment: bool = False

@dataclass
class ModelConfig:
    architecture: str
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.2
    activation: str = "relu"

@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 10
    optimizer: str = "adam"
    early_stopping_patience: int = 5

@dataclass
class ExperimentConfig:
    experiment_name: str
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    seed: int = 42
    use_gpu: bool = True
    tags: List[str] = field(default_factory=list)

    def to_json(self):
        return json.dumps(asdict(self), indent=2)

config = ExperimentConfig(
    experiment_name="mnist_v1",
    dataset=DatasetConfig("MNIST", "./data/mnist"),
    model=ModelConfig("simple_cnn", hidden_layers=[128, 64]),
    training=TrainingConfig(learning_rate=0.0005, epochs=20),
    tags=["vision", "cnn"]
)

print(config.to_json())
```

```
# Output:
{
  "experiment_name": "mnist_v1",
  "dataset": {
    "name": "MNIST",
    "path": "./data/mnist",
    "validation_split": 0.2,
    "shuffle": true,
    "augment": false
  },
  "model": {
    "architecture": "simple_cnn",
    "hidden_layers": [
      128,
      64
    ],
    "dropout": 0.2,
    "activation": "relu"
  },
  "training": {
    "learning_rate": 0.0005,
    "batch_size": 64,
    "epochs": 20,
    "optimizer": "adam",
    "early_stopping_patience": 5
  },
  "seed": 42,
  "use_gpu": true,
  "tags": [
    "vision",
    "cnn"
  ]
}
```

## Common Mistakes

1. **Using mutable default values** — `grades: List[float] = []` creates one list shared across all instances. Use `field(default_factory=list)`.
2. **Forgetting `default_factory` for mutable types** — `field(default_factory=dict)`, `field(default_factory=list)`, etc.
3. **Setting `frozen=True` and then trying to modify** — raises `FrozenInstanceError`. Use `object.__setattr__()` in `__post_init__` if needed.
4. **Comparing order-enabled data classes with incompatible types** — `order=True` generates comparison methods that may fail with mixed types.
5. **Overriding `__init__` in a data class** — the decorator generates `__init__`; overriding it defeats the purpose. Use `__post_init__` instead.
6. **Forgetting type annotations** — fields without type annotations are ignored by `@dataclass`.
7. **Using `@dataclass` for classes with complex behavior** — data classes are for data containers. If you have many methods, a regular class may be better.
8. **Not considering `eq` and `order` parameters together** — setting `order=True` implies `eq=True`; setting `eq=False` with `order=True` raises `ValueError`.

## Interview Questions

### Beginner

1. What is a data class in Python?
2. What methods does `@dataclass` auto-generate?
3. How do you set a default value in a data class?
4. What is the `field()` function used for?
5. How do you make a data class immutable?

### Intermediate

1. What is the difference between `field(default=[])` and `field(default_factory=list)`?
2. What is `__post_init__` and when would you use it?
3. How does `asdict()` handle nested data classes?
4. What does `order=True` do in a data class?
5. How do `@dataclass` and `__slots__` interact?

### Advanced

1. How does `@dataclass` interact with inheritance?
2. How would you implement a custom `__hash__` on a frozen data class?
3. Explain the internal mechanics of how `@dataclass` generates methods.

## Practice Problems

### Easy

1. Create a `Book` data class with `title`, `author`, `year`, `isbn`.
2. Create an `InventoryItem` data class with `name`, `price`, `quantity` and a default quantity of 0.
3. Create a `Point3D` data class with `x`, `y`, `z` and ordering enabled.
4. Create a frozen `Hyperparameters` data class with `lr`, `batch_size`, `epochs`.
5. Create an `Address` data class with `street`, `city`, `state`, `zip` and a default city.

### Medium

1. Add `__post_init__` to validate that `Price` is positive and `quantity` is non-negative.
2. Create a nested data class structure: `Order` contains `Customer` and `List[Item]`.
3. Use `field(repr=False)` to hide passwords in a `User` data class.
4. Create a `Range` data class with `start`, `end`, `step` and a `__post_init__` that normalizes values.
5. Implement a `LogEntry` data class with timestamp auto-generation in `__post_init__`.

### Hard

1. Create a data class that tracks its own creation order using a class-level counter in `__post_init__`.
2. Implement a `JSONSerializable` mixin that provides `to_json()` and `from_json()` for any data class.
3. Build a data class registry that automatically discovers and indexes all data classes in a module.

## Solutions

### Easy — Solution 1

```python
from dataclasses import dataclass

@dataclass
class Book:
    title: str
    author: str
    year: int
    isbn: str

b = Book("1984", "George Orwell", 1949, "978-0451524935")
print(b)
print(b.title)
```

```
# Output:
Book(title='1984', author='George Orwell', year=1949, isbn='978-0451524935')
1984
```

### Medium — Solution 1

```python
from dataclasses import dataclass

@dataclass
class Price:
    amount: float
    currency: str = "USD"

    def __post_init__(self):
        if self.amount <= 0:
            raise ValueError(f"Price must be positive, got {self.amount}")

@dataclass
class Item:
    name: str
    price: Price
    quantity: int = 0

    def __post_init__(self):
        if self.quantity < 0:
            raise ValueError(f"Quantity must be non-negative, got {self.quantity}")

    def total(self):
        return self.price.amount * self.quantity

item = Item("Widget", Price(10.99), 5)
print(item)
print(f"Total: ${item.total():.2f}")

try:
    bad = Item("Bad", Price(-5), 1)
except ValueError as e:
    print(e)
```

```
# Output:
Item(name='Widget', price=Price(amount=10.99, currency='USD'), quantity=5)
Total: $54.95
Price must be positive, got -5
```

### Hard — Solution 1

```python
from dataclasses import dataclass, field

class CreationCounter:
    _count = 0

@dataclass
class TrackedItem:
    name: str
    value: float
    creation_order: int = field(init=False, repr=False)

    def __post_init__(self):
        CreationCounter._count += 1
        object.__setattr__(self, 'creation_order', CreationCounter._count)

a = TrackedItem("first", 1.0)
b = TrackedItem("second", 2.0)
c = TrackedItem("third", 3.0)

print(f"{a.name}: order={a.creation_order}")
print(f"{b.name}: order={b.creation_order}")
print(f"{c.name}: order={c.creation_order}")
```

```
# Output:
first: order=1
second: order=2
third: order=3
```

## Related Concepts

- Classes and Objects
- Type Hints
- Magic Methods

## Next Concepts

- Properties and Descriptors
- Generators and Iterators
- Context Managers

## Summary

Data classes, introduced in Python 3.7 via `@dataclass`, automatically generate `__init__`, `__repr__`, `__eq__`, and optional comparison methods from type-annotated fields. The `field()` function provides fine-grained control over defaults, comparison, and representation. `frozen=True` creates immutable instances. `__post_init__` enables validation and derived attributes. `asdict()` and `astuple()` convert nested data classes to plain data structures. Data classes are ideal for configuration, DTOs, and any class whose primary purpose is holding data.

## Key Takeaways

- `@dataclass` auto-generates `__init__`, `__repr__`, `__eq__` from type annotations
- Use `field(default_factory=list)` for mutable defaults, not `[]`
- `frozen=True` makes instances immutable
- `order=True` generates comparison methods (`<`, `<=`, `>`, `>=`)
- `__post_init__` runs after `__init__` for validation and derived fields
- `asdict()` recursively converts to dicts; `astuple()` converts to tuples
- Data classes reduce boilerplate, improve readability, and make intent clear
- They are the preferred way to model data in modern Python
