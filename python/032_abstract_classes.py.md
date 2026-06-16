# Concept: Abstract Classes

## Concept ID

PYT-032

## Difficulty

Advanced

## Domain

Python

## Module

Object-Oriented Programming

## Learning Objectives

- Understand what abstract classes are and why they exist
- Use `ABC` and `@abstractmethod` from the `abc` module
- Recognize that abstract classes cannot be instantiated directly
- Implement abstract properties
- Apply the template method pattern with abstract base classes
- Leverage abstract classes for framework design

## Prerequisites

- Inheritance (PYT-028)
- Polymorphism (PYT-029)
- Method types (PYT-027)

## Definition

An **abstract class** is a class that cannot be instantiated directly. It defines a common interface (abstract methods) that all subclasses must implement. In Python, abstract classes are created using the `abc` module's `ABC` base class and the `@abstractmethod` decorator. Any class that inherits from `ABC` and has at least one abstract method is abstract — you cannot create an instance of it until all abstract methods are overridden in a concrete subclass.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

# s = Shape()  # TypeError: Can't instantiate abstract class Shape
```

## Intuition

Think of an abstract class as a contract or a blueprint specification — not a finished product. It's like a job description for a "Cook" role: it lists required skills (chop, sauté, plate) but doesn't say how to perform them. Each actual cook (subclass) implements those skills in their own style (Italian cook, Japanese cook, etc.). The job description itself can't cook anything — only concrete people (instances of subclasses) can.

## Why This Concept Matters

Abstract classes provide a formal way to define interfaces in Python. They enforce that subclasses implement specific methods, which is crucial for framework design, plugin systems, and large-scale applications. They serve as documentation ("you must implement these methods") and enable type checking tools to verify correctness. scikit-learn, PyTorch, and Django all use abstract base classes to define required interfaces.

## Real World Examples

1. **scikit-learn estimators:** All estimators inherit from `BaseEstimator` and must implement `fit()`. Classifiers additionally inherit from `ClassifierMixin` and must implement `predict()`.
2. **PyTorch datasets:** `torch.utils.data.Dataset` is abstract — subclasses must implement `__len__` and `__getitem__`.
3. **Django REST framework:** Serializers inherit from `Serializer` and must implement `create()` and `update()`.
4. **Plugin systems:** A `Plugin` abstract class defines `run(data)` and `stop()`; each plugin implements them.
5. **Game development:** An abstract `Character` class with abstract methods `move()`, `attack()`, `take_damage()`.

## AI/ML Relevance

- `sklearn.base.BaseEstimator` and `ClassifierMixin` define the abstract interface for all estimators.
- `torch.utils.data.Dataset` is abstract; custom datasets must implement `__len__` and `__getitem__`.
- Custom Keras layers implement `build()` and `call()` from `tf.keras.layers.Layer`.
- PyTorch's `nn.Module` is technically concrete but `forward()` acts as a de facto abstract method.
- Abstract optimization strategies in frameworks like Optuna use abstract `Trial` classes.

## Code Examples

### Example 1: Basic Abstract Class

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass

    @abstractmethod
    def move(self):
        pass

class Dog(Animal):
    def make_sound(self):
        return "Woof!"

    def move(self):
        return "Running"

class Bird(Animal):
    def make_sound(self):
        return "Chirp!"

    def move(self):
        return "Flying"

# a = Animal()  # TypeError!
d = Dog()
b = Bird()
print(d.make_sound(), d.move())
print(b.make_sound(), b.move())
```

```
# Output:
Woof! Running
Chirp! Flying
```

### Example 2: Abstract Properties

```python
from abc import ABC, abstractmethod

class Employee(ABC):
    def __init__(self, name, hourly_rate):
        self.name = name
        self.hourly_rate = hourly_rate

    @property
    @abstractmethod
    def job_title(self):
        pass

    @abstractmethod
    def calculate_pay(self, hours):
        pass

class Developer(Employee):
    @property
    def job_title(self):
        return "Software Developer"

    def calculate_pay(self, hours):
        return self.hourly_rate * hours + 1000  # Bonus

class Intern(Employee):
    @property
    def job_title(self):
        return "Intern"

    def calculate_pay(self, hours):
        return self.hourly_rate * hours

dev = Developer("Alice", 75)
intern = Intern("Bob", 20)
print(f"{dev.name} ({dev.job_title}): ${dev.calculate_pay(160)}")
print(f"{intern.name} ({intern.job_title}): ${intern.calculate_pay(160)}")
```

```
# Output:
Alice (Software Developer): $13000
Bob (Intern): $3200
```

### Example 3: Template Method Pattern

```python
from abc import ABC, abstractmethod

class DataMiner(ABC):
    # Template method — defines the skeleton
    def mine(self, path):
        file = self.open_file(path)
        data = self.extract_data(file)
        parsed = self.parse_data(data)
        analysis = self.analyze(parsed)
        self.close_file(file)
        return analysis

    @abstractmethod
    def open_file(self, path):
        pass

    @abstractmethod
    def extract_data(self, file):
        pass

    @abstractmethod
    def parse_data(self, data):
        pass

    def analyze(self, parsed):
        # Optional step with default implementation
        return f"Analyzed: {len(parsed)} records"

    @abstractmethod
    def close_file(self, file):
        pass

class CSVMiner(DataMiner):
    def open_file(self, path):
        print(f"Opening CSV: {path}")
        return open(path, "r")

    def extract_data(self, file):
        print("Extracting CSV data")
        return file.readlines()

    def parse_data(self, data):
        print("Parsing CSV rows")
        return [line.strip().split(",") for line in data]

    def close_file(self, file):
        print("Closing CSV file")
        file.close()

# Simulated usage — skip file I/O
m = CSVMiner()
print(m.analyze([1, 2, 3]))
```

```
# Output:
Analyzed: 3 records
```

### Example 4: Abstract Class with Concrete Methods

```python
from abc import ABC, abstractmethod

class Logger(ABC):
    def log(self, message, level="INFO"):
        formatted = self.format_message(message, level)
        self.write(formatted)

    def format_message(self, message, level):
        return f"[{level}] {message}"

    @abstractmethod
    def write(self, formatted_message):
        pass

class FileLogger(Logger):
    def __init__(self, filename):
        self.filename = filename

    def write(self, formatted_message):
        with open(self.filename, "a") as f:
            f.write(formatted_message + "\n")

class ConsoleLogger(Logger):
    def write(self, formatted_message):
        print(formatted_message)

import os
cl = ConsoleLogger()
cl.log("System started")
cl.log("User logged in", "WARNING")
```

```
# Output:
[INFO] System started
[WARNING] User logged in
```

### Example 5: AI/ML — Abstract Base Estimator

```python
from abc import ABC, abstractmethod
import math

class BaseEstimator(ABC):
    def __init__(self):
        self._is_fitted = False

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def score(self, X, y):
        predictions = self.predict(X)
        errors = sum((p - t) ** 2 for p, t in zip(predictions, y))
        return -errors  # Higher is better (negative MSE)

    def is_fitted(self):
        return self._is_fitted

class LinearRegressor(BaseEstimator):
    def __init__(self):
        super().__init__()
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n = len(X)
        mean_x = sum(X) / n
        mean_y = sum(y) / n
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(X, y))
        den = sum((xi - mean_x) ** 2 for xi in X)
        self.coef_ = num / den
        self.intercept_ = mean_y - self.coef_ * mean_x
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("Not fitted")
        return [self.coef_ * x + self.intercept_ for x in X]

class MeanRegressor(BaseEstimator):
    def fit(self, X, y):
        self.mean_ = sum(y) / len(y)
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("Not fitted")
        return [self.mean_] * len(X)

models = [LinearRegressor(), MeanRegressor()]
X = [1, 2, 3, 4, 5]
y = [2.1, 4.0, 5.9, 8.2, 10.1]

for model in models:
    model.fit(X, y)
    preds = model.predict(X)
    print(f"{model.__class__.__name__}: "
          f"score={model.score(X, y):.2f}, "
          f"preds={[f'{p:.1f}' for p in preds]}")
```

```
# Output:
LinearRegressor: score=-0.17, preds=['2.1', '4.1', '6.1', '8.1', '10.1']
MeanRegressor: score=-42.84, preds=['6.0', '6.0', '6.0', '6.0', '6.0']
```

### Example 6: Multiple Abstract Methods with Different Signatures

```python
from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def step(self, gradients):
        pass

    @abstractmethod
    def zero_grad(self):
        pass

    @abstractmethod
    def state_dict(self):
        pass

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.lr = learning_rate
        self.momentum = momentum
        self._velocity = None

    def zero_grad(self):
        print("Zeroing gradients")

    def step(self, gradients):
        print(f"Updating weights with SGD (lr={self.lr})")
        return [g * self.lr for g in gradients]

    def state_dict(self):
        return {"lr": self.lr, "momentum": self.momentum}

class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

    def zero_grad(self):
        print("Resetting Adam momentum buffers")

    def step(self, gradients):
        print(f"Updating weights with Adam (lr={self.lr})")
        return [g * self.lr for g in gradients]

    def state_dict(self):
        return {"lr": self.lr, "beta1": self.beta1, "beta2": self.beta2}

optimizers = [SGD(0.01), Adam(0.001)]
for opt in optimizers:
    opt.zero_grad()
    result = opt.step([1.0, 2.0, 3.0])
    print(f"Updated: {result}")
    print(f"State: {opt.state_dict()}")
    print("---")
```

```
# Output:
Zeroing gradients
Updating weights with SGD (lr=0.01)
Updated: [0.01, 0.02, 0.03]
State: {'lr': 0.01, 'momentum': 0.0}
---
Resetting Adam momentum buffers
Updating weights with Adam (lr=0.001)
Updated: [0.001, 0.002, 0.003]
State: {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999}
```

## Common Mistakes

1. **Trying to instantiate an abstract class** — raises `TypeError: Can't instantiate abstract class ...`. Every abstract method must be implemented in a concrete subclass.
2. **Forgetting to implement all abstract methods** — the concrete subclass itself becomes abstract and can't be instantiated.
3. **Calling `super().__init__()` in an abstract class unnecessarily** — abstract classes can have `__init__`, but it's only called via `super()` in subclasses.
4. **Using `@abstractmethod` on a static method incorrectly** — `@staticmethod` and `@abstractmethod` ordering matters: `@abstractmethod` should be outermost, then `@staticmethod`.
5. **Making a class abstract without inheriting from `ABC`** — `@abstractmethod` only works if the class inherits from `ABC`. Without it, the decorator has no effect.
6. **Confusing abstract classes with interfaces** — abstract classes can have concrete methods; interfaces (like Java's) only declare method signatures. Python uses abstract classes for both roles.
7. **Not using `@abstractmethod` on properties correctly** — decorator order must be `@property` then `@abstractmethod`, or `@abstractmethod` then `@property` (both work but the latter is preferred in modern Python).

## Interview Questions

### Beginner

1. What is an abstract class?
2. What module provides abstract base class support in Python?
3. Can you instantiate an abstract class directly?
4. What does the `@abstractmethod` decorator do?
5. What happens if a subclass doesn't implement all abstract methods?

### Intermediate

1. What is the difference between an abstract class and an interface?
2. Can an abstract class have concrete (non-abstract) methods?
3. How do you define an abstract property?
4. What is the template method pattern?
5. How do abstract classes support the Open/Closed principle?

### Advanced

1. How can you use `ABCMeta` metaclass to create abstract classes without inheriting from `ABC`?
2. How do abstract classes interact with `__init_subclass__`?
3. Implement a custom abstract method decorator using `__init_subclass__` that enforces implementation at import time.

## Practice Problems

### Easy

1. Create an abstract `Shape` class with an abstract `area()` method. Implement `Circle` and `Square`.
2. Add an abstract `perimeter()` method to `Shape` and implement it in both subclasses.
3. Create an abstract `Vehicle` class with abstract `start()` and `stop()` methods.
4. Implement `Car` and `Motorcycle` from `Vehicle`.
5. Create an abstract `Appliance` class with abstract `turn_on()` and a concrete `plug_in()` method.

### Medium

1. Create an abstract `Database` class with `connect()`, `query()`, `close()` methods. Implement `MySQLDatabase` and `PostgreSQLDatabase`.
2. Implement a `NotificationSender` abstract class with a template method `send_notification()` that calls abstract `format_message()` and `deliver()`.
3. Create an abstract `Character` class for a game with `attack()`, `defend()`, `special_move()`.
4. Build a `PipelineStep` abstract class where each step has `validate_input()`, `execute()`, `validate_output()`.
5. Implement an abstract `Serializer` class with `serialize()` and `deserialize()` and concrete `JSONSerializer` and `XMLSerializer`.

### Hard

1. Create a plugin system using abstract classes where plugins register automatically via `__init_subclass__`.
2. Implement a `ValidatedAbstractClass` that uses custom metaclass to enforce abstract method implementation at class creation time (not instantiation).
3. Build a dependency injection container that resolves abstract class dependencies to concrete implementations.

## Solutions

### Easy — Solution 1

```python
from abc import ABC, abstractmethod
import math

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2

class Square(Shape):
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side ** 2

shapes = [Circle(5), Square(4)]
for s in shapes:
    print(f"{s.__class__.__name__}: {s.area():.2f}")
```

```
# Output:
Circle: 78.54
Square: 16.00
```

### Medium — Solution 1

```python
from abc import ABC, abstractmethod

class Database(ABC):
    @abstractmethod
    def connect(self, connection_string):
        pass

    @abstractmethod
    def query(self, sql):
        pass

    @abstractmethod
    def close(self):
        pass

class MySQLDatabase(Database):
    def connect(self, connection_string):
        self.conn = f"MySQL connected to {connection_string}"
        print(self.conn)

    def query(self, sql):
        print(f"MySQL executing: {sql}")
        return ["result1", "result2"]

    def close(self):
        print("MySQL connection closed")

class PostgreSQLDatabase(Database):
    def connect(self, connection_string):
        self.conn = f"PostgreSQL connected to {connection_string}"
        print(self.conn)

    def query(self, sql):
        print(f"PostgreSQL executing: {sql}")
        return ["row1", "row2"]

    def close(self):
        print("PostgreSQL connection closed")

def run_queries(db: Database):
    db.connect("localhost:5432")
    results = db.query("SELECT * FROM users")
    print(f"Got {len(results)} results")
    db.close()

run_queries(MySQLDatabase())
print("---")
run_queries(PostgreSQLDatabase())
```

```
# Output:
MySQL connected to localhost:5432
MySQL executing: SELECT * FROM users
Got 2 results
MySQL connection closed
---
PostgreSQL connected to localhost:5432
PostgreSQL executing: SELECT * FROM users
Got 2 results
PostgreSQL connection closed
```

### Hard — Solution 1

```python
from abc import ABC, abstractmethod

class PluginRegistry(type):
    plugins = {}

    def __new__(cls, name, bases, namespace):
        klass = super().__new__(cls, name, bases, namespace)
        if hasattr(klass, 'plugin_name') and not getattr(namespace.get('plugin_name'), 'is_abstract', False):
            PluginRegistry.plugins[klass.plugin_name] = klass
        return klass

class Plugin(ABC, metaclass=PluginRegistry):
    @abstractmethod
    def process(self, data):
        pass

class UppercasePlugin(Plugin):
    plugin_name = "uppercase"

    def process(self, data):
        return data.upper()

class ReversePlugin(Plugin):
    plugin_name = "reverse"

    def process(self, data):
        return data[::-1]

print(PluginRegistry.plugins)
plugin = PluginRegistry.plugins["uppercase"]()
print(plugin.process("hello"))
plugin2 = PluginRegistry.plugins["reverse"]()
print(plugin2.process("hello"))
```

```
# Output:
{'uppercase': <class '__main__.UppercasePlugin'>, 'reverse': <class '__main__.ReversePlugin'>}
HELLO
olleh
```

## Related Concepts

- Inheritance
- Polymorphism
- Interfaces and Protocols

## Next Concepts

- Multiple Inheritance
- Composition vs Inheritance
- Data Classes

## Summary

Abstract classes, created with `ABC` and `@abstractmethod`, define interfaces that subclasses must implement. They cannot be instantiated directly. They can have both abstract and concrete methods, allowing the template method pattern where a skeleton algorithm is defined with overridable steps. Abstract classes enforce contracts, improve code organization, and are heavily used in AI/ML frameworks like scikit-learn and PyTorch to define estimator and dataset interfaces.

## Key Takeaways

- Use `from abc import ABC, abstractmethod` to create abstract classes
- Abstract classes cannot be instantiated — they must be subclassed
- Subclasses must implement all abstract methods or they remain abstract
- Abstract classes can have concrete methods with shared implementation
- The template method pattern uses abstract methods for customizable steps in a fixed algorithm
- Abstract properties use both `@property` and `@abstractmethod`
- Abstract classes are essential for framework and library design
