# Concept: Composition vs Inheritance

## Concept ID

PYT-034

## Difficulty

Intermediate

## Domain

Python

## Module

Object-Oriented Programming

## Learning Objectives

- Distinguish between "is-a" (inheritance) and "has-a" (composition) relationships
- Understand the principle "favor composition over inheritance"
- Implement delegation through composition
- Use `__getattr__` for automatic delegation
- Decide when to use inheritance vs composition
- Recognize the trade-offs of each approach

## Prerequisites

- Inheritance (PYT-028)
- Multiple Inheritance (PYT-033)
- Classes and objects (PYT-026)

## Definition

**Inheritance** models an "is-a" relationship: a `Dog` is an `Animal`. The child class extends or overrides the parent's behavior.

**Composition** models a "has-a" relationship: a `Car` has an `Engine`. The containing class delegates work to contained objects, which are passed in or created internally.

The principle "**favor composition over inheritance**" (from Gang of Four's Design Patterns) advises that composition is usually more flexible and less brittle than inheritance for code reuse.

```python
class Animal:
    def breathe(self): return "Breathing"
class Dog(Animal):
    def bark(self): return "Woof!"

class Engine:
    def start(self): return "Engine started"
class Car:
    def __init__(self):
        self.engine = Engine()
    def start(self): return self.engine.start()
```

## Intuition

Think of inheritance as being born into a family — you automatically get all the family traits, but you're stuck with the family tree. If your parent changes, you change too, potentially breaking things.

Composition is like hiring employees for your business. You pick exactly the skills you need (a `Logger`, a `Database`, a `Mailer`), combine them in your class, and if one service becomes problematic, you swap it out without affecting the others. You have full control over your interface.

## Why This Concept Matters

Choosing between composition and inheritance is one of the most important design decisions in OOP. Inheritance creates tight coupling — a change in the parent can break all subclasses. Composition creates loose coupling — objects interact through well-defined interfaces and can be easily replaced, tested, and extended. Most real-world frameworks recommend composition over inheritance for code reuse.

## Real World Examples

1. **Strategy pattern:** A `Sorter` class takes a `SortingStrategy` (QuickSort, MergeSort) via composition rather than subclassing.
2. **Django middleware:** Middleware classes are composed, not inherited. Each middleware component wraps the next.
3. **PyTorch `nn.Sequential`:** Composes layers sequentially; doesn't require inheritance.
4. **Python's `io` module:** `BufferedReader` composes a `RawIOBase` object, inheriting behavior without subclassing the raw reader.
5. **Logging:** A `Service` class composes a `Logger` object rather than inheriting from a `Logger` class.

## AI/ML Relevance

- PyTorch models compose layers (`nn.Linear`, `nn.Conv2d`) rather than inheriting from them.
- Feature extractors are composed: a pipeline chains a vectorizer, scaler, and PCA transform.
- Preprocessing pipelines use composition (each step is a separate transform object).
- Training loops compose a model, optimizer, loss function, and data loader.
- Ensemble models compose multiple base models via aggregation.

## Code Examples

### Example 1: Inheritance Tight Coupling

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

animals = [Dog("Rex"), Cat("Whiskers")]
for a in animals:
    print(f"{a.name} says {a.speak()}")
```

```
# Output:
Rex says Woof!
Whiskers says Meow!
```

### Example 2: Composition with Delegation

```python
class Engine:
    def start(self):
        return "Engine started"

    def stop(self):
        return "Engine stopped"

class Wheels:
    def rotate(self):
        return "Wheels rotating"

class Car:
    def __init__(self):
        self.engine = Engine()
        self.wheels = Wheels()

    def start(self):
        return self.engine.start()

    def drive(self):
        return f"{self.engine.start()}, {self.wheels.rotate()}"

c = Car()
print(c.start())
print(c.drive())
```

```
# Output:
Engine started
Engine started, Wheels rotating
```

### Example 3: Using `__getattr__` for Delegation

```python
class Logging:
    def log_error(self, msg):
        return f"ERROR: {msg}"

    def log_info(self, msg):
        return f"INFO: {msg}"

class Service:
    def __init__(self):
        self.logger = Logging()

    def __getattr__(self, name):
        # Delegate unknown attribute access to logger
        if hasattr(self.logger, name):
            return getattr(self.logger, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def do_work(self):
        return self.log_info("Doing work")

s = Service()
print(s.do_work())
print(s.log_error("Something broke"))
```

```
# Output:
INFO: Doing work
ERROR: Something broke
```

### Example 4: When Inheritance Makes Sense

```python
class PaymentMethod:
    def __init__(self, amount):
        self.amount = amount

    def process(self):
        raise NotImplementedError

class CreditCard(PaymentMethod):
    def process(self):
        return f"Processing ${self.amount} via Credit Card"

class PayPal(PaymentMethod):
    def process(self):
        return f"Processing ${self.amount} via PayPal"

def checkout(payment: PaymentMethod):
    return payment.process()

print(checkout(CreditCard(100)))
print(checkout(PayPal(50)))
```

```
# Output:
Processing $100 via Credit Card
Processing $50 via PayPal
```

### Example 5: AI/ML — Composing Model Components

```python
class LinearLayer:
    def __init__(self, in_features, out_features):
        self.weights = [0.5] * in_features * out_features
        self.bias = 0.0

    def forward(self, x):
        return sum(w * xi for w, xi in zip(self.weights[:len(x)], x)) + self.bias

class ReLU:
    def forward(self, x):
        return max(0, x)

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __call__(self, x):
        return self.forward(x)

model = Sequential([
    LinearLayer(3, 4),
    ReLU(),
    LinearLayer(4, 1)
])

result = model([1.0, 2.0, 3.0])
print(f"Output: {result:.2f}")
```

```
# Output:
Output: 3.00
```

### Example 6: Composition with Dependency Injection

```python
class Database:
    def save(self, data):
        return f"Saved: {data}"

class FileStorage:
    def save(self, data):
        return f"Wrote to file: {data}"

class CloudStorage:
    def save(self, data):
        return f"Uploaded to cloud: {data}"

class DataProcessor:
    def __init__(self, storage):
        self.storage = storage

    def process_and_save(self, data):
        processed = data.upper()
        return self.storage.save(processed)

# Swap storage backends easily
processor1 = DataProcessor(Database())
processor2 = DataProcessor(FileStorage())
processor3 = DataProcessor(CloudStorage())

print(processor1.process_and_save("hello"))
print(processor2.process_and_save("hello"))
print(processor3.process_and_save("hello"))
```

```
# Output:
Saved: HELLO
Wrote to file: HELLO
Uploaded to cloud: HELLO
```

## Common Mistakes

1. **Using inheritance for code reuse alone** — if there's no "is-a" relationship, use composition. `Stack` inheriting from `list` is a classic misuse.
2. **Creating deep inheritance hierarchies** — more than 2-3 levels makes code fragile and hard to debug.
3. **Breaking encapsulation via inheritance** — child classes depend on parent's internal implementation; a parent change can break children.
4. **Overusing `__getattr__` for composition** — it hides the actual interface, making code harder to understand and debug.
5. **Assuming composition is always better** — inheritance is the right choice for true "is-a" relationships where Liskov substitution holds.
6. **Forgetting to delegate `__init__` arguments** — when composing, ensure all components are properly initialized.
7. **Modifying composed objects from outside** — leaking internal objects breaks encapsulation; provide controlled access.
8. **Circular composition** — object A has B, B has A creates reference cycles and memory issues.

## Interview Questions

### Beginner

1. What is the difference between composition and inheritance?
2. What does "favor composition over inheritance" mean?
3. Give an example of an "is-a" relationship and a "has-a" relationship.
4. How do you delegate a method call in composition?
5. Can you use both composition and inheritance in the same class?

### Intermediate

1. When would you choose inheritance over composition?
2. What is the strategy pattern and how does it use composition?
3. How does `__getattr__` help with delegation in composition?
4. What are the disadvantages of deep inheritance hierarchies?
5. How does dependency injection relate to composition?

### Advanced

1. Explain the fragile base class problem.
2. How would you refactor a deep inheritance chain to use composition?
3. Implement a decorator pattern using composition.

## Practice Problems

### Easy

1. Create `Author` and `Book` classes using composition (Book has an Author).
2. Create a `Computer` class that composes a `CPU`, `RAM`, and `Storage`.
3. Create `Engineer` and `Manager` classes that both inherit from `Employee`.
4. Create a `Restaurant` class that composes multiple `MenuItem` objects.
5. Create a `Team` class that composes multiple `Player` objects.

### Medium

1. Implement a `Logger` class and compose it in a `BankAccount` class.
2. Refactor a `Vehicle` -> `Car` -> `ElectricCar` hierarchy into composition.
3. Build a `Notifier` that composes `EmailSender`, `SMSSender`, `PushSender`.
4. Create a `SortStrategy` abstract class and compose it in a `Sorter` class.
5. Implement a `Pipeline` class that composes multiple `Processor` objects.

### Hard

1. Implement the decorator pattern using composition for a `TextProcessor`.
2. Build a dependency injection container that resolves composed objects.
3. Refactor a complex multiple inheritance hierarchy to use composition with `__getattr__` delegation.

## Solutions

### Easy — Solution 1

```python
class Author:
    def __init__(self, name, email):
        self.name = name
        self.email = email

class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages

    def info(self):
        return f"'{self.title}' by {self.author.name} ({self.pages} pages)"

author = Author("George Orwell", "orwell@example.com")
book = Book("1984", author, 328)
print(book.info())
```

```
# Output:
'1984' by George Orwell (328 pages)
```

### Medium — Solution 1

```python
class Logger:
    def __init__(self, prefix=""):
        self.prefix = prefix

    def log(self, message, level="INFO"):
        print(f"[{level}] {self.prefix}{message}")

class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self._balance = balance
        self.logger = Logger(f"{owner}: ")

    @property
    def balance(self):
        return self._balance

    def deposit(self, amount):
        if amount <= 0:
            self.logger.log("Invalid deposit", "ERROR")
            return False
        self._balance += amount
        self.logger.log(f"Deposited ${amount}, balance=${self._balance}")
        return True

    def withdraw(self, amount):
        if amount <= 0 or amount > self._balance:
            self.logger.log("Withdrawal failed", "ERROR")
            return False
        self._balance -= amount
        self.logger.log(f"Withdrew ${amount}, balance=${self._balance}")
        return True

acc = BankAccount("Alice", 1000)
acc.deposit(500)
acc.withdraw(200)
```

```
# Output:
[INFO] Alice: Deposited $500, balance=$1500
[INFO] Alice: Withdrew $200, balance=$1300
```

### Hard — Solution 1

```python
class TextProcessor:
    def process(self, text):
        return text

class BoldDecorator:
    def __init__(self, processor):
        self.processor = processor

    def process(self, text):
        result = self.processor.process(text)
        return f"**{result}**"

class ItalicDecorator:
    def __init__(self, processor):
        self.processor = processor

    def process(self, text):
        result = self.processor.process(text)
        return f"*{result}*"

class UpperDecorator:
    def __init__(self, processor):
        self.processor = processor

    def process(self, text):
        result = self.processor.process(text)
        return result.upper()

base = TextProcessor()
bold = BoldDecorator(base)
italic = ItalicDecorator(base)
bold_italic = BoldDecorator(ItalicDecorator(base))
all_decorated = UpperDecorator(BoldDecorator(ItalicDecorator(base)))

print(bold.process("hello"))
print(italic.process("hello"))
print(bold_italic.process("hello"))
print(all_decorated.process("hello"))
```

```
# Output:
**hello**
*hello*
***hello***
** * HELLO **
```

## Related Concepts

- Inheritance
- Multiple Inheritance
- Design Patterns

## Next Concepts

- Data Classes
- Magic Methods
- Properties and Descriptors

## Summary

Inheritance ("is-a") and composition ("has-a") are two approaches to code reuse. Inheritance creates tight coupling and can lead to fragile base class problems, but models natural hierarchies well. Composition creates loose coupling, is more flexible, and supports dependency injection. The principle "favor composition over inheritance" guides developers to prefer composition unless a true "is-a" relationship exists. Use `__getattr__` for delegation in composition, but beware of hiding the interface.

## Key Takeaways

- Inheritance: "is-a" relationship, tight coupling, code reuse via extension
- Composition: "has-a" relationship, loose coupling, code reuse via delegation
- Favor composition over inheritance for most code reuse scenarios
- Use inheritance when Liskov substitution clearly holds (subtypes are substitutable)
- `__getattr__` enables automatic delegation in composition
- Composition supports dependency injection and runtime behavior swapping
- Deep inheritance hierarchies are fragile; deep composition trees are flexible
- Both approaches can and should be used together in well-designed systems
