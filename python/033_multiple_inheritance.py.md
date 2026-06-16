# Concept: Multiple Inheritance

## Concept ID

PYT-033

## Difficulty

Advanced

## Domain

Python

## Module

Object-Oriented Programming

## Learning Objectives

- Understand how multiple inheritance works in Python
- Recognize the diamond problem and how Python resolves it
- Understand MRO (Method Resolution Order) using C3 linearization
- Use `super()` correctly in multiple inheritance hierarchies
- Inspect `__mro__` attribute
- Design mixin classes for reusable behavior

## Prerequisites

- Inheritance (PYT-028)
- Abstract Classes (PYT-032)
- Understanding of `super()` and method resolution

## Definition

Multiple inheritance is when a class inherits from more than one parent class. Python supports multiple inheritance directly. The method resolution order (MRO) determines which parent's method is called when there's a conflict. Python uses the **C3 linearization algorithm** to compute a consistent MRO that preserves monotonicity and local precedence ordering.

```python
class A:
    def method(self): return "A"

class B(A):
    def method(self): return "B"

class C(A):
    def method(self): return "C"

class D(B, C):
    pass

print(D.mro())
print(D().method())
```

```
# Output:
[<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.A'>, <class 'object'>]
B
```

## Intuition

Imagine a family tree where a child has two parents. The child inherits traits from both. If both parents have a "cook" trait, which one does the child follow? Python solves this with MRO — it creates a consistent, linear order that respects the hierarchy. It's like saying "check the first parent and all their ancestors first, then the second parent and all their ancestors." This prevents ambiguity and keeps behavior predictable.

## Why This Concept Matters

Multiple inheritance is powerful but controversial. It allows for **mixin classes** — small, focused classes that add specific functionality (like serialization, logging, validation) without deep inheritance chains. Frameworks like Django use mixins extensively (`LoginRequiredMixin`, `PermissionRequiredMixin`). Understanding MRO is critical when using `super()` in complex hierarchies.

## Real World Examples

1. **Django class-based views:** `class MyView(LoginRequiredMixin, ListView):` — mixin adds authentication, ListView provides the view logic.
2. **Django models:** `class Product(SoftDeleteMixin, TimestampMixin, models.Model):` — each mixin adds specific fields/behavior.
3. **PyTorch:** `class MyModel(ResNetMixin, CustomHeadMixin, nn.Module):` — mixins add different architectural features.
4. **Logging mixins:** `class LoggedClass(LogMixin, SaveMixin, BaseClass):` — adds logging and save capability.
5. **REST framework:** mixins for Create, Read, Update, Delete operations combined in a single view.

## AI/ML Relevance

- Mixins for serialization (`TorchScriptMixin`, `ONNXExportMixin`) that add export capabilities to models.
- Mixins for logging (`MetricLoggerMixin`, `CheckpointMixin`) for training loops.
- Mixins for model validation (`CrossValidateMixin`, `HyperoptMixin`).
- Combining multiple feature extractors via multiple inheritance.
- `sklearn.base.ClassifierMixin`, `RegressorMixin`, `TransformerMixin` — all mixins that add default behavior.

## Code Examples

### Example 1: Basic Multiple Inheritance

```python
class Flyer:
    def fly(self):
        return "Flying"

class Swimmer:
    def swim(self):
        return "Swimming"

class Duck(Flyer, Swimmer):
    def quack(self):
        return "Quack!"

d = Duck()
print(d.fly())
print(d.swim())
print(d.quack())
print(Duck.mro())
```

```
# Output:
Flying
Swimming
Quack!
[<class '__main__.Duck'>, <class '__main__.Flyer'>, <class '__main__.Swimmer'>, <class 'object'>]
```

### Example 2: The Diamond Problem

```python
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B"

class C(A):
    def method(self):
        return "C"

class D(B, C):
    pass

d = D()
print(d.method())
print(D.__mro__)
```

```
# Output:
B
(<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.A'>, <class 'object'>)
```

### Example 3: `super()` in Multiple Inheritance

```python
class A:
    def __init__(self):
        print("A.__init__")

class B(A):
    def __init__(self):
        print("B.__init__")
        super().__init__()

class C(A):
    def __init__(self):
        print("C.__init__")
        super().__init__()

class D(B, C):
    def __init__(self):
        print("D.__init__")
        super().__init__()

d = D()
print(D.__mro__)
```

```
# Output:
D.__init__
B.__init__
C.__init__
A.__init__
(<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.A'>, <class 'object'>)
```

### Example 4: Mixin Classes

```python
class LogMixin:
    def log(self, message):
        print(f"[LOG] {self.__class__.__name__}: {message}")

class TimestampMixin:
    def __init__(self, *args, **kwargs):
        import datetime
        self.created_at = datetime.datetime.now()
        super().__init__(*args, **kwargs)

class SerializeMixin:
    def to_dict(self):
        return self.__dict__

class User(LogMixin, TimestampMixin, SerializeMixin):
    def __init__(self, username, email):
        self.username = username
        self.email = email
        super().__init__()

user = User("alice", "alice@example.com")
user.log("User created")
print(user.to_dict())
print(user.created_at)
```

```
# Output:
[LOG] User: User created
{'username': 'alice', 'email': 'alice@example.com', 'created_at': datetime.datetime(...)}
...
```

### Example 5: AI/ML — Mixins for Model Features

```python
class CheckpointMixin:
    def save_checkpoint(self, path):
        import json
        data = {"state": self.__dict__, "class": self.__class__.__name__}
        with open(path, "w") as f:
            json.dump(data, f, default=str)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        import json
        with open(path) as f:
            data = json.load(f)
        self.__dict__.update(data["state"])
        print(f"Checkpoint loaded from {path}")

class LogMetricsMixin:
    def log_metrics(self, metrics):
        log_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"[METRICS] {log_str}")

class Trainer(CheckpointMixin, LogMetricsMixin):
    def __init__(self, model_name):
        self.model_name = model_name
        self.epoch = 0
        self.losses = []

    def train_epoch(self, loss):
        self.epoch += 1
        self.losses.append(loss)
        self.log_metrics({"epoch": self.epoch, "loss": loss})

trainer = Trainer("my_model")
trainer.train_epoch(0.5)
trainer.train_epoch(0.3)
trainer.save_checkpoint("checkpoint.json")
trainer.log_metrics({"accuracy": 0.95, "f1": 0.93})

import os
os.remove("checkpoint.json")
```

```
# Output:
[METRICS] epoch=1 | loss=0.5000
[METRICS] epoch=2 | loss=0.3000
Checkpoint saved to checkpoint.json
[METRICS] accuracy=0.9500 | f1=0.9300
```

### Example 6: Inspecting MRO and Method Resolution

```python
class Base:
    def identify(self):
        return "Base"

class MixinA:
    def identify(self):
        return "MixinA"

class MixinB:
    def identify(self):
        return "MixinB"

class Concrete(MixinA, Base, MixinB):
    pass

c = Concrete()
print(c.identify())
print(Concrete.__mro__)
for cls in Concrete.__mro__:
    if 'identify' in cls.__dict__:
        print(f"  {cls.__name__} defines identify")
```

```
# Output:
MixinA
(<class '__main__.Concrete'>, <class '__main__.MixinA'>, <class '__main__.Base'>, <class '__main__.MixinB'>, <class 'object'>)
  Concrete defines identify
  MixinA defines identify
  Base defines identify
  MixinB defines identify
```

## Common Mistakes

1. **Assuming `super()` always calls the direct parent** — in multiple inheritance, `super()` follows MRO, which may skip the direct parent.
2. **Creating deep diamond hierarchies** — too many layers make MRO hard to reason about.
3. **Using multiple inheritance when composition would be simpler** — many "mixin" use cases are better served by composition.
4. **Inconsistent `__init__` signatures across parents** — if parents have different `__init__` parameters, calling `super().__init__()` with wrong args causes errors.
5. **Forgetting to call `super().__init__()` in mixins** — mixins that don't call `super()` break the chain, preventing subsequent classes from initializing.
6. **Overriding methods ambiguously without understanding MRO** — you may think you're calling one parent's method but are actually getting another.
7. **Circular inheritance** — Python detects this at class creation time and raises `TypeError`.
8. **Mutating shared class attributes** — if both parents define the same class attribute, the MRO determines which one is used, which can be surprising.

## Interview Questions

### Beginner

1. What is multiple inheritance?
2. How do you define a class that inherits from two parents?
3. What is the diamond problem?
4. How can you check the method resolution order of a class?
5. What is a mixin class?

### Intermediate

1. How does Python resolve the diamond problem?
2. What is C3 linearization?
3. How does `super()` work in multiple inheritance?
4. Why must mixins call `super().__init__()`?
5. What is the difference between `__mro__` and `mro()`?

### Advanced

1. Explain the C3 linearization algorithm step by step.
2. What is monotonicity in MRO and why does it matter?
3. How would you design a class hierarchy to avoid the diamond problem entirely?

## Practice Problems

### Easy

1. Create `Walker` and `Jumper` classes with `move()` methods. Create a `Kangaroo` class that inherits both.
2. Create `Reader` and `Writer` mixins and a `FileHandler` class that uses both.
3. Create `A`, `B(A)`, `C(A)`, `D(B, C)` and print MRO.
4. Add a `greet()` method to each class in a diamond and see which is called.
5. Create mixins `JSONMixin` and `XMLMixin` each with a `serialize()` method.

### Medium

1. Create a `ValidatableMixin` that adds `validate()` and a `SavableMixin` that adds `save()`. Combine in `Model`.
2. Implement `SerializableMixin` with `to_json()` and `LogMixin` with `log()`. Create a `User` class that uses both.
3. Create a diamond with `__init__` taking different parameters and use `super()` correctly.
4. Build a `PermissionMixin` and `CacheMixin` for a web view class.
5. Create a `TimerMixin` that tracks elapsed time for any method call.

### Hard

1. Implement a cooperative multiple inheritance hierarchy with 4+ levels where all `__init__` calls succeed.
2. Build a plugin system where mixins register themselves into a central registry via MRO inspection.
3. Implement an `@implements` decorator that checks at class creation whether a class satisfies all abstract methods from multiple parent classes.

## Solutions

### Easy — Solution 1

```python
class Walker:
    def move(self):
        return "Walking"

class Jumper:
    def move(self):
        return "Jumping"

class Kangaroo(Walker, Jumper):
    pass

k = Kangaroo()
print(k.move())
print(Kangaroo.__mro__)
```

```
# Output:
Walking
(<class '__main__.Kangaroo'>, <class '__main__.Walker'>, <class '__main__.Jumper'>, <class 'object'>)
```

### Medium — Solution 1

```python
class ValidatableMixin:
    def validate(self):
        errors = []
        for field, rules in getattr(self, 'validations', {}).items():
            value = getattr(self, field, None)
            for rule in rules:
                error = rule(field, value)
                if error:
                    errors.append(error)
        return errors

class SavableMixin:
    def save(self):
        if hasattr(self, 'validate'):
            errors = self.validate()
            if errors:
                raise ValueError(f"Validation failed: {errors}")
        print(f"Saving {self.__class__.__name__}: {self.__dict__}")

class Model(ValidatableMixin, SavableMixin):
    validations = {}

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class User(Model):
    validations = {
        "email": [
            lambda f, v: f"{f} is required" if not v else None,
        ],
        "age": [
            lambda f, v: f"{f} must be positive" if v is not None and v <= 0 else None,
        ],
    }

user = User(email="alice@example.com", age=25)
print(user.validate())
user.save()

bad_user = User(email="", age=0)
print(bad_user.validate())
```

```
# Output:
[]
Saving User: {'email': 'alice@example.com', 'age': 25}
[None, 'age must be positive']
```

### Hard — Solution 1

```python
class A:
    def __init__(self, **kwargs):
        print(f"A.__init__")
        self.a_value = kwargs.get('a_value', 'default_a')
        super().__init__(**kwargs)

class B(A):
    def __init__(self, **kwargs):
        print(f"B.__init__")
        self.b_value = kwargs.get('b_value', 'default_b')
        super().__init__(**kwargs)

class C(A):
    def __init__(self, **kwargs):
        print(f"C.__init__")
        self.c_value = kwargs.get('c_value', 'default_c')
        super().__init__(**kwargs)

class D(B, C):
    def __init__(self, **kwargs):
        print(f"D.__init__")
        self.d_value = kwargs.get('d_value', 'default_d')
        super().__init__(**kwargs)

obj = D(a_value="A1", b_value="B1", c_value="C1", d_value="D1")
print(f"Values: {obj.__dict__}")
print(D.__mro__)
```

```
# Output:
D.__init__
B.__init__
C.__init__
A.__init__
Values: {'d_value': 'D1', 'b_value': 'B1', 'c_value': 'C1', 'a_value': 'A1'}
(<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.A'>, <class 'object'>)
```

## Related Concepts

- Inheritance
- Abstract Classes
- Composition vs Inheritance

## Next Concepts

- Composition vs Inheritance
- Magic Methods
- Data Classes

## Summary

Multiple inheritance allows a class to inherit from multiple parent classes. Python resolves method conflicts using the MRO (Method Resolution Order), computed via C3 linearization. The `super()` function follows MRO, enabling cooperative multiple inheritance when all classes in the hierarchy use it consistently. Mixins are small, focused classes designed to add specific functionality via multiple inheritance. While powerful, multiple inheritance should be used judiciously — composition is often a simpler alternative.

## Key Takeaways

- Python supports multiple inheritance with `class Child(Parent1, Parent2):`
- The diamond problem is resolved by C3 linearization MRO
- `__mro__` shows the method resolution order
- `super()` follows MRO, not just the direct parent
- Mixins should always call `super().__init__()` for cooperative initialization
- Use mixins for focused, reusable behaviors (logging, serialization, validation)
- Prefer composition over multiple inheritance when possible
- Keep hierarchies shallow to avoid confusing MRO
