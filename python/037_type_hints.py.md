# Concept: Type Hints

## Concept ID

PYT-037

## Difficulty

Intermediate

## Domain

Python

## Module

Intermediate Python

## Learning Objectives

- Understand the purpose and benefits of type hints in Python
- Annotate variables, parameters, and return values with basic types
- Use generic types from the `typing` module: `List`, `Dict`, `Tuple`, `Set`
- Master optional types with `Optional`, `Union`, and `Any`
- Define callable signatures with `Callable`
- Create reusable type aliases with `TypeVar`
- Use structural subtyping with `Protocol`
- Constrain values with `Literal` and `TypedDict`
- Apply type hints to AI/ML model inputs and outputs

## Prerequisites

- Python functions and parameters
- Understanding of lists, dictionaries, and other collections
- Basic knowledge of object-oriented programming (classes, inheritance)

## Definition

Type hints (also called type annotations) are a formal syntax for declaring the expected types of variables, function parameters, and return values in Python. Introduced in PEP 484 and refined in subsequent PEPs, type hints enable static type checking, improve code documentation, and enhance IDE support — all while remaining optional at runtime.

Type hints do not enforce types at runtime. They serve as metadata that external tools like `mypy`, `pyright`, `pylance`, and IDEs use to analyze code before execution.

## Intuition

Think of type hints as documentation that machines can read. Instead of writing `# name is a string` in a comment, you write `name: str`. Instead of guessing what a function returns, you see `-> List[float]` in the signature. Type hints make your intentions explicit, catch bugs early, and make your code self-documenting.

## Why This Concept Matters

Type hints transform Python from a purely dynamic language into a gradually typed language. They are essential in large codebases where understanding function signatures from documentation alone is impractical. Type hints prevent entire categories of bugs — passing `None` when an object is expected, mixing up strings and integers, returning inconsistent types. In AI/ML projects, where data pipelines involve complex transformations, type hints clarify the shape and type of data at each stage.

## Real World Examples

- API endpoint signatures with request/response models
- Data processing pipelines with typed transformation steps
- Configuration schemas defined as `TypedDict`
- Library APIs for autocomplete and documentation generation
- Large-scale codebases at companies like Dropbox, Google, and Instagram

## AI/ML Relevance

Type hints are increasingly critical in AI/ML code:
- Annotating model `predict()` and `fit()` method signatures
- Typing numpy arrays as `ndarray` or using `ArrayLike`
- Defining `TypedDict` for model configuration dicts
- Typing dataset loaders with `Iterator[Batch]`
- Using `Protocol` to define duck-typed model interfaces
- Ensuring data pipeline steps receive and return expected shapes and types

## Code Examples

### Example 1: Basic function annotations

```python
def greet(name: str, age: int) -> str:
    return f'{name} is {age} years old.'

result: str = greet('Alice', 30)
print(result)

# Output:
# Alice is 30 years old.
```

### Example 2: Collections from typing

```python
from typing import List, Dict, Tuple, Set

def process_scores(scores: List[float]) -> Dict[str, float]:
    return {
        'mean': sum(scores) / len(scores),
        'max': max(scores),
        'min': min(scores),
    }

def lookup_coordinates() -> Tuple[float, float]:
    return (40.7128, -74.0060)

x: Tuple[float, float] = lookup_coordinates()
print(x)

# Output:
# (40.7128, -74.0060)
```

### Example 3: Optional, Union, Any

```python
from typing import Optional, Union, Any

def find_user(user_id: int) -> Optional[str]:
    db = {1: 'Alice', 2: 'Bob'}
    return db.get(user_id)

def format_value(value: Union[int, float, str]) -> str:
    return f'Value: {value}'

def log_message(message: Any) -> None:
    print(f'LOG: {message}')

print(find_user(1))
print(find_user(99))
print(format_value(42))

# Output:
# Alice
# None
# Value: 42
```

### Example 4: TypeVar and generic functions

```python
from typing import TypeVar, List

T = TypeVar('T')

def first_element(items: List[T]) -> T:
    return items[0]

def reverse_list(items: List[T]) -> List[T]:
    return items[::-1]

print(first_element([1, 2, 3]))
print(first_element(['a', 'b', 'c']))
print(reverse_list([10, 20, 30]))

# Output:
# 1
# a
# [30, 20, 10]
```

### Example 5: Callable type

```python
from typing import Callable

def apply_twice(func: Callable[[int], int], x: int) -> int:
    return func(func(x))

def square(n: int) -> int:
    return n * n

result = apply_twice(square, 3)
print(result)

# Output:
# 81
```

### Example 6: Protocol (structural subtyping)

```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> str:
        ...

class Circle:
    def draw(self) -> str:
        return 'Drawing a circle'

class Square:
    def draw(self) -> str:
        return 'Drawing a square'

def render(shape: Drawable) -> None:
    print(shape.draw())

render(Circle())
render(Square())

# Output:
# Drawing a circle
# Drawing a square
```

### Example 7: Literal and TypedDict

```python
from typing import Literal, TypedDict

def set_status(status: Literal['active', 'inactive', 'pending']) -> str:
    return f'Status set to {status}'

class UserConfig(TypedDict):
    name: str
    age: int
    email: str
    is_admin: bool

config: UserConfig = {
    'name': 'Alice',
    'age': 30,
    'email': 'alice@example.com',
    'is_admin': True,
}

print(set_status('active'))
print(config['name'])

# Output:
# Status set to active
# Alice
```

### Example 8: Typing for AI/ML

```python
from typing import List, Dict, Tuple, Optional
import numpy as np

class MLModel:
    def predict(self, features: np.ndarray) -> np.ndarray:
        return features @ self.weights + self.bias

TrainingBatch = Tuple[np.ndarray, np.ndarray]

def train_model(
    model: MLModel,
    data: List[TrainingBatch],
    learning_rate: float = 0.01,
    epochs: int = 10
) -> Dict[str, List[float]]:
    history: Dict[str, List[float]] = {'loss': [], 'accuracy': []}
    return history

print(type(TrainingBatch))
print(type(train_model))

# Output:
# <class 'typing.GenericAlias'>
# <class 'function'>
```

## Common Mistakes

1. Using type hints at runtime for validation — they are ignored by the interpreter
2. Forgetting to import types from `typing` module (e.g., `List` instead of `list` in Python < 3.9)
3. Over-annotating with `Any` everywhere, which defeats the purpose of type hints
4. Mixing up `Optional[X]` and `Union[X, None]` — they are equivalent
5. Annotating variables but never running a type checker to validate them

## Interview Questions

### Beginner - 5

1. What are type hints in Python and why were they introduced?
2. How do you annotate a function parameter and return type?
3. What is the difference between `list[int]` and `List[int]`?
4. What does `Optional[str]` mean?
5. How do you annotate a variable that can be either int or float?

### Intermediate - 5

1. Explain the difference between nominal typing and structural typing in Python type hints.
2. What is a `TypeVar` and when would you use it?
3. How does `Callable[[Arg1Type, Arg2Type], ReturnType]` work?
4. What is a `TypedDict` and how does it differ from a regular dictionary annotation?
5. How do you annotate a generator function?

### Advanced - 3

1. How do you use `overload` decorators to provide multiple type signatures for a single function?
2. Explain how variance works with `TypeVar` (covariant, contravariant, invariant).
3. How does `Self` type work in Python 3.11+ and how does it simplify class method annotations?

## Practice Problems

### Easy - 5

1. Write a function `add(a, b) -> int` with type hints that adds two integers.
2. Annotate a variable `names: ...` as a list of strings.
3. Write a function `get_first(items) -> T` with a generic type hint.
4. Annotate a function `process(data)` that can accept either str or bytes.
5. Write a function `nothing() -> None` and annotate it properly.

### Medium - 5

1. Write a type-annotated function that takes a list of floats and returns a dictionary mapping string keys to float values.
2. Create a `TypedDict` for a `Student` with name, age, and list of grades.
3. Write a generic function `pairs(a, b) -> List[Tuple[T, T]]` that zips two lists into pairs.
4. Use `Callable` to type a higher-order function that takes a transformation function and a list.
5. Write a type-annotated decorator that preserves the original function signature.

### Hard - 3

1. Implement a generic `Registry[T]` class using `Protocol` and `TypeVar` with proper variance.
2. Write a type-safe event emitter using `overload` and `Callable`.
3. Design and implement a fully typed dependency injection container with generic type resolution.

## Solutions

### Easy 1

```python
def add(a: int, b: int) -> int:
    return a + b

print(add(3, 5))

# Output:
# 8
```

### Easy 2

```python
names: list[str] = ['Alice', 'Bob', 'Charlie']
print(names)

# Output:
# ['Alice', 'Bob', 'Charlie']
```

### Medium 1

```python
from typing import Dict, List

def analyze(numbers: List[float]) -> Dict[str, float]:
    return {
        'sum': sum(numbers),
        'count': len(numbers),
        'average': sum(numbers) / len(numbers) if numbers else 0,
    }

result = analyze([10, 20, 30])
print(result)

# Output:
# {'sum': 60, 'count': 3, 'average': 20.0}
```

### Hard 1 (simplified)

```python
from typing import Protocol, TypeVar, Generic, List

T = TypeVar('T', covariant=True)

class Named(Protocol):
    name: str

class Registry(Generic[T]):
    def __init__(self) -> None:
        self._items: List[T] = []

    def register(self, item: T) -> None:
        self._items.append(item)

    def get_all(self) -> List[T]:
        return self._items

reg: Registry[Named] = Registry()
print('Registry ready')

# Output:
# Registry ready
```

## Related Concepts

- PEP 484, PEP 526, PEP 544, PEP 586, PEP 589
- Static type checking with mypy, pyright, pytype
- Pydantic for runtime type validation
- Dataclasses with type-annotated fields
- Mypy configuration and strict mode
- Type stubs (`.pyi` files)

## Next Concepts

- Protocols and structural subtyping (PYT-037 advanced)
- Descriptors (PYT-040)
- Dataclasses and attrs
- Pydantic for data validation

## Summary

Type hints bring static typing benefits to Python while preserving its dynamic nature. They improve code clarity, catch bugs at analysis time, enhance IDE support, and are essential for large-scale Python projects. With the `typing` module providing generic types, protocols, and advanced constructs, type hints can express complex type relationships clearly and precisely.

## Key Takeaways

- Type hints are optional metadata — they do not affect runtime behavior
- Basic annotations use `: type` for parameters and `-> type` for returns
- The `typing` module provides `List`, `Dict`, `Tuple`, `Set`, `Optional`, `Union`, and more
- `TypeVar` enables type-safe generic functions and classes
- `Protocol` enables structural subtyping (duck typing with type safety)
- `Literal` and `TypedDict` provide precise type constraints
- Always run a type checker like mypy to validate your hints
- Type hints are especially valuable in complex AI/ML data pipelines
