# Concept: Functions

## Concept ID

PYT-016

## Difficulty

Beginner

## Domain

Python

## Module

Functions and Modules

## Learning Objectives

- Understand how to define and call functions using the `def` keyword
- Master parameter passing and return values
- Write effective docstrings following PEP 257
- Use type hints to annotate function signatures
- Recognize functions as first-class objects in Python
- Identify and avoid the mutable default argument pitfall

## Prerequisites

- Basic Python syntax (variables, data types, conditionals, loops)
- Understanding of indentation and code blocks
- Familiarity with basic data structures (lists, dicts)

## Definition

A **function** in Python is a reusable block of organized, executable code that performs a specific task. Functions are defined using the `def` keyword, can accept input parameters, and optionally return values. In Python, functions are **first-class objects**, meaning they can be passed around as arguments, returned from other functions, and assigned to variables.

The general syntax is:

```python
def function_name(parameters):
    """Optional docstring."""
    # body
    return value
```

## Intuition

Think of a function like a recipe. The recipe has a name (function name), a list of ingredients (parameters), a set of instructions (the body), and an output (return value). Once written, you can reuse the recipe any number of times without rewriting the instructions. When you need to make a small variation, you change the ingredients rather than rewriting the whole procedure.

## Why This Concept Matters

Functions are the fundamental building blocks of readable, maintainable, and reusable code. Without functions, programs become monolithic, repetitive, and impossible to debug. In the context of AI/ML, every model architecture, loss function, training loop, and preprocessing pipeline is structured using functions. Understanding functions deeply is the prerequisite for understanding decorators, generators, and functional programming paradigms in Python.

## Real World Examples

1. **Web development**: A Flask route handler is a function decorated with `@app.route('/')`.
2. **Data science**: A pandas transformation like `df.apply(lambda x: x**2)` uses functions as arguments.
3. **Automation**: A script that renames files in a directory uses a function to encapsulate the renaming logic.
4. **Testing**: Unit test frameworks like pytest expect test functions named `test_*`.
5. **AI/ML**: A PyTorch model's `forward` method is a function that defines the computation from input to output.

## AI/ML Relevance

Functions are the backbone of all AI/ML code. Activation functions like ReLU, sigmoid, and tanh are simple Python functions. Custom loss functions for specialized training objectives are written as functions. Training loops, evaluation metrics, data loaders, and preprocessing pipelines are all structured as functions. Libraries like NumPy, PyTorch, and TensorFlow expose vectorized functions that operate on tensors. Understanding how to write efficient, well-typed functions is essential for any machine learning engineer.

## Code Examples

### Example 1: Basic function definition and return

```python
def add(a, b):
    return a + b

result = add(3, 5)
print(result)
# Output: 8
```

### Example 2: Function with default arguments (the mutable default gotcha)

```python
def append_to_list(value, target=[]):
    target.append(value)
    return target

print(append_to_list(1))
# Output: [1]
print(append_to_list(2))
# Output: [1, 2]  # The same list object is reused!

def append_to_list_fixed(value, target=None):
    if target is None:
        target = []
    target.append(value)
    return target

print(append_to_list_fixed(1))
# Output: [1]
print(append_to_list_fixed(2))
# Output: [2]  # New list each time
```

### Example 3: Type hints and docstrings (PEP 257)

```python
from typing import List, Optional

def calculate_mean(values: List[float]) -> Optional[float]:
    """Calculate the arithmetic mean of a list of numbers.

    Args:
        values: A list of floating-point numbers.

    Returns:
        The mean as a float, or None if the list is empty.
    """
    if not values:
        return None
    return sum(values) / len(values)

print(calculate_mean([1.0, 2.0, 3.0]))
# Output: 2.0
print(calculate_mean([]))
# Output: None
```

### Example 4: Functions as first-class objects

```python
def square(x):
    return x ** 2

def cube(x):
    return x ** 3

def apply(func, value):
    return func(value)

print(apply(square, 4))
# Output: 16
print(apply(cube, 4))
# Output: 64

# Assigning function to a variable
my_func = square
print(my_func(5))
# Output: 25
```

### Example 5: Activation functions as Python functions (AI/ML context)

```python
import math

def sigmoid(x: float) -> float:
    """Sigmoid activation function."""
    return 1.0 / (1.0 + math.exp(-x))

def relu(x: float) -> float:
    """ReLU (Rectified Linear Unit) activation function."""
    return max(0.0, x)

def tanh(x: float) -> float:
    """Hyperbolic tangent activation function."""
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def custom_loss(y_true: float, y_pred: float) -> float:
    """Custom loss: squared error with L2 penalty."""
    return (y_true - y_pred) ** 2 + 0.01 * y_pred ** 2

print(sigmoid(0.0))
# Output: 0.5
print(relu(-3.0))
# Output: 0.0
print(tanh(0.0))
# Output: 0.0
print(custom_loss(1.0, 0.8))
# Output: 0.0464
```

### Example 6: Nested functions and closures

```python
def make_multiplier(factor: float):
    """Return a function that multiplies by the given factor."""
    def multiplier(x: float) -> float:
        return x * factor
    return multiplier

double = make_multiplier(2.0)
triple = make_multiplier(3.0)

print(double(5))
# Output: 10.0
print(triple(5))
# Output: 15.0
```

### Example 7: Function annotations and return type

```python
from typing import Union

def divide(a: Union[int, float], b: Union[int, float]) -> float:
    """Divide two numbers safely."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

print(divide(10, 3))
# Output: 3.3333333333333335
```

## Common Mistakes

1. **Mutable default arguments**: Using `[]`, `{}`, or `set()` as default arguments causes shared state across function calls. Use `None` instead and initialize inside the function.
2. **Forgetting the return statement**: A function without `return` implicitly returns `None`.
3. **Modifying global variables inside functions**: Without the `global` keyword, assigning to a variable inside a function creates a local variable, shadowing the global.
4. **Confusing `return` with `print`**: Printing a value does not make it available to the caller; you must use `return`.
5. **Overcomplicating function signatures**: Too many parameters make functions hard to test and understand; prefer fewer, well-named parameters or use data classes.

## Interview Questions

### Beginner

1. What is the `def` keyword used for in Python?
2. How do you return a value from a function?
3. What happens if a function does not have a `return` statement?
4. Can you assign a function to a variable? Explain with an example.
5. What is a docstring and how do you access it programmatically?

### Intermediate

1. Explain the mutable default argument problem and how to fix it.
2. What are type hints and how do they improve code quality?
3. How do nested functions work? What is a closure?
4. Write a function that accepts another function as an argument and applies it to a list of numbers.
5. Explain the difference between `return` and `yield` in a function.

### Advanced

1. Implement a decorator that measures the execution time of a function.
2. How does Python resolve variable names in nested scopes using the LEGB rule?
3. Write a function that can accept an arbitrary number of keyword arguments and returns only those with string values.

## Practice Problems

### Easy

1. Write a function `is_even(n)` that returns `True` if `n` is even.
2. Write a function `greet(name)` that returns `"Hello, {name}!"`.
3. Write a function `max_of_three(a, b, c)` that returns the largest of three numbers.
4. Write a function `factorial(n)` that computes n! recursively.
5. Write a function `celsius_to_fahrenheit(c)` that converts temperature.

### Medium

1. Write a function `count_vowels(s)` that counts vowels in a string.
2. Write a function `flatten_list(nested)` that flattens a list of lists.
3. Write a function `fibonacci(n)` that returns the first n Fibonacci numbers using a generator.
4. Write a function `validate_email(email)` that validates an email address using regex and returns a boolean.
5. Write a function `time_it(func, *args)` that times how long a function takes to execute.

### Hard

1. Implement a decorator `@retry(max_attempts=3)` that retries a function if it raises an exception.
2. Write a function `compose(*funcs)` that returns the composition of an arbitrary number of functions.
3. Implement a function `memoize(fn)` that caches results of expensive function calls based on arguments.

## Solutions

### Easy Solutions

```python
# 1
def is_even(n): return n % 2 == 0

# 2
def greet(name): return f"Hello, {name}!"

# 3
def max_of_three(a, b, c): return max(a, b, c)

# 4
def factorial(n):
    if n <= 1: return 1
    return n * factorial(n - 1)

# 5
def celsius_to_fahrenheit(c): return (c * 9/5) + 32
```

### Medium Solutions

```python
# 1
def count_vowels(s):
    return sum(1 for ch in s.lower() if ch in "aeiou")

# 2
def flatten_list(nested):
    return [item for sublist in nested for item in sublist]

# 3
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# 4
import re
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

# 5
import time
def time_it(func, *args):
    start = time.perf_counter()
    result = func(*args)
    elapsed = time.perf_counter() - start
    return result, elapsed
```

### Hard Solutions

```python
# 1
import functools
import time

def retry(max_attempts=3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
            return None
        return wrapper
    return decorator

# 2
from functools import reduce
def compose(*funcs):
    def composed(arg):
        return reduce(lambda x, f: f(x), funcs, arg)
    return composed

# 3
def memoize(fn):
    cache = {}
    @functools.wraps(fn)
    def wrapper(*args):
        if args not in cache:
            cache[args] = fn(*args)
        return cache[args]
    return wrapper
```

## Related Concepts

- PYT-017: Arguments and Parameters
- PYT-018: Lambda Functions
- PYT-019: Scope and Namespace
- PYT-024: Decorators

## Next Concepts

- PYT-025: Generators (using yield instead of return)
- PYT-030: Exception Handling
- PYT-035: Decorators in Depth

## Summary

Functions are reusable blocks of code defined with `def`. They accept parameters, return values, and support docstrings and type hints. Python treats functions as first-class objects, enabling higher-order programming patterns. The mutable default argument pitfall occurs when using `[]` or `{}` as defaults; the fix is to use `None` and initialize inside the function. Functions are central to AI/ML for activations, losses, training loops, and data pipelines.

## Key Takeaways

- Use `def` to define, `return` to send back a value.
- Always use `None` instead of mutable objects as default arguments.
- Write docstrings following PEP 257 for documentation.
- Add type hints to improve readability and enable static analysis.
- Functions can be passed, returned, and assigned like any other object.
- Nested functions capture enclosing scope, forming closures.
