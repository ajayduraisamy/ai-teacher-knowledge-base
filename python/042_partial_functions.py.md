# Concept: Partial Functions

## Concept ID

PYT-042

## Difficulty

Advanced

## Domain

Python

## Module

Advanced Python

## Learning Objectives

- Understand what partial functions are and when to use them
- Use `functools.partial` to fix arguments of existing functions
- Distinguish between partial functions and lambda expressions
- Apply partial functions to methods using `partialmethod`
- Recognize common patterns and best practices
- Leverage partials in AI/ML workflows

## Prerequisites

- Functions and function arguments (positional, keyword, default)
- First-class functions (assigning functions to variables)
- Basic understanding of the `functools` module
- Lambda functions (helpful for comparison)

## Definition

A **partial function** is a function derived from an existing function by fixing (pre-filling) some of its arguments, producing a new function with fewer parameters. `functools.partial` is a higher-order function that takes a callable and a set of arguments, returning a new callable with those arguments bound.

Partial functions are a form of function specialization — they "remember" some arguments so you don't have to repeat them in every call.

## Intuition

Think of a partial function as a function with some arguments "already dialed in." Imagine a coffee machine that takes (beans, water, temperature). A partial function might pre-set the temperature to 95°C, giving you a new function that only needs beans and water. You're not changing the coffee machine — you're creating a specialized version of it.

## Why This Concept Matters

Partial functions reduce code repetition, make code more readable, and enable function composition patterns. They are particularly useful when passing functions to higher-order functions like `map`, `filter`, and `sorted` where you need to lock in some arguments. Partials also bridge the gap between APIs that take different numbers of arguments, enabling adapters and callbacks with customized behavior.

## Real World Examples

- Pre-configuring API client functions with authentication tokens
- Creating specialized mathematical functions (e.g., base-2 log from generic log)
- Fixing sort keys with custom comparators
- Binding database connection parameters to query functions
- Creating reusable preprocessing pipelines in data science
- Adapter patterns in event-driven architectures

## AI/ML Relevance

Partial functions in AI/ML workflows:
- Pre-configuring loss functions with specific parameters
- Creating specialized activation functions with fixed hyperparameters
- Binding model save paths and logging configurations
- Pre-filling data loader arguments for different datasets
- Creating multiple optimizer variants from a single optimizer factory

## Code Examples

### Example 1: Basic partial function

```python
from functools import partial

def power(base, exp):
    return base ** exp

square = partial(power, exp=2)
cube = partial(power, exp=3)

print(square(5))
print(cube(3))
print(power(2, 10))

# Output:
# 25
# 27
# 1024
```

### Example 2: Partial with positional arguments

```python
from functools import partial

def multiply(a, b, c):
    return a * b * c

double_and_more = partial(multiply, 2)
print(double_and_more(3, 5))

fixed_first_two = partial(multiply, 2, 3)
print(fixed_first_two(5))

# Output:
# 30
# 30
```

### Example 3: Partial for file processing

```python
from functools import partial

def process_line(line, prefix, suffix):
    return f'{prefix}: {line.strip()} {suffix}'

with_log = partial(process_line, prefix='LOG', suffix='[END]')

lines = ['First entry', 'Second entry', 'Third entry']
for line in lines:
    print(with_log(line))

# Output:
# LOG: First entry [END]
# LOG: Second entry [END]
# LOG: Third entry [END]
```

### Example 4: Partial vs lambda comparison

```python
from functools import partial

def greet(greeting, name):
    return f'{greeting}, {name}!'

# Using partial
say_hello = partial(greet, 'Hello')
say_hi = partial(greet, 'Hi')

# Using lambda
say_hello_lambda = lambda name: greet('Hello', name)
say_hi_lambda = lambda name: greet('Hi', name)

print(say_hello('Alice'))
print(say_hello_lambda('Alice'))
print(say_hi('Bob'))
print(say_hi_lambda('Bob'))

# partial provides more information (__name__, __doc__ are preserved from wrapped)
print(f'partial __name__: {say_hello.__name__}')
print(f'lambda __name__: {say_hello_lambda.__name__}')

# Output:
# Hello, Alice!
# Hello, Alice!
# Hi, Bob!
# Hi, Bob!
# partial __name__: partial
# lambda __name__: <lambda>
```

### Example 5: Partial with map and sorted

```python
from functools import partial

numbers = [1, 2, 3, 4, 5]

double = partial(map, lambda x: x * 2)
print(list(double(numbers)))

# Real-world: sorting by absolute value
points = [(-3, 1), (-1, 5), (-2, 3), (0, 2)]
sort_by_x = partial(sorted, key=lambda p: p[0])
sort_by_y = partial(sorted, key=lambda p: p[1])

print(sort_by_x(points))
print(sort_by_y(points))

# Output:
# [2, 4, 6, 8, 10]
# [(-3, 1), (-2, 3), (-1, 5), (0, 2)]
# [(-3, 1), (0, 2), (-2, 3), (-1, 5)]
```

### Example 6: partialmethod for methods

```python
from functools import partialmethod

class Window:
    def __init__(self, title):
        self.title = title
        self.visible = False

    def set_visibility(self, visible):
        self.visible = visible
        status = 'visible' if visible else 'hidden'
        print(f'{self.title} is now {status}')

    show = partialmethod(set_visibility, True)
    hide = partialmethod(set_visibility, False)

w = Window('Main Window')
w.show()
w.hide()

# Output:
# Main Window is now visible
# Main Window is now hidden
```

### Example 7: AI/ML — partial for model configuration

```python
from functools import partial

def train_model(model, data, learning_rate, epochs, log_every=10):
    print(f'Training {model.__class__.__name__}: lr={learning_rate}, epochs={epochs}')
    # Simulated training loop
    for epoch in range(epochs):
        if epoch % log_every == 0:
            print(f'  Epoch {epoch}/{epochs}')
    print(f'Finished training {model.__class__.__name__}')
    return {'model': model, 'trained': True}

class NeuralNetwork:
    def __init__(self, name):
        self.name = name
    def __class__(self):
        return type(self)

# Create specialized trainers
fast_trainer = partial(train_model, learning_rate=0.01, epochs=5)
slow_trainer = partial(train_model, learning_rate=0.001, epochs=20)

model = NeuralNetwork('Classifier')
result = fast_trainer(model, 'dataset.csv')
print(result)

# Output:
# Training NeuralNetwork: lr=0.01, epochs=5
#   Epoch 0/5
#   ...
# Finished training NeuralNetwork
# {'model': ..., 'trained': True}
```

## Common Mistakes

1. Using `partial` when a default parameter would work — prefer defaults for permanent bindings
2. Forgetting that `partial` binds arguments at definition time but evaluates them eagerly (mutable defaults are shared)
3. Confusing positional and keyword argument binding order in `partial`
4. Using `partial` for extremely simple cases where a lambda is clearer
5. Overusing partials and making code harder to trace and debug

## Interview Questions

### Beginner - 5

1. What is a partial function in Python?
2. How do you create a partial function using `functools.partial`?
3. What is the difference between a partial function and a lambda?
4. Can you combine positional and keyword arguments in a partial?
5. How do you inspect the original function and bound arguments of a partial?

### Intermediate - 5

1. What is `partialmethod` and how does it differ from regular `partial`?
2. How do partial functions work with methods and `self`?
3. What happens to `__doc__` and `__name__` when you create a partial?
4. How would you create a partial that keeps the remaining function's signature visible?
5. When would you choose `partial` over a closure for fixing arguments?

### Advanced - 3

1. How does `functools.partial` interact with Python's descriptor protocol?
2. Implement `partial` from scratch and explain the edge cases.
3. How would you create a "curried" version of a function using partials, and what are the limitations compared to true currying?

## Practice Problems

### Easy - 5

1. Create a partial of `int()` that always converts to base 16 (hex).
2. Use partial to create `double` from a generic `multiply` function.
3. Create a partial `print_to_file` that prints to a specific file.
4. Create a partial of `round` that always rounds to 2 decimal places.
5. Use partial to create `is_even` from a generic `divisible_by` function.

### Medium - 5

1. Create a URL builder partial that fixes the base URL and protocol.
2. Use partial to create a custom `sorted` that always sorts by the second element.
3. Implement a logging system using partials with different log levels.
4. Create a mathematical function factory using partials and `pow`.
5. Build a partial-based configuration system for data preprocessing.

### Hard - 3

1. Implement a `curry` function that converts any function into a chain of partials.
2. Build a `partial_recorder` that logs all calls and arguments to a partial function.
3. Design and implement a partial-based middleware composition system for API requests.

## Solutions

### Easy 1

```python
from functools import partial

hex_int = partial(int, base=16)
print(hex_int('FF'))
print(hex_int('A'))
print(hex_int('10'))

# Output:
# 255
# 10
# 16
```

### Medium 1

```python
from functools import partial

def build_url(base, protocol, path, params=''):
    url = f'{protocol}://{base}/{path}'
    if params:
        url += f'?{params}'
    return url

api_url = partial(build_url, 'api.example.com', 'https')
print(api_url('users'))
print(api_url('products', 'page=1&limit=10'))

# Output:
# https://api.example.com/users
# https://api.example.com/products?page=1&limit=10
```

### Hard 1

```python
from functools import partial

def curry(fn, *args, **kwargs):
    if args or kwargs:
        return partial(curry, fn, *args, **kwargs)
    return fn()

def add(a, b, c):
    return a + b + c

curried_add = curry(add)
print(curried_add(1)(2)(3))

# Works for any number of arguments
print(curried_add(10, 20)(30))

# Output:
# 6
# 60
```

## Related Concepts

- Lambda functions
- Closures (PYT-041)
- `functools` module (PYT-043)
- Higher-order functions
- Function composition
- Currying (functional programming)
- Default function parameters

## Next Concepts

- `functools` module (PYT-043)
- `itertools` module (PYT-044)
- Decorators and function wrapping
- Functional programming in Python

## Summary

Partial functions allow you to fix some arguments of a function, creating a new callable with fewer parameters. `functools.partial` is the primary tool for this, offering a readable alternative to lambdas for many use cases. `partialmethod` extends this pattern to class methods. Partials reduce repetition, enable function customization, and are valuable in both general programming and specialized AI/ML workflows.

## Key Takeaways

- `functools.partial` binds arguments to a function, returning a new callable
- Partials can bind both positional and keyword arguments
- Partials preserve the wrapped function's identity better than lambdas
- `partialmethod` works like `partial` but for class methods
- Use partials when you repeatedly pass the same arguments to a function
- Prefer default parameters when you control the function definition
- Partials are excellent for pre-configuring functions in AI/ML pipelines
