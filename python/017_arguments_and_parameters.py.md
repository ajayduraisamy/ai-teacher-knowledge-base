# Concept: Arguments and Parameters

## Concept ID

PYT-017

## Difficulty

Intermediate

## Domain

Python

## Module

Functions and Modules

## Learning Objectives

- Distinguish between parameters and arguments
- Master positional and keyword argument passing
- Use `*args` for variable positional arguments
- Use `**kwargs` for variable keyword arguments
- Understand parameter ordering rules
- Implement keyword-only arguments using the bare `*` separator
- Build flexible function interfaces for real-world applications

## Prerequisites

- PYT-016: Functions (def, return, basic parameters)
- Understanding of dictionaries and tuples
- Familiarity with unpacking operators `*` and `**`

## Definition

**Parameters** are the variables listed in a function's definition. **Arguments** are the actual values passed to the function when it is called. Python supports several types of parameters: positional, keyword, default, variable positional (`*args`), variable keyword (`**kwargs`), and keyword-only parameters.

The complete parameter ordering rule is:

```python
def func(pos, pos_default=val, *args, keyword_only, **kwargs):
    pass
```

1. Standard positional parameters
2. Default parameters
3. `*args` (variable positional)
4. Keyword-only parameters (after `*` or `*args`)
5. `**kwargs` (variable keyword)

## Intuition

Think of a function like a vending machine. Positional parameters are the slots where you insert coins in order. Keyword parameters are labeled slots that accept specific items. The `*args` is like a catch-all tray that collects any extra coins, and `**kwargs` is like a note pad for any special instructions. Keywords-only parameters are like buttons you must press by name — you cannot just press the first button and hope.

## Why This Concept Matters

Flexible function signatures are essential for building APIs, libraries, and frameworks. In AI/ML, model classes and training functions accept dozens of hyperparameters through `**kwargs`. Config objects, data loaders, and pipeline stages all benefit from variable argument patterns. Understanding parameter ordering prevents subtle bugs and enables clean, expressive function calls.

## Real World Examples

1. **Web frameworks**: Flask route decorators accept optional methods via keyword arguments.
2. **Data science**: Pandas `read_csv` has dozens of parameters; users pass only the ones they need via keyword arguments.
3. **Deep learning**: PyTorch `nn.Linear(in_features, out_features, bias=True)` uses default parameters.
4. **Logging**: `logging.info(msg, *args, **kwargs)` allows flexible structured logging.
5. **Configuration**: ML training scripts accept hyperparameters via `**kwargs` to support experiment configs.

## AI/ML Relevance

In machine learning, flexible function signatures are everywhere. Model constructors accept numerous hyperparameters through `**kwargs`, allowing config dictionaries to be unpacked directly. Training loops accept callbacks and schedulers as keyword arguments. Data loaders use `*args` to handle variable-length inputs. The `*` separator for keyword-only arguments forces explicit naming of critical hyperparameters like `learning_rate`, preventing accidental misordering.

## Code Examples

### Example 1: Positional vs keyword arguments

```python
def describe_person(name, age, city):
    return f"{name} is {age} years old and lives in {city}."

# Positional
print(describe_person("Alice", 30, "New York"))
# Output: Alice is 30 years old and lives in New York.

# Keyword (order doesn't matter)
print(describe_person(city="London", age=25, name="Bob"))
# Output: Bob is 25 years old and lives in London.

# Mixed: positional before keyword
print(describe_person("Charlie", city="Paris", age=35))
# Output: Charlie is 35 years old and lives in Paris.
```

### Example 2: Default parameters

```python
def greet(name, greeting="Hello", punctuation="!"):
    return f"{greeting}, {name}{punctuation}"

print(greet("Alice"))
# Output: Hello, Alice!
print(greet("Bob", "Hi"))
# Output: Hi, Bob!
print(greet("Charlie", punctuation="?"))
# Output: Hello, Charlie?
```

### Example 3: `*args` for variable positional arguments

```python
def sum_all(*args):
    """Sum an arbitrary number of numeric arguments."""
    total = 0
    for num in args:
        total += num
    return total

print(sum_all(1, 2, 3))
# Output: 6
print(sum_all(10, 20, 30, 40, 50))
# Output: 150
print(sum_all())
# Output: 0

# Unpacking a list into positional arguments
numbers = [1, 2, 3, 4]
print(sum_all(*numbers))
# Output: 10
```

### Example 4: `**kwargs` for variable keyword arguments

```python
def build_profile(name, **kwargs):
    """Build a user profile with arbitrary attributes."""
    profile = {"name": name}
    profile.update(kwargs)
    return profile

print(build_profile("Alice", age=30, city="NYC", role="Engineer"))
# Output: {'name': 'Alice', 'age': 30, 'city': 'NYC', 'role': 'Engineer'}

# Unpacking a dictionary into keyword arguments
config = {"learning_rate": 0.001, "epochs": 10, "batch_size": 32}
print(build_profile("Experiment1", **config))
# Output: {'name': 'Experiment1', 'learning_rate': 0.001, 'epochs': 10, 'batch_size': 32}
```

### Example 5: Keyword-only arguments with bare `*`

```python
def configure_model(layers, *, learning_rate, activation="relu"):
    """Force explicit naming of critical hyperparameters."""
    return {
        "layers": layers,
        "learning_rate": learning_rate,
        "activation": activation
    }

# This works:
print(configure_model([64, 32], learning_rate=0.001))
# Output: {'layers': [64, 32], 'learning_rate': 0.001, 'activation': 'relu'}

# This raises TypeError:
# configure_model([64, 32], 0.001)
# TypeError: configure_model() takes 1 positional argument but 2 were given
```

### Example 6: Complete parameter ordering

```python
def comprehensive(a, b=2, *args, kw_only, **kwargs):
    """Demonstrate the complete Python parameter ordering."""
    return {
        "a": a,
        "b": b,
        "args": args,
        "kw_only": kw_only,
        "kwargs": kwargs
    }

result = comprehensive(1, 3, 4, 5, kw_only=99, extra="x", verbose=True)
print(result)
# Output: {'a': 1, 'b': 3, 'args': (4, 5), 'kw_only': 99, 'kwargs': {'extra': 'x', 'verbose': True}}
```

### Example 7: Flexible model constructor (AI/ML context)

```python
def create_model(input_dim, output_dim, *,
                 hidden_layers=[128, 64],
                 activation="relu",
                 dropout=0.2,
                 **kwargs):
    """Flexible neural network model constructor."""
    config = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_layers": hidden_layers,
        "activation": activation,
        "dropout": dropout,
    }
    config.update(kwargs)
    return config

# Typical usage with hyperparameter config
hyperparams = {"learning_rate": 0.001, "optimizer": "adam"}
model_config = create_model(784, 10, **hyperparams)
print(model_config)
# Output: {'input_dim': 784, 'output_dim': 10, 'hidden_layers': [128, 64],
#          'activation': 'relu', 'dropout': 0.2, 'learning_rate': 0.001,
#          'optimizer': 'adam'}
```

### Example 8: Combining `*args` and `**kwargs` in delegation

```python
def log_message(level, msg, *args, **kwargs):
    """Log a message with optional formatting."""
    prefix = f"[{level.upper()}]"
    formatted = msg.format(*args, **kwargs) if args or kwargs else msg
    return f"{prefix} {formatted}"

print(log_message("info", "User {} logged in from {}", "Alice", "192.168.1.1"))
# Output: [INFO] User Alice logged in from 192.168.1.1

print(log_message("error", "Disk usage at {percent}%", percent=95))
# Output: [ERROR] Disk usage at 95%
```

## Common Mistakes

1. **Wrong parameter order**: Putting `*args` before default parameters causes syntax errors. Defaults must come before `*args`.
2. **Forgetting to unpack**: Passing a list as `args` instead of `*args` passes the list as a single argument rather than unpacking it.
3. **Keyword argument after `**kwargs`**: No parameter can follow `**kwargs` in the signature.
4. **Mutable default arguments with `*args`**: The same mutable default problem applies if you use mutable defaults alongside `*args`.
5. **Overusing `**kwargs`**: Accepting everything with `**kwargs` makes the function signature opaque; document expected keys.

## Interview Questions

### Beginner

1. What is the difference between a parameter and an argument?
2. How do you define a function with default parameter values?
3. Can you mix positional and keyword arguments in a function call? What is the rule?
4. What happens if you pass more positional arguments than parameters? (Without `*args`)
5. What does the `*` operator do when used in a function call (argument unpacking)?

### Intermediate

1. Explain the complete parameter ordering rule in Python function definitions.
2. What is the difference between `*args` and `**kwargs`? When would you use each?
3. How do you define keyword-only arguments? Why would you use them?
4. Write a function that accepts a variable number of arguments and returns their product.
5. What is the purpose of the bare `*` in a function signature?

### Advanced

1. Implement a function that can be called with both positional and keyword arguments and returns a dictionary of all received arguments with their types.
2. How would you design a function that forwards unknown keyword arguments to another function while still processing known ones?
3. Explain how Python resolves argument passing under the hood. What is the role of `METH_VARARGS` in CPython?

## Practice Problems

### Easy

1. Write a function `multiply(a, b, c=1)` that multiplies three numbers with one having a default.
2. Write a function `repeat(msg, times=2)` that repeats a message.
3. Write a function `power(base, exp=2)` that computes base^exp.
4. Write a function `create_point(x, y, z=0)` that creates a 3D point tuple.
5. Write a function `format_date(day, month, year, sep="-")` that formats a date string.

### Medium

1. Write a function `average(*nums)` that returns the average of an arbitrary number of numbers.
2. Write a function `pipeline(*funcs, initial=None)` that applies a chain of functions to `initial`.
3. Write a function `register_user(name, **details)` that returns a user dict with name and all extra details.
4. Write a function `safe_divide(a, b, **kwargs)` that forwards keyword arguments to division but returns infinity on division by zero.
5. Write a function `merge_configs(default, **overrides)` that merges a default config dict with overrides.

### Hard

1. Implement a flexible event handler system using `**kwargs` where handlers can subscribe to specific event keys.
2. Write a function `call_with_config(func, config)` that inspects func's signature and passes only relevant keys from config.
3. Design a validation decorator `@validate_params` that checks argument types and ranges using keyword-only parameters for constraints.

## Solutions

### Easy Solutions

```python
# 1
def multiply(a, b, c=1): return a * b * c

# 2
def repeat(msg, times=2): return (msg + " ") * times

# 3
def power(base, exp=2): return base ** exp

# 4
def create_point(x, y, z=0): return (x, y, z)

# 5
def format_date(day, month, year, sep="-"):
    return f"{day:02d}{sep}{month:02d}{sep}{year}"
```

### Medium Solutions

```python
# 1
def average(*nums):
    return sum(nums) / len(nums) if nums else 0.0

# 2
def pipeline(*funcs, initial=None):
    result = initial
    for func in funcs:
        result = func(result) if result is not None else func()
    return result

# 3
def register_user(name, **details):
    return {"name": name, **details}

# 4
def safe_divide(a, b, **kwargs):
    try:
        return a / b
    except ZeroDivisionError:
        return float("inf")

# 5
def merge_configs(default, **overrides):
    config = default.copy()
    config.update(overrides)
    return config
```

### Hard Solutions

```python
# 2
import inspect

def call_with_config(func, config):
    sig = inspect.signature(func)
    filtered = {k: v for k, v in config.items()
                if k in sig.parameters}
    return func(**filtered)

# 3
import functools
def validate_params(**constraints):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound = inspect.signature(func).bind(*args, **kwargs)
            bound.apply_defaults()
            for param, value in bound.arguments.items():
                if param in constraints:
                    constraint = constraints[param]
                    if callable(constraint) and not constraint(value):
                        raise ValueError(f"{param}={value} fails constraint")
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

## Related Concepts

- PYT-016: Functions
- PYT-018: Lambda Functions
- PYT-019: Scope and Namespace
- PYT-024: Decorators

## Next Concepts

- PYT-030: Exception Handling (validating arguments)
- PYT-035: Advanced Function Patterns
- PYT-040: Configuration Management

## Summary

Functions support positional, keyword, default, `*args`, `**kwargs`, and keyword-only parameters. The ordering rule is strict: positional, defaults, `*args`, keyword-only, `**kwargs`. The bare `*` enforces keyword-only arguments, which is valuable for API design. Unpacking with `*` and `**` works in both function definitions and calls. Proper use of flexible signatures enables elegant, maintainable interfaces for libraries, frameworks, and ML pipelines.

## Key Takeaways

- Parameters are the definition; arguments are the values passed.
- Use `*args` for any number of positional arguments; `**kwargs` for any number of keyword arguments.
- Place keyword-only arguments after a bare `*` to enforce explicit naming.
- The complete parameter order is: positional, defaults, `*args`, keyword-only, `**kwargs`.
- Unpack lists/tuples with `*` and dicts with `**` in function calls.
- Mutable default gotcha applies to all parameter types.
