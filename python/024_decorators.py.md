# Concept: Decorators

## Concept ID

PYT-024

## Difficulty

Advanced

## Domain

Python

## Module

Functions and Modules

## Learning Objectives

- Understand the `@decorator` syntax and how it transforms functions
- Write custom decorators using `*args` and `**kwargs` wrappers
- Use `functools.wraps` to preserve function metadata
- Create decorators that accept arguments
- Implement class-based decorators
- Stack multiple decorators and understand execution order
- Apply AI/ML-specific decorators for timing, caching, and no-grad contexts

## Prerequisites

- PYT-016: Functions (first-class objects, closures)
- PYT-017: Arguments and Parameters (`*args`, `**kwargs`)
- PYT-019: Scope and Namespace (closures, nonlocal)

## Definition

A **decorator** is a function that takes another function as an argument and extends its behavior without explicitly modifying it. The `@decorator` syntax is syntactic sugar for `function = decorator(function)`. Decorators are a form of metaprogramming — code that modifies other code at definition time.

```python
@decorator
def func():
    pass
# Equivalent to: func = decorator(func)
```

## Intuition

Think of a decorator as a camera lens filter. The original function is the camera lens — it captures the image. The filter (decorator) wraps around the lens, modifying the light before it reaches the sensor. You can stack multiple filters (multiple decorators), and each one adds or modifies behavior. The lens itself remains unchanged; you just change what wraps it.

## Why This Concept Matters

Decorators enable clean, reusable cross-cutting concerns. Logging, timing, authentication, caching, input validation, and error handling can be applied to any function with a single line. In AI/ML, decorators are used for `@torch.no_grad()` (disable gradient computation), `@tf.function` (compile to graph), timing training epochs, caching model outputs, and registering callbacks. Understanding decorators is essential for working with modern Python frameworks.

## Real World Examples

1. **Web frameworks**: `@app.route('/')` in Flask, `@api.get('/items')` in FastAPI.
2. **Testing**: `@pytest.fixture`, `@pytest.mark.parametrize`.
3. **ML**: `@torch.no_grad()` disables gradient tracking during inference.
4. **Caching**: `@functools.lru_cache(maxsize=128)` memoizes function results.
5. **Authentication**: `@login_required` checks user auth before allowing access.

## AI/ML Relevance

Decorators are deeply embedded in ML frameworks. PyTorch's `@torch.no_grad()` disables gradient computation during inference, saving memory and computation. TensorFlow's `@tf.function` compiles a function into a TensorFlow graph for optimization. `@functools.lru_cache` caches expensive data loading operations. Custom decorators time training epochs, log metrics, cache model outputs, and manage device placement. Every ML engineer should know how to write and compose decorators.

## Code Examples

### Example 1: Basic decorator structure

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Before {func.__name__}")
        result = func(*args, **kwargs)
        print(f"After {func.__name__}")
        return result
    return wrapper

@my_decorator
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))
# Output:
# Before greet
# After greet
# Hello, Alice!
```

### Example 2: Using `functools.wraps` to preserve metadata

```python
import functools

def preserve_metadata(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper docstring."""
        return func(*args, **kwargs)
    return wrapper

@preserve_metadata
def say_hello(name):
    """Say hello to someone."""
    return f"Hi, {name}!"

# Without @functools.wraps, these would show wrapper's metadata
print(say_hello.__name__)
# Output: say_hello
print(say_hello.__doc__)
# Output: Say hello to someone.
```

### Example 3: Timing decorator (AI/ML relevant)

```python
import functools
import time

def timer(func):
    """Decorator that prints the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timer
def train_epoch(n_batches):
    for i in range(n_batches):
        time.sleep(0.01)  # Simulate training step
    return "Epoch complete"

print(train_epoch(10))
# Output:
# train_epoch took 0.1001s
# Epoch complete
```

### Example 4: Decorator with arguments

```python
import functools

def repeat(n=2):
    """Decorator that repeats a function n times."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(n):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(n=3)
def announce(msg):
    print(msg)
    return msg

announce("Hello!")
# Output:
# Hello!
# Hello!
# Hello!
```

### Example 5: Logging decorator for ML training

```python
import functools
import logging
import time

logging.basicConfig(level=logging.INFO)

def log_training(func):
    """Decorator that logs training step information."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        logging.info(f"Starting {func.__name__} with batch_size={self.batch_size}")
        start = time.perf_counter()
        result = func(self, *args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.info(f"Completed {func.__name__} in {elapsed:.2f}s, loss={result:.4f}")
        return result
    return wrapper

class Trainer:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size

    @log_training
    def train_step(self, data):
        # Simulate training
        time.sleep(0.05)
        return 0.5  # Simulated loss

trainer = Trainer()
trainer.train_step("data")
# Output:
# INFO:root:Starting train_step with batch_size=32
# INFO:root:Completed train_step in 0.05s, loss=0.5000
```

### Example 6: Class-based decorator

```python
import functools

class CountCalls:
    """Class-based decorator that counts function calls."""

    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        print(f"Call {self.calls} of {self.func.__name__}")
        return self.func(*args, **kwargs)

@CountCalls
def predict(x):
    return x * 2

print(predict(5))
# Output: Call 1 of predict
#         10
print(predict(10))
# Output: Call 2 of predict
#         20
print(predict.calls)
# Output: 2
```

### Example 7: Multiple decorators (stacking)

```python
import functools
import time

def bold(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return f"<b>{func(*args, **kwargs)}</b>"
    return wrapper

def italic(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return f"<i>{func(*args, **kwargs)}</i>"
    return wrapper

@bold
@italic
def format_text(text):
    return text

print(format_text("Hello"))
# Output: <b><i>Hello</i></b>
# Order: first italic wraps, then bold wraps the result
# Equivalent to: format_text = bold(italic(format_text))
```

### Example 8: `@torch.no_grad()` equivalent in pure Python

```python
import functools

def no_grad(func):
    """Simulate PyTorch's @torch.no_grad() decorator.

    In PyTorch, this disables gradient computation during inference.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("[no_grad] Disabling gradient tracking")
        # In real PyTorch: with torch.no_grad():
        result = func(*args, **kwargs)
        print("[no_grad] Re-enabling gradient tracking")
        return result
    return wrapper

@no_grad
def predict(model, input_data):
    """Run inference without gradient computation."""
    print(f"Running inference on {len(input_data)} samples")
    return [x * 0.5 for x in input_data]

predictions = predict("MyModel", [1.0, 2.0, 3.0])
print(predictions)
# Output:
# [no_grad] Disabling gradient tracking
# Running inference on 3 samples
# [no_grad] Re-enabling gradient tracking
# [0.5, 1.0, 1.5]
```

### Example 9: Caching/Memoization decorator

```python
import functools
import time

def cache(func):
    """Simple in-memory cache decorator."""
    saved = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args in saved:
            print(f"Cache hit for {args}")
            return saved[args]
        print(f"Cache miss for {args}")
        result = func(*args)
        saved[args] = result
        return result
    return wrapper

@cache
def expensive_computation(n):
    """Simulate an expensive computation."""
    time.sleep(1)
    return n * n

print(expensive_computation(5))
# Output: Cache miss for (5,)
#         25
print(expensive_computation(5))
# Output: Cache hit for (5,)
#         25 (returned instantly)
```

### Example 10: Device placement decorator (AI/ML)

```python
import functools

def to_device(device="cpu"):
    """Decorator that logs which device a function runs on."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"[{func.__name__}] Running on {device}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@to_device(device="cuda:0")
def train_step(batch):
    return f"Processed batch of {len(batch)} items on GPU"

@to_device(device="cpu")
def preprocess(data):
    return f"Preprocessed {data} on CPU"

print(preprocess("raw_data"))
# Output: [preprocess] Running on cpu
#         Preprocessed raw_data on CPU
print(train_step([1, 2, 3]))
# Output: [train_step] Running on cuda:0
#         Processed batch of 3 items on GPU
```

## Common Mistakes

1. **Forgetting `functools.wraps`**: Without it, the decorated function loses its `__name__`, `__doc__`, and `__module__` metadata.
2. **Wrong nesting for decorator arguments**: A decorator that takes arguments needs three levels of nesting: `def decorator(args): def inner(func): def wrapper(...)`.
3. **Assuming decorators run at call time**: Most of the decorator code (except the wrapper) runs at function definition time, not call time.
4. **Forgetting `return` in the wrapper**: If the wrapper doesn't return the function's result, the decorated function returns `None`.
5. **Losing the `self` parameter in method decorators**: When decorating a class method, the wrapper must accept `self` as the first argument.

## Interview Questions

### Beginner

1. What is a decorator in Python? What is the `@` syntax?
2. What does `@decorator` desugar to in plain Python?
3. Can you define a decorator without using the `@` syntax?
4. What is the purpose of `functools.wraps` in a decorator?
5. Write a simple decorator that prints "Function called" before execution.

### Intermediate

1. How do you create a decorator that accepts arguments?
2. Explain how multiple decorators are applied when stacked.
3. Write a decorator that measures and prints execution time.
4. What is the difference between a function decorator and a class-based decorator?
5. Write a decorator that retries a function up to 3 times if it raises an exception.

### Advanced

1. Implement a decorator that can be used with or without arguments (e.g., `@retry` and `@retry(3)` both work).
2. How do decorators interact with class methods, static methods, and class methods?
3. Write a decorator that validates function arguments based on type hints.

## Practice Problems

### Easy

1. Write a decorator that prints "Start" before and "End" after a function.
2. Write a decorator that converts the return value to a string.
3. Write a decorator that logs the function name and arguments.
4. Write a decorator that runs a function twice and returns the second result.
5. Write a decorator that ensures the function is only called once (singleton call).

### Medium

1. Write a `@timeout` decorator that raises an exception if the function takes too long.
2. Write a `@debug` decorator that prints the function's arguments and return value.
3. Write a `@deprecated` decorator that warns when a function is called.
4. Write a `@rate_limit` decorator that limits calls to N per second.
5. Write a decorator that caches results to disk (not just memory).

### Hard

1. Implement a `@singleton` decorator for classes that ensures only one instance is created.
2. Write a decorator that creates a retry mechanism with exponential backoff.
3. Implement a decorator that intercepts and transforms exceptions into custom error types.

## Solutions

### Easy Solutions

```python
# 1
def print_start_end(func):
    def wrapper(*args, **kwargs):
        print("Start")
        result = func(*args, **kwargs)
        print("End")
        return result
    return wrapper

# 2
def to_string(func):
    def wrapper(*args, **kwargs):
        return str(func(*args, **kwargs))
    return wrapper

# 3
def log_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}({args}, {kwargs})")
        return func(*args, **kwargs)
    return wrapper

# 4
def twice(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper

# 5
def once(func):
    has_run = False
    result = None
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal has_run, result
        if not has_run:
            result = func(*args, **kwargs)
            has_run = True
        return result
    return wrapper
```

### Medium Solutions

```python
# 1 — Timeout
import signal
def timeout(seconds):
    def decorator(func):
        def handler(signum, frame):
            raise TimeoutError(f"{func.__name__} timed out")
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
        return wrapper
    return decorator

# 2 — Debug
def debug(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result!r}")
        return result
    return wrapper

# 3 — Deprecated
import warnings
def deprecated(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(f"{func.__name__} is deprecated", DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return wrapper

# 5 — Disk cache
import pickle
import os
def disk_cache(cache_dir=".cache"):
    os.makedirs(cache_dir, exist_ok=True)
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str((args, kwargs))
            cache_file = os.path.join(cache_dir, f"{func.__name__}_{hash(key)}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            result = func(*args, **kwargs)
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
            return result
        return wrapper
    return decorator
```

### Hard Solutions

```python
# 1 — Singleton class decorator
def singleton(cls):
    instances = {}
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Database:
    def __init__(self):
        print("Creating database connection")

db1 = Database()
db2 = Database()
print(db1 is db2)  # True

# 2 — Retry with exponential backoff
import time
def retry(max_attempts=3, base_delay=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    delay = base_delay * (2 ** attempt)
                    print(f"Attempt {attempt + 1} failed, retrying in {delay}s...")
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

# 3 — Argument validator from type hints
import inspect
def validate_types(func):
    sig = inspect.signature(func)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, value in bound.arguments.items():
            param = sig.parameters[name]
            if param.annotation != inspect.Parameter.empty:
                if not isinstance(value, param.annotation):
                    raise TypeError(f"{name} should be {param.annotation}, got {type(value)}")
        return func(*args, **kwargs)
    return wrapper
```

## Related Concepts

- PYT-016: Functions (first-class objects)
- PYT-017: Arguments and Parameters (`*args`, `**kwargs`)
- PYT-019: Scope and Namespace (closures, nonlocal)
- PYT-018: Lambda Functions

## Next Concepts

- PYT-025: Generators
- PYT-030: Context Managers (the `with` statement)
- PYT-050: Metaclasses

## Summary

Decorators are functions that modify other functions using the `@` syntax. They enable clean separation of cross-cutting concerns like logging, timing, caching, and authentication. The `functools.wraps` decorator preserves function metadata. Decorators can accept arguments through three-level nesting. Multiple decorators stack from bottom to top. Class-based decorators use `__call__`. In AI/ML, decorators disable gradients, compile graphs, time training, and cache results.

## Key Takeaways

- `@decorator` is syntactic sugar for `func = decorator(func)`.
- Always use `@functools.wraps(func)` in your wrapper to preserve metadata.
- Three levels of nesting for decorators that accept arguments: `outer(args) -> decorator -> wrapper`.
- Stacking order: decorators closest to the function run first; they are applied bottom-up.
- Use class-based decorators when you need to maintain state (like call counters).
- Real ML frameworks use decorators extensively: `@torch.no_grad`, `@tf.function`, `@lru_cache`.
- Decorators execute at function definition time, not call time.
