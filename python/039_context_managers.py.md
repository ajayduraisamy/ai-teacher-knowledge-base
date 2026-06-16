# Concept: Context Managers

## Concept ID

PYT-039

## Difficulty

Advanced

## Domain

Python

## Module

Advanced Python

## Learning Objectives

- Understand the purpose and use of the `with` statement
- Implement custom context managers using `__enter__` and `__exit__`
- Handle exceptions correctly within `__exit__`
- Use `contextlib.contextmanager` to create context managers from generator functions
- Leverage `contextlib.suppress` for ignoring specific exceptions
- Use `contextlib.closing` for objects with a `close()` method
- Master `ExitStack` for dynamic context management
- Apply context managers to AI/ML workflows

## Prerequisites

- Basic class definition and methods
- Exception handling with `try`/`except`/`finally`
- Generators and the `yield` keyword (for `contextmanager` decorator)

## Definition

A **context manager** is an object that defines a runtime context to be established when entering a block of code and torn down when leaving it. Context managers use the `with` statement to ensure that setup and cleanup actions are executed reliably, even if exceptions occur.

The context manager protocol consists of two methods: `__enter__()` (called when entering the `with` block) and `__exit__()` (called when leaving the `with` block, regardless of how the block exits).

## Intuition

Think of a context manager as a "guaranteed cleanup" contract. When you open a file with `with open(...) as f:`, you are guaranteed that the file will be closed when the block ends — whether it ends normally, by exception, or by `return`. Context managers wrap a resource lifecycle in a safe, composable envelope.

## Why This Concept Matters

Resource management is one of the most error-prone aspects of programming. Context managers eliminate entire categories of bugs: forgetting to close files, release locks, commit or rollback transactions, or clean up temporary resources. They make code more readable by separating the resource management logic from the business logic, and they guarantee cleanup no matter how the code block is exited.

## Real World Examples

- File I/O (`with open(...) as f`)
- Database connections and transactions
- Thread locks and synchronization primitives
- Temporary directory and file management
- Timing and profiling code blocks
- Redirecting stdout/stderr temporarily
- Managing network connections (sockets, HTTP sessions)

## AI/ML Relevance

Context managers are particularly valuable in AI/ML workflows:
- Managing GPU memory contexts (e.g., `torch.no_grad()`, `tf.GradientTape()`)
- Model training sessions with automatic checkpoint saving and logging
- Experiment tracking context (MLflow, Weights & Biases runs)
- Temporary model evaluation environments
- Managing temporary data files for preprocessing pipelines

## Code Examples

### Example 1: Custom context manager with __enter__ and __exit__

```python
class Timer:
    def __enter__(self):
        import time
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.elapsed = time.time() - self.start
        print(f'Elapsed: {self.elapsed:.4f}s')
        return False

with Timer() as t:
    sum(range(10_000_000))
print(f'Timer recorded: {t.elapsed:.4f}s')

# Output:
# Elapsed: 0.2345s
# Timer recorded: 0.2345s
```

### Example 2: Handling exceptions in __exit__

```python
class SafeDivision:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ZeroDivisionError:
            print(f'Caught: {exc_val}')
            return True
        return False

with SafeDivision():
    result = 10 / 0
    print('This will not print')

print('Execution continues here')

# Output:
# Caught: division by zero
# Execution continues here
```

### Example 3: contextlib.contextmanager decorator

```python
from contextlib import contextmanager

@contextmanager
def open_file(filename, mode):
    print(f'Opening {filename}')
    f = open(filename, mode)
    try:
        yield f
    finally:
        print(f'Closing {filename}')
        f.close()

import tempfile
import os

tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w')
tmp_path = tmp.name
tmp.close()

with open_file(tmp_path, 'w') as f:
    f.write('Hello, World!')

with open_file(tmp_path, 'r') as f:
    content = f.read()
    print(content)

os.unlink(tmp_path)

# Output:
# Opening ...\tmp....txt
# Closing ...\tmp....txt
# Opening ...\tmp....txt
# Closing ...\tmp....txt
# Hello, World!
```

### Example 4: contextlib.suppress

```python
from contextlib import suppress
import os

# Instead of:
# try:
#     os.remove('nonexistent.txt')
# except FileNotFoundError:
#     pass

with suppress(FileNotFoundError):
    os.remove('nonexistent.txt')
    print('This line runs, but exception is suppressed')

print('Clean exit')

# Note: the print inside the with block would NOT run
# because os.remove raises before it

# Correct usage:
with suppress(FileNotFoundError):
    os.remove('nonexistent.txt')

print('No error raised')

# Output:
# No error raised
```

### Example 5: contextlib.closing

```python
from contextlib import closing
import urllib.request

url = 'http://httpbin.org/get'
try:
    with closing(urllib.request.urlopen(url)) as response:
        data = response.read()
        print(f'Got {len(data)} bytes')
except Exception as e:
    print(f'Error: {e}')
```

### Example 6: ExitStack for dynamic cleanup

```python
from contextlib import ExitStack

def process_files(filenames):
    with ExitStack() as stack:
        files = []
        for name in filenames:
            f = stack.enter_context(open(name, 'r'))
            files.append(f)

        for f, name in zip(files, filenames):
            first_line = f.readline().strip()
            print(f'{name}: {first_line}')

    # All files are automatically closed here

import tempfile
import os

f1 = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
f1.write('First file content\n')
f1.close()

f2 = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
f2.write('Second file content\n')
f2.close()

process_files([f1.name, f2.name])

os.unlink(f1.name)
os.unlink(f2.name)

# Output:
# ...\tmp...txt: First file content
# ...\tmp...txt: Second file content
```

### Example 7: AI/ML — torch.no_grad style context manager

```python
from contextlib import contextmanager

@contextmanager
def no_grad():
    print('Disabling gradient tracking')
    # In PyTorch, this would be: torch.set_grad_enabled(False)
    try:
        yield
    finally:
        print('Re-enabling gradient tracking')
        # torch.set_grad_enabled(True)

with no_grad():
    print('Performing inference without gradients')

# Output:
# Disabling gradient tracking
# Performing inference without gradients
# Re-enabling gradient tracking
```

### Example 8: Database transaction context manager

```python
from contextlib import contextmanager

@contextmanager
def transaction():
    print('BEGIN TRANSACTION')
    try:
        yield
        print('COMMIT')
    except Exception as e:
        print(f'ROLLBACK (error: {e})')
        raise

with transaction():
    print('Doing work...')

print('---')

with transaction():
    print('Doing work...')
    raise ValueError('Something went wrong')

# Output:
# BEGIN TRANSACTION
# Doing work...
# COMMIT
# ---
# BEGIN TRANSACTION
# Doing work...
# ROLLBACK (error: Something went wrong)
# Traceback (most recent call last):
#   ...
# ValueError: Something went wrong
```

## Common Mistakes

1. Returning `True` from `__exit__` without understanding that it suppresses exceptions
2. Forgetting that `__exit__` is called even when an exception occurs — do not double-clean
3. Using `@contextmanager` on a generator without wrapping the `yield` in `try`/`finally`
4. Not propagating exceptions from `__exit__` when they should not be suppressed
5. Assuming context managers are thread-safe by default — they are not
6. Nesting too many context managers without `ExitStack`
7. Forgetting that `contextmanager`-decorated functions cannot yield multiple times

## Interview Questions

### Beginner - 5

1. What is the `with` statement used for in Python?
2. Name three built-in context managers in Python.
3. What methods must a context manager implement?
4. What does it mean when `__exit__` returns `True`?
5. How does `with open(...) as f` guarantee the file is closed?

### Intermediate - 5

1. Implement a context manager that measures execution time of a block.
2. How does `contextlib.contextmanager` work under the hood?
3. What is the difference between `contextlib.closing` and just calling `.close()` manually?
4. How do you create a context manager that acquires and releases a lock?
5. What happens if an exception is raised inside a `with` block and `__exit__` returns `False`?

### Advanced - 3

1. How does `ExitStack` work and when would you use it over nested `with` statements?
2. Explain how Python's `contextlib.contextmanager` decorator converts a generator into a context manager.
3. How would you implement a context manager that supports asynchronous setup and teardown (`__aenter__` / `__aexit__`)?

## Practice Problems

### Easy - 5

1. Create a context manager that prints "Entering" and "Exiting" before and after a block.
2. Use `with open(...)` to write and then read a file safely.
3. Implement a context manager that temporarily changes the current working directory.
4. Use `contextlib.suppress` to silently ignore a `KeyError`.
5. Create a context manager using `@contextmanager` that logs the start and end of a block.

### Medium - 5

1. Implement a `Timed` context manager that prints elapsed time with microsecond precision.
2. Create a `RedirectStdout` context manager that captures all print output to a string.
3. Build a `DatabaseSession` context manager that autocommits on success and rolls back on error.
4. Use `ExitStack` to manage multiple temporary files dynamically.
5. Create a nested context manager that accumulates all exceptions and re-raises them at the end.

### Hard - 3

1. Implement a context manager that supports entering multiple times (reusable context manager with `__init__` and `__enter__`/`__exit__`).
2. Build an `AtomicFileWriter` context manager that writes to a temp file and atomically replaces the target on success.
3. Design and implement an asynchronous context manager for managing an aiohttp client session.

## Solutions

### Easy 1

```python
class EnterExit:
    def __enter__(self):
        print('Entering')
        return self

    def __exit__(self, *args):
        print('Exiting')

with EnterExit():
    print('Inside block')

# Output:
# Entering
# Inside block
# Exiting
```

### Medium 1

```python
import time
from contextlib import contextmanager

@contextmanager
def timed():
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - start) * 1_000_000
        print(f'{elapsed:.0f}us elapsed')

with timed():
    sum(range(1_000_000))

# Output:
# 12345us elapsed
```

### Hard 1

```python
class ReusableTimer:
    def __init__(self):
        self.start = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f'Elapsed: {self.elapsed:.4f}s')

timer = ReusableTimer()
with timer:
    sum(range(1_000_000))
with timer:
    sum(range(2_000_000))

# Output:
# Elapsed: 0.0123s
# Elapsed: 0.0256s
```

## Related Concepts

- Exception handling (`try`/`except`/`finally`)
- Generators and `yield`
- Resource management patterns
- The `with` statement grammar
- Asynchronous context managers (`__aenter__`, `__aexit__`)
- `contextlib` module utilities

## Next Concepts

- Descriptors (PYT-040)
- Closures (PYT-041)
- Decorators and function wrapping
- Asynchronous programming with async/await

## Summary

Context managers provide a clean, reliable way to manage resources in Python. By implementing `__enter__` and `__exit__`, any object can participate in the `with` statement protocol. The `contextlib` module offers powerful helpers like `@contextmanager`, `suppress`, `closing`, and `ExitStack` that simplify common resource management patterns. Context managers are a cornerstone of robust Python programming.

## Key Takeaways

- Context managers guarantee proper resource cleanup via the `with` statement
- Implement `__enter__` and `__exit__` for class-based context managers
- Use `@contextmanager` for simple, generator-based context managers
- Return `True` from `__exit__` to suppress exceptions
- `ExitStack` enables dynamic, composable context management
- Context managers are critical for safe resource handling in any Python application
- AI/ML workflows benefit from context managers for GPU contexts, training sessions, and experiment tracking
