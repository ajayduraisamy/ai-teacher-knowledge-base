# Concept: Error Handling

## Concept ID

PYT-014

## Difficulty

INTERMEDIATE

## Domain

Python

## Module

Python Basics

## Learning Objectives

- Understand Python's exception hierarchy and how try/except works
- Handle specific exception types (ValueError, TypeError, FileNotFoundError, etc.)
- Use the full try/except/else/finally block structure
- Raise exceptions with raise and create custom exception classes
- Use assertions for debugging and validation
- Understand exception chaining with raise ... from
- Apply best practices for error handling in real-world code

## Prerequisites

- Basic Python syntax and control flow
- Understanding of functions and scope
- Familiarity with file operations and user input

## Definition

Error handling in Python is the practice of anticipating, catching, and responding to runtime errors (exceptions) that occur during program execution. Python uses a try/except/else/finally mechanism to gracefully handle errors without crashing the program. Exceptions are objects that represent error conditions, and they follow a class hierarchy rooted in `BaseException`. Python's philosophy is "it's easier to ask for forgiveness than permission" (EAFP), meaning you attempt an operation and handle errors if they occur, rather than checking all preconditions first.

## Intuition

Think of error handling like a safety net in a circus. The performer (your code) walks a tightrope (executes an operation). If they fall (an exception occurs), the safety net (the except block) catches them safely instead of letting them hit the ground (crash the program). The `else` block is like the applause after a successful performance — it runs only when no fall occurs. The `finally` block is the cleanup crew that always comes out, regardless of what happened — they always lower the safety net and pack up the equipment.

## Why This Concept Matters

Robust error handling separates professional, production-ready code from fragile scripts. Users expect applications to handle invalid input, missing files, network failures, and unexpected states gracefully. In data pipelines and AI/ML workflows, a single malformed data point shouldn't crash a multi-hour training job. Proper error handling enables:
- Graceful degradation (continuing with partial results)
- Clear, actionable error messages
- Resource cleanup (closing files, database connections)
- Debugging and logging for root cause analysis
- Validation of assumptions via assertions

## Real World Examples

- **User input validation**: Catching `ValueError` when converting string input to integer
- **File operations**: Catching `FileNotFoundError` when opening a missing configuration file
- **Network requests**: Catching connection timeouts and retrying
- **Data parsing**: Handling malformed JSON, CSV, or XML gracefully
- **API development**: Returning appropriate HTTP error codes with structured error responses
- **Database operations**: Catching integrity errors and constraint violations

## AI/ML Relevance

- **Data loading errors**: Handling missing files, corrupt data, or unexpected formats when loading training datasets
- **Graceful degradation**: Skipping bad samples in a batch during training and logging warnings instead of crashing
- **Validation in preprocessing**: Catching invalid feature values (NaN, infinity, wrong types) during data transformation
- **Model serving**: Handling prediction requests with malformed input and returning meaningful errors
- **Hyperparameter tuning**: Managing failed training runs (NaN loss, convergence errors) without aborting the entire search
- **Checkpoint management**: Handling corrupted checkpoint files when resuming training
- **Resource management**: Ensuring GPU memory is freed and files are closed even after exceptions

## Code Examples

### Example 1: Basic try/except

```python
# Without error handling
try:
    num = int(input("Enter a number: "))
    result = 10 / num
    print(f"Result: {result}")
except ValueError:
    print("That's not a valid number!")
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Output (if user enters "abc"):
# That's not a valid number!

# Catching multiple exceptions in one block
try:
    data = [1, 2, 3]
    index = 5
    value = data[index]
except (IndexError, TypeError) as e:
    print(f"Error: {e}")
# Output: Error: list index out of range
```

### Example 2: try/except/else/finally

```python
def safe_divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Cannot divide by zero!")
        return None
    except TypeError:
        print("Both arguments must be numbers!")
        return None
    else:
        print(f"Division successful: {result}")
        return result
    finally:
        print("Cleanup: this always runs.")

print(safe_divide(10, 2))
# Output:
# Division successful: 5.0
# Cleanup: this always runs.
# 5.0

print(safe_divide(10, 0))
# Output:
# Cannot divide by zero!
# Cleanup: this always runs.
# None
```

### Example 3: Raising Exceptions

```python
def withdraw(balance, amount):
    if amount < 0:
        raise ValueError("Withdrawal amount cannot be negative")
    if amount > balance:
        raise ValueError(f"Insufficient funds: have {balance}, need {amount}")
    return balance - amount

try:
    new_balance = withdraw(100, 150)
except ValueError as e:
    print(f"Withdrawal failed: {e}")
# Output: Withdrawal failed: Insufficient funds: have 100, need 150

# Re-raising exceptions
def process_data(data):
    try:
        result = data["key"] / data["divisor"]
    except KeyError as e:
        print(f"Missing key: {e}")
        raise  # re-raise the same exception
    return result

try:
    process_data({"divisor": 5})
except KeyError:
    print("Handled at outer level")
# Output:
# Missing key: 'key'
# Handled at outer level
```

### Example 4: Custom Exception Classes

```python
class InsufficientFundsError(Exception):
    """Raised when an account has insufficient funds for a transaction."""
    pass

class NegativeAmountError(Exception):
    """Raised when a transaction amount is negative."""
    def __init__(self, amount, message="Amount cannot be negative"):
        self.amount = amount
        self.message = f"{message}: {amount}"
        super().__init__(self.message)

class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def withdraw(self, amount):
        if amount < 0:
            raise NegativeAmountError(amount)
        if amount > self.balance:
            raise InsufficientFundsError(
                f"{self.owner} needs {amount} but only has {self.balance}"
            )
        self.balance -= amount
        return self.balance

acc = BankAccount("Alice", 100)
try:
    acc.withdraw(-50)
except NegativeAmountError as e:
    print(f"Error: {e}")
# Output: Error: Amount cannot be negative: -50

try:
    acc.withdraw(200)
except InsufficientFundsError as e:
    print(f"Error: {e}")
# Output: Error: Alice needs 200 but only has 100
```

### Example 5: Exception Chaining

```python
# Implicit chaining (__context__)
try:
    data = {"value": "not_a_number"}
    num = int(data["value"])
except KeyError as e:
    print(f"Key error: {e}")
except ValueError as e:
    print(f"Value error: {e}")
    # Implicit chaining — the ValueError is the context

# Explicit chaining with raise ... from
class DataProcessingError(Exception):
    pass

def load_and_process(filename):
    try:
        with open(filename) as f:
            data = f.read()
        return int(data.strip())
    except FileNotFoundError as e:
        raise DataProcessingError(f"Could not find data file: {filename}") from e
    except ValueError as e:
        raise DataProcessingError(f"Invalid data in file: {filename}") from e

try:
    load_and_process("nonexistent.txt")
except DataProcessingError as e:
    print(f"Error: {e}")
    print(f"Caused by: {e.__cause__}")
# Output:
# Error: Could not find data file: nonexistent.txt
# Caused by: [Errno 2] No such file or directory: 'nonexistent.txt'
```

### Example 6: Assertions

```python
def calculate_bmi(weight_kg, height_m):
    assert weight_kg > 0, "Weight must be positive"
    assert height_m > 0, "Height must be positive"
    return weight_kg / (height_m ** 2)

print(calculate_bmi(70, 1.75))
# Output: 22.857142857142858

try:
    calculate_bmi(-70, 1.75)
except AssertionError as e:
    print(f"Assertion failed: {e}")
# Output: Assertion failed: Weight must be positive

# Using assert for debugging invariants
def process_list(items):
    assert len(items) > 0, "Cannot process empty list"
    for item in items:
        assert isinstance(item, str), f"Expected string, got {type(item)}"
    return [item.upper() for item in items]
```

### Example 7: Common Built-in Exceptions

```python
# ValueError — invalid value for an operation
try:
    int("hello")
except ValueError as e:
    print(f"ValueError: {e}")
# Output: ValueError: invalid literal for int() with base 10: 'hello'

# TypeError — operation on incompatible type
try:
    "hello" + 5
except TypeError as e:
    print(f"TypeError: {e}")
# Output: TypeError: can only concatenate str (not "int") to str

# KeyError — missing dictionary key
try:
    {}["missing"]
except KeyError as e:
    print(f"KeyError: {e}")
# Output: KeyError: 'missing'

# IndexError — list index out of range
try:
    [1, 2, 3][10]
except IndexError as e:
    print(f"IndexError: {e}")
# Output: IndexError: list index out of range

# AttributeError — object lacks attribute
try:
    None.upper()
except AttributeError as e:
    print(f"AttributeError: {e}")
# Output: AttributeError: 'NoneType' object has no attribute 'upper'
```

### Example 8: Context Manager (with statement) and Exceptions

```python
# The with statement handles exceptions during file operations
class ManagedFile:
    def __init__(self, filename, mode="r"):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        # Returning False (default) propagates exceptions
        # Returning True suppresses exceptions
        if exc_type is FileNotFoundError:
            print(f"File not found: {self.filename}")
            return True  # suppress the exception
        return False  # let other exceptions propagate

with ManagedFile("nonexistent.txt") as f:
    content = f.read()
print("Continues after suppressed exception")
# Output:
# File not found: nonexistent.txt
# Continues after suppressed exception
```

## Common Mistakes

1. **Bare except clauses**: `except:` catches everything including `KeyboardInterrupt` and `SystemExit`. Always specify exception types. If you must catch broadly, use `except Exception:` instead.
2. **Swallowing exceptions silently**: `except: pass` hides errors and makes debugging impossible. At minimum, log the exception.
3. **Too broad exception handling**: Catching `Exception` when you only expect `ValueError` can mask unexpected bugs. Be as specific as possible.
4. **Placing code that can fail in `else` or `finally`**: `else` runs when no exception occurs — don't put code there that might raise the same exception type you're catching. `finally` should only contain cleanup code.
5. **Raising generic `Exception`**: Always raise specific exception types or custom exceptions. `raise Exception("something wrong")` forces callers to catch broadly.
6. **Forgetting that `assert` can be disabled**: Assertions are removed when Python runs with `-O` (optimize) flag. Never use `assert` for critical validation that must always run — use `if/raise` instead.
7. **Modifying mutable default arguments while catching exceptions**: Exception handling doesn't protect you from mutable default argument pitfalls.
8. **Returning in both `try` and `finally`**: If both blocks have `return`, the `finally` return value wins, which is almost never what you want.

## Interview Questions

### Beginner

1. **Q**: What is the difference between `try/except` and `if/else` for error checking?
   **A**: `try/except` follows Python's "EAFP" (Easier to Ask Forgiveness than Permission) philosophy — you attempt the operation and handle errors after. `if/else` follows "LBYL" (Look Before You Leap) — you check conditions before attempting. EAFP is often cleaner when checks are complex or race conditions exist.

2. **Q**: What is the purpose of the `finally` block?
   **A**: The `finally` block always executes, whether an exception was raised, caught, or not. It's used for cleanup operations like closing files, releasing network connections, or freeing resources.

3. **Q**: What happens if an exception is not caught?
   **A**: The exception propagates up the call stack. If it reaches the top level and is still unhandled, the program terminates and prints a traceback showing the exception type, message, and where it occurred.

4. **Q**: Can you have multiple `except` blocks for one `try`?
   **A**: Yes, you can have multiple `except` blocks to handle different exception types differently. They are checked in order, so put more specific exceptions before more general ones.

5. **Q**: What is `raise` used for?
   **A**: `raise` intentionally triggers an exception. It can raise a new exception or re-raise the current exception (with no argument inside an except block).

### Intermediate

1. **Q**: How do you create a custom exception in Python?
   **A**: Define a class that inherits from `Exception` (or a more specific subclass). The class can have custom attributes and methods. Example: `class MyError(Exception): pass`.

2. **Q**: What is exception chaining and when would you use it?
   **A**: Exception chaining (`raise ... from ...`) preserves the original exception as the `__cause__` of the new exception. Use it when you catch a low-level exception and want to raise a higher-level, domain-specific exception while preserving the original error context for debugging.

3. **Q**: What is the difference between `__context__` and `__cause__` in exceptions?
   **A**: `__context__` is set implicitly when an exception is raised during handling of another exception. `__cause__` is set explicitly with `raise X from Y`. If `__cause__` is set, it's displayed as "The above exception was the direct cause..." while `__context__` is displayed as "During handling of the above exception, another exception occurred:".

4. **Q**: What are the best practices for using assertions in production code?
   **A**: Use assertions for debugging, documenting invariants, and testing preconditions that should never happen if the code is correct. Never use assertions for input validation, data validation, or any check that must always run — assertions are disabled with `-O` flag.

5. **Q**: What is the exception hierarchy in Python?
   **A**: `BaseException` is the root, with `SystemExit`, `KeyboardInterrupt`, `GeneratorExit`, and `Exception` as direct subclasses. `Exception` is the base for all non-system-exiting exceptions (ValueError, TypeError, etc.). `ArithmeticError`, `LookupError`, `EnvironmentError` are mid-level groupings. The hierarchy helps you catch exceptions at the right level of specificity.

### Advanced

1. **Q**: How does Python's exception handling work at the interpreter level?
   **A**: When an exception occurs, Python unwinds the call stack looking for matching `except` blocks. Each frame's `f_lineno` is updated, and the traceback objects (`tb_frame`, `tb_lineno`, `tb_next`) are built as the stack unwinds. If no handler is found, the interpreter prints the traceback and exits. The `sys.excepthook` function controls the unhandled exception output.

2. **Q**: What are the performance implications of try/except blocks?
   **A**: The `try` block itself has negligible overhead when no exception is raised (Python uses a "fast path" with a jump table). The performance cost comes only when an exception is actually raised — raising an exception involves constructing the exception object, building the traceback, and unwinding the stack, which is relatively expensive. Therefore, don't use exceptions for control flow in tight loops.

3. **Q**: How would you implement a retry decorator that handles transient failures with exponential backoff?
   **A**: Create a decorator that wraps the function in a loop. On catching specified exceptions, wait with exponential backoff (initially short, doubling each retry), optionally adding jitter. Maximum retries and delay caps prevent indefinite retries. The decorator re-raises the last exception after all retries are exhausted.

## Practice Problems

### Easy

1. **Safe Division**: Write a function that divides two numbers and returns the result, or returns `None` if division by zero or invalid types occur.

2. **Integer Input**: Write a function that asks the user for an integer and keeps asking until valid input is provided.

3. **File Reader**: Write a function that reads a file and returns its content, or returns an empty string if the file doesn't exist.

4. **List Access**: Write a function that safely accesses a list element by index, returning a default value if the index is out of range.

5. **Dictionary Lookup**: Write a function that looks up a key in a dictionary and returns the value, or a default if the key is missing.

### Medium

1. **Calculator**: Write a simple calculator that handles invalid operations (e.g., division by zero, invalid operators) with appropriate error messages.

2. **JSON Parser**: Write a function that safely parses JSON from a string, returning the parsed data or `None` with a descriptive error message.

3. **Nested Key Access**: Write a function that safely accesses a nested dictionary using a dot-separated path, raising a custom `PathNotFoundError` if the path doesn't exist.

4. **Retry Wrapper**: Write a function that retries another function up to N times if it raises a specific exception.

5. **Configuration Loader**: Write a config loader that tries multiple file paths (config.yaml, config.json, config.ini) and raises a custom `ConfigNotFoundError` if none exist.

### Hard

1. **Retry Decorator with Backoff**: Implement a decorator `@retry(max_attempts=3, delay=1, backoff=2)` that retries a function on failure with exponential backoff and jitter.

2. **Transaction Manager**: Implement a context manager that simulates a database transaction — it commits if no exception occurs and rolls back if an exception is raised.

3. **Graceful Pipeline**: Build a data processing pipeline where each stage catches and logs errors but continues processing, collecting errors at the end for reporting.

## Solutions

### Easy Solutions

**1. Safe Division**
```python
def safe_divide(a, b):
    try:
        return a / b
    except (ZeroDivisionError, TypeError):
        return None

print(safe_divide(10, 2), safe_divide(10, 0), safe_divide("a", 2))
# Output: 5.0 None None
```

**2. Integer Input**
```python
def get_integer(prompt="Enter an integer: "):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Invalid integer. Try again.")

# Usage: num = get_integer()
```

**3. File Reader**
```python
def read_file_safe(filename):
    try:
        with open(filename, "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""

print(repr(read_file_safe("nonexistent.txt")))
# Output: ''
```

**4. List Access**
```python
def safe_get(lst, index, default=None):
    try:
        return lst[index]
    except IndexError:
        return default

print(safe_get([10, 20, 30], 1), safe_get([10, 20, 30], 10, "N/A"))
# Output: 20 N/A
```

**5. Dictionary Lookup**
```python
def safe_lookup(d, key, default=None):
    try:
        return d[key]
    except KeyError:
        return default

print(safe_lookup({"a": 1, "b": 2}, "a"), safe_lookup({"a": 1}, "z"))
# Output: 1 None
```

### Medium Solutions

**1. Calculator**
```python
def calculator(a, op, b):
    try:
        if op == "+":
            return a + b
        elif op == "-":
            return a - b
        elif op == "*":
            return a * b
        elif op == "/":
            return a / b
        else:
            raise ValueError(f"Unknown operator: {op}")
    except ZeroDivisionError:
        return "Error: Division by zero"
    except TypeError:
        return "Error: Invalid operand type"

print(calculator(10, "/", 0), calculator(10, "+", "x"))
# Output: Error: Division by zero Error: Invalid operand type
```

**2. JSON Parser**
```python
import json

def safe_json_parse(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        return None

print(safe_json_parse('{"a": 1}'))
# Output: {'a': 1}
print(safe_json_parse('not json'))
# Output: JSON parse error: Expecting value: line 1 column 1 (char 0)
# Output: None
```

**3. Nested Key Access**
```python
class PathNotFoundError(Exception):
    pass

def deep_get(d, path):
    keys = path.split(".")
    current = d
    for key in keys:
        try:
            current = current[key]
        except (KeyError, TypeError, IndexError):
            raise PathNotFoundError(f"Path '{path}' not found at key '{key}'")
    return current

data = {"a": {"b": {"c": 42}}}
print(deep_get(data, "a.b.c"))
# Output: 42
try:
    deep_get(data, "a.x.y")
except PathNotFoundError as e:
    print(e)
# Output: Path 'a.x.y' not found at key 'x'
```

**4. Retry Wrapper**
```python
import time

def retry(func, max_attempts=3, exceptions=(Exception,)):
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except exceptions as e:
            if attempt == max_attempts:
                raise
            print(f"Attempt {attempt} failed: {e}. Retrying...")
            time.sleep(0.5)

# Usage:
# def unstable_network_call():
#     if random.random() < 0.7:
#         raise ConnectionError("Timeout")
#     return "Success"
# result = retry(unstable_network_call, max_attempts=5)
```

**5. Configuration Loader**
```python
import json
import yaml  # Assume PyYAML is installed

class ConfigNotFoundError(Exception):
    pass

def load_config(paths):
    for path in paths:
        try:
            with open(path) as f:
                if path.endswith(".json"):
                    return json.load(f)
                elif path.endswith(".yaml") or path.endswith(".yml"):
                    return yaml.safe_load(f)
                else:
                    return f.read()
        except FileNotFoundError:
            continue
    raise ConfigNotFoundError(f"None of the paths exist: {paths}")

# try:
#     config = load_config(["config.json", "config.yaml", "config.ini"])
# except ConfigNotFoundError as e:
#     print(e)
```

### Hard Solutions

**1. Retry Decorator with Backoff**
```python
import time
import random
from functools import wraps

def retry(max_attempts=3, delay=1, backoff=2, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise
                    jitter = random.uniform(0, current_delay)
                    print(f"Attempt {attempt}/{max_attempts} failed: {e}")
                    print(f"Retrying in {current_delay + jitter:.2f}s...")
                    time.sleep(current_delay + jitter)
                    current_delay *= backoff
            return None
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5, backoff=2, exceptions=(ConnectionError,))
def unstable_request():
    import random
    if random.random() < 0.6:
        raise ConnectionError("Server timeout")
    return "OK"

# result = unstable_request()
```

**2. Transaction Manager**
```python
class TransactionError(Exception):
    pass

class Transaction:
    def __init__(self):
        self.state = "initial"
        self.operations = []

    def execute(self, op_name, func, *args, **kwargs):
        self.operations.append((op_name, func, args, kwargs))

    def __enter__(self):
        self.state = "active"
        print("Transaction started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.state = "committed"
            print("Transaction committed")
            return False
        else:
            self.state = "rolled_back"
            print(f"Transaction rolled back due to {exc_type.__name__}: {exc_val}")
            return True  # suppress exception

# Usage:
# with Transaction() as tx:
#     tx.execute("insert", db.insert, {"id": 1, "name": "Alice"})
#     tx.execute("update", db.update, {"id": 1, "balance": 100})
```

**3. Graceful Pipeline**
```python
class PipelineError:
    def __init__(self, stage, item, exception):
        self.stage = stage
        self.item = item
        self.exception = exception

    def __repr__(self):
        return f"[{self.stage}] Failed on {self.item}: {self.exception}"

def pipeline_pipeline(data, stages):
    errors = []
    results = []

    for item in data:
        current = item
        failed = False
        for stage_name, stage_func in stages:
            try:
                current = stage_func(current)
            except Exception as e:
                errors.append(PipelineError(stage_name, item, e))
                failed = True
                break
        if not failed:
            results.append(current)

    return results, errors

# Usage:
# def clean(s):
#     return s.strip().lower()
# def parse(s):
#     name, age = s.split(",")
#     return {"name": name, "age": int(age)}
# def validate(rec):
#     if rec["age"] < 0:
#         raise ValueError("Negative age")
#     return rec
#
# data = ["Alice,30", "Bob,invalid", "Charlie,-5", "Diana,25"]
# stages = [("clean", clean), ("parse", parse), ("validate", validate)]
# results, errors = run_pipeline(data, stages)
# print(f"Success: {results}")
# print(f"Errors: {errors}")
```

## Related Concepts

- Context managers (with statement) — built on exception handling
- Logging — recording exceptions for debugging
- Debugging and tracebacks — understanding error output
- Unit testing — testing error paths with pytest's `pytest.raises`
- Type hints and validation — preventing errors at boundaries
- Cleanup patterns (try/finally) — resource management

## Next Concepts

- File I/O (reading/writing files with proper error handling)
- Context managers and the `with` statement
- Custom context managers with `__enter__` and `__exit__`
- Logging module for production error tracking
- Testing exceptions with pytest

## Summary

Python's exception handling system allows programs to detect and respond to runtime errors gracefully. The `try/except/else/finally` block provides comprehensive control over error recovery. Specific exception types enable precise error handling, while custom exception classes create domain-specific error hierarchies. Exception chaining preserves error context during abstraction layers. Assertions serve as debugging aids and invariant documentation. Following EAFP (Easier to Ask Forgiveness than Permission) leads to cleaner, more robust code. In AI/ML pipelines, proper error handling is critical for reliable data processing and long-running training jobs.

## Key Takeaways

- Use `try/except` for error handling, never `except:` (bare) — specify exception types
- `else` runs when no exception occurs; `finally` always runs (cleanup)
- Raise specific built-in exceptions or custom exception subclasses of `Exception`
- Use `raise ... from` for explicit exception chaining
- Custom exceptions make code more readable and debuggable
- Assertions (`assert`) are for debugging and invariants, not input validation
- EAFP (try first, handle errors) is Pythonic; LBYL (if checks) is less so
- Be specific in catch clauses — too broad hides bugs, too narrow crashes on unexpected errors
- Never swallow exceptions silently; log them at minimum
- Context managers (`with`) handle resource cleanup even when exceptions occur
