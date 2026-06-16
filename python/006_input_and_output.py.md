# Concept: Input and Output

## Concept ID

PYT-006

## Difficulty

BEGINNER

## Prerequisites

- PYT-001: Introduction to Python
- PYT-002: Variables and Data Types
- PYT-005: Operators and Expressions

## Domain

Python

## Module

Python Basics

## Learning Objectives

- Understand how to display output using the `print()` function with custom `sep`, `end`, and `file` parameters
- Read user input from the keyboard using the `input()` function and process the returned string
- Format strings using f-strings, `str.format()`, and %-formatting
- Write output to standard error (`stderr`) for error messages and diagnostics
- Use pretty printing (`pprint.pprint()`) to display complex data structures legibly

## Definition

Input and output (I/O) operations allow a program to communicate with the outside world. **Output** sends data from the program to a destination such as the console, a file, or an error stream. **Input** reads data from the user or external sources into the program for processing. Python provides built-in functions `print()` for output and `input()` for input, along with powerful string formatting tools.

## Intuition

Think of a Python program as a black box. You need a way to push data into the box (input) and a way to see what the box produces (output). Without I/O, a program is completely isolated — it cannot interact with users, log progress, or persist results. I/O is the bridge between your code and the real world.

## Why This Concept Matters

Every non-trivial program involves I/O. Command-line tools print results, web servers send HTTP responses, data pipelines write to files, and interactive applications prompt users. Mastering `print()`, `input()`, and formatting is essential before moving to files, networking, or user interfaces. Understanding the difference between `stdout` and `stderr` is crucial for writing professional command-line tools. Pretty printing helps during debugging and when presenting structured data.

## Real World Examples

1. A calculator reads two numbers and an operator from the user (input), computes the result, then prints it (output).
2. A password generator prints a random string with a customisable length and character set.
3. A data-cleaning script reads a CSV file and prints progress bars and statistics to `stdout` while logging warnings to `stderr`.
4. A command-line TODO app uses `input()` to accept new tasks and `print()` to display the list in a formatted table.
5. A DevOps monitoring script uses `pprint` to dump a dictionary of server metrics in a readable format.

## AI/ML Relevance

- **Training logs**: ML frameworks print loss, accuracy, and learning rate at each epoch using formatted output.
- **Hyperparameter configuration**: Dictionaries of hyperparameters are printed or logged using `pprint` for readability.
- **Data exploration**: Summaries of datasets (shape, columns, sample rows) are displayed with formatted `print()` calls.
- **Progress bars**: Libraries like `tqdm` build on `print()` and `sys.stdout` to show training progress.
- **Error handling**: Warnings about convergence failures or data issues are written to `stderr`.

## Code Examples

### Example 1: Basic `print()` and the `sep` / `end` parameters

The `print()` function accepts any number of positional arguments and writes them to the console. By default, items are separated by a space and a newline character is appended at the end. Both behaviours can be customised.

```python
print("Hello", "World", "from", "Python")
# Output: Hello World from Python

print("apple", "banana", "cherry", sep=", ")
# Output: apple, banana, cherry

print("Loading", end="...")
print("Done")
# Output: Loading...Done

print("a", "b", "c", sep=" | ", end="\n\n")
print("Section 2")
# Output: a | b | c
# Output:
# Output: Section 2
```

### Example 2: Reading input with `input()`

The `input()` function reads one line from the user and always returns a **string**. You must convert to the desired type using `int()`, `float()`, etc.

```python
name = input("Enter your name: ")
print(f"Hello, {name}!")

age_str = input("Enter your age: ")
age = int(age_str)
print(f"Next year you will be {age + 1} years old.")
```

### Example 3: Formatted output with f-strings

F-strings (formatted string literals) were introduced in Python 3.6. They allow inline expressions inside curly braces and support format specifiers.

```python
pi = 3.1415926535
print(f"Pi rounded to 2 decimals: {pi:.2f}")
# Output: Pi rounded to 2 decimals: 3.14

score = 95
total = 100
print(f"Score: {score}/{total} = {score / total:.1%}")
# Output: Score: 95/100 = 95.0%

value = 42
print(f"Pad left: [{value:>10}]")
# Output: Pad left: [        42]
```

### Example 4: `str.format()` and %-formatting

Older but still used in legacy code bases. `str.format()` uses `{}` placeholders, while %-formatting uses `%s`, `%d`, `%f`, etc.

```python
name = "Alice"
score = 92.5

print("Student: {}, Score: {}".format(name, score))
# Output: Student: Alice, Score: 92.5

print("Student: {n}, Score: {s:.1f}".format(n=name, s=score))
# Output: Student: Alice, Score: 92.5

print("Student: %s, Score: %.1f" % (name, score))
# Output: Student: Alice, Score: 92.5
```

### Example 5: Writing to `stdout` and `stderr`

The `sys.stdout` and `sys.stderr` file objects represent the standard output and standard error streams. Redirecting output to a file or logging errors separately is easier when you use `sys.stderr`.

```python
import sys

# Write to stdout (normal output)
print("This is standard output", file=sys.stdout)

# Write to stderr (error / diagnostic messages)
print("ERROR: Something went wrong!", file=sys.stderr)

# Redirect print output to a file
with open("output_log.txt", "w") as f:
    print("Log entry 1", file=f)
    print("Log entry 2", file=f)
# The file output_log.txt now contains two lines.
```

### Example 6: Pretty printing with `pprint`

When printing nested dictionaries, lists, or custom objects, the default `print()` output can be hard to read. `pprint` formats the data with indentation and sorted keys.

```python
from pprint import pprint

data = {
    "name": "model_v1",
    "layers": [
        {"type": "Dense", "units": 128, "activation": "relu"},
        {"type": "Dropout", "rate": 0.5},
        {"type": "Dense", "units": 10, "activation": "softmax"},
    ],
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
}

pprint(data, width=60, sort_dicts=False)
# Output: {'layers': [{'activation': 'relu',
# Output:               'type': 'Dense',
# Output:               'units': 128},
# Output:              {'rate': 0.5, 'type': 'Dropout'},
# Output:              {'activation': 'softmax',
# Output:               'type': 'Dense',
# Output:               'units': 10}],
# Output:  'loss': 'categorical_crossentropy',
# Output:  'name': 'model_v1',
# Output:  'optimizer': 'adam'}
```

## Common Mistakes

1. **Forgetting that `input()` returns a string** — performing arithmetic on the result without conversion raises a `TypeError`.
2. **Adding a space before `sep` or `end`** — `print("hello", sep=",")` does nothing because `sep` only applies between multiple arguments.
3. **Using `+` for concatenation without converting non-strings** — `print("The answer is " + 42)` raises `TypeError`. Use `f-strings` or `str()` instead.
4. **Assuming `print()` flushes immediately in all environments** — output may be buffered. Use `flush=True` when you need real-time output (progress bars).
5. **Using `%` formatting with mismatched types** — `"Value: %d" % "text"` raises `TypeError`. Always match the format specifier to the type.
6. **Confusing `sys.stdout` and `sys.stderr`** — both print to the console by default, but redirecting one does not redirect the other.
7. **Forgetting the trailing comma for single-element tuples with `pprint`** — Not a `pprint` issue, but common when printing data structures.

## Interview Questions

### Beginner

1. **Q:** What does the `print()` function do?  
   **A:** It writes one or more objects to the console (or a file), separated by `sep` and terminated by `end`.

2. **Q:** How do you read a number from the user?  
   **A:** Call `input()` to get a string, then convert with `int()` or `float()`.

3. **Q:** What is the default value of `sep` in `print()`?  
   **A:** A single space `' '`.

4. **Q:** How can you prevent `print()` from adding a newline at the end?  
   **A:** Set `end=""`.

5. **Q:** What is an f-string?  
   **A:** A formatted string literal prefixed with `f` that allows embedding expressions inside `{}`.

### Intermediate

1. **Q:** How do you redirect `print()` output to a file?  
   **A:** Pass `file=file_object` where `file_object` is an open file handle.

2. **Q:** Explain the difference between `str.format()` and f-strings.  
   **A:** `str.format()` is a method call on a template string; f-strings are evaluated at compile time and are generally faster and more readable.

3. **Q:** What does `flush=True` do in `print()`?  
   **A:** It forces the output buffer to be written immediately instead of waiting for buffer fill or program exit.

4. **Q:** How do you write to `stderr` without using `print()`?  
   **A:** Use `sys.stderr.write("message\n")` or `sys.stderr.writelines(lines)`.

5. **Q:** What is the purpose of `pprint`?  
   **A:** To print complex data structures (nested lists, dicts) in a human-readable, indented format.

### Advanced

1. **Q:** How would you capture `print()` output into a string?  
   **A:** Use `io.StringIO` as the `file` target:  
   ```python
   import io
   buf = io.StringIO()
   print("Hello", file=buf)
   result = buf.getvalue()
   ```

2. **Q:** Explain how `print()` handles Unicode and encoding.  
   **A:** `print()` encodes the string using `sys.stdout.encoding` (usually UTF-8). In terminals that do not support Unicode, errors may occur; setting `PYTHONIOENCODING` or using `errors='replace'` can help.

3. **Q:** Implement a custom `print`-like function that prepends a timestamp.  
   **A:**  
   ```python
   import sys
   from datetime import datetime

   def tprint(*args, **kwargs):
       timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
       print(timestamp, *args, **kwargs)
   ```

## Practice Problems

### Easy

1. **Greet User** — Prompt the user for their name and print a personalised greeting.
2. **Sum of Two Numbers** — Read two integers, print their sum.
3. **Rectangle Area** — Read width and height, print the area.
4. **Formatted Price** — Given a float `price`, print it with 2 decimal places and a `$` prefix.
5. **Pi Digits** — Print the first 5 digits of pi using f-string formatting.

### Medium

1. **Multiplication Table** — Read an integer `n` and print its 1-to-10 table in aligned columns.
2. **Log to File** — Write a function that accepts a message and logs it to `app.log` with a timestamp.
3. **Progress Bar** — Print a simple text progress bar (e.g., `[####    ] 40%`) that updates in place using `end="\r"`.
4. **Pretty Print Config** — Given a nested dictionary representing a config, use `pprint` to display it sorted by keys.
5. **CSV Formatter** — Read a list of rows (list of lists) and print it as a formatted table with aligned columns.

### Hard

1. **Custom Print with Levels** — Implement a `logger` function that supports `DEBUG`, `INFO`, `WARNING`, and `ERROR` levels, writing to `stdout` for `INFO` and `stderr` for `ERROR`.
2. **Interactive REPL** — Build a mini REPL (Read-Eval-Print Loop) that reads Python expressions, evaluates them, prints the result, and catches exceptions — all using `input()` and `print()`.
3. **Progress Bar Decorator** — Write a decorator that prints a progress bar around any iterable-based function.

## Solutions

### Easy

```python
# 1. Greet User
name = input("Enter your name: ")
print(f"Hello, {name}!")

# 2. Sum of Two Numbers
a = int(input())
b = int(input())
print(a + b)

# 3. Rectangle Area
w = float(input("Width: "))
h = float(input("Height: "))
print(f"Area: {w * h}")

# 4. Formatted Price
price = 49.956
print(f"${price:.2f}")
# Output: $49.96

# 5. Pi Digits
pi = 3.1415926535
print(f"{pi:.5f}")
# Output: 3.14159
```

### Medium

```python
# 1. Multiplication Table
n = int(input("Table for: "))
for i in range(1, 11):
    print(f"{n:3} x {i:2} = {n * i:4}")

# 2. Log to File
from datetime import datetime
def log_message(msg):
    with open("app.log", "a") as f:
        print(f"[{datetime.now()}] {msg}", file=f)

# 3. Progress Bar
import time
for i in range(1, 101):
    bar = "#" * (i // 4) + " " * (25 - i // 4)
    print(f"\r[{bar}] {i}%", end="")
    time.sleep(0.05)
print()

# 4. Pretty Print Config
from pprint import pprint
config = {"version": 2, "logging": {"formatters": {"simple": {"format": "%(message)s"}}}}
pprint(config, sort_dicts=True)

# 5. CSV Formatter
rows = [["Name", "Age", "City"], ["Alice", 30, "NYC"], ["Bob", 25, "LA"]]
for row in rows:
    print(f"{row[0]:<10} {row[1]:<5} {row[2]:<10}")
```

### Hard

```python
# 1. Custom Print with Levels
import sys
def log(level, msg):
    if level == "ERROR":
        print(f"[{level}] {msg}", file=sys.stderr)
    else:
        print(f"[{level}] {msg}")

log("INFO", "Server started")
log("ERROR", "Disk full")

# 2. Mini REPL
while True:
    expr = input(">>> ")
    if expr.strip().lower() == "exit":
        break
    try:
        result = eval(expr)
        print(repr(result))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

# 3. Progress Bar Decorator
def progress(iterable):
    items = list(iterable)
    total = len(items)
    for i, item in enumerate(items, 1):
        pct = i / total * 100
        bar = "#" * (int(pct) // 4)
        print(f"\r[{bar:<25}] {pct:.0f}%", end="")
        yield item
    print()

for _ in progress(range(50)):
    pass
```

## Related Concepts

- **Variables and Data Types** — The values you print and read are variables of specific types.
- **Operators and Expressions** — Expressions inside f-string braces use operators.
- **File I/O** — `print()` with `file=` is a bridge to file operations.
- **String Methods** — `str.format()` is a string method.

## Next Concepts

- **Conditionals** — Control flow to react to user input.
- **Loops** — Repeatedly read/process data.
- **Functions** — Encapsulate I/O logic in reusable units.
- **Exception Handling** — Gracefully handle bad input.

## Summary

Python's `print()` and `input()` functions provide simple but powerful I/O. The `sep`, `end`, and `file` parameters make `print()` flexible. String formatting via f-strings, `str.format()`, and %-formatting allows clean output. Writing to `stderr` is important for robust CLI tools, and `pprint` makes debugging complex data easy. Understanding I/O is the first step toward building interactive and professional Python applications.

## Key Takeaways

- `print()` accepts `sep` (separator between arguments) and `end` (line terminator).
- `input()` always returns a string; convert types explicitly.
- F-strings (Python 3.6+) are the preferred way to format strings.
- Use `file=sys.stderr` to write error messages to the error stream.
- `pprint.pprint()` produces human-readable output for nested data structures.
- Use `flush=True` for real-time output when buffering is undesirable.
