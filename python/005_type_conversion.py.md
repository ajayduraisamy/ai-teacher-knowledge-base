# Concept: Type Conversion

## Concept ID

PYT-005

## Difficulty

BEGINNER

## Domain

Python

## Module

Python Basics

## Learning Objectives

After completing this lesson, you will be able to:

- Convert between data types using `int()`, `float()`, `str()`, `bool()`, `list()`, `tuple()`, `set()`, and `dict()`
- Distinguish between implicit and explicit type conversion
- Handle common conversion errors (`ValueError`, `TypeError`)
- Apply safe conversion patterns to avoid runtime errors
- Use type conversion for input validation and data cleaning
- Understand when Python automatically converts types and when explicit conversion is required

## Prerequisites

- Understanding of variables and data types (PYT-001)
- Familiarity with numbers (PYT-002), strings (PYT-003), and booleans (PYT-004)
- Basic understanding of collections: lists, tuples, sets, dictionaries

## Definition

**Type conversion**, also called **type casting**, is the process of converting a value from one data type to another. **Implicit conversion** (coercion) happens automatically when Python determines that a conversion is safe and necessary. **Explicit conversion** is performed by the programmer using built-in functions like `int()`, `float()`, `str()`, and `bool()`. Successful conversion depends on the source value being compatible with the target type — converting `"123"` to an integer works, but converting `"hello"` to an integer raises a `ValueError`.

## Intuition

Think of type conversion like translating between languages. The number `42` in "English" (integer) can be translated to "Spanish" (string) as `"42"`, and then "Spanish" can be translated back to "English" as `42`. But the word `"hello"` cannot be translated into "Math" (integer) — it has no numeric meaning. Some translations are automatic (like when a float country speaks to an integer country, they both use float), but most require an explicit translator (the conversion function).

## Why This Concept Matters

In real-world programming, data rarely arrives in the type you need. User input comes as strings but must become integers for calculation. API responses contain strings that need to become floats for analysis. Files store text that must become booleans for configuration. Understanding which conversions are valid, which are automatic, and how to handle failures is essential for building robust programs that gracefully handle unexpected data.

## Real World Examples

1. **Reading user age from input** — `input()` returns a string; convert to `int` for age calculations.
2. **Parsing a CSV file** — all values are strings initially; convert numbers to `int` or `float`.
3. **JSON API response parsing** — `json.loads()` returns strings, numbers, booleans, etc. in correct types already, but nested data often needs explicit conversion.
4. **Configuration files** — reading boolean values like `"true"` from `.env` or config files and converting to Python `bool`.
5. **Database queries** — converting query results (often tuples) into dictionaries for easier access.

## AI/ML Relevance

Type conversion is critical in data preprocessing pipelines for machine learning.

- **Loading datasets** — CSV/Excel files load everything as strings; numeric columns need conversion to `float` before training.
- **Feature engineering** — converting categorical string labels to integer codes for model consumption.
- **Data cleaning** — handling missing values represented as `"N/A"` or `"null"` strings, converting them to `None` or `np.nan`.
- **Normalization** — converting integer pixel values (0-255) to float (0.0-1.0) for neural network input.
- **Boolean feature creation** — converting conditions like `income > 50000` from a comparison to a boolean column (0/1).

```python
# AI/ML Example: Loading and converting a simple dataset
raw_data = [
    "Alice,25,55000",
    "Bob,30,62000",
    "Charlie,35,48000"
]

processed = []
for row in raw_data:
    name, age_str, income_str = row.split(",")
    age = int(age_str)         # str -> int
    income = float(income_str) # str -> float
    processed.append({"name": name, "age": age, "income": income})

print(processed)
# Output:
# [{'name': 'Alice', 'age': 25, 'income': 55000.0},
#  {'name': 'Bob', 'age': 30, 'income': 62000.0},
#  {'name': 'Charlie', 'age': 35, 'income': 48000.0}]

# Normalize pixel values (image processing)
pixel_value = 128
normalized = pixel_value / 255.0  # implicit int -> float
print(f"{pixel_value} -> {normalized:.4f}")
# Output: 128 -> 0.5020
```

## Code Examples

### Example 1: Implicit Type Conversion (Coercion)

```python
# Python automatically converts in safe situations

# int + float -> float
result = 5 + 3.2
print(result, type(result))  # 8.2 <class 'float'>

# int + complex -> complex
result = 5 + (2 + 3j)
print(result, type(result))  # (7+3j) <class 'complex'>

# Boolean in arithmetic (True=1, False=0)
print(True + 5)     # 6
print(False * 10)   # 0

# Division always returns float
print(10 / 2, type(10 / 2))  # 5.0 <class 'float'>

# Implicit conversion does NOT work for str + int
# print("Age: " + 25)  # TypeError: can only concatenate str (not "int") to str
```

### Example 2: Explicit Conversion — `int()`

```python
# Valid string to int
print(int("42"))          # 42
print(int("  -10  "))     # -10 (whitespace stripped)

# Float to int (truncates toward zero)
print(int(3.14))          # 3
print(int(-3.14))         # -3
print(int(5.999))         # 5 (does NOT round)

# Base conversion (binary, hex, octal)
print(int("1010", 2))     # 10 (binary to int)
print(int("FF", 16))      # 255 (hex to int)
print(int("77", 8))       # 63 (octal to int)

# Invalid conversions raise ValueError
# int("hello")      # ValueError: invalid literal for int()
# int("3.14")       # ValueError: invalid literal for int()
```

### Example 3: Explicit Conversion — `float()`

```python
# Valid string to float
print(float("3.14"))      # 3.14
print(float("  -2.5  "))  # -2.5
print(float("1e-3"))      # 0.001 (scientific notation)

# Integer to float
print(float(42))          # 42.0

# Special values
print(float("inf"))       # inf
print(float("-inf"))      # -inf
print(float("nan"))       # nan

# Invalid conversions
# float("hello")    # ValueError
# float("1,5")      # ValueError (use decimal point, not comma)
```

### Example 4: Explicit Conversion — `str()`

```python
# str() can convert almost anything
print(str(42))            # "42"
print(str(3.14))          # "3.14"
print(str(True))          # "True"
print(str(None))          # "None"
print(str([1, 2, 3]))     # "[1, 2, 3]"
print(str({"a": 1}))       # "{'a': 1}"

# Useful for concatenation
age = 30
message = "I am " + str(age) + " years old."
print(message)            # "I am 30 years old."
```

### Example 5: Explicit Conversion — `bool()`

```python
# False-y values
print(bool(0))            # False
print(bool(0.0))          # False
print(bool(""))           # False
print(bool([]))           # False
print(bool(None))         # False
print(bool({}))           # False

# Truth-y values
print(bool(1))            # True
print(bool(-1))           # True
print(bool(42))           # True
print(bool("hello"))      # True
print(bool("False"))      # True (non-empty string!)
print(bool([0]))          # True (non-empty list)
print(bool(" "))          # True (space is a character)

# Beware: bool("False") is True
# For parsing config values, use explicit checks
config_value = "false"
is_enabled = config_value.lower() == "true"
print(is_enabled)         # False
```

### Example 6: Collection Type Conversions

```python
# str -> list (split into characters)
print(list("hello"))      # ['h', 'e', 'l', 'l', 'o']

# str -> list (split by delimiter)
print("a,b,c".split(",")) # ['a', 'b', 'c']

# list -> tuple (immutable)
print(tuple([1, 2, 3]))   # (1, 2, 3)

# tuple -> list (mutable)
print(list((1, 2, 3)))    # [1, 2, 3]

# list -> set (removes duplicates, loses order)
print(set([1, 2, 2, 3, 3, 3]))  # {1, 2, 3}

# set -> list (back to list, order not guaranteed)
print(list({3, 1, 2}))    # [1, 2, 3] (or similar)

# list of pairs -> dict
pairs = [("name", "Alice"), ("age", 30)]
print(dict(pairs))        # {'name': 'Alice', 'age': 30}

# zip to dict
keys = ["a", "b", "c"]
values = [1, 2, 3]
print(dict(zip(keys, values)))  # {'a': 1, 'b': 2, 'c': 3}
```

### Example 7: Safe Conversion Patterns

```python
# Pattern 1: Try-except for safe conversion
def safe_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

print(safe_int("42"))           # 42
print(safe_int("hello"))        # 0
print(safe_int(None))           # 0
print(safe_int("3.14"))         # 0 (cannot convert float string to int directly)

# Pattern 2: Safe float
def safe_float(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

print(safe_float("3.14"))       # 3.14
print(safe_float("invalid"))    # 0.0

# Pattern 3: Parsing comma-separated numbers
def parse_numbers(text, separator=","):
    parts = text.split(separator)
    result = []
    for part in parts:
        try:
            result.append(float(part.strip()))
        except ValueError:
            continue  # skip invalid entries
    return result

print(parse_numbers("1,2.5,abc,3.7"))
# Output: [1.0, 2.5, 3.7]
```

### Example 8: Converting Between `str` and `bytes`

```python
# str -> bytes (encoding)
text = "Hello, 世界"
encoded = text.encode("utf-8")
print(encoded)            # b'Hello, \xe4\xb8\x96\xe7\x95\x8c'
print(type(encoded))      # <class 'bytes'>

# bytes -> str (decoding)
decoded = encoded.decode("utf-8")
print(decoded)            # Hello, 世界

# Common encodings
print("ABC".encode("ascii"))       # b'ABC'
print("ABC".encode("utf-16"))      # b'\xff\xfeA\x00B\x00C\x00'

# Errors in encoding/decoding
# "Hello 世界".encode("ascii")  # UnicodeEncodeError
# Safe encoding
print("Hello 世界".encode("ascii", errors="ignore"))   # b'Hello '
print("Hello 世界".encode("ascii", errors="replace"))  # b'Hello ??'
```

## Common Mistakes

### 1. Trying to convert an invalid string to int/float
```python
# ValueError: invalid literal for int() with base 10: '3.14'
# int("3.14")   # Wrong — use float() first, then int()

# Correct approach:
value = int(float("3.14"))  # 3

# ValueError: 'hello' has no numeric meaning
# int("hello")  # Always fails
```

### 2. Using comma instead of decimal point
```python
# float("3,14")  # ValueError
# European decimal format uses commas
# Correct: float("3.14")
```

### 3. Assuming `bool("False")` is False
```python
print(bool("False"))    # True! (non-empty string)
print(bool(""))         # False (only empty string is falsy)

# For string-to-boolean parsing:
str_bool = "false"
actual_bool = str_bool.lower() in ("true", "1", "yes")
```

### 4. Forgetting that `int()` truncates, not rounds
```python
print(int(3.99))    # 3 (not 4)
print(int(-3.99))   # -3 (not -4)
# Use round() for rounding behavior
print(round(3.99))  # 4
```

### 5. Converting a float string to int directly
```python
# Wrong:
# int("3.14")     # ValueError

# Correct two-step:
int(float("3.14"))  # 3
```

### 6. Assuming all strings can be converted
```python
# int("  10  ")   # Works (whitespace stripped)
# int("10 5")     # ValueError (contains space in the middle)
# int("10 years") # ValueError (non-numeric characters)
```

### 7. Losing data when converting collections
```python
# set removes duplicates
print(list(set([1, 2, 2, 3])))  # [1, 2, 3] — duplicate 2 lost

# dict requires pairs
# dict([1, 2, 3])  # ValueError: dictionary update sequence element #0 has length 1; 2 is required
```

## Interview Questions

### Beginner - 5

1. What is the difference between implicit and explicit type conversion in Python?
2. How do you convert a string `"123"` to an integer? What happens if you try to convert `"abc"`?
3. What does `int(3.7)` return? Why doesn't it return 4?
4. How do you convert a number to a string? Give an example of why you would need to.
5. What does `bool("False")` return? Why is this surprising?

### Intermediate - 5

1. Explain the difference between `int()` and `round()` for converting a float to a whole number.
2. How do you safely convert a user input string that might not be a valid number?
3. How does Python handle type conversion between `int` and `float`? Why does `5 + 3.2` return `8.2`?
4. How do you convert between string and bytes in Python? What is encoding?
5. How do you convert a list of tuples like `[("a", 1), ("b", 2)]` into a dictionary?

### Advanced - 3

1. Explain how Python's type conversion for numeric types follows the numeric tower (`int` -> `float` -> `complex`). How does `__int__`, `__float__`, and `__complex__` protocol work?
2. What is the difference between `str()` and `repr()` for non-string types? When would you use each?
3. How does Python's `json` module handle type conversion automatically? What types map to what JSON types?

## Practice Problems

### Easy - 5 Questions

**Problem 1:** Convert the string `"42"` to an integer and add 8 to it. Print the result.

**Problem 2:** Convert the integer `100` to a float, then to a string, then concatenate it with `" percent"`. Print the result.

**Problem 3:** Use `bool()` to check the truth value of `0`, `-1`, `""`, and `" "`.

**Problem 4:** Convert the list `[1, 2, 2, 3, 3, 3]` to a set and back to a list. Print the result.

**Problem 5:** Use `int()` with base 16 to convert the hex string `"1A"` to an integer.

### Medium - 5 Questions

**Problem 6:** Write a function `to_int_safe(value)` that tries to convert a value to int, returning `None` if it fails.

**Problem 7:** Parse the string `"10,20,30,40,50"` into a list of integers. Use a list comprehension.

**Problem 8:** Convert a list of two-element tuples `[("x", 5), ("y", 10), ("z", 15)]` into a dictionary, then back to a list of tuples.

**Problem 9:** Write a function `str_to_bool(value)` that returns `True` for `"true"`, `"yes"`, `"1"` (case-insensitive) and `False` otherwise.

**Problem 10:** Given a list of mixed types `[42, "3.14", "abc", 7.5, "10"]`, write a program that extracts only the numeric values (as floats) using safe conversion.

### Hard - 3 Questions

**Problem 11:** Write a function `convert_table(data)` that takes a list of lists (like from a CSV), where all values are strings, and automatically converts each cell to the most appropriate type (`int`, `float`, or keep as `str`).

**Problem 12:** Write a program that demonstrates implicit type conversion with `int`, `float`, `complex`, and `bool` in arithmetic operations. Explain each conversion.

**Problem 13:** Implement a safe converter that tries `int()`, then `float()`, then falls back to the original string. Handle edge cases like `None`, empty strings, and strings with surrounding whitespace.

## Solutions

### Problem 1
```python
result = int("42") + 8
print(result)  # 50
```

### Problem 2
```python
num = 100
f = float(num)
s = str(f)
print(s + " percent")  # "100.0 percent"
```

### Problem 3
```python
print(bool(0))    # False
print(bool(-1))   # True
print(bool(""))   # False
print(bool(" "))  # True
```

### Problem 4
```python
items = [1, 2, 2, 3, 3, 3]
unique = list(set(items))
print(unique)  # [1, 2, 3]
```

### Problem 5
```python
result = int("1A", 16)
print(result)  # 26
```

### Problem 6
```python
def to_int_safe(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

print(to_int_safe("42"))     # 42
print(to_int_safe("abc"))    # None
print(to_int_safe(None))     # None
```

### Problem 7
```python
data = "10,20,30,40,50"
numbers = [int(x) for x in data.split(",")]
print(numbers)  # [10, 20, 30, 40, 50]
```

### Problem 8
```python
pairs = [("x", 5), ("y", 10), ("z", 15)]
d = dict(pairs)
print(d)  # {'x': 5, 'y': 10, 'z': 15}
back = list(d.items())
print(back)  # [('x', 5), ('y', 10), ('z', 15)]
```

### Problem 9
```python
def str_to_bool(value):
    if not isinstance(value, str):
        return False
    return value.lower() in ("true", "yes", "1")

print(str_to_bool("True"))   # True
print(str_to_bool("false"))  # False
print(str_to_bool("YES"))    # True
print(str_to_bool("no"))     # False
print(str_to_bool("1"))      # True
```

### Problem 10
```python
data = [42, "3.14", "abc", 7.5, "10"]

def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

numerics = [safe_float(x) for x in data if safe_float(x) is not None]
print(numerics)  # [42.0, 3.14, 7.5, 10.0]
```

### Problem 11
```python
def convert_cell(value):
    """Convert a string cell to int, float, or str."""
    value = value.strip()
    if not value:
        return None
    # Try int first
    try:
        return int(value)
    except ValueError:
        pass
    # Try float
    try:
        return float(value)
    except ValueError:
        pass
    # Fall back to string
    return value

def convert_table(data):
    return [[convert_cell(cell) for cell in row] for row in data]

csv_data = [
    ["name", "age", "score"],
    ["Alice", "30", "95.5"],
    ["Bob", "25", "88.0"],
    ["Charlie", "35", "invalid"]
]

converted = convert_table(csv_data)
for row in converted:
    print(row)
# Output:
# ['name', 'age', 'score']
# ['Alice', 30, 95.5]
# ['Bob', 25, 88.0]
# ['Charlie', 35, 'invalid']
```

### Problem 12
```python
# Implicit conversions in arithmetic
a = 5       # int
b = 2.5     # float
c = 1 + 2j  # complex

# int + float -> float
print(f"{a} + {b} = {a + b}, type: {type(a + b)}")
# 5 + 2.5 = 7.5, type: <class 'float'>

# int + complex -> complex
print(f"{a} + {c} = {a + c}, type: {type(a + c)}")
# 5 + (1+2j) = (6+2j), type: <class 'complex'>

# float + complex -> complex
print(f"{b} + {c} = {b + c}, type: {type(b + c)}")
# 2.5 + (1+2j) = (3.5+2j), type: <class 'complex'>

# bool in arithmetic (True=1, False=0)
print(f"True + 5 = {True + 5}")    # 6
print(f"False * 3 = {False * 3}")  # 0
```

### Problem 13
```python
def safe_convert(value):
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return None
    # Try int
    try:
        return int(value)
    except ValueError:
        pass
    # Try float
    try:
        return float(value)
    except ValueError:
        pass
    # Return as-is
    return value

test_values = ["42", "3.14", "hello", "   ", None, "  -10  ", "1e3", "True"]
for v in test_values:
    result = safe_convert(v)
    print(f"'{v}' -> {result} ({type(result).__name__ if result is not None else 'NoneType'})")
# Output:
# '42' -> 42 (int)
# '3.14' -> 3.14 (float)
# 'hello' -> hello (str)
# '   ' -> None (NoneType)
# None -> None (NoneType)
# '  -10  ' -> -10 (int)
# '1e3' -> 1000.0 (float)
# 'True' -> True (str)
```

## Related Concepts

- **Variables and Data Types** — Understanding the types you are converting between
- **Numbers and Arithmetic** — Converting between numeric types for calculations
- **Strings** — Parsing text into other types
- **Booleans and Comparisons** — Boolean conversion and truthiness
- **Collections** — Converting between lists, tuples, sets, and dictionaries
- **Error Handling** — Try/except for safe conversion patterns

## Next Concepts

- **Conditional Statements** — Using converted values in `if` conditions
- **Loops** — Processing and converting data in loops
- **Functions** — Writing reusable conversion utilities

## Summary

Type conversion is the process of changing a value from one data type to another. Python performs **implicit conversion** (coercion) automatically when safe — for example, `int + float` yields `float`. **Explicit conversion** uses built-in functions: `int()`, `float()`, `str()`, `bool()`, `list()`, `tuple()`, `set()`, and `dict()`. Not all conversions are valid — trying to convert `"hello"` to `int` raises `ValueError`. Safe conversion patterns use try/except to handle failures gracefully, returning a default value or `None` instead of crashing. Understanding type conversion is essential for processing user input, file data, API responses, and any situation where data arrives in a different type than your program expects.

## Key Takeaways

- Implicit conversion happens automatically when safe (`int` -> `float` -> `complex`)
- Explicit conversion uses functions like `int()`, `float()`, `str()`, `bool()`
- `int("3.14")` fails — use `int(float("3.14"))` instead
- `int(3.99)` truncates toward zero (returns 3), does NOT round
- `bool("False")` is `True` — only empty string is falsy
- Collection conversions: `list()`, `tuple()`, `set()`, `dict()` convert between types
- `set()` removes duplicates; `dict()` needs pairs (tuples of length 2)
- Safe conversion: wrap in try/except for `ValueError` and `TypeError`
- String/bytes conversion requires encoding (use `.encode()` and `.decode()`)
- Always handle invalid conversions gracefully, never assume input is valid
