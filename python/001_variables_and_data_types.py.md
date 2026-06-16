# Concept: Variables and Data Types

## Concept ID

PYT-001

## Difficulty

BEGINNER

## Domain

Python

## Module

Python Basics

## Learning Objectives

After completing this lesson, you will be able to:

- Declare and assign variables using proper Python syntax
- Identify and use the fundamental data types: `int`, `float`, `str`, `bool`, and `NoneType`
- Use the `type()` function to inspect the data type of any value
- Apply Python's dynamic typing to reassign variables to different types
- Follow standard naming conventions including snake_case and constants
- Perform multiple assignments in a single line of code
- Understand variable immutability for specific types

## Prerequisites

- No prior programming experience required
- Python 3.x installed on your system
- A text editor or IDE (such as VS Code, PyCharm, or IDLE)
- Basic familiarity with the command line or terminal

## Definition

A **variable** is a named reference to a value stored in memory. In Python, variables act as labels that point to objects. The **data type** of a variable determines what kind of data it holds — numbers, text, boolean values, or more complex structures — and what operations can be performed on it.

Python is a **dynamically typed** language, meaning you do not need to declare the type of a variable explicitly. The interpreter infers the type from the value you assign.

## Intuition

Think of variables as sticky notes you place on different boxes. Each sticky note has a name (the variable name), and each box contains some data (the value). You can peel the note off one box and stick it onto a completely different box — this is exactly what happens when you reassign a variable in Python. The type of data in the new box can be entirely different from the old one, and Python handles this seamlessly.

## Why This Concept Matters

Variables and data types are the absolute foundation of every Python program. Without variables, you cannot store, retrieve, or manipulate data. Understanding data types is critical because each type supports different operations and behaves differently under the hood. A string supports concatenation with `+`, while a number supports arithmetic — mixing them up leads to errors. Mastering this topic early prevents countless bugs and builds the mental model needed for everything that follows: loops, conditionals, functions, classes, and data analysis.

## Real World Examples

1. **Storing a user's name and age** in a registration form — name as a string, age as an integer.
2. **Tracking a product price** on an e-commerce site — price as a float to support decimal values.
3. **Checking if a light is on or off** in a smart home system — boolean `True` or `False`.
4. **A sensor that has not returned a reading yet** — represented as `None` to indicate missing data.
5. **A database connection status** — a boolean indicating connected or disconnected.

## AI/ML Relevance

In artificial intelligence and machine learning, variables store everything from raw input data to trained model parameters.

- **Feature vectors** are typically stored as lists of floats or NumPy arrays.
- **Labels** for classification tasks are often integers (e.g., `0` for cat, `1` for dog) or strings.
- **Model weights and biases** are floating-point numbers stored in tensors.
- **Hyperparameters** like learning rate, number of epochs, and batch size are stored as float and int variables.
- **Flags** such as `is_training` or `use_dropout` are booleans that control model behavior.

```python
# AI/ML Example: Storing ML hyperparameters
learning_rate = 0.001       # float
num_epochs = 50             # int
model_name = "resnet50"     # str
use_pretrained = True       # bool
best_accuracy = None        # NoneType — not yet evaluated

print(f"Training {model_name} for {num_epochs} epochs")
print(f"Learning rate: {learning_rate}, Use pretrained: {use_pretrained}")
print(f"Best accuracy so far: {best_accuracy}")
# Output:
# Training resnet50 for 50 epochs
# Learning rate: 0.001, Use pretrained: True
# Best accuracy so far: None
```

## Code Examples

### Example 1: Basic Variable Assignment and Data Types

```python
# Assigning values to variables
name = "Alice"           # str
age = 25                 # int
height = 5.6             # float
is_student = True        # bool
middle_name = None       # NoneType

print(name)
print(age)
print(height)
print(is_student)
print(middle_name)
# Output:
# Alice
# 25
# 5.6
# True
# None
```

### Example 2: Using the type() Function

```python
# Inspecting types with type()
print(type(42))            # <class 'int'>
print(type(3.14))          # <class 'float'>
print(type("Hello"))       # <class 'str'>
print(type(True))          # <class 'bool'>
print(type(None))          # <class 'NoneType'>
print(type([1, 2, 3]))     # <class 'list'>
```

### Example 3: Dynamic Typing — Reassigning Variables

```python
# Python allows a variable to change type at runtime
value = 100
print(f"Value is {value}, type is {type(value)}")

value = "Now I am a string!"
print(f"Value is {value}, type is {type(value)}")

value = 3.14
print(f"Value is {value}, type is {type(value)}")

value = False
print(f"Value is {value}, type is {type(value)}")
# Output:
# Value is 100, type is <class 'int'>
# Value is Now I am a string!, type is <class 'str'>
# Value is 3.14, type is <class 'float'>
# Value is False, type is <class 'bool'>
```

### Example 4: Variable Naming Conventions and Multiple Assignment

```python
# snake_case naming convention (PEP 8)
first_name = "John"
last_name = "Doe"
user_email_address = "john@example.com"
total_price = 49.99

# Multiple assignment in one line
x, y, z = 10, 20, 30
print(x, y, z)             # 10 20 30

# Swapping variables (Pythonic way)
a, b = 5, 10
a, b = b, a
print(f"a = {a}, b = {b}")  # a = 10, b = 5

# Assigning the same value to multiple variables
p = q = r = 0
print(p, q, r)             # 0 0 0
```

### Example 5: Constants Convention and Type Checking

```python
# Constants are written in UPPER_CASE by convention
# Python does not enforce const — it's a convention
PI = 3.14159
MAX_RETRIES = 5
DEFAULT_TIMEOUT = 30.0
API_ENDPOINT = "https://api.example.com/v1"

# Type checking with isinstance()
print(isinstance(42, int))       # True
print(isinstance(3.14, float))   # True
print(isinstance("Hi", str))     # True
print(isinstance(True, bool))    # True
print(isinstance([], list))      # True

# Checking against multiple types
print(isinstance(42, (int, float)))  # True
```

### Example 6: Deleting Variables

```python
# The del statement removes a variable reference
temp = "This will be deleted"
print(temp)

del temp
# print(temp)  # Would raise NameError: name 'temp' is not defined

# Variables can also go out of scope naturally
```

## Common Mistakes

### 1. Using undefined variable names
```python
# This raises NameError
# print(undefined_variable)
```

### 2. Confusing assignment `=` with equality `==`
```python
# Wrong: if x = 5:   (assignment)
# Correct: if x == 5: (comparison)
```

### 3. Using reserved keywords as variable names
```python
# Invalid variable names
# class = "Math"     # 'class' is a keyword
# for = "loop"       # 'for' is a keyword
# True = 1           # 'True' is a keyword
```

### 4. Starting a variable name with a number
```python
# Invalid
# 1st_place = "Alice"
# Correct
first_place = "Alice"
```

### 5. Forgetting that variables are references, not boxes
```python
a = [1, 2, 3]
b = a          # b is a reference to the same list
b.append(4)
print(a)       # [1, 2, 3, 4] — a changed too!
```

### 6. Mixing incompatible types in operations
```python
# Raises TypeError
# result = "Age: " + 25
# Correct
result = "Age: " + str(25)
```

## Interview Questions

### Beginner - 5

1. What is the difference between `=` and `==` in Python?
2. List five basic data types in Python and give an example of each.
3. What does the `type()` function do? Provide a code example.
4. What is dynamic typing in Python?
5. How do you swap two variables in Python without using a temporary variable?

### Intermediate - 5

1. Explain the difference between a mutable and immutable data type. Give examples.
2. What is the difference between `is` and `==`? When should you use each?
3. How does Python handle variable scope? Explain LEGB rule.
4. What are `None`, `True`, and `False` in Python — are they variables or constants?
5. Explain what happens when you use `del` on a variable. Can you delete a variable that was created in a different scope?

### Advanced - 3

1. How does Python's memory management work for small integers and string interning? What is the range of integers that are cached?
2. Explain the concept of "duck typing" and how it relates to Python's variable type system.
3. How does garbage collection work in CPython? Explain reference counting and the generational garbage collector.

## Practice Problems

### Easy - 5 Questions

**Problem 1:** Create three variables `city`, `population`, and `area_sq_km`. Assign them a string, integer, and float respectively. Print all three.

**Problem 2:** Use the `type()` function to print the type of `100`, `10.5`, `"Python"`, and `False`.

**Problem 3:** Create a variable `score` and assign it `95`. Then reassign it to `"Ninety-five"` and print the new value and type.

**Problem 4:** Use multiple assignment to create variables `a`, `b`, `c` with values `1`, `2`, `3` in one line. Print their sum.

**Problem 5:** Swap the values of two variables `x = 10` and `y = 20` using Python's multiple assignment syntax.

### Medium - 5 Questions

**Problem 6:** Write a program that takes a user's name and age as input, stores them in variables, and prints a sentence using f-strings.

**Problem 7:** Create three variables representing the length, width, and height of a box. Calculate and print the volume. Ensure length and width are integers, height is a float.

**Problem 8:** Write code that demonstrates how two variables can reference the same list and how modifying one affects the other.

**Problem 9:** Create a variable `count` and set it to `None`. Then create an `if` statement that checks if `count is None` and prints "No count available" if true.

**Problem 10:** Write a program that uses `isinstance()` to check if a variable `data` is either an `int` or a `float`. Test with `data = 42.0`.

### Hard - 3 Questions

**Problem 11:** Explain and demonstrate what happens when you assign a variable to another variable, then change the original. Do this for both mutable (list) and immutable (int) types.

**Problem 12:** Write a program that demonstrates Python's integer caching by using the `is` operator to compare small integers (-5 to 256) and large integers.

**Problem 13:** Create a system of variables that simulates a simple bank account (name, balance, account_number, is_active). Write code that shows variable reassignment, deleting a variable, and the effect of mutability.

## Solutions

### Problem 1
```python
city = "Tokyo"
population = 13960000
area_sq_km = 2194.0
print(city, population, area_sq_km)
# Output: Tokyo 13960000 2194.0
```

### Problem 2
```python
print(type(100))       # <class 'int'>
print(type(10.5))      # <class 'float'>
print(type("Python"))  # <class 'str'>
print(type(False))     # <class 'bool'>
```

### Problem 3
```python
score = 95
score = "Ninety-five"
print(score)           # Ninety-five
print(type(score))     # <class 'str'>
```

### Problem 4
```python
a, b, c = 1, 2, 3
print(a + b + c)       # 6
```

### Problem 5
```python
x, y = 10, 20
x, y = y, x
print(x, y)            # 20 10
```

### Problem 6
```python
name = input("Enter your name: ")
age = int(input("Enter your age: "))
print(f"Hello {name}, you are {age} years old.")
```

### Problem 7
```python
length = 10
width = 5
height = 3.5
volume = length * width * height
print(f"Volume: {volume}")
# Output: Volume: 175.0
```

### Problem 8
```python
list1 = [1, 2, 3]
list2 = list1
list2.append(4)
print(list1)  # [1, 2, 3, 4]
```

### Problem 9
```python
count = None
if count is None:
    print("No count available")
# Output: No count available
```

### Problem 10
```python
data = 42.0
if isinstance(data, (int, float)):
    print(f"{data} is a number")
# Output: 42.0 is a number
```

### Problem 11
```python
# Immutable example
x = 5
y = x
x = 10
print(x, y)  # 10 5 — y is unaffected

# Mutable example
a = [1, 2, 3]
b = a
a.append(4)
print(a, b)  # [1, 2, 3, 4] [1, 2, 3, 4] — b changed too!
```

### Problem 12
```python
a = 5
b = 5
print(a is b)  # True (cached)

c = 1000
d = 1000
print(c is d)  # False (not cached, though may be True in some implementations)
print(c == d)  # True (always correct way to compare)
```

### Problem 13
```python
name = "Alice"
balance = 1000.50
account_number = "ACC-12345"
is_active = True

print(f"Account: {account_number}, Name: {name}, Balance: ${balance}")

# Reassign
balance = 1500.75
print(f"New balance: ${balance}")

# Delete
del account_number
# print(account_number)  # Would raise NameError

# Mutability demo
transactions = ["Deposit $500", "Withdraw $200"]
history = transactions
transactions.append("Deposit $300")
print(history)  # ['Deposit $500', 'Withdraw $200', 'Deposit $300']
```

## Related Concepts

- **Operators** — Using variables with arithmetic and logical operators
- **Input/Output** — Reading data into variables and displaying results
- **Functions** — Passing variables as arguments and returning values
- **Scope** — Where variables are accessible in your code
- **Data Structures** — Lists, tuples, dictionaries, and sets hold collections of values

## Next Concepts

- **Numbers and Arithmetic** — Deep dive into numeric types and operations
- **Strings** — Text manipulation and methods
- **Type Conversion** — Converting between data types safely

## Summary

Variables in Python are named references to objects in memory. Python supports five fundamental data types: `int` (integers), `float` (decimal numbers), `str` (text), `bool` (True/False), and `NoneType` (absence of value). Python is dynamically typed — variables can change type on reassignment. The `type()` function reveals a value's data type, and `isinstance()` checks against one or more types. Variable names follow the snake_case convention, constants are written in UPPER_CASE by convention only, and multiple assignment enables elegant patterns like swapping values. Understanding variables and types is the essential first step to mastering Python.

## Key Takeaways

- Variables are labels referencing objects in memory
- Python has five basic types: `int`, `float`, `str`, `bool`, `NoneType`
- Use `type()` to inspect a value's type
- Python is dynamically typed — no need to declare types upfront
- Follow snake_case naming; use UPPER_CASE for constants by convention
- Multiple assignment allows elegant value swapping: `a, b = b, a`
- Use `isinstance()` for safe type checking
- Variables referencing mutable objects share the same underlying data
