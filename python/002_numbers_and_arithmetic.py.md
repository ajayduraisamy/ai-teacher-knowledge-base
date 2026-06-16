# Concept: Numbers and Arithmetic

## Concept ID

PYT-002

## Difficulty

BEGINNER

## Domain

Python

## Module

Python Basics

## Learning Objectives

After completing this lesson, you will be able to:

- Distinguish between `int` and `float` types and their use cases
- Use all seven arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- Understand and apply operator precedence (PEMDAS)
- Use the `math` module for advanced mathematical operations
- Work with complex numbers using `j` notation
- Understand Python's numeric type hierarchy and implicit type conversion
- Avoid common pitfalls with floating-point arithmetic

## Prerequisites

- Understanding of variables and data types (PYT-001)
- Basic mathematical knowledge (addition, subtraction, multiplication, division)
- Python 3.x installed

## Definition

Numbers in Python are objects that represent numeric values. Python provides three distinct numeric types: **integers** (`int`) for whole numbers, **floating-point numbers** (`float`) for decimal values, and **complex numbers** (`complex`) for values with real and imaginary parts. **Arithmetic operators** are symbols that perform mathematical operations on numeric operands and produce a result.

## Intuition

Think of integers as counting numbers — you use them when you need exact, whole values like the number of students in a class or the index of an item in a list. Floating-point numbers are measurements — they handle fractions and decimals, like a person's height in meters or the temperature outside. Just as a ruler has finite precision (you cannot measure infinitely small distances), floats have limited precision due to how computers represent them in binary.

## Why This Concept Matters

Virtually every program you write will involve numbers. Whether you are counting iterations in a loop, calculating statistics on a dataset, computing gradients in a neural network, or processing financial transactions, you rely on Python's numeric types and arithmetic operators. Understanding how integers and floats behave differently, how operator precedence affects your expressions, and how floating-point precision works will prevent subtle bugs and help you write efficient, correct code.

## Real World Examples

1. **Shopping cart total** — summing item prices (float) and applying a discount percentage.
2. **Age calculation** — subtracting birth year from current year (int).
3. **Averaging test scores** — dividing total score by number of tests.
4. **Compound interest** — using exponentiation `**` to compute growth over time.
5. **Splitting a bill** — integer division `//` to determine equal shares and modulus `%` to find remainder.

## AI/ML Relevance

Numbers are the lifeblood of AI and machine learning. Every algorithm is ultimately a sequence of mathematical operations on numeric data.

- **Gradient descent** uses floating-point arithmetic to update model weights by subtracting a fraction of the gradient.
- **Loss functions** like mean squared error involve squaring differences (`**`), summing them, and dividing.
- **Matrix multiplications** in neural networks consist of repeated multiplication and addition operations.
- **Normalization and scaling** transform features using division by the standard deviation.
- **Hyperparameter tuning** involves searching over learning rates (e.g., `0.001`, `0.0001`) and other numeric values.

```python
# AI/ML Example: Gradient descent update step
weight = 0.5
learning_rate = 0.01
gradient = 0.1

# Update weight using gradient descent
weight = weight - learning_rate * gradient
print(f"Updated weight: {weight}")
# Output: Updated weight: 0.499

# Mean squared error calculation
predictions = [0.8, 0.1, 0.6]
actuals = [1.0, 0.0, 1.0]
squared_errors = [(p - a) ** 2 for p, a in zip(predictions, actuals)]
mse = sum(squared_errors) / len(squared_errors)
print(f"Mean Squared Error: {mse}")
# Output: Mean Squared Error: 0.08666666666666667
```

## Code Examples

### Example 1: Integers and Floats — Creation and Type Checking

```python
# Integers
a = 10
b = -5
c = 0
large = 1_000_000  # Underscores improve readability

print(type(a))    # <class 'int'>
print(large)      # 1000000

# Floats
x = 3.14
y = -0.001
z = 1.0           # Even whole numbers written with decimal are floats
sci = 1.5e-3      # Scientific notation: 1.5 × 10^-3 = 0.0015

print(type(x))    # <class 'float'>
print(sci)        # 0.0015

# Check if a value is a specific numeric type
print(isinstance(10, int))      # True
print(isinstance(3.14, float))  # True
print(isinstance(10, (int, float)))  # True
```

### Example 2: Basic Arithmetic Operators

```python
# Addition
print(10 + 3)      # 13

# Subtraction
print(10 - 3)      # 7

# Multiplication
print(10 * 3)      # 30

# Division (always returns a float)
print(10 / 3)      # 3.3333333333333335
print(10 / 2)      # 5.0 (note: float, not int)

# Integer (floor) division
print(10 // 3)     # 3

# Modulus (remainder)
print(10 % 3)      # 1

# Exponentiation
print(2 ** 3)      # 8
print(4 ** 0.5)    # 2.0
```

### Example 3: How Different Operators Handle int vs float

```python
# int + int -> int
print(5 + 3)       # 8, type: int

# int + float -> float
print(5 + 3.0)     # 8.0, type: float

# int / int -> float (always!)
print(10 / 2)      # 5.0

# int // int -> int
print(10 // 3)     # 3

# float // float -> float
print(10.0 // 3.0) # 3.0

# Negative division
print(-10 // 3)    # -4 (floors toward negative infinity)
print(-10 % 3)     # 2 (remainder satisfies: a = (a//b)*b + (a%b))
```

### Example 4: Operator Precedence (PEMDAS)

```python
# Python follows standard mathematical precedence:
# 1. Parentheses ()
# 2. Exponentiation **
# 3. Unary +, -
# 4. Multiplication *, Division /, Floor division //, Modulus %
# 5. Addition +, Subtraction -

# Without parentheses — multiplication happens first
result = 5 + 3 * 2
print(result)     # 11 (not 16)

# With parentheses
result = (5 + 3) * 2
print(result)     # 16

# Exponentiation is right-associative
result = 2 ** 3 ** 2
print(result)     # 512 (2 ** (3 ** 2) = 2 ** 9)

# Chained exponentiation with parentheses
result = (2 ** 3) ** 2
print(result)     # 64
```

### Example 5: The math Module

```python
import math

# Constants
print(math.pi)      # 3.141592653589793
print(math.e)       # 2.718281828459045
print(math.tau)     # 6.283185307179586
print(math.inf)     # inf (infinity)
print(math.nan)     # nan (not a number)

# Rounding functions
print(math.ceil(4.2))    # 5
print(math.floor(4.2))   # 4
print(math.trunc(4.2))   # 4 (truncates toward zero)
print(round(4.5))        # 4 (banker's rounding — toward even)
print(round(5.5))        # 6

# Power and roots
print(math.sqrt(16))     # 4.0
print(math.pow(2, 10))   # 1024.0
print(math.isclose(0.1 + 0.2, 0.3))  # True (safe float comparison)
```

### Example 6: Complex Numbers

```python
# Creating complex numbers
z1 = 3 + 4j
z2 = complex(1, 2)  # 1 + 2j

print(type(z1))             # <class 'complex'>
print(z1)                   # (3+4j)

# Real and imaginary parts
print(z1.real)              # 3.0
print(z1.imag)              # 4.0

# Arithmetic with complex numbers
print(z1 + z2)              # (4+6j)
print(z1 * z2)              # (-5+10j)
print(z1 ** 2)              # (-7+24j)

# Magnitude (using abs)
print(abs(z1))              # 5.0 (sqrt(3^2 + 4^2))
```

### Example 7: Numeric Type Hierarchy and Implicit Conversion

```python
# Python's numeric hierarchy: int -> float -> complex
# Operations with mixed types promote to the wider type

# int + float -> float
result = 10 + 2.5
print(result, type(result))  # 12.5 <class 'float'>

# int + complex -> complex
result = 5 + (2 + 3j)
print(result, type(result))  # (7+3j) <class 'complex'>

# float + complex -> complex
result = 1.5 + (2 + 3j)
print(result, type(result))  # (3.5+3j) <class 'complex'>

# Explicit conversion (narrowing)
print(int(3.9))              # 3 (truncates toward zero)
print(float(5))              # 5.0
print(complex(3))            # (3+0j)
```

## Common Mistakes

### 1. Floating-point precision issues
```python
# Binary representation cannot represent all decimals exactly
print(0.1 + 0.2 == 0.3)          # False!
print(0.1 + 0.2)                  # 0.30000000000000004

# Fix: use math.isclose()
import math
print(math.isclose(0.1 + 0.2, 0.3))  # True
```

### 2. Integer division when you expect float division
```python
# Using / instead of // or vice versa
average = 10 / 3
print(average)      # 3.333... (float — correct for averages)

# But when you need integer result:
items_per_box = 10 // 3
print(items_per_box)  # 3 (correct: integer division)
```

### 3. Modulus with negative numbers
```python
print(-10 % 3)     # 2 (not -1)
print(10 % -3)     # -2 (surprising to many beginners)
```

### 4. Integer overflow confusion
```python
# Python 3 integers have arbitrary precision — no overflow
big = 10 ** 100
print(big)  # Works fine (unlike C or Java)

# But floats overflow to infinity
print(1e308)            # 1e308
# print(1e309)          # OverflowError or inf
```

### 5. Confusing `=` with `==` in expressions
```python
# Wrong (assignment inside expression):
# if x = 5:

# Correct (comparison):
x = 5
if x == 5:
    print("x is 5")
```

### 6. Misunderstanding floor division for negative numbers
```python
print(-10 // 3)     # -4 (floors down, not toward zero)
print(-10 / 3)      # -3.333...
# In many languages, integer division truncates toward zero (-3)
# Python floors toward negative infinity (-4)
```

## Interview Questions

### Beginner - 5

1. What is the difference between `/` and `//` in Python?
2. What does the `%` operator do? Give an example of checking if a number is even.
3. What is the result of `type(10 / 2)` and why?
4. Explain operator precedence. What is the result of `3 + 4 * 2`?
5. How do you write a comment in Python?

### Intermediate - 5

1. Explain floating-point precision issues. Why does `0.1 + 0.2 != 0.3`? How do you safely compare floats?
2. What is the difference between `math.floor()` and `int()` for negative numbers?
3. How does Python handle integer overflow compared to languages like C or Java?
4. What are complex numbers in Python? How do you access their real and imaginary parts?
5. What is the result of `2 ** 3 ** 2` and why?

### Advanced - 3

1. Explain Python's numeric tower (type hierarchy) and how `__rsub__`, `__radd__`, etc., enable mixed-type arithmetic.
2. How does Python's `round()` implement "banker's rounding" (round half to even)? Give examples where it rounds up vs. down.
3. What is `decimal.Decimal` and when would you use it instead of `float`?

## Practice Problems

### Easy - 5 Questions

**Problem 1:** Calculate the area of a rectangle with width 12.5 and height 8.3.

**Problem 2:** Write a program that checks whether a given number is even or odd using the modulus operator.

**Problem 3:** Compute the result of `(10 + 5) * 2 - 8 / 4`. Print the result.

**Problem 4:** Use integer division to determine how many full boxes of 12 eggs you can fill with 50 eggs, and how many eggs are left over.

**Problem 5:** Compute `2^10` (2 to the power of 10) using the exponentiation operator.

### Medium - 5 Questions

**Problem 6:** Write a program that converts a temperature from Celsius to Fahrenheit using the formula `F = C * 9/5 + 32`.

**Problem 7:** Use `math.sqrt()` to calculate the hypotenuse of a right triangle with legs 3 and 4.

**Problem 8:** Write a program that calculates compound interest: `A = P * (1 + r/n)^(n*t)` where `P = 1000`, `r = 0.05`, `n = 4`, `t = 10`.

**Problem 9:** Calculate the sum of the first 100 natural numbers using the formula `n*(n+1)/2`. Check whether the result is a float or int.

**Problem 10:** Write a program that uses `math.isclose()` to compare `0.3` with `0.1 + 0.2` and prints whether they are equal.

### Hard - 3 Questions

**Problem 11:** Implement a function `is_prime(n)` that returns `True` if a positive integer `n` is prime, using the modulus operator and `math.isqrt()`.

**Problem 12:** Write a program that uses the `decimal` module to accurately compute `0.1 + 0.2` and compare it with the `float` result.

**Problem 13:** Create a program that computes the nth Fibonacci number using Binet's formula `phi^n / sqrt(5)` and compares it with an iterative solution.

## Solutions

### Problem 1
```python
width = 12.5
height = 8.3
area = width * height
print(area)  # 103.75
```

### Problem 2
```python
number = 7
if number % 2 == 0:
    print(f"{number} is even")
else:
    print(f"{number} is odd")
# Output: 7 is odd
```

### Problem 3
```python
result = (10 + 5) * 2 - 8 / 4
print(result)  # 28.0
```

### Problem 4
```python
eggs = 50
full_boxes = eggs // 12
leftover = eggs % 12
print(f"Full boxes: {full_boxes}, Leftover eggs: {leftover}")
# Output: Full boxes: 4, Leftover eggs: 2
```

### Problem 5
```python
print(2 ** 10)  # 1024
```

### Problem 6
```python
celsius = 25
fahrenheit = celsius * 9 / 5 + 32
print(f"{celsius}°C = {fahrenheit}°F")
# Output: 25°C = 77.0°F
```

### Problem 7
```python
import math
a, b = 3, 4
hypotenuse = math.sqrt(a ** 2 + b ** 2)
print(hypotenuse)  # 5.0
```

### Problem 8
```python
P = 1000
r = 0.05
n = 4
t = 10
A = P * (1 + r / n) ** (n * t)
print(f"Amount after {t} years: ${A:.2f}")
# Output: Amount after 10 years: $1643.62
```

### Problem 9
```python
n = 100
sum_n = n * (n + 1) // 2
print(sum_n)          # 5050
print(type(sum_n))    # <class 'int'>
```

### Problem 10
```python
import math
result = 0.1 + 0.2
if math.isclose(result, 0.3):
    print("0.1 + 0.2 equals 0.3 (within tolerance)")
else:
    print("Not equal")
# Output: 0.1 + 0.2 equals 0.3 (within tolerance)
```

### Problem 11
```python
import math

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

print(is_prime(17))   # True
print(is_prime(25))   # False
```

### Problem 12
```python
from decimal import Decimal

float_result = 0.1 + 0.2
decimal_result = Decimal("0.1") + Decimal("0.2")

print(f"Float: {float_result}")            # 0.30000000000000004
print(f"Decimal: {decimal_result}")        # 0.3
print(f"Float == 0.3: {float_result == 0.3}")    # False
print(f"Decimal == 0.3: {decimal_result == Decimal('0.3')}")  # True
```

### Problem 13
```python
import math

def fibonacci_binet(n):
    phi = (1 + math.sqrt(5)) / 2
    return round(phi ** n / math.sqrt(5))

def fibonacci_iterative(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

n = 10
print(f"Binet's formula: F({n}) = {fibonacci_binet(n)}")           # 55
print(f"Iterative: F({n}) = {fibonacci_iterative(n)}")             # 55
```

## Related Concepts

- **Variables and Data Types** — Understanding how numbers fit into Python's type system
- **Type Conversion** — Converting between int, float, and other numeric types
- **Strings** — Formatting numbers into strings for display using f-strings
- **Lists and Loops** — Processing collections of numbers in data analysis tasks
- **Boolean Logic** — Using comparison results with arithmetic in algorithms

## Next Concepts

- **Strings** — Text manipulation and methods
- **Booleans and Comparisons** — Logical operations on numeric values
- **Type Conversion** — Converting between types safely

## Summary

Python provides three numeric types: `int` for exact whole numbers, `float` for approximate real numbers, and `complex` for numbers with imaginary parts. The seven arithmetic operators (`+`, `-`, `*`, `/`, `//`, `%`, `**`) follow standard mathematical precedence (PEMDAS), with exponentiation being right-associative. The `math` module supplies constants (`pi`, `e`) and functions (`sqrt`, `ceil`, `floor`, `isclose`). Floating-point numbers have limited precision due to binary representation, so comparing them requires `math.isclose()` rather than `==`. Python integers have arbitrary precision, eliminating overflow concerns. Understanding these concepts is essential for scientific computing, data analysis, and machine learning.

## Key Takeaways

- Python has three numeric types: `int`, `float`, and `complex`
- Use `//` for floor division and `%` for modulus (remainder)
- Division `/` always returns a `float`, even with two integers
- Operator precedence follows PEMDAS; use parentheses for clarity
- The `math` module provides advanced math functions and constants
- Floating-point arithmetic is approximate — use `math.isclose()` for comparison
- Python integers never overflow (arbitrary precision)
- Complex numbers use `j` (not `i`) to denote imaginary parts
- Numeric types widen automatically: `int` -> `float` -> `complex`
