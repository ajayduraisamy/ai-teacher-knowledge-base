# Concept: Lambda Functions

## Concept ID

PYT-018

## Difficulty

Intermediate

## Domain

Python

## Module

Functions and Modules

## Learning Objectives

- Understand lambda functions as anonymous, single-expression functions
- Use lambdas effectively with `sorted()`, `map()`, `filter()`, and `functools.reduce()`
- Recognize the limitations of lambda functions
- Distinguish when to use a lambda versus a named function
- Apply lambdas in data transformation and AI/ML preprocessing

## Prerequisites

- PYT-016: Functions (definition, return values, first-class functions)
- PYT-017: Arguments and Parameters
- Basic understanding of lists, sorting, and iterables
- Familiarity with `sorted()`, `map()`, `filter()` built-ins

## Definition

A **lambda function** is a small, anonymous function defined using the `lambda` keyword instead of `def`. It can take any number of arguments but returns only a single expression. The syntax is:

```python
lambda arguments: expression
```

Lambdas cannot contain statements, assignments, or multiple expressions. They are syntactically restricted to a single expression that is evaluated and returned automatically.

## Intuition

Think of a lambda as a disposable sticky note — you write a quick expression on it, use it immediately, and throw it away. You don't need to give it a formal name because you're only using it once, right where you need it. If you find yourself writing the same lambda logic repeatedly, it is time to promote it to a named function.

## Why This Concept Matters

Lambdas enable functional programming patterns in Python. They make code more concise by eliminating the need to define one-off functions. They are essential for callbacks, key functions for sorting, and transformations in data pipelines. In AI/ML, lambdas are commonly used for quick preprocessing transformations, sorting model results, and defining lightweight metrics without cluttering the namespace.

## Real World Examples

1. **Sorting**: Sorting a list of dictionaries by a specific key with `sorted(data, key=lambda x: x['name'])`.
2. **Data transformation**: Applying a transformation to every element in a pandas Series with `series.map(lambda x: x.strip())`.
3. **UI callbacks**: Tkinter or PyQt button click handlers like `button.config(command=lambda: do_something())`.
4. **API responses**: Sorting API results by a computed field.
5. **ML pipelines**: Defining a custom preprocessing step as `lambda x: (x - mean) / std`.

## AI/ML Relevance

Lambdas are used in AI/ML for lightweight, inline operations. They appear in data preprocessing pipelines where a quick transformation is needed without defining a separate function. Model evaluation often involves sorting results by confidence scores, which uses `sorted(results, key=lambda x: x['confidence'], reverse=True)`. Pandas workflows frequently use lambdas with `.apply()` for element-wise operations. However, lambdas are not suitable for complex ML logic; they are best for simple, on-the-fly transformations.

## Code Examples

### Example 1: Basic lambda syntax

```python
# Named function equivalent
def add(a, b):
    return a + b

# Lambda equivalent
add_lambda = lambda a, b: a + b

print(add(3, 4))
# Output: 7
print(add_lambda(3, 4))
# Output: 7
print((lambda x: x ** 2)(5))
# Output: 25
```

### Example 2: Using lambda with `sorted()` and custom key

```python
students = [
    {"name": "Alice", "grade": 85},
    {"name": "Bob", "grade": 72},
    {"name": "Charlie", "grade": 95},
    {"name": "Diana", "grade": 68},
]

# Sort by grade ascending
sorted_by_grade = sorted(students, key=lambda s: s["grade"])
print(sorted_by_grade)
# Output: [{'name': 'Diana', 'grade': 68}, {'name': 'Bob', 'grade': 72},
#          {'name': 'Alice', 'grade': 85}, {'name': 'Charlie', 'grade': 95}]

# Sort by name descending
sorted_by_name_desc = sorted(students, key=lambda s: s["name"], reverse=True)
print(sorted_by_name_desc)
# Output: [{'name': 'Diana', 'grade': 68}, {'name': 'Charlie', 'grade': 95},
#          {'name': 'Bob', 'grade': 72}, {'name': 'Alice', 'grade': 85}]
```

### Example 3: Lambda with `map()`

```python
numbers = [1, 2, 3, 4, 5]

squared = list(map(lambda x: x ** 2, numbers))
print(squared)
# Output: [1, 4, 9, 16, 25]

celsius = [0, 10, 20, 30, 40]
fahrenheit = list(map(lambda c: (c * 9/5) + 32, celsius))
print(fahrenheit)
# Output: [32.0, 50.0, 68.0, 86.0, 104.0]
```

### Example 4: Lambda with `filter()`

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)
# Output: [2, 4, 6, 8, 10]

words = ["hello", "", "world", " ", "python", ""]
non_empty = list(filter(lambda w: w.strip() != "", words))
print(non_empty)
# Output: ['hello', 'world', ' ', 'python']
```

### Example 5: Lambda with `functools.reduce()`

```python
from functools import reduce

numbers = [1, 2, 3, 4, 5]
product = reduce(lambda a, b: a * b, numbers)
print(product)
# Output: 120

# Find maximum using reduce
max_val = reduce(lambda a, b: a if a > b else b, numbers)
print(max_val)
# Output: 5

# Concatenate strings
words = ["Hello", " ", "World", "!"]
sentence = reduce(lambda a, b: a + b, words)
print(sentence)
# Output: Hello World!
```

### Example 6: Lambda with multiple arguments and conditional expressions

```python
# Conditional expression in lambda
max_lambda = lambda a, b: a if a > b else b
print(max_lambda(10, 20))
# Output: 20

# Sorting by multiple criteria (tuples as key)
data = [(1, "z"), (2, "a"), (1, "a"), (2, "z")]
sorted_data = sorted(data, key=lambda x: (x[1], x[0]))
print(sorted_data)
# Output: [(1, 'a'), (2, 'a'), (1, 'z'), (2, 'z')]
```

### Example 7: AI/ML context — sorting model predictions

```python
predictions = [
    {"class": "cat", "confidence": 0.85},
    {"class": "dog", "confidence": 0.92},
    {"class": "bird", "confidence": 0.78},
    {"class": "fish", "confidence": 0.95},
]

# Sort by confidence descending to get top predictions
top_predictions = sorted(predictions, key=lambda p: p["confidence"], reverse=True)
print(top_predictions)
# Output: [{'class': 'fish', 'confidence': 0.95}, {'class': 'dog', 'confidence': 0.92},
#          {'class': 'cat', 'confidence': 0.85}, {'class': 'bird', 'confidence': 0.78}]

# Get top 2 classes
top_2 = [p["class"] for p in top_predictions[:2]]
print(top_2)
# Output: ['fish', 'dog']
```

### Example 8: Lambda with pandas-like transformation (pure Python)

```python
# Simulating a preprocessing pipeline with lambdas
data = ["  hello  ", "WORLD", "  Python  ", "ML  "]

pipeline = [
    lambda s: s.strip(),
    lambda s: s.lower(),
    lambda s: s.capitalize(),
]

result = data
for transform in pipeline:
    result = list(map(transform, result))

print(result)
# Output: ['Hello', 'World', 'Python', 'Ml']
```

## Common Mistakes

1. **Using statements inside lambdas**: Lambdas only allow a single expression. You cannot use `print`, `return`, `pass`, or assignment `=` inside a lambda.
2. **Overusing lambdas**: Complex lambdas harm readability. If the logic is more than a simple expression, use a `def` function.
3. **Lambda pitfalls with closures in loops**: Lambdas in a loop capture variables by reference, not value. They all see the final iteration value. Use a default argument to capture the current value.
4. **Expecting lambdas to be faster than named functions**: Lambdas are not inherently faster; they are identical to `def` functions in performance.
5. **Trying to type hint lambdas**: Lambda parameters and return values cannot be annotated with type hints. Use `def` if you need type hints.

## Interview Questions

### Beginner

1. What is a lambda function in Python? What keyword is used to define it?
2. What is the main syntactical restriction of a lambda function?
3. How do you call a lambda function immediately after defining it?
4. Write a lambda that doubles a number.
5. Can a lambda function take multiple arguments? Give an example.

### Intermediate

1. Explain the closure problem with lambdas in loops and how to fix it.
2. Write a lambda that sorts a list of tuples by the second element.
3. How would you use a lambda with `filter()` to get all positive numbers from a list?
4. Compare `map()` with lambda versus a list comprehension. Which is more Pythonic?
5. Write a lambda that returns `True` if a number is prime (up to 100).

### Advanced

1. Implement a `compose` function that chains multiple lambdas together.
2. How could you use lambdas to implement a simple neural network layer forward pass?
3. Explain the bytecode differences between a lambda and a named function. Are there any performance differences?

## Practice Problems

### Easy

1. Write a lambda that returns the square of a number.
2. Write a lambda that returns `True` if a string starts with a vowel.
3. Use `map` with a lambda to convert a list of strings to uppercase.
4. Use `filter` with a lambda to keep only numbers greater than 10.
5. Use `sorted` with a lambda to sort a list of strings by their length.

### Medium

1. Write a lambda that takes a tuple `(name, age)` and returns `True` if age >= 18.
2. Use `reduce` with a lambda to compute the sum of squares of a list of numbers.
3. Given a list of dicts `[{"x": 1, "y": 2}, {"x": 3, "y": 4}]`, use lambdas to sort by the product `x * y`.
4. Write a lambda-based pipeline that strips whitespace, converts to lowercase, and removes non-alphanumeric characters.
5. Use `map` with a lambda to convert a list of temperature tuples `[(c, f), ...]` to just Fahrenheit.

### Hard

1. Implement a lambda-based expression evaluator that can handle `+`, `-`, `*`, `/` for two numbers using a dictionary of lambdas.
2. Use lambdas to implement a simple decision tree node that compares a feature value against a threshold and returns left or right.
3. Write a function `apply_activation(x, activation_name)` that uses a dictionary of lambdas to apply activation functions (relu, sigmoid, tanh) without if-elif chains.

## Solutions

### Easy Solutions

```python
# 1
square = lambda x: x ** 2

# 2
starts_with_vowel = lambda s: s[0].lower() in "aeiou"

# 3
result = list(map(lambda s: s.upper(), ["a", "b", "c"]))

# 4
result = list(filter(lambda x: x > 10, [5, 15, 8, 20]))

# 5
sorted(["apple", "kiwi", "banana"], key=lambda s: len(s))
```

### Medium Solutions

```python
from functools import reduce

# 1
is_adult = lambda t: t[1] >= 18

# 2
sum_sq = reduce(lambda a, b: a + b**2, [1, 2, 3], 0)

# 3
data = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
sorted(data, key=lambda d: d["x"] * d["y"])

# 4
import re
pipeline = [
    lambda s: s.strip(),
    lambda s: s.lower(),
    lambda s: re.sub(r'[^a-zA-Z0-9]', '', s)
]

# 5
temps = [(0, 32), (10, 50), (20, 68)]
list(map(lambda t: t[1], temps))
```

### Hard Solutions

```python
# 1
ops = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b if b != 0 else float("inf"),
}
def eval_expr(a, op, b):
    return ops[op](a, b)

# 2
decision_node = lambda features, threshold, left_val, right_val: (
    left_val if features[0] < threshold else right_val
)

# 3
activations = {
    "relu": lambda x: max(0, x),
    "sigmoid": lambda x: 1 / (1 + 2.71828 ** (-x)),
    "tanh": lambda x: (2.71828 ** x - 2.71828 ** (-x)) / (2.71828 ** x + 2.71828 ** (-x)),
}
def apply_activation(x, name):
    return activations[name](x)
```

## Related Concepts

- PYT-016: Functions
- PYT-017: Arguments and Parameters
- PYT-024: Decorators
- PYT-035: Functional Programming Tools

## Next Concepts

- PYT-030: List Comprehensions (alternative to map/filter)
- PYT-050: Functional Programming in Python
- PYT-060: Higher-Order Functions

## Summary

Lambda functions are anonymous, single-expression functions defined with the `lambda` keyword. They are ideal for short, throwaway operations in `sorted()`, `map()`, `filter()`, and `reduce()`. Lambdas cannot contain statements, assignments, or type hints. The closure-in-loops pitfall can be solved with default argument capture. In AI/ML, lambdas are useful for quick preprocessing, sorting predictions, and defining simple transformations without polluting the namespace.

## Key Takeaways

- Syntax: `lambda args: expression` — always a single expression, never statements.
- Use lambdas for simple, one-off functions; use `def` for anything complex.
- `sorted(data, key=lambda x: x['key'])` is the most common lambda usage.
- Lambdas in loops capture by reference; use default arguments to fix.
- Lambdas cannot use type hints, assignment (`=`), or contain `return`.
- Prefer list comprehensions over `map()`/`filter()` with lambdas for readability.
