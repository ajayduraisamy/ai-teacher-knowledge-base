# Concept: Scope and Namespace

## Concept ID

PYT-019

## Difficulty

Intermediate

## Domain

Python

## Module

Functions and Modules

## Learning Objectives

- Understand Python's scope resolution through the LEGB rule
- Use `global` to modify global variables inside functions
- Use `nonlocal` to modify enclosing scope variables in nested functions
- Recognize the role of `__builtins__` in the built-in scope
- Understand variable scoping in comprehensions
- Explain closures as a mechanism for capturing enclosing scope

## Prerequisites

- PYT-016: Functions (nested functions, closures)
- Understanding of variable assignment and lookup
- Familiarity with modules and the `import` statement

## Definition

**Scope** refers to the region of a program where a variable is accessible. **Namespace** is a mapping from names to objects (implemented as dictionaries). Python uses the **LEGB rule** to resolve variable names: **L**ocal, **E**nclosing, **G**lobal, **B**uilt-in. Scopes are determined statically (at definition time), but variable lookup happens at runtime.

The four scopes are:

- **Local (L)**: Inside the current function.
- **Enclosing (E)**: Any enclosing function's local scope (for nested functions).
- **Global (G)**: The top-level scope of the current module.
- **Built-in (B)**: The scope containing Python's built-in names (`print`, `len`, `range`, etc.).

## Intuition

Imagine a university building. The built-in scope is the entire campus — everyone has access to common resources (libraries, cafeterias). The global scope is a specific building — accessible to everyone in that building. The enclosing scope is a floor within the building — accessible to everyone on that floor. The local scope is a single room — only the people inside that room can access its contents. When Python needs to find a name, it starts in the room (local), then checks the floor (enclosing), then the building (global), and finally the campus (built-in).

## Why This Concept Matters

Understanding scope prevents name collision bugs and enables powerful patterns like closures. In AI/ML, scope management is critical when building complex nested functions for training loops, loss functions, and model architectures. The `nonlocal` keyword is essential for implementing stateful closures like counters, caches, and accumulators. Without a clear grasp of scoping, developers inadvertently shadow variables, modify globals unintentionally, or fail to persist state across function calls.

## Real World Examples

1. **Configuration management**: Global config objects accessed by multiple functions.
2. **Caching**: A closure that caches expensive computation results in its enclosing scope.
3. **Training loops**: An outer function defines hyperparameters that inner functions (training steps) access via enclosing scope.
4. **Counter functions**: A `make_counter()` function uses `nonlocal` to maintain a count across calls.
5. **Logging**: A logger factory that creates loggers with a shared parent namespace.

## AI/ML Relevance

In AI/ML, scope and namespace concepts are fundamental when organizing code into modules and classes. Training scripts often define global hyperparameters and use nested functions for training steps, validation, and logging. The `nonlocal` keyword appears in decorators for timing training epochs. Understanding scope is essential for debugging variable shadowing issues in large PyTorch/TensorFlow projects where multiple modules define overlapping names. The `global` keyword is sometimes used for shared state like random seeds and device configurations.

## Code Examples

### Example 1: Basic LEGB lookup

```python
x = "global"

def outer():
    x = "enclosing"

    def inner():
        x = "local"
        print(x)

    inner()
    print(x)

outer()
print(x)
# Output:
# local
# enclosing
# global
```

### Example 2: Python's scope resolution step by step

```python
# Built-in scope
print("print is in built-in scope")

# Global scope
x_global = 100

def demo():
    x_local = 20
    print(x_local)   # Local scope
    print(x_global)  # Global scope (read access)
    print(len([1, 2]))  # Built-in scope

demo()
# Output:
# 20
# 100
# 2
```

### Example 3: The `global` keyword

```python
counter = 0

def increment():
    global counter
    counter += 1
    print(f"Counter inside: {counter}")

increment()
increment()
print(f"Counter outside: {counter}")
# Output:
# Counter inside: 1
# Counter inside: 2
# Counter outside: 2

# Without global — creates a local variable
def increment_wrong():
    counter = counter + 1  # UnboundLocalError!
```

### Example 4: The `nonlocal` keyword

```python
def make_counter():
    count = 0  # Enclosing scope

    def increment():
        nonlocal count
        count += 1
        return count

    return increment

counter_a = make_counter()
counter_b = make_counter()

print(counter_a())
# Output: 1
print(counter_a())
# Output: 2
print(counter_b())
# Output: 1  (Independent counter)
```

### Example 5: Closures as scope examples

```python
def create_multiplier(factor):
    """Return a function that multiplies any number by factor."""
    def multiplier(x):
        return x * factor  # factor is captured from enclosing scope
    return multiplier

double = create_multiplier(2)
triple = create_multiplier(3)

print(double(5))
# Output: 10
print(triple(5))
# Output: 15

# Inspect the closure
print(double.__closure__[0].cell_contents)
# Output: 2
print(triple.__closure__[0].cell_contents)
# Output: 3
```

### Example 6: Scope in comprehensions (Python 3)

```python
# In Python 3, list comprehensions have their own scope
x = "original"
result = [x for x in range(3)]  # x in comprehension is local
print(x)
# Output: original  (Python 3: global x is unchanged)

# In Python 2, this would have leaked x as 2

# Nested comprehension scope
matrix = [[i + j for j in range(3)] for i in range(3)]
print(matrix)
# Output: [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
```

### Example 7: `__builtins__` and shadowing

```python
print("Original print function")

# Shadow the built-in print (dangerous!)
print = lambda x: x.upper()

result = print("hello")
print(result)
# Output: HELLO

# But we broke the original print!
# To restore, use import builtins
import builtins
builtins.print("Back to normal")
# Output: Back to normal

# Inspect built-in scope
print(len(dir(__builtins__)))
# Output: e.g., 152 (varies by Python version)
```

### Example 8: Scope in nested functions with multiple levels

```python
def level1():
    a = 1

    def level2():
        a = 2  # Shadows level1's a

        def level3():
            nonlocal a  # Refers to level2's a
            a = 3
            print(f"Level3: a={a}")

        level3()
        print(f"Level2: a={a}")

    level2()
    print(f"Level1: a={a}")

level1()
# Output:
# Level3: a=3
# Level2: a=3
# Level1: a=1
```

## Common Mistakes

1. **UnboundLocalError**: Assigning to a variable inside a function makes it local, even if a global variable with the same name exists. Use `global` keyword to fix.
2. **Accidental shadowing**: Naming a local variable the same as a built-in (e.g., `list`, `str`, `len`) shadows the built-in, potentially breaking code.
3. **Confusing `global` and `nonlocal`**: `global` refers to module-level scope; `nonlocal` refers to the nearest enclosing function scope. They are not interchangeable.
4. **Forgetting that `nonlocal` requires the variable to exist in an enclosing scope**: Unlike `global`, `nonlocal` cannot create a new variable; it must refer to an existing one.
5. **Assuming comprehensions leak variables**: In Python 3, comprehensions have their own scope. However, generator expressions also use their own scope.

## Interview Questions

### Beginner

1. What is the LEGB rule in Python?
2. What is the difference between a scope and a namespace?
3. What happens if you try to access a variable that is not defined in any scope?
4. Can you modify a global variable inside a function without using the `global` keyword? What happens?
5. What is a built-in scope? Give three examples of built-in names.

### Intermediate

1. Explain the difference between `global` and `nonlocal` with code examples.
2. What is a closure? How does it relate to scope?
3. How does scope work in list comprehensions in Python 3 compared to Python 2?
4. Why does the following code raise `UnboundLocalError`?
   ```python
   x = 10
   def f():
       print(x)
       x = 20
   ```
5. How do you inspect the variables in the current scope?

### Advanced

1. Explain how Python implements scopes internally using `LOAD_FAST`, `LOAD_DEREF`, and `LOAD_GLOBAL` bytecode instructions.
2. Write a decorator that counts function calls using a `nonlocal` variable.
3. How does Python resolve names when multiple `nonlocal` statements could apply in deeply nested functions?

## Practice Problems

### Easy

1. Write a function that uses `global` to increment a counter variable.
2. Write a nested function that accesses a variable from its enclosing scope (without modifying it).
3. Identify the scope of `x` in: `x = 5; def f(): print(x)`.
4. Write a function that shadows a built-in name and explain the consequence.
5. Write a function that demonstrates that comprehension variables don't leak in Python 3.

### Medium

1. Implement a `make_adder(n)` function that returns a function adding `n` to its argument using closures.
2. Write a function that counts how many times it has been called using `nonlocal` in a nested function.
3. Write a decorator `@count_calls` that tracks the number of times a function is called.
4. Write a function that demonstrates the difference between `global` and `nonlocal` across three levels of nesting.
5. Create a memoization function using closures (manual, not `functools.lru_cache`).

### Hard

1. Implement a module-level configuration system where functions can read and update global settings with proper scoping.
2. Write a function that captures a variable in a closure but allows the captured value to be updated from outside.
3. Analyze and fix a buggy nested function that incorrectly shadows variables across three levels.

## Solutions

### Easy Solutions

```python
# 1
count = 0
def increment():
    global count
    count += 1

# 2
def outer():
    msg = "Hello"
    def inner():
        print(msg)
    inner()
outer()

# 4
# def f(): str = 5; print(str) — shadows built-in str()

# 5
x = "original"
[x for x in range(5)]
print(x)  # Still "original"
```

### Medium Solutions

```python
# 1
def make_adder(n):
    def adder(x):
        return x + n
    return adder

# 2
def make_counter():
    count = 0
    def counter():
        nonlocal count
        count += 1
        return count
    return counter

# 3
def count_calls(func):
    count = 0
    def wrapper(*args, **kwargs):
        nonlocal count
        count += 1
        print(f"Call {count} of {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

# 5
def memoize(f):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = f(*args)
        return cache[args]
    return wrapper
```

### Hard Solutions

```python
# 1
class Config:
    _settings = {}

    @classmethod
    def set(cls, key, value):
        cls._settings[key] = value

    @classmethod
    def get(cls, key, default=None):
        return cls._settings.get(key, default)

# 3 — debugging scope issues
def fixed():
    x = 1
    def level2():
        nonlocal x
        x = 2
        def level3():
            nonlocal x
            x = 3
            print(x)  # 3
        level3()
        print(x)  # 3 (not 1, because level2 modified it via nonlocal)
    level2()
    print(x)  # 1 (level1's x is untouched)
```

## Related Concepts

- PYT-016: Functions
- PYT-017: Arguments and Parameters
- PYT-020: Modules and Imports
- PYT-024: Decorators

## Next Concepts

- PYT-035: Advanced Function Patterns
- PYT-100: Python's Execution Model in Depth
- PYT-110: Bytecode and Disassembly

## Summary

Python uses the LEGB rule (Local, Enclosing, Global, Built-in) for variable name resolution. The `global` keyword is needed to modify module-level variables inside functions. The `nonlocal` keyword modifies variables in the nearest enclosing function scope. Closures capture variables from their enclosing scope, enabling stateful functions. Comprehensions in Python 3 have their own scope, preventing variable leakage. Understanding scope is essential for avoiding `UnboundLocalError`, shadowing bugs, and writing correct nested functions.

## Key Takeaways

- Variable lookup follows LEGB order: Local → Enclosing → Global → Built-in.
- Use `global` to write to a module-level variable inside a function.
- Use `nonlocal` to write to an enclosing function's variable from a nested function.
- Closures capture enclosing scope variables by reference, enabling factory functions.
- In Python 3, comprehensions and generator expressions have their own scope.
- Avoid shadowing built-in names to prevent subtle bugs.
