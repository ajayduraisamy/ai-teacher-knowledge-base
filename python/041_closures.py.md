# Concept: Closures

## Concept ID

PYT-041

## Difficulty

Advanced

## Domain

Python

## Module

Advanced Python

## Learning Objectives

- Understand what a closure is and how it captures outer scope variables
- Differentiate closures from regular nested functions
- Inspect closures using `__closure__` and `cell` objects
- Use closures for factory functions and function factories
- Compare closures with classes for state management
- Recognize common closure patterns and pitfalls
- Apply closures in AI/ML contexts

## Prerequisites

- Nested function definitions
- Python scoping rules (LEGB rule)
- First-class functions (passing functions as arguments)
- Basic understanding of variable lifetime

## Definition

A **closure** is a function object that remembers values in enclosing lexical scope even when the program flow is no longer in that scope. In Python, a closure occurs when a nested function references a variable from its enclosing function's scope and the enclosing function has finished execution. The nested function "closes over" the captured variables, preserving them for later use.

## Intuition

Think of a closure as a function with a backpack. The function carries its own code (what to do) and a backpack of variables (values it needs) that were available where it was defined. Even if you leave the neighborhood where the function was created, it still has those values in its backpack.

A classic analogy: a closure is like a cake recipe that remembers which kitchen it was learned in — it still works even when you move to a different kitchen.

## Why This Concept Matters

Closures enable powerful programming patterns: factory functions that create specialized functions, maintaining private state without classes, function decorators, and callback customization. Closures are fundamental to decorators (a core Python feature) and to functional programming patterns. Understanding closures deepens your understanding of Python's scoping rules and first-class function support.

## Real World Examples

- Function decorators that remember configuration parameters
- Factory functions that create specialized mathematical operations
- Event handlers that capture context at creation time
- Callback customization in GUI and web frameworks
- Memoization and caching with persistent state
- Configuration presets for data processing pipelines

## AI/ML Relevance

Closures in AI/ML development:
- Creating custom activation functions with learnable parameters
- Building loss function factories with configurable parameters
- Implementing gradient computation wrappers that capture context
- Creating data augmentation pipelines with configurable transforms
- Building metric functions that remember running statistics

## Code Examples

### Example 1: Basic closure

```python
def make_multiplier(factor):
    def multiplier(x):
        return x * factor
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))
print(triple(5))

# Output:
# 10
# 15
```

### Example 2: Inspecting a closure with __closure__

```python
def make_greeter(greeting):
    def greet(name):
        return f'{greeting}, {name}!'
    return greet

hello = make_greeter('Hello')
hi = make_greeter('Hi')

print(hello('Alice'))
print(hi('Bob'))

print(hello.__closure__)
print(hello.__closure__[0].cell_contents)

# Output:
# Hello, Alice!
# Hi, Bob!
# (<cell at 0x...: str object at 0x...>,)
# Hello
```

### Example 3: Closure for private state (counter)

```python
def make_counter():
    count = 0
    def counter():
        nonlocal count
        count += 1
        return count
    return counter

counter_a = make_counter()
counter_b = make_counter()

print(counter_a())
print(counter_a())
print(counter_b())
print(counter_a())
print(counter_b())

# Output:
# 1
# 2
# 1
# 3
# 2
```

### Example 4: Factory functions with closures

```python
def create_power_function(exp):
    def power(base):
        return base ** exp
    return power

square = create_power_function(2)
cube = create_power_function(3)

print(square(4))
print(cube(3))
print(square(10))

# Output:
# 16
# 27
# 100
```

### Example 5: AI/ML — custom activation function with parameter

```python
def make_activation(a, b):
    def activation(x):
        return 1 / (1 + (a * x) ** (-b))
    return activation

sigmoid_like = make_activation(1.0, 1.0)
steep_sigmoid = make_activation(2.0, 3.0)

print(f'{sigmoid_like(0):.4f}')
print(f'{sigmoid_like(2):.4f}')
print(f'{steep_sigmoid(2):.4f}')

# Output:
# 0.5000
# 0.6667
# 0.9412
```

### Example 6: Closure vs class comparison

```python
# Using a class
class CounterClass:
    def __init__(self):
        self.count = 0
    def increment(self):
        self.count += 1
        return self.count

# Using a closure
def make_counter_fn():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

c_obj = CounterClass()
c_fn = make_counter_fn()

print(c_obj.increment())
print(c_obj.increment())
print(c_fn())
print(c_fn())

# Both produce the same results, but the closure
# is simpler for single-method cases

# Output:
# 1
# 2
# 1
# 2
```

### Example 7: Closure for memoization

```python
def make_memoized(func):
    cache = {}
    def memoized(n):
        if n not in cache:
            print(f'Computing {n}...')
            cache[n] = func(n)
        return cache[n]
    return memoized

@make_memoized
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(10))

# Output:
# Computing 10...
# Computing 9...
# Computing 8...
# Computing 7...
# Computing 6...
# Computing 5...
# Computing 4...
# Computing 3...
# Computing 2...
# Computing 1...
# Computing 0...
# 55
```

### Example 8: Closure with late binding gotcha

```python
def create_functions():
    funcs = []
    for i in range(3):
        def func():
            return i
        funcs.append(func)
    return funcs

functions = create_functions()
for f in functions:
    print(f())

# Output:
# 2
# 2
# 2
```

## Common Mistakes

1. Late binding closures — variables captured by reference, not value; all closures see the final value
2. Forgetting `nonlocal` when modifying captured variables in nested functions
3. Expecting closures to capture the current value at creation time (they capture the variable itself)
4. Using closures when a simple lambda or class would be clearer
5. Assuming closures have access to the enclosing scope's `self` for class methods
6. Creating closures inside loops without using default arguments to capture current values

## Interview Questions

### Beginner - 5

1. What is a closure in Python?
2. How do you create a closure?
3. What is the `nonlocal` keyword and why is it needed in closures?
4. What is `__closure__` and what does it contain?
5. Can closures modify variables from the enclosing scope?

### Intermediate - 5

1. What is the late binding problem with closures in loops and how do you fix it?
2. Compare closures and classes for maintaining state — when would you use each?
3. How do closures relate to decorators?
4. What is a `cell` object in Python and how is it related to closures?
5. How do closures interact with Python's garbage collection?

### Advanced - 3

1. Explain the precise conditions required for a closure to be created (vs. a regular nested function).
2. How would you implement a closure that correctly captures loop variables by value?
3. Discuss the memory implications of closures — what happens to the captured variables' lifetime?

## Practice Problems

### Easy - 5

1. Write a closure that adds a fixed number to any input.
2. Create a greeting closure factory — pass a greeting word, get back a function that greets a name.
3. Write a closure that counts how many times it has been called.
4. Create a closure that multiplies two numbers but remembers a fixed multiplier.
5. Write a closure that returns the absolute difference from a fixed reference value.

### Medium - 5

1. Implement a `make_repeater(n)` function that returns a closure repeating a string `n` times.
2. Build a `make_logger(prefix)` closure that prints messages with a configurable prefix.
3. Create a `make_averager()` closure that maintains a running average of all values passed to it.
4. Implement a `rate_limiter(max_calls, period)` closure that limits how often a function can be called.
5. Write a closure-based `cached_property` that computes once and caches the result.

### Hard - 3

1. Implement a closure-based finite state machine where each state is a closure accepting events.
2. Build a dependency injection container using closures as factories with captured configuration.
3. Design and implement a closure-based middleware pipeline where each middleware wraps the next.

## Solutions

### Easy 1

```python
def make_adder(n):
    def adder(x):
        return x + n
    return adder

add5 = make_adder(5)
add10 = make_adder(10)

print(add5(3))
print(add10(3))

# Output:
# 8
# 13
```

### Medium 1

```python
def make_repeater(n):
    def repeater(s):
        return s * n
    return repeater

repeat3 = make_repeater(3)
repeat5 = make_repeater(5)

print(repeat3('Hello '))
print(repeat5('Hi '))

# Output:
# Hello Hello Hello
# Hi Hi Hi Hi Hi
```

### Hard 1 (simplified FSM)

```python
def create_fsm():
    state = 'idle'

    def idle(event):
        nonlocal state
        if event == 'start':
            state = 'running'
            return 'Started'
        return 'Ignored'

    def running(event):
        nonlocal state
        if event == 'stop':
            state = 'idle'
            return 'Stopped'
        if event == 'pause':
            state = 'paused'
            return 'Paused'
        return 'Running...'

    def paused(event):
        nonlocal state
        if event == 'resume':
            state = 'running'
            return 'Resumed'
        if event == 'stop':
            state = 'idle'
            return 'Stopped'
        return 'Paused...'

    def transition(event):
        nonlocal state
        handlers = {'idle': idle, 'running': running, 'paused': paused}
        return handlers[state](event)

    return transition

fsm = create_fsm()
print(fsm('start'))
print(fsm('pause'))
print(fsm('stop'))

# Output:
# Started
# Paused
# Stopped
```

## Related Concepts

- First-class functions
- Scoping rules (LEGB)
- The `nonlocal` and `global` keywords
- Decorators
- Lambda functions
- Functional programming patterns
- Function factories

## Next Concepts

- Partial functions (PYT-042)
- `functools` module (PYT-043)
- Decorator patterns
- Context managers and closures

## Summary

Closures are functions that capture and remember variables from their enclosing scope even after the outer function has finished executing. They enable factory functions, private state management, decorators, and many functional programming patterns. Understanding closures is essential for mastering decorators, understanding Python's scoping, and writing clean, expressive code.

## Key Takeaways

- A closure captures variables from the enclosing scope by reference
- Use `nonlocal` to modify captured variables within a closure
- Closures are lighter than classes for single-method stateful objects
- Late binding causes all closures in a loop to see the final loop variable
- Inspect closures with `__closure__` and `cell_contents`
- Closures are the foundation of Python's decorator system
- AI/ML uses include parameterized activation functions, configurable loss functions, and augmented data pipelines
