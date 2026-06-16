# Concept: Conditionals

## Concept ID

PYT-007

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

- Write `if`, `elif`, and `else` clauses with correct indentation
- Understand truthiness and which values are considered `False`
- Nest conditionals appropriately for complex branching logic
- Use the ternary expression (`x if cond else y`) for inline decisions
- Employ the `match`/`case` statement (Python 3.10+) for pattern matching
- Recognise common conditional patterns and pitfalls

## Definition

Conditionals are programming constructs that execute different blocks of code depending on whether a condition evaluates to `True` or `False`. In Python, this is primarily achieved with the `if`/`elif`/`else` keywords, the ternary conditional expression, and (in Python 3.10+) the `match`/`case` structural pattern matching statement.

## Intuition

Life is full of decisions: *if* it is raining, take an umbrella; *else* wear sunglasses. Conditionals allow a program to make similar decisions. They evaluate a Boolean expression (a yes/no question) and follow the path corresponding to the answer. Without conditionals, every program would run the same sequence of instructions every time, making it impossible to respond to different inputs or situations.

## Why This Concept Matters

Conditionals are the foundation of all non-trivial program logic. Validation (checking user input), branching (choosing between algorithms), error handling (checking return codes), and state machines all depend on conditionals. In AI/ML, conditionals control data preprocessing pipelines (e.g., if a column is missing, impute it), hyperparameter selection (if accuracy drops, reduce learning rate), and model evaluation (if loss increases, stop training early).

## Real World Examples

1. **Login system**: *If* the password matches the hash, grant access; *else* show an error.
2. **Discount calculator**: *If* the user is a member, apply a 10% discount; *elif* it is Black Friday, apply 20%; *else* no discount.
3. **Traffic light**: *If* the light is green, go; *elif* yellow, slow down; *else* stop.
4. **Data validation**: *If* the user's age is less than 0 or greater than 150, reject the input.
5. **ML early stopping**: *If* validation loss has not improved for 10 epochs, stop training.

## AI/ML Relevance

- **Data preprocessing**: Conditionals check for missing values, outliers, or invalid data before feeding it to a model.
- **Feature engineering**: New features may be created conditionally — e.g., *if* the time is between 6 PM and 6 AM, mark as "night".
- **Hyperparameter tuning**: Grid search evaluates many combinations and conditionally updates the best configuration.
- **Decision trees**: The quintessential ML model is built entirely on nested conditionals.
- **Model serving**: In production, conditionals route requests to different model versions based on A/B testing flags.

## Code Examples

### Example 1: Basic `if`/`elif`/`else`

Python uses indentation (typically 4 spaces) to define blocks. The colon `:` introduces a new block.

```python
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

print(f"Grade: {grade}")
# Output: Grade: B
```

### Example 2: Truthiness and Falsy values

In Python, some values are considered `False` in a Boolean context: `None`, `0`, `0.0`, empty strings `""`, empty lists `[]`, empty dicts `{}`, empty sets `set()`, and empty tuples `()`. Everything else is truthy.

```python
items = []
if items:               # Equivalent to: if len(items) > 0
    print("List has items")
else:
    print("List is empty")
# Output: List is empty

name = ""
if not name:
    print("Name is empty")
# Output: Name is empty
```

### Example 3: Nested conditionals

Conditionals can be placed inside other conditionals. Use nesting sparingly — deep nesting hurts readability. Consider refactoring or using `and`/`or`.

```python
age = 20
has_id = True

if age >= 18:
    if has_id:
        print("Allowed to enter")
    else:
        print("Need ID")
else:
    print("Too young")
# Output: Allowed to enter
```

### Example 4: Ternary conditional expression

The ternary expression `x if condition else y` evaluates to `x` when the condition is truthy and `y` otherwise. Use it for simple inline decisions.

```python
x = 15
result = "even" if x % 2 == 0 else "odd"
print(f"{x} is {result}")
# Output: 15 is odd

# Nested ternary (use sparingly)
y = -5
sign = "positive" if y > 0 else "negative" if y < 0 else "zero"
print(f"{y} is {sign}")
# Output: -5 is negative
```

### Example 5: `match`/`case` (Python 3.10+)

Structural pattern matching is similar to `switch`/`case` in other languages but far more powerful. It can match literals, sequences, mappings, and even custom objects.

```python
status_code = 404

match status_code:
    case 200:
        print("OK")
    case 201:
        print("Created")
    case 301 | 302:
        print("Redirect")
    case 400:
        print("Bad Request")
    case 401 | 403:
        print("Unauthorized")
    case 404:
        print("Not Found")
    case 500:
        print("Server Error")
    case _:
        print("Unknown code")
# Output: Not Found
```

### Example 6: Matching sequences and guards

`match`/`case` can destructure sequences and apply guards (additional conditions with `if`).

```python
point = (3, 0)

match point:
    case (0, 0):
        print("Origin")
    case (0, y):
        print(f"On Y axis at y={y}")
    case (x, 0):
        print(f"On X axis at x={x}")
    case (x, y) if x == y:
        print(f"On diagonal at ({x}, {y})")
    case (x, y):
        print(f"Point at ({x}, {y})")
# Output: On X axis at x=3
```

## Common Mistakes

1. **Forgetting the colon `:`** at the end of `if`, `elif`, `else`, `match`, and `case` lines.
2. **Using `=` instead of `==`** — `if x = 5:` assigns 5 instead of comparing, causing a `SyntaxError`.
3. **Inconsistent indentation** — mixing tabs and spaces or using the wrong number of spaces causes `IndentationError`.
4. **Misunderstanding truthiness** — checking `if x == True` instead of `if x`, or assuming non-empty values are falsy.
5. **Using `else if`** instead of Python's `elif`.
6. **Forgetting that `match`/`case` requires Python 3.10+** — using it in older versions raises `SyntaxError`.
7. **Deeply nested conditionals** — over-nesting makes code hard to follow; use logical operators or early returns.

## Interview Questions

### Beginner

1. **Q:** What is the syntax of an `if` statement in Python?  
   **A:** `if condition:` followed by an indented block, optionally `elif condition:` and `else:`.

2. **Q:** What values are considered falsy in Python?  
   **A:** `None`, `False`, `0`, `0.0`, `""`, `[]`, `{}`, `()`, `set()`.

3. **Q:** How do you write a ternary conditional in Python?  
   **A:** `value_if_true if condition else value_if_false`.

4. **Q:** What does `elif` stand for?  
   **A:** "else if".

5. **Q:** What happens if you forget the colon after an `if` line?  
   **A:** Python raises a `SyntaxError`.

### Intermediate

1. **Q:** Can you have an `if` without an `else`?  
   **A:** Yes. `else` is optional.

2. **Q:** How does short-circuit evaluation work with `and`/`or`?  
   **A:** For `and`, if the left operand is falsy, the right operand is never evaluated. For `or`, if the left operand is truthy, the right operand is never evaluated.

3. **Q:** What is the difference between `is` and `==` in conditions?  
   **A:** `==` checks value equality; `is` checks identity (same object). Use `is` with `None` but `==` for numbers/strings.

4. **Q:** How do you check multiple conditions concisely?  
   **A:** Use `and`, `or`, `not`, or containment with `in` (e.g., `if x in (2, 4, 6)`).

5. **Q:** What is a "guard" in `match`/`case`?  
   **A:** An `if` clause after a case pattern that adds an extra condition for the pattern to match.

### Advanced

1. **Q:** How would you simulate a `switch`/`case` statement in Python 3.8?  
   **A:** Using a dictionary mapping:  
   ```python
   def handle(code):
       return {200: "OK", 404: "Not Found"}.get(code, "Unknown")
   ```

2. **Q:** Explain the difference between `if a and b` and `if a & b`.  
   **A:** `and` is a logical operator (short-circuiting); `&` is a bitwise operator (evaluates both operands). Using `&` with non-integer types may produce unexpected results or errors.

3. **Q:** Write a pattern-match example that destructures a nested list and matches a specific shape.  
   **A:**  
   ```python
   data = [1, [2, 3]]
   match data:
       case [int(), [int(), int()]]:
           print("Matched nested ints")
   ```

## Practice Problems

### Easy

1. **Even or Odd** — Accept an integer and print whether it is even or odd.
2. **Positive, Negative, or Zero** — Accept a number and classify it.
3. **Voting Age** — Accept age and print "Can vote" or "Cannot vote".
4. **Max of Two** — Accept two numbers and print the larger one.
5. **Leap Year** — Accept a year and print whether it is a leap year.

### Medium

1. **Grade Calculator** — Accept a score (0–100) and return A/B/C/D/F.
2. **Triangle Type** — Accept three side lengths and classify as equilateral, isosceles, or scalene.
3. **Rock Paper Scissors** — Accept two player moves and determine the winner.
4. **Date Validator** — Accept day, month, year and validate the date (consider leap years).
5. **Simple Calculator** — Accept two numbers and an operator (`+`, `-`, `*`, `/`), perform the operation.

### Hard

1. **FizzBuzz with Match** — Implement FizzBuzz (multiples of 3 → Fizz, 5 → Buzz, both → FizzBuzz) using `match`/`case` in Python 3.10+.
2. **Tax Calculator** — Given income, compute tax based on progressive brackets (e.g., 0% up to 10,000; 10% up to 50,000; 20% above). Use chain of conditionals.
3. **Command Parser** — Write a function that parses a simple command string like `"move 5"` or `"rotate 90"` using `match`/`case` and returns an action dictionary.

## Solutions

### Easy

```python
# 1. Even or Odd
n = int(input())
print("even" if n % 2 == 0 else "odd")

# 2. Positive, Negative, or Zero
n = float(input())
if n > 0:
    print("positive")
elif n < 0:
    print("negative")
else:
    print("zero")

# 3. Voting Age
age = int(input())
print("Can vote" if age >= 18 else "Cannot vote")

# 4. Max of Two
a, b = map(int, input().split())
print(a if a > b else b)

# 5. Leap Year
year = int(input())
if year % 400 == 0 or (year % 4 == 0 and year % 100 != 0):
    print("Leap year")
else:
    print("Not a leap year")
```

### Medium

```python
# 1. Grade Calculator
score = int(input())
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"
print(grade)

# 2. Triangle Type
a, b, c = map(float, input().split())
if a == b == c:
    print("equilateral")
elif a == b or b == c or a == c:
    print("isosceles")
else:
    print("scalene")

# 3. Rock Paper Scissors
p1, p2 = input().split()
if p1 == p2:
    print("draw")
elif (p1 == "rock" and p2 == "scissors") or \
     (p1 == "scissors" and p2 == "paper") or \
     (p1 == "paper" and p2 == "rock"):
    print("player 1 wins")
else:
    print("player 2 wins")

# 4. Date Validator
d, m, y = map(int, input().split("/"))
days_in_month = [31, 29 if (y % 400 == 0 or (y % 4 == 0 and y % 100 != 0)) else 28,
                 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
if 1 <= m <= 12 and 1 <= d <= days_in_month[m - 1]:
    print("Valid")
else:
    print("Invalid")

# 5. Simple Calculator
a, op, b = input().split()
a, b = float(a), float(b)
if op == "+":
    print(a + b)
elif op == "-":
    print(a - b)
elif op == "*":
    print(a * b)
elif op == "/":
    print(a / b if b != 0 else "Error: division by zero")
else:
    print("Invalid operator")
```

### Hard

```python
# 1. FizzBuzz with Match
for n in range(1, 101):
    match (n % 3 == 0, n % 5 == 0):
        case (True, True):
            print("FizzBuzz")
        case (True, False):
            print("Fizz")
        case (False, True):
            print("Buzz")
        case _:
            print(n)

# 2. Tax Calculator
income = float(input("Income: "))
tax = 0.0
if income > 50000:
    tax += (income - 50000) * 0.20
    income = 50000
if income > 10000:
    tax += (income - 10000) * 0.10
print(f"Tax: {tax:.2f}")

# 3. Command Parser
def parse(cmd):
    parts = cmd.split()
    match parts:
        case ["move", distance]:
            return {"action": "move", "distance": int(distance)}
        case ["rotate", angle]:
            return {"action": "rotate", "angle": int(angle)}
        case ["shoot"]:
            return {"action": "shoot"}
        case _:
            return {"action": "unknown"}

print(parse("move 10"))
# Output: {'action': 'move', 'distance': 10}
```

## Related Concepts

- **Operators and Expressions** — Comparison operators (`==`, `<`, `>`, etc.) produce the Boolean values conditionals evaluate.
- **Boolean Logic** — `and`, `or`, `not` combine conditions.
- **Loops** — Loops often use conditionals (`break` when condition met).
- **Functions** — Functions can return different values based on conditionals.

## Next Concepts

- **Loops** — Repeat actions based on conditions.
- **Exception Handling** — `try`/`except` is another form of conditional flow control.
- **Match/Case (advanced)** — Deeper pattern matching with classes and guards.

## Summary

Conditionals in Python — `if`/`elif`/`else`, the ternary expression, and `match`/`case` — give programs the ability to make decisions. Python relies on indentation to define blocks. Understanding truthiness is critical because many values are implicitly falsy. The `match`/`case` statement (Python 3.10+) offers a powerful alternative to long if-elif chains, especially for destructuring sequences and mappings.

## Key Takeaways

- Use `if`, `elif`, and `else` with a colon and consistent indentation (4 spaces).
- Falsy values include `None`, `0`, `""`, `[]`, `{}`, `()`, and `False`.
- The ternary form `x if cond else y` is concise for simple branches.
- Nested conditionals work but deep nesting harms readability.
- `match`/`case` (Python 3.10+) provides structural pattern matching beyond simple value comparison.
- Use `in` with collections (`if x in (a, b, c)`) for multi-value checks.
