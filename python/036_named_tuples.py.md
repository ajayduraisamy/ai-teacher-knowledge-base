# Concept: Named Tuples

## Concept ID

PYT-036

## Difficulty

Intermediate

## Domain

Python

## Module

Intermediate Python

## Learning Objectives

- Understand what named tuples are and why they exist
- Create named tuples using the `namedtuple()` factory function
- Access fields by name and index
- Use `_fields`, `_asdict()`, `_replace()` helper methods
- Create typed named tuples using the typing module syntax
- Compare named tuples with regular tuples, dictionaries, and dataclasses
- Choose the right data structure for the problem

## Prerequisites

- Python tuples and tuple unpacking
- Dictionaries and key-value access
- Basic object-oriented concepts (classes, attributes)

## Definition

A **named tuple** is a subclass of `tuple` that allows accessing elements both by positional index (like a regular tuple) and by named attribute (like an object). Named tuples are part of the `collections` module and provide a lightweight, immutable data structure with named fields.

Named tuples combine the immutability and performance of tuples with the readability of class attributes. They are defined using the `namedtuple()` factory function or, in modern Python, with typed syntax inheriting from `NamedTuple`.

## Intuition

Think of a named tuple as a tuple where each position has a label. A regular 2D point as a tuple `(3, 4)` forces you to remember that index `0` is x and index `1` is y. A named tuple `Point(x=3, y=4)` is self-documenting — the field names carry semantic meaning.

Named tuples sit between plain tuples (positional, fast, compact) and full classes (named, flexible, heavier). They give you named access without the overhead of a full class definition.

## Why This Concept Matters

Named tuples are fundamental to writing clean, readable, and self-documenting Python code. They eliminate "magic numbers" when indexing tuples and make return values from functions clearly structured. They are particularly valuable in data processing pipelines where each record has a fixed schema, such as CSV rows, database results, or configuration entries.

## Real World Examples

- Representing a database row (id, name, email) with named access
- Returning multiple values from a function with meaningful field names
- Parsing CSV or log file records into structured objects
- Configuration entries where each field has a specific meaning
- Geometry applications (Point, Vector, Rectangle) with XY access

## AI/ML Relevance

Named tuples are frequently used in AI/ML code for:
- Representing model evaluation metrics (accuracy, precision, recall, f1)
- Storing hyperparameter configurations in a structured, immutable way
- Returning prediction results with associated metadata
- Defining dataset sample schemas (features, label, id)
- Organizing experiment results for logging and comparison

## Code Examples

### Example 1: Basic namedtuple creation and access

```python
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
p = Point(3, 4)

print(p.x)
print(p.y)
print(p[0])
print(p)

# Output:
# 3
# 4
# 3
# Point(x=3, y=4)
```

### Example 2: Helper methods — _fields, _asdict, _replace

```python
from collections import namedtuple

Person = namedtuple('Person', 'name age city')
alice = Person('Alice', 30, 'New York')

print(alice._fields)

d = alice._asdict()
print(d)
print(type(d))

bob = alice._replace(name='Bob', age=25)
print(bob)

# Output:
# ('name', 'age', 'city')
# {'name': 'Alice', 'age': 30, 'city': 'New York'}
# <class 'dict'>
# Person(name='Bob', age=25, city='New York')
```

### Example 3: Typed NamedTuple (Python 3.6+)

```python
from typing import NamedTuple

class Student(NamedTuple):
    student_id: int
    name: str
    gpa: float
    active: bool = True

s = Student(101, 'Alice', 3.8)
print(s)
print(s.name)
print(s.gpa)
print(s.active)

# Output:
# Student(student_id=101, name='Alice', gpa=3.8, active=True)
# Alice
# 3.8
# True
```

### Example 4: Named tuple with default values and docstring

```python
from collections import namedtuple

Rectangle = namedtuple('Rectangle', ['width', 'height'])
Rectangle.__doc__ = 'A 2D rectangle dimensions'
Rectangle.width.__doc__ = 'Width of the rectangle'
Rectangle.height.__doc__ = 'Height of the rectangle'

r = Rectangle(10, 20)
print(r)
print(f'Area: {r.width * r.height}')

# Output:
# Rectangle(width=10, height=20)
# Area: 200
```

### Example 5: Named tuples in a real scenario (CSV-like data)

```python
from collections import namedtuple

Record = namedtuple('Record', ['id', 'name', 'score', 'grade'])

data = [
    Record(1, 'Alice', 92, 'A'),
    Record(2, 'Bob', 78, 'C'),
    Record(3, 'Charlie', 85, 'B'),
]

for rec in data:
    print(f'{rec.name} (ID: {rec.id}) scored {rec.score} = {rec.grade}')

# Filter with named access
honors = [r for r in data if r.score >= 90]
print(f'Honors students: {[r.name for r in honors]}')

# Output:
# Alice (ID: 1) scored 92 = A
# Bob (ID: 2) scored 78 = C
# Charlie (ID: 3) scored 85 = B
# Honors students: ['Alice']
```

### Example 6: Named tuples are immutable

```python
from collections import namedtuple

Point = namedtuple('Point', 'x y')
p = Point(1, 2)

try:
    p.x = 10
except AttributeError as e:
    print(f'Error: {e}')

# Output:
# Error: can't set attribute
```

## Common Mistakes

1. Forgetting that named tuples are immutable — attempting to assign to a field raises `AttributeError`
2. Confusing the first argument (typename string) with the class variable name — they should match or use `rename=True`
3. Using mutable default values like lists as field defaults — they are shared across all instances
4. Overlooking memory overhead when creating millions of named tuples — consider `__slots__` or arrays for extreme scale
5. Nesting named tuples without careful schema design — deeply nested access can become unwieldy

## Interview Questions

### Beginner - 5

1. What is a named tuple in Python and how is it different from a regular tuple?
2. How do you create a named tuple using the `namedtuple()` factory?
3. How do you access fields in a named tuple — by index and by name?
4. What does the `_asdict()` method return?
5. How do you create a new named tuple with one field changed?

### Intermediate - 5

1. Compare named tuples and dataclasses. When would you choose one over the other?
2. How do you add default values to fields in a named tuple?
3. What is the typed `NamedTuple` syntax and how does it differ from the factory function approach?
4. What is the `rename` parameter in `namedtuple()` used for?
5. How do named tuples compare to dictionaries in terms of memory and access speed?

### Advanced - 3

1. How are named tuples implemented internally? Explain the metaclass and descriptor mechanics.
2. Can named tuples be subclassed? What are the implications and limitations?
3. How would you design a system that includes both mutable and immutable record types, choosing between named tuples, dataclasses, and attrs?

## Practice Problems

### Easy - 5

1. Create a `Book` named tuple with fields `title`, `author`, `year`. Create an instance and print each field.
2. Given a list of `City` named tuples (name, population), write a function to find the city with the highest population.
3. Convert a named tuple to a dictionary using `_asdict()`, modify the dictionary, and create a new named tuple.
4. Create a `Movie` named tuple with default `rating=None`. Create instances with and without the rating.
5. Write a function that takes a named tuple and returns a string representation with one field per line.

### Medium - 5

1. Parse a CSV string into a list of named tuples with appropriate field names and types.
2. Create a hierarchy of named tuples for a university system (Person, Student, Professor) using inheritance or composition.
3. Write a generic function that converts any named tuple to a JSON-serializable dictionary.
4. Implement a cache system where function arguments are named tuples and results are stored using the named tuple as a dictionary key.
5. Create a `Deck` of `Card` named tuples and implement shuffling, dealing, and sorting by rank and suit.

### Hard - 3

1. Implement a class that dynamically creates named tuple subclasses from a dictionary of field names and types, complete with validation methods.
2. Build a lightweight ORM where database rows are represented as named tuples with automatic type conversion from SQL types to Python types.
3. Design and implement a configuration system that uses nested named tuples for hierarchical settings with validation and default propagation.

## Solutions

### Easy 1

```python
from collections import namedtuple

Book = namedtuple('Book', ['title', 'author', 'year'])
book = Book('1984', 'George Orwell', 1949)
print(book.title)
print(book.author)
print(book.year)

# Output:
# 1984
# George Orwell
# 1949
```

### Easy 2

```python
from collections import namedtuple

City = namedtuple('City', 'name population')
cities = [
    City('Tokyo', 13929286),
    City('Delhi', 16787941),
    City('Shanghai', 24870895),
]

largest = max(cities, key=lambda c: c.population)
print(f'{largest.name}: {largest.population}')

# Output:
# Shanghai: 24870895
```

### Medium 1

```python
from collections import namedtuple

csv_data = """id,name,score
1,Alice,92
2,Bob,78
3,Charlie,85"""

Record = namedtuple('Record', 'id name score')
lines = csv_data.strip().split('\n')
records = []
for line in lines[1:]:
    parts = line.split(',')
    records.append(Record(int(parts[0]), parts[1], int(parts[2])))

for r in records:
    print(r)

# Output:
# Record(id=1, name='Alice', score=92)
# Record(id=2, name='Bob', score=78)
# Record(id=3, name='Charlie', score=85)
```

### Hard 1

```python
from collections import namedtuple
from typing import get_type_hints

def typed_named_tuple(name, fields):
    return namedtuple(name, fields.keys())

class DynamicRecord:
    @staticmethod
    def create(name, schema):
        return typed_named_tuple(name, schema)

Person = DynamicRecord.create('Person', {'name': str, 'age': int, 'email': str})
p = Person('Alice', 30, 'alice@example.com')
print(p)

# Output:
# Person(name='Alice', age=30, email='alice@example.com')
```

## Related Concepts

- Regular tuples and tuple unpacking
- Dictionary data type
- Dataclasses (`@dataclass`)
- The `collections` module
- `__slots__` for memory optimization
- Data classes vs named tuples vs attrs

## Next Concepts

- Type hints and typing module (PYT-037)
- Dataclasses (advanced usage)
- Serialization with JSON (PYT-048)
- Data validation with Pydantic

## Summary

Named tuples provide an elegant middle ground between positional tuples and full class definitions. They offer named field access, immutability, lightweight memory usage, and self-documenting code. With both the `namedtuple()` factory and the typed `NamedTuple` syntax, Python developers have flexible tools to create structured, readable data containers without the complexity of full classes.

## Key Takeaways

- Named tuples combine tuple performance with named attribute access
- Use `_fields`, `_asdict()`, and `_replace()` for introspection and modification
- Typed `NamedTuple` syntax provides type hints and a class-like declaration
- Named tuples are immutable — use `_replace()` to create modified copies
- Choose named tuples when you need immutable records with named fields and minimal overhead
- For mutable objects or complex behavior, prefer dataclasses or regular classes
- Named tuples hash by value, making them suitable as dictionary keys and set members
