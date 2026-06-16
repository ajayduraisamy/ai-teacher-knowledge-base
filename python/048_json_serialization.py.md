# Concept: JSON Serialization

## Concept ID

PYT-048

## Difficulty

Intermediate

## Domain

Python

## Module

Intermediate Python

## Learning Objectives

- Understand JSON as a data interchange format
- Serialize Python objects with `json.dumps()` and `json.dump()`
- Deserialize JSON with `json.loads()` and `json.load()`
- Handle non-serializable types with custom encoders and decoders
- Pretty-print JSON for readability
- Manage datetime, Decimal, and custom object serialization
- Explore performance alternatives: `orjson`, `ujson`
- Apply JSON to AI/ML model configurations and dataset metadata

## Prerequisites

- Python dictionaries, lists, strings, numbers, and booleans
- File I/O with `open()`
- Basic class definitions and inheritance
- Understanding of serialization concepts

## Definition

**JSON** (JavaScript Object Notation) is a lightweight, text-based data interchange format that is easy for humans to read and write and easy for machines to parse and generate. The `json` module provides an API for encoding Python objects to JSON strings/files and decoding JSON strings/files back to Python objects.

JSON supports four primitive types (strings, numbers, booleans, null) and two composite types (objects/dicts, arrays/lists).

## Intuition

Think of JSON as a universal language for data exchange. When a Python program needs to talk to a JavaScript web app, or when you need to save structured data to a file, JSON is the common ground. Serialization is like packing your Python objects into a suitcase (JSON string) that anyone can unpack, regardless of what programming language they speak.

## Why This Concept Matters

JSON is the dominant data interchange format on the web. Every REST API, every configuration file, every NoSQL database (MongoDB, CouchDB) uses JSON. Python's `json` module makes it trivial to convert between Python objects and JSON, enabling seamless integration with web services, data storage, and configuration management.

## Real World Examples

- REST API request and response bodies
- Configuration files for applications and tools
- Data export/import between systems
- Storing structured data in databases (PostgreSQL JSON fields, MongoDB)
- Web browser localStorage and sessionStorage
- Tool configuration (ESLint, Prettier, VS Code settings)

## AI/ML Relevance

JSON in AI/ML:
- Model configuration files (hyperparameters, architecture params)
- Dataset metadata and annotation formats (COCO JSON, LabelMe)
- Experiment tracking outputs (MLflow, Weights and Biases)
- Tokenizer configuration files for NLP models
- Training job specifications and pipeline configurations
- Serializing model evaluation results for reporting

## Code Examples

### Example 1: Basic serialization and deserialization

```python
import json

# Python dict to JSON string
data = {
    'name': 'Alice',
    'age': 30,
    'is_student': False,
    'scores': [95, 87, 92],
    'address': None,
}

json_string = json.dumps(data)
print(json_string)
print(type(json_string))

# JSON string back to Python dict
parsed = json.loads(json_string)
print(parsed)
print(parsed['name'])
print(type(parsed))

# Output:
# {"name": "Alice", "age": 30, "is_student": false, "scores": [95, 87, 92], "address": null}
# <class 'str'>
# {'name': 'Alice', 'age': 30, 'is_student': False, 'scores': [95, 87, 92], 'address': None}
# Alice
# <class 'dict'>
```

### Example 2: File I/O with json.dump and json.load

```python
import json
import tempfile
import os

data = {'name': 'Alice', 'age': 30, 'city': 'New York'}

# Write to file
tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
json.dump(data, tmp)
tmp_path = tmp.name
tmp.close()

# Read from file
with open(tmp_path, 'r') as f:
    loaded = json.load(f)

print(loaded)
print(loaded == data)

os.unlink(tmp_path)

# Output:
# {'name': 'Alice', 'age': 30, 'city': 'New York'}
# True
```

### Example 3: Pretty printing

```python
import json

data = {
    'name': 'Alice',
    'age': 30,
    'address': {
        'street': '123 Main St',
        'city': 'New York',
        'zip': '10001'
    },
    'hobbies': ['reading', 'hiking', 'coding']
}

# Compact JSON
print(json.dumps(data))

# Pretty-printed JSON
print(json.dumps(data, indent=2))

# Sorted keys
print(json.dumps(data, indent=2, sort_keys=True))

# Custom separators
print(json.dumps(data, separators=(',', ':')))

# Output:
# {"name": "Alice", "age": 30, "address": {"street": "123 Main St", "city": "New York", "zip": "10001"}, "hobbies": ["reading", "hiking", "coding"]}
# {
#   "name": "Alice",
#   "age": 30,
#   "address": {
#     "street": "123 Main St",
#     "city": "New York",
#     "zip": "10001"
#   },
#   "hobbies": [
#     "reading",
#     "hiking",
#     "coding"
#   ]
# }
# {"address": {"city": "New York", "street": "123 Main St", "zip": "10001"}, "age": 30, "hobbies": ["coding", "hiking", "reading"], "name": "Alice"}
# {"name":"Alice","age":30,"address":{"street":"123 Main St","city":"New York","zip":"10001"},"hobbies":["reading","hiking","coding"]}
```

### Example 4: Handling non-serializable types — datetime

```python
import json
from datetime import datetime, date

def datetime_encoder(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

def datetime_decoder(dct):
    for key, value in dct.items():
        if isinstance(value, str):
            try:
                dct[key] = datetime.fromisoformat(value)
            except (ValueError, TypeError):
                pass
    return dct

data = {
    'event': 'Conference',
    'date': datetime(2024, 12, 25, 10, 30, 0),
    'created': date(2024, 1, 15),
}

json_str = json.dumps(data, default=datetime_encoder, indent=2)
print(json_str)

decoded = json.loads(json_str, object_hook=datetime_decoder)
print(decoded)
print(type(decoded['date']))

# Output:
# {
#   "event": "Conference",
#   "date": "2024-12-25T10:30:00",
#   "created": "2024-01-15"
# }
# {'event': 'Conference', 'date': datetime.datetime(2024, 12, 25, 10, 30), 'created': datetime.date(2024, 1, 15)}
# <class 'datetime.datetime'>
```

### Example 5: Custom class encoder/decoder

```python
import json

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def to_json(self):
        return {'__type__': 'Person', 'name': self.name, 'age': self.age}

    @classmethod
    def from_json(cls, data):
        return cls(data['name'], data['age'])

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_json'):
            return obj.to_json()
        return super().default(obj)

def custom_decoder(dct):
    if '__type__' in dct:
        if dct['__type__'] == 'Person':
            return Person.from_json(dct)
    return dct

person = Person('Alice', 30)
json_str = json.dumps(person, cls=CustomEncoder, indent=2)
print(json_str)

decoded = json.loads(json_str, object_hook=custom_decoder)
print(type(decoded))
print(decoded.name, decoded.age)

# Output:
# {
#   "__type__": "Person",
#   "name": "Alice",
#   "age": 30
# }
# <class '__main__.Person'>
# Alice 30
```

### Example 6: Handling edge cases

```python
import json

# NaN and Infinity
import math
data = {'value': float('nan'), 'big': float('inf'), 'neg': float('-inf')}

try:
    print(json.dumps(data))
except ValueError as e:
    print(f'Error: {e}')

# Allow NaN and Infinity
print(json.dumps(data, allow_nan=True))

# Non-ASCII characters
data = {'message': 'Hello, 世界'}
print(json.dumps(data))
print(json.dumps(data, ensure_ascii=False))

# Output:
# Error: Out of range float values are not JSON compliant
# {"value": NaN, "big": Infinity, "neg": -Infinity}
# {"message": "Hello, \u4e16\u754c"}
# {"message": "Hello, 世界"}
```

### Example 7: AI/ML model configuration

```python
import json

model_config = {
    'model': {
        'type': 'Transformer',
        'params': {
            'vocab_size': 30000,
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 2048,
            'dropout': 0.1,
            'activation': 'gelu',
        }
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'warmup_steps': 1000,
        'max_epochs': 100,
        'early_stopping_patience': 5,
    },
    'data': {
        'train_path': 'data/train.csv',
        'val_path': 'data/val.csv',
        'test_path': 'data/test.csv',
        'max_seq_length': 512,
        'num_workers': 4,
    }
}

config_json = json.dumps(model_config, indent=2)
print(config_json[:200] + '...')

# Save and load
with open('config.json', 'w') as f:
    json.dump(model_config, f, indent=2)

with open('config.json', 'r') as f:
    loaded_config = json.load(f)

print(loaded_config['model']['type'])
print(loaded_config['training']['learning_rate'])

# Output:
# {
#   "model": {
#     "type": "Transformer",
#     ...
# Transformer
# 0.0001
```

## Common Mistakes

1. Trying to serialize unhashable or non-serializable types (sets, datetime, custom objects) without custom encoders
2. Forgetting that JSON keys must be strings — integer keys become strings when serialized
3. Using single quotes or trailing commas in JSON strings (invalid JSON syntax)
4. Assuming JSON preserves key order (it does in Python 3.7+, but not guaranteed in the spec)
5. Loading untrusted JSON without validation — potential for large or malicious payloads

## Interview Questions

### Beginner - 5

1. What does JSON stand for and what is it used for?
2. What is the difference between `json.dumps()` and `json.dump()`?
3. What Python types map to JSON types and vice versa?
4. How do you pretty-print JSON output?
5. How do you read JSON from a file?

### Intermediate - 5

1. How do you serialize a `datetime` object to JSON?
2. How do you create a custom JSON encoder for a user-defined class?
3. What is the `object_hook` parameter in `json.loads()` used for?
4. How do you handle sets or other non-serializable types?
5. What is the difference between `json` and `orjson`/`ujson`?

### Advanced - 3

1. How would you implement a JSON encoder that handles circular references?
2. Explain the security considerations when loading JSON from untrusted sources.
3. How would you implement a streaming JSON parser for very large files?

## Practice Problems

### Easy - 5

1. Convert a Python dictionary to a JSON string and back.
2. Write a list of dictionaries to a JSON file with pretty printing.
3. Read a JSON file and access nested values.
4. Serialize a tuple to JSON and handle the type conversion.
5. Create a JSON string with sorted keys.

### Medium - 5

1. Create a custom JSON encoder for `Decimal` types.
2. Implement a JSON decoder that converts ISO date strings to `datetime` objects.
3. Write a function that merges two JSON configuration files.
4. Serialize a complex nested data structure with mixed types.
5. Implement a JSON validation function that checks required keys and types.

### Hard - 3

1. Implement a JSON serializer that handles circular references with a depth limit.
2. Build a streaming JSON parser that processes a large JSON array without loading it entirely into memory.
3. Design a JSON-based configuration system with inheritance (base config + overrides).

## Solutions

### Easy 1

```python
import json
data = {'name': 'Python', 'version': 3.12, 'features': ['dynamic', 'typed']}
json_str = json.dumps(data)
print(json_str)
parsed = json.loads(json_str)
print(parsed)

# Output:
# {"name": "Python", "version": 3.12, "features": ["dynamic", "typed"]}
# {'name': 'Python', 'version': 3.12, 'features': ['dynamic', 'typed']}
```

### Medium 1

```python
import json
from decimal import Decimal

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

data = {'price': Decimal('19.99'), 'tax': Decimal('2.50')}
json_str = json.dumps(data, cls=DecimalEncoder)
print(json_str)

# Output:
# {"price": 19.99, "tax": 2.5}
```

### Hard 1

```python
import json

class SafeEncoder(json.JSONEncoder):
    def __init__(self, max_depth=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_depth = max_depth
        self.seen = set()

    def default(self, obj):
        if id(obj) in self.seen:
            return '<circular reference>'
        self.seen.add(id(obj))
        if hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items()}
        return super().default(obj)

class Node:
    def __init__(self, name):
        self.name = name
        self.child = None

root = Node('root')
child = Node('child')
root.child = child
child.child = root  # circular

try:
    json.dumps(root, cls=SafeEncoder, indent=2)
    print('Encoded without error')
except RecursionError:
    print('Recursion error')

# Output:
# Encoded without error
```

## Related Concepts

- YAML and TOML as alternatives to JSON
- XML and protocol buffers for data interchange
- Pickle (Python-specific serialization)
- `jsonlines` / NDJSON format
- REST API design
- Data validation with JSON Schema

## Next Concepts

- Logging (PYT-049)
- Unit testing (PYT-050)
- API development with FastAPI
- Data serialization in ML pipelines

## Summary

JSON serialization is essential for data exchange, configuration, and storage in modern Python applications. The `json` module provides a complete API for encoding Python objects to JSON and decoding JSON back to Python. With custom encoders and decoders, you can handle any Python type. For performance-critical applications, `orjson` and `ujson` offer significant speed improvements.

## Key Takeaways

- `json.dumps()`/`json.dump()` serialize Python to JSON; `json.loads()`/`json.load()` deserialize
- JSON supports: object/dict, array/list, string, number, boolean, null
- Use `indent` for pretty-printing, `sort_keys` for sorted output
- Custom encoders via `default` parameter or `JSONEncoder` subclass
- Custom decoders via `object_hook` parameter
- `orjson` and `ujson` offer faster alternatives to the standard `json` module
- AI/ML: JSON is ideal for model configs, dataset metadata, and experiment tracking output
