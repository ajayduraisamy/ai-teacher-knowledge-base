# Concept: Modules and Imports

## Concept ID

PYT-020

## Difficulty

Intermediate

## Domain

Python

## Module

Functions and Modules

## Learning Objectives

- Understand what a module is and how Python organizes code into modules
- Differentiate between `import`, `from ... import`, and `import ... as`
- Use the `if __name__ == "__main__"` guard correctly
- Understand `sys.path` and Python's module search order
- Use `dir()` and `__all__` to control module exports
- Structure ML code into reusable modules

## Prerequisites

- PYT-016: Functions (definition and organization)
- Basic understanding of files and directories
- PYT-019: Scope and Namespace (module-level scope)

## Definition

A **module** in Python is a file containing Python definitions and statements. The file name is the module name with a `.py` suffix. Modules allow you to logically organize Python code into reusable units. The `import` statement loads a module, executes its code, and makes its names available in the importing namespace.

```python
# Import the entire module
import math
print(math.sqrt(16))  # Access names via the module namespace

# Import specific names
from math import sqrt, pi
print(sqrt(16))
print(pi)

# Import with alias
import numpy as np
```

## Intuition

Think of modules as chapters in a textbook. Each chapter covers a specific topic (math, statistics, file I/O) and contains related functions, classes, and variables. You don't need to read the whole book to use one chapter — you just open the chapter you need. Similarly, Python modules let you load only the functionality you need without executing unrelated code.

## Why This Concept Matters

Modular code is the foundation of maintainable software. Without modules, all code would be in a single monolithic file. Python's standard library is organized into hundreds of modules covering everything from regular expressions to concurrent programming. In AI/ML, you import PyTorch, NumPy, scikit-learn, and your own custom modules to build pipelines. Understanding the import system is essential for managing dependencies, avoiding circular imports, and organizing large codebases.

## Real World Examples

1. **Data science**: `import pandas as pd; import numpy as np; import matplotlib.pyplot as plt`
2. **Web development**: `from flask import Flask, request, jsonify`
3. **Deep learning**: `import torch; import torch.nn as nn; from torch.utils.data import DataLoader`
4. **Testing**: `import pytest; from mymodule import my_function`
5. **API development**: `from fastapi import FastAPI; from pydantic import BaseModel`

## AI/ML Relevance

Modern AI/ML relies heavily on modular imports. A typical ML project imports:
- `numpy` for numerical operations
- `torch` or `tensorflow` for deep learning
- `sklearn` for preprocessing and metrics
- `matplotlib` or `seaborn` for visualization
- Custom modules for data loading, model definitions, training loops, and evaluation

Structuring ML code into modules — `data/`, `models/`, `train/`, `utils/` — promotes reusability and collaboration. The `if __name__ == "__main__"` guard allows modules to be both importable and runnable, which is essential for ML experiment scripts.

## Code Examples

### Example 1: Basic import styles

```python
# Style 1: Import entire module
import math
print(math.factorial(5))
# Output: 120

# Style 2: Import specific names
from math import factorial, gcd
print(factorial(5))
# Output: 120
print(gcd(12, 8))
# Output: 4

# Style 3: Import with alias
import datetime as dt
print(dt.date.today())
# Output: 2026-06-16

# Style 4: Import all names (not recommended)
# from math import *  — pollutes namespace
```

### Example 2: The `if __name__ == "__main__"` guard

```python
# File: utils.py
def preprocess(data):
    return [x.strip().lower() for x in data]

def analyze(data):
    return {"count": len(data), "unique": len(set(data))}

# This code only runs when utils.py is executed directly,
# NOT when imported as a module.
if __name__ == "__main__":
    sample = [" Hello ", "WORLD", "  Python  "]
    cleaned = preprocess(sample)
    print(analyze(cleaned))
    # Output when run directly: {'count': 3, 'unique': 3}
```

### Example 3: Module search order and `sys.path`

```python
import sys

# Print the module search path
for path in sys.path:
    print(path)
# Output: (list of directories)
# 1. Directory of the script being run (or current directory)
# 2. PYTHONPATH environment variable directories
# 3. Standard library directories
# 4. Site-packages directories

# Adding a custom path
sys.path.append("/my/custom/modules")

# Now you can import from that directory
# import my_custom_module
```

### Example 4: `dir()` and `__all__`

```python
import math

# List all names in the math module
print([name for name in dir(math) if not name.startswith("_")])
# Output: ['acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh', ...]

# Defining __all__ in a module (e.g., mymodule.py)
"""
__all__ = ["preprocess", "analyze", "load_data"]

def preprocess(x): ...
def analyze(x): ...
def load_data(path): ...
def _helper(x): ...  # Private, won't be exported with *
"""

# When importing with `from mymodule import *`
# Only names in __all__ are imported.
```

### Example 5: Creating and importing a custom module

```python
# File: data_utils.py
"""Data utility functions for ML preprocessing."""

import csv
from typing import List, Dict

__all__ = ["load_csv", "normalize"]

def load_csv(path: str) -> List[Dict[str, str]]:
    """Load CSV file as list of dictionaries."""
    with open(path, "r") as f:
        return list(csv.DictReader(f))

def normalize(values: List[float]) -> List[float]:
    """Min-max normalize a list of values."""
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.0] * len(values)
    return [(v - min_val) / (max_val - min_val) for v in values]

def _validate(data: List[Dict]) -> bool:
    """Internal helper, not exported with *."""
    return len(data) > 0

# Usage:
# >>> from data_utils import load_csv, normalize
# >>> data = load_csv("train.csv")
# >>> normalized = normalize([float(r["age"]) for r in data])
```

### Example 6: Reloading modules

```python
import importlib
import mymodule  # Assume this exists

# After modifying mymodule.py during development
importlib.reload(mymodule)

# Now mymodule reflects the latest changes
# Note: existing references to old objects are NOT updated
# from mymodule import some_func  # This still points to the old function!
```

### Example 7: Structuring ML code into modules

```python
"""
Typical ML project structure:

project/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── loader.py       # Data loading
│   └── preprocessing.py # Cleaning, normalization
├── models/
│   ├── __init__.py
│   ├── architecture.py # Model definition
│   └── layers.py       # Custom layers
├── train/
│   ├── __init__.py
│   ├── trainer.py      # Training loop
│   └── metrics.py      # Evaluation metrics
├── utils/
│   ├── __init__.py
│   ├── config.py       # Hyperparameter config
│   └── logging.py      # Logging setup
└── main.py             # Entry point
"""

# main.py imports:
"""
from data.preprocessing import load_and_clean
from models.architecture import NeuralNetwork
from train.trainer import Trainer
from utils.config import Config

config = Config.from_yaml("config.yaml")
model = NeuralNetwork(config)
trainer = Trainer(model, config)
trainer.train()
"""
```

### Example 8: Conditional imports (handling optional dependencies)

```python
# Module that works with or without optional dependency
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Define fallback implementations
    class np_fallback:
        @staticmethod
        def array(data):
            return list(data)
    np = np_fallback()

def as_array(data):
    """Convert data to array-like object."""
    if HAS_NUMPY:
        return np.array(data)
    return list(data)

print(as_array([1, 2, 3]))
# Output: [1, 2, 3]
```

## Common Mistakes

1. **Circular imports**: Module A imports module B, which imports module A. This causes `ImportError`. Restructure code to avoid circular dependencies.
2. **Using `from module import *`**: Pollutes the namespace, makes code unclear, and can silently override existing names. Use explicit imports instead.
3. **Shadowing standard library modules**: Naming your file `math.py` or `json.py` will shadow the standard library module when you try `import math`.
4. **Forgetting the `__name__` guard**: Code at module level runs on import. Without `if __name__ == "__main__":`, importing a module can have side effects like running tests or printing output.
5. **Importing inside functions unnecessarily**: While possible, it adds overhead and reduces clarity. Import at the top of the module unless you have a specific reason (e.g., optional dependencies).

## Interview Questions

### Beginner

1. What is a module in Python?
2. What is the difference between `import math` and `from math import sqrt`?
3. What does `if __name__ == "__main__":` do? Why is it important?
4. How do you create a module? (What file extension, naming conventions)
5. What is `sys.path` and how does it affect imports?

### Intermediate

1. Explain Python's module search order.
2. What is `__all__` and how does it control imports?
3. How do you reload a module without restarting Python?
4. What causes a circular import and how do you resolve it?
5. What is the difference between absolute and relative imports?

### Advanced

1. Explain how Python caches modules in `sys.modules` and what happens when you import a module twice.
2. How do namespace packages work (PEP 420)?
3. Write a custom import hook that logs every import statement.

## Practice Problems

### Easy

1. Create a module `greetings.py` with a function `say_hello(name)` and import it.
2. Write a script that imports `math` and uses `sqrt`, `sin`, and `pi`.
3. Use `from ... import ... as` to import `datetime.datetime` as `dt`.
4. Write a module with the `__name__` guard that prints something when run directly.
5. Use `dir()` to list all names in the `random` module.

### Medium

1. Create a project with two modules: `calculator.py` (add, subtract) and `main.py` that imports both.
2. Write a module that conditionally imports `numpy` and provides a fallback if not available.
3. Implement `__all__` in a module and verify that `from module import *` only imports the listed names.
4. Create a reusable `config.py` module that reads from a JSON file and makes settings available.
5. Write a `utils.py` module with `__all__`, private functions (prefixed with `_`), and verify behavior.

### Hard

1. Design a plugin system where modules in a `plugins/` directory are discovered and loaded dynamically.
2. Implement a lazy import mechanism that only imports a module when its function is first called.
3. Convert a monolithic ML training script into a modular project with `data/`, `models/`, `train/`, and `utils/` packages.

## Solutions

### Easy Solutions

```python
# greetings.py
def say_hello(name):
    return f"Hello, {name}!"

# main.py
import greetings
print(greetings.say_hello("Alice"))

# 3
from datetime import datetime as dt
print(dt.now())

# 4
if __name__ == "__main__":
    print("Running directly!")

# 5
import random
print([name for name in dir(random) if not name.startswith("_")])
```

### Medium Solutions

```python
# calculator.py
def add(a, b): return a + b
def subtract(a, b): return a - b

# main.py
from calculator import add, subtract

# 2 — Conditional import
try:
    import numpy as np
except ImportError:
    import math
    np = type("numpy", (), {"array": lambda x: [float(v) for v in x]})()

# 3
# mymodule.py
__all__ = ["public_func"]
def public_func(): return "public"
def _private_func(): return "private"

# 4
import json
class Config:
    _config = {}
    @classmethod
    def load(cls, path):
        with open(path) as f:
            cls._config = json.load(f)
    @classmethod
    def get(cls, key, default=None):
        return cls._config.get(key, default)
```

### Hard Solutions

```python
# 1 — Plugin system
import importlib
import pkgutil
import inspect

def discover_plugins(package):
    plugins = {}
    for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{modname}")
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and hasattr(obj, "_plugin"):
                plugins[name] = obj
    return plugins

# 2 — Lazy import
class LazyImport:
    def __init__(self, module_name):
        self.module_name = module_name
        self._module = None
    def __getattr__(self, name):
        if self._module is None:
            self._module = __import__(self.module_name)
        return getattr(self._module, name)

# Usage: np = LazyImport("numpy")
```

## Related Concepts

- PYT-019: Scope and Namespace
- PYT-021: Standard Library
- PYT-022: Packages and __init__.py
- PYT-023: Virtual Environments

## Next Concepts

- PYT-030: File I/O
- PYT-040: Package Distribution (setup.py, pyproject.toml)
- PYT-050: Dependency Management

## Summary

Modules are `.py` files that organize Python code into reusable units. The `import` system supports different styles: `import module`, `from module import name`, and `import module as alias`. The `if __name__ == "__main__"` guard prevents code from running on import. Python searches for modules in `sys.path`. The `__all__` variable controls what is exported with `from module import *`. Properly structuring ML code into modules improves reusability, testability, and collaboration.

## Key Takeaways

- Every `.py` file is a module; import it using its filename (without `.py`).
- Use `if __name__ == "__main__"` to make modules both importable and executable.
- Prefer explicit `from module import name` over `import *`.
- Module search order: current dir → PYTHONPATH → standard lib → site-packages.
- Use `sys.modules` to see all currently imported modules.
- Avoid circular imports by restructuring or using lazy imports.
