# Concept: Packages and __init__.py

## Concept ID

PYT-022

## Difficulty

Intermediate

## Domain

Python

## Module

Functions and Modules

## Learning Objectives

- Understand packages as directory-based module collections
- Use `__init__.py` to initialize packages and control imports
- Define `__all__` in `__init__.py` to restrict wildcard imports
- Differentiate between absolute and relative imports
- Use relative imports (`.` and `..`) to import within packages
- Understand namespace packages (PEP 420)

## Prerequisites

- PYT-020: Modules and Imports
- Understanding of directory structures
- Familiarity with `__all__` and `sys.path`

## Definition

A **package** is a directory containing Python modules and a special `__init__.py` file (in Python 3.3+, `__init__.py` can be omitted for namespace packages). Packages allow you to organize related modules hierarchically. The `__init__.py` file is executed when the package is imported and can initialize package-level state, define `__all__`, or import submodules.

## Intuition

If a module is a single chapter, a package is a book — a collection of related chapters organized into sections. The `__init__.py` is like the book's table of contents and introduction: it tells you what the book contains and exports the most important concepts. The package structure mirrors the import syntax: `from book.chapter1.section2 import topic` matches the directory hierarchy `book/chapter1/section2.py`.

## Why This Concept Matters

Packages are essential for organizing large codebases. In AI/ML projects, you typically have packages for `data/`, `models/`, `train/`, `evaluate/`, and `utils/`. The `__init__.py` file controls what users of your package can access and provides a clean API surface. Understanding absolute vs relative imports prevents import errors in complex projects.

## Real World Examples

1. **NumPy**: `import numpy` loads the numpy package; submodules like `numpy.linalg` are lazily loaded.
2. **PyTorch**: The `torch` package contains `torch.nn`, `torch.optim`, `torch.utils.data` as subpackages.
3. **Scikit-learn**: `sklearn.ensemble`, `sklearn.preprocessing`, `sklearn.metrics` are subpackages.
4. **Django**: The `django.urls`, `django.db`, `django.views` packages organize the web framework.
5. **Custom ML projects**: `from myproject.data.loader import DataLoader; from myproject.models.resnet import ResNet18`

## AI/ML Relevance

ML projects are typically organized as packages. A project called `ml_project/` might have:
- `ml_project/data/` — data loading and preprocessing
- `ml_project/models/` — model architectures
- `ml_project/training/` — training loops and callbacks
- `ml_project/evaluation/` — metrics and visualization
- `ml_project/config/` — configuration management

The `__init__.py` files re-export key classes and functions so users can write `from ml_project import Trainer` instead of deep import paths.

## Code Examples

### Example 1: Basic package structure

```
mypackage/
├── __init__.py
├── module_a.py
└── module_b.py
```

```python
# __init__.py
print("Initializing mypackage")

# module_a.py
def hello():
    return "Hello from module_a"

# module_b.py
def goodbye():
    return "Goodbye from module_b"
```

```python
# Usage
import mypackage
# Output: Initializing mypackage

from mypackage import module_a, module_b
print(module_a.hello())
# Output: Hello from module_a
print(module_b.goodbye())
# Output: Goodbye from module_b
```

### Example 2: `__init__.py` with import shortcuts

```python
# mypackage/__init__.py
"""Machine learning utility package."""

from .module_a import hello
from .module_b import goodbye

__all__ = ["hello", "goodbye", "Utils"]

class Utils:
    @staticmethod
    def version():
        return "1.0.0"
```

```python
# Now users can do:
from mypackage import hello, goodbye, Utils
print(hello())
# Output: Hello from module_a
print(Utils.version())
# Output: 1.0.0

# Or with star import:
from mypackage import *
print(hello())
# Output: Hello from module_a
```

### Example 3: Nested package structure

```
project/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── loader.py
│   └── preprocessing.py
├── models/
│   ├── __init__.py
│   ├── cnn.py
│   └── rnn.py
└── utils/
    ├── __init__.py
    └── metrics.py
```

```python
# project/data/__init__.py
from .loader import DataLoader
from .preprocessing import normalize, standardize

__all__ = ["DataLoader", "normalize", "standardize"]

# project/__init__.py
from . import data, models, utils

__all__ = ["data", "models", "utils"]
```

```python
# Usage
import project
loader = project.data.DataLoader("train.csv")

# Or
from project.data import DataLoader
from project.models.cnn import CNN
from project.utils.metrics import accuracy
```

### Example 4: Relative imports (`.` and `..`)

```python
# project/data/preprocessing.py
"""Contains preprocessing functions."""

# Absolute import
from project.utils.metrics import accuracy

# Relative import (same as absolute above)
from ..utils.metrics import accuracy

# Relative import of sibling module
from .loader import DataLoader

def preprocess_pipeline(data):
    loader = DataLoader()
    # ... preprocessing logic
    return data
```

### Example 5: Package `__init__.py` with lazy imports

```python
# mypackage/__init__.py
"""Package with lazy imports for faster startup."""

__all__ = ["heavy_function"]

def heavy_function():
    """Import heavy dependencies only when needed."""
    import numpy as np  # Lazy import
    return np.array([1, 2, 3])
```

### Example 6: `__all__` in `__init__.py`

```python
# mypackage/__init__.py
"""Package with controlled exports."""

from .module_a import public_func as pf
from .module_b import _private_helper  # Still importable but not in __all__

__all__ = ["pf", "UtilityClass"]

class UtilityClass:
    pass
```

```python
# from mypackage import *
# Only imports pf and UtilityClass
# _private_helper is NOT imported (though it's accessible via mypackage._private_helper)
```

### Example 7: Namespace packages (Python 3.3+, PEP 420)

```
project/
├── package/
│   ├── sub1/
│   │   └── module.py     # No __init__.py
│   └── sub2/
│       └── module.py     # No __init__.py
```

```python
# Without __init__.py files, this is a namespace package.
# Multiple directories can contribute to the same namespace.

# import package.sub1.module
# import package.sub2.module
```

### Example 8: Complete ML package example

```
ml_project/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── settings.py
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── transforms.py
├── models/
│   ├── __init__.py
│   ├── base.py
│   ├── resnet.py
│   └── vit.py
└── train/
    ├── __init__.py
    ├── trainer.py
    └── callbacks.py
```

```python
# ml_project/__init__.py
from . import config, data, models, train
from .config.settings import Config
from .train.trainer import Trainer

__version__ = "0.1.0"
__all__ = ["Config", "Trainer"]

# From another script:
# import ml_project
# config = ml_project.Config()
# trainer = ml_project.Trainer(config)
```

```python
# ml_project/data/__init__.py
from .dataset import ImageDataset, TextDataset
from .transforms import normalize, augment

__all__ = ["ImageDataset", "TextDataset", "normalize", "augment"]
```

## Common Mistakes

1. **Forgetting `__init__.py` in Python < 3.3**: Without `__init__.py`, Python versions before 3.3 would not recognize the directory as a package.
2. **Circular imports with relative imports**: If `module_a` imports from `module_b` and vice versa, you get circular import errors. Restructure or use lazy imports.
3. **Using relative imports outside packages**: Relative imports (`.`, `..`) only work inside packages. Running a script directly that uses relative imports raises `ImportError`.
4. **Importing `__init__.py` contents incorrectly**: If `__init__.py` imports from submodules, ensure those submodules are importable before `__init__.py` finishes executing.
5. **Accidental namespace packages**: If you forget `__init__.py` in Python and the directory is on `sys.path`, it becomes a namespace package, which may not behave as expected.

## Interview Questions

### Beginner

1. What is a package in Python? How does it differ from a module?
2. What is the purpose of `__init__.py` in a package?
3. How do you import a module from a subpackage?
4. What is the difference between absolute and relative imports?
5. What is `__all__` in an `__init__.py` file?

### Intermediate

1. Explain how relative imports work with `.` (single dot) and `..` (double dot).
2. What are namespace packages (PEP 420) and when would you use them?
3. How does `__init__.py` affect the import performance of a package?
4. Write the contents of an `__init__.py` that re-exports specific functions from submodules.
5. What happens if you run a script that uses relative imports directly (not as a module)?

### Advanced

1. Design a package structure for a deep learning library that supports plugin architectures.
2. How would you implement lazy loading of submodules in `__init__.py`?
3. Explain how Python resolves imports in a deeply nested package structure. What is the role of `__path__`?

## Practice Problems

### Easy

1. Create a package `shapes` with modules `circle.py` and `square.py`.
2. Add an `__init__.py` that imports both modules.
3. Create a subpackage `shapes/3d` with a module `sphere.py`.
4. Import a function from a sibling module using absolute import.
5. Define `__all__` in a package to export only two functions.

### Medium

1. Create a package with three levels of nesting and import a function from the deepest level.
2. Implement an `__init__.py` that lazily imports a heavy module only when accessed.
3. Create a namespace package split across two directories.
4. Write an `__init__.py` that reads a version from a `VERSION` file.
5. Create a package that re-exports functions from submodules at the top level.

### Hard

1. Design a plugin system where plugins are discovered in a `plugins/` subpackage and registered automatically.
2. Implement a package that can be imported both as a package and as a module (dual-mode).
3. Write an `__init__.py` that dynamically imports all submodules based on configuration.

## Solutions

### Easy Solutions

```python
# shapes/__init__.py
from . import circle, square

# shapes/circle.py
def area(r): return 3.14159 * r ** 2

# shapes/square.py
def area(s): return s ** 2

# Usage: from shapes import circle; circle.area(5)

# shapes/__init__.py with __all__
__all__ = ["circle"]
```

### Medium Solutions

```python
# Lazy import __init__.py
class LazyLoader:
    def __init__(self, module_name):
        self.module_name = module_name
        self._module = None
    def __getattr__(self, name):
        if self._module is None:
            import importlib
            self._module = importlib.import_module(self.module_name)
        return getattr(self._module, name)

heavy_module = LazyLoader(".heavy_module")

# Version from file
import os
with open(os.path.join(os.path.dirname(__file__), "VERSION")) as f:
    __version__ = f.read().strip()
```

### Hard Solutions

```python
# 1 — Plugin system
import pkgutil
import importlib

def discover_plugins():
    plugins = {}
    package = importlib.import_module(__name__)
    for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
        if modname.startswith("plugin_"):
            module = importlib.import_module(f".{modname}", __name__)
            if hasattr(module, "register"):
                plugins[modname] = module.register()
    return plugins

# 3 — Dynamic import
import importlib
import pkgutil

def _auto_import():
    __all__ = []
    for importer, modname, ispkg in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f".{modname}", __name__)
        globals()[modname] = module
        __all__.append(modname)
    return __all__

__all__ = _auto_import()
```

## Related Concepts

- PYT-020: Modules and Imports
- PYT-021: Standard Library
- PYT-023: Virtual Environments

## Next Concepts

- PYT-040: Package Distribution (setup.py, pyproject.toml)
- PYT-050: Creating and Publishing Packages
- PYT-060: Dependency Management

## Summary

Packages are directories of modules with optional `__init__.py` files. The `__init__.py` file is executed on package import and can re-export names, initialize state, or define `__all__`. Relative imports (`.` for current package, `..` for parent) simplify intra-package imports. Namespace packages (no `__init__.py`) allow a package to span multiple directories. In ML projects, packages organize data loading, models, training, and evaluation into clean, hierarchical namespaces.

## Key Takeaways

- A package is a directory with an optional `__init__.py` file.
- Use `__init__.py` to re-export submodule contents for a clean API.
- `__all__` in `__init__.py` controls what `from package import *` exports.
- Relative imports (`.` and `..`) are for importing within a package.
- Absolute imports specify the full path from the project root.
- Namespace packages (PEP 420) allow splitting a package across directories.
- Organize ML projects into logical packages: `data/`, `models/`, `train/`, `utils/`.
