# Concept: Virtual Environments

## Concept ID

PYT-023

## Difficulty

Intermediate

## Domain

Python

## Module

Functions and Modules

## Learning Objectives

- Understand the purpose and benefits of virtual environments
- Create and manage virtual environments with `python -m venv`
- Activate and deactivate virtual environments on different platforms
- Install, upgrade, and remove packages with `pip`
- Create and use `requirements.txt` for reproducible environments
- Compare `venv`, `conda`, and other environment management tools
- Manage ML library versions with CUDA compatibility in mind

## Prerequisites

- PYT-020: Modules and Imports
- Basic familiarity with command-line interface
- Understanding of package dependencies

## Definition

A **virtual environment** is an isolated Python environment that maintains its own Python interpreter, site-packages, and scripts. It allows different projects to use different versions of the same library without conflicts. The built-in `venv` module creates virtual environments:

```bash
python -m venv myenv
```

Virtual environments solve the "dependency hell" problem where two projects require conflicting versions of a package.

## Intuition

Imagine you have two kitchens (projects). One kitchen needs a specific brand of blender (library v1.0), and the other needs a different brand (library v2.0). Without virtual environments, you can only have one blender brand in your single kitchen. Virtual environments are like having separate, self-contained kitchens — each with its own appliances, utensils, and ingredients. You can switch between kitchens without interference.

## Why This Concept Matters

Virtual environments are non-negotiable in professional Python development. They prevent version conflicts, enable reproducible builds, and isolate project-specific dependencies. In AI/ML, where library versions are critical for reproducibility and GPU compatibility, virtual environments are essential. PyTorch 2.x may require CUDA 11.8, while TensorFlow 2.x may need CUDA 11.2 — a virtual environment ensures each project gets exactly the libraries it needs.

## Real World Examples

1. **ML research**: One project uses PyTorch 1.13 (CUDA 11.6), another uses PyTorch 2.0 (CUDA 11.8). Two virtual environments keep them separate.
2. **Web development**: A Flask 1.x project and a FastAPI project can coexist with different dependency trees.
3. **Legacy code**: Maintaining an old Django 2.2 project alongside new Django 4.x projects.
4. **CI/CD**: Each CI job creates a fresh virtual environment for clean, reproducible builds.
5. **Deployment**: Virtual environments ensure the production environment matches the development environment exactly.

## AI/ML Relevance

AI/ML projects are particularly sensitive to version mismatches. PyTorch, TensorFlow, and JAX each have specific CUDA and cuDNN requirements. A virtual environment for each project ensures:
- Correct CUDA toolkit version
- Compatible PyTorch/TensorFlow builds
- Isolated Python package dependencies
- Reproducible research results
- Clean upgrade paths

The typical ML workflow: create a venv, install PyTorch with the right CUDA version, then install project dependencies from `requirements.txt`.

## Code Examples

### Example 1: Creating and activating a virtual environment

```bash
# Create a virtual environment
python -m venv ml_project_env

# Activate on Windows (Command Prompt)
ml_project_env\Scripts\activate

# Activate on Windows (PowerShell)
ml_project_env\Scripts\Activate.ps1

# Activate on macOS/Linux
source ml_project_env/bin/activate

# After activation, the prompt changes to show the environment name
# (ml_project_env) D:\Ajay\ai-teacher-knowledge-base>

# Deactivate
deactivate
```

### Example 2: Installing packages with pip

```bash
# After activation, install packages
pip install numpy pandas matplotlib

# Install a specific version
pip install torch==2.0.1

# Install with CUDA support (PyTorch example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install from requirements file
pip install -r requirements.txt

# Upgrade a package
pip install --upgrade numpy

# Uninstall a package
pip uninstall matplotlib
```

### Example 3: `pip freeze` and `requirements.txt`

```bash
# Generate a requirements file with pinned versions
pip freeze > requirements.txt
```

```
# requirements.txt content example:
numpy==1.26.2
pandas==2.1.4
torch==2.1.2
torchvision==0.16.2
scikit-learn==1.3.2
matplotlib==3.8.2
jupyter==1.0.0
```

```bash
# Recreate the exact environment
python -m venv new_env
new_env\Scripts\activate
pip install -r requirements.txt
```

### Example 4: Listing installed packages

```bash
# List all installed packages
pip list

# Sample output:
# Package            Version
# ------------------ -------
# numpy              1.26.2
# pandas             2.1.4
# pip                23.3.1
# torch              2.1.2
# ...

# Show information about a specific package
pip show torch

# Sample output:
# Name: torch
# Version: 2.1.2
# Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
# Requires: filelock, typing-extensions, sympy, networkx, jinja2, fsspec
```

### Example 5: Virtual environment directory structure

```
ml_project_env/
├── Include/
├── Lib/
│   └── site-packages/    # Installed packages live here
├── Scripts/
│   ├── activate          # Activation script (Unix)
│   ├── activate.bat      # Activation script (Windows CMD)
│   ├── Activate.ps1      # Activation script (Windows PowerShell)
│   ├── deactivate.tcl
│   ├── pip.exe
│   ├── python.exe
│   └── pythonw.exe
└── pyvenv.cfg            # Configuration file
```

```python
# Check the Python interpreter location inside a venv
import sys
print(sys.executable)
# Output: D:\Ajay\ai-teacher-knowledge-base\ml_project_env\Scripts\python.exe
```

### Example 6: Using a virtual environment in VS Code

```python
# In VS Code, select the Python interpreter from the virtual environment:
# 1. Press Ctrl+Shift+P
# 2. Type "Python: Select Interpreter"
# 3. Choose the one in your venv (e.g., .\ml_project_env\Scripts\python.exe)
# 4. VS Code will use the venv's Python and packages
```

### Example 7: Reproducible ML environment setup

```python
# setup_ml_env.py
"""
Script to set up a complete ML environment.

Run: python setup_ml_env.py
Creates a virtual environment and installs ML dependencies.
"""

import subprocess
import sys
import os

VENV_NAME = "ml_env"
PYTHON = sys.executable

def run(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def setup():
    # 1. Create virtual environment
    run(f"{PYTHON} -m venv {VENV_NAME}")

    # 2. Determine the pip path
    if sys.platform == "win32":
        pip_path = os.path.join(VENV_NAME, "Scripts", "pip")
        python_path = os.path.join(VENV_NAME, "Scripts", "python")
    else:
        pip_path = os.path.join(VENV_NAME, "bin", "pip")
        python_path = os.path.join(VENV_NAME, "bin", "python")

    # 3. Upgrade pip
    run(f"{python_path} -m pip install --upgrade pip")

    # 4. Install core ML packages
    run(f"{pip_path} install numpy pandas scikit-learn matplotlib jupyter")

    # 5. Install PyTorch (CPU version for compatibility)
    run(f"{pip_path} install torch torchvision torchaudio")

    # 6. Freeze the environment
    run(f"{pip_path} freeze > requirements.txt")

    print(f"\nVirtual environment '{VENV_NAME}' created!")
    print(f"Activate with: {VENV_NAME}\\Scripts\\activate")

if __name__ == "__main__":
    setup()
```

### Example 8: Managing multiple environments with a requirements directory

```
project/
├── requirements/
│   ├── base.txt          # Common dependencies
│   ├── dev.txt           # Development dependencies (testing, linting)
│   ├── train.txt         # Training dependencies (torch, CUDA)
│   └── deploy.txt        # Production dependencies (flask, gunicorn)
├── scripts/
│   └── setup_env.py
└── README.md
```

```bash
# Creating different environments for different stages
python -m venv dev_env
dev_env\Scripts\activate
pip install -r requirements/base.txt
pip install -r requirements/dev.txt
```

```bash
# base.txt content:
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
python-dotenv>=1.0.0

# dev.txt content:
-r base.txt
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.11.0
flake8>=6.1.0

# train.txt content:
-r base.txt
torch>=2.0.0
torchvision>=0.15.0
tensorboard>=2.14.0

# deploy.txt content:
-r base.txt
flask>=3.0.0
gunicorn>=21.2.0
```

### Example 9: Conda environments (alternative to venv)

```bash
# Conda is more than a virtual environment — it's a package manager
# that handles non-Python dependencies (CUDA, cuDNN, etc.)

# Create a conda environment with specific Python version
conda create -n ml_env python=3.10

# Activate
conda activate ml_env

# Install packages from conda-forge
conda install -c conda-forge numpy pandas matplotlib

# Install PyTorch with CUDA via conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Export environment
conda env export > environment.yml

# Create from file
conda env create -f environment.yml

# Deactivate
conda deactivate

# List environments
conda env list
```

## Common Mistakes

1. **Committing the virtual environment directory**: Never commit the `venv/` directory to version control. Add it to `.gitignore`.
2. **Forgetting to activate the environment**: Running `pip install` without activation installs packages globally.
3. **Using `pip freeze` without activation**: `pip freeze` lists globally installed packages, not your project's dependencies.
4. **Not pinning versions in `requirements.txt`**: Without versions, builds are not reproducible. Use `pip freeze > requirements.txt` after installing.
5. **Mixing conda and pip**: Installing the same package with both conda and pip can cause conflicts. Prefer one or the other where possible.

## Interview Questions

### Beginner

1. What is a virtual environment and why is it useful?
2. How do you create a virtual environment using the built-in `venv` module?
3. How do you activate and deactivate a virtual environment on Windows?
4. What command installs packages from a `requirements.txt` file?
5. What is `pip freeze` used for?

### Intermediate

1. What is the difference between `venv` and `conda`?
2. How do you create a virtual environment with a specific Python version?
3. What should you include in `.gitignore` regarding virtual environments?
4. Explain the structure of a generated `requirements.txt` file.
5. How do you upgrade all packages in a virtual environment?

### Advanced

1. How do virtual environments work internally? What happens when you activate one?
2. Explain the difference between `setup.py`, `requirements.txt`, and `pyproject.toml`.
3. How would you manage environments in a monorepo with multiple Python services?

## Practice Problems

### Easy

1. Create a virtual environment named `test_env`.
2. Activate the environment and install `requests`.
3. Deactivate the environment and verify `requests` is not available globally.
4. Generate a `requirements.txt` file from an activated environment.
5. Create a new environment and install packages from the `requirements.txt`.

### Medium

1. Create a `requirements.txt` that pins specific versions of `numpy`, `pandas`, `matplotlib`.
2. Set up a virtual environment for a PyTorch project with CUDA support.
3. Create a script that automatically creates a virtual environment and installs dependencies.
4. Compare the package lists between two virtual environments with different Python versions.
5. Create a `.gitignore` file that excludes virtual environment directories.

### Hard

1. Write a Makefile or PowerShell script that sets up a complete development environment (venv, install deps, create config files).
2. Implement a dependency conflict resolver that checks `requirements.txt` for incompatible version constraints.
3. Create a multi-stage Docker build that uses virtual environments for dependency isolation.

## Solutions

### Easy Solutions

```bash
# 1
python -m venv test_env

# 2 (Windows)
test_env\Scripts\activate
pip install requests

# 3
deactivate
python -c "import requests"  # Should fail or import global one

# 4
pip freeze > requirements.txt

# 5
python -m venv new_env
new_env\Scripts\activate
pip install -r requirements.txt
```

### Medium Solutions

```bash
# 2 — PyTorch with CUDA
python -m venv torch_env
torch_env\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

```python
# 3 — Setup script
import subprocess, sys, os

VENV = "auto_env"
PY = sys.executable

subprocess.check_call([PY, "-m", "venv", VENV])
pip = os.path.join(VENV, "Scripts", "pip") if sys.platform == "win32" else os.path.join(VENV, "bin", "pip")
subprocess.check_call([pip, "install", "-r", "requirements.txt"])
print(f"Environment {VENV} ready")
```

```gitignore
# 4 — .gitignore
venv/
.env/
*.egg-info/
dist/
build/
```

### Hard Solutions

```powershell
# 1 — PowerShell setup script
param(
    [string]$EnvName = "project_env",
    [string]$Requirements = "requirements.txt"
)

# Create venv
python -m venv $EnvName
$pip = Join-Path $EnvName "Scripts\pip.exe"

# Upgrade pip
& $pip install --upgrade pip

# Install dependencies
if (Test-Path $Requirements) {
    & $pip install -r $Requirements
} else {
    Write-Warning "$Requirements not found"
}

Write-Host "Environment '$EnvName' created successfully" -ForegroundColor Green
```

```python
# 3 — Dockerfile with venv
"""
FROM python:3.10-slim

WORKDIR /app

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run with venv Python
CMD ["python", "main.py"]
"""
```

## Related Concepts

- PYT-020: Modules and Imports
- PYT-021: Standard Library
- PYT-022: Packages and __init__.py

## Next Concepts

- PYT-040: Package Distribution
- PYT-050: CI/CD for Python
- PYT-060: Docker for Python Applications

## Summary

Virtual environments isolate Python dependencies per project using `python -m venv`. Activation modifies `PATH` and `sys.prefix` to point to the environment's interpreter and site-packages. The `requirements.txt` file enables reproducible environments via `pip freeze`. Conda environments extend this to non-Python dependencies like CUDA. In AI/ML, virtual environments are critical for managing library version conflicts and ensuring reproducible research.

## Key Takeaways

- Always use virtual environments for project isolation.
- Create with `python -m venv env_name`; activate with `env_name\Scripts\activate`.
- Use `pip freeze > requirements.txt` to lock dependency versions.
- Never commit the virtual environment directory; do commit `requirements.txt`.
- Use `pip install -r requirements.txt` to recreate an environment.
- Conda manages both Python and non-Python dependencies (like CUDA).
- For AI/ML, always create a fresh venv per project to avoid CUDA/library conflicts.
