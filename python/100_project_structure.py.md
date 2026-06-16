# Concept: Project Structure for ML/AI

## Concept ID

PYT-100

## Difficulty

Intermediate

## Domain

Python

## Module

Python for ML/AI

## Learning Objectives

- Structure ML projects using the src/ layout with proper `__init__.py` files
- Create `setup.py` and `pyproject.toml` for packaging
- Write type hints and Google/Numpy-style docstrings
- Set up testing with pytest, linting with pre-commit, and environments with tox
- Organize a complete ML project with README, LICENSE, and configuration

## Prerequisites

- Python project experience (modules, packages, imports)
- Familiarity with ML workflow (data processing, training, evaluation)
- Basic command-line knowledge (pip, virtual environments)

## Definition

A well-structured ML project follows conventions that make code reproducible, shareable, and production-ready. The standard approach is the src/ layout:

```
my_project/
  pyproject.toml           # Project metadata & build config
  README.md                # Project description & usage
  LICENSE                  # Open-source license
  setup.cfg                # Package configuration
  src/
    my_project/
      __init__.py          # Makes it a package
      data/
        __init__.py
        make_dataset.py
        preprocessing.py
      features/
        __init__.py
        build_features.py
      models/
        __init__.py
        train_model.py
        predict_model.py
      visualization/
        __init__.py
        visualize.py
  tests/
    __init__.py
    test_data.py
    test_models.py
  notebooks/
    01_eda.ipynb
  data/
    raw/
    processed/
  models/                  # Saved model files (gitignored)
  reports/
  .pre-commit-config.yaml
  tox.ini
```

Key practices: Type hints for function signatures, Google/Numpy-style docstrings, pytest for unit tests, tox for multi-environment testing, and pre-commit for automated code quality checks.

## Intuition

Think of an ML project like a restaurant kitchen. The src/ directory is the kitchen with organized stations (data prep, modeling, evaluation). notebooks/ is the test kitchen for experimentation. data/ is the pantry with raw ingredients (raw) and prepared items (processed). tests/ are quality checks ensuring consistency. pyproject.toml is the menu telling others what the project offers.

The src/ layout prevents import confusion: `pip install -e .` makes your package importable, and tests import from the installed package (not the local directory), mirroring real-world usage.

## Why This Concept Matters

- **Reproducibility:** Others can understand and run the project
- **Collaboration:** Team members work independently on different modules
- **Deployment:** Clean structure simplifies Docker builds, CI/CD, and API serving
- **Reusability:** Well-packaged code can be pip-installed into other projects
- **Professionalism:** Following standards signals quality to employers and collaborators

## Real World Examples

1. **Kaggle Competition:** A clean src/ layout lets a team work on feature engineering, modeling, and ensembling simultaneously.
2. **ML Research Lab:** Each paper has a src/ structured repository enabling reproduction.
3. **Enterprise ML Platform:** A central ml_toolkit/ package with standardized preprocessing and model modules used across teams.
4. **Open Source ML Library:** A well-structured library with pyproject.toml, pytest tests, and pre-commit hooks.
5. **Client-Facing ML Product:** A production ML service with modular data processing, model inference, and API code.

## AI/ML Relevance

- **ML Pipelines Everywhere:** Every ML project benefits from clean structure
- **MLOps Integration:** Structured projects integrate with MLflow, DVC, Weights and Biases
- **Model Serving:** Clean separation of preprocessing, model, and API makes serving straightforward
- **CI/CD for ML:** Automated testing, linting, and validation in ML pipelines
- **Collaboration:** Data scientists with different skill sets work on the same project

## Code Examples

### Example 1: pyproject.toml
```toml
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "my_ml_project"
version = "0.1.0"
description = "ML project for customer churn prediction"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Data Scientist", email = "ds@example.com"}
]
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "torch>=2.0",
    "matplotlib>=3.7",
    "seaborn>=0.12",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "ruff>=0.1",
    "pre-commit>=3.0",
    "tox>=4.0",
    "mypy>=1.0",
]

[tool.setuptools]
package-dir = {"" = "src"}
packages = {find = {where = ["src"]}}

[tool.ruff]
line-length = 100
target-version = "py39"
select = ["E", "F", "I", "N", "W"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```
```
# Output: A complete pyproject.toml with build config, dependencies, dev extras, and tool settings.
```

### Example 2: src layout with __init__.py
```python
# src/my_project/__init__.py
"""My ML Project: Customer Churn Prediction.

Modules:
    data: Data loading and preprocessing
    features: Feature engineering
    models: Model training and prediction
    visualization: Plotting utilities
"""

__version__ = "0.1.0"
```

```python
# src/my_project/data/__init__.py
from .make_dataset import load_data, split_data
from .preprocessing import scale_features, encode_categories

__all__ = ["load_data", "split_data", "scale_features", "encode_categories"]
```

```python
# src/my_project/models/train_model.py
from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any]
) -> Pipeline:
    """Train a RandomForest classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        params: Model hyperparameters.

    Returns:
        Fitted sklearn Pipeline.

    Raises:
        ValueError: If X_train and y_train have mismatched sizes.
    """
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train ({len(X_train)}) and y_train ({len(y_train)}) "
            f"must have the same number of samples."
        )

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """Evaluate a trained model.

    Args:
        model: Fitted model pipeline.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dictionary of evaluation metrics.
    """
    from sklearn.metrics import accuracy_score, f1_score

    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average="weighted")),
    }
```
```
# Output: Package init files enabling clean imports, and a train_model module with type hints and Google-style docstrings.
```

### Example 3: Type hints throughout
```python
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


def preprocess_data(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    target_col: str,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split and preprocess data into train/test sets.

    Args:
        df: Input DataFrame.
        numeric_cols: List of numeric column names.
        categorical_cols: List of categorical column names.
        target_col: Name of the target column.
        test_size: Fraction of data for testing (default 0.2).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).

    Example:
        >>> X_train, X_test, y_train, y_test = preprocess_data(
        ...     df, ["age", "income"], ["gender"], "churn"
        ... )
    """
    X = df[numeric_cols + categorical_cols]
    y = df[target_col]

    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size,
                           random_state=random_state, stratify=y)
```
```
# Output: A function with full type hints, Google-style docstring including Example section, and clear return type.
```

### Example 4: Numpy-style docstring
```python
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels, shape (n_samples,).
    y_pred : np.ndarray
        Predicted labels, shape (n_samples,).

    Returns
    -------
    metrics : Dict[str, float]
        Dictionary containing 'accuracy', 'precision', 'recall', 'f1'.

    Examples
    --------
    >>> y_true = np.array([0, 1, 0, 1])
    >>> y_pred = np.array([0, 1, 1, 1])
    >>> calculate_metrics(y_true, y_pred)
    {'accuracy': 0.75, 'precision': 0.67, 'recall': 1.0, 'f1': 0.8}
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
```
```
# Output: Numpy-style docstring with Parameters, Returns, and Examples sections.
```

### Example 5: pytest tests
```python
# tests/test_preprocessing.py
import pytest
import pandas as pd
import numpy as np
from my_project.data.preprocessing import scale_features, encode_categories


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "age": [25, 30, 35, 40],
        "income": [50000, 60000, 70000, 80000],
        "gender": ["M", "F", "F", "M"],
        "target": [0, 1, 0, 1],
    })


class TestPreprocessing:
    """Tests for preprocessing functions."""

    def test_scale_features(self, sample_df: pd.DataFrame):
        """Test that scaled features have mean approx 0 and std approx 1."""
        scaled = scale_features(sample_df, ["age", "income"])
        assert isinstance(scaled, pd.DataFrame)
        np.testing.assert_almost_equal(scaled["age"].mean(), 0.0, decimal=1)
        np.testing.assert_almost_equal(scaled["age"].std(), 1.0, decimal=1)

    def test_scale_features_empty_cols(self, sample_df: pd.DataFrame):
        """Test scaling with empty column list returns original."""
        result = scale_features(sample_df, [])
        assert result.equals(sample_df)

    def test_encode_categories(self, sample_df: pd.DataFrame):
        """Test one-hot encoding produces expected columns."""
        encoded = encode_categories(sample_df, ["gender"])
        assert "gender_M" in encoded.columns
        assert "gender_F" in encoded.columns
        assert encoded.shape[1] == sample_df.shape[1] + 1  # 2 cols - 1 original + 4 original

    def test_encode_categories_unknown_handling(self):
        """Test that unknown categories are handled."""
        train = pd.DataFrame({"cat": ["a", "b", "c"]})
        test = pd.DataFrame({"cat": ["a", "d"]})
        encoded_train, encoder = encode_categories(train, ["cat"], return_encoder=True)
        encoded_test = encoder.transform(test)
        # 'd' should be all zeros
        assert encoded_test[encoded_test.columns[-1]].sum() > 0  # at least 'a' is encoded
```
```
# Output: Complete pytest test file with fixtures, test classes, and multiple test functions.
```

### Example 6: pre-commit configuration
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=500']

  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black
        args: [--line-length=100]

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.8
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]
        additional_dependencies: [types-pandas, types-sklearn]
```
```
# Output: Pre-commit config with hooks for whitespace, black formatting, ruff linting, and mypy type checking.
```

### Example 7: tox configuration
```ini
# tox.ini
[tox]
envlist = py39, py310, py311
isolated_build = True

[testenv]
deps =
    pytest>=7.0
    pytest-cov>=4.0
    numpy
    pandas
    scikit-learn
commands =
    pytest tests/ --cov=src/my_project --cov-report=html --cov-report=term

[testenv:lint]
deps =
    black
    ruff
commands =
    black --check --diff src/ tests/
    ruff check src/ tests/

[testenv:typecheck]
deps =
    mypy
    types-pandas
    types-sklearn
commands =
    mypy src/ --ignore-missing-imports

[testenv:docs]
deps =
    sphinx
    sphinx-rtd-theme
commands =
    sphinx-build -b html docs/source docs/build
```
```
# Output: Tox configuration running tests across Python 3.9-3.11, plus lint, typecheck, and docs environments.
```

### Example 8: Config file (YAML)
```yaml
# configs/config.yaml
data:
  raw_path: "data/raw/customers.csv"
  processed_path: "data/processed/customers_clean.csv"
  test_size: 0.2
  random_state: 42

features:
  numeric: ["age", "income", "tenure", "usage_frequency"]
  categorical: ["gender", "region", "plan_type"]
  target: "churn"

model:
  name: "random_forest"
  params:
    n_estimators: 200
    max_depth: 10
    min_samples_split: 5
    min_samples_leaf: 2
    random_state: 42

training:
  cv_folds: 5
  scoring: "roc_auc"
  search_type: "randomized"
  n_iter: 50
```
```python
# src/my_project/config.py
from typing import Dict, Any
import yaml
from pathlib import Path

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Dictionary with configuration values.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# Usage
config = load_config()
print(f"Model: {config['model']['name']}")
print(f"Features: {config['features']['numeric'] + config['features']['categorical']}")
```
```
# Output: YAML config file with data/features/model/training sections, plus loader function.
```

### Example 9: MLflow integration
```python
# src/my_project/models/train_with_mlflow.py
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def train_with_tracking(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict[str, Any],
    experiment_name: str = "churn_prediction",
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """Train model with MLflow experiment tracking.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        params: Model hyperparameters.
        experiment_name: MLflow experiment name.

    Returns:
        Tuple of (trained_model, metrics_dict).
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)

        # Train
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average="weighted"),
        }

        # Log metrics and model
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print(f"Logged to MLflow run: {mlflow.active_run().info.run_id}")

    return model, metrics
```
```
# Output: Training function with MLflow tracking — logs params, metrics, and model artifact.
```

### Example 10: Makefile for common commands
```makefile
# Makefile
.PHONY: install test lint typecheck clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ --cov=src/my_project -v

lint:
	ruff check src/ tests/
	black --check --diff src/ tests/

typecheck:
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	ruff check --fix src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

precommit:
	pre-commit run --all-files

tox:
	tox

help:
	@echo "Available commands:"
	@echo "  install    - Install package in dev mode"
	@echo "  test       - Run tests with coverage"
	@echo "  lint       - Check code style"
	@echo "  format     - Auto-format code"
	@echo "  typecheck  - Run mypy type checking"
	@echo "  clean      - Remove build artifacts"
```
```
# Output: Makefile with common dev commands — install, test, lint, format, typecheck, clean.
```

### Example 11: README.md template
```markdown
# Customer Churn Prediction

ML project to predict customer churn using behavioral and demographic data.

## Project Structure

```
src/my_project/          # Main package
  data/                  # Data loading & preprocessing
  features/              # Feature engineering
  models/                # Training & prediction
  visualization/         # Plotting utilities
notebooks/               # EDA and experimentation
tests/                   # Unit tests
data/                    # Data files (gitignored)
models/                  # Saved models (gitignored)
```

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```python
from my_project.data.make_dataset import load_data
from my_project.models.train_model import train_model

df = load_data("data/raw/customers.csv")
model = train_model(X_train, y_train, {"n_estimators": 200})
```

## Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Random Forest | 0.892 | 0.871 |
| Logistic Regression | 0.834 | 0.812 |

## License

MIT
```
```
# Output: README.md with project description, structure, installation, usage, and results table.
```

## Common Mistakes

1. **Putting code in notebooks without refactoring into src/.** Notebooks are for exploration, not production. Always refactor reusable logic into src/ modules.
2. **Missing `__init__.py` files.** Without them, Python doesn't recognize directories as packages. All subdirectories under src/ need one.
3. **Hardcoding paths.** Use `pathlib.Path` or config files. Never hardcode `data/raw/file.csv` — use relative paths from a config.
4. **Committing large data files or model binaries.** Add `data/processed/`, `models/`, and `*.pth`/`*.h5` to `.gitignore`. Use DVC for data versioning.
5. **No type hints.** Type hints catch bugs early and serve as documentation. Especially important in team projects.
6. **No tests.** ML code is prone to silent failures (wrong data shapes, leaky preprocessing). pytest catches these.
7. **Mixing raw and processed data.** Keep raw data immutable. Never overwrite raw files. Processed data should be reproducible from raw + scripts.
8. **Not pinning dependency versions.** Without version pins, `pip install` may break months later. Use `pip freeze > requirements.txt` or precise version specifiers in `pyproject.toml`.

## Interview Questions

### Beginner - 5

1. **Q:** What is the src/ layout and why use it?  
   **A:** All package code goes under src/my_project/, separating it from tests, notebooks, and config. It prevents import confusion and mirrors real-world package installation.

2. **Q:** What does `__init__.py` do?  
   **A:** It marks a directory as a Python package, enabling imports like `from my_project.data import load_data`. Can also define `__all__` for explicit exports.

3. **Q:** What are type hints in Python?  
   **A:** Type hints annotate function parameters and return values, e.g., `def add(x: int, y: int) -> int:`. They improve readability and enable static type checking with mypy.

4. **Q:** What is pytest used for?  
   **A:** pytest is a testing framework for writing and running unit tests. It finds files named `test_*.py` and runs functions named `test_*`.

5. **Q:** What is the purpose of a README.md?  
   **A:** It provides project overview, installation instructions, usage examples, and results. It's the first thing people see on GitHub or PyPI.

### Intermediate - 5

1. **Q:** What is the difference between requirements.txt and pyproject.toml?  
   **A:** requirements.txt lists exact packages for a specific environment (pip freeze). pyproject.toml defines the project metadata, dependencies with version ranges, and tool configurations. Modern projects use pyproject.toml.

2. **Q:** What does `pip install -e .` do and why use it in ML projects?  
   **A:** It installs the package in editable (development) mode. Changes to source code are immediately reflected without reinstalling. Essential for development.

3. **Q:** How do you organize configuration for different environments (dev, staging, prod)?  
   **A:** Use YAML configs with environment-specific overrides: `configs/config.yaml` for defaults, `configs/prod.yaml` for overrides. Load and merge with a config loader.

4. **Q:** What is the role of pre-commit hooks in an ML project?  
   **A:** They automatically check code quality before each commit: trailing whitespace, formatting (black), linting (ruff), type checking (mypy). Prevents bad code from entering the repository.

5. **Q:** How do you version data alongside code in an ML project?  
   **A:** Use DVC (Data Version Control) which integrates with Git. DVC tracks data files in .dvc files (committed to Git) while the actual data lives in remote storage (S3, GCS).

### Advanced - 3

1. **Q:** Design a CI/CD pipeline for an ML project using GitHub Actions.  
   **A:** Trigger on PR: run lint (ruff), typecheck (mypy), test (pytest with coverage). On merge to main: run full test suite, train model, evaluate, and if improved, register in MLflow model registry and deploy to staging.

2. **Q:** How would you structure a monorepo containing multiple ML projects with shared utilities?  
   **A:** Top-level `src/` with `shared/` package (common utils, base models). Each project in `projects/project_a/`, `projects/project_b/` with its own src/ and tests/. Use namespace packages or editable installs. Single tox.ini and pre-commit config at root.

3. **Q:** Explain how to implement reproducible ML pipelines using the src/ layout and a workflow manager (e.g., ZenML, Kubeflow, or Airflow).  
   **A:** Each pipeline step (load, preprocess, train, evaluate) is a Python function in src/ with a decorator (e.g., `@step`). Steps are composed into a DAG. Artifacts are versioned. Configs control step parameters. The src/ layout keeps step logic clean and testable.

## Practice Problems

### Easy - 5

1. **E1:** Create a minimal `pyproject.toml` for an ML project with numpy, pandas, and scikit-learn as dependencies.
2. **E2:** Write a function with type hints that takes a DataFrame and returns a scaled DataFrame.
3. **E3:** Create an `__init__.py` that imports and exposes 3 functions from submodules.
4. **E4:** Write a pytest test for a function that adds two numbers.
5. **E5:** Create a `.gitignore` file for an ML project (data/, models/, __pycache__/, .pytest_cache/).

### Medium - 5

1. **M1:** Organize a small ML project (binary classification on iris) into the src/ layout with tests.
2. **M2:** Add Google-style docstrings and type hints to a train_model function.
3. **M3:** Create a pre-commit config with black, ruff, and trailing-whitespace hooks.
4. **M4:** Write a tox.ini that runs pytest with coverage on Python 3.10 and 3.11.
5. **M5:** Create a config.yaml for a project with data paths, model params, and training settings.

### Hard - 3

1. **H1:** Convert a Jupyter notebook into a structured src/ layout with tests and a CLI entry point.
2. **H2:** Set up a GitHub Actions workflow that runs lint, typecheck, and tests on every PR.
3. **H3:** Implement a full ML pipeline (load -> preprocess -> train -> evaluate -> register) using MLflow tracking, with all code in src/ and configs in YAML.

## Solutions

### E1 Solution
```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "my_project"
version = "0.1.0"
dependencies = ["numpy", "pandas", "scikit-learn"]
requires-python = ">=3.9"
```

### E2 Solution
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df
```

### E3-E5 Solutions follow patterns from examples.

### M1-M5 Solutions extend the examples with full project setup.

### H1-H3 Solutions require complete project implementation.

## Related Concepts

- 093 — sklearn Basics (ML pipeline components)
- 094 — Preprocessing (data transformation modules)
- 095 — Model Evaluation (evaluation in ML projects)
- 096 — PyTorch Tensors (deep learning project structure)

## Next Concepts

- Deployment (FastAPI, Docker, cloud serving)
- MLOps (MLflow, DVC, CI/CD for ML)
- Advanced packaging (namespace packages, C extensions)

## Summary

Well-structured ML projects follow the src/ layout with proper packaging (pyproject.toml), type hints, Google/Numpy-style docstrings, pytest tests, pre-commit hooks, tox for multi-environment testing, and config files. This structure ensures reproducibility, collaboration, and production readiness. The src/ layout, in particular, prevents import confusion and mirrors real-world package usage.

## Key Takeaways

- Use src/ layout: all package code under src/my_project/
- pyproject.toml for project metadata and build configuration
- __init__.py files make directories importable
- Type hints and docstrings improve readability and catch bugs
- pytest for unit tests; test files start with test_
- pre-commit hooks automate code quality checks
- tox tests across multiple Python versions
- Gitignore data/, models/, cache directories
- Keep YAML/JSON configs separate from code
- Use MLflow or similar for experiment tracking
- Always pin dependency versions for reproducibility
