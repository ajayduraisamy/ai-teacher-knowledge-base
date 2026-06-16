# Concept: Unit Testing

## Concept ID

PYT-050

## Difficulty

Intermediate

## Domain

Python

## Module

Intermediate Python

## Learning Objectives

- Understand the purpose and benefits of unit testing
- Write tests using `unittest.TestCase`
- Use pytest fixtures, parametrize, and assertions
- Mock external dependencies with `unittest.mock` and `patch`
- Organize tests with discovery and test suites
- Measure code coverage with `coverage.py`
- Apply Test-Driven Development (TDD) principles
- Test AI/ML pipelines and data validation

## Prerequisites

- Basic Python functions and classes
- Understanding of modules and imports
- Familiarity with the `assert` statement
- Basic knowledge of exceptions

## Definition

**Unit testing** is a software testing method where individual units of source code (functions, methods, classes) are tested to determine whether they are fit for use. The `unittest` module (built-in) and `pytest` (third-party) are the primary testing frameworks in Python. Unit tests provide automated verification that code behaves correctly, enabling confident refactoring and regression prevention.

## Intuition

Think of unit tests as a safety net for your code. When you change something, you run your tests to make sure nothing broke. Tests are executable documentation — they show exactly how your code is supposed to work. Without tests, every change is risky; with tests, you can refactor fearlessly.

## Why This Concept Matters

Unit testing is a cornerstone of professional software development. It catches bugs early, documents expected behavior, enables refactoring, and serves as a regression safety net. In data science and AI/ML, testing is equally critical — data pipelines, preprocessing steps, and model evaluation code all benefit from automated verification. Tests prevent silent data corruption, incorrect metrics, and deployment failures.

## Real World Examples

- Testing API endpoints return correct status codes and structures
- Validating data preprocessing functions handle edge cases
- Testing database queries return expected results
- Verifying configuration parsing and validation
- Testing mathematical computations and transformations
- Ensuring error handling behaves correctly under failure conditions

## AI/ML Relevance

Unit testing in AI/ML:
- Testing data preprocessing and cleaning functions
- Validating feature extraction and transformation logic
- Testing model evaluation metrics implementations
- Ensuring data split integrity (no leakage between train/test)
- Testing data validation rules and schema checks
- Verifying model serialization/deserialization round trips
- Testing pipeline stage outputs match expected shapes and types

## Code Examples

### Example 1: Basic unittest.TestCase

```python
import unittest

def add(a, b):
    return a + b

def divide(a, b):
    if b == 0:
        raise ValueError('Cannot divide by zero')
    return a / b

class TestMathFunctions(unittest.TestCase):
    def test_add_positive(self):
        self.assertEqual(add(2, 3), 5)

    def test_add_negative(self):
        self.assertEqual(add(-1, 1), 0)

    def test_add_zero(self):
        self.assertEqual(add(0, 0), 0)

    def test_divide_normal(self):
        self.assertEqual(divide(10, 2), 5.0)

    def test_divide_by_zero(self):
        with self.assertRaises(ValueError):
            divide(10, 0)

if __name__ == '__main__':
    unittest.main(verbosity=2)

# Output:
# test_add_negative (__main__.TestMathFunctions) ... ok
# test_add_positive (__main__.TestMathFunctions) ... ok
# test_add_zero (__main__.TestMathFunctions) ... ok
# test_divide_by_zero (__main__.TestMathFunctions) ... ok
# test_divide_normal (__main__.TestMathFunctions) ... ok
#
# ----------------------------------------------------------------------
# Ran 5 tests in 0.001s
#
# OK
```

### Example 2: pytest syntax

```python
# Save as test_functions.py and run with: pytest test_functions.py -v

import pytest

def multiply(a, b):
    return a * b

def is_even(n):
    return n % 2 == 0

class TestMultiplication:
    def test_basic(self):
        assert multiply(3, 4) == 12

    def test_zero(self):
        assert multiply(5, 0) == 0

    def test_negative(self):
        assert multiply(-2, 3) == -6

class TestEven:
    def test_even_number(self):
        assert is_even(4) == True

    def test_odd_number(self):
        assert is_even(7) == False

    def test_zero(self):
        assert is_even(0) == True

# Run with: pytest test_functions.py -v
# Output:
# ============================= test session starts ==============================
# collected 6 items
#
# test_functions.py::TestMultiplication::test_basic PASSED
# test_functions.py::TestMultiplication::test_zero PASSED
# test_functions.py::TestMultiplication::test_negative PASSED
# test_functions.py::TestEven::test_even_number PASSED
# test_functions.py::TestEven::test_odd_number PASSED
# test_functions.py::TestEven::test_zero PASSED
#
# ============================== 6 passed in 0.02s ===============================
```

### Example 3: pytest fixtures

```python
import pytest
import tempfile
import os

class DataProcessor:
    def __init__(self, data):
        self.data = data

    def clean(self):
        return [x for x in self.data if x is not None]

    def save(self, path):
        with open(path, 'w') as f:
            for item in self.clean():
                f.write(f'{item}\n')

@pytest.fixture
def sample_data():
    return [1, None, 2, None, 3]

@pytest.fixture
def temp_file():
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)

class TestDataProcessor:
    def test_clean_removes_none(self, sample_data):
        processor = DataProcessor(sample_data)
        result = processor.clean()
        assert None not in result
        assert len(result) == 3

    def test_clean_preserves_order(self, sample_data):
        processor = DataProcessor(sample_data)
        result = processor.clean()
        assert result == [1, 2, 3]

    def test_save_to_file(self, sample_data, temp_file):
        processor = DataProcessor(sample_data)
        processor.save(temp_file)
        with open(temp_file, 'r') as f:
            lines = f.read().strip().split('\n')
        assert lines == ['1', '2', '3']

# Run with: pytest test_processor.py -v
```

### Example 4: pytest parametrize

```python
import pytest

def fibonacci(n):
    if n < 0:
        raise ValueError('n must be non-negative')
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

@pytest.mark.parametrize('n, expected', [
    (0, 0),
    (1, 1),
    (2, 1),
    (3, 2),
    (4, 3),
    (5, 5),
    (10, 55),
    (20, 6765),
])
def test_fibonacci(n, expected):
    assert fibonacci(n) == expected

@pytest.mark.parametrize('n', [-1, -5, -10])
def test_fibonacci_negative(n):
    with pytest.raises(ValueError):
        fibonacci(n)

# Run with: pytest test_fib.py -v
# Output:
# ============================= test session starts ==============================
# collected 11 items
#
# test_fib.py::test_fibonacci[0-0] PASSED
# test_fib.py::test_fibonacci[1-1] PASSED
# test_fib.py::test_fibonacci[2-1] PASSED
# test_fib.py::test_fibonacci[3-2] PASSED
# test_fib.py::test_fibonacci[4-3] PASSED
# test_fib.py::test_fibonacci[5-5] PASSED
# test_fib.py::test_fibonacci[10-55] PASSED
# test_fib.py::test_fibonacci[20-6765] PASSED
# test_fib.py::test_fibonacci_negative[-1] PASSED
# test_fib.py::test_fibonacci_negative[-5] PASSED
# test_fib.py::test_fibonacci_negative[-10] PASSED
#
# ============================== 11 passed in 0.03s ===============================
```

### Example 5: Mocking with unittest.mock

```python
import pytest
from unittest.mock import Mock, patch
import requests

class WeatherService:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://api.weather.com'

    def get_temperature(self, city):
        url = f'{self.base_url}/v1/current?city={city}&key={self.api_key}'
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()['temp']
        return None

def test_get_temperature_success():
    # Mock the requests.get call
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'temp': 22.5}

    with patch('requests.get', return_value=mock_response):
        service = WeatherService('fake-key')
        temp = service.get_temperature('London')

    assert temp == 22.5

def test_get_temperature_failure():
    mock_response = Mock()
    mock_response.status_code = 404

    with patch('requests.get', return_value=mock_response):
        service = WeatherService('fake-key')
        temp = service.get_temperature('Nowhere')

    assert temp is None

# Note: These tests do not make actual HTTP requests
# Run with: pytest -v
```

### Example 6: Testing AI/ML data preprocessing

```python
import pytest
import numpy as np

class DataPreprocessor:
    def __init__(self, scaler_params=None):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        return self

    def transform(self, data):
        if self.mean is None or self.std is None:
            raise ValueError('Preprocessor not fitted')
        return (data - self.mean) / self.std

    def remove_outliers(self, data, threshold=3):
        z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
        mask = np.all(z_scores < threshold, axis=1)
        return data[mask]

def test_preprocessor_fit_transform():
    data = np.array([[1, 2], [3, 4], [5, 6]])
    preprocessor = DataPreprocessor()
    preprocessor.fit(data)
    transformed = preprocessor.transform(data)
    assert np.allclose(np.mean(transformed, axis=0), [0, 0], atol=1e-10)
    assert np.allclose(np.std(transformed, axis=0), [1, 1], atol=1e-10)

def test_preprocessor_not_fitted():
    preprocessor = DataPreprocessor()
    with pytest.raises(ValueError, match='not fitted'):
        preprocessor.transform(np.array([[1, 2]]))

def test_remove_outliers():
    data = np.array([[1, 2], [2, 3], [3, 4], [100, 200]])
    preprocessor = DataPreprocessor()
    cleaned = preprocessor.remove_outliers(data, threshold=2)
    assert len(cleaned) == 3
    assert 100 not in cleaned[:, 0]

def test_preprocessor_invalid_input():
    preprocessor = DataPreprocessor()
    with pytest.raises(Exception):
        preprocessor.fit('invalid')

# Run with: pytest -v
```

### Example 7: Testing with temporary files

```python
import pytest
import tempfile
import os
import json

class ConfigManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = {}

    def load(self):
        if not os.path.exists(self.config_path):
            return {}
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        return self.config

    def save(self, config):
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        self.config = config

    def get(self, key, default=None):
        return self.config.get(key, default)

def test_config_load_save(tmp_path):
    config_file = tmp_path / 'config.json'
    manager = ConfigManager(str(config_file))

    config = {'learning_rate': 0.001, 'epochs': 10}
    manager.save(config)

    loaded = manager.load()
    assert loaded == config

def test_config_missing_file(tmp_path):
    config_file = tmp_path / 'nonexistent.json'
    manager = ConfigManager(str(config_file))
    assert manager.load() == {}

def test_config_default_value(tmp_path):
    config_file = tmp_path / 'config.json'
    manager = ConfigManager(str(config_file))
    assert manager.get('missing', 'default') == 'default'

# Run with: pytest -v
```

### Example 8: Test-Driven Development (TDD) workflow

```python
import pytest

# Step 1: Write the test first (RED)
def test_calculate_mean():
    from stats import calculate_mean
    assert calculate_mean([1, 2, 3, 4, 5]) == 3.0
    assert calculate_mean([10]) == 10.0
    assert calculate_mean([]) == 0.0

def test_calculate_mean_with_floats():
    from stats import calculate_mean
    result = calculate_mean([1.5, 2.5, 3.0])
    assert result == pytest.approx(2.333, rel=1e-3)

# Step 2: Implement the function (GREEN)
# Save as stats.py:
# def calculate_mean(numbers):
#     if not numbers:
#         return 0.0
#     return sum(numbers) / len(numbers)

# Step 3: Refactor as needed
# Now we add more tests...

def test_calculate_median():
    from stats import calculate_median
    assert calculate_median([1, 3, 5]) == 3
    assert calculate_median([1, 2, 3, 4]) == 2.5

# Run with: pytest -v
```

## Common Mistakes

1. Testing implementation details instead of behavior — tests break when refactoring
2. Writing tests that are too large or test too many things at once
3. Forgetting to test edge cases (empty input, None, boundary values)
4. Using global state that leaks between tests (use fixtures for clean state)
5. Making HTTP/database calls in unit tests — use mocks to isolate the unit
6. Testing only the "happy path" without error and edge cases
7. Not running tests as part of CI/CD pipeline

## Interview Questions

### Beginner - 5

1. What is unit testing and why is it important?
2. What is the difference between `unittest` and `pytest`?
3. How do you assert that a function raises an exception?
4. What is a test fixture?
5. How do you run all tests in a directory?

### Intermediate - 5

1. How does pytest parametrize work and when would you use it?
2. What is mocking and how does `unittest.mock.patch` work?
3. How do you use pytest fixtures for setup and teardown?
4. What is test coverage and how do you measure it?
5. How do you organize tests in a larger project?

### Advanced - 3

1. How would you test asynchronous code in Python?
2. How do you implement test fixtures that are scoped (module, session)?
3. How would you test an AI/ML pipeline that involves random number generation and non-deterministic behavior?

## Practice Problems

### Easy - 5

1. Write a test for a function that returns the square of a number.
2. Test that a function raises TypeError for invalid input types.
3. Write a test using pytest parametrize for a palindrome checker.
4. Test a function that converts Celsius to Fahrenheit.
5. Write a test for a function that returns the length of a list.

### Medium - 5

1. Write tests for a stack data structure (push, pop, peek, is_empty, size).
2. Mock an external API call and test a service function that uses it.
3. Write parametrized tests for a function that validates email addresses.
4. Test a data preprocessing function that handles missing values.
5. Write tests using temporary files for a CSV reader/writer.

### Hard - 3

1. Write a test suite for a machine learning model's predict method using mocked model weights.
2. Implement property-based tests using Hypothesis for a data transformation function.
3. Design a complete testing strategy for an ML training pipeline including data validation, preprocessing, training, and evaluation stages.

## Solutions

### Easy 1

```python
def square(n):
    return n * n

import pytest

def test_square_positive():
    assert square(3) == 9

def test_square_negative():
    assert square(-4) == 16

def test_square_zero():
    assert square(0) == 0

@pytest.mark.parametrize('n, expected', [(1, 1), (2, 4), (5, 25), (10, 100)])
def test_square_parametrized(n, expected):
    assert square(n) == expected

# Run with: pytest -v
```

### Medium 1

```python
import pytest
from unittest.mock import Mock, patch

class Stack:
    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError('pop from empty stack')
        return self._items.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError('peek from empty stack')
        return self._items[-1]

    def is_empty(self):
        return len(self._items) == 0

    def size(self):
        return len(self._items)

class TestStack:
    def test_push_and_pop(self):
        stack = Stack()
        stack.push(1)
        stack.push(2)
        assert stack.pop() == 2
        assert stack.pop() == 1

    def test_is_empty_new(self):
        assert Stack().is_empty() == True

    def test_is_empty_after_push(self):
        stack = Stack()
        stack.push(1)
        assert stack.is_empty() == False

    def test_pop_empty(self):
        with pytest.raises(IndexError):
            Stack().pop()

    def test_peek(self):
        stack = Stack()
        stack.push(42)
        assert stack.peek() == 42
        assert stack.size() == 1

    def test_size(self):
        stack = Stack()
        stack.push(1)
        stack.push(2)
        stack.push(3)
        assert stack.size() == 3
        stack.pop()
        assert stack.size() == 2

# Run with: pytest -v
```

### Hard 1

```python
import pytest
import numpy as np
from unittest.mock import Mock

class SimpleModel:
    def __init__(self, weights=None):
        self.weights = weights if weights is not None else np.array([1.0, 1.0])
        self.bias = 0.0

    def predict(self, features):
        return np.dot(features, self.weights) + self.bias

    def predict_proba(self, features):
        raw = self.predict(features)
        return 1 / (1 + np.exp(-raw))

def test_model_predict_shape():
    model = SimpleModel()
    features = np.array([[1.0, 2.0], [3.0, 4.0]])
    predictions = model.predict(features)
    assert predictions.shape == (2,)

def test_model_predict_with_known_weights():
    model = SimpleModel(weights=np.array([2.0, 3.0]))
    features = np.array([[1.0, 0.0]])
    assert model.predict(features) == pytest.approx(2.0)

def test_model_predict_proba():
    model = SimpleModel(weights=np.array([0.0, 0.0]))
    features = np.array([[1.0, 2.0]])
    proba = model.predict_proba(features)
    assert 0 <= proba[0] <= 1
    assert proba[0] == pytest.approx(0.5)

def test_model_predict_zero_features():
    model = SimpleModel()
    features = np.array([[0.0, 0.0]])
    assert model.predict(features) == pytest.approx(0.0)

# Run with: pytest -v
```

## Related Concepts

- Test-Driven Development (TDD)
- Behavior-Driven Development (BDD)
- Integration testing vs unit testing
- Test coverage analysis
- Property-based testing (Hypothesis)
- Mocking and stubbing patterns
- CI/CD pipeline integration

## Next Concepts

- Advanced testing patterns
- Integration and end-to-end testing
- Performance testing
- Continuous integration with GitHub Actions

## Summary

Unit testing is an essential practice for producing reliable, maintainable software. Python provides both `unittest` (built-in) and `pytest` (third-party) frameworks, with `pytest` being the more popular choice for its simplicity, powerful fixtures, parametrization, and plugin ecosystem. Mocking enables isolated testing of components with external dependencies. In AI/ML workflows, testing ensures data integrity, correct preprocessing, and reliable model evaluation.

## Key Takeaways

- Write tests before or alongside code to catch bugs early
- Use pytest for simpler syntax, fixtures, and parametrized tests
- Mock external dependencies to isolate the unit under test
- Test edge cases, error conditions, and normal behavior
- Measure coverage to identify untested code paths
- Organize tests mirroring your source code structure
- AI/ML: test data preprocessing, feature extraction, metrics, and pipeline integrity
- Run tests automatically in CI/CD to prevent regressions
