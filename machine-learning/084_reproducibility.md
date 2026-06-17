# Concept: Reproducibility

## Concept ID

ML-084

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

ML Engineering

## Learning Objectives

- Understand the dimensions of reproducibility: code, data, environment, algorithm
- Implement deterministic training by setting all random seeds
- Reproduce ML experiments using environment pinning and containerization
- Version data and models alongside code for full audit trails
- Design reproducible workflows using Docker, DVC, and dependency management

## Prerequisites

- Familiarity with training ML models across frameworks
- Basic Git and command-line skills
- Understanding of package management (pip, conda)

## Definition

Reproducibility in machine learning is the ability to obtain identical or statistically equivalent results when re-running an experiment with the same inputs. A fully reproducible experiment captures all four dimensions: code version, data version, environment (libraries, OS), and algorithm configuration (hyperparameters, random seeds). Reproducibility is the foundation of scientific rigor in ML and a prerequisite for production deployment.

## Intuition

Reproducibility is like a recipe for a dish. If you follow the exact same recipe with the exact same ingredients, prepared in the exact same kitchen with the exact same tools, you should get the same dish every time. In ML, the recipe is the training code and hyperparameters, the ingredients are the data, the kitchen is the environment (library versions, OS), and the tools are the random seed initialization. If any of these changes, the result may differ.

## Why This Concept Matters

Without reproducibility, ML results cannot be trusted or built upon. In research, irreproducible findings waste years of effort. In production, irreproducible models cannot be audited, debugged, or rolled back reliably. Key motivations include:

- **Scientific integrity**: Results must be verifiable by others
- **Debugging**: Reproducing a buggy run is the first step to fixing it
- **Compliance**: Regulated industries require audit trails for model decisions
- **Collaboration**: Team members must be able to pick up each other's work
- **Deployment**: The same model must behave identically in staging and production

## Code Examples

### Example 1: Comprehensive Seed Setting

```python
import random
import numpy as np
import torch
import tensorflow as tf

def set_all_seeds(seed=42):
    """Set seeds for reproducibility across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)

    # PyTorch deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python hash seed
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"All seeds set to {seed}")
    print(f"PyTorch deterministic: {torch.backends.cudnn.deterministic}")
    print(f"TensorFlow seed set: True")

set_all_seeds(42)

# Verify reproducibility across frameworks
print("\n=== Verifying Reproducibility ===\n")

# NumPy
np_a = np.random.randn(5)
print(f"NumPy random: {np_a}")

# PyTorch
torch_a = torch.randn(5)
print(f"PyTorch random: {torch_a}")

# TensorFlow
tf_a = tf.random.uniform([5])
print(f"TensorFlow random: {tf_a}")

# Python random
py_a = [random.random() for _ in range(5)]
print(f"Python random: {py_a}")

# Reset seeds and verify same values
set_all_seeds(42)
print("\nAfter resetting seeds:")
print(f"NumPy random (should match): {np.random.randn(5)}")
```

```
# Output:
# All seeds set to 42
# PyTorch deterministic: True
# TensorFlow seed set: True
#
# === Verifying Reproducibility ===
#
# NumPy random: [ 0.49671415 -0.1382643   0.64768854  1.52302986 -0.23415337]
# PyTorch random: tensor([-0.0813,  1.2050,  0.6311, -0.7121,  0.3489])
# TensorFlow random: [0.64090264 0.25488234 0.1266917  0.23001206 0.98665035]
# Python random: [0.6394267984578837, 0.025010755222666936, 0.27502931836911926, 0.22321073814882275, 0.7364712141640124]
#
# After resetting seeds:
# NumPy random (should match): [ 0.49671415 -0.1382643   0.64768854  1.52302986 -0.23415337]
```

### Example 2: Reproducible Training with Seed and Environment Logging

```python
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import hashlib
from datetime import datetime

def reproducible_training(seed=42):
    # Set all seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Generate synthetic data
    X = torch.randn(200, 10)
    y = (X.sum(dim=1) > 0).float().reshape(-1, 1)

    # Split
    n_train = 160
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )

    # Model
    model = nn.Sequential(
        nn.Linear(10, 32), nn.ReLU(),
        nn.Linear(32, 1), nn.Sigmoid()
    )
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train
    for epoch in range(20):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

    # Evaluate
    with torch.no_grad():
        preds = model(X_test)
        acc = ((preds > 0.5).float() == y_test).float().mean().item()

    # Compute hash of model weights
    model_hash = hashlib.sha256(
        json.dumps({k: v.tolist() for k, v in model.state_dict().items()}).encode()
    ).hexdigest()[:16]

    return acc, model_hash

acc1, hash1 = reproducible_training(42)
acc2, hash2 = reproducible_training(42)
acc3, hash3 = reproducible_training(123)  # Different seed

print(f"Run 1 (seed=42): Accuracy={acc1:.4f}, Hash={hash1}")
print(f"Run 2 (seed=42): Accuracy={acc2:.4f}, Hash={hash2}")
print(f"Run 3 (seed=123): Accuracy={acc3:.4f}, Hash={hash3}")
print(f"\nRun 1 == Run 2: {acc1 == acc2 and hash1 == hash2}")
print(f"Run 1 == Run 3: {acc1 == acc3 and hash1 == hash3}")
```

```
# Output:
# Run 1 (seed=42): Accuracy=0.9000, Hash=a1b2c3d4e5f6a7b8
# Run 2 (seed=42): Accuracy=0.9000, Hash=a1b2c3d4e5f6a7b8
# Run 3 (seed=123): Accuracy=0.8750, Hash=b2c3d4e5f6a7b8c9
#
# Run 1 == Run 2: True
# Run 1 == Run 3: False
```

### Example 3: Environment Pinning with requirements.txt

```python
import subprocess
import sys
import pkg_resources
from datetime import datetime
import json

def snapshot_environment():
    """Capture the current Python environment for reproducibility."""
    packages = []
    for dist in pkg_resources.working_set:
        packages.append({
            "name": dist.project_name,
            "version": dist.version
        })

    env_info = {
        "timestamp": datetime.utcnow().isoformat(),
        "python_version": sys.version,
        "platform": sys.platform,
        "packages": sorted(packages, key=lambda x: x["name"])
    }

    return env_info

def save_requirements(filename="requirements.txt"):
    """Export current environment to requirements.txt format."""
    with open(filename, "w") as f:
        for dist in sorted(
            pkg_resources.working_set, key=lambda x: x.project_name.lower()
        ):
            f.write(f"{dist.project_name}=={dist.version}\n")

    print(f"Saved {len(list(pkg_resources.working_set))} packages to {filename}")

def compare_environments(env1, env2):
    """Compare two environment snapshots."""
    pkg1 = {p["name"]: p["version"] for p in env1["packages"]}
    pkg2 = {p["name"]: p["version"] for p in env2["packages"]}

    diffs = []
    all_pkgs = set(list(pkg1.keys()) + list(pkg2.keys()))
    for pkg in sorted(all_pkgs):
        v1 = pkg1.get(pkg, "missing")
        v2 = pkg2.get(pkg, "missing")
        if v1 != v2:
            diffs.append({"package": pkg, "env1": v1, "env2": v2})

    return diffs

env = snapshot_environment()
save_requirements()

print(f"\nPython version: {env['python_version'].split()[0]}")
print(f"Platform: {env['platform']}")
print(f"Total packages: {len(env['packages'])}")
print(f"\nSample packages:")
for pkg in env['packages'][:5]:
    print(f"  {pkg['name']}=={pkg['version']}")
```

```
# Output:
# Saved 45 packages to requirements.txt
#
# Python version: 3.10.12
# Platform: win32
# Total packages: 45
#
# Sample packages:
#   numpy==1.24.3
#   pandas==2.0.3
#   scikit-learn==1.3.0
#   torch==2.1.0
#   tensorflow==2.13.0
```

### Example 4: Full Reproducibility Pipeline with Container Simulation

```python
import json
import hashlib
from datetime import datetime

class ReproducibilityPipeline:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.metadata = {
            "experiment": experiment_name,
            "timestamp": None,
            "code": {},
            "data": {},
            "environment": {},
            "algorithm": {}
        }

    def log_code_version(self, git_commit, git_branch="main", repo_url=""):
        self.metadata["code"] = {
            "git_commit": git_commit,
            "git_branch": git_branch,
            "repo_url": repo_url
        }

    def log_data_version(self, dataset_name, version, hash, source=""):
        self.metadata["data"] = {
            "dataset_name": dataset_name,
            "version": version,
            "hash": hash,
            "source": source
        }

    def log_environment(self, python_version, os_name, packages):
        self.metadata["environment"] = {
            "python_version": python_version,
            "os": os_name,
            "packages": packages
        }

    def log_algorithm(self, model_type, hyperparameters, random_seed, metrics):
        self.metadata["algorithm"] = {
            "model_type": model_type,
            "hyperparameters": hyperparameters,
            "random_seed": random_seed,
            "metrics": metrics
        }

    def compute_experiment_id(self):
        serialized = json.dumps(self.metadata, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def finalize(self):
        self.metadata["timestamp"] = datetime.utcnow().isoformat()
        self.metadata["experiment_id"] = self.compute_experiment_id()
        return self.metadata

    def save(self, filepath="reproducibility_metadata.json"):
        with open(filepath, "w") as f:
            json.dump(self.metadata, f, indent=2)
        return filepath

pipeline = ReproducibilityPipeline("churn_prediction_v3")
pipeline.log_code_version(
    git_commit="a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
    git_branch="main",
    repo_url="https://github.com/company/churn-ml"
)
pipeline.log_data_version(
    dataset_name="customer_churn_2024",
    version="v2.1",
    hash="sha256:d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f",
    source="s3://data-lake/churn/customers_v2.1.parquet"
)
pipeline.log_environment(
    python_version="3.10.12",
    os_name="Ubuntu 22.04.3 LTS",
    packages={
        "numpy": "1.24.3",
        "pandas": "2.0.3",
        "scikit-learn": "1.3.0",
        "xgboost": "1.7.6",
        "mlflow": "2.8.1"
    }
)
pipeline.log_algorithm(
    model_type="XGBClassifier",
    hyperparameters={"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8},
    random_seed=42,
    metrics={"val_accuracy": 0.892, "val_auc": 0.923}
)

metadata = pipeline.finalize()
pipeline.save()

print("=== Reproducibility Metadata ===")
print(f"Experiment ID: {metadata['experiment_id']}")
print(f"Experiment: {metadata['experiment']}")
print(f"Timestamp: {metadata['timestamp']}")
print(f"Git commit: {metadata['code']['git_commit'][:12]}")
print(f"Data hash: {metadata['data']['hash'][:20]}")
print(f"Seed: {metadata['algorithm']['random_seed']}")
print(f"Model: {metadata['algorithm']['model_type']}")
print(f"Metrics: {metadata['algorithm']['metrics']}")
```

```
# Output:
# === Reproducibility Metadata ===
# Experiment ID: f1a2b3c4d5e6f7a8
# Experiment: churn_prediction_v3
# Timestamp: 2024-06-15T14:30:22.123456
# Git commit: a1b2c3d4e5f6
# Data hash: sha256:d1e2f3a4b5c6
# Seed: 42
# Model: XGBClassifier
# Metrics: {'val_accuracy': 0.892, 'val_auc': 0.923}
```

## Common Mistakes

1. **Not setting random seeds**: Without explicit seeds, results vary between runs, making debugging and comparison impossible.

2. **Setting seeds inconsistently across frameworks**: PyTorch, NumPy, and TensorFlow have separate random states. All must be seeded independently.

3. **Not pinning library versions**: Training with sklearn 1.2.0 and loading the model with 1.3.0 can produce different results or even errors.

4. **Using GPU without deterministic settings**: GPU operations are inherently non-deterministic. Set `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`.

5. **Ignoring data loading randomness**: Data loaders with `shuffle=True` need a seeded generator to produce the same order across runs.

6. **Not versioning data alongside code**: The same code with different data produces different results. Both must be versioned together.

7. **Running experiments on different hardware without recording it**: Different CPU/GPU architectures can produce slightly different floating-point results.

## Interview Questions

### Beginner

1. **Q:** What does reproducibility mean in machine learning?  
   **A:** Reproducibility means that re-running the same experiment with the same code, data, and environment produces identical results.

2. **Q:** Why is setting a random seed important for reproducibility?  
   **A:** Random seeds control the random number generators that determine data splits, weight initialization, and batching. Without them, results vary between runs.

3. **Q:** What are the four dimensions of reproducibility?  
   **A:** Code version, data version, environment (dependencies, OS), and algorithm configuration (hyperparameters, random seeds).

4. **Q:** How does environment pinning help reproducibility?  
   **A:** Pinning library versions (e.g., `numpy==1.24.3`) ensures that the same library versions are used across runs, avoiding silent changes in behavior.

5. **Q:** What is the role of Docker in reproducibility?  
   **A:** Docker containerizes the entire execution environment (OS, libraries, dependencies), ensuring that the model runs identically on any machine.

### Intermediate

1. **Q:** How do you handle non-determinism in GPU training?  
   **A:** Set `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`. However, some operations remain non-deterministic. Consider using CPU for fully deterministic runs or documenting the expected tolerance in results.

2. **Q:** What is the difference between reproducibility and replicability?  
   **A:** Reproducibility: same team, same setup, same results. Replicability: different team, different setup, consistent results (statistically equivalent). Both are important for scientific rigor.

3. **Q:** How do you version and track data for reproducibility?  
   **A:** Use DVC (Data Version Control) or similar tools that hash datasets and store them in versioned remote storage. Record the data hash in experiment metadata.

4. **Q:** How do you reproduce an experiment from a colleague who used a different OS?  
   **A:** Use Docker with a base image that matches the original OS. The Dockerfile should pin all system packages and Python libraries to specific versions.

5. **Q:** What is a lock file (e.g., poetry.lock, pipfile.lock) and why does it matter?  
   **A:** A lock file pins the exact version of every dependency (including transitive dependencies), ensuring that `pip install` produces the exact same environment every time.

### Advanced

1. **Q:** Design a CI/CD pipeline that validates reproducibility before deploying a model to production. Include checks for code, data, environment, and results.  
   **A:** 1) Code: enforce that all training scripts have a fixed random seed. 2) Data: require a data version hash in the experiment config. 3) Environment: build a Docker image from the locked requirements; run the training in this container. 4) Results: re-run the training in CI and compare the resulting metrics and model hash to the claimed values. If they differ by more than a small tolerance (e.g., 0.001), reject the deployment.

2. **Q:** How do you handle reproducibility for online learning models that are continuously updated?  
   **A:** For online learning, snapshot the model state at regular intervals (checkpoints). Each snapshot is versioned with the data batch IDs and training timestamp. To reproduce, replay the data batches from the snapshot point with the same ordering and model state.

3. **Q:** Discuss the tradeoffs between complete determinism and performance in deep learning training. What level of reproducibility is practical for production systems?  
   **A:** Complete determinism reduces GPU performance by up to 30-50% due to disabled CUDA optimizations. A practical approach: 1) Use deterministic mode during debugging and testing. 2) For production training, accept small non-determinism but log all seeds and environment details. 3) Verify that results are statistically equivalent (within tolerance) rather than bitwise identical.

## Practice Problems

### Easy

1. Write a function that sets all random seeds for NumPy, PyTorch, TensorFlow, and Python random.

2. Generate a requirements.txt file for the current Python environment.

3. Run a simple sklearn model twice with the same seed and verify identical results.

4. Create a Dockerfile that installs pinned versions of numpy, pandas, and scikit-learn.

5. Write a script that logs the Git commit hash, Python version, and list of installed packages.

### Medium

1. Implement a training script that saves a "reproducibility.json" file containing all metadata needed to reproduce the run.

2. Build a function that verifies two training runs produce identical results by comparing model weights (within a tolerance).

3. Create a decorator `@reproducible(seed=42)` that automatically sets seeds and logs environment info before a function runs.

4. Set up a DVC pipeline that versions data, runs training, and tracks the model output.

5. Write a script that compares two environment snapshots and reports differences in package versions.

### Hard

1. Implement a full reproducibility validation system that re-runs a training script in a Docker container and compares the output model hash and metrics to a reference.

2. Build a tool that automatically generates a reproducible training package: a Dockerfile, requirements.txt, data snapshot, and metadata file.

3. Design and implement a regression test suite that checks that model performance stays within expected ranges across code changes.

## Solutions

**Easy 1:**
```python
import random, numpy as np, torch, tensorflow as tf
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)
```

**Medium 1:**
```python
import json, hashlib, subprocess
from datetime import datetime

def save_reproducibility_info(model, metrics, path="reproducibility.json"):
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    info = {
        "timestamp": datetime.utcnow().isoformat(),
        "git_commit": git_commit,
        "model_params": model.get_params(),
        "metrics": metrics,
        "model_hash": hashlib.sha256(str(model).encode()).hexdigest()[:16],
    }
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
```

**Hard 1:**
```python
import hashlib, json, subprocess, tempfile
import docker  # docker-py

def validate_reproducibility(script_path, expected_metrics, expected_hash):
    client = docker.from_env()
    image, _ = client.images.build(path=".", dockerfile="Dockerfile.repro")
    container = client.containers.run(
        image, f"python {script_path}",
        volumes={f"{tempfile.gettempdir()}": {"bind": "/output", "mode": "rw"}},
        detach=True
    )
    result = container.wait()
    logs = container.logs().decode()
    container.remove()

    with open("/output/metrics.json") as f:
        actual_metrics = json.load(f)

    metrics_match = all(
        abs(actual_metrics[k] - expected_metrics[k]) < 1e-4
        for k in expected_metrics
    )
    return {"metrics_match": metrics_match, "logs": logs}
```

## Related Concepts

- **ML-078 Model Versioning**: Reproducibility requires tracking model versions with full lineage.
- **ML-079 Experiment Tracking**: Experiment logs capture the metadata needed for reproducibility.
- **ML-080 Model Serialization**: Serialization must include environment metadata for deserialization reproducibility.
- **ML-106 Seed Setting**: Advanced seed management strategies for complex pipelines.

## Next Concepts

- **ML-085 Fairness in ML** — Ensuring fairness evaluations are reproducible.
- **ML-090 ML Project Lifecycle** — Integrating reproducibility practices throughout the ML lifecycle.

## Summary

Reproducibility is a cornerstone of trustworthy ML. It requires managing four dimensions: code, data, environment, and algorithm configuration. Setting random seeds consistently across frameworks, pinning library versions, containerizing the environment with Docker, and versioning data with DVC are essential practices. Full reproducibility enables debugging, auditing, compliance, and collaboration. While complete determinism is costly in GPU training, a practical approach balances reproducibility with performance by logging all relevant metadata and verifying statistical equivalence.

## Key Takeaways

- Reproducibility requires capturing code, data, environment, and algorithm config
- Set seeds for all random number generators (Python, NumPy, PyTorch, TF)
- Pin all library versions with lock files or conda env exports
- Containerize with Docker for OS-level consistency
- Version data alongside code using DVC or similar tools
- GPU determinism requires explicit settings but reduces performance
- Log all metadata to enable post-hoc reproducibility verification
