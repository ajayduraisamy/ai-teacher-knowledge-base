# Concept: Model Versioning

## Concept ID

ML-078

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

ML Engineering

## Learning Objectives

- Understand the importance of model versioning in production ML
- Use DVC (Data Version Control) for versioning datasets and models
- Navigate the MLflow Model Registry for managing model lifecycle
- Track model lineage linking code, data, and hyperparameters
- Manage model stages: staging, production, archived

## Prerequisites

- Basic understanding of Git for version control
- Familiarity with ML training workflows
- Knowledge of model serialization formats

## Definition

Model versioning is the practice of tracking and managing different versions of machine learning models along with their associated metadata, including training code, hyperparameters, datasets, evaluation metrics, and artifacts. It extends standard code versioning (Git) to handle large binary files, data snapshots, and model lineage, enabling reproducibility, auditability, and controlled rollouts.

## Intuition

Think of model versioning like software versioning but for ML artifacts. Just as you would never deploy untracked code changes to production, you should never deploy untracked model changes. A model version captures the complete snapshot of everything that produced a model: the exact data, the code, the environment, and the training configuration. This means you can always reproduce any model, compare versions side-by-side, and roll back to a previous version if a deployment goes wrong.

## Why This Concept Matters

Without model versioning, organizations face several critical problems: they cannot reproduce past results, they cannot audit which model is in production, they cannot roll back safely, and they cannot compare model performance systematically. As ML models become regulated (finance, healthcare, auto), model versioning becomes a compliance requirement. A robust versioning strategy enables:

- **Reproducibility**: Exact recreation of any historical model
- **Auditability**: Full lineage from raw data to deployed model
- **Controlled rollouts**: Staged deployments with automatic rollback
- **Collaboration**: Multiple data scientists work on model improvements concurrently
- **Model governance**: Track which models are in staging, production, or archived

## Code Examples

### Example 1: DVC for Data and Model Versioning

```python
import os
import sys
import hashlib
import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Simulate DVC commands via Python subprocess
# In practice, these are shell commands, but shown here for illustration

print("=== DVC Model Versioning Demo ===\n")

# Step 1: Initialize DVC (one-time)
# dvc init

# Step 2: Track data directory
# dvc add data/

# Step 3: Create a model
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Save model and metrics
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model_v1.joblib')

metrics = {'accuracy': round(acc, 4), 'n_estimators': 100}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

# Compute model hash
with open('models/model_v1.joblib', 'rb') as f:
    model_hash = hashlib.sha256(f.read()).hexdigest()

# Step 4: Track model with DVC
# dvc add models/model_v1.joblib

print(f"Model saved: models/model_v1.joblib")
print(f"Model hash (SHA256): {model_hash[:16]}...")
print(f"Accuracy: {acc:.4f}")
print(f"Metrics saved: {metrics}")
print(f"\nNext steps:")
print(f"  git add .")
print(f"  git commit -m 'Model v1: accuracy {acc:.4f}'")
print(f"  dvc push")
```

```
# Output:
# === DVC Model Versioning Demo ===
#
# Model saved: models/model_v1.joblib
# Model hash (SHA256): a1b2c3d4e5f6a7b8...
# Accuracy: 1.0000
# Metrics saved: {'accuracy': 1.0, 'n_estimators': 100}
#
# Next steps:
#   git add .
#   git commit -m 'Model v1: accuracy 1.0000'
#   dvc push
```

### Example 2: MLflow Model Registry

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("wine-quality-model")

# Load data
data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="gbm_v1") as run:
    # Log hyperparameters
    params = {
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
    }
    mlflow.log_params(params)

    # Train model
    model = GradientBoostingClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    # Log metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_weighted", f1)

    # Log model
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="WineQualityClassifier"
    )

    run_id = run.info.run_id

print(f"Run ID: {run_id}")
print(f"Logged parameters: {params}")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Weighted: {f1:.4f}")

# Now transition the model to different stages
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get the latest version
model_versions = client.get_latest_versions("WineQualityClassifier")
latest_version = model_versions[0].version

# Transition to staging
client.transition_model_version_stage(
    name="WineQualityClassifier",
    version=latest_version,
    stage="Staging"
)

print(f"\nModel 'WineQualityClassifier' version {latest_version} transitioned to Staging")
print(f"Current stages for WineQualityClassifier:")
for mv in client.get_latest_versions("WineQualityClassifier"):
    print(f"  Version {mv.version}: {mv.current_stage}")
```

```
# Output:
# Run ID: 7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d
# Logged parameters: {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.1, 'subsample': 0.8}
# Accuracy: 0.9722
# F1 Weighted: 0.9721
#
# Model 'WineQualityClassifier' version 1 transitioned to Staging
# Current stages for WineQualityClassifier:
#   Version 1: Staging
```

### Example 3: Model Lineage Tracking

```python
import mlflow
import json
import hashlib
from datetime import datetime

# Simulate tracking metadata for model lineage
lineage = {
    "model_name": "customer_churn_v3",
    "model_version": "3.2.1",
    "timestamp": datetime.utcnow().isoformat(),
    "git_commit": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
    "dvc_data_hash": "d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f",
    "code_hash": hashlib.sha256(
        open("train_churn.py", "rb").read()
    ).hexdigest() if hasattr(open, '__call__') else "mock_hash",
    "dataset": "churn_data_v2024_01_15",
    "features": [
        "avg_call_duration_7d",
        "num_complaints_30d",
        "contract_length_days",
        "monthly_charges",
        "tenure_months"
    ],
    "hyperparameters": {
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 200,
        "subsample": 0.75
    },
    "metrics": {
        "val_auc": 0.892,
        "val_precision": 0.845,
        "val_recall": 0.812,
        "val_f1": 0.828
    },
    "training_env": {
        "python_version": "3.10.12",
        "mlflow_version": "2.8.1",
        "sklearn_version": "1.3.2",
        "os": "Ubuntu 22.04"
    },
    "artifact_uri": "s3://mlflow-artifacts/3/c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7/artifacts/model"
}

print("=== Model Lineage ===")
print(f"Model: {lineage['model_name']} v{lineage['model_version']}")
print(f"Trained: {lineage['timestamp']}")
print(f"Git Commit: {lineage['git_commit'][:12]}")
print(f"Data Version: {lineage['dvc_data_hash'][:12]}")
print(f"Dataset: {lineage['dataset']}")
print(f"Validation AUC: {lineage['metrics']['val_auc']:.4f}")
print(f"Artifact URI: {lineage['artifact_uri']}")
```

```
# Output:
# === Model Lineage ===
# Model: customer_churn_v3 v3.2.1
# Trained: 2024-06-15T14:30:22.123456
# Git Commit: a1b2c3d4e5f6
# Data Version: d1e2f3a4b5c6
# Dataset: churn_data_v2024_01_15
# Validation AUC: 0.8920
# Artifact URI: s3://mlflow-artifacts/3/c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7/artifacts/model
```

### Example 4: Model Version Comparison

```python
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

client = MlflowClient()
model_name = "WineQualityClassifier"

# Get all versions
versions = client.search_model_versions(f"name='{model_name}'")
version_data = []

for v in versions:
    run = client.get_run(v.run_id)
    metrics = run.data.metrics
    params = run.data.params

    version_data.append({
        "version": v.version,
        "stage": v.current_stage,
        "accuracy": metrics.get("accuracy", "N/A"),
        "f1_weighted": metrics.get("f1_weighted", "N/A"),
        "n_estimators": params.get("n_estimators", "N/A"),
        "max_depth": params.get("max_depth", "N/A"),
        "status": v.status,
        "run_id": v.run_id[:8]
    })

df = pd.DataFrame(version_data)
print("=== Model Version Comparison ===")
print(df.to_string(index=False))
```

```
# Output:
# === Model Version Comparison ===
#  version   stage  accuracy f1_weighted n_estimators max_depth   status    run_id
#       1  Staging    0.9722      0.9721          150         4    READY  a1b2c3d4
#       2   None       0.9583      0.9581          100         3    READY  e5f6a7b8
#       3   None       0.9861      0.9860          200         5    READY  c9d0e1f2
```

## Common Mistakes

1. **Versioning only the model file without metadata**: Storing just the `.pkl` file without code version, data version, hyperparameters, or metrics makes the model unreproducible.

2. **Using Git LFS without DVC for large model files**: Git LFS stores files but does not provide dataset versioning, pipeline tracking, or easy comparison between model versions.

3. **Forgetting to tag model versions with metrics**: Without tagging, you cannot easily query which version had the best performance for a specific metric.

4. **Overwriting model artifacts in place**: Saving `model.pkl` to the same path without versioning means you lose the previous model state.

5. **Not archiving old model versions**: Keeping every version in the active registry leads to clutter. Models that are no longer in use should be moved to an "Archived" stage.

6. **Mismatched environment dependencies**: A model trained with one library version may fail to load with another. Always record the exact environment.

7. **Inconsistent naming conventions for model versions**: Without a consistent versioning scheme (e.g., semantic versioning), it becomes difficult to understand the relationship between versions.

## Interview Questions

### Beginner

1. **Q:** Why is Git alone insufficient for model versioning?  
   **A:** Git is designed for text files. ML models are large binary files, and Git does not efficiently handle them. Also, Git does not track data versions or model lineage (code + data + hyperparameters).

2. **Q:** What is DVC and how does it relate to Git?  
   **A:** DVC (Data Version Control) is a tool that extends Git to handle large files and datasets. It stores pointers in Git while the actual data lives in remote storage (S3, GCS, etc.).

3. **Q:** What is the MLflow Model Registry?  
   **A:** It is a centralized model store that manages model versions, stages (Staging, Production, Archived), and annotations, integrated with MLflow Tracking.

4. **Q:** What does model lineage mean?  
   **A:** Model lineage is the complete trace of what code, data, hyperparameters, and environment produced a specific model version.

5. **Q:** What are the typical model stages in a registry?  
   **A:** Staging (for testing), Production (in use), Archived (retired), and None (unregistered).

### Intermediate

1. **Q:** How do you handle model versioning for ensemble or multi-model systems?  
   **A:** Each component model is versioned independently. The ensemble configuration (which model versions, how they combine) is stored as a separate artifact with its own version.

2. **Q:** Explain how DVC pipelines track data transformations end-to-end.  
   **A:** DVC pipelines define stages where each stage has inputs, outputs, and a command. DVC tracks dependencies and re-runs only stages whose inputs have changed, using content-addressed storage for reproducibility.

3. **Q:** How do you set up automated model promotion from staging to production?  
   **A:** Define validation gates (performance thresholds, data drift checks, bias tests). If the candidate model passes all gates and scores better than the current production model, automatically transition it via the registry API.

4. **Q:** What is model registry-based deployment?  
   **A:** Deployment systems watch the registry for version stage changes. When a model transitions to "Production," a CI/CD pipeline automatically deploys it to the serving infrastructure.

5. **Q:** How do you ensure that a model version can be reproduced exactly?  
   **A:** Pin the data version (DVC hash), code version (Git commit), hyperparameters, and environment (conda/Docker). Use `mlflow.pyfunc` models that bundle the environment.

### Advanced

1. **Q:** Design a multi-region model versioning strategy for a global platform with regulatory requirements in Europe, North America, and Asia. Consider data sovereignty and model compliance.  
   **A:** Each region has its own model registry configured with the region's compliance rules. Global model versions are promoted through a central registry, then replicated to regional registries only if compliant with local regulations. Data lineage must include region of origin tags. Model explainability reports are versioned alongside models.

2. **Q:** How do you manage version conflicts when two data scientists train models from the same feature store but with different point-in-time queries?  
   **A:** The feature store version (or the timestamp range) must be part of the model lineage. Each model version records the exact feature view hash and the query time range used. The registry should warn if two models share the same version name but different feature configurations.

3. **Q:** Describe a zero-downtime model version update strategy for a high-throughput real-time inference system.  
   **A:** Use canary or blue-green deployment: deploy the new model version to a subset of traffic, monitor performance and drift, then gradually shift traffic. The registry stores deployment metadata (traffic %, start time, rollback condition). If a rollback is needed, the previous version is still in the registry and served from the same infrastructure.

## Practice Problems

### Easy

1. Initialize a DVC project and track a CSV dataset.

2. Log a simple sklearn model to MLflow with a single metric (accuracy).

3. Register a model in the MLflow Model Registry and transition it to Staging.

4. Use `joblib.dump` and save a model with a versioned filename (e.g., `model_v2.joblib`).

5. Query the MLflow registry to list all versions of a registered model.

### Medium

1. Set up a DVC pipeline with three stages: data preparation, feature engineering, and training.

2. Write a script that compares two model versions from the MLflow registry and prints the performance differences.

3. Implement automated model archiving: move any model that has been in Staging for more than 30 days to Archived.

4. Create a model lineage JSON document for a given experiment run, capturing all required metadata.

5. Set up a GitHub Action that validates model registry promotions against minimum accuracy thresholds.

### Hard

1. Build a custom model versioning system using a database (SQLite) and S3-compatible storage with full lineage tracking.

2. Implement a multi-model A/B testing framework that uses the model registry to route traffic between versions.

3. Design and implement a rollback mechanism for a production model that automatically reverts to the previous version if a drift metric exceeds a threshold.

## Solutions

**Easy 1:**
```bash
# Shell commands (shown for reference)
# dvc init
# dvc add data/dataset.csv
# git add data/dataset.csv.dvc .dvc/config
# git commit -m "Add dataset"
# dvc remote add -d myremote s3://mybucket/dvcstore
# dvc push
```

**Medium 1:**
```python
# DVC pipeline defined in dvc.yaml
# stages:
#   prepare:
#     cmd: python prepare.py data/raw data/processed
#     deps:
#       - data/raw
#       - prepare.py
#     outs:
#       - data/processed
#   featurize:
#     cmd: python featurize.py data/processed data/features
#     deps:
#       - data/processed
#       - featurize.py
#     outs:
#       - data/features
#   train:
#     cmd: python train.py data/features models/model.joblib
#     deps:
#       - data/features
#       - train.py
#     outs:
#       - models/model.joblib
#     metrics:
#       - metrics.json
```

**Hard 1:**
```python
import sqlite3
import boto3
import json
import hashlib
from datetime import datetime

class ModelRegistry:
    def __init__(self, db_path, bucket):
        self.conn = sqlite3.connect(db_path)
        self.s3 = boto3.client('s3')
        self.bucket = bucket
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                version TEXT,
                git_commit TEXT,
                data_hash TEXT,
                params TEXT,
                metrics TEXT,
                artifact_path TEXT,
                created_at TEXT,
                stage TEXT DEFAULT 'None'
            )
        """)
        self.conn.commit()

    def register(self, model_name, version, artifact_path, git_commit, data_hash, params, metrics):
        artifact_s3_key = f"{model_name}/{version}/model.pkl"
        self.s3.upload_file(artifact_path, self.bucket, artifact_s3_key)
        self.conn.execute("""
            INSERT INTO model_versions
            (model_name, version, git_commit, data_hash, params, metrics, artifact_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (model_name, version, git_commit, data_hash,
              json.dumps(params), json.dumps(metrics),
              f"s3://{self.bucket}/{artifact_s3_key}", datetime.utcnow().isoformat()))
        self.conn.commit()
```

## Related Concepts

- **ML-077 Feature Stores**: Feature versions in a feature store should align with model versions for full reproducibility.
- **ML-079 Experiment Tracking**: Experiment tracking logs runs, while model versioning manages the resulting artifacts through their lifecycle.
- **ML-080 Model Serialization**: Serialization formats (joblib, ONNX, SavedModel) are the underlying representation stored in model registries.
- **ML-084 Reproducibility**: Model versioning is a prerequisite for reproducibility, linking artifacts to the environment that produced them.

## Next Concepts

- **ML-080 Model Serialization** — How models are serialized and deserialized for storage and deployment.
- **ML-081 Serving Models** — Deploying specific model versions to inference endpoints.

## Summary

Model versioning is a critical MLOps practice that enables reproducibility, auditability, and controlled deployment of ML models. Tools like DVC handle data and model versioning alongside Git, while MLflow Model Registry provides lifecycle management with stages (Staging, Production, Archived). Model lineage captures the full provenance — code, data, hyperparameters, environment, and metrics — for each version. A robust versioning strategy is essential for teams scaling from experimental notebooks to production ML systems.

## Key Takeaways

- Model versioning tracks the full lineage: code, data, hyperparameters, and environment
- DVC extends Git to version large datasets and models with content-addressed storage
- MLflow Model Registry manages model lifecycle through staging, production, and archived stages
- Model lineage enables full reproducibility and auditability
- Semantic versioning and consistent naming conventions are critical for scale
- Versioning is the foundation for A/B testing, rollbacks, and model governance
