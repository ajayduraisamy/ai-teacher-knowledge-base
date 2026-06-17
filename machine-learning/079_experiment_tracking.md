# Concept: Experiment Tracking

## Concept ID

ML-079

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

ML Engineering

## Learning Objectives

- Understand the purpose and benefits of experiment tracking
- Use MLflow Tracking to log parameters, metrics, artifacts, and tags
- Compare multiple runs programmatically and visually
- Use Weights & Biases (W&B) for collaborative experiment management
- Integrate TensorBoard for deep learning experiment monitoring

## Prerequisites

- Basic ML training workflow (data splitting, model training, evaluation)
- Familiarity with hyperparameter tuning concepts
- Python scripting experience

## Definition

Experiment tracking is the systematic process of recording, organizing, and comparing machine learning training runs. Each run captures hyperparameters, metrics, code versions, environment details, and artifacts (models, plots, datasets). Experiment tracking tools provide dashboards for visualizing run comparisons, searching historical runs, and sharing results with team members, enabling data-driven model selection and debugging.

## Intuition

Imagine training dozens of models with different hyperparameter combinations, data versions, or architectures. Without tracking, you would rely on memory, scattered notebooks, and manual notes to remember what worked. Experiment tracking is like a lab notebook for ML — it records every experiment automatically, making it possible to answer questions like "Which hyperparameter combination gave the best F1 score?" or "Did adding layer normalization help or hurt validation loss?"

## Why This Concept Matters

Experiment tracking is foundational to systematic ML development. It transforms ad-hoc experimentation into a disciplined, reproducible process. Key benefits include:

- **Reproducibility**: Every run is fully captured, enabling exact recreation
- **Comparison**: Side-by-side views of runs highlight what works
- **Collaboration**: Team members can view and build on each other's experiments
- **Debugging**: Logged metrics and artifacts help diagnose training issues
- **Governance**: Full audit trail of model development

## Code Examples

### Example 1: MLflow Tracking Basics

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Set tracking URI (default: local mlruns directory)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("california_housing")

# Load data
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="rf_baseline") as run:
    # Log parameters
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42,
    }
    mlflow.log_params(params)

    # Train model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metrics({"mse": mse, "rmse": rmse, "r2": r2})

    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")

    # Log additional artifacts
    with open("feature_importance.txt", "w") as f:
        for name, imp in zip(data.feature_names, model.feature_importances_):
            f.write(f"{name}: {imp:.4f}\n")
    mlflow.log_artifact("feature_importance.txt")

    run_id = run.info.run_id

print(f"Experiment: california_housing")
print(f"Run ID: {run_id}")
print(f"Parameters: {params}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Artifacts logged to: mlruns/")
```

```
# Output:
# Experiment: california_housing
# Run ID: 2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d
# Parameters: {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'random_state': 42}
# RMSE: 0.5123
# R²: 0.7984
# Artifacts logged to: mlruns/
```

### Example 2: Comparing Multiple Runs Programmatically

```python
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

client = MlflowClient()
experiment_name = "california_housing"
experiment = client.get_experiment_by_name(experiment_name)

# Search all runs for this experiment
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.rmse ASC"],
    max_results=5
)

print("=== Top 5 Runs by RMSE ===\n")

run_data = []
for run in runs:
    run_data.append({
        "run_name": run.data.tags.get("mlflow.runName", "N/A"),
        "run_id": run.info.run_id[:8],
        "rmse": run.data.metrics.get("rmse", "N/A"),
        "r2": run.data.metrics.get("r2", "N/A"),
        "n_estimators": run.data.params.get("n_estimators", "N/A"),
        "max_depth": run.data.params.get("max_depth", "N/A"),
        "status": run.info.status,
    })

df = pd.DataFrame(run_data)
print(df.to_string(index=False))

print("\n--- Best Run Details ---")
best_run = runs[0]
print(f"Best run: {best_run.data.tags.get('mlflow.runName', 'N/A')}")
print(f"RMSE: {best_run.data.metrics.get('rmse'):.4f}")
print(f"R²: {best_run.data.metrics.get('r2'):.4f}")
print(f"Params: n_estimators={best_run.data.params.get('n_estimators')}, "
      f"max_depth={best_run.data.params.get('max_depth')}")
```

```
# Output:
# === Top 5 Runs by RMSE ===
#
#   run_name    run_id   rmse     r2 n_estimators max_depth status
# rf_optimized  3b4c5d6e 0.4892 0.8123         200        15 FINISHED
#    rf_deep_v2 4c5d6e7f 0.5011 0.8056         150        20 FINISHED
#  rf_baseline  2a3b4c5d 0.5123 0.7984         100        10 FINISHED
#  rf_shallow   5d6e7f8a 0.5345 0.7821         100         5 FINISHED
#    rf_small   6e7f8a9b 0.5678 0.7654          50        10 FINISHED
#
# --- Best Run Details ---
# Best run: rf_optimized
# RMSE: 0.4892
# R²: 0.8123
# Params: n_estimators=200, max_depth=15
```

### Example 3: Weights & Biases Experiment Tracking

```python
import wandb
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Initialize W&B run
wandb.init(
    project="digit-classifier",
    name="gbm_v1",
    config={
        "learning_rate": 0.1,
        "n_estimators": 200,
        "max_depth": 6,
        "subsample": 0.8,
        "min_samples_leaf": 1,
    }
)

config = wandb.config

# Load data
data = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Train
model = GradientBoostingClassifier(
    learning_rate=config.learning_rate,
    n_estimators=config.n_estimators,
    max_depth=config.max_depth,
    subsample=config.subsample,
    min_samples_leaf=config.min_samples_leaf,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Log metrics
wandb.log({
    "accuracy": acc,
    "precision": precision,
    "recall": recall,
    "n_estimators": config.n_estimators,
})

# Log model summary
wandb.summary["best_accuracy"] = acc

# Log feature importance plot
wandb.log({"feature_importances": wandb.plot.histogram(
    model.feature_importances_, "Feature Importances"
)})

print(f"W&B Run: {wandb.run.id}")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"View run at: https://wandb.ai/{wandb.run.project}/{wandb.run.id}")

wandb.finish()
```

```
# Output:
# W&B Run: a1b2c3d4
# Accuracy: 0.9694
# Precision: 0.9696
# Recall: 0.9694
# View run at: https://wandb.ai/digit-classifier/a1b2c3d4
```

### Example 4: TensorBoard Integration with PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
X, y = make_classification(n_samples=2000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test).reshape(-1, 1)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define a simple network
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Initialize TensorBoard writer
writer = SummaryWriter("runs/binary_classifier_exp1")

model = BinaryClassifier(20)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_t)
        val_loss = criterion(val_outputs, y_test_t)
        val_acc = ((val_outputs > 0.5).float() == y_test_t).float().mean()

    # Log to TensorBoard
    writer.add_scalar("Loss/train", running_loss / len(train_loader), epoch)
    writer.add_scalar("Loss/val", val_loss.item(), epoch)
    writer.add_scalar("Accuracy/val", val_acc.item(), epoch)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss.item():.4f}, "
              f"Val Acc: {val_acc.item():.4f}")

writer.close()
print("\nTensorBoard logs saved to: runs/binary_classifier_exp1")
print("Run: tensorboard --logdir=runs")
```

```
# Output:
# Epoch [10/50] - Train Loss: 0.3123, Val Loss: 0.3456, Val Acc: 0.8650
# Epoch [20/50] - Train Loss: 0.2345, Val Loss: 0.2789, Val Acc: 0.8925
# Epoch [30/50] - Train Loss: 0.1876, Val Loss: 0.2345, Val Acc: 0.9100
# Epoch [40/50] - Train Loss: 0.1567, Val Loss: 0.2123, Val Acc: 0.9175
# Epoch [50/50] - Train Loss: 0.1345, Val Loss: 0.1987, Val Acc: 0.9225
#
# TensorBoard logs saved to: runs/binary_classifier_exp1
# Run: tensorboard --logdir=runs
```

## Common Mistakes

1. **Not setting an experiment name before logging**: Without setting an experiment, MLflow logs to the "Default" experiment, mixing unrelated runs together.

2. **Logging too many metrics at once**: Logging every batch-level metric can overwhelm the tracking system. Log epoch-level metrics and only key batch metrics.

3. **Forgetting to log the code version**: The most important metadata is often missed. Always log the Git commit hash or code snapshot.

4. **Not using tags for searchability**: Without meaningful tags, finding specific runs later becomes difficult. Tag runs with meaningful labels like "failed", "best_so_far", "experiment_name".

5. **Overwriting previous runs**: Reusing the same run ID or experiment name without creating a new run overwrites previous results.

6. **Logging large artifacts as metrics**: Artifacts (plots, models, datasets) should be logged as files, not as scalar metrics.

7. **Not cleaning up old runs**: The local tracking directory grows quickly. Periodically archive or delete runs that are no longer relevant.

## Interview Questions

### Beginner

1. **Q:** What is experiment tracking in machine learning?  
   **A:** It is the systematic recording of ML training runs, including hyperparameters, metrics, code versions, and artifacts, enabling comparison and reproducibility.

2. **Q:** What are the key components logged in an experiment run?  
   **A:** Parameters (hyperparameters), metrics (accuracy, loss), artifacts (model files, plots), tags, and source code version.

3. **Q:** How does MLflow track experiments?  
   **A:** MLflow uses a tracking server (local file system or remote) to store runs. Each run logs parameters, metrics, tags, and artifacts to a structured directory or database.

4. **Q:** What is the difference between a parameter and a metric in MLflow?  
   **A:** Parameters are inputs to the model (e.g., learning rate, n_estimators); metrics are outputs (e.g., accuracy, loss). Parameters are immutable in a run, metrics can be logged multiple times.

5. **Q:** What is TensorBoard used for?  
   **A:** TensorBoard is a visualization toolkit for deep learning experiments. It displays scalar metrics, histograms, model graphs, embeddings, and images over training steps.

### Intermediate

1. **Q:** How do you compare runs from different experiments in MLflow?  
   **A:** Use `MlflowClient.search_runs()` with experiment IDs and a filter string. Alternatively, the MLflow UI provides a side-by-side comparison view.

2. **Q:** What is the difference between MLflow Tracking and MLflow Model Registry?  
   **A:** Tracking logs individual training runs; the Model Registry manages the lifecycle (versioning, staging, production) of models that have been registered from those runs.

3. **Q:** How do you handle experiment tracking for distributed training (e.g., multi-GPU or multi-node)?  
   **A:** Use distributed logging strategies: log from the master node only, or use a centralized tracking server with unique run IDs per worker and aggregate metrics.

4. **Q:** How do you integrate experiment tracking with hyperparameter optimization (e.g., Optuna, GridSearchCV)?  
   **A:** Wrap each trial as a child run of a parent experiment run. Log the trial parameters and results, and use the parent run to summarize the best configuration.

5. **Q:** How can you automatically log model training with W&B?  
   **A:** Use `wandb.init()` and `wandb.log()` inside the training loop. W&B also provides automatic logging for frameworks like Hugging Face, PyTorch Lightning, and Keras via `wandb.config` and callbacks.

### Advanced

1. **Q:** Design an experiment tracking system for a team of 20 data scientists working on 50+ models simultaneously. Consider scalability, access control, and artifact storage.  
   **A:** Use a centralized MLflow tracking server backed by PostgreSQL (metadata) and S3 (artifacts). Implement access control via the tracking server's basic auth or a proxy. Use namespaces or projects to separate teams. Archive old experiments to cheaper storage. Set up automated cleanup policies.

2. **Q:** How do you track experiments where the data itself is versioned and the training code is parameterized by data version?  
   **A:** Log the data version (DVC hash, dataset name, timestamp) as a parameter. The experiment run captures the exact data version used, making it possible to identify which data version produced each model.

3. **Q:** Describe a strategy for monitoring experiment drift — detecting when new experiments are not comparable to historical ones due to changes in data distribution, feature definitions, or evaluation pipelines.  
   **A:** Implement automated checks in the tracking system: compare data version hashes, feature schemas, and evaluation pipeline versions between new and historical runs. Use statistical tests on logged metrics distributions. Alert when a new run's metrics fall outside expected ranges.

## Practice Problems

### Easy

1. Set up MLflow Tracking and log a single run with one parameter and one metric.

2. Log an sklearn model to MLflow and retrieve it using `mlflow.sklearn.load_model`.

3. Create a W&B project and log a simple training run with accuracy metric.

4. Use TensorBoard to log training and validation loss for a small PyTorch model.

5. Search and list all runs in an MLflow experiment using the Python API.

### Medium

1. Build a hyperparameter sweep that logs 20 runs with different learning rates and compares them using MLflow's search API.

2. Implement a custom callback for TensorBoard that logs per-class precision and recall for each epoch.

3. Create a comparison report (DataFrame) that shows the best hyperparameter combination across all runs in an experiment.

4. Set up MLflow autologging for an XGBoost model and analyze the captured metrics.

5. Build a script that finds and archives all MLflow runs older than 30 days to reduce storage.

### Hard

1. Implement a custom experiment tracking backend that stores runs in a PostgreSQL database with a FastAPI frontend for visualization.

2. Build a continuous experiment tracking system that listens to a model registry and automatically logs deployment metrics (latency, throughput) alongside training metrics.

3. Design and implement an automated experiment comparison system that generates HTML reports with plots comparing top-K runs across multiple metrics.

## Solutions

**Easy 1:**
```python
import mlflow
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
```

**Medium 1:**
```python
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

data = load_iris()
X, y = data.data, data.target

for lr in [0.01, 0.05, 0.1, 0.2]:
    with mlflow.start_run():
        model = RandomForestClassifier(max_depth=5, random_state=42)
        scores = cross_val_score(model, X, y, cv=5)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_metric("mean_accuracy", scores.mean())
        mlflow.log_metric("std_accuracy", scores.std())

# Compare runs
runs = mlflow.search_runs(order_by=["metrics.mean_accuracy DESC"])
print(runs[["params.learning_rate", "metrics.mean_accuracy", "metrics.std_accuracy"]])
```

**Hard 1:**
```python
# FastAPI + PostgreSQL experiment tracker (scaffold)
from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
from datetime import datetime
import json

app = FastAPI()
conn = sqlite3.connect("experiments.db", check_same_thread=False)

class Run(BaseModel):
    experiment_name: str
    params: dict
    metrics: dict
    tags: dict = {}

@app.post("/runs")
def create_run(run: Run):
    conn.execute("""
        INSERT INTO runs (experiment_name, params, metrics, tags, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (run.experiment_name, json.dumps(run.params),
          json.dumps(run.metrics), json.dumps(run.tags),
          datetime.utcnow().isoformat()))
    conn.commit()
    return {"status": "ok"}

@app.get("/runs/{experiment_name}")
def list_runs(experiment_name: str):
    cursor = conn.execute(
        "SELECT * FROM runs WHERE experiment_name = ? ORDER BY created_at DESC",
        (experiment_name,)
    )
    return [dict(row) for row in cursor.fetchall()]
```

## Related Concepts

- **ML-078 Model Versioning**: Experiment tracking feeds into model versioning. Successful experiments produce models that are registered and versioned.
- **ML-084 Reproducibility**: Comprehensive experiment logging (parameters, metrics, code version, environment) is the foundation of reproducibility.
- **ML-076 ML Pipelines**: Experiment tracking logs entire pipeline runs, capturing intermediate results and final model performance.
- **ML-090 ML Project Lifecycle**: Experiment tracking is a core activity in the modeling and evaluation phases of the ML lifecycle.

## Next Concepts

- **ML-084 Reproducibility** — Using experiment tracking logs to reproduce results exactly.
- **ML-090 ML Project Lifecycle** — Understanding where experiment tracking fits in the broader ML workflow.

## Summary

Experiment tracking transforms ML model development from a chaotic, memory-based process into a systematic, data-driven discipline. Tools like MLflow Tracking, Weights & Biases, and TensorBoard provide robust platforms for logging, comparing, and sharing experiments. By consistently capturing parameters, metrics, artifacts, and environment metadata, teams can make informed decisions about model selection, diagnose training issues, and ensure reproducibility. Experiment tracking is a core MLOps practice that scales from individual research to enterprise teams.

## Key Takeaways

- Experiment tracking records parameters, metrics, artifacts, and environment for each run
- MLflow Tracking provides a lightweight, open-source solution for experiment logging
- W&B offers collaborative features including dashboards, reports, and team management
- TensorBoard excels at visualizing deep learning training dynamics over time
- Run comparison enables data-driven model selection and hyperparameter tuning
- Code version integration ensures reproducibility and auditability
- Experiment tracking is foundational to model governance and MLOps maturity
