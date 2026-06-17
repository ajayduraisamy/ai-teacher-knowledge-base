# Concept: Serving Models

## Concept ID

ML-081

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

ML Engineering

## Learning Objectives

- Understand the architecture of model serving systems
- Deploy a model as a REST API using Flask and FastAPI
- Containerize a model serving application with Docker
- Use specialized serving frameworks: TensorFlow Serving, TorchServe, BentoML
- Design proper request/response formats and error handling

## Prerequisites

- Basic web development concepts (HTTP, REST APIs)
- Docker fundamentals (images, containers, Dockerfiles)
- Model serialization skills (ML-080)

## Definition

Model serving is the practice of deploying trained machine learning models to production environments where they can receive input data and return predictions in real time or in batches. Serving infrastructure wraps the model in an API layer that handles authentication, request validation, batching, load balancing, and monitoring, while abstracting away the underlying ML framework.

## Intuition

A serving system is like a restaurant kitchen. The trained model is the chef who knows how to cook. The API layer is the waitstaff who takes orders from customers, translates them into kitchen tickets, and delivers the finished dishes. The serving infrastructure (Docker, load balancers) is the restaurant layout that ensures orders flow smoothly even during peak hours.

## Why This Concept Matters

Model serving is where the value of ML is realized — a model that cannot be served is useless. Serving frameworks handle critical production concerns: latency, throughput, resource management, model versioning, and graceful degradation. The choice of serving approach (custom API vs. specialized framework) affects operational complexity, scalability, and maintainability.

## Code Examples

### Example 1: Flask REST API for Model Serving

```python
import pickle
import numpy as np
from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Train a simple model (in practice, load from disk)
data = load_iris()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(data.data, data.target)

# Save model
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Create Flask app
app = Flask(__name__)

# Load model at startup
with open("iris_model.pkl", "rb") as f:
    clf = pickle.load(f)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Validate input
        if "features" not in data:
            return jsonify({"error": "Missing 'features' field"}), 400

        features = np.array(data["features"])

        # Handle single vs batch
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.shape[1] != 4:
            return jsonify({"error": "Expected 4 features"}), 400

        # Predict
        predictions = clf.predict(features)
        probabilities = clf.predict_proba(features)

        # Format response
        response = {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "class_names": data.target_names.tolist() if hasattr(data, 'target_names') else None
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Iris model server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
```

```python
# Client code
import requests
import json

# Health check
resp = requests.get("http://localhost:5000/health")
print(f"Health: {resp.json()}")

# Single prediction
single_input = {"features": [5.1, 3.5, 1.4, 0.2]}
resp = requests.post("http://localhost:5000/predict", json=single_input)
print(f"Single prediction: {resp.json()}")

# Batch prediction
batch_input = {"features": [
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 3.4, 5.4, 2.3],
    [5.9, 3.0, 4.2, 1.5]
]}
resp = requests.post("http://localhost:5000/predict", json=batch_input)
print(f"Batch predictions: {json.dumps(resp.json(), indent=2)}")
```

```
# Output:
# Health: {'status': 'healthy'}
# Single prediction: {'predictions': [0], 'probabilities': [[0.97, 0.03, 0.0]], 'class_names': None}
# Batch predictions: {
#   "predictions": [0, 2, 1],
#   "probabilities": [[0.97, 0.03, 0.0], [0.0, 0.02, 0.98], [0.01, 0.94, 0.05]],
#   "class_names": null
# }
```

### Example 2: FastAPI with Async Support and Validation

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing

# Train model
data = fetch_california_housing()
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(data.data, data.target)
joblib.dump(model, "housing_model.joblib")
joblib.dump(data.feature_names, "housing_features.joblib")

# Load model
model = joblib.load("housing_model.joblib")
feature_names = joblib.load("housing_features.joblib")

app = FastAPI(title="California Housing Price Predictor")

class PredictionInput(BaseModel):
    features: List[float] = Field(..., description="Feature values in order")
    features_dict: Optional[dict] = None

class BatchPredictionInput(BaseModel):
    instances: List[PredictionInput]

class PredictionOutput(BaseModel):
    prediction: float
    feature_names: List[str]

class BatchPredictionOutput(BaseModel):
    predictions: List[float]

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "GradientBoostingRegressor"}

@app.get("/metadata")
async def metadata():
    return {
        "model_type": type(model).__name__,
        "features": feature_names,
        "n_features": len(feature_names),
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(input: PredictionInput):
    if len(input.features) != len(feature_names):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(feature_names)} features, got {len(input.features)}"
        )
    features = np.array(input.features).reshape(1, -1)
    prediction = model.predict(features)[0]
    return {"prediction": float(prediction), "feature_names": feature_names}

@app.post("/predict_batch", response_model=BatchPredictionOutput)
async def predict_batch(input: BatchPredictionInput):
    features_list = [inst.features for inst in input.instances]
    features_array = np.array(features_list)
    predictions = model.predict(features_array)
    return {"predictions": predictions.tolist()}
```

```
# Output (tested with httpx):
# Health: {"status":"healthy","model":"GradientBoostingRegressor"}
# Metadata: {"model_type":"GradientBoostingRegressor","features":["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"],"n_features":8}
# Predict: {"prediction":2.3456,"feature_names":["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]}
# Predict batch: {"predictions":[2.3456,1.2345,3.4567]}
```

### Example 3: Docker Containerization

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application code
COPY model/ ./model/
COPY app.py .

# Create non-root user
RUN useradd -m -u 1000 mluser && chown -R mluser:mluser /app
USER mluser

# Expose port
EXPOSE 8000

# Run with Uvicorn for FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```txt
# requirements.txt
fastapi==0.104.1
uvicorn==0.24.0
joblib==1.3.2
numpy==1.24.3
scikit-learn==1.3.2
pydantic==2.5.0
```

```bash
# Build and run
docker build -t housing-predictor .
docker run -d -p 8000:8000 --name housing-api housing-predictor

# Test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [8.3252, 41.0, 6.9841, 1.0238, 322.0, 2.5556, 37.88, -122.23]}'
```

```
# Output:
# {"prediction":4.526,"feature_names":["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]}
```

### Example 4: TensorFlow Serving with SavedModel

```python
import tensorflow as tf
import numpy as np

# Train a simple model
tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')

# Generate fake data
X = np.random.randn(1000, 10).astype(np.float32)
y = (X.sum(axis=1) > 0).astype(np.float32)

model.fit(X, y, epochs=10, verbose=0, validation_split=0.2)

# Define serving signature
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 10], dtype=tf.float32)])
def serving_default_fn(inputs):
    return {"prediction": model(inputs)}

# Save with signature
tf.saved_model.save(
    model,
    "tf_serving_model/0001",
    signatures={"serving_default": serving_default_fn}
)

print("Model saved to tf_serving_model/0001/")
print("Saved artifacts:")
import os
for root, dirs, files in os.walk("tf_serving_model/0001/"):
    for f in files:
        print(f"  {os.path.join(root, f)}")

# Serve with TensorFlow Serving:
# docker run -p 8501:8501 --mount type=bind,source=$(pwd)/tf_serving_model,target=/models/housing \
#   -e MODEL_NAME=housing -t tensorflow/serving

# Test client
import json
import requests

test_input = {
    "instances": np.random.randn(3, 10).tolist()
}
resp = requests.post(
    "http://localhost:8501/v1/models/housing:predict",
    json=test_input
)
print(f"\nTF Serving response: {resp.json()}")
```

```
# Output:
# Model saved to tf_serving_model/0001/
# Saved artifacts:
#   tf_serving_model/0001/saved_model.pb
#   tf_serving_model/0001/variables/variables.index
#   tf_serving_model/0001/variables/variables.data-00000-of-00001
#   tf_serving_model/0001/assets/
#
# TF Serving response: {"predictions": [[0.234], [0.678], [0.123]]}
```

### Example 5: BentoML for Full Serving Pipeline

```python
import bentoml
import numpy as np
from bentoml.io import JSON, PandasDataFrame
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

# Train model
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save with BentoML
bento_model = bentoml.sklearn.save_model("housing_price_predictor", model)
print(f"Model saved: {bento_model}")

# Create a service
@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class HousingPredictor:
    def __init__(self):
        self.model = bentoml.sklearn.load_model("housing_price_predictor:latest")
        self.feature_names = data.feature_names.tolist()

    @bentoml.api(input=JSON(), output=JSON())
    def predict(self, input_data: dict) -> dict:
        features = np.array(input_data["features"]).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        return {
            "prediction": float(prediction),
            "features_used": self.feature_names
        }

    @bentoml.api(input=PandasDataFrame(), output=JSON())
    def predict_batch(self, df: pd.DataFrame) -> dict:
        predictions = self.model.predict(df.values)
        return {"predictions": predictions.tolist()}

print("BentoML service defined. Run with: bentoml serve service.py:svc")
```

```
# Output:
# Model saved: Model(tag="housing_price_predictor:qwerty1234")
# BentoML service defined. Run with: bentoml serve service.py:svc
```

## Common Mistakes

1. **Not handling batch prediction in the API design**: Serving systems should accept both single instances and batches to efficiently handle varying load patterns.

2. **Loading the model on every request**: Loading the model from disk for each request adds seconds of latency. Load once at startup and reuse.

3. **Missing health check endpoints**: Orchestrators (Kubernetes, Docker Swarm) require health checks to manage container lifecycle.

4. **Exposing the model file directly**: Serving the raw pickle file via a static endpoint is insecure and inefficient. Use a proper API layer.

5. **Ignoring request validation**: Malformed input can crash the serving process. Validate inputs with Pydantic schemas or equivalent.

6. **Not setting timeouts**: Long-running inference requests can exhaust server resources. Configure appropriate timeout values.

7. **Mixing GPU and CPU model paths**: Models trained on GPU may fail on CPU if not configured for device-agnostic execution.

## Interview Questions

### Beginner

1. **Q:** What is the difference between model serving and model training infrastructure?  
   **A:** Training infrastructure is optimized for high throughput and batch processing. Serving infrastructure is optimized for low latency, high availability, and handling diverse traffic patterns.

2. **Q:** What is a REST API for model serving?  
   **A:** A REST API exposes HTTP endpoints (typically `/predict`) that accept JSON input and return JSON predictions, following RESTful principles.

3. **Q:** Why would you use a specialized serving framework like TensorFlow Serving instead of a custom Flask app?  
   **A:** TF Serving provides automatic batching, model version management, GPU acceleration, and optimized serving performance without writing custom infrastructure code.

4. **Q:** What is the purpose of Docker in model serving?  
   **A:** Docker containerizes the model and its dependencies into a reproducible, portable unit that can be deployed consistently across development, staging, and production environments.

5. **Q:** What is a health check endpoint and why is it important?  
   **A:** A health check (e.g., `/health`) returns the service status. Orchestrators use it to determine if the container is alive and ready to receive traffic.

### Intermediate

1. **Q:** Compare FastAPI and Flask for model serving. Which would you choose and why?  
   **A:** FastAPI provides automatic request validation (Pydantic), async support, auto-generated OpenAPI docs, and better performance. Flask is simpler but requires more manual setup for validation and documentation. Choose FastAPI for most production use cases.

2. **Q:** How does automatic batching work in TensorFlow Serving?  
   **A:** TF Serving queues incoming requests and groups them into batches based on configurable parameters (max_batch_size, batch_timeout_micros). This increases throughput at the cost of slight latency increase.

3. **Q:** How do you handle model versioning in a serving system?  
   **A:** Use a model registry that stores multiple versions. The serving infrastructure loads a specific version or can route traffic across versions for A/B testing. TF Serving supports serving multiple versions simultaneously.

4. **Q:** What is BentoML and how does it simplify model serving?  
   **A:** BentoML is a unified model serving framework that handles model packaging, dependency management, API generation, and containerization. It provides built-in support for monitoring, batching, and distributed serving.

5. **Q:** How do you scale a model serving system horizontally?  
   **A:** Deploy multiple replicas behind a load balancer. Use a shared model registry or distributed cache to avoid redundant model loading. Consider partitioning by model version or customer segment.

### Advanced

1. **Q:** Design a multi-model serving architecture for a platform that hosts 100+ models with different resource requirements (CPU vs GPU, small vs large).  
   **A:** Use Kubernetes with node pools for CPU and GPU. Deploy models as microservices, each in its own pod with resource requests/limits. Use a service mesh (Istio) for traffic routing. Implement a model router that maps model IDs to backend services. Use horizontal pod autoscaling based on request latency and throughput.

2. **Q:** How do you implement canary deployments for ML models in production?  
   **A:** Deploy the new model version alongside the current version. Route a small percentage of traffic (e.g., 5%) to the canary. Monitor metrics (latency, prediction distribution, business KPIs). Gradually increase traffic if metrics are stable, or roll back if anomalies are detected. The model registry tracks which version is serving which percentage of traffic.

3. **Q:** Describe a strategy for serving large language models (LLMs) with acceptable latency. Include model optimization, hardware selection, and request handling.  
   **A:** Use model quantization (INT8, FP16) and pruning to reduce model size. Deploy on GPUs with high VRAM (A100, H100). Implement KV-cache for autoregressive generation. Use continuous batching to maximize GPU utilization. Set up a response streaming API for token-by-token delivery. Use speculative decoding for latency reduction.

## Practice Problems

### Easy

1. Create a Flask endpoint that loads a pickle model and returns predictions for a single input.

2. Add a `/health` endpoint and a `/metadata` endpoint to a FastAPI model server.

3. Write a Dockerfile for a FastAPI model serving application.

4. Use `curl` to test a model serving endpoint with a sample input.

5. Implement request validation in FastAPI using Pydantic models for a serving endpoint.

### Medium

1. Build a FastAPI model server that supports both single and batch prediction endpoints.

2. Containerize a TorchServe deployment with a custom handler for a PyTorch vision model.

3. Implement request caching so that identical inputs within a time window return cached predictions.

4. Set up TensorFlow Serving with model version management and test version-based routing.

5. Create a BentoML service that includes pre-processing and post-processing transforms.

### Hard

1. Build a gRPC-based model serving endpoint and compare its latency vs. REST for 10,000 requests.

2. Implement a multi-model serving system where a single endpoint routes to different models based on a `model_id` in the request.

3. Design and deploy a complete model serving pipeline with Kubernetes: model registry, canary deployment, auto-scaling, and monitoring with Prometheus/Grafana.

## Solutions

**Easy 1:**
```python
from flask import Flask, request, jsonify
import joblib
app = Flask(__name__)
model = joblib.load("model.joblib")
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    pred = model.predict([data["features"]])[0]
    return jsonify({"prediction": float(pred)})
```

**Medium 1:**
```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model.joblib")

class SingleInput(BaseModel):
    features: List[float]

class BatchInput(BaseModel):
    instances: List[List[float]]

@app.post("/predict")
def predict_single(input: SingleInput):
    pred = model.predict([input.features])[0]
    return {"prediction": float(pred)}

@app.post("/predict_batch")
def predict_batch(input: BatchInput):
    preds = model.predict(input.instances)
    return {"predictions": preds.tolist()}
```

**Hard 1:**
```python
# gRPC server (protocol buffer definition)
# syntax = "proto3";
# service PredictionService {
#   rpc Predict (PredictionRequest) returns (PredictionResponse);
# }
# message PredictionRequest {
#   repeated float features = 1;
# }
# message PredictionResponse {
#   float prediction = 1;
# }
```

## Related Concepts

- **ML-080 Model Serialization**: Models must be serialized before they can be served.
- **ML-078 Model Versioning**: The model registry tracks which model versions are deployed and serving.
- **ML-082 Batch vs Realtime**: Serving architecture depends on whether predictions are needed in real-time or can be processed in batches.
- **ML-084 Reproducibility**: Containerization ensures the serving environment matches the training environment.

## Next Concepts

- **ML-082 Batch vs Realtime** — Choosing the right serving paradigm for your use case.
- **ML-090 ML Project Lifecycle** — Understanding where model serving fits in the overall ML lifecycle.

## Summary

Model serving is the production deployment of ML models behind API endpoints. Custom frameworks (Flask, FastAPI) offer flexibility and simplicity for small-scale deployments, while specialized frameworks (TensorFlow Serving, TorchServe, BentoML) provide optimization, version management, and scalability. Docker containerization ensures environment consistency from training through deployment. Key considerations include request validation, batching, health checks, monitoring, and graceful handling of traffic spikes. A well-architected serving system is essential for reliable, low-latency ML inference in production.

## Key Takeaways

- Model serving exposes trained models as HTTP APIs for inference
- FastAPI provides async support, auto-validation, and OpenAPI docs
- Docker ensures consistent environments from training to serving
- TF Serving, TorchServe, and BentoML offer framework-specific optimizations
- Always implement health checks, request validation, and error handling
- Consider batch vs. real-time tradeoffs in API design
- Containerization + orchestration enables horizontal scaling
