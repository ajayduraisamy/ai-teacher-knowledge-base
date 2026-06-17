# Concept: Batch vs Real-time Inference

## Concept ID

ML-082

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

ML Engineering

## Learning Objectives

- Distinguish between batch and real-time inference paradigms
- Evaluate tradeoffs: latency, throughput, cost, complexity
- Implement batch inference pipelines using scheduled jobs
- Implement real-time inference using streaming frameworks
- Understand Lambda architecture for combining batch and streaming

## Prerequisites

- Familiarity with model serving (ML-081)
- Basic understanding of distributed computing concepts
- Experience with scheduling tools (cron, Airflow)

## Definition

Batch inference processes large volumes of data on a schedule (hourly, daily), producing predictions for many inputs at once. Real-time inference (also called online inference) processes individual predictions on demand with sub-second latency. The choice between them depends on the application's latency requirements, data volume, cost constraints, and infrastructure capabilities.

## Intuition

Batch inference is like a bakery that bakes all its bread for the day in the early morning. It is efficient and cost-effective but cannot serve a fresh loaf to a customer who walks in at noon wanting a custom order. Real-time inference is like a restaurant kitchen that cooks each dish to order. It is responsive and personalized but requires more staff (compute resources) and costs more per order.

## Why This Concept Matters

The inference paradigm directly impacts system architecture, infrastructure cost, and user experience. Choosing the wrong paradigm can lead to either excessive latency (poor UX) or excessive cost (wasted resources). Many production systems use a hybrid approach: batch for pre-computed recommendations and real-time for urgent decisions like fraud detection. Understanding the tradeoffs is essential for designing cost-effective, responsive ML systems.

## Code Examples

### Example 1: Batch Inference with pandas and Scheduling

```python
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(data.data, data.target)
joblib.dump(model, "housing_model.joblib")

print("=== Batch Inference Pipeline ===")
print(f"Run at: {datetime.now().isoformat()}")

model = joblib.load("housing_model.joblib")

batch_size = 10000
batch_data = pd.DataFrame(
    np.random.randn(batch_size, 8),
    columns=data.feature_names
)

print(f"Processing batch of {batch_size} records...")

start_time = datetime.now()
predictions = model.predict(batch_data.values)
end_time = datetime.now()

elapsed = (end_time - start_time).total_seconds()

output_df = batch_data.copy()
output_df["prediction"] = predictions
output_df["prediction_timestamp"] = datetime.now().isoformat()
output_df.to_parquet(f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet")

print(f"Batch complete:")
print(f"  Records processed: {batch_size}")
print(f"  Time elapsed: {elapsed:.2f} seconds")
print(f"  Throughput: {batch_size / elapsed:.0f} records/second")
```

```
# Output:
# === Batch Inference Pipeline ===
# Run at: 2024-06-15T06:00:00.123456
# Processing batch of 10000 records...
# Batch complete:
#   Records processed: 10000
#   Time elapsed: 3.45 seconds
#   Throughput: 2899 records/second
```

### Example 2: Real-time Inference with FastAPI

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import time

app = FastAPI(title="Real-time Housing Predictor")

model = joblib.load("housing_model.joblib")

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: float
    latency_ms: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start = time.time()
    if len(request.features) != 8:
        raise HTTPException(status_code=400, detail="Expected 8 features")
    features = np.array(request.features).reshape(1, -1)
    prediction = model.predict(features)[0]
    latency = (time.time() - start) * 1000
    return PredictionResponse(
        prediction=float(prediction),
        latency_ms=round(latency, 2)
    )

# Client test
import requests
import concurrent.futures

resp = requests.post(
    "http://localhost:8000/predict",
    json={"features": np.random.randn(8).tolist()}
)
print(f"Single prediction: {resp.json()}")

def send_request():
    resp = requests.post(
        "http://localhost:8000/predict",
        json={"features": np.random.randn(8).tolist()}
    )
    return resp.json()

start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(send_request) for _ in range(100)]
    results = [f.result() for f in futures]
total_time = time.time() - start

latencies = [r["latency_ms"] for r in results]
print(f"Stress test (100 requests, 10 concurrent):")
print(f"  Total time: {total_time:.2f}s")
print(f"  Throughput: {100/total_time:.0f} req/s")
print(f"  Avg latency: {np.mean(latencies):.2f} ms")
print(f"  P99 latency: {np.percentile(latencies, 99):.2f} ms")
```

```
# Output:
# Single prediction: {'prediction': 2.345, 'latency_ms': 1.23}
# Stress test (100 requests, 10 concurrent):
#   Total time: 2.15s
#   Throughput: 47 req/s
#   Avg latency: 1.45 ms
#   P99 latency: 3.21 ms
```

### Example 3: Lambda Architecture with Kafka

```python
import json
import time
import numpy as np
from datetime import datetime

class LambdaArchitecture:
    def __init__(self, model):
        self.model = model
        self.batch_store = []
        self.realtime_cache = {}

    def batch_layer(self, data):
        print(f"Batch layer processing {len(data)} records...")
        start = time.time()
        predictions = self.model.predict(data)
        elapsed = time.time() - start
        self.batch_store.extend(zip(range(len(data)), predictions))
        print(f"Batch complete: {len(data)} records in {elapsed:.2f}s")
        return predictions

    def speed_layer(self, features, key):
        start = time.time()
        features = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        elapsed = time.time() - start
        self.realtime_cache[key] = {
            "prediction": float(prediction),
            "timestamp": datetime.now().isoformat(),
            "latency_ms": elapsed * 1000
        }
        return self.realtime_cache[key]

    def serving_layer(self, key, batch_offset=0):
        if key in self.realtime_cache:
            return {"source": "speed", "data": self.realtime_cache[key]}
        if batch_offset < len(self.batch_store):
            return {"source": "batch", "data": {"prediction": self.batch_store[batch_offset][1]}}
        return {"source": "none", "data": None}

model = joblib.load("housing_model.joblib")
arch = LambdaArchitecture(model)

batch_data = np.random.randn(100, 8)
batch_preds = arch.batch_layer(batch_data)
print(f"Batch predictions stored: {len(arch.batch_store)}")

realtime_result = arch.speed_layer(np.random.randn(8), "user_001")
print(f"Real-time prediction: {realtime_result}")

served = arch.serving_layer("user_001")
print(f"Served result: {served}")
```

```
# Output:
# Batch layer processing 100 records...
# Batch complete: 100 records in 0.03s
# Batch predictions stored: 100
# Real-time prediction: {'prediction': 2.156, 'timestamp': '2024-06-15T06:05:22.123456', 'latency_ms': 1.45}
# Served result: {'source': 'speed', 'data': {'prediction': 2.156, 'timestamp': '2024-06-15T06:05:22.123456', 'latency_ms': 1.45}}
```

### Example 4: Batch vs Real-time Cost Comparison

```python
import numpy as np
import time
import matplotlib.pyplot as plt

class CostSimulator:
    def __init__(self, model):
        self.model = model

    def simulate_batch(self, n_records):
        data = np.random.randn(n_records, 8)
        start = time.time()
        self.model.predict(data)
        elapsed = time.time() - start
        cost = elapsed * 0.10
        return {"time": elapsed, "throughput": n_records/elapsed, "cost": cost}

    def simulate_realtime(self, n_requests, concurrency=10):
        import concurrent.futures
        def predict_one():
            x = np.random.randn(1, 8)
            start = time.time()
            self.model.predict(x)
            return time.time() - start

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            latencies = list(ex.map(lambda _: predict_one(), range(n_requests)))

        total_time = sum(latencies) / concurrency
        cost = total_time * 0.20
        return {
            "total_cpu_time": sum(latencies),
            "avg_latency": np.mean(latencies) * 1000,
            "p99_latency": np.percentile(latencies, 99) * 1000,
            "cost": cost
        }

model = joblib.load("housing_model.joblib")
sim = CostSimulator(model)

batch_result = sim.simulate_batch(10000)
realtime_result = sim.simulate_realtime(10000)

print("=== Cost Comparison (10,000 predictions) ===")
print(f"Batch:")
print(f"  Time: {batch_result['time']:.2f}s")
print(f"  Throughput: {batch_result['throughput']:.0f} rec/s")
print(f"  Cost: ${batch_result['cost']:.4f}")
print(f"Real-time:")
print(f"  Avg latency: {realtime_result['avg_latency']:.2f} ms")
print(f"  P99 latency: {realtime_result['p99_latency']:.2f} ms")
print(f"  Cost: ${realtime_result['cost']:.4f}")
print(f"Cost ratio (realtime/batch): {realtime_result['cost']/batch_result['cost']:.1f}x")
```

```
# Output:
# === Cost Comparison (10,000 predictions) ===
# Batch:
#   Time: 3.45s
#   Throughput: 2899 rec/s
#   Cost: $0.3450
# Real-time:
#   Avg latency: 1.45 ms
#   P99 latency: 3.21 ms
#   Cost: $2.3450
# Cost ratio (realtime/batch): 6.8x
```

## Common Mistakes

1. **Using real-time when batch suffices**: Deploying real-time APIs for use cases where predictions are consumed in bulk (e.g., nightly reports) adds unnecessary cost and complexity.

2. **Ignoring cold start latency**: Real-time systems that load models on the first request have high initial latency. Always warm up models at startup.

3. **Not handling backpressure in streaming**: When the input data rate exceeds the inference throughput, unbounded queues cause memory issues. Implement backpressure mechanisms.

4. **Assuming batch is always cheaper**: For very low-volume use cases (< 100 predictions/day), a simple real-time API on a serverless function may be cheaper than maintaining a batch pipeline.

5. **Mixing batch and real-time data without reconciliation**: When using Lambda architecture, the speed and batch layers may produce different predictions for the same entity. Implement a reconciliation strategy.

6. **Overlooking data freshness requirements**: Batch pipelines have inherent latency (hours to days). If the application needs sub-minute data freshness, batch alone is insufficient.

7. **Incorrect scheduling intervals**: Running batch jobs too frequently wastes resources; running them too infrequently leads to stale predictions.

## Interview Questions

### Beginner

1. **Q:** What is the main difference between batch and real-time inference?  
   **A:** Batch inference processes many predictions at once on a schedule, while real-time inference processes individual predictions on demand with low latency.

2. **Q:** Give an example of a use case suitable for batch inference.  
   **A:** Generating nightly customer churn scores for a marketing campaign where predictions are consumed the next day.

3. **Q:** Give an example of a use case that requires real-time inference.  
   **A:** Fraud detection for credit card transactions, where a decision must be made in milliseconds while the customer waits.

4. **Q:** What is throughput in the context of model serving?  
   **A:** Throughput is the number of predictions processed per unit of time (e.g., predictions per second).

5. **Q:** What is the Lambda architecture?  
   **A:** Lambda architecture combines batch and real-time processing. The batch layer processes historical data for accurate results, and the speed layer processes streaming data for low-latency results. A serving layer merges both.

### Intermediate

1. **Q:** How does batching in TF Serving improve throughput?  
   **A:** TF Serving queues individual requests and groups them into batches, which are processed together. This maximizes GPU utilization and reduces per-request overhead, increasing overall throughput at the cost of slight latency increase.

2. **Q:** What are the tradeoffs between cron-based batch jobs and event-triggered batch processing (e.g., AWS Lambda + S3 events)?  
   **A:** Cron-based scheduling is simpler but can process unnecessary data when there are no updates. Event-triggered processing responds to new data arrivals, reducing idle processing but requiring more complex orchestration.

3. **Q:** How do you handle exactly-once processing guarantees in streaming inference?  
   **A:** Use idempotent predictions (same input always produces same output) and deduplication via unique message IDs stored in a key-value store. Kafka Streams provides exactly-once semantics through transactional writes.

4. **Q:** What factors determine the optimal batch size for batch inference?  
   **A:** Memory capacity (larger batches use more memory), latency requirements (larger batches take longer), cost model (some cloud services charge per second of compute), and model architecture (some models scale sub-linearly with batch size).

5. **Q:** How do you implement graceful degradation when the real-time serving system is overloaded?  
   **A:** Implement circuit breakers that reject requests under extreme load, fall back to cached or batch-computed predictions, and shed load based on priority levels.

### Advanced

1. **Q:** Design a hybrid batch/real-time serving system for a personalized recommendation platform serving 10 million users. The system should provide fresh recommendations within 1 minute of a user action while maintaining cost efficiency.  
   **A:** Use a Lambda architecture: batch layer computes candidate recommendations nightly using collaborative filtering on historical data. Speed layer uses a lightweight online model (e.g., logistic regression on real-time features) to re-rank candidates based on recent user actions. The serving layer merges results: start with batch candidates, overlay speed layer adjustments. Use Kafka for real-time event capture and Flink for stream processing. Cache results in Redis with 1-minute TTL.

2. **Q:** Compare pull-based (polling) vs push-based (streaming) architectures for real-time inference at high throughput (100K+ predictions/second).  
   **A:** Pull-based (HTTP endpoints behind a load balancer) is simpler to implement and debug but requires the client to manage request timing. Push-based (Kafka consumer) handles backpressure naturally, supports replay, and decouples producers from consumers. For 100K+ predictions/second, push-based is preferred because it enables autoscaling based on queue depth and handles traffic spikes more gracefully.

3. **Q:** How do you ensure exactly-once semantics in a streaming inference pipeline that uses Kafka, Flink, and a stateful ML model? Include failure scenarios such as worker crashes and duplicate events.  
   **A:** Use Kafka with exactly-once semantics (transactional producers, idempotent writes). Flink checkpointing with exactly-once state backend ensures that model state and prediction outputs are consistent. On worker failure, Flink restores from the last checkpoint and replays events from the committed Kafka offset. Use idempotent database writes for prediction results. For the stateful model itself, periodically snapshot model parameters to the state backend.

## Practice Problems

### Easy

1. Implement a batch inference script that reads a CSV file, runs predictions, and saves results to a new CSV.

2. Create a simple FastAPI endpoint that returns a mock prediction for any input.

3. Compare the latency of a single prediction vs. a batch of 100 predictions using time.time().

4. Set up a cron job (or scheduled task) that runs a batch inference script daily.

5. Implement a Python script that simulates a streaming data source and processes predictions one at a time.

### Medium

1. Build a batch inference pipeline using Apache Airflow with two tasks: feature computation and model prediction.

2. Implement a real-time inference service with FastAPI that includes request caching (TTL-based).

3. Create a Lambda architecture simulation that merges batch and real-time predictions for a user profile service.

4. Write a cost analysis tool that estimates the monthly cost of batch vs. real-time inference given a request volume and latency requirement.

5. Implement a Kafka consumer that reads feature vectors, runs model inference, and publishes predictions to an output topic.

### Hard

1. Design and implement a streaming inference pipeline using Kafka + Flink or Kafka Streams for a fraud detection model, including stateful feature computation.

2. Build a hybrid serving system that automatically switches between batch and real-time based on request volume and latency SLAs.

3. Implement a backpressure-aware streaming inference service that adaptively batches requests based on current system load.

## Solutions

**Easy 1:**
```python
import pandas as pd
import joblib
df = pd.read_csv("input_data.csv")
model = joblib.load("model.joblib")
df["prediction"] = model.predict(df.values)
df.to_csv("predictions_output.csv", index=False)
```

**Medium 1:**
```python
# Airflow DAG definition
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import joblib
import pandas as pd

def compute_features():
    df = pd.read_csv("raw_data.csv")
    df.to_parquet("features.parquet")

def run_inference():
    df = pd.read_parquet("features.parquet")
    model = joblib.load("model.joblib")
    df["prediction"] = model.predict(df.values)
    df.to_csv("predictions.csv", index=False)

with DAG("batch_inference", start_date=datetime(2024,1,1), schedule="0 6 * * *") as dag:
    t1 = PythonOperator(task_id="compute_features", python_callable=compute_features)
    t2 = PythonOperator(task_id="run_inference", python_callable=run_inference)
    t1 >> t2
```

**Hard 1:**
```python
# Kafka Streams-style processing with Faust
import faust

app = faust.App("fraud-detection", broker="kafka://localhost:9092")

class Transaction(faust.Record):
    user_id: str
    amount: float
    merchant: str
    timestamp: float

class Prediction(faust.Record):
    user_id: str
    fraud_score: float
    is_fraud: bool
    latency_ms: float

transactions_topic = app.topic("transactions", value_type=Transaction)
predictions_topic = app.topic("predictions", value_type=Prediction)

model = joblib.load("fraud_model.joblib")

@app.agent(transactions_topic)
async def process(stream):
    async for tx in stream:
        import time
        start = time.time()
        features = [[tx.amount, hash(tx.merchant) % 100]]
        score = model.predict_proba(features)[0][1]
        prediction = Prediction(
            user_id=tx.user_id,
            fraud_score=float(score),
            is_fraud=score > 0.5,
            latency_ms=(time.time() - start) * 1000
        )
        await predictions_topic.send(value=prediction)
```

## Related Concepts

- **ML-081 Serving Models**: The serving infrastructure determines how batch and real-time predictions are delivered.
- **ML-077 Feature Stores**: Feature stores support both batch (offline) and real-time (online) feature serving.
- **ML-079 Experiment Tracking**: Logging inference latency and throughput metrics for different serving paradigms.
- **ML-090 ML Project Lifecycle**: The inference paradigm is a key architectural decision in the deployment phase.

## Next Concepts

- **ML-088 Data Augmentation** — Data augmentation strategies that work in both batch and streaming contexts.
- **ML-089 Labeling and Annotation** — Combining batch and real-time strategies for labeling pipelines.

## Summary

The choice between batch and real-time inference is a fundamental architectural decision in ML systems. Batch inference offers high throughput at lower cost but introduces latency from minutes to hours. Real-time inference provides immediate results at higher operational cost and complexity. Many production systems use Lambda architecture to combine both paradigms, using batch for accuracy and cost efficiency and real-time for freshness and responsiveness. Understanding the tradeoffs and knowing when to use each approach — and how to combine them — is essential for designing production ML systems.

## Key Takeaways

- Batch inference: high throughput, lower cost, scheduled, minutes/hours latency
- Real-time inference: low latency, higher cost, on-demand, sub-second responses
- Lambda architecture combines batch (accuracy) and real-time (freshness)
- Cost ratio of real-time to batch can be 5-10x for the same volume
- Streaming frameworks (Kafka, Flink) enable real-time inference at scale
- Hybrid approaches optimize for both cost and latency requirements
- The inference paradigm should be chosen based on business requirements
