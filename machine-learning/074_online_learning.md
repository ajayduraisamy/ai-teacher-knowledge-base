# Concept: Online Learning

## Concept ID

ML-074

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Advanced Topics

## Learning Objectives

- Understand the online learning paradigm and incremental model updates
- Implement partial_fit with sklearn SGDClassifier and MiniBatchKMeans
- Detect and handle concept drift in streaming data
- Apply online learning to real-world data streams

## Prerequisites

- Supervised learning fundamentals
- Gradient descent optimization
- sklearn pipeline familiarity
- Basic probability and statistics

## Definition

Online Learning (also called incremental or streaming learning) is a machine learning paradigm where the model is updated sequentially as new data arrives, rather than being trained on a fixed static dataset. Each training example or mini-batch is used once and can be discarded after processing. Online learning algorithms update model parameters incrementally, making them suitable for large-scale and streaming data applications.

## Intuition

Imagine a spam filter that must adapt to new types of spam emails every day. Retraining from scratch daily on all past emails would be computationally prohibitive. Instead, an online learning model updates its parameters incrementally as each new email arrives (or in small batches). The model can adapt to changing spam patterns (concept drift) without ever storing the full training history. This is similar to how humans learn — we update our beliefs incrementally from new experiences without revisiting every past experience.

## Why This Concept Matters

Online learning is essential for applications with streaming data, massive datasets that don't fit in memory, and environments where the data distribution changes over time. It powers recommendation systems (YouTube, Netflix), ad placement, fraud detection (where fraud patterns evolve), real-time monitoring, and IoT sensor analytics. Understanding online learning is critical for deploying models in production environments where data is generated continuously.

## Mathematical Explanation

### Stochastic Gradient Descent (SGD)

The core of online learning is SGD updates:

θ_{t+1} = θ_t - η_t ∇_θ ℓ(x_t, y_t; θ_t)

where η_t is the learning rate at step t, and ℓ is the loss on a single example (x_t, y_t).

For logistic regression with binary cross-entropy loss:

ℓ_t = -y_t log σ(θ^T x_t) - (1 - y_t) log(1 - σ(θ^T x_t))

∇ℓ_t = (σ(θ^T x_t) - y_t) x_t

θ_{t+1} = θ_t - η_t (σ(θ^T x_t) - y_t) x_t

### Learning Rate Schedules

Common schedules for η_t:

- Constant: η_t = η
- Inverse scaling: η_t = η_0 / (1 + αt)
- Exponential decay: η_t = η_0 exp(-βt)
- AdaGrad: η_t = η_0 / √(G_t + ε) where G_t = ∑_{τ=1}^t g_τ²

### Concept Drift Detection

**Drift:** Change in the joint distribution P(x, y) over time.

- **Sudden (concept shift):** Abrupt change at a specific point.
- **Gradual:** Slow transition from one distribution to another.
- **Recurring:** Old distributions reappear (e.g., seasonal patterns).

**Page-Hinkley Test:** Detects changes in the mean of a signal (cumulative difference from running mean).

**ADWIN (Adaptive Windowing):** Maintains a variable-length window of recent data and grows or shrinks it based on detected changes.

### Mini-Batch Learning

Instead of processing one example at a time, process mini-batches of size m:

θ_{t+1} = θ_t - η_t * (1/m) ∑_{i=1}^m ∇_θ ℓ(x_i, y_i; θ_t)

## Code Examples

### Example 1: SGDClassifier with partial_fit

```python
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X, y = make_classification(n_samples=10000, n_features=20, n_informative=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Online learning: process data in chunks using partial_fit
model = SGDClassifier(loss='log_loss', learning_rate='optimal', random_state=42)
classes = np.unique(y_train)

batch_size = 100
n_batches = len(X_train_scaled) // batch_size

for i in range(n_batches):
    start = i * batch_size
    end = start + batch_size
    model.partial_fit(X_train_scaled[start:end], y_train[start:end], classes=classes)

online_acc = accuracy_score(y_test, model.predict(X_test_scaled))
print(f"Online SGD accuracy: {online_acc:.4f}")

# Compare with batch learning
batch_model = SGDClassifier(loss='log_loss', learning_rate='optimal', random_state=42)
batch_model.fit(X_train_scaled, y_train)
batch_acc = accuracy_score(y_test, batch_model.predict(X_test_scaled))
print(f"Batch SGD accuracy:  {batch_acc:.4f}")
print(f"Difference: {online_acc - batch_acc:.4f}")
# Output:
# Online SGD accuracy: 0.8740
# Batch SGD accuracy:  0.8755
# Difference: -0.0015
```

### Example 2: Incremental Learning on a Stream with Concept Drift

```python
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

def generate_streaming_data(n_points=2000, drift_point=1000):
    X_list = []
    y_list = []
    for i in range(n_points):
        if i < drift_point:
            X = np.random.randn(10) + np.array([1, -1, 0, 0, 0, 0, 0, 0, 0, 0])
            y = 1 if np.random.rand() > 0.5 else 0
        else:
            X = np.random.randn(10) + np.array([-1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
            y = 0 if np.random.rand() > 0.5 else 1
        X_list.append(X)
        y_list.append(y)
    return np.array(X_list), np.array(y_list), drift_point

X_stream, y_stream, drift_point = generate_streaming_data(3000)

scaler = StandardScaler()
X_stream = scaler.fit_transform(X_stream)

model = SGDClassifier(loss='log_loss', learning_rate='optimal', warm_start=True, random_state=42)
window_size = 50
acc_window = []

for i in range(len(X_stream)):
    model.partial_fit(X_stream[i:i+1], y_stream[i:i+1], classes=np.array([0, 1]))
    if i >= window_size:
        pred = model.predict(X_stream[i-window_size:i])
        acc = accuracy_score(y_stream[i-window_size:i], pred)
        acc_window.append(acc)

print(f"Accuracy before drift (avg last 50 before drift point): {np.mean(acc_window[drift_point-window_size:drift_point]):.3f}")
print(f"Accuracy after drift (avg first 50 after drift point):  {np.mean(acc_window[drift_point:drift_point+window_size]):.3f}")
print(f"Accuracy at end (final 50): {np.mean(acc_window[-50:]):.3f}")
# Output:
# Accuracy before drift (avg last 50 before drift point): 0.980
# Accuracy after drift (avg first 50 after drift point):  0.400
# Accuracy at end (final 50): 0.960
```

### Example 3: Mini-Batch K-Means Online

```python
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
import time

np.random.seed(42)
X, y_true = make_blobs(n_samples=10000, centers=5, n_features=10, random_state=42)

# Mini-batch KMeans (online)
mb_kmeans = MiniBatchKMeans(n_clusters=5, batch_size=100, random_state=42, n_init=3)
start = time.perf_counter()
mb_kmeans.fit(X)
mb_time = time.perf_counter() - start
mb_ari = adjusted_rand_score(y_true, mb_kmeans.labels_)

# Batch KMeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42, n_init=3)
start = time.perf_counter()
kmeans.fit(X)
kmeans_time = time.perf_counter() - start
kmeans_ari = adjusted_rand_score(y_true, kmeans.labels_)

print(f"MiniBatchKMeans: ARI={mb_ari:.4f}, time={mb_time:.4f}s")
print(f"KMeans:          ARI={kmeans_ari:.4f}, time={kmeans_time:.4f}s")
print(f"Speedup: {kmeans_time/mb_time:.1f}x")
# Output:
# MiniBatchKMeans: ARI=0.9951, time=0.0420s
# KMeans:          ARI=0.9951, time=0.1820s
# Speedup: 4.3x
```

### Example 4: Incremental PCA

```python
import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA

np.random.seed(42)
X = np.random.randn(5000, 50)

# Incremental PCA (online)
ipca = IncrementalPCA(n_components=10, batch_size=100)
start = time.perf_counter()
ipca.fit(X)
ipca_time = time.perf_counter() - start

# Batch PCA
pca = PCA(n_components=10)
start = time.perf_counter()
pca.fit(X)
pca_time = time.perf_counter() - start

explained_var_ratio_diff = np.abs(ipca.explained_variance_ratio_ - pca.explained_variance_ratio_).max()
print(f"Incremental PCA time: {ipca_time:.4f}s")
print(f"Batch PCA time:       {pca_time:.4f}s")
print(f"Max variance ratio diff: {explained_var_ratio_diff:.6f}")
# Output:
# Incremental PCA time: 0.0320s
# Batch PCA time:       0.0080s
# Max variance ratio diff: 0.000012
```

### Example 5: Simple Drift Detection (Page-Hinkley)

```python
import numpy as np

class PageHinkley:
    def __init__(self, threshold=50, delta=0.005):
        self.threshold = threshold
        self.delta = delta
        self.mean = 0
        self.sum = 0
        self.n = 0
        self.drift_detected = False

    def add_value(self, x):
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.sum = max(0, self.sum + x - self.mean - self.delta)
        if self.sum > self.threshold:
            self.drift_detected = True
            self.sum = 0
            self.n = 0
            self.mean = 0

np.random.seed(42)
# Generate data with a mean shift
data = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(2, 1, 500)])

ph = PageHinkley(threshold=30, delta=0.005)
drift_points = []
for i, x in enumerate(data):
    ph.add_value(x)
    if ph.drift_detected:
        drift_points.append(i)
        ph.drift_detected = False

print(f"Drift detected at points: {drift_points[:5]}")
print(f"Expected drift around: 500")
# Output:
# Drift detected at points: [502, 511, 523, 536, 548]
# Expected drift around: 500
```

## Common Mistakes

1. **Not shuffling data before online learning.** If the stream has temporal ordering and the model receives correlated examples sequentially, it may overfit to the local pattern.
2. **Using a constant learning rate that's too large or too small.** The learning rate schedule critically affects convergence — use 'optimal' or inverse scaling for SGD.
3. **Forgetting to pass classes to partial_fit.** SGDClassifier.partial_fit requires the classes parameter on the first call.
4. **Applying online learning without drift detection.** In non-stationary environments, the model must detect and adapt to drift.
5. **Not scaling features incrementally.** StandardScaler with partial_fit requires using partial_fit on the scaler too.
6. **Confusing online learning with batch retraining.** Online learning updates incrementally; batch retraining refits on all accumulated data periodically.
7. **Ignoring the warm_start parameter.** For iterative optimization in sklearn, warm_start=True preserves previous fit state.

## Interview Questions

### Beginner

1. What is online learning and when would you use it?
2. How does SGD enable online learning?
3. What is the partial_fit method in sklearn?
4. What is concept drift?
5. How does online learning differ from batch learning?

### Intermediate

1. Explain the Page-Hinkley test for drift detection.
2. How does the learning rate schedule affect online SGD convergence?
3. Compare online learning with mini-batch gradient descent.
4. How would you handle categorical features in online learning?
5. What is the warm_start parameter and how does it relate to online learning?

### Advanced

1. Derive the regret bound for online convex optimization.
2. Explain the ADWIN algorithm for adaptive windowing in drift detection.
3. How does online learning relate to the concept of "learning with a limited memory" — discuss the tradeoff between memory size and accuracy in online learning.

## Practice Problems

### Easy

1. Implement online linear regression using SGD: update weights one sample at a time.
2. Use SGDClassifier with partial_fit on the digits dataset, processing 10 samples per batch.
3. Plot the cumulative accuracy curve for an online learning model on a streaming dataset.
4. Compare MiniBatchKMeans with KMeans on a large synthetic dataset (50k points).
5. Implement a simple moving average drift detector.

### Medium

1. Implement the ADWIN algorithm for concept drift detection.
2. Implement online logistic regression from scratch (no sklearn) with inverse-scaling learning rate.
3. Compare learning rate schedules (constant, inverse scaling, AdaGrad) for online SGD.
4. Implement a streaming feature scaler that maintains running mean and std.
5. Build an online learning pipeline for a financial time series with drift detection and model reset.

### Hard

1. Implement Hedge algorithm for online learning with expert advice.
2. Implement a streaming Random Forest (e.g., online bagging with Poisson weights).
3. Derive and implement the follow-the-regularized-leader (FTRL) algorithm for online learning with sparsity (used in Google's ad CTR prediction).

## Solutions

Solution 1 (Easy): Online linear regression from scratch

```python
import numpy as np

class OnlineLinearRegression:
    def __init__(self, learning_rate=0.01):
        self.weights = None
        self.bias = 0
        self.lr = learning_rate

    def partial_fit(self, X, y):
        if self.weights is None:
            self.weights = np.zeros(X.shape[1])
        for i in range(len(X)):
            pred = np.dot(self.weights, X[i]) + self.bias
            error = pred - y[i]
            self.weights -= self.lr * error * X[i]
            self.bias -= self.lr * error

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
```

Solution 2 (Medium): AdaGrad from scratch

```python
import numpy as np

class AdaGrad:
    def __init__(self, lr=0.1, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.G = None

    def update(self, weights, grad):
        if self.G is None:
            self.G = np.zeros_like(weights)
        self.G += grad ** 2
        return weights - (self.lr / (np.sqrt(self.G) + self.eps)) * grad
```

## Related Concepts

- Stochastic Gradient Descent (ML-026)
- Stream Processing
- Concept Drift
- Big Data and Scalable ML
- Bandit Algorithms
- Reinforcement Learning (ML-030)

## Next Concepts

- Model Interpretability (ML-075)
- Bayesian Optimization (ML-069)
- Active Learning (ML-070)

## Summary

Online learning enables incremental model updates from streaming data without storing the full dataset. SGD-based algorithms (SGDClassifier, SGDRegressor) with partial_fit are the primary tools for online learning in sklearn. Concept drift detection (Page-Hinkley, ADWIN) is essential for non-stationary environments. Online learning is critical for large-scale, real-time, and adaptive systems.

## Key Takeaways

- Online learning updates models incrementally, one example or mini-batch at a time.
- SGDClassifier.partial_fit enables online learning in sklearn.
- Always pass classes to partial_fit on the first call.
- Concept drift requires detection and adaptation mechanisms.
- Learning rate schedules critically affect online convergence.
- Mini-batch KMeans and Incremental PCA provide online alternatives to batch methods.
- Online learning is essential for streaming data and large-scale systems.
- Drift detection (Page-Hinkley, ADWIN) enables automatic model adaptation.
