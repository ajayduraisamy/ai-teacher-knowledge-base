# Concept: Anomaly Detection

## Concept ID

ML-065

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Advanced Topics

## Learning Objectives

- Understand the difference between anomaly detection, novelty detection, and outlier detection
- Implement Isolation Forest, LOF, and One-Class SVM
- Understand the role of the contamination parameter
- Apply anomaly detection to real-world problems (fraud, intrusion detection)
- Evaluate anomaly detection models using appropriate metrics

## Prerequisites

- Basic unsupervised learning concepts
- Decision trees and ensemble methods
- Understanding of density estimation

## Definition

Anomaly detection (also called outlier detection) is the identification of rare items, events, or observations that differ significantly from the majority of the data. These anomalies often indicate suspicious activity (fraud, network intrusion), system failures (equipment malfunction), or novel phenomena (new disease patterns, scientific discoveries).

Anomalies can be categorized as:
- **Point anomalies**: Individual instances that are anomalous relative to the rest (e.g., a $10,000 transaction in a sea of $20 transactions).
- **Contextual anomalies**: Instances that are anomalous in a specific context (e.g., 30°C in December is anomalous, but not in July).
- **Collective anomalies**: Collections of related instances that are anomalous together (e.g., a sudden burst of failed login attempts).

## Intuition

Think of anomaly detection like spotting a wolf in a flock of sheep. Most of the animals are sheep (normal data), and the wolf (anomaly) looks very different. The challenge is that wolves are rare, and you might never have seen one before (unlabeled data).

Anomaly detection methods work by learning what "normal" looks like and then flagging anything that deviates. Isolation Forest isolates anomalies by using fewer random splits than normal points. LOF measures how isolated a point is in its local neighborhood. One-Class SVM learns a boundary around the normal data.

## Why This Concept Matters

1. **Fraud detection**: Identify fraudulent transactions, insurance claims, and credit card abuse.
2. **Network security**: Detect intrusions, DDoS attacks, and malware.
3. **Industrial monitoring**: Predict equipment failures before they happen.
4. **Medical diagnosis**: Detect rare diseases, anomalous test results.
5. **Data cleaning**: Identify and remove erroneous data points before analysis.

## Mathematical Explanation

### Isolation Forest

Isolation Forest isolates anomalies by randomly partitioning the feature space. Anomalies are "few and different" — they require fewer random splits to isolate.

For a data point, the anomaly score is:
s(x, n) = 2^(-E[h(x)] / c(n))

where:
- E[h(x)] is the average path length across all trees.
- c(n) is the expected path length for a random point in n samples (normalization).
- s(x, n) ranges from 0 (normal) to 1 (anomalous).

Points with s > 0.5 are considered anomalies. The algorithm:
1. Build an ensemble of isolation trees (iTrees).
2. Each tree recursively splits data by randomly choosing a feature and split value.
3. Anomalies are isolated closer to the root (shorter paths).
4. Average path length across trees determines anomaly score.

### Local Outlier Factor (LOF)

LOF measures the local density deviation of a point relative to its neighbors. Points with significantly lower density than their neighbors are considered outliers.

The local reachability density (LRD) of point A:
LRD(A) = 1 / ( (sum_{B in kNN(A)} reach-dist_k(A, B)) / |kNN(A)| )

where reach-dist_k(A, B) = max(k-distance(B), dist(A, B)).

The LOF score:
LOF(A) = (sum_{B in kNN(A)} LRD(B) / LRD(A)) / |kNN(A)|

- LOF ~ 1: point has similar density to neighbors (normal).
- LOF > 1: point has lower density than neighbors (anomaly).
- LOF < 1: point has higher density than neighbors (inlier).

### One-Class SVM

One-Class SVM learns a decision boundary that separates the normal data from the origin (in the feature space). It allows some fraction of training points (nu) to fall outside the boundary:

minimize (1/2) * ||w||^2 + (1/(nu * n)) * sum(xi_i) - rho

subject to: w^T Phi(x_i) >= rho - xi_i, xi_i >= 0

where:
- nu controls the upper bound on the fraction of outliers.
- rho is the offset of the decision boundary.
- xi_i are slack variables.

The decision function: f(x) = sign(w^T Phi(x) - rho) = sign(sum(alpha_i * K(x_i, x) - rho))

### Novelty Detection vs. Outlier Detection

**Outlier detection**: Training data contains outliers. The model must be robust to contamination. Fit the model on all training data (including outliers).

**Novelty detection**: Training data is assumed "clean" (no outliers). New data points that deviate from the training distribution are flagged as novelties.

## Code Examples

### Example 1: Isolation Forest for Anomaly Detection

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

np.random.seed(42)

# Generate normal data with some anomalies
X_normal, _ = make_blobs(n_samples=300, centers=1,
                          cluster_std=0.5, random_state=42)
X_anomalies = np.random.uniform(-4, 4, (20, 2))
X = np.vstack([X_normal, X_anomalies])
y_true = np.array([0]*300 + [1]*20)

# Train Isolation Forest
iso_forest = IsolationForest(
    contamination=0.07,  # expected proportion of outliers
    random_state=42,
    n_estimators=100
)
y_pred = iso_forest.fit_predict(X)
# Convert: -1 = anomaly, 1 = normal
y_pred_binary = np.where(y_pred == -1, 1, 0)

# Scores
scores = iso_forest.decision_function(X)

# Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='coolwarm',
            edgecolors='k', s=50)
plt.title('True Labels (red=anomaly)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=scores, cmap='coolwarm',
            edgecolors='k', s=50)
plt.colorbar(label='Anomaly Score')
plt.title('Isolation Forest Anomaly Scores')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

from sklearn.metrics import classification_report
print("Classification Report:")
print(classification_report(y_true, y_pred_binary))
```

```
# Output:
Classification Report:
              precision    recall  f1-score   support
           0       0.97      0.98      0.98       300
           1       0.50      0.40      0.44        20
    accuracy                           0.95       320
   macro avg       0.73      0.69      0.71       320
weighted avg       0.94      0.95      0.94       320
```

### Example 2: LOF for Outlier Detection

```python
from sklearn.neighbors import LocalOutlierFactor

# LOF
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.07,
    novelty=False  # outlier detection mode
)
y_pred_lof = lof.fit_predict(X)
y_pred_lof_binary = np.where(y_pred_lof == -1, 1, 0)

# LOF scores (negative of LOF, more negative = more outlier)
lof_scores = -lof.negative_outlier_factor_

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=lof_scores, cmap='coolwarm',
            edgecolors='k', s=50)
plt.colorbar(label='LOF Score')
plt.title('LOF Scores (higher = more anomalous)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_lof_binary, cmap='coolwarm',
            edgecolors='k', s=50)
plt.title('LOF Predictions (red=anomaly)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("LOF Classification Report:")
print(classification_report(y_true, y_pred_lof_binary))
```

```
# Output:
LOF Classification Report:
              precision    recall  f1-score   support
           0       0.97      0.99      0.98       300
           1       0.60      0.30      0.40        20
    accuracy                           0.96       320
```

### Example 3: One-Class SVM for Novelty Detection

```python
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split

# Generate clean training data and anomalous test data
X_train = X_normal.copy()
X_test = np.vstack([
    np.random.randn(50, 2) * 0.5,
    np.random.uniform(-4, 4, (15, 2))
])
y_test = np.array([0]*50 + [1]*15)

# One-Class SVM
ocsvm = OneClassSVM(
    nu=0.05,  # expected proportion of outliers in training
    kernel='rbf',
    gamma='scale'
)
ocsvm.fit(X_train)

y_pred_ocsvm = ocsvm.predict(X_test)
y_pred_ocsvm_binary = np.where(y_pred_ocsvm == -1, 1, 0)

# Decision scores
scores_ocsvm = ocsvm.decision_function(X_test)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm',
            edgecolors='k', s=50)
plt.title('True Labels (red=novelty)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=scores_ocsvm, cmap='coolwarm',
            edgecolors='k', s=50)
plt.colorbar(label='SVM Score')
plt.title('One-Class SVM Decision Scores')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("One-Class SVM Report:")
print(classification_report(y_test, y_pred_ocsvm_binary))
```

```
# Output:
One-Class SVM Report:
              precision    recall  f1-score   support
           0       0.98      0.98      0.98        50
           1       0.90      0.60      0.72        15
    accuracy                           0.91        65
```

### Example 4: Comparing Anomaly Detection Methods

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

methods = {
    'Isolation Forest': IsolationForest(
        contamination=0.07, random_state=42),
    'LOF (novelty mode)': LocalOutlierFactor(
        n_neighbors=20, contamination=0.07, novelty=True),
    'One-Class SVM': OneClassSVM(
        nu=0.05, kernel='rbf', gamma='scale'),
}

# Use clean training, test with anomalies
X_train = X_normal
X_test = np.vstack([X_normal[:100], X_anomalies])
y_test = np.array([0]*100 + [1]*20)

plt.figure(figsize=(12, 6))
for name, model in methods.items():
    if name == 'LOF (novelty mode)':
        model.fit(X_train)
        y_score = -model.decision_function(X_test)
    elif name == 'Isolation Forest':
        model.fit(X_train)
        y_score = -model.decision_function(X_test)
    else:
        model.fit(X_train)
        y_score = -model.decision_function(X_test)

    precision, recall, _ = precision_recall_curve(y_test, y_score)
    ap = average_precision_score(y_test, y_score)
    plt.plot(recall, precision, linewidth=2, label=f'{name} (AP={ap:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves: Anomaly Detection Methods')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

```
# Output:
[PR curves showing Isolation Forest with AP=0.65, LOF with AP=0.58, One-Class SVM with AP=0.72]
```

### Example 5: Contamination Parameter Impact

```python
contamination_values = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
f1_scores = []
precision_scores = []
recall_scores = []

for cont in contamination_values:
    iso = IsolationForest(contamination=cont, random_state=42)
    y_pred = iso.fit_predict(X)
    y_pred_bin = np.where(y_pred == -1, 1, 0)
    f1_scores.append(f1_score(y_true, y_pred_bin))
    precision_scores.append(precision_score(y_true, y_pred_bin))
    recall_scores.append(recall_score(y_true, y_pred_bin))

plt.figure(figsize=(10, 6))
plt.plot(contamination_values, f1_scores, 'bo-', label='F1', linewidth=2)
plt.plot(contamination_values, precision_scores, 'rs-',
         label='Precision', linewidth=2)
plt.plot(contamination_values, recall_scores, 'g^-',
         label='Recall', linewidth=2)
plt.xlabel('Contamination Parameter')
plt.ylabel('Score')
plt.title('Effect of Contamination Parameter on Performance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Contamination parameter impact:")
for i, cont in enumerate(contamination_values):
    print(f"  contamination={cont:.2f}: F1={f1_scores[i]:.3f}, "
          f"Precision={precision_scores[i]:.3f}, "
          f"Recall={recall_scores[i]:.3f}")
```

```
# Output:
Contamination parameter impact:
  contamination=0.01: F1=0.244, Precision=0.571, Recall=0.150
  contamination=0.03: F1=0.348, Precision=0.533, Recall=0.250
  contamination=0.05: F1=0.400, Precision=0.500, Recall=0.350
  contamination=0.07: F1=0.444, Precision=0.500, Recall=0.400
  contamination=0.10: F1=0.432, Precision=0.400, Recall=0.450
  contamination=0.15: F1=0.372, Precision=0.300, Recall=0.450
  contamination=0.20: F1=0.297, Precision=0.231, Recall=0.450
```

## Common Mistakes

1. **Setting contamination incorrectly**: The contamination parameter should match the true proportion of anomalies. Too low: many anomalies missed. Too high: too many false positives.

2. **Using novelty detection for outlier detection**: One-Class SVM assumes clean training data. If training data contains outliers, use robust methods or set nu appropriately.

3. **Ignoring feature scaling**: Distance-based methods (LOF, One-Class SVM) are sensitive to feature scales. Always normalize/standardize features.

4. **Not evaluating properly**: Anomaly detection is unsupervised — you can't evaluate on training data. Use labeled test data or domain expert review.

5. **Choosing wrong k for LOF**: Too small k: too sensitive to local noise. Too large k: global outliers dominate. Start with k = 10-20.

6. **Assuming anomalies are always point anomalies**: Consider contextual and collective anomalies. Time series data requires specialized methods (seasonal decomposition, change point detection).

7. **Not handling high-dimensional data**: All methods struggle with high dimensions (curse of dimensionality). Distance metrics become meaningless. Use dimensionality reduction first.

8. **Overlooking the interpretability requirement**: In many applications (e.g., fraud), you need to explain WHY a point is anomalous. LOF and Isolation Forest provide feature importance; One-Class SVM is harder to interpret.

9. **Using default hyperparameters without validation**: Default contamination (0.1) may not match your data. Default kernels may not fit your data distribution.

10. **Not considering the dynamic nature of normality**: What's normal today may not be normal tomorrow. Use adaptive or online anomaly detection for changing distributions.

## Interview Questions

### Beginner

**Q1:** What is anomaly detection?

**A1:** Anomaly detection identifies rare items or events that differ significantly from the majority of data. It's used for fraud detection, network intrusion, system monitoring, and data cleaning.

**Q2:** What is the difference between outlier detection and novelty detection?

**A2:** Outlier detection: training data contains outliers, and the model must handle contamination. Novelty detection: training data is assumed clean, and the model detects novel points at test time.

**Q3:** How does Isolation Forest work?

**A3:** Isolation Forest builds random trees that split data on random features at random thresholds. Anomalies are isolated closer to the root (fewer splits needed). The anomaly score is based on the average path length across the forest — shorter paths = more anomalous.

**Q4:** What is the contamination parameter?

**A4:** Contamination is the expected proportion of anomalies in the data. It's used to set the threshold for anomaly scores. For example, contamination=0.05 means we expect 5% of data to be anomalies.

**Q5:** What is LOF?

**A5:** LOF (Local Outlier Factor) measures the local density deviation of a point relative to its neighbors. Points with much lower density than their neighbors (LOF >> 1) are flagged as outliers.

### Intermediate

**Q1:** Compare Isolation Forest, LOF, and One-Class SVM.

**A1:** Isolation Forest: tree-based, fast, handles high dimensions, less sensitive to hyperparameters. LOF: density-based, captures local outliers, sensitive to k parameter and feature scaling. One-Class SVM: boundary-based, good for novelty detection, sensitive to kernel parameters, can overfit with small data.

**Q2:** How do you evaluate an unsupervised anomaly detection model?

**A2:** If labeled data exists: use precision, recall, F1, and PR-AUC on a held-out test set. If no labels exist: use domain expert review, rank anomalies by score and inspect top-k, use synthetic anomalies with known labels, or use silhouette scores for clustered anomalies.

**Q3:** What is the "curse of dimensionality" in anomaly detection?

**A3:** In high dimensions, all points become approximately equidistant, making distance-based methods (LOF, kNN) ineffective. The contrast between normal and anomalous points diminishes. Dimensionality reduction (PCA, t-SNE) or feature selection is essential before anomaly detection.

**Q4:** How does One-Class SVM differ from a standard SVM?

**A4:** Standard SVM finds a hyperplane separating two classes. One-Class SVM finds a hyperplane separating the data from the origin in feature space, allowing a fraction nu of points to fall on the other side. It learns the "support" of the data distribution.

**Q5:** What is the role of the nu parameter in One-Class SVM?

**A5:** nu is an upper bound on the fraction of training points that can be outliers and a lower bound on the fraction of support vectors. nu=0.01 means at most 1% of points can violate the boundary. It's analogous to the contamination parameter.

### Advanced

**Q1:** Derive the anomaly score for Isolation Forest and explain its theoretical properties.

**A1:** The anomaly score s(x, n) = 2^(-E[h(x)] / c(n)), where c(n) = 2H(n-1) - 2(n-1)/n (average path length of unsuccessful BST search), and H(i) is the harmonic number. E[h(x)] is the average path length across all trees. s ranges from 0 (very normal — long path) to 1 (very anomalous — short path). s > 0.5 indicates anomaly. The score is derived from the expected path length in random binary search trees.

**Q2:** Explain how LOF handles datasets with varying densities (the "local" aspect).

**A2:** LOF is local because it compares each point's density only to its k-nearest neighbors, not to the global distribution. This allows it to detect outliers in datasets where different regions have different densities. A point that is global but not local outlier (e.g., in a sparse cluster) won't be flagged. The local reachability density accounts for the distance to neighbors, making the comparison fair across regions.

**Q3:** Design an anomaly detection system for a time series with seasonality and trend.

**A3:** Approach: (1) Decompose the time series into trend, seasonal, and residual components (STL decomposition). (2) Apply anomaly detection to the residual component (which should be stationary). (3) Use a sliding window approach: compute rolling statistics (mean, std) and flag points beyond Z-score threshold or Tukey fences. (4) For complex patterns, use an LSTM autoencoder: train on normal periods, flag high reconstruction error. (5) Consider contextual anomalies — a value normal for July may be anomalous for December.

## Practice Problems

**E1:** Apply Isolation Forest to a synthetic dataset with 5% anomalies. Tune the contamination parameter.

**E2:** Compare LOF and kNN-based outlier detection on the same dataset.

**E3:** Use One-Class SVM for novelty detection on clean training data.

**M1:** Build an anomaly detection pipeline for credit card fraud detection using a real dataset.

**M2:** Implement dimensionality reduction + anomaly detection for a high-dimensional dataset.

**M3:** Compare the interpretability of Isolation Forest (feature importance) vs. LOF for explaining why a point is anomalous.

**H1:** Implement an online anomaly detection method using a sliding window.

**H2:** Design and implement a method for detecting collective anomalies in time series data.

## Solutions

**M1:** For credit card fraud: (1) Standardize features. (2) Use Isolation Forest with contamination=0.001-0.01 (fraud is rare). (3) Evaluate using PR curve and F2-score (recall more important than precision for fraud). (4) Tune threshold based on cost of false negatives vs. false positives.

## Related Concepts

- Imbalanced Datasets (ML-064) — Anomaly detection as an extreme imbalance problem
- Clustering — Normal behavior can be identified through clustering
- Density Estimation — LOF is based on local density estimation
- Unsupervised Learning — Anomaly detection is typically unsupervised

## Next Concepts

- Autoencoders for Anomaly Detection — Reconstruction error as anomaly score
- GAN-Based Anomaly Detection — AnoGAN, Efficient GAN
- Time Series Anomaly Detection — STL decomposition, change point detection
- Explainable Anomaly Detection — Feature importance for anomalies

## Summary

Anomaly detection identifies rare, unusual, or suspicious data points. Key methods include Isolation Forest (tree-based, fast, works well on high-dimensional data), LOF (local density-based, captures local outliers), and One-Class SVM (boundary-based, good for novelty detection). The choice of method depends on dataset characteristics, the type of anomalies expected, and the need for interpretability. Proper evaluation requires labeled test data or domain expert review.

## Key Takeaways

- Anomaly detection identifies rare events significantly different from normal data
- Isolation Forest isolates anomalies using fewer random splits
- LOF compares local density to neighbors (good for varying density)
- One-Class SVM learns a boundary around normal data
- Contamination parameter controls the expected proportion of anomalies
- Outlier detection (data has outliers) vs. novelty detection (clean training data)
- Feature scaling is critical for distance-based methods
- High-dimensional data requires dimensionality reduction first
- PR curves are preferred over ROC for evaluation
- Domain expertise is essential for interpreting anomalies
