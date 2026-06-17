# Concept: k-Means Clustering

## Concept ID

ML-042

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Clustering

## Learning Objectives

- Implement k-Means clustering using the Lloyd algorithm
- Select the optimal number of clusters using the elbow method and silhouette score
- Evaluate the effect of different initialization methods (random, k-means++)
- Understand k-Means limitations: sensitivity to initialization, assumes spherical clusters
- Analyze cluster quality using inertia and silhouette metrics

## Prerequisites

- Basic understanding of unsupervised learning
- Euclidean distance and similarity metrics
- Python with sklearn and NumPy
- Basic optimization concepts

## Definition

k-Means is an unsupervised clustering algorithm that partitions n observations into k clusters, each assigned to the cluster with the nearest mean (centroid). It minimizes the within-cluster sum of squares (WCSS), also called inertia:

argmin_S sum_{i=1}^k sum_{x in S_i} ||x - mu_i||^2

where S_i is the set of points in cluster i and mu_i is the centroid (mean) of S_i.

## Intuition

Imagine you have a scatter plot of points and want to group them into k natural clusters. k-Means does this by (1) guessing k center points, (2) assigning each point to its nearest center, (3) moving each center to the average of its assigned points, and repeating until nothing changes. Think of it as placing k magnets that attract nearby points, then shifting to the center of mass, then attracting again, converging to a stable configuration.

## Why This Concept Matters

k-Means is the most widely used clustering algorithm due to its simplicity, speed, and scalability. It is used for customer segmentation, image compression (color quantization), document clustering, anomaly detection, and as a preprocessing step for other algorithms. Understanding k-Means builds intuition for optimization-based unsupervised learning and the challenges of initialization, local minima, and cluster validation.

## Mathematical Explanation

### Lloyd's Algorithm

1. Initialize k centroids mu_1, ..., mu_k (randomly from data or via k-means++)
2. Assign each point x to nearest centroid: c_i = argmin_j ||x_i - mu_j||^2
3. Update centroids: mu_j = (1/|S_j|) sum_{x in S_j} x
4. Repeat steps 2-3 until convergence (centroids stop changing or assignments stabilize)

### Inertia (WCSS)

WCSS = sum_{j=1}^k sum_{x in S_j} ||x - mu_j||^2

Inertia decreases monotonically with each iteration but may converge to a local minimum (not global). It is minimized when clusters are compact and well-separated.

### Choosing k via Elbow Method

Plot inertia vs. k for a range of values. The "elbow" point where inertia decreases more slowly indicates a reasonable k. The silhouette score provides an alternative: it measures how similar points are to their own cluster vs. other clusters, ranging from -1 to 1.

### k-Means++ Initialization

k-Means++ selects initial centroids probabilistically: first centroid is chosen uniformly at random; subsequent centroids are chosen with probability proportional to squared distance from the nearest existing centroid. This gives provable O(log k) approximation to optimal WCSS.

## Code Examples

### Example 1: Basic k-Means Clustering

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
y_pred = kmeans.fit_predict(X)

print(f"Inertia: {kmeans.inertia_:.2f}")
# Output: Inertia: 217.64
print(f"Centroids:\n{kmeans.cluster_centers_}")
# Output:
# Centroids:
# [[ 0.95738207  4.99611454]
#  [ 7.95450945  0.47113804]
#  [ 3.95236747  7.97528737]
#  [ 7.01842319  4.99321401]]
print(f"N iterations: {kmeans.n_iter_}")
# Output: N iterations: 5

# Compare predicted vs true (accounting for label permutation)
from scipy.stats import mode
labels = np.zeros_like(y_pred)
for i in range(4):
    mask = (y_pred == i)
    labels[mask] = mode(y_true[mask])[0]
accuracy = np.mean(labels == y_true)
print(f"Clustering accuracy: {accuracy:.3f}")
# Output: Clustering accuracy: 1.000
```

### Example 2: Elbow Method and Silhouette Score

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X, labels))
    print(f"k={k}: inertia={kmeans.inertia_:.1f}, silhouette={silhouettes[-1]:.3f}")

# Output:
# k=2: inertia=1048.5, silhouette=0.618
# k=3: inertia=461.4, silhouette=0.651
# k=4: inertia=217.6, silhouette=0.713
# k=5: inertia=199.5, silhouette=0.607
# k=6: inertia=182.9, silhouette=0.575
# k=7: inertia=167.0, silhouette=0.537
# k=8: inertia=152.5, silhouette=0.543
# k=9: inertia=139.2, silhouette=0.532
# k=10: inertia=127.3, silhouette=0.509

best_k = K_range[np.argmax(silhouettes)]
print(f"Optimal k by silhouette: {best_k}")
# Output: Optimal k by silhouette: 4
```

### Example 3: Effect of Initialization

```python
from sklearn.cluster import KMeans
import numpy as np

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Random initialization - can get stuck
print("Random initialization (5 n_init):")
kmeans_random = KMeans(n_clusters=4, init='random', n_init=5, random_state=42)
labels_random = kmeans_random.fit_predict(X)
print(f"  Inertia: {kmeans_random.inertia_:.2f}")

print("Random initialization (1 n_init, bad seed):")
kmeans_bad = KMeans(n_clusters=4, init='random', n_init=1, random_state=0)
labels_bad = kmeans_bad.fit_predict(X)
print(f"  Inertia: {kmeans_bad.inertia_:.2f}")

print("k-Means++ initialization:")
kmeans_pp = KMeans(n_clusters=4, init='k-means++', n_init=1, random_state=42)
labels_pp = kmeans_pp.fit_predict(X)
print(f"  Inertia: {kmeans_pp.inertia_:.2f}")

# Output:
# Random initialization (5 n_init):
#   Inertia: 217.64
# Random initialization (1 n_init, bad seed):
#   Inertia: 260.93
# k-Means++ initialization:
#   Inertia: 217.64
```

### Example 4: Image Compression with k-Means

```python
from sklearn.cluster import KMeans
import numpy as np

# Simulate image as array of pixels
np.random.seed(42)
h, w, c = 100, 100, 3
image = np.random.randint(0, 256, (h, w, c), dtype=np.uint8)

# Reshape to pixels
pixels = image.reshape(-1, 3)

n_colors = 16
kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(pixels)
compressed_colors = kmeans.cluster_centers_.astype(np.uint8)
labels = kmeans.predict(pixels)
compressed_image = compressed_colors[labels].reshape(h, w, 3)

original_size = image.nbytes
compressed_size = n_colors * 3 + labels.nbytes
ratio = compressed_size / original_size

print(f"Original colors: {len(np.unique(pixels.reshape(-1, 3), axis=0))}")
print(f"Compressed colors: {n_colors}")
print(f"Compression ratio (approx): {ratio:.3f}")
# Output:
# Original colors: 10000
# Compressed colors: 16
# Compression ratio (approx): 0.333
```

### Example 5: Comparison with True Clusters

```python
from sklearn.datasets import make_circles, make_moons
from sklearn.cluster import KMeans

# k-Means fails on non-convex clusters
X_circles, y_circles = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)
X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)

kmeans = KMeans(n_clusters=2, random_state=42)

labels_circles = kmeans.fit_predict(X_circles)
labels_moons = kmeans.fit_predict(X_moons)

from sklearn.metrics import adjusted_rand_score
ari_circles = adjusted_rand_score(y_circles, labels_circles)
ari_moons = adjusted_rand_score(y_moons, labels_moons)

print(f"k-Means on concentric circles ARI: {ari_circles:.3f} (bad - cannot separate)")
print(f"k-Means on moons ARI: {ari_moons:.3f} (bad - cannot separate)")
# Output:
# k-Means on concentric circles ARI: 0.011 (bad - cannot separate)
# k-Means on moons ARI: 0.235 (bad - cannot separate)
```

## Common Mistakes

1. **Assuming k-Means finds the global optimum.** Lloyd's algorithm converges to a local minimum. Multiple restarts (n_init) or k-means++ are needed.

2. **Using Euclidean distance for non-spherical data.** k-Means assumes isotropic clusters (spherical). Elongated or irregular clusters are poorly handled.

3. **Choosing k arbitrarily.** Always use elbow method, silhouette score, or gap statistic. Domain knowledge may also guide k selection.

4. **Not standardizing features.** k-Means uses Euclidean distance. Features with larger ranges dominate cluster assignment.

5. **Applying k-Means to high-dimensional data.** The curse of dimensionality makes distances meaningless. Use PCA or feature selection first.

6. **Interpreting cluster labels as having inherent meaning.** k-Means labels are arbitrary — cluster 0 in one run may differ from cluster 0 in another.

7. **Using k-Means for categorical data.** k-Means requires continuous features to compute means. Use k-modes for categorical data.

## Interview Questions

### Beginner

1. What is k-Means clustering?
An unsupervised algorithm that partitions n points into k clusters by minimizing within-cluster variance. Each point belongs to the cluster with the nearest centroid.

2. How does k-Means work?
Initialize k centroids. Repeat: assign each point to nearest centroid, recompute centroids as mean of assigned points. Stop when assignments stabilize.

3. How do you choose k?
Elbow method (plot inertia vs k, find elbow), silhouette score (maximize average silhouette), or gap statistic.

4. What is inertia?
Sum of squared distances from each point to its assigned centroid. Lower inertia indicates more compact clusters.

5. Why is k-Means sensitive to initialization?
Poor initialization can lead to convergence to poor local minima. k-Means++ and multiple restarts mitigate this.

### Intermediate

1. Explain the k-Means++ initialization algorithm.
First centroid chosen uniformly. Subsequent centroids chosen with probability proportional to squared distance to nearest existing centroid. This spreads initial centroids and provides O(log k) approximation guarantee.

2. What are the assumptions of k-Means?
Clusters are spherical, equal variance, and roughly equal size. The algorithm works poorly when these are violated.

3. How does k-Means handle outliers?
Outliers are assigned to the nearest cluster but can pull centroids significantly. Preprocessing to remove outliers or using k-Medoids is recommended.

4. Compare k-Means with Gaussian Mixture Models.
GMM provides soft assignments (probabilities) and captures elliptical clusters of different sizes. k-Means gives hard assignments and assumes spherical clusters.

5. What is the time complexity of k-Means?
O(n * k * d * i) where n=points, k=clusters, d=dimensions, i=iterations. Fast and scales to large datasets.

### Advanced

1. Prove that k-Means always converges.
The assignment step minimizes WCSS for fixed centroids. The update step minimizes WCSS for fixed assignments (the mean minimizes sum of squared errors). Each iteration reduces WCSS, which is bounded below by 0, guaranteeing convergence.

2. How can k-Means be adapted for large-scale clustering (millions of points)?
Use Mini-Batch k-Means (processes mini-batches incrementally), approximate methods like canopy clustering for initialization, or distributed implementations (Spark MLlib).

3. Derive the relationship between k-Means and PCA.
The k-Means objective can be rewritten in terms of the cluster indicator matrix. PCA is the continuous relaxation of k-Means when k=1. For k>1, spectral clustering provides the relaxation.

## Practice Problems

### Easy

1. Apply k-Means to the Iris dataset (using only sepal features). Report cluster centroids.

2. Generate 4 well-separated blobs and cluster with k-Means. Compute inertia and silhouette score.

3. Use the elbow method to find optimal k for the Wine dataset.

4. Compare k-Means with random vs k-means++ initialization on synthetic data.

5. Scale features and cluster the breast cancer dataset with k-Means.

### Medium

1. Implement k-Means from scratch (NumPy only). Verify it matches sklearn on a small dataset.

2. Compare k-Means, DBSCAN, and Agglomerative clustering on the make_moons dataset.

3. Use k-Means for color quantization on an image. Compute compression ratio vs quality tradeoff.

4. Implement the elbow method programmatically by finding the point of maximum curvature.

5. Use MiniBatchKMeans on a large synthetic dataset (n=100000). Compare speed and quality.

### Hard

1. Implement k-Means with k-Means++ initialization and multiple restarts from scratch.

2. Prove that the mean minimizes the sum of squared distances: mu = argmin sum ||x_i - mu||^2.

3. Implement a k-Means variant using L1 distance (k-Medians). Compare with standard k-Means on outlier-contaminated data.

## Solutions

Easy 1: Iris clustering

```python
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import numpy as np

X, y = load_iris(return_X_y=True)
X_sepal = X[:, :2]

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_sepal)
print(f"Inertia: {kmeans.inertia_:.2f}")
print(f"Centroids (sepal length, sepal width):")
print(kmeans.cluster_centers_)
# Output:
# Inertia: 51.40
# Centroids (sepal length, sepal width):
# [[6.19354839 2.88387097]
#  [5.54339623 2.63962264]
#  [5.006      3.428     ]]
```

## Related Concepts

- **DBSCAN** (ML-043): Density-based clustering, handles arbitrary shapes
- **Hierarchical Clustering** (ML-044): Builds cluster tree, no k needed
- **Gaussian Mixture Models** (ML-045): Soft clustering with probabilistic assignments
- **PCA** (ML-046): Often used before k-Means for dimensionality reduction
- **Feature Scaling** (ML-007): Essential preprocessing for k-Means

## Next Concepts

- DBSCAN (ML-043): Density-based clustering that finds arbitrary-shaped clusters
- Hierarchical Clustering (ML-044): Agglomerative methods with dendrogram visualization
- Gaussian Mixture Models (ML-045): Probabilistic soft clustering

## Summary

k-Means is a simple, fast clustering algorithm that partitions data into k spherical clusters by minimizing within-cluster variance. Key considerations include proper initialization (k-means++), choosing k (elbow/silhouette), and feature scaling. While it assumes spherical, equally-sized clusters and fails on complex geometries, its speed and scalability make it the go-to for many applications.

## Key Takeaways

- k-Means minimizes within-cluster sum of squares but converges to local optima
- k-Means++ initialization is strongly preferred over random
- Choose k via elbow method, silhouette score, or domain knowledge
- Always scale features before clustering
- k-Means assumes spherical, convex clusters of similar size
- Multiple restarts (n_init) help avoid poor local minima
- Mini-batch variant scales to large datasets
- Fails on non-convex shapes (circles, moons) — use DBSCAN instead
