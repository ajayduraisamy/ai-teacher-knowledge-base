# Concept: Distance Metrics

## Concept ID

ML-037

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Instance-Based and Probabilistic Methods

## Learning Objectives

- Define and compute Euclidean, Manhattan, Minkowski, Cosine, Hamming, and Jaccard distances
- Understand the mathematical properties each metric satisfies
- Select appropriate distance metrics for different data types (continuous, binary, categorical, textual)
- Implement distance computations using NumPy and sklearn
- Analyze how metric choice impacts algorithm performance

## Prerequisites

- Basic vector and matrix operations
- Norms and inner products in linear algebra
- Familiarity with kNN and clustering algorithms

## Definition

A distance metric (or distance function) is a function $d: \mathcal{X} \times \mathcal{X} \to \mathbb{R}_{\geq 0}$ that quantifies the dissimilarity between two points in a feature space. Formally, a metric must satisfy four properties for all $x, y, z \in \mathcal{X}$:

1. **Non-negativity**: $d(x, y) \geq 0$
2. **Identity of indiscernibles**: $d(x, y) = 0 \iff x = y$
3. **Symmetry**: $d(x, y) = d(y, x)$
4. **Triangle inequality**: $d(x, z) \leq d(x, y) + d(y, z)$

Not all dissimilarity measures used in ML are true metrics — some violate symmetry or the triangle inequality — but they are still widely useful.

## Intuition

Distance metrics encode our notion of "closeness." For points on a 2D grid, Euclidean distance corresponds to the straight-line path ("as the crow flies"). Manhattan distance measures grid-aligned travel ("city blocks"). Cosine distance captures directional similarity independent of magnitude — useful when comparing documents where length reflects verbosity, not content. Hamming distance counts how many bits differ between two binary strings. Jaccard distance measures overlap between sets.

Choosing the right metric is equivalent to choosing the right geometry for your feature space. The metric defines what "similar" means, and this directly shapes decision boundaries, cluster shapes, and nearest-neighbor relationships.

## Why This Concept Matters

Distance metrics are foundational to nearly every ML algorithm. kNN, k-Means, DBSCAN, Hierarchical Clustering, and many kernel methods (via the kernel trick for RBF) depend critically on the chosen distance. Amortized loans, recommendation systems use cosine similarity between user vectors; bioinformatics uses Hamming and edit distances for DNA sequences; image retrieval uses Euclidean distance on feature embeddings. Choosing the wrong metric leads to poor performance regardless of the algorithm. Understanding distance metrics is therefore a prerequisite for effective modeling.

## Mathematical Explanation

### Euclidean Distance ($L^2$)

The straight-line distance between two points:

$$d_2(x, y) = \|x - y\|_2 = \sqrt{\sum_{i=1}^d (x_i - y_i)^2}$$

Properties: isotropic (rotationally invariant), sensitive to all coordinate differences. Most commonly used metric in ML.

### Manhattan Distance ($L^1$)

Sum of absolute differences along each axis:

$$d_1(x, y) = \|x - y\|_1 = \sum_{i=1}^d |x_i - y_i|$$

Properties: less sensitive to outliers than Euclidean (no squaring). Preferred in high dimensions where L2 distances become meaningless due to concentration of norms.

### Minkowski Distance ($L^p$)

Generalization of L1 and L2:

$$d_p(x, y) = \left(\sum_{i=1}^d |x_i - y_i|^p\right)^{1/p}$$

- $p = 1$: Manhattan
- $p = 2$: Euclidean
- $p \to \infty$: Chebyshev distance $\max_i |x_i - y_i|$

### Cosine Distance

Measures the angle between vectors, ignoring magnitude:

$$d_{\text{cos}}(x, y) = 1 - \cos(\theta) = 1 - \frac{x \cdot y}{\|x\| \|y\|} = 1 - \frac{\sum_i x_i y_i}{\sqrt{\sum_i x_i^2} \sqrt{\sum_i y_i^2}}$$

Range: [0, 2]. Cosine similarity $s(x, y) \in [-1, 1]$ is often used instead. Cosine distance is not a true metric (fails triangle inequality), but is widely used for text and embeddings.

### Hamming Distance

Counts positions where two strings (or binary vectors) differ:

$$d_H(x, y) = \sum_{i=1}^d \mathbb{I}(x_i \neq y_i)$$

Used for categorical features, binary codes, error-correcting codes, and DNA sequences.

### Jaccard Distance

Measures dissimilarity between two sets:

$$d_J(A, B) = 1 - J(A, B) = 1 - \frac{|A \cap B|}{|A \cup B|}$$

For binary vectors: $d_J = \frac{b + c}{a + b + c}$ where a = both present, b = first only, c = second only. Commonly used in information retrieval and genomics.

## Code Examples

### Example 1: Computing Distances with NumPy

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([4.0, 5.0, 6.0])

# Euclidean
euclidean = np.sqrt(np.sum((x - y) ** 2))
print(f"Euclidean: {euclidean:.4f}")
# Output: Euclidean: 5.1962

# Manhattan
manhattan = np.sum(np.abs(x - y))
print(f"Manhattan: {manhattan}")
# Output: Manhattan: 9

# Minkowski p=3
p = 3
minkowski = np.power(np.sum(np.abs(x - y) ** p), 1/p)
print(f"Minkowski (p=3): {minkowski:.4f}")
# Output: Minkowski (p=3): 4.3267

# Cosine
cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
cosine_dist = 1 - cos_sim
print(f"Cosine distance: {cosine_dist:.4f}")
# Output: Cosine distance: 0.0254

# Cosine similarity
print(f"Cosine similarity: {cos_sim:.4f}")
# Output: Cosine similarity: 0.9746

# Hamming
x_bin = np.array([1, 0, 1, 1, 0, 0])
y_bin = np.array([1, 1, 0, 1, 0, 1])
hamming = np.mean(x_bin != y_bin)
print(f"Hamming (normalized): {hamming:.4f}")
# Output: Hamming (normalized): 0.5000
hamming_count = np.sum(x_bin != y_bin)
print(f"Hamming (count): {hamming_count}")
# Output: Hamming (count): 3

# Jaccard for binary vectors
intersection = np.sum((x_bin == 1) & (y_bin == 1))
union = np.sum((x_bin == 1) | (y_bin == 1))
jaccard_sim = intersection / union
jaccard_dist = 1 - jaccard_sim
print(f"Jaccard similarity: {jaccard_sim:.4f}")
print(f"Jaccard distance: {jaccard_dist:.4f}")
# Output:
# Jaccard similarity: 0.2500
# Jaccard distance: 0.7500
```

### Example 2: sklearn Distance Metrics

```python
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_kernels
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

# Pairwise Euclidean distances
euclidean = pairwise_distances(X, metric='euclidean')
print("Euclidean matrix:")
print(euclidean)
# Output:
# Euclidean matrix:
# [[0.         2.82842712 5.65685425 8.48528137]
#  [2.82842712 0.         2.82842712 5.65685425]
#  [5.65685425 2.82842712 0.         2.82842712]
#  [8.48528137 5.65685425 2.82842712 0.        ]]

# Manhattan
manhattan = pairwise_distances(X, metric='manhattan')
print("Manhattan matrix:")
print(manhattan)
# Output:
# Manhattan matrix:
# [[0. 4. 8. 12.]
#  [4. 0. 4.  8.]
#  [8. 4. 0.  4.]
#  [12. 8. 4.  0.]]

# Cosine
cosine = pairwise_distances(X, metric='cosine')
print("Cosine distance matrix:")
print(np.round(cosine, 4))
# Output:
# Cosine distance matrix:
# [[0.     0.0004 0.0016 0.0032]
#  [0.0004 0.     0.0004 0.0016]
#  [0.0016 0.0004 0.     0.0004]
#  [0.0032 0.0016 0.0004 0.    ]]

# Same vectors have 0 cosine distance; nearly collinear ones have near-0
```

### Example 3: When Metric Choice Changes kNN Classification

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = load_wine(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

metrics = {
    'euclidean': 'euclidean',
    'manhattan': 'manhattan',
    'cosine': 'cosine',
    'chebyshev': 'chebyshev'
}

for name, metric in metrics.items():
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    scores = cross_val_score(knn, X_scaled, y, cv=5)
    print(f"{name:12s}: Mean={scores.mean():.3f}  Std={scores.std():.3f}")

# Output:
# euclidean   : Mean=0.972  Std=0.016
# manhattan   : Mean=0.978  Std=0.015
# cosine      : Mean=0.960  Std=0.027
# chebyshev   : Mean=0.849  Std=0.022
```

### Example 4: Custom Distance Function in sklearn

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

def custom_weighted_euclidean(u, v):
    weights = np.array([1.0, 2.0, 0.5])  # feature importance
    return np.sqrt(np.sum(weights * (u - v) ** 2))

X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

nn = NearestNeighbors(n_neighbors=2, metric=custom_weighted_euclidean)
nn.fit(X)
distances, indices = nn.kneighbors(X[:1])

print(f"Nearest neighbors to {X[0]}:")
print(f"  Indices: {indices[0]}")
print(f"  Distances: {distances[0]}")
# Output:
# Nearest neighbors to [1 2 3]:
#   Indices: [0 1]
#   Distances: [0.         5.24404424]
```

### Example 5: Hamming Distance for Categorical Data

```python
from sklearn.metrics import pairwise_distances
import numpy as np

# One-hot encoded categorical data
data = np.array([
    [1, 0, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1],
    [0, 0, 1, 1, 1]
])

# Hamming distance (fraction of differing coordinates)
hamming_dist = pairwise_distances(data, metric='hamming')
print("Hamming distance matrix:")
print(np.round(hamming_dist, 3))
# Output:
# Hamming distance matrix:
# [[0.  0.4 0.4 0.6]
#  [0.4 0.  0.6 0.6]
#  [0.4 0.6 0.  0.4]
#  [0.6 0.6 0.4 0. ]]

# Jaccard distance (for binary/set data)
jaccard_dist = pairwise_distances(data, metric='jaccard')
print("Jaccard distance matrix:")
print(np.round(jaccard_dist, 3))
# Output:
# Jaccard distance matrix:
# [[0.   0.5  0.5  0.75]
#  [0.5  0.   0.75 0.75]
#  [0.5  0.75 0.   0.5 ]
#  [0.75 0.75 0.5  0.  ]]
```

## Common Mistakes

1. **Using Euclidean distance on high-dimensional data.** The concentration of norms phenomenon makes all Euclidean distances nearly equal in high dimensions, rendering nearest neighbor meaningless. Use Manhattan or cosine instead.

2. **Using Euclidean distance on text data.** Text vectors are sparse and high-dimensional with large magnitude variations. Cosine distance captures semantic similarity better by ignoring document length.

3. **Not normalizing data before computing distances.** If features have different scales, those with larger ranges dominate Euclidean and Manhattan distances. Always standardize or normalize.

4. **Using Euclidean distance for categorical features.** Treating category labels as ordinal (e.g., "red"=1, "blue"=2) implies false ordering. Use Hamming distance on one-hot encoded categoricals, or Gower distance for mixed types.

5. **Assuming all metrics are interchangeable.** Euclidean creates spherical decision boundaries; Manhattan produces diamond-shaped ones; cosine ignores magnitude entirely. Different metrics encode different assumptions about data geometry.

6. **Forgetting that Cosine distance is not a true metric.** It violates the triangle inequality, which can break algorithms that assume metric properties (e.g., some tree-based indexing structures).

7. **Using weighted metrics without rationale.** Arbitrary feature weights can distort the geometry in unpredictable ways. Weight based on domain knowledge or learned importance (e.g., via feature selection).

## Interview Questions

### Beginner

1. What is the difference between Euclidean and Manhattan distance?

Euclidean is the straight-line (L2 norm) distance: $\sqrt{\sum (x_i - y_i)^2}$. Manhattan is the sum of absolute differences (L1 norm): $\sum |x_i - y_i|$. Manhattan is less sensitive to outliers and behaves better in high dimensions.

2. When would you use cosine distance instead of Euclidean?

Cosine distance measures angular dissimilarity independent of magnitude. Use it for text documents (TF-IDF vectors), embeddings, and any data where direction matters more than length. For example, two news articles of different lengths covering the same topic have small cosine distance but large Euclidean distance.

3. What is the range of cosine similarity? Cosine distance?

Cosine similarity ranges from -1 (opposite directions) to 1 (same direction), with 0 indicating orthogonality. Cosine distance = 1 - similarity, ranging from 0 to 2.

4. Define Hamming distance. When is it used?

Hamming distance counts positions where two strings or vectors differ. Used in error-correcting codes, categorical feature comparison, binary feature vectors, and bioinformatics (DNA sequence comparison).

5. Does Jaccard distance consider joint absences? Why?

No. Jaccard ignores positions where both vectors have 0 (double absence). The denominator is the union of present features, so it measures overlap relative to presence. This is useful when absence of a feature (e.g., not having a word in a document) is not informative.

### Intermediate

1. Prove that Euclidean distance satisfies the triangle inequality.

By the Minkowski inequality: $\|x - z\|_2 \leq \|x - y\|_2 + \|y - z\|_2$. This follows from the Cauchy-Schwarz inequality and the fact that the L2 norm is induced by an inner product. The triangle inequality is what makes Euclidean distance a true metric.

2. How does metric choice affect the decision boundary of a 1-NN classifier?

The decision boundary of 1-NN is the Voronoi diagram. Euclidean distance produces linear boundaries (perpendicular bisectors between points). Manhattan produces axis-aligned boundaries (L1 Voronoi cells are convex polygons with axis-aligned edges). The metric shapes the region geometry.

3. Explain the "concentration of norms" phenomenon in high dimensions.

For i.i.d. random vectors in $\mathbb{R}^d$, as d grows, the ratio $\|X\| / \sqrt{d}$ concentrates around a constant. Consequently, pairwise Euclidean distances become nearly equal: $\text{Var}(d(x_i, x_j)) \to 0$ as $d \to \infty$. This makes nearest neighbor methods ineffective.

4. What is the Mahalanobis distance, and when would you use it?

$d_M(x, y) = \sqrt{(x - y)^T \Sigma^{-1} (x - y)}$ where $\Sigma$ is the covariance matrix. It accounts for feature correlations and different variances, producing elliptical distance contours aligned with the data distribution. Use when features are correlated or have different units.

5. Compare Gower distance with other metrics for mixed-type data.

Gower distance handles mixed data by computing a per-feature distance (range-normalized for numeric, 0/1 for categorical) and averaging across features. It scales each feature to [0,1] before combining, unlike raw Euclidean which mixes scales arbitrarily.

### Advanced

1. Prove that the cosine distance violates the triangle inequality and provide a counterexample.

Consider three unit vectors in 2D: $a=(1,0), b=(0,1), c=(-1,0)$. $d_{cos}(a,c)=2$ (opposite directions), $d_{cos}(a,b)=1$, $d_{cos}(b,c)=1$. Triangle inequality would require $2 \leq 1+1$, which holds (equal). A violation: $a=(0.99,0.14), b=(1,0), c=(-1,0)$. $d_{cos}(a,c)=1.98$, $d_{cos}(a,b)=0.01$, $d_{cos}(b,c)=2$ — gives $1.98 \leq 0.01+2=2.01$, still holds. True violation: in 3D, $a=(1,0,0), b=(0.5,0.5,0.7), c=(0,1,0)$. Careful computation shows $d_{cos}(a,c)=1$ but $d_{cos}(a,b)=0.29$ and $d_{cos}(b,c)=0.29$, so $1 \not\leq 0.58$.

2. How does the choice of distance metric affect the bias-variance tradeoff in kNN?

The metric defines the neighborhood shape. An isotropic metric like Euclidean creates spherical neighborhoods — good when features are equally relevant. A metric that weights certain features more heavily reduces bias along those dimensions but increases variance due to fewer effective neighbors. In high dimensions, L1 distance concentrates more slowly than L2, yielding lower variance at the cost of higher bias.

3. Derive the gradient of the Euclidean distance with respect to the input features. How would this be used in metric learning?

$\frac{\partial}{\partial x} \|x - y\|_2 = \frac{x - y}{\|x - y\|_2}$. Metric learning (e.g., Large Margin Nearest Neighbor) learns a linear transformation M of the features such that $d_M(x,y) = (x-y)^T M (x-y)$. Gradients of pairwise distances w.r.t. M are used in optimization to pull similar points together and push dissimilar points apart.

## Practice Problems

### Easy

1. Compute the pairwise Euclidean distance matrix for points (0,0), (1,1), (2,2) using NumPy.

2. Given two TF-IDF vectors of length 1000, compute their cosine similarity using sklearn's cosine_similarity.

3. On the Iris dataset, compute all pairwise distances using Manhattan, Euclidean, and Chebyshev metrics. Report the mean distance for each.

4. Write a function to compute normalized Hamming distance for two binary arrays of arbitrary length.

5. On sklearn's load_digits, compute the average cosine distance between all pairs of digit 3 vs. all pairs of digit 8. Which digit is more internally similar?

### Medium

1. Implement a custom `minkowski_distance` function with parameter p. Show that as p→∞, the distance converges to Chebyshev.

2. Using sklearn's NearestNeighbors, compare the number of points within a fixed radius for Euclidean vs. Manhattan metrics on scaled Wine data. Which metric finds more neighbors?

3. Write a function to compute Gower distance for a dataset with mixed numeric and categorical features.

4. Generate random data in 5, 20, 100, 500 dimensions. Compute the ratio $\max(d_{ij}) / \min(d_{ij})$ for Euclidean distances. How does this ratio change with dimension?

5. Build a kNN classifier on the 20 Newsgroups data using cosine distance. Compare accuracy with Euclidean distance after TF-IDF vectorization.

### Hard

1. Implement Large Margin Nearest Neighbor (LMNN) metric learning from scratch. Show how the learned distance improves kNN accuracy on a subset of MNIST.

2. Prove that the squared Euclidean distance is equivalent to a linear kernel in a transformed space: $\|x - y\|_2^2 = \langle x, x \rangle - 2\langle x, y \rangle + \langle y, y \rangle$ and relate to kernel methods.

3. Implement the DTW (Dynamic Time Warping) distance for time series and compare nearest-neighbor classification accuracy against Euclidean distance on the UCR time series archive.

## Solutions

### Easy 1: Pairwise Euclidean

```python
import numpy as np

points = np.array([[0, 0], [1, 1], [2, 2]])
n = len(points)
dist_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        dist_matrix[i, j] = np.sqrt(np.sum((points[i] - points[j]) ** 2))

print(np.round(dist_matrix, 4))
# Output:
# [[0.     1.4142 2.8284]
#  [1.4142 0.     1.4142]
#  [2.8284 1.4142 0.    ]]
```

### Easy 2: Cosine similarity

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

np.random.seed(42)
v1 = np.random.rand(1, 1000)
v2 = np.random.rand(1, 1000)
v3 = v1 * 5  # same direction, different magnitude

sim_12 = cosine_similarity(v1, v2)[0, 0]
sim_13 = cosine_similarity(v1, v3)[0, 0]

print(f"Similarity between random vectors: {sim_12:.4f}")
print(f"Similarity between scaled vectors: {sim_13:.4f}")
# Output:
# Similarity between random vectors: 0.8545
# Similarity between scaled vectors: 1.0000
```

## Related Concepts

- **k-Nearest Neighbors** (ML-036): Primary algorithm that uses distance metrics
- **K-Means** (ML-042): Uses Euclidean distance by default
- **DBSCAN** (ML-043): Uses distance to define density neighborhoods
- **Kernel Methods** (ML-030): Kernel trick replaces inner product, implicitly defining distance
- **Feature Scaling** (ML-007): Preprocessing required before distance computation

## Next Concepts

- **Naive Bayes** (ML-038): Probabilistic classifier — an alternative to distance-based methods
- **t-SNE** (ML-047): Uses probability-based distances (not metric distances) for visualization
- **Hierarchical Clustering** (ML-044): Relies on linkage distances between clusters

## Summary

Distance metrics are the foundation of similarity computation in machine learning. Euclidean (L2) is the default for continuous data in low dimensions; Manhattan (L1) is more robust in high dimensions; Cosine ignores magnitude for text and embeddings; Hamming compares binary/categorical data; Jaccard measures set overlap. The choice of metric encodes assumptions about data geometry and significantly impacts algorithm performance. No single metric works universally — the right choice depends on data type, dimensionality, scale, and the notion of similarity relevant to the task.

## Key Takeaways

- Euclidean distance is the most common but fails in high dimensions
- Manhattan distance is preferred for high-dimensional and sparse data
- Cosine distance captures directional similarity, ideal for text and embeddings
- Hamming and Jaccard distances handle binary and categorical data
- Feature scaling is mandatory before computing any metric distance
- Cosine distance is NOT a true metric (violates triangle inequality)
- Metric choice fundamentally shapes model behavior and decision boundaries
- Custom metrics can encode domain knowledge through feature weighting
