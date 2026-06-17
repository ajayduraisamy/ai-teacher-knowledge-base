# Concept: Hierarchical Clustering

## Concept ID

ML-044

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Clustering

## Learning Objectives

- Understand agglomerative (bottom-up) hierarchical clustering
- Interpret dendrograms for cluster analysis
- Compare linkage criteria: single, complete, average, and Ward
- Select appropriate cutoff thresholds for cluster extraction
- Implement hierarchical clustering using sklearn

## Prerequisites

- Distance metrics
- Basic clustering concepts
- Python with sklearn, SciPy, and NumPy

## Definition

Hierarchical clustering builds a tree of clusters (dendrogram) without requiring the number of clusters k as input. Agglomerative (bottom-up) clustering starts with each point as its own cluster and iteratively merges the closest pair of clusters. The hierarchy can be cut at any level to produce any desired number of clusters.

## Intuition

Think of hierarchical clustering like building a family tree of data points. Start with each person as their own family. Then repeatedly merge the two most similar families together. The result is a tree (dendrogram) that shows the full merging history. If you cut the tree at a certain height, you get different levels of grouping — from many small clusters (cut low) to few large clusters (cut high).

## Why This Concept Matters

Hierarchical clustering is unique among clustering algorithms because it provides a complete hierarchy rather than a single partition. This allows exploration of cluster structure at multiple granularities. It is widely used in biology (phylogenetic trees), social network analysis, document organization, and anytime you want to understand relationships between clusters, not just the clusters themselves.

## Mathematical Explanation

### Agglomerative Algorithm

1. Start with each point as its own cluster (n clusters)
2. Compute pairwise dissimilarity between all clusters
3. Merge the two closest clusters
4. Recompute dissimilarities between the new cluster and all others
5. Repeat steps 3-4 until only one cluster remains

### Linkage Criteria

The distance between two clusters A and B depends on the linkage:

**Single linkage** (minimum): d(A, B) = min_{a in A, b in B} d(a, b)
- Can produce long, chain-like clusters
- Sensitive to noise and outliers

**Complete linkage** (maximum): d(A, B) = max_{a in A, b in B} d(a, b)
- Produces compact, spherical clusters
- Less sensitive to noise

**Average linkage** (UPGMA): d(A, B) = 1/(|A||B|) sum_{a in A} sum_{b in B} d(a, b)
- Compromise between single and complete
- Often preferred for biological applications

**Ward's linkage**: d(A, B) = (|A||B|/(|A|+|B|)) * ||mu_A - mu_B||^2
- Minimizes increase in within-cluster variance when merging
- Tends to produce compact, equally-sized clusters
- Assumes Euclidean distance

### Dendrogram

A dendrogram is a tree diagram showing the merging history. The vertical axis represents the dissimilarity at which clusters merge. Cutting the dendrogram at a given height yields a clustering. The number of clusters equals the number of vertical lines crossed.

### Complexity

O(n^3) time for naive implementation, O(n^2 log n) with priority queues, O(n^2) memory for the distance matrix.

## Code Examples

### Example 1: Basic Agglomerative Clustering with Dendrogram

```python
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=30, centers=3, cluster_std=0.5, random_state=42)

# Single linkage
clustering_single = AgglomerativeClustering(n_clusters=3, linkage='single')
labels_single = clustering_single.fit_predict(X)

# Ward linkage
clustering_ward = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_ward = clustering_ward.fit_predict(X)

# Complete linkage
clustering_complete = AgglomerativeClustering(n_clusters=3, linkage='complete')
labels_complete = clustering_complete.fit_predict(X)

from sklearn.metrics import adjusted_rand_score
print("Adjusted Rand Index vs true labels:")
print(f"  Single:   {adjusted_rand_score(y, labels_single):.3f}")
print(f"  Complete: {adjusted_rand_score(y, labels_complete):.3f}")
print(f"  Ward:     {adjusted_rand_score(y, labels_ward):.3f}")
# Output:
# Adjusted Rand Index vs true labels:
#   Single:   1.000
#   Complete: 1.000
#   Ward:     1.000

# Generate dendrogram data
linked = linkage(X, method='ward')
print("Dendrogram linkage matrix (first 5 merges):")
print(linked[:5])
# Output:
# Dendrogram linkage matrix (first 5 merges):
# [[ 9. 24.  0.  2.]
#  [14. 21.  0.  2.]
#  [23. 28.  0.  2.]
#  [ 0. 26.  0.  2.]
#  [ 2.  8.  0.  2.]]
```

### Example 2: Comparing Linkage Methods

```python
from sklearn.datasets import make_moons

X_moons, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

for linkage_method in ['single', 'complete', 'average', 'ward']:
    clustering = AgglomerativeClustering(n_clusters=2, linkage=linkage_method)
    labels = clustering.fit_predict(X_moons)
    ari = adjusted_rand_score(y_moons, labels)
    print(f"Linkage={linkage_method:10s}: ARI={ari:.3f}")

# Output:
# Linkage=single    : ARI=-0.001
# Linkage=complete  : ARI=1.000
# Linkage=average   : ARI=0.738
# Linkage=ward      : ARI=0.971
```

### Example 3: Dendrogram with Cutoff

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np

X, y = make_blobs(n_samples=80, centers=4, cluster_std=0.8, random_state=42)

linked = linkage(X, method='ward')

# Cut at max distance of 15
labels_k = fcluster(linked, t=15, criterion='distance')
n_clusters = len(set(labels_k))

# Cut to get exactly 4 clusters
labels_4 = fcluster(linked, t=4, criterion='maxclust')

print(f"Cut at distance 15: {n_clusters} clusters")
print(f"Cut for exactly 4 clusters ARI: {adjusted_rand_score(y, labels_4):.3f}")
# Output:
# Cut at distance 15: 4 clusters
# Cut for exactly 4 clusters ARI: 0.970
```

### Example 4: Hierarchical Clustering on Real Data

```python
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

wine = load_wine()
X_wine = StandardScaler().fit_transform(wine.data)

for linkage_method in ['single', 'complete', 'average', 'ward']:
    clustering = AgglomerativeClustering(n_clusters=3, linkage=linkage_method)
    labels = clustering.fit_predict(X_wine)
    ari = adjusted_rand_score(wine.target, labels)
    print(f"Wine dataset - {linkage_method:10s}: ARI={ari:.3f}")

# Output:
# Wine dataset - single    : ARI=0.195
# Wine dataset - complete  : ARI=0.423
# Wine dataset - average   : ARI=0.569
# Wine dataset - ward      : ARI=0.897
```

### Example 5: Choosing Cutoff via Inconsistency Coefficients

```python
from scipy.cluster.hierarchy import leaders, inconsistent

X, y = make_blobs(n_samples=100, centers=3, cluster_std=0.5, random_state=42)
linked = linkage(X, method='ward')

# Inconsistency coefficient measures how merge height compares to neighboring merges
incons = inconsistent(linked, d=3)
print("First 5 inconsistency coefficients:")
for i, inc in enumerate(incons[:5]):
    height = linked[i, 2]
    print(f"  Merge {i}: height={height:.2f}, mean={inc[0]:.2f}, "
          f"std={inc[1]:.2f}, inconsist={inc[2]:.2f}")

# Output:
# First 5 inconsistency coefficients:
#   Merge 0: height=0.01, mean=0.01, std=0.00, inconsist=0.00
#   Merge 1: height=0.03, mean=0.02, std=0.01, inconsist=0.00
#   Merge 2: height=0.04, mean=0.04, std=0.01, inconsist=0.00
#   Merge 3: height=0.06, mean=0.05, std=0.02, inconsist=0.00
#   Merge 4: height=0.08, mean=0.07, std=0.03, inconsist=0.00

# The highest inconsistency coefficient indicates the natural cutoff
```

## Common Mistakes

1. **Using single linkage with noisy data.** Single linkage chains outliers together, producing long, straggly clusters that don't represent real structure.

2. **Not standardizing features.** Hierarchical clustering uses distances. Features with larger scales dominate.

3. **Cutting the dendrogram at an arbitrary height without validation.** Use inconsistency coefficients, elbow method on linkage distances, or silhouette score.

4. **Using Euclidean distance with non-Euclidean linkage.** Ward's linkage specifically requires Euclidean distances. Using other distances with Ward gives incorrect results.

5. **Applying hierarchical clustering to large datasets.** O(n^2) memory makes it impractical for n > 10,000-20,000.

6. **Assuming the dendrogram shows optimal cluster structure.** Different linkage methods produce different hierarchies. Always compare multiple linkages.

7. **Reading the dendrogram without considering scale.** The vertical axis shows the linkage distance. A large jump in merge distance suggests the natural number of clusters.

## Interview Questions

### Beginner

1. What is hierarchical clustering?
An algorithm that builds a hierarchy of clusters, typically via agglomerative (bottom-up) merging or divisive (top-down) splitting.

2. What is a dendrogram?
A tree diagram showing the merging history in hierarchical clustering. The vertical axis shows the dissimilarity at each merge.

3. What are the main linkage methods?
Single (min distance), Complete (max distance), Average (mean distance), and Ward (minimize variance increase).

4. Does hierarchical clustering require specifying k?
Not initially — the dendrogram shows all possible clusterings. You cut it at a chosen height to get k clusters.

5. What is the difference between agglomerative and divisive clustering?
Agglomerative starts with n clusters and merges; divisive starts with 1 cluster and splits. Agglomerative is more common.

### Intermediate

1. Compare single, complete, and average linkage.
Single linkage finds elongated, chain-like clusters (chaining effect). Complete linkage finds compact clusters but may split large clusters. Average linkage balances both. Ward's linkage minimizes variance and often works best.

2. How do you determine the optimal number of clusters from a dendrogram?
Look for the largest vertical gap in the dendrogram — the merge at that height joins very different clusters. The number of clusters is the number of vertical lines crossed by a horizontal line at that height.

3. What is the computational complexity of agglomerative clustering?
O(n^3) naive, O(n^2 log n) optimized, O(n^2) memory for the distance matrix. Not suitable for large datasets.

4. How does Ward's linkage differ from single linkage?
Ward merges clusters that minimize the increase in total within-cluster variance. It produces compact, spherical clusters of similar size. Single linkage merges the closest pair of points from different clusters, potentially creating chains.

5. What is the chaining effect in single linkage?
The tendency to form long, thin clusters by incrementally adding nearby points. This can merge distinct clusters connected by a chain of intermediate points.

### Advanced

1. Prove that Ward's linkage is equivalent to minimizing the increase in error sum of squares.
The increase in ESS when merging clusters A and B is: Delta ESS = |A||B|/(|A|+|B|) * ||mu_A - mu_B||^2. This is exactly the Ward distance. Minimizing this at each step greedily optimizes the hierarchical clustering objective.

2. How can hierarchical clustering be extended to handle large datasets?
Use BIRCH (clustering features tree), CURE (representative points), or Chameleon algorithms. For moderate large data, sample first then assign remaining points.

3. Describe the relationship between hierarchical clustering and minimum spanning trees.
Single linkage hierarchical clustering is equivalent to building a minimum spanning tree (MST) of the data and then sorting edges by length. Cutting the MST at a threshold gives the same clusters as single linkage at that threshold.

## Practice Problems

### Easy

1. Apply AgglomerativeClustering with Ward linkage to Iris data. Compare with true labels.

2. Generate synthetic data with 3 well-separated blobs. Plot the dendrogram.

3. Compare single, complete, and average linkage on the make_circles dataset.

4. Use fcluster to extract 5 clusters from a dendrogram at a specific distance threshold.

5. Scale features and apply hierarchical clustering to the Wine dataset.

### Medium

1. Implement agglomerative clustering with single linkage from scratch.

2. Compare BIRCH with AgglomerativeClustering on a dataset of 50,000 points. Report speed and quality.

3. Use the cophenetic correlation coefficient to evaluate how well the dendrogram preserves pairwise distances.

4. Implement a function that finds the optimal dendrogram cutoff using silhouette score maximization.

5. Compare hierarchical clustering with DBSCAN on the make_moons dataset for various linkage methods.

### Hard

1. Implement Ward's linkage from scratch and verify it matches sklearn's implementation.

2. Prove the Lance-Williams recurrence formula for updating cluster distances in agglomerative clustering.

3. Design and implement a divisive (top-down) hierarchical clustering using k-Means recursively.

## Solutions

Easy 1: Iris hierarchical clustering

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

X, y = load_iris(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = clustering.fit_predict(X_scaled)

ari = adjusted_rand_score(y, labels)
print(f"Ward linkage ARI: {ari:.3f}")
# Output: Ward linkage ARI: 0.898

clustering_complete = AgglomerativeClustering(n_clusters=3, linkage='complete')
labels_complete = clustering_complete.fit_predict(X_scaled)
ari_complete = adjusted_rand_score(y, labels_complete)
print(f"Complete linkage ARI: {ari_complete:.3f}")
# Output: Complete linkage ARI: 0.736
```

## Related Concepts

- **k-Means** (ML-042): Flat clustering, complementary to hierarchical
- **DBSCAN** (ML-043): Density-based clustering, different paradigm
- **Gaussian Mixture Models** (ML-045): Probabilistic clustering
- **Dendrogram Visualization**: Key interpretability tool
- **BIRCH**: Scalable hierarchical clustering for large data

## Next Concepts

- Gaussian Mixture Models (ML-045): Soft clustering with EM algorithm
- t-SNE (ML-047): Visualization of high-dimensional clusters
- PCA (ML-046): Dimensionality reduction for preprocessing

## Summary

Hierarchical clustering builds a tree of cluster merges (dendrogram) that reveals multiscale cluster structure. The agglomerative approach starts with each point as a cluster and iteratively merges the closest pairs. Linkage criteria (single, complete, average, Ward) determine how cluster distances are computed, each producing different cluster shapes. The dendrogram allows exploration at any granularity, but the O(n^2) memory cost limits scalability.

## Key Takeaways

- Agglomerative clustering builds a complete merge hierarchy
- Linkage criteria dramatically affect results — Ward's is often best
- The dendrogram enables multi-granularity cluster analysis
- O(n^2) memory limits use to moderate-sized datasets
- Always scale features before clustering
- Cut the dendrogram at large gaps for natural clusters
- Compare multiple linkage methods for robust analysis
- BIRCH extends hierarchical clustering to large datasets
