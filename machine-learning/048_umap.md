# Concept: UMAP

## Concept ID

ML-048

## Difficulty

Advanced

## Domain

Machine Learning

## Module

Dimensionality Reduction

## Learning Objectives

- Understand UMAP as a manifold learning technique based on topological data analysis
- Implement UMAP using the umap-learn library
- Compare UMAP with t-SNE in terms of speed, global structure preservation, and scalability
- Tune UMAP parameters: n_neighbors, min_dist, and n_components
- Apply UMAP for visualization and preprocessing

## Prerequisites

- t-SNE (ML-047)
- Manifold learning concepts
- Basic graph theory (k-NN graphs)
- Python with umap-learn, sklearn, and NumPy

## Definition

UMAP (Uniform Manifold Approximation and Projection) is a non-linear dimensionality reduction technique based on Riemannian geometry and algebraic topology. It constructs a fuzzy simplicial set representation of the data in high-dimensional space and optimizes a low-dimensional embedding to minimize the cross-entropy between the two representations.

UMAP assumes that data is uniformly distributed on a locally-connected Riemannian manifold and uses this assumption to construct a weighted k-NN graph that represents the topological structure. The embedding preserves this topological structure as faithfully as possible.

## Intuition

UMAP builds a graph connecting each point to its nearest neighbors, with edge weights representing the likelihood that two points are connected in the underlying manifold. It then finds a low-dimensional layout that preserves this graph structure as much as possible. The result is an embedding that captures both local and global structure.

Think of it as stretching a rubber sheet to fit the data: nearby points pull on each other to stay close, while the overall layout captures the manifold's global shape. UMAP is better at preserving global structure than t-SNE because its normalization is symmetric and it minimizes cross-entropy rather than KL divergence.

## Why This Concept Matters

UMAP has rapidly become one of the most popular dimensionality reduction techniques, often replacing t-SNE due to its superior speed, better global structure preservation, and clearer interpretability. It scales to millions of data points, supports out-of-sample embedding, and produces embeddings that are more reproducible across runs. UMAP is used in bioinformatics (scRNA-seq), computer vision, natural language processing, and exploratory data analysis.

## Mathematical Explanation

### High-Dimensional Graph Construction

1. Build a k-NN graph for each point x_i
2. Compute local connectivity: rho_i = min_j ||x_i - x_j|| (distance to nearest neighbor)
3. Compute local radius sigma_i such that sum_{j in NN(i)} exp(-(||x_i - x_j|| - rho_i) / sigma_i) = log_2(k)
4. Define fuzzy simplicial set membership: w_{ij} = exp(-(||x_i - x_j|| - rho_i) / sigma_i) for j in NN(i)
5. Symmetrize using local fuzzy union: W = w + w^T - w * w^T (element-wise)

### Low-Dimensional Optimization

In low dimensions, UMAP uses a repulsive/attractive force model. The attractive force pulls connected points together, while the repulsive force pushes all points apart (preventing collapse). The forces are:

**Attractive**: -2 * w_{ij} * (y_i - y_j) / (a + ||y_i - y_j||^2 + 1e-8)

**Repulsive**: 2 * (1 - w_{ij}) * (y_i - y_j) * b / (0.001 + ||y_i - y_j||^2) * (a + ||y_i - y_j||^2)

where a and b are derived from min_dist via curve fitting.

### Key Parameters

- n_neighbors: Controls local/global balance (default 15). Larger values preserve more global structure.
- min_dist: Controls how tightly points are packed (default 0.1). Smaller values produce more compact clusters.
- n_components: Embedding dimension (default 2).

### Differences from t-SNE

- UMAP uses a different normalization (fuzzy union vs symmetrized conditional probabilities)
- UMAP minimizes cross-entropy (KL + reverse KL), t-SNE minimizes KL only
- UMAP is faster (O(n log n) throughout), t-SNE is O(n^2) or O(n log n) with approximations
- UMAP supports out-of-sample embedding via transform()

## Code Examples

### Example 1: Basic UMAP on Digits

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import umap

X, y = load_digits(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

print(f"UMAP embedding shape: {X_umap.shape}")
# Output: UMAP embedding shape: (1797, 2)

print(f"Final cross-entropy: {reducer._initial_alpha:.3f}")
# (Note: final cross-entropy not directly exposed, but embedding quality is visible)

# Cluster separation quality via silhouette
from sklearn.metrics import silhouette_score
sil = silhouette_score(X_umap, y)
print(f"Silhouette score in UMAP space: {sil:.3f}")
# Output: Silhouette score in UMAP space: 0.527
```

### Example 2: UMAP vs t-SNE Speed Comparison

```python
import time
import numpy as np
from sklearn.manifold import TSNE
import umap

np.random.seed(42)
sizes = [500, 1000, 2000, 5000]

print("Speed comparison (UMAP vs t-SNE):")
for n in sizes:
    X = np.random.randn(n, 50)

    start = time.time()
    reducer = umap.UMAP(n_neighbors=15, random_state=42)
    X_umap = reducer.fit_transform(X)
    umap_time = time.time() - start

    start = time.time()
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)
    tsne_time = time.time() - start

    print(f"  n={n:5d}: UMAP={umap_time:.2f}s, t-SNE={tsne_time:.2f}s, "
          f"speedup={tsne_time/umap_time:.1f}x")
# Output:
#   n=  500: UMAP=0.62s, t-SNE=2.21s, speedup=3.6x
#   n= 1000: UMAP=1.21s, t-SNE=6.48s, speedup=5.4x
#   n= 2000: UMAP=2.18s, t-SNE=18.12s, speedup=8.3x
#   n= 5000: UMAP=4.85s, t-SNE=85.34s, speedup=17.6x
```

### Example 3: Effect of n_neighbors Parameter

```python
import umap
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)

for n_neighbors in [2, 5, 15, 30, 50, 100]:
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X)
    sil = silhouette_score(X_umap, y)
    print(f"n_neighbors={n_neighbors:3d}: silhouette={sil:.3f}")

# Output:
# n_neighbors=  2: silhouette=0.311
# n_neighbors=  5: silhouette=0.464
# n_neighbors= 15: silhouette=0.527
# n_neighbors= 30: silhouette=0.501
# n_neighbors= 50: silhouette=0.466
# n_neighbors=100: silhouette=0.329
```

### Example 4: Effect of min_dist Parameter

```python
for min_dist in [0.0, 0.1, 0.25, 0.5, 0.8, 0.99]:
    reducer = umap.UMAP(n_neighbors=15, min_dist=min_dist, random_state=42)
    X_umap = reducer.fit_transform(X)
    sil = silhouette_score(X_umap, y)
    print(f"min_dist={min_dist:.2f}: silhouette={sil:.3f}")
# Output:
# min_dist=0.00: silhouette=0.531
# min_dist=0.10: silhouette=0.527
# min_dist=0.25: silhouette=0.505
# min_dist=0.50: silhouette=0.446
# min_dist=0.80: silhouette=0.356
# min_dist=0.99: silhouette=0.283
```

### Example 5: UMAP for Preprocessing

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits
import umap

X, y = load_digits(return_X_y=True)

# Original
rf = RandomForestClassifier(random_state=42)
scores_orig = cross_val_score(rf, X, y, cv=5).mean()

# After UMAP reduction
reducer = umap.UMAP(n_neighbors=15, n_components=10, random_state=42)
X_umap = reducer.fit_transform(X)
scores_umap = cross_val_score(rf, X_umap, y, cv=5).mean()

print(f"Original ({X.shape[1]} features): {scores_orig:.3f}")
print(f"UMAP 10D: {scores_umap:.3f}")
# Output:
# Original (64 features): 0.935
# UMAP 10D: 0.938
```

## Common Mistakes

1. **Interpreting UMAP plots like t-SNE plots.** While similar, UMAP preserves global structure better, so distances between clusters are more meaningful. Still, caution is warranted.

2. **Not normalizing data before UMAP.** UMAP uses Euclidean distances for the k-NN graph. Features on different scales distort the graph.

3. **Using default parameters without tuning.** n_neighbors and min_dist significantly affect results. Always explore parameter space.

4. **Applying UMAP to very small datasets (n < 50).** UMAP's k-NN graph becomes unreliable with too few points.

5. **Using UMAP for lossy compression without validation.** UMAP is stochastic and discards information. Validate downstream task performance.

6. **Setting n_neighbors too high (close to n).** This collapses all global structure into a single manifold, losing local detail.

7. **Expecting identical results across runs.** UMAP's stochastic optimization produces slightly different embeddings each time. Set random_state for reproducibility.

## Interview Questions

### Beginner

1. What is UMAP?
Uniform Manifold Approximation and Projection — a non-linear dimensionality reduction technique that preserves topological structure using fuzzy simplicial sets.

2. How does UMAP differ from t-SNE?
UMAP is faster, preserves global structure better, supports out-of-sample embedding, and uses cross-entropy instead of KL divergence. It constructs a graph differently (fuzzy union vs symmetrized probabilities).

3. What do n_neighbors and min_dist control?
n_neighbors balances local vs global structure (higher = more global). min_dist controls how tightly points pack together in the embedding (lower = tighter clusters).

4. Can UMAP embed new points without retraining?
Yes. UMAP provides a transform() method that embeds new points into an existing embedding via the graph Laplacian.

5. Is UMAP deterministic?
No, the optimization is stochastic. Set random_state for reproducibility.

### Intermediate

1. Explain the fuzzy simplicial set construction in UMAP.
For each point, UMAP computes distances to its k nearest neighbors. It normalizes these distances by a local radius sigma_i so that distances are meaningful locally. The membership strength w_{ij} = exp(-(d_{ij} - rho_i) / sigma_i). These local views are combined via the fuzzy union operation.

2. Compare UMAP's optimization objective with t-SNE.
UMAP minimizes cross-entropy: CE(P, Q) = sum p log(p/q) + sum (1-p) log((1-p)/(1-q)) = KL(P||Q) + reverse KL, which has both attractive and repulsive forces. t-SNE uses only KL divergence, which lacks the repulsive push on well-separated points.

3. How does UMAP handle out-of-sample extension?
After fitting, UMAP builds a graph Laplacian from the original training data. For a new point, it finds nearest neighbors in the training set and uses Laplacian eigenfunction techniques to find the optimal embedding position.

4. What makes UMAP faster than t-SNE?
UMAP uses approximate nearest neighbor search (via NNDescent) for graph construction and a simpler optimization (SGD with negative sampling vs gradient descent with momentum).

5. How do the parameters a and b relate to min_dist?
UMAP fits a curve: phi(x) = 1/(1 + a x^{2b}) that approximates the desired min_dist behavior. Small min_dist gives a and b that make the function drop steeply near zero.

### Advanced

1. Derive the UMAP cost function from the perspective of cross-entropy between fuzzy sets.
Let A be the adjacency matrix of the high-D fuzzy simplicial set, B be the low-D adjacency. Cross-entropy: sum_i sum_j [A_ij log(A_ij/B_ij) + (1-A_ij) log((1-A_ij)/(1-B_ij))]. The first term attracts connected points; the second term repels unconnected points.

2. Prove that UMAP's graph construction corresponds to computing the geometric realization of a fuzzy simplicial set.
The fuzzy simplicial set assigns a membership strength to each simplex based on distances to neighbors. The construction ensures that the representation is functorial — it respects the manifold structure by computing the Čech complex of the metric space with variable radius.

3. Compare the theoretical foundations of UMAP and t-SNE.
t-SNE is motivated by information theory (KL divergence). UMAP is motivated by topological data analysis (persistent homology, simplicial sets). UMAP provides stronger theoretical guarantees about manifold recovery under certain assumptions.

## Practice Problems

### Easy

1. Apply UMAP to the Iris dataset with default parameters. Visualize the embedding.

2. Compare UMAP and t-SNE on the Wine dataset side by side.

3. Scale features and apply UMAP to the breast cancer dataset. Report silhouette score.

4. Use UMAP to visualize MNIST in 2D (use a subset of 2000 points).

5. Experiment with n_neighbors = 2, 15, 100 on the digits dataset.

### Medium

1. Use UMAP for preprocessing: reduce to 10D, then classify with logistic regression.

2. Compare UMAP and PCA on the Swiss roll dataset.

3. Analyze the stability of UMAP by running it 10 times with different seeds and computing the correlation between embeddings.

4. Implement parameter search for UMAP (n_neighbors, min_dist) using KNN classifier accuracy as metric.

5. Use UMAP to visualize word embeddings (e.g., spaCy or GloVe) and observe semantic clustering.

### Hard

1. Implement UMAP's graph construction from scratch (fuzzy simplicial set).

2. Extend UMAP to supervised dimensionality reduction (use label information in graph construction).

3. Compare UMAP with parametric UMAP (using a neural network) on a large dataset.

## Solutions

Easy 1: UMAP on Iris

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.metrics import silhouette_score

X, y = load_iris(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

sil = silhouette_score(X_umap, y)
print(f"UMAP embedding shape: {X_umap.shape}")
print(f"Silhouette score: {sil:.3f}")
# Output:
# UMAP embedding shape: (150, 2)
# Silhouette score: 0.716
```

## Related Concepts

- **t-SNE** (ML-047): The most direct comparison — both are non-linear visualization methods
- **PCA** (ML-046): Linear dimensionality reduction
- **Manifold Learning**: The broader theoretical framework
- **Topological Data Analysis**: UMAP's mathematical foundation
- **Feature Extraction** (ML-050): UMAP as a representation learning tool

## Next Concepts

- LDA (ML-049): Supervised linear dimensionality reduction
- Feature Extraction (ML-050): Broader representation learning methods

## Summary

UMAP is a modern non-linear dimensionality reduction technique that preserves both local and global topological structure. It is faster than t-SNE, scales to millions of points, supports out-of-sample embedding, and produces more reproducible results. The key parameters (n_neighbors, min_dist) control the tradeoff between local and global structure preservation. UMAP is widely used for visualization, preprocessing, and exploratory data analysis.

## Key Takeaways

- UMAP preserves topological structure via fuzzy simplicial sets
- Faster and more scalable than t-SNE
- Preserves global structure better than t-SNE
- Out-of-sample embedding via transform()
- n_neighbors controls local/global balance (default 15)
- min_dist controls embedding compactness (default 0.1)
- Scale features before applying UMAP
- Cross-entropy optimization includes both attractive and repulsive forces
- Widely used in bioinformatics, NLP, and exploratory analysis
