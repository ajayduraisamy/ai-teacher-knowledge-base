# Concept: t-SNE

## Concept ID

ML-047

## Difficulty

Advanced

## Domain

Machine Learning

## Module

Dimensionality Reduction

## Learning Objectives

- Understand t-SNE as a non-linear dimensionality reduction technique for visualization
- Explain how t-SNE preserves local structure using probability distributions
- Use sklearn's TSNE for visualizing high-dimensional data
- Interpret t-SNE plots correctly, understanding their limitations
- Tune the perplexity parameter and understand its effect

## Prerequisites

- Basic dimensionality reduction concepts
- Probability distributions (Gaussian, Student-t)
- KL divergence
- Python with sklearn and NumPy

## Definition

t-SNE (t-distributed Stochastic Neighbor Embedding) is a non-linear dimensionality reduction technique designed specifically for visualizing high-dimensional data in 2D or 3D. It converts pairwise similarities between data points into joint probabilities and minimizes the KL divergence between the high-dimensional and low-dimensional probability distributions.

Given high-dimensional data X = {x_1, ..., x_n}, t-SNE produces low-dimensional embeddings Y = {y_1, ..., y_n} in R^d (typically d=2 or 3) by minimizing:

C = KL(P || Q) = sum_i sum_j p_{ij} log(p_{ij} / q_{ij})

where P is the pairwise similarity distribution in the original space and Q in the embedding space.

## Intuition

t-SNE preserves local structure by making sure points that are close in high-dimensional space remain close in the 2D/3D embedding. It uses a heavy-tailed Student-t distribution in the low-dimensional space to address the "crowding problem" — moderate distances in high-dimensional space are exaggerated in the embedding, creating gaps between clusters.

Think of it as arranging points on a map such that nearby cities in reality are nearby on the map, but distances between far-away cities are not meaningful. The goal is to see the cluster structure, not to preserve global geometry.

## Why This Concept Matters

t-SNE has become the de facto standard for visualizing high-dimensional data in machine learning. It produces stunning 2D visualizations of complex datasets like MNIST, ImageNet features, and word embeddings where clusters clearly emerge. Understanding t-SNE is crucial for exploratory data analysis, presenting results, and debugging learned representations.

## Mathematical Explanation

### High-Dimensional Similarities

For each point x_i, define conditional probabilities:

p_{j|i} = exp(-||x_i - x_j||^2 / 2 sigma_i^2) / sum_{k != i} exp(-||x_i - x_k||^2 / 2 sigma_i^2)

where sigma_i is set so that the perplexity of the conditional distribution equals a user-specified value.

The joint probability is symmetrized:

p_{ij} = (p_{j|i} + p_{i|j}) / (2n)

### Low-Dimensional Similarities

Using Student-t distribution with 1 degree of freedom (Cauchy):

q_{ij} = (1 + ||y_i - y_j||^2)^{-1} / sum_{k != l} (1 + ||y_k - y_l||^2)^{-1}

The heavy tails of the t-distribution allow moderate distances in high-D to be represented as larger distances in low-D, alleviating the crowding problem.

### Optimization

KL divergence is minimized via gradient descent:

dC/dy_i = 4 sum_j (p_{ij} - q_{ij})(y_i - y_j)(1 + ||y_i - y_j||^2)^{-1}

### Perplexity

Perplexity = 2^{H(P_i)}, where H(P_i) is the entropy of P_i. Perplexity controls sigma_i — it can be interpreted as the effective number of neighbors. Typical values: 5-50.

## Code Examples

### Example 1: t-SNE on Digits Dataset

```python
import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
X, y = digits.data, digits.target

tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
X_tsne = tsne.fit_transform(X)

print(f"Final KL divergence: {tsne.kl_divergence_:.2f}")
print(f"KL divergence at last iteration: {tsne.kl_divergence_:.2f}")
# Output: Final KL divergence: 1.02

# Visualization
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.8, s=10)
plt.colorbar(scatter, ticks=range(10))
plt.title("t-SNE visualization of Digits dataset")
plt.xlabel("t-SNE component 1")
plt.ylabel("t-SNE component 2")
plt.tight_layout()
plt.savefig('tsne_digits.png', dpi=150)
print("Digits t-SNE plot saved.")
# Output: Digits t-SNE plot saved.
```

### Example 2: Effect of Perplexity

```python
import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)

perplexities = [5, 10, 30, 50, 80, 100]
print("Effect of perplexity on KL divergence:")
for perp in perplexities:
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X)
    print(f"  perplexity={perp:3d}: KL div={tsne.kl_divergence_:.3f}, "
          f"iterations={tsne.n_iter_}")

# Output:
#   perplexity=  5: KL div=0.886, iterations=1000
#   perplexity= 10: KL div=0.981, iterations=1000
#   perplexity= 30: KL div=1.018, iterations=1000
#   perplexity= 50: KL div=1.008, iterations=1000
#   perplexity= 80: KL div=0.983, iterations=1000
#   perplexity=100: KL div=0.969, iterations=1000
```

### Example 3: t-SNE vs PCA on Swiss Roll

```python
import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
# Output: PCA explained variance: 0.664

print(f"t-SNE final KL divergence: {tsne.kl_divergence_:.3f}")
# Output: t-SNE final KL divergence: 0.116

# t-SNE unfolds the Swiss roll, PCA collapses it
# Correlation with original manifold position (t)
from scipy.stats import spearmanr
pca_corr, _ = spearmanr(t, X_pca[:, 0])
tsne_corr, _ = spearmanr(t, X_tsne[:, 0])
print(f"Spearman correlation with manifold position:")
print(f"  PCA: {pca_corr:.3f}")
print(f"  t-SNE: {tsne_corr:.3f}")
# Output:
# Spearman correlation with manifold position:
#   PCA: 0.876
#   t-SNE: 0.992
```

### Example 4: Multiple Runs and Randomness

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)

print("Multiple t-SNE runs (same parameters, different random states):")
for seed in [42, 123, 256, 789]:
    tsne = TSNE(n_components=2, perplexity=30, random_state=seed, n_iter=1000)
    X_tsne = tsne.fit_transform(X)
    kl_div = tsne.kl_divergence_
    print(f"  seed={seed}: KL={kl_div:.3f}")

# Output:
#   seed=42: KL=1.018
#   seed=123: KL=1.034
#   seed=256: KL=1.006
#   seed=789: KL=1.003

# Note: Each run produces a different embedding with similar KL.
```
the t-SNE plot.

### Example 5: Barnes-Hut t-SNE for Large Datasets

```python
import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml

# Use method='exact' for small data, 'barnes_hut' for large
# Barnes-Hut is O(n log n), exact is O(n^2)

np.random.seed(42)
X_large = np.random.randn(5000, 50)

# Barnes-Hut (default for n > 200)
tsne_bh = TSNE(n_components=2, perplexity=30, method='barnes_hut', random_state=42)
X_bh = tsne_bh.fit_transform(X_large)
print(f"Barnes-Hut: {tsne_bh.kl_divergence_:.3f}, time: O(n log n)")

# For comparison on a small subset
X_small = X_large[:200]
tsne_exact = TSNE(n_components=2, perplexity=30, method='exact', random_state=42)
X_exact = tsne_exact.fit_transform(X_small)
print(f"Exact: {tsne_exact.kl_divergence_:.3f}, time: O(n^2)")

# Output:
# Barnes-Hut: 1.341, time: O(n log n)
# Exact: 1.128, time: O(n^2)
```

## Common Mistakes

1. **Interpreting distances between clusters.** t-SNE preserves local structure; distances between well-separated clusters are not meaningful.

2. **Interpreting cluster sizes.** t-SNE inflates sparse clusters and shrinks dense ones. Cluster size in the plot does not reflect cluster size in the data.

3. **Using t-SNE for dimensionality reduction before other algorithms.** t-SNE is designed for visualization, not as a preprocessing step. It discards global structure and is stochastic.

4. **Not trying multiple perplexity values.** Different perplexities reveal different structure. Always try a range (5-50).

5. **Ignoring the stochastic nature of t-SNE.** Each run produces different results. To see stable patterns, run multiple times with different seeds.

6. **Using t-SNE for very large datasets without subsampling.** t-SNE is O(n^2) for exact and O(n log n) for Barnes-Hut, which still becomes slow for n > 100,000. Subsample first.

7. **Assuming t-SNE finds global structure.** t-SNE does not preserve global distances or density. For global structure preservation, use UMAP or PCA.

## Interview Questions

### Beginner

1. What is t-SNE used for?
Non-linear dimensionality reduction for visualizing high-dimensional data in 2D or 3D. It preserves local structure so nearby points remain nearby.

2. What is perplexity in t-SNE?
The effective number of neighbors for each point. Typical values 5-50. Lower perplexity focuses on very local structure; higher perplexity considers more global structure.

3. How does t-SNE differ from PCA?
t-SNE is non-linear, stochastic, and preserves local distances. PCA is linear, deterministic, and preserves global variance. t-SNE is for visualization; PCA is for dimensionality reduction.

4. Why does t-SNE use a Student-t distribution in low dimensions?
The heavy tails prevent the crowding problem — moderate distances in high-D can be represented as larger distances in low-D, creating gaps between clusters.

5. What is the KL divergence in t-SNE?
It measures how well the low-dimensional distribution Q matches the high-dimensional distribution P. Lower KL means better preservation of pairwise similarities.

### Intermediate

1. Explain the crowding problem in t-SNE.
Points that are moderately distant in high-dimensional space have too little room in low-dimensional space (the "crowding" of many medium distances). The Student-t distribution's heavy tails push these points farther apart in embedding space, alleviating crowding.

2. How do you choose perplexity?
Try values from 5 to 50. Lower perplexity captures fine-grained structure; higher perplexity captures broader patterns. No single optimal value exists — explore multiple.

3. What are the limitations of t-SNE?
Global distances not meaningful, stochastic (different runs differ), computationally expensive, sensitive to perplexity, cluster sizes not interpretable, and cannot embed new points without retraining.

4. Compare Barnes-Hut t-SNE with exact t-SNE.
Barnes-Hut approximates forces using a quadtree structure, reducing complexity from O(n^2) to O(n log n). It is suitable for n > 10,000 but introduces approximation error.

5. Why can't t-SNE embed new points (no out-of-sample extension)?
t-SNE optimizes positions of all points jointly. To embed a new point, you must run t-SNE again on the full dataset. Parametric t-SNE (via neural networks) addresses this.

### Advanced

1. Derive the gradient of the t-SNE cost function.
dC/dy_i = 4 sum_j (p_{ij} - q_{ij})(y_i - y_j)(1 + ||y_i - y_j||^2)^{-1}. This has a physical interpretation: the total force on point i is the sum of attractive forces (p_{ij} > q_{ij}) and repulsive forces (p_{ij} < q_{ij}) from all other points.

2. How does t-SNE relate to other manifold learning methods?
t-SNE is related to SNE (Stochastic Neighbor Embedding) which uses Gaussian in both spaces. t-SNE improves on SNE by using the t-distribution in low-D. It is also related to UMAP, which uses a different normalization and optimization approach.

3. Prove that t-SNE preserves local neighborhoods in expectation under certain conditions.
For points that are very close in high-D (||x_i - x_j|| << sigma_i), p_{ij} is large and the gradient maintains their proximity. For distant points, q_{ij} is forced to be very small, and the repulsive forces prevent them from collapsing, preserving the neighborhood structure.

## Practice Problems

### Easy

1. Run t-SNE on the Iris dataset with perplexity=10, 30, 50. Compare the plots.

2. Apply t-SNE to the Wine dataset reduced to 2D. Color by wine class.

3. Compare PCA and t-SNE projections of the digits dataset side by side.

4. Run t-SNE on random noise data. Observe that no clusters form.

5. Apply t-SNE to the breast cancer dataset. Check if the two classes separate.

### Medium

1. Investigate the effect of perplexity on the digits dataset for values 2, 5, 30, 100. Describe how cluster structure changes.

2. Compare t-SNE with PCA on the Swiss roll dataset. Show that t-SNE unfolds the manifold.

3. Run t-SNE multiple times on the same data with different seeds. Measure the pairwise similarity of the resulting embeddings.

4. Use t-SNE to visualize word embeddings (e.g., GloVe vectors for a subset of words). Verify that semantic relationships appear.

5. Implement a parametric t-SNE using a neural network (autoencoder-like) that can embed new points.

### Hard

1. Implement t-SNE from scratch (without sklearn). Include both exact and Barnes-Hut approximations.

2. Derive the gradient of the t-SNE objective and implement gradient descent with momentum.

3. Compare t-SNE with UMAP on a dataset with known hierarchical structure. Show that UMAP preserves global structure better.

## Solutions

Easy 1: t-SNE on Iris

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
import numpy as np

X, y = load_iris(return_X_y=True)

for perp in [10, 30, 50]:
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    X_tsne = tsne.fit_transform(X)
    kl = tsne.kl_divergence_
    print(f"perplexity={perp}: KL={kl:.3f}")
# Output:
# perplexity=10: KL=0.060
# perplexity=30: KL=0.093
# perplexity=50: KL=0.105
```

## Related Concepts

- **PCA** (ML-046): Linear dimensionality reduction — complementary to t-SNE
- **UMAP** (ML-048): Modern non-linear method, preserves global structure better
- **Manifold Learning**: The broader class t-SNE belongs to
- **KL Divergence**: The optimization objective in t-SNE

## Next Concepts

- UMAP (ML-048): Faster, more scalable manifold learning
- Feature Extraction (ML-050): Learning representations for downstream tasks

## Summary

t-SNE is a powerful non-linear dimensionality reduction technique designed for visualizing high-dimensional data. It preserves local pairwise similarities by minimizing KL divergence between high-D and low-D probability distributions. The Student-t distribution in low-D addresses the crowding problem. Key considerations include perplexity tuning, the stochastic nature of the algorithm, and interpreting t-SNE plots correctly (local distances only, not global distances or cluster sizes).

## Key Takeaways

- t-SNE preserves local structure for visualization
- Perplexity controls the balance between local and global structure
- Uses Student-t distribution to avoid crowding
- KL divergence is minimized via gradient descent
- Global distances and cluster sizes are not meaningful
- Multiple runs with different seeds give different views
- Barnes-Hut approximation scales to larger datasets
- t-SNE cannot embed new points (no out-of-sample extension)
- Best used for exploration, not preprocessing
