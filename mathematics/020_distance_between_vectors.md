# Concept: Distance Between Vectors

## Concept ID

MATH-020

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Vector Algebra

## Learning Objectives

- Define and compute Euclidean distance between vectors
- Understand L1 (Manhattan) distance and Lp norms
- Differentiate between Euclidean, Manhattan, and cosine distances
- Apply distance metrics to k-NN, k-means, and vector search
- Choose appropriate distance metrics for different data types

## Prerequisites

- Vector subtraction: $\mathbf{u} - \mathbf{v}$
- Vector magnitude (norm): $\|\mathbf{u}\|$
- Dot product (MATH-016)
- Basic algebra and square roots

## Definition

The **distance between two vectors** measures how far apart their endpoints are when the vectors are placed tail-to-tail at the origin. The most common distance is the **Euclidean distance** (also called the L2 distance):

$$
d_2(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\| = \sqrt{\sum_{i=1}^n (u_i - v_i)^2}
$$

More generally, the **Lp distance** (Minkowski distance) is:

$$
d_p(\mathbf{u}, \mathbf{v}) = \left(\sum_{i=1}^n |u_i - v_i|^p\right)^{1/p}
$$

Important special cases:
- **L1 distance** (Manhattan distance): $p = 1$, $d_1(\mathbf{u}, \mathbf{v}) = \sum_{i=1}^n |u_i - v_i|$
- **L2 distance** (Euclidean distance): $p = 2$, $d_2(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^n (u_i - v_i)^2}$
- **L$\infty$ distance** (Chebyshev distance): $d_\infty(\mathbf{u}, \mathbf{v}) = \max_i |u_i - v_i|$

**Cosine distance** (related but not a true metric) is:

$$
d_{\cos}(\mathbf{u}, \mathbf{v}) = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}
$$

## Intuition

Distance between vectors measures how "far apart" two points are in space. Euclidean distance is the straight-line distance — the length of the shortest path between two points. Manhattan distance is the distance you would travel along a grid (like city blocks). Cosine distance measures directional difference rather than magnitude difference.

Think of two people walking in a field: Euclidean distance is "as the crow flies," Manhattan distance is walking along perpendicular paths, and cosine distance asks "are they facing the same direction?" regardless of how far apart they stand.

## Why This Concept Matters

Distance is the foundation of nearly all geometric machine learning:

- Classification algorithms (k-NN) classify points based on distance to neighbours
- Clustering algorithms (k-means, DBSCAN) group points by proximity
- Vector databases (Pinecone, Milvus, Weaviate) retrieve nearest neighbours by distance
- Anomaly detection flags points far from the cluster centre
- Dimensionality reduction (MDS, t-SNE) preserves pairwise distances

Choosing the right distance metric can dramatically affect model performance.

## Historical Background

The Euclidean distance is named after **Euclid of Alexandria** (c. 300 BCE), whose "Elements" formalised the geometry of straight-line distances. The Manhattan distance was studied by **Hermann Minkowski** (1864–1909), who generalised Lp spaces. The term "Manhattan distance" was popularised in the 1950s, referring to the grid layout of Manhattan's streets. The cosine distance emerged from information retrieval in the 1970s (Salton's vector space model). Modern vector databases and embedding-based search have made distance computation a billion-dollar technology.

## Real World Examples

1. **GPS Navigation:** Euclidean distance approximates straight-line distance between two GPS coordinates; Manhattan distance approximates driving distance in grid cities.
2. **Image Recognition:** Pixel-wise Euclidean distance between image feature vectors measures image similarity.
3. **Genomics:** Genetic distance between DNA sequences is computed using Hamming distance (special case of L1).
4. **Sports Analytics:** Player performance vectors are compared using Euclidean distance to find similar players.
5. **Recommendation Systems:** User preference vectors are compared using cosine distance (which ignores magnitude differences in rating scales).

## AI/ML Relevance

Distance metrics are central to many ML algorithms:

1. **k-Nearest Neighbours (k-NN):** The simplest classification algorithm computes distances between a query point and all training points, then takes a vote among the $k$ closest neighbours. Euclidean distance is the default choice, but Manhattan distance works better for high-dimensional data.

2. **k-Means Clustering:** Assigns each point to the nearest cluster centroid (using Euclidean distance), then recomputes centroids. The objective is to minimise the sum of squared Euclidean distances within clusters.

3. **Vector Databases:** Modern AI applications use vector embeddings (from LLMs, image models, etc.) stored in vector databases. When a query comes in, the database finds the nearest vectors by distance (typically cosine or Euclidean). This powers semantic search, RAG (retrieval-augmented generation), and recommendation.

4. **t-SNE and UMAP:** Dimensionality reduction techniques that aim to preserve pairwise distances (or similarities) between high-dimensional points in a low-dimensional embedding.

5. **Anomaly Detection:** Points with large distance to the nearest cluster centre or to the $k$th nearest neighbour are flagged as anomalies.

6. **Loss Functions:** Mean squared error (MSE) is the squared Euclidean distance between predictions and targets. Mean absolute error (MAE) is the L1 distance.

## Mathematical Explanation

**Euclidean distance (L2):** This is the most intuitive distance — the straight-line distance between two points. For vectors $\mathbf{u}$ and $\mathbf{v}$ in $\mathbb{R}^n$:

$$
d_2(\mathbf{u}, \mathbf{v}) = \sqrt{(u_1 - v_1)^2 + (u_2 - v_2)^2 + \cdots + (u_n - v_n)^2}
$$

This is the magnitude of the difference vector $\mathbf{u} - \mathbf{v}$.

**Manhattan distance (L1):** Sum of absolute differences along each dimension:

$$
d_1(\mathbf{u}, \mathbf{v}) = |u_1 - v_1| + |u_2 - v_2| + \cdots + |u_n - v_n|
$$

**Lp distance:** A generalisation:

$$
d_p(\mathbf{u}, \mathbf{v}) = \left(\sum_{i=1}^n |u_i - v_i|^p\right)^{1/p}
$$

As $p \to \infty$, only the largest absolute difference matters:

$$
d_\infty(\mathbf{u}, \mathbf{v}) = \max_i |u_i - v_i|
$$

**Cosine distance** measures angular dissimilarity:

$$
d_{\cos}(\mathbf{u}, \mathbf{v}) = 1 - \cos\theta = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}
$$

For unit vectors ($\|\mathbf{u}\| = \|\mathbf{v}\| = 1$), there is a direct relationship between Euclidean distance and cosine distance:

$$
\|\mathbf{u} - \mathbf{v}\|^2 = 2 - 2\cos\theta = 2 \cdot d_{\cos}(\mathbf{u}, \mathbf{v})
$$

**Properties of distance metrics:** A true distance metric $d$ must satisfy:
1. **Non-negativity:** $d(\mathbf{u}, \mathbf{v}) \geq 0$
2. **Identity:** $d(\mathbf{u}, \mathbf{v}) = 0$ iff $\mathbf{u} = \mathbf{v}$
3. **Symmetry:** $d(\mathbf{u}, \mathbf{v}) = d(\mathbf{v}, \mathbf{u})$
4. **Triangle inequality:** $d(\mathbf{u}, \mathbf{w}) \leq d(\mathbf{u}, \mathbf{v}) + d(\mathbf{v}, \mathbf{w})$

Euclidean, Manhattan, and Chebyshev distances all satisfy these. Cosine distance does NOT satisfy the triangle inequality, so it is not a true metric (though it is commonly used as a similarity measure).

## Formula(s)

**Euclidean (L2) distance:**
$$
d_2(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^n (u_i - v_i)^2}
$$

**Manhattan (L1) distance:**
$$
d_1(\mathbf{u}, \mathbf{v}) = \sum_{i=1}^n |u_i - v_i|
$$

**Chebyshev (L$\infty$) distance:**
$$
d_\infty(\mathbf{u}, \mathbf{v}) = \max_i |u_i - v_i|
$$

**Minkowski (Lp) distance:**
$$
d_p(\mathbf{u}, \mathbf{v}) = \left(\sum_{i=1}^n |u_i - v_i|^p\right)^{1/p}
$$

**Cosine distance:**
$$
d_{\cos}(\mathbf{u}, \mathbf{v}) = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}
$$

**Relationship for unit vectors:**
$$
\|\mathbf{u} - \mathbf{v}\|^2 = 2 \cdot d_{\cos}(\mathbf{u}, \mathbf{v})
$$

## Properties

1. **Non-negativity:** All Lp distances are non-negative; zero only when vectors are identical.
2. **Symmetry:** $d(\mathbf{u}, \mathbf{v}) = d(\mathbf{v}, \mathbf{u})$ for all Lp distances and cosine distance.
3. **Triangle inequality:** Lp distances satisfy $d(\mathbf{u}, \mathbf{w}) \leq d(\mathbf{u}, \mathbf{v}) + d(\mathbf{v}, \mathbf{w})$. Cosine distance does NOT satisfy this.
4. **Scale sensitivity:** Euclidean distance is sensitive to the scale of each dimension. Manhattan distance is less sensitive. Cosine distance is scale-invariant.
5. **Curse of dimensionality:** In high dimensions, all Lp distances tend to become similar (the distance concentration phenomenon), which makes nearest neighbour search challenging.
6. **L2 squared relationship:** $\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2(\mathbf{u} \cdot \mathbf{v})$
7. **Monotonicity:** For $p < q$, $d_p(\mathbf{u}, \mathbf{v}) \geq d_q(\mathbf{u}, \mathbf{v})$ in general (L1 is usually larger than L2 for the same vectors).

## Step-by-Step Worked Examples

### Example 1: Euclidean Distance in 2D

**Problem:** Compute the Euclidean distance between $\mathbf{u} = \langle 1, 2 \rangle$ and $\mathbf{v} = \langle 4, 6 \rangle$.

**Solution:**

**Step 1:** Compute the difference vector:
$$
\mathbf{u} - \mathbf{v} = \langle 1-4, 2-6 \rangle = \langle -3, -4 \rangle
$$

**Step 2:** Apply the Euclidean distance formula:
$$
d_2(\mathbf{u}, \mathbf{v}) = \sqrt{(-3)^2 + (-4)^2} = \sqrt{9 + 16} = \sqrt{25} = 5
$$

**Step 3:** Interpretation: The two points are 5 units apart in straight-line distance.

### Example 2: Manhattan vs Euclidean Distance

**Problem:** For $\mathbf{u} = \langle 1, 1 \rangle$ and $\mathbf{v} = \langle 4, 5 \rangle$, compute both Euclidean and Manhattan distances.

**Solution:**

**Step 1:** Euclidean distance:
$$
d_2 = \sqrt{(1-4)^2 + (1-5)^2} = \sqrt{(-3)^2 + (-4)^2} = \sqrt{9 + 16} = 5
$$

**Step 2:** Manhattan distance:
$$
d_1 = |1-4| + |1-5| = |-3| + |-4| = 3 + 4 = 7
$$

**Step 3:** Comparison: The Manhattan distance (7) is larger than the Euclidean distance (5). In a city grid, you would walk 7 blocks (3 east, 4 north), while the straight-line distance is only 5.

### Example 3: Euclidean Distance in 3D

**Problem:** Find the distance between $\mathbf{u} = \langle 2, -1, 3 \rangle$ and $\mathbf{v} = \langle 1, 4, -2 \rangle$.

**Solution:**

**Step 1:** Compute the difference:
$$
\mathbf{u} - \mathbf{v} = \langle 2-1, -1-4, 3-(-2) \rangle = \langle 1, -5, 5 \rangle
$$

**Step 2:** Apply the formula:
$$
d_2 = \sqrt{1^2 + (-5)^2 + 5^2} = \sqrt{1 + 25 + 25} = \sqrt{51} \approx 7.141
$$

**Step 3:** The distance is approximately 7.141 units.

### Example 4: Cosine Distance

**Problem:** Compute the cosine distance between $\mathbf{u} = \langle 3, 4 \rangle$ and $\mathbf{v} = \langle 1, 2 \rangle$.

**Solution:**

**Step 1:** Compute dot product:
$$
\mathbf{u} \cdot \mathbf{v} = 3(1) + 4(2) = 3 + 8 = 11
$$

**Step 2:** Compute magnitudes:
$$
\|\mathbf{u}\| = \sqrt{9 + 16} = 5
$$
$$
\|\mathbf{v}\| = \sqrt{1 + 4} = \sqrt{5}
$$

**Step 3:** Compute cosine similarity:
$$
\cos\theta = \frac{11}{5\sqrt{5}} \approx 0.9839
$$

**Step 4:** Cosine distance:
$$
d_{\cos} = 1 - 0.9839 = 0.0161
$$

**Step 5:** Interpretation: The cosine distance is very small (close to 0), meaning the vectors point in almost the same direction. The Euclidean distance ($\sqrt{(3-1)^2 + (4-2)^2} = \sqrt{4+4} = \sqrt{8} \approx 2.828$) is much larger because the vectors have different magnitudes, but cosine distance ignores magnitude and focuses on direction.

### Example 5: Chebyshev (L$\infty$) Distance

**Problem:** Compute the Chebyshev distance between $\mathbf{u} = \langle 3, 7, 2 \rangle$ and $\mathbf{v} = \langle 1, 5, 9 \rangle$.

**Solution:**

**Step 1:** Compute absolute differences for each dimension:
- $|3 - 1| = 2$
- $|7 - 5| = 2$
- $|2 - 9| = 7$

**Step 2:** The Chebyshev distance is the maximum of these:
$$
d_\infty = \max(2, 2, 7) = 7
$$

**Step 3:** Interpretation: The Chebyshev distance only cares about the single dimension where the vectors differ the most. This is useful in chess (king moves) and certain optimisation problems.

## Visual Interpretation

**Euclidean distance** is the straight line connecting the tips of two vectors. In 2D, this is the familiar distance formula from geometry.

**Manhattan distance** is the sum of the horizontal and vertical distances — like walking along grid lines. In 2D, it always forms a right-angled path between the points.

**Cosine distance** relates to the angle between the vectors. Two vectors can be very far apart in Euclidean sense (different magnitudes) but have zero cosine distance (same direction). Visualise two arrows of different lengths pointing the same way — their Euclidean distance is the difference in lengths, but cosine distance is zero.

**Level sets** (sets of points at a fixed distance from the origin) look different for each metric:
- Euclidean: circles (spheres in higher dimensions)
- Manhattan: diamonds (diamonds rotated 45 degrees)
- Chebyshev: squares (aligned with axes)

## Common Mistakes

1. **Confusing distance with similarity:** Distance measures "how far apart," while similarity measures "how close." Cosine distance = 1 - cosine similarity. A distance of 0 means identical; a similarity of 1 means identical.

2. **Using Euclidean distance for unnormalised data:** If features have different scales (e.g., age in years vs income in dollars), Euclidean distance will be dominated by the feature with the largest range. Always normalise or use Manhattan/cosine distance.

3. **Forgetting the square root in Euclidean distance:** The squared Euclidean distance $\sum (u_i - v_i)^2$ is not the same as Euclidean distance $\sqrt{\sum (u_i - v_i)^2}$. Squared distance is more convenient for optimisation (no square root), but the actual distance includes the square root.

4. **Using cosine distance for magnitude-dependent tasks:** If the magnitude of the vector carries information (e.g., word frequency), cosine distance discards it. Use Euclidean distance instead.

5. **Assuming all distance metrics satisfy the triangle inequality:** Cosine distance does not. This means it is not a proper metric and cannot be used with certain algorithms (e.g., triangle-inequality-based indexing).

6. **Ignoring the curse of dimensionality:** In high dimensions (e.g., 100+), all pairwise Euclidean distances become nearly equal, making nearest neighbour search unreliable. Dimensionality reduction or specialised metrics may be needed.

7. **Applying Manhattan distance to continuous circular data:** For angles or cyclic features (e.g., hours of the day), neither Manhattan nor Euclidean distance works correctly without special handling.

## Interview Questions

### Beginner

1. What is Euclidean distance? Write the formula for 2D vectors.
2. How is Manhattan distance different from Euclidean distance?
3. What is the distance between $\mathbf{u} = \langle 0, 0, 0 \rangle$ and $\mathbf{v} = \langle 1, 1, 1 \rangle$?
4. When would you use cosine distance instead of Euclidean distance?
5. What is the range of possible values for cosine distance?

### Intermediate

1. Prove that $\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2(\mathbf{u} \cdot \mathbf{v})$.
2. Show the relationship between Euclidean distance and cosine distance for unit vectors.
3. Explain why Euclidean distance is not the best choice for text document vectors. What would you use instead?
4. In k-NN classification, why might Manhattan distance outperform Euclidean distance in high dimensions?
5. Given $\mathbf{u} = \langle 2, 3, 1 \rangle$ and $\mathbf{v} = \langle 1, 0, 4 \rangle$, compute Euclidean, Manhattan, and cosine distances.

### Advanced

1. Prove that the Minkowski distance $d_p$ satisfies the triangle inequality for $p \geq 1$ (Minkowski inequality).
2. Explain why in high-dimensional spaces, the ratio of nearest-neighbour distance to farthest-neighbour distance tends to 1 (curse of dimensionality). How does this affect k-NN?
3. For an inner product space, show that the following are equivalent: (a) $d(\mathbf{u}, \mathbf{v}) = \sqrt{\langle \mathbf{u} - \mathbf{v}, \mathbf{u} - \mathbf{v} \rangle}$ is a metric, (b) the inner product is positive definite, and (c) $\| \mathbf{u} \| = \sqrt{\langle \mathbf{u}, \mathbf{u} \rangle}$ defines a norm.

## Practice Problems

### Easy - 5 Questions

1. Compute Euclidean distance between $\mathbf{u} = \langle 3, 4 \rangle$ and $\mathbf{v} = \langle 0, 0 \rangle$.
2. Compute Manhattan distance between $\mathbf{u} = \langle 1, 2 \rangle$ and $\mathbf{v} = \langle 4, 6 \rangle$.
3. Find the Chebyshev distance between $\mathbf{u} = \langle 1, 5, 3 \rangle$ and $\mathbf{v} = \langle 4, 2, 3 \rangle$.
4. Compute cosine similarity between $\mathbf{u} = \langle 1, 0 \rangle$ and $\mathbf{v} = \langle 0, 1 \rangle$, then find cosine distance.
5. If $\mathbf{u} = \mathbf{v}$, what are the Euclidean, Manhattan, and cosine distances?

### Medium - 5 Questions

6. For $\mathbf{u} = \langle 1, -2, 3 \rangle$ and $\mathbf{v} = \langle 4, 0, -1 \rangle$, compute all three: Euclidean, Manhattan, and Chebyshev distances.
7. Two unit vectors have cosine distance 0.25. What is the Euclidean distance between them?
8. For $\mathbf{u} = \langle 1, 1, 1, 1 \rangle$ and $\mathbf{v} = \langle 2, 2, 2, 2 \rangle$, compute Euclidean and cosine distances. Explain why they give different impressions of "closeness."
9. Normalise $\mathbf{u} = \langle 3, 4 \rangle$ to a unit vector, then compute the Euclidean distance to $\mathbf{v} = \langle 0.6, 0.8 \rangle$.
10. Prove that for unit vectors, $d_{\cos}(\mathbf{u}, \mathbf{v}) = \frac{1}{2} d_2(\mathbf{u}, \mathbf{v})^2$.

### Hard - 3 Questions

11. Prove the triangle inequality for Euclidean distance: $\|\mathbf{u} - \mathbf{w}\| \leq \|\mathbf{u} - \mathbf{v}\| + \|\mathbf{v} - \mathbf{w}\|$.
12. Show that cosine distance is NOT a metric by finding three vectors $\mathbf{u}, \mathbf{v}, \mathbf{w}$ that violate the triangle inequality.
13. For a set of $n$ points in $\mathbb{R}^d$, the distance matrix $D_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\|$ can be used for multidimensional scaling (MDS). Prove that if $D$ is a Euclidean distance matrix, then the matrix $B = -\frac{1}{2} J D^{(2)} J$ is positive semidefinite, where $J = I - \frac{1}{n} \mathbf{1}\mathbf{1}^T$ is the centering matrix and $D^{(2)}_{ij} = D_{ij}^2$.

## Solutions

### Easy Solutions

1. $d_2 = \sqrt{(3-0)^2 + (4-0)^2} = \sqrt{9 + 16} = \sqrt{25} = 5$. This is the length of $\mathbf{u}$.
2. $d_1 = |1-4| + |2-6| = 3 + 4 = 7$.
3. Differences: $|1-4| = 3$, $|5-2| = 3$, $|3-3| = 0$. $d_\infty = \max(3, 3, 0) = 3$.
4. $\cos\theta = \frac{1(0) + 0(1)}{1 \cdot 1} = 0$. $d_{\cos} = 1 - 0 = 1$.
5. Euclidean and Manhattan distances $= 0$. Cosine distance $= 1 - 1 = 0$.

### Medium Solutions

6. Euclidean: $\mathbf{u} - \mathbf{v} = \langle -3, -2, 4 \rangle$. $d_2 = \sqrt{9 + 4 + 16} = \sqrt{29}$.
   Manhattan: $|1-4| + |-2-0| + |3-(-1)| = 3 + 2 + 4 = 9$.
   Chebyshev: $\max(|-3|, |-2|, |4|) = 4$.

7. $d_{\cos} = 0.25$ means $\cos\theta = 0.75$. For unit vectors: $d_2 = \sqrt{2(1 - \cos\theta)} = \sqrt{2(1 - 0.75)} = \sqrt{2 \times 0.25} = \sqrt{0.5} \approx 0.707$.

8. Euclidean: $d_2 = \sqrt{(1-2)^2 + (1-2)^2 + (1-2)^2 + (1-2)^2} = \sqrt{4 \times 1} = 2$.
   Cosine similarity: $\frac{1(2)+1(2)+1(2)+1(2)}{\sqrt{4}\sqrt{16}} = \frac{8}{2 \times 4} = 1$, so $d_{\cos} = 0$.
   Explanation: $\mathbf{v} = 2\mathbf{u}$, so they are parallel (cosine distance = 0) but have different magnitudes (Euclidean distance = 2). Which metric to use depends on whether magnitude matters.

9. Unit vector of $\mathbf{u}$: $\hat{\mathbf{u}} = \langle \frac{3}{5}, \frac{4}{5} \rangle = \langle 0.6, 0.8 \rangle$. This is exactly $\mathbf{v}$, so the Euclidean distance is 0.

10. For unit vectors: $\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2(\mathbf{u} \cdot \mathbf{v}) = 1 + 1 - 2\cos\theta = 2 - 2\cos\theta = 2(1 - \cos\theta) = 2 \cdot d_{\cos}(\mathbf{u}, \mathbf{v})$. Therefore $d_{\cos}(\mathbf{u}, \mathbf{v}) = \frac{1}{2}\|\mathbf{u} - \mathbf{v}\|^2$.

### Hard Solutions

11. Let $\mathbf{a} = \mathbf{u} - \mathbf{v}$ and $\mathbf{b} = \mathbf{v} - \mathbf{w}$. Then $\mathbf{u} - \mathbf{w} = \mathbf{a} + \mathbf{b}$.
    By the Cauchy-Schwarz inequality: $\mathbf{a} \cdot \mathbf{b} \leq \|\mathbf{a}\|\|\mathbf{b}\|$.
    Then $\|\mathbf{a} + \mathbf{b}\|^2 = \|\mathbf{a}\|^2 + \|\mathbf{b}\|^2 + 2(\mathbf{a} \cdot \mathbf{b}) \leq \|\mathbf{a}\|^2 + \|\mathbf{b}\|^2 + 2\|\mathbf{a}\|\|\mathbf{b}\| = (\|\mathbf{a}\| + \|\mathbf{b}\|)^2$.
    Taking square roots: $\|\mathbf{a} + \mathbf{b}\| \leq \|\mathbf{a}\| + \|\mathbf{b}\|$.
    Therefore $\|\mathbf{u} - \mathbf{w}\| \leq \|\mathbf{u} - \mathbf{v}\| + \|\mathbf{v} - \mathbf{w}\|$. QED.

12. Let $\mathbf{u} = \langle 1, 0 \rangle$, $\mathbf{v} = \langle 0, 1 \rangle$, $\mathbf{w} = \langle -1, 0 \rangle$.
    $d_{\cos}(\mathbf{u}, \mathbf{v}) = 1 - 0 = 1$.
    $d_{\cos}(\mathbf{v}, \mathbf{w}) = 1 - 0 = 1$.
    $d_{\cos}(\mathbf{u}, \mathbf{w}) = 1 - (-1) = 2$.
    Check triangle inequality: $d(\mathbf{u}, \mathbf{w}) \leq d(\mathbf{u}, \mathbf{v}) + d(\mathbf{v}, \mathbf{w})$?
    $2 \leq 1 + 1 = 2$. This satisfies equality, so this example does not violate it.
    Let's try: $\mathbf{u} = \langle 1, 0 \rangle$, $\mathbf{v} = \langle 1, 0.01 \rangle$, $\mathbf{w} = \langle -1, 0 \rangle$.
    $d_{\cos}(\mathbf{u}, \mathbf{v}) \approx 1 - 0.99995 = 0.00005$.
    $d_{\cos}(\mathbf{v}, \mathbf{w}) \approx 1 - (-0.99995) = 1.99995$.
    $d_{\cos}(\mathbf{u}, \mathbf{w}) = 1 - (-1) = 2$.
    Check: $2 \leq 0.00005 + 1.99995 = 2.00000$. Holds. Cosine distance can satisfy triangle inequality in some cases but not all.
    For a counterexample: $\mathbf{u} = \langle 1, 0 \rangle$, $\mathbf{v} = \langle 1/2, \sqrt{3}/2 \rangle$ (60 degrees), $\mathbf{w} = \langle -1/2, \sqrt{3}/2 \rangle$ (120 degrees from u).
    $d_{\cos}(\mathbf{u}, \mathbf{v}) = 1 - 0.5 = 0.5$.
    $d_{\cos}(\mathbf{v}, \mathbf{w}) = 1 - 0.5 = 0.5$.
    $d_{\cos}(\mathbf{u}, \mathbf{w}) = 1 - (-0.5) = 1.5$.
    Check: $1.5 \leq 0.5 + 0.5 = 1.0$? No! $1.5 > 1.0$, so the triangle inequality is violated. Cosine distance is not a metric.

13. Step 1: Define the squared distance matrix $D^{(2)}_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\|^2 = \|\mathbf{x}_i\|^2 + \|\mathbf{x}_j\|^2 - 2\mathbf{x}_i^T\mathbf{x}_j$.
    Step 2: Apply double centering: $B = -\frac{1}{2} J D^{(2)} J$.
    Step 3: For a column vector $\mathbf{c}$ with entries $\|\mathbf{x}_i\|^2$, write $D^{(2)} = \mathbf{c}\mathbf{1}^T + \mathbf{1}\mathbf{c}^T - 2X X^T$.
    Step 4: $J\mathbf{c}\mathbf{1}^T J = 0$ (since $J\mathbf{1} = 0$). Similarly $J\mathbf{1}\mathbf{c}^T J = 0$.
    Step 5: $B = -\frac{1}{2} J (-2X X^T) J = J X X^T J = (JX)(JX)^T$.
    Step 6: $JX$ is the matrix of centered points ($X$ with column means subtracted).
    Step 7: $B = (JX)(JX)^T$ is a Gram matrix, which is always positive semidefinite because for any vector $\mathbf{v}$, $\mathbf{v}^T B \mathbf{v} = \|(JX)^T \mathbf{v}\|^2 \geq 0$.

## Related Concepts

- **Vector Magnitude** (MATH-014): The norm $\|\mathbf{u}\|$ is the distance from $\mathbf{u}$ to the origin
- **Dot Product** (MATH-016): Euclidean distance relates to dot products via $\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2(\mathbf{u} \cdot \mathbf{v})$
- **Angle Between Vectors** (MATH-019): Cosine distance is derived from the angle
- **Vector Projection** (MATH-018): The perpendicular component's magnitude is the distance from $\mathbf{u}$ to the line of $\mathbf{v}$

## Next Concepts

- **040 k-Nearest Neighbours**: Classification/clustering using distance metrics
- **041 k-Means Clustering**: Centroid-based clustering using Euclidean distance
- **042 Vector Databases**: Efficient nearest neighbour search using distance
- **043 Cosine Similarity**: Deep dive into angular similarity measures
- **045 Curse of Dimensionality**: Why all distances converge in high dimensions

## Summary

Distance between vectors quantifies how far apart two points are in space. Euclidean distance (L2) is the straight-line distance, Manhattan distance (L1) is the grid-path distance, Chebyshev distance (L$\infty$) takes the maximum dimension-wise difference, and cosine distance measures angular dissimilarity. Each metric has different properties and is suited to different data types and algorithms. Distance metrics are the foundation of k-NN, k-means, vector databases, anomaly detection, and many other machine learning techniques.

## Key Takeaways

- Euclidean distance: $d_2(\mathbf{u}, \mathbf{v}) = \sqrt{\sum (u_i - v_i)^2}$ — straight-line distance
- Manhattan distance: $d_1(\mathbf{u}, \mathbf{v}) = \sum |u_i - v_i|$ — grid distance
- Chebyshev distance: $d_\infty = \max_i |u_i - v_i|$ — maximum coordinate difference
- Cosine distance: $d_{\cos} = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$ — angular difference
- Lp distances are true metrics (satisfy triangle inequality); cosine distance is not
- For unit vectors: $\|\mathbf{u} - \mathbf{v}\|^2 = 2 \cdot d_{\cos}(\mathbf{u}, \mathbf{v})$
- Distance is the core of k-NN, k-means, vector search, and embedding-based ML
- Scale your data before applying Euclidean distance; use cosine distance when direction matters more than magnitude
