# Concept: Principal Component Analysis

## Concept ID

MATH-043

## Difficulty

ADVANCED

## Domain

Mathematics

## Module

Linear Algebra

## Learning Objectives

- Explain PCA as an orthogonal linear transformation that maximizes variance in the projected subspace
- Centre the data and compute the covariance matrix
- Perform eigenvalue decomposition of the covariance matrix to obtain principal components
- Compute the proportion of variance explained by each principal component
- Perform PCA via SVD of the centred data matrix and relate the SVD formulation to the covariance formulation
- Project high-dimensional data onto a lower-dimensional subspace using the top-$k$ principal components
- Interpret the loadings (principal component directions) and scores in the context of real data
- Apply PCA for dimensionality reduction, visualization, noise reduction, and feature extraction in ML pipelines
- Understand the connection between PCA, SVD, and whitening

## Prerequisites

- Matrix multiplication and transpose
- Eigenvalues and eigenvectors (MATH-010)
- Singular Value Decomposition (MATH-042)
- Covariance and variance (basic statistics)
- Orthogonal projections
- Linear combinations of vectors

## Definition

**Principal Component Analysis** (PCA) is a statistical procedure that orthogonally transforms a set of possibly correlated variables into a set of linearly uncorrelated variables called **principal components**. The transformation is defined such that:

- The **first principal component** has the largest possible variance (i.e., accounts for as much of the variability in the data as possible).
- Each succeeding component has the highest possible variance under the constraint that it is **orthogonal** (uncorrelated) to the preceding components.

Given a data matrix $X \in \mathbb{R}^{n \times d}$ with $n$ observations and $d$ features, PCA seeks a set of orthonormal vectors $\mathbf{w}_1, \mathbf{w}_2, \dots, \mathbf{w}_d \in \mathbb{R}^d$ such that the projection of the centred data onto $\mathbf{w}_1$ has maximal variance, the projection onto $\mathbf{w}_2$ has maximal variance subject to being orthogonal to $\mathbf{w}_1$, and so on.

**Formal definition (covariance method):**

1. Centre the data: $\tilde{X} = X - \bar{X}$ where $\bar{X}$ is the $n \times d$ matrix where each row is the mean vector $\boldsymbol{\mu} = \frac{1}{n}\sum_{i=1}^n \mathbf{x}_i$.
2. Compute the sample covariance matrix: $C = \frac{1}{n-1} \tilde{X}^T \tilde{X} \in \mathbb{R}^{d \times d}$.
3. Perform eigenvalue decomposition: $C = V \Lambda V^T$ where $V$ is orthogonal and $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_d)$ with $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0$.
4. The **principal component directions** (loadings) are the columns of $V$ (eigenvectors of $C$).
5. The **principal component scores** for observation $\mathbf{x}_i$ are $\mathbf{t}_i = \tilde{\mathbf{x}}_i V \in \mathbb{R}^{d}$.
6. For dimensionality reduction to $k$ dimensions, project onto the first $k$ columns of $V$: $\tilde{X} V_k \in \mathbb{R}^{n \times k}$.

**Equivalent definition (SVD method):**

Perform SVD on the centred data matrix: $\tilde{X} = U \Sigma V^T$. Then:
- Principal components (loadings) = columns of $V$.
- Principal component scores = $U \Sigma = \tilde{X} V$.
- Singular values $\sigma_i = \sqrt{(n-1)\lambda_i}$, where $\lambda_i$ are eigenvalues of $C$.

## Intuition

Imagine you have a cloud of points in $d$-dimensional space. PCA finds the "most important" directions — the ones that capture the greatest spread of the data.

- The **first principal component** points in the direction of maximum variance. If you project all points onto this line, the spread (variance) is maximized.
- The **second principal component** picks the direction of maximum variance among all directions orthogonal to the first. It captures the next most important pattern.
- And so on.

Think of PCA as rotating the coordinate system to align with the natural axes of the data. If you have a cigar-shaped cloud of points, the first PC points along the long axis of the cigar. If the data lies roughly on a 2D plane in 100-dimensional space, the first two PCs span that plane.

The proportion of variance explained by the $i$-th PC is $\lambda_i / \sum_{j=1}^d \lambda_j$. If the first $k$ PCs explain 95% of the variance, we can safely reduce the data to $k$ dimensions without losing much information.

## Why This Concept Matters

PCA is arguably the most widely used dimensionality reduction technique in all of data science. It addresses the **curse of dimensionality**: as the number of features grows, the data becomes sparse, distances become meaningless, and models overfit. PCA provides a principled way to:

- **Simplify**: Reduce thousands of features to a handful of interpretable components.
- **Visualize**: Project high-dimensional data to 2D or 3D for plotting.
- **Denoise**: Remove low-variance components that often correspond to noise.
- **Decorrelate**: Transform correlated features into uncorrelated components.
- **Compress**: Store data using fewer numbers with minimal information loss.
- **Speed up ML**: Reduce feature count so that downstream algorithms train faster and generalize better.

PCA underlies eigenfaces, population genetics structure analysis (EIGENSTRAT), and quality control in genomics. It is a foundational tool in exploratory data analysis.

## Historical Background

PCA was invented independently multiple times:

- **Karl Pearson** (1901) first described the concept in "On Lines and Planes of Closest Fit to Systems of Points in Space." He derived the principal axes as the lines that minimize the orthogonal distances from points to the line (total least squares).
- **Harold Hotelling** (1933) independently developed PCA in a series of papers, giving it its modern name "Principal Component Analysis" and connecting it to the eigenvalue decomposition of the covariance matrix.
- **John Edward Jackson** (1950s-1980s) developed many of the control-chart-based applications of PCA.
- **Svante Wold** (1987) coined the term "multivariate statistical process control" using PCA.
- The connection between PCA and SVD was clarified in the 1960s-1970s as numerical linear algebra matured.

PCA remains one of the most active areas of methodological research, with extensions including kernel PCA, robust PCA, sparse PCA, and probabilistic PCA.

## Real World Examples

1. **Genetics (Population Structure)**: PCA applied to genome-wide SNP data reveals population structure. The first PC often corresponds to geographic ancestry, separating populations by continent. The 1000 Genomes Project uses PCA for quality control and population stratification correction.

2. **Face Recognition (Eigenfaces)**: In the classic "eigenfaces" approach (Turk & Pentland, 1991), each face image is treated as a high-dimensional vector. PCA finds the "eigenfaces" (principal components) that span the face space. New faces are recognized by projecting onto the first 100-200 eigenfaces, dramatically reducing the dimensionality from tens of thousands of pixels.

3. **Finance**: PCA decomposes asset returns into systematic risk factors. The first PC often represents the "market" factor, the second a "size" factor, and so on. Portfolio managers use PCA for risk decomposition and hedging.

4. **Neuroscience (EEG/fMRI)**: PCA reduces high-dimensional brain imaging data to a small number of spatial or temporal patterns. It removes noise and identifies dominant brain networks.

5. **Quality Control (Manufacturing)**: In semiconductor manufacturing, hundreds of sensors monitor each wafer. PCA identifies anomalous wafers by detecting large deviations in the principal component subspace (Hotelling's $T^2$ statistic).

## AI/ML Relevance

PCA is deeply embedded in machine learning both as a preprocessing step and as a core algorithm.

### Dimensionality Reduction

Given $n$ samples with $d$ features (where $d$ can be millions for images, text, or genomics), PCA reduces to $k \ll d$ features while retaining most information:

$$
X_{\text{reduced}} = \tilde{X} V_k \in \mathbb{R}^{n \times k}
$$

This speeds up training, reduces overfitting, and improves model generalization.

### Visualization

PCA projects data to 2D or 3D for visualization:

```python
import sklearn.decomposition as dec
pca = dec.PCA(n_components=2)
X_2d = pca.fit_transform(X)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels)
```

### Feature Extraction

The principal components are new features — linear combinations of the original features. This can reveal hidden structure. For example, in text analytics, PCA on the term-frequency matrix can extract latent topics.

### Noise Reduction

Assuming noise has small variance, it will be captured by later PCs. Reconstructing data using only the first $k$ PCs removes noise:

$$
X_{\text{denoised}} = \tilde{X} V_k V_k^T + \bar{X}
$$

### Preprocessing Before ML Models

Many models (linear regression, logistic regression, SVMs, k-NN) benefit from decorrelated features. PCA whitening transforms $X$ to have identity covariance, which stabilizes training:

$$
X_{\text{white}} = \tilde{X} V \Lambda^{-1/2}
$$

### Eigenfaces Example

In face recognition, a $256 \times 256$ image = 65,536-dimensional vector. PCA reduces to $k=100$ components. The reconstruction produces a recognizable face with 99.8% compression. The eigenfaces (columns of $V_k$) visually resemble face-like patterns (eyes, noses, lighting variations).

### PCA for Anomaly Detection

A new point $\mathbf{x}$ can be assessed by computing its reconstruction error:

$$
\text{SPE} = \|\tilde{\mathbf{x}} - \tilde{\mathbf{x}} V_k V_k^T\|^2
$$

Large squared prediction error (SPE) suggests an anomaly.

## Mathematical Explanation

### Derivation (Maximizing Variance)

Let $X \in \mathbb{R}^{n \times d}$ be the data matrix. Centre it: $\tilde{X} = X - \mathbf{1}\boldsymbol{\mu}^T$ where $\boldsymbol{\mu} = \frac{1}{n}\sum_{i=1}^n \mathbf{x}_i$.

We seek a unit vector $\mathbf{w} \in \mathbb{R}^d$ ($\|\mathbf{w}\| = 1$) such that the variance of the projected data $\tilde{X}\mathbf{w}$ is maximized.

The sample variance of the projection is:

$$
\text{Var}(\tilde{X}\mathbf{w}) = \frac{1}{n-1} \|\tilde{X}\mathbf{w}\|^2 = \frac{1}{n-1} \mathbf{w}^T \tilde{X}^T \tilde{X} \mathbf{w} = \mathbf{w}^T C \mathbf{w}
$$

where $C = \frac{1}{n-1} \tilde{X}^T \tilde{X}$ is the sample covariance matrix.

We maximize $\mathbf{w}^T C \mathbf{w}$ subject to $\mathbf{w}^T \mathbf{w} = 1$. Using a Lagrange multiplier $\lambda$:

$$
\mathcal{L}(\mathbf{w}, \lambda) = \mathbf{w}^T C \mathbf{w} - \lambda(\mathbf{w}^T \mathbf{w} - 1)
$$

Taking gradient and setting to zero:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 2C\mathbf{w} - 2\lambda \mathbf{w} = 0 \quad \Rightarrow \quad C\mathbf{w} = \lambda \mathbf{w}
$$

Thus $\mathbf{w}$ is an **eigenvector** of the covariance matrix $C$, and $\lambda$ is the corresponding eigenvalue. The variance of the projection is:

$$
\mathbf{w}^T C \mathbf{w} = \mathbf{w}^T \lambda \mathbf{w} = \lambda
$$

To maximize variance, we pick the eigenvector with the **largest eigenvalue**. This is the first principal component.

For the second PC, we constrain $\mathbf{w}_2 \perp \mathbf{w}_1$ and maximize $\mathbf{w}_2^T C \mathbf{w}_2$, yielding the eigenvector with the second-largest eigenvalue, and so on.

### The Full Solution

The eigenvalue decomposition of $C$ gives:

$$
C = V \Lambda V^T
$$

where $V = [\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_d]$ (columns are eigenvectors/PC directions) and $\Lambda = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_d)$ with $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0$.

The principal component scores are:

$$
T = \tilde{X} V \in \mathbb{R}^{n \times d}
$$

### Connection to SVD

Consider the SVD of the centred data:

$$
\tilde{X} = U \Sigma V^T
$$

Then:

$$
C = \frac{1}{n-1} \tilde{X}^T \tilde{X} = \frac{1}{n-1} (U \Sigma V^T)^T (U \Sigma V^T) = \frac{1}{n-1} V \Sigma^T U^T U \Sigma V^T = V \frac{\Sigma^T \Sigma}{n-1} V^T
$$

Thus:
- The right singular vectors $V$ are the principal component directions (same as eigenvectors of $C$).
- The eigenvalues of $C$ are $\lambda_i = \frac{\sigma_i^2}{n-1}$.
- The principal component scores $T = \tilde{X} V = U \Sigma V^T V = U \Sigma$.

The SVD approach is numerically more stable because it avoids computing $C$ explicitly.

## Formula(s)

### Centring
$$
\tilde{X} = X - \mathbf{1} \boldsymbol{\mu}^T, \quad \boldsymbol{\mu} = \frac{1}{n} \sum_{i=1}^n \mathbf{x}_i
$$

### Covariance Matrix
$$
C = \frac{1}{n-1} \tilde{X}^T \tilde{X}
$$

### Eigenvalue Decomposition
$$
C = V \Lambda V^T, \quad \Lambda = \text{diag}(\lambda_1, \dots, \lambda_d), \quad \lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0
$$

### Principal Component Scores
$$
T = \tilde{X} V, \quad T \in \mathbb{R}^{n \times d}
$$

### Dimensionality Reduction ($k$ components)
$$
X_{\text{reduced}} = \tilde{X} V_k \in \mathbb{R}^{n \times k}
$$

### Reconstruction (approximate)
$$
X_{\text{approx}} = \tilde{X} V_k V_k^T + \mathbf{1} \boldsymbol{\mu}^T
$$

### Variance Explained
$$
\text{VarExplained}_i = \frac{\lambda_i}{\sum_{j=1}^d \lambda_j}, \quad \text{Cumulative}_k = \frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^d \lambda_j}
$$

### PCA via SVD
$$
\tilde{X} = U \Sigma V^T, \quad \lambda_i = \frac{\sigma_i^2}{n-1}, \quad T = U \Sigma
$$

### Whitening (PCA Whitening)
$$
X_{\text{white}} = \tilde{X} V \Lambda^{-1/2} = U \sqrt{n-1}
$$

## Properties

1. **Maximal Variance**: The $k$-th PC direction maximizes the variance of the projected data among all directions orthogonal to the first $k-1$ PCs.
2. **Orthogonality**: Principal component directions are orthonormal: $V^T V = I$.
3. **Uncorrelated Scores**: The columns of $T$ (PC scores) are uncorrelated: $\text{Cov}(T_i, T_j) = 0$ for $i \neq j$.
4. **Variance = Eigenvalue**: $\text{Var}(T_{:,i}) = \lambda_i$.
5. **Total Variance Conserved**: $\sum_{i=1}^d \text{Var}(\text{original features}) = \sum_{i=1}^d \lambda_i = \text{tr}(C)$.
6. **Optimal Reconstruction**: PCA minimizes the sum of squared reconstruction errors $\|\tilde{X} - \tilde{X} V_k V_k^T\|_F^2$ among all rank-$k$ orthogonal projections (Eckart-Young via SVD).
7. **Scale Sensitivity**: PCA is not scale-invariant. Variables with larger variance dominate. Always standardize (z-score) features if they are in different units.
8. **Centring Required**: PCA must be performed on centred data. The first PC does not necessarily pass through the origin otherwise.
9. **Rotation Invariance**: PCA is rotationally equivariant: if we rotate the data, the PCs rotate correspondingly.
10. **Linearity**: PCA captures only linear relationships. For nonlinear structures, kernel PCA or manifold learning is needed.

## Step-by-Step Worked Examples

### Example 1: PCA on a Small 2D Dataset by Hand

Consider the following 2D dataset with $n = 5$ points:

$$
X = \begin{bmatrix}
1 & 1 \\
2 & 3 \\
3 & 5 \\
4 & 4 \\
5 & 2
\end{bmatrix}
$$

Each row is a point $(x_1, x_2)$.

**Step 1: Centre the data**

Compute the mean of each feature:

$$
\mu_1 = \frac{1 + 2 + 3 + 4 + 5}{5} = \frac{15}{5} = 3
$$

$$
\mu_2 = \frac{1 + 3 + 5 + 4 + 2}{5} = \frac{15}{5} = 3
$$

Subtract the mean:

$$
\tilde{X} = \begin{bmatrix}
-2 & -2 \\
-1 & 0 \\
0 & 2 \\
1 & 1 \\
2 & -1
\end{bmatrix}
$$

**Step 2: Compute the covariance matrix**

$$
C = \frac{1}{4} \tilde{X}^T \tilde{X}
$$

Compute $\tilde{X}^T \tilde{X}$:

- $(1,1)$: $(-2)^2 + (-1)^2 + 0^2 + 1^2 + 2^2 = 10$
- $(1,2)$: $(-2)(-2) + (-1)(0) + (0)(2) + (1)(1) + (2)(-1) = 3$
- $(2,1)$: same as $(1,2) = 3$
- $(2,2)$: $(-2)^2 + 0^2 + 2^2 + 1^2 + (-1)^2 = 10$

Thus:

$$
\tilde{X}^T \tilde{X} = \begin{bmatrix} 10 & 3 \\ 3 & 10 \end{bmatrix}
$$

$$
C = \frac{1}{4} \begin{bmatrix} 10 & 3 \\ 3 & 10 \end{bmatrix} = \begin{bmatrix} 2.5 & 0.75 \\ 0.75 & 2.5 \end{bmatrix}
$$

**Step 3: Compute eigenvalues and eigenvectors**

Solve $\det(C - \lambda I) = 0$:

$$
\det \begin{bmatrix} 2.5 - \lambda & 0.75 \\ 0.75 & 2.5 - \lambda \end{bmatrix} = (2.5 - \lambda)^2 - 0.75^2 = 0
$$

$$
(2.5 - \lambda)^2 = 0.5625
$$

$$
2.5 - \lambda = \pm 0.75
$$

So $\lambda_1 = 3.25$, $\lambda_2 = 1.75$. Sum = $5.0 = \text{tr}(C)$. ?

**Step 4: Find eigenvectors (PC directions)**

For $\lambda_1 = 3.25$:

$$
(C - 3.25I) \mathbf{v} = \begin{bmatrix} -0.75 & 0.75 \\ 0.75 & -0.75 \end{bmatrix} \mathbf{v} = \mathbf{0}
$$

$-0.75 v_1 + 0.75 v_2 = 0 \Rightarrow v_1 = v_2$.

$$
\mathbf{v}_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix} \approx \begin{bmatrix} 0.7071 \\ 0.7071 \end{bmatrix}
$$

For $\lambda_2 = 1.75$:

$$
(C - 1.75I) \mathbf{v} = \begin{bmatrix} 0.75 & 0.75 \\ 0.75 & 0.75 \end{bmatrix} \mathbf{v} = \mathbf{0}
$$

$0.75 v_1 + 0.75 v_2 = 0 \Rightarrow v_1 = -v_2$.

$$
\mathbf{v}_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \end{bmatrix} \approx \begin{bmatrix} 0.7071 \\ -0.7071 \end{bmatrix}
$$

**Step 5: Compute PC scores**

$$
T = \tilde{X} V = \begin{bmatrix}
-2 & -2 \\
-1 & 0 \\
0 & 2 \\
1 & 1 \\
2 & -1
\end{bmatrix}
\frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
$$

$t_{11} = (-2-2)/\sqrt{2} = -4/\sqrt{2} \approx -2.828$
$t_{12} = (-2+2)/\sqrt{2} = 0$

$t_{21} = (-1+0)/\sqrt{2} = -0.707$
$t_{22} = (-1+0)/\sqrt{2} = -0.707$

$t_{31} = (0+2)/\sqrt{2} = 1.414$
$t_{32} = (0-2)/\sqrt{2} = -1.414$

$t_{41} = (1+1)/\sqrt{2} = 1.414$
$t_{42} = (1-1)/\sqrt{2} = 0$

$t_{51} = (2-1)/\sqrt{2} = 0.707$
$t_{52} = (2+1)/\sqrt{2} = 2.121$

Thus:

$$
T = \begin{bmatrix}
-2.828 & 0 \\
-0.707 & -0.707 \\
1.414 & -1.414 \\
1.414 & 0 \\
0.707 & 2.121
\end{bmatrix}
$$

**Step 6: Verify variances**

Var(PC1) = $[8.0 + 0.5 + 2.0 + 2.0 + 0.5]/4 = 13.0/4 = 3.25 = \lambda_1$ ?

Var(PC2) = $[0 + 0.5 + 2.0 + 0 + 4.5]/4 = 7.0/4 = 1.75 = \lambda_2$ ?

Cov(PC1, PC2) = $[0 + 0.5 - 2.0 + 0 + 1.5]/4 = 0.0/4 = 0$ ? (uncorrelated)

**Step 7: Variance explained**

PC1: $3.25/5.0 = 65\%$, PC2: $1.75/5.0 = 35\%$

Reducing to 1D retains 65% of variance.

**Step 8: Verify via SVD**

$\tilde{X}^T \tilde{X} = \begin{bmatrix} 10 & 3 \\ 3 & 10 \end{bmatrix}$ has eigenvalues 13 and 7.

$\sigma_1 = \sqrt{13} \approx 3.606$, $\sigma_2 = \sqrt{7} \approx 2.646$.

$\lambda_1 = 13/4 = 3.25$ ?, $\lambda_2 = 7/4 = 1.75$ ?

$V$ from SVD of $\tilde{X}$ = $\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$ ?

### Example 2: PCA for Dimensionality Reduction (2D to 1D)

Continue with the data from Example 1. Reduce to 1D and reconstruct.

**Step 1: Project to 1D**

Keep only the first PC: $V_1 = \begin{bmatrix} 0.7071 \\ 0.7071 \end{bmatrix}$.

$$
T_1 = \tilde{X} V_1 = \begin{bmatrix} -2.828 \\ -0.707 \\ 1.414 \\ 1.414 \\ 0.707 \end{bmatrix}
$$

**Step 2: Reconstruct in original space**

$\tilde{X}_{\text{approx}} = T_1 V_1^T$

Point 1: $(-2.828)\begin{bmatrix}0.7071\\0.7071\end{bmatrix} = \begin{bmatrix}-2.0\\-2.0\end{bmatrix}$, add mean: $\begin{bmatrix}1.0\\1.0\end{bmatrix}$

Point 2: $(-0.707)\begin{bmatrix}0.7071\\0.7071\end{bmatrix} = \begin{bmatrix}-0.5\\-0.5\end{bmatrix}$, add mean: $\begin{bmatrix}2.5\\2.5\end{bmatrix}$

Point 3: $(1.414)\begin{bmatrix}0.7071\\0.7071\end{bmatrix} = \begin{bmatrix}1.0\\1.0\end{bmatrix}$, add mean: $\begin{bmatrix}4.0\\4.0\end{bmatrix}$

Point 4: $(1.414)\begin{bmatrix}0.7071\\0.7071\end{bmatrix} = \begin{bmatrix}1.0\\1.0\end{bmatrix}$, add mean: $\begin{bmatrix}4.0\\4.0\end{bmatrix}$

Point 5: $(0.707)\begin{bmatrix}0.7071\\0.7071\end{bmatrix} = \begin{bmatrix}0.5\\0.5\end{bmatrix}$, add mean: $\begin{bmatrix}3.5\\3.5\end{bmatrix}$

**Step 3: Reconstruction error**

| Point | Original | Reconstructed | Squared Error |
|-------|----------|---------------|---------------|
| 1 | (1, 1) | (1.0, 1.0) | 0 |
| 2 | (2, 3) | (2.5, 2.5) | 0.5 |
| 3 | (3, 5) | (4.0, 4.0) | 2.0 |
| 4 | (4, 4) | (4.0, 4.0) | 0 |
| 5 | (5, 2) | (3.5, 3.5) | 4.5 |

Total squared error = $7.0 = \sigma_2^2$. Frobenius error = $\sqrt{7} \approx 2.646 = \sigma_2$. Confirms Eckart-Young.

### Example 3: PCA on a 3D Dataset

Given:

$$
X = \begin{bmatrix}
1 & 4 & 2 \\
2 & 5 & 3 \\
3 & 6 & 4 \\
4 & 3 & 5 \\
5 & 2 & 6 \\
6 & 1 & 7
\end{bmatrix}
$$

**Step 1: Compute means**

$\mu_1 = 3.5$, $\mu_2 = 3.5$, $\mu_3 = 4.5$

**Step 2: Centre the data**

$$
\tilde{X} = \begin{bmatrix}
-2.5 & 0.5 & -2.5 \\
-1.5 & 1.5 & -1.5 \\
-0.5 & 2.5 & -0.5 \\
0.5 & -0.5 & 0.5 \\
1.5 & -1.5 & 1.5 \\
2.5 & -2.5 & 2.5
\end{bmatrix}
$$

**Step 3: Covariance matrix**

$C = \frac{1}{5} \tilde{X}^T \tilde{X}$:

$$
\tilde{X}^T \tilde{X} = \begin{bmatrix}
17.5 & -13.5 & 17.5 \\
-13.5 & 17.5 & -13.5 \\
17.5 & -13.5 & 17.5
\end{bmatrix},\quad
C = \begin{bmatrix}
3.5 & -2.7 & 3.5 \\
-2.7 & 3.5 & -2.7 \\
3.5 & -2.7 & 3.5
\end{bmatrix}
$$

**Step 4: Eigenvalues**

Row 1 = Row 3, so rank = 2. One eigenvalue is 0.

$\mathbf{v}_3 = [1, 0, -1]^T/\sqrt{2}$ is eigenvector for $\lambda_3 = 0$ (check: $C\mathbf{v}_3 = \mathbf{0}$).

Solve for non-zero eigenvalues using symmetry $[a, b, a]^T$:

$C[a, b, a]^T = [7a - 2.7b, -5.4a + 3.5b, 7a - 2.7b]^T$

Setting $C\mathbf{v} = \lambda \mathbf{v}$:

$7a - 2.7b = \lambda a$ and $-5.4a + 3.5b = \lambda b$

From first: $b = \frac{7-\lambda}{2.7}a$. Substituting into second:

$\lambda^2 - 10.5\lambda + 9.92 = 0$

$\lambda = \frac{10.5 \pm \sqrt{70.57}}{2} = \frac{10.5 \pm 8.40}{2}$

$\lambda_1 \approx 9.45$, $\lambda_2 \approx 1.05$, $\lambda_3 = 0$

Sum = $10.5 = \text{tr}(C)$. ?

**Step 5: Eigenvectors**

For $\lambda_1 = 9.45$: $b = \frac{7-9.45}{2.7}a \approx -0.907a$

$\mathbf{v}_1 \propto [1, -0.907, 1]^T$, normalized:

$$
\mathbf{v}_1 \approx \begin{bmatrix} 0.595 \\ -0.540 \\ 0.595 \end{bmatrix}
$$

For $\lambda_2 = 1.05$: $b = \frac{7-1.05}{2.7}a \approx 2.204a$

$\mathbf{v}_2 \propto [1, 2.204, 1]^T$, normalized:

$$
\mathbf{v}_2 \approx \begin{bmatrix} 0.382 \\ 0.842 \\ 0.382 \end{bmatrix}
$$

**Step 6: Variance explained**

PC1: $9.45/10.5 = 90\%$, PC2: $1.05/10.5 = 10\%$, PC3: $0\%$

The data lies almost on a line in 3D space (90% of variance captured by first PC).

## Visual Interpretation

Geometrically, PCA can be understood as:

1. **Centring**: Shift the data so its centroid is at the origin.
2. **Finding the first PC**: Place a line through the origin. Rotate it until the sum of squared distances from points to their projections on the line is minimized (equivalently, the variance of the projected points is maximized).
3. **Finding the second PC**: Place a second line through the origin, orthogonal to the first, again maximizing the variance of projections.
4. **And so on**.

In 2D, the PCs are the axes of the **data ellipse** (the ellipse that best fits the scatterplot). The lengths of the ellipse axes are proportional to $\sqrt{\lambda_i}$.

The "scree plot" (eigenvalues vs. component index) shows the variance captured by each PC. An elbow in the scree plot suggests the optimal number of components to keep.

The "biplot" overlays the original feature vectors on the PC score plot, showing which features contribute most to each component.

## Common Mistakes

1. **Not centring the data**: PCA without centring finds components that pass through the origin, which may not align with the directions of maximum variance. The first component might simply capture the mean.

2. **Not standardizing when features have different units**: If features are in different scales (e.g., height in cm and weight in kg), the feature with larger variance dominates PCA. Always use the correlation matrix (standardize to z-scores) when features are not commensurate.

3. **Confusing loadings and scores**: Loadings are the eigenvectors/PC directions (columns of $V$), giving the weights of original features. Scores are the transformed coordinates ($T = \tilde{X}V$), one row per observation.

4. **Assuming PCA components are interpretable**: PCs are linear combinations of all original features. The loadings may be dense (non-zero entries for all features), making interpretation difficult. Sparse PCA addresses this.

5. **Using PCA without checking for linearity**: PCA captures only linear structure. If the data lies on a nonlinear manifold (e.g., a Swiss roll), PCA will fail to capture the true structure.

6. **Over-interpreting explained variance**: A high cumulative explained variance (e.g., 95%) does not guarantee that the reduced data preserves all important structure. Some phenomena (e.g., rare events) may have small variance but be critically important.

7. **Applying PCA on the wrong matrix**: Some textbooks define PCA on the covariance matrix; others on the correlation matrix. They give different results. Choose based on whether features have comparable units.

8. **Forgetting that PCA maximizes variance, not separation**: PCA is unsupervised. It does not know about class labels. A 2D PCA projection may not show class separation well even if classes are clearly separable in the original space.

9. **Ignoring the $n < d$ case**: When $n < d$ (more features than samples), the covariance matrix is singular. PCA is still valid, but at most $n-1$ PCs have non-zero variance. Use SVD on $\tilde{X}$ directly.

10. **Thinking PCA removes all correlations**: PCA decorrelates features in the training set. But new test data, when projected, may still show correlations in the reduced space if the test distribution differs.

## Interview Questions

### Beginner

**Q1**: What is Principal Component Analysis?

**A1**: PCA is an unsupervised dimensionality reduction technique that transforms a set of possibly correlated features into a smaller set of uncorrelated features called principal components. Each principal component is a linear combination of the original features, ordered by the amount of variance they capture from the data.

**Q2**: How many principal components can you compute for a dataset with $d$ features?

**A2**: At most $d$ principal components (equal to the number of features). However, if $n < d$ (fewer samples than features), at most $n-1$ components have non-zero variance.

**Q3**: What does "variance explained" mean in PCA?

**A3**: Variance explained by the $i$-th PC is the ratio $\lambda_i / \sum_{j=1}^d \lambda_j$, where $\lambda_i$ is the eigenvalue associated with that PC. It represents the proportion of total dataset variance captured by that component.

**Q4**: Should you centre the data before PCA?

**A4**: Yes. PCA always requires centred data (mean-subtracted). Without centring, the first PC would be pulled toward the mean rather than the direction of maximum variance.

**Q5**: How can you decide how many principal components to keep?

**A5**: Common methods: (1) Scree plot elbow — look for the point where eigenvalues level off. (2) Cumulative variance threshold — keep enough PCs to explain 90-95% of variance. (3) Kaiser rule — keep PCs with eigenvalue $> 1$ (for correlation matrix). (4) Cross-validation — evaluate downstream task performance for different $k$.

### Intermediate

**Q1**: Derive PCA as the solution to a variance-maximization problem.

**A1**: Given centred data $\tilde{X}$, we seek a unit vector $\mathbf{w}$ maximizing $\text{Var}(\tilde{X}\mathbf{w}) = \mathbf{w}^T C \mathbf{w}$ where $C = \frac{1}{n-1}\tilde{X}^T\tilde{X}$. Using Lagrange multiplier $\lambda$, set $\nabla(\mathbf{w}^T C \mathbf{w} - \lambda(\mathbf{w}^T\mathbf{w} - 1)) = 0$, giving $C\mathbf{w} = \lambda\mathbf{w}$. Thus $\mathbf{w}$ is an eigenvector of $C$, and the variance is $\lambda$. To maximize variance, take the eigenvector with the largest eigenvalue.

**Q2**: Explain the relationship between PCA and SVD.

**A2**: Let $\tilde{X}$ be the centred data matrix. Compute SVD: $\tilde{X} = U\Sigma V^T$. Then the covariance matrix $C = \frac{1}{n-1}\tilde{X}^T\tilde{X} = V\frac{\Sigma^T\Sigma}{n-1}V^T$. The principal component directions are the columns of $V$ (right singular vectors). The eigenvalues of $C$ are $\lambda_i = \sigma_i^2/(n-1)$. The PC scores are $T = \tilde{X}V = U\Sigma$.

**Q3**: When would you use the correlation matrix instead of the covariance matrix for PCA?

**A3**: Use the correlation matrix (standardize to z-scores) when features are measured in different units or have substantially different variances. The covariance matrix is appropriate when all features are in the same units and comparable scales.

**Q4**: What is the computational complexity of PCA?

**A4**: Standard PCA via eigenvalue decomposition of $C$: $O(d^3 + nd^2)$. Via SVD of $\tilde{X}$: $O(nd \min(n,d))$. For large $d$, randomized PCA takes $O(ndk)$ for $k$ components.

**Q5**: How would you use PCA for anomaly detection?

**A5**: Fit PCA on normal data. For a new point $\mathbf{x}$, centre it and compute its reconstruction $\hat{\mathbf{x}} = \bar{X} + \tilde{\mathbf{x}} V_k V_k^T$. Compute the squared reconstruction error (SPE): $\|\tilde{\mathbf{x}} - \tilde{\mathbf{x}} V_k V_k^T\|^2$. If this exceeds a threshold, label as anomalous.

### Advanced

**Q1**: Prove that PCA gives the best rank-$k$ linear approximation to the data in terms of minimizing reconstruction error.

**A1**: We minimize $\|\tilde{X} - \tilde{X} W_k W_k^T\|_F^2$ over orthonormal $W_k$. This equals $\|\tilde{X}\|_F^2 - \|\tilde{X} W_k\|_F^2$, so minimizing error = maximizing $\|\tilde{X} W_k\|_F^2 = (n-1)\text{tr}(W_k^T C W_k)$. By the Ky Fan theorem, the maximizer is the top $k$ eigenvectors of $C$, giving minimum error $\sum_{i=k+1}^d \lambda_i$ (times $n-1$). Equivalent to Eckart-Young on $\tilde{X}$.

**Q2**: Explain how PCA relates to whitening and when you would use ZCA whitening instead.

**A2**: PCA whitening: $X_{\text{white}} = \tilde{X} V \Lambda^{-1/2} = U \sqrt{n-1}$. ZCA whitening: $X_{\text{ZCA-white}} = \tilde{X} V \Lambda^{-1/2} V^T = U \sqrt{n-1} V^T$. ZCA rotates back, preserving the original feature orientations. ZCA is preferred for images because whitened images look like edge-enhanced originals; PCA-whitened images are unrecognizable.

**Q3**: Derive probabilistic PCA (PPCA) and relate it to standard PCA.

**A3**: PPCA models $\mathbf{x}_i = W \mathbf{z}_i + \boldsymbol{\mu} + \boldsymbol{\epsilon}_i$ with $\mathbf{z}_i \sim \mathcal{N}(0, I_k)$ and $\boldsymbol{\epsilon}_i \sim \mathcal{N}(0, \sigma^2 I_d)$. MLE: $W_{\text{MLE}} = V_k (\Lambda_k - \sigma^2 I)^{1/2} R$, where $\sigma^2 = \frac{1}{d-k}\sum_{i=k+1}^d \lambda_i$. As $\sigma^2 \to 0$, latent variables converge to PCA scores. PPCA enables Bayesian extensions, missing data imputation, and model selection via likelihood.

## Practice Problems

### Easy

1. For $X = \begin{bmatrix} 1 & 2 \\ 2 & 4 \\ 3 & 6 \end{bmatrix}$, find the first PC direction.

2. A covariance matrix has eigenvalues $\lambda = [10, 4, 3, 2, 1]$. What % of variance is explained by the first two PCs?

3. If $\mathbf{v}_1 = [0.6, 0.8]^T$ and $\tilde{\mathbf{x}} = [3, -1]^T$, what is the PC1 score?

4. True or False: PCA components are always orthogonal to each other.

5. How many PCs can you extract from a dataset with 10 features and 100 samples?

### Medium

1. For $X = \begin{bmatrix} 0 & 0 \\ 0 & 2 \\ 2 & 0 \\ 2 & 2 \end{bmatrix}$, perform PCA by hand. Find PC directions and variance explained.

2. A dataset has $C = \begin{bmatrix} 4 & 2 \\ 2 & 3 \end{bmatrix}$. Find PCs and variance explained.

3. Given $\tilde{X} = \begin{bmatrix} 1 & -1 \\ -1 & 1 \\ 0 & 0 \end{bmatrix}$, compute SVD and use it to find PCs and scores.

4. Standardize $X = \begin{bmatrix} 10 & 100 \\ 20 & 200 \\ 30 & 300 \end{bmatrix}$ and perform PCA. Explain why standardization is needed.

5. Given $V = \begin{bmatrix} 0.5 & 0.5 \\ 0.5 & -0.5 \\ 0.5 & 0.5 \\ 0.5 & -0.5 \end{bmatrix}$, interpret PC1. What pattern does it capture?

### Hard

1. Prove PCs are eigenvectors of $C$ by solving $\max_{\mathbf{w}} \mathbf{w}^T C \mathbf{w}$ s.t. $\mathbf{w}^T\mathbf{w}=1$ and $\mathbf{w}^T\mathbf{v}_1=0$ (second PC).

2. For $X = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix}$, compute PCA via SVD. Reconstruct using only PC1 and compute error. Verify it matches $\sigma_2^2$.

3. Given $\tilde{X} = U\Sigma V^T$, show PC scores $T = \tilde{X}V = U\Sigma$ have zero mean and diagonal covariance. Prove columns of $T$ are uncorrelated.

## Solutions

### Easy Solutions

**Solution 1**: $X = \begin{bmatrix} 1 & 2 \\ 2 & 4 \\ 3 & 6 \end{bmatrix}$.

Mean: $\mu_1 = 2$, $\mu_2 = 4$. Centred: $\tilde{X} = \begin{bmatrix} -1 & -2 \\ 0 & 0 \\ 1 & 2 \end{bmatrix}$.

$C = \frac{1}{2} \tilde{X}^T \tilde{X} = \frac{1}{2} \begin{bmatrix} 2 & 4 \\ 4 & 8 \end{bmatrix} = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$.

Eigenvalues: $\det \begin{bmatrix} 1-\lambda & 2 \\ 2 & 4-\lambda \end{bmatrix} = \lambda^2 - 5\lambda = \lambda(\lambda-5) = 0$.

$\lambda_1 = 5$, $\lambda_2 = 0$.

For $\lambda_1 = 5$: $\begin{bmatrix} -4 & 2 \\ 2 & -1 \end{bmatrix} \Rightarrow v_2 = 2v_1$.

$\mathbf{v}_1 = \frac{1}{\sqrt{5}}[1, 2]^T \approx [0.447, 0.894]^T$.

**Solution 2**: Total variance = $10+4+3+2+1 = 20$. First two: $14$. Percentage = $14/20 = 70\%$.

**Solution 3**: $t_1 = \tilde{\mathbf{x}}^T \mathbf{v}_1 = 3(0.6) + (-1)(0.8) = 1.8 - 0.8 = 1.0$.

**Solution 4**: True. PCs are eigenvectors of a symmetric matrix (covariance matrix), which are orthogonal for distinct eigenvalues.

**Solution 5**: $\min(d, n-1) = \min(10, 99) = 10$ PCs.

### Medium Solutions

**Solution 1**: $X = \begin{bmatrix} 0 & 0 \\ 0 & 2 \\ 2 & 0 \\ 2 & 2 \end{bmatrix}$.

Mean: $\mu_1 = 1$, $\mu_2 = 1$. Centred: $\tilde{X} = \begin{bmatrix} -1 & -1 \\ -1 & 1 \\ 1 & -1 \\ 1 & 1 \end{bmatrix}$.

$C = \frac{1}{3} \tilde{X}^T \tilde{X} = \frac{1}{3} \begin{bmatrix} 4 & 0 \\ 0 & 4 \end{bmatrix} = \begin{bmatrix} 4/3 & 0 \\ 0 & 4/3 \end{bmatrix}$.

$\lambda_1 = \lambda_2 = 4/3$. Equal eigenvalues ? any orthonormal basis is valid. 50% variance each.

**Solution 2**: $C = \begin{bmatrix} 4 & 2 \\ 2 & 3 \end{bmatrix}$.

$\det \begin{bmatrix} 4-\lambda & 2 \\ 2 & 3-\lambda \end{bmatrix} = \lambda^2 - 7\lambda + 8 = 0$.

$\lambda = \frac{7 \pm \sqrt{17}}{2}$. $\lambda_1 \approx 5.562$, $\lambda_2 \approx 1.438$.

PC1 explains $5.562/7 = 79.5\%$, PC2 explains $20.5\%$.

For $\lambda_1$: $(4-5.562)v_1 + 2v_2 = 0 \Rightarrow v_2 = 0.781v_1$.

$\mathbf{v}_1 \approx [0.788, 0.615]^T$, $\mathbf{v}_2 \approx [-0.615, 0.788]^T$.

**Solution 3**: $\tilde{X} = \begin{bmatrix} 1 & -1 \\ -1 & 1 \\ 0 & 0 \end{bmatrix}$.

$\tilde{X}^T \tilde{X} = \begin{bmatrix} 2 & -2 \\ -2 & 2 \end{bmatrix}$. Eigenvalues: 4, 0. $\sigma_1 = 2$, $\sigma_2 = 0$.

For $\lambda=4$: $\begin{bmatrix} -2 & -2 \\ -2 & -2 \end{bmatrix} \Rightarrow v_1 = -v_2$. $\mathbf{v}_1 = \frac{1}{\sqrt{2}}[1, -1]^T$.

$\mathbf{v}_2 = \frac{1}{\sqrt{2}}[1, 1]^T$.

$U_1 = \frac{\tilde{X} V_1}{\sigma_1} = \frac{1}{2} \begin{bmatrix} 1 & -1 \\ -1 & 1 \\ 0 & 0 \end{bmatrix} \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \end{bmatrix} = \frac{1}{2\sqrt{2}}\begin{bmatrix} 2 \\ -2 \\ 0 \end{bmatrix} = \begin{bmatrix} 1/\sqrt{2} \\ -1/\sqrt{2} \\ 0 \end{bmatrix}$.

Scores: $T = U\Sigma = \begin{bmatrix} \sqrt{2} \\ -\sqrt{2} \\ 0 \end{bmatrix}$ (column vector). PCs = columns of $V$.

**Solution 4**: Standardize: $z_{ij} = (x_{ij} - \mu_j)/\sigma_j$.
$\mu_1 = 20$, $\sigma_1 \approx 8.165$; $\mu_2 = 200$, $\sigma_2 \approx 81.65$.
$Z = \begin{bmatrix} -1.225 & -1.225 \\ 0 & 0 \\ 1.225 & 1.225 \end{bmatrix}$.

$C_Z = \begin{bmatrix} 1.5 & 1.5 \\ 1.5 & 1.5 \end{bmatrix}$. Eigenvalues: 3, 0. PC1 = $[1,1]^T/\sqrt{2}$.

Standardization is necessary because feature 2 has 10x the variance of feature 1; without it, PC1 would almost entirely be feature 2.

**Solution 5**: PC1 = $[0.5, 0.5, 0.5, 0.5]^T$ — all features with equal positive weight. This captures the "average" or "overall size" pattern. PC2 = $[0.5, -0.5, 0.5, -0.5]^T$ — contrasts features 1 & 3 vs 2 & 4.

### Hard Solutions

**Solution 1**: For the second PC, maximize $\mathbf{w}^T C \mathbf{w}$ s.t. $\mathbf{w}^T\mathbf{w}=1$ and $\mathbf{w}^T\mathbf{v}_1=0$. Lagrangian: $\mathcal{L} = \mathbf{w}^T C \mathbf{w} - \lambda(\mathbf{w}^T\mathbf{w}-1) - \mu\mathbf{w}^T\mathbf{v}_1$. Gradient: $2C\mathbf{w} - 2\lambda\mathbf{w} - \mu\mathbf{v}_1 = 0$. Left-multiply by $\mathbf{v}_1^T$: $2\mathbf{v}_1^T C \mathbf{w} - 2\lambda\mathbf{v}_1^T\mathbf{w} - \mu\mathbf{v}_1^T\mathbf{v}_1 = 0$. Since $\mathbf{v}_1$ is eigenvector of $C$ with eigenvalue $\lambda_1$, $\mathbf{v}_1^T C = \lambda_1\mathbf{v}_1^T$. So $2\lambda_1\mathbf{v}_1^T\mathbf{w} - 0 - \mu = 0 \Rightarrow \mu = 0$. Then $C\mathbf{w} = \lambda\mathbf{w}$. With orthogonality constraint, $\mathbf{w}$ must be eigenvector orthogonal to $\mathbf{v}_1$: the eigenvector for $\lambda_2$.

**Solution 2**: $X = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix}$.

Mean: $\mu_1 = 3$, $\mu_2 = 4$.

$\tilde{X} = \begin{bmatrix} -2 & -2 \\ 0 & 0 \\ 2 & 2 \end{bmatrix}$.

$\tilde{X}^T \tilde{X} = \begin{bmatrix} 8 & 8 \\ 8 & 8 \end{bmatrix}$. Eigenvalues: 16, 0. $\sigma_1 = 4$, $\sigma_2 = 0$.

$V_1 = \frac{1}{\sqrt{2}}[1, 1]^T$, $V_2 = \frac{1}{\sqrt{2}}[1, -1]^T$.

$U_1 = \frac{\tilde{X} V_1}{\sigma_1} = \frac{1}{4} \begin{bmatrix} -2 & -2 \\ 0 & 0 \\ 2 & 2 \end{bmatrix} \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix} = \frac{1}{4\sqrt{2}}\begin{bmatrix} -4 \\ 0 \\ 4 \end{bmatrix} = \begin{bmatrix} -1/\sqrt{2} \\ 0 \\ 1/\sqrt{2} \end{bmatrix}$.

PC scores (1D): $T_1 = \tilde{X} V_1 = \begin{bmatrix} -2\sqrt{2} \\ 0 \\ 2\sqrt{2} \end{bmatrix}$.

Reconstruction: $\tilde{X}_{\text{approx}} = T_1 V_1^T = \begin{bmatrix} -2\sqrt{2} \\ 0 \\ 2\sqrt{2} \end{bmatrix} \frac{1}{\sqrt{2}}[1, 1] = \begin{bmatrix} -2 & -2 \\ 0 & 0 \\ 2 & 2 \end{bmatrix} = \tilde{X}$.

Since $\sigma_2 = 0$, reconstruction is exact. Reconstruction error = $0 = \sigma_2^2$. ?

**Solution 3**: $T = \tilde{X}V = U\Sigma V^T V = U\Sigma$.

Mean of scores: $\frac{1}{n}\sum_{i=1}^n \mathbf{t}_i = \frac{1}{n} \mathbf{1}^T U\Sigma$.

But $\mathbf{1}^T \tilde{X} = \mathbf{0}^T$ (columns of centred data sum to zero), so $\mathbf{1}^T U\Sigma V^T = \mathbf{0}^T \Rightarrow \mathbf{1}^T U\Sigma = \mathbf{0}^T$ (since $V$ is invertible). Thus scores have zero mean.

Covariance: $\frac{1}{n-1} T^T T = \frac{1}{n-1} \Sigma^T U^T U \Sigma = \frac{1}{n-1} \Sigma^T \Sigma = \Lambda$ (diagonal).

Thus $\text{Cov}(T_i, T_j) = 0$ for $i \neq j$ (uncorrelated), and $\text{Var}(T_{:,i}) = \lambda_i$. ?

## Related Concepts

- **Singular Value Decomposition (MATH-042)**: PCA = SVD on centred data. The right singular vectors are the PC directions; singular values relate to eigenvalues via $\sigma_i^2 = (n-1)\lambda_i$.
- **Factor Analysis**: A related latent variable model that distinguishes shared vs unique variance. PCA explains total variance; factor analysis explains only common variance.
- **Independent Component Analysis (ICA)**: Finds statistically independent (not just uncorrelated) components. PCA is a preprocessing step for ICA (whitening).
- **Kernel PCA**: Nonlinear extension using the kernel trick to find PCs in a high-dimensional feature space.
- **t-SNE / UMAP**: Nonlinear dimensionality reduction methods for visualization. Unlike PCA, they preserve local neighbourhood structure rather than global variance.
- **Linear Discriminant Analysis (LDA)**: Supervised dimensionality reduction that maximizes class separation rather than variance.
- **Canonical Correlation Analysis (CCA)**: Finds linear relationships between two sets of variables; generalizes PCA to paired datasets.
- **Random Projection**: An alternative dimensionality reduction method based on the Johnson-Lindenstrauss lemma, computationally cheaper than PCA for massive datasets.

## Next Concepts

- **Kernel PCA**: Generalizes PCA to nonlinear manifolds using kernel functions.
- **Probabilistic PCA (PPCA)**: Provides a probabilistic framework for PCA, enabling Bayesian treatment, missing data, and model comparison.
- **Sparse PCA**: Produces PCs with sparse loadings (many zeros), improving interpretability.
- **Robust PCA**: Decomposes data into low-rank + sparse components, handling outliers robustly.
- **Autoencoders**: Neural network-based nonlinear dimensionality reduction; PCA is a linear autoencoder with a single hidden layer.
- **Matrix Factorization**: PCA is a special case; other factorizations (NMF, SVD++) are widely used in recommendation systems.

## Summary

Principal Component Analysis (PCA) is the most widely used linear dimensionality reduction technique. It finds orthogonal directions (principal components) that maximize the variance of projected data, obtained as the eigenvectors of the covariance matrix (or equivalently, the right singular vectors of the centred data matrix via SVD). The eigenvalues quantify the variance captured by each component. PCA decorrelates features, removes noise, enables visualization, and speeds up downstream ML algorithms. It is equivalent to SVD on centred data, with principal components = right singular vectors $V$ and scores = $U\Sigma$. The proportion of variance explained guides the choice of how many components to retain. PCA can also be used for whitening (decorrelation + variance normalization) and anomaly detection via reconstruction error.

## Key Takeaways

1. **PCA = eigenvalue decomposition of covariance matrix $C$** or **SVD of centred data $\tilde{X}$**.
2. **Principal components** = eigenvectors of $C$ (columns of $V$), sorted by decreasing eigenvalue.
3. **Scores** $T = \tilde{X}V = U\Sigma$ give the data coordinates in PC space.
4. **Variance explained** by PC $i$ = $\lambda_i / \sum \lambda_j$. A scree plot shows diminishing returns.
5. **Dimensionality reduction**: keep top-$k$ PCs to reduce $d$ to $k$ features while maximizing retained variance.
6. **PCA decorrelates**: transformed features have diagonal covariance matrix (uncorrelated).
7. **Standardize when features differ in scale**; use correlation matrix PCA for mixed-unit data.
8. **PCA is linear**: it captures only linear relationships; use kernel PCA for nonlinear structure.
9. **Optimal reconstruction**: PCA minimizes $\|\tilde{X} - \tilde{X} V_k V_k^T\|_F^2$ among all rank-$k$ projections (Eckart-Young).
10. **Widely used in ML**: dimensionality reduction, visualization, denoising, whitening, preprocessing, anomaly detection, eigenfaces, and genetics.
