# Concept: Singular Value Decomposition

## Concept ID

MATH-042

## Difficulty

ADVANCED

## Domain

Mathematics

## Module

Linear Algebra

## Learning Objectives

- Decompose any real or complex matrix $A \in \mathbb{R}^{m \times n}$ into the product $U\Sigma V^T$ and interpret each factor geometrically and algebraically
- Compute the full and reduced (thin/economy) SVD for small matrices by hand
- Relate singular values and singular vectors to eigenvalues and eigenvectors of $AA^T$ and $A^TA$
- Apply the Eckart-Young theorem to construct optimal low-rank approximations $A_k$
- Derive and use the Moore-Penrose pseudo-inverse $A^+$ via SVD
- Implement dimensionality reduction, recommendation systems, and image compression using SVD
- Explain data whitening and its connection to SVD in the context of machine learning pipelines

## Prerequisites

- Matrix multiplication, matrix transpose, matrix inverse
- Eigenvalues and eigenvectors (MATH-010)
- Orthogonal matrices and orthonormal bases
- Vector norms and inner products
- Rank of a matrix
- Basic understanding of linear transformations

## Definition

The **Singular Value Decomposition** (SVD) of a real $m \times n$ matrix $A$ is a factorization of the form:

$$
A = U \Sigma V^T
$$

where:

- $U$ is an $m \times m$ **orthogonal matrix** ($U^T U = I_m$). Its columns are the **left singular vectors** of $A$. These are the eigenvectors of $AA^T$.
- $V$ is an $n \times n$ **orthogonal matrix** ($V^T V = I_n$). Its columns are the **right singular vectors** of $A$. These are the eigenvectors of $A^T A$.
- $\Sigma$ is an $m \times n$ **diagonal matrix** containing the **singular values** $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_p \ge 0$ along its main diagonal, where $p = \min(m, n)$. The singular values are the square roots of the eigenvalues of $A^T A$ (or equivalently $AA^T$): $\sigma_i = \sqrt{\lambda_i}$.

The SVD exists for **every** matrix — real or complex, square or rectangular, full rank or rank-deficient. This universality is one of its most powerful properties.

**Full SVD:**

$$
A_{m \times n} = U_{m \times m} \Sigma_{m \times n} V^T_{n \times n}
$$

**Reduced (Thin/Economy) SVD:**

When $m \ge n$, we can write the reduced form where $\Sigma$ is $n \times n$ diagonal and $U$ is $m \times n$ with orthonormal columns:

$$
A_{m \times n} = U_{m \times n} \Sigma_{n \times n} V^T_{n \times n}
$$

For $m \le n$, the reduced SVD keeps $\Sigma$ as $m \times m$ and $V$ as $n \times m$.

**Rank-$k$ Truncated SVD:**

If we keep only the $k$ largest singular values ($k < r = \text{rank}(A)$), we obtain the truncated SVD:

$$
A_k = U_k \Sigma_k V_k^T
$$

where $U_k$ is $m \times k$, $\Sigma_k$ is $k \times k$, and $V_k$ is $n \times k$.

## Intuition

Geometrically, SVD reveals that **any linear transformation** can be decomposed into three elementary steps:

1. **Rotation/Reflection** ($V^T$): The input vector is rotated (or reflected) by the orthogonal matrix $V^T$ from the standard basis into the basis of right singular vectors.
2. **Scaling** ($\Sigma$): The rotated vector is stretched or compressed along each coordinate axis by the singular values $\sigma_i$. If $\sigma_i = 0$, that direction is completely flattened (projected to zero).
3. **Rotation/Reflection** ($U$): The scaled vector is rotated (or reflected) by the orthogonal matrix $U$ into the output coordinate system.

$$
A\mathbf{x} = U (\Sigma (V^T \mathbf{x}))
$$

Think of it this way: every matrix sends the unit sphere to an ellipsoid. The left singular vectors $U$ give the **principal axes directions** of that ellipsoid in the output space. The singular values $\sigma_i$ give the **lengths of those axes**. The right singular vectors $V$ are the **pre-images** of those axes — the input directions that map to the principal axes.

The rank of $A$ equals the number of non-zero singular values. If $\sigma_k \gg \sigma_{k+1}$, the matrix can be well-approximated by a rank-$k$ matrix.

## Why This Concept Matters

SVD is arguably the most numerically stable and revealing matrix factorization. Every matrix has an SVD; it is the culmination of linear algebra's fundamental theorems. It simultaneously provides:

- The **rank** of the matrix (number of non-zero singular values)
- The **2-norm** (largest singular value: $\|A\|_2 = \sigma_1$)
- The **Frobenius norm** ($\|A\|_F = \sqrt{\sum \sigma_i^2}$)
- The **condition number** ($\kappa(A) = \sigma_1 / \sigma_k$ for rank-$k$)
- The **range** (column space), **nullspace**, **row space**, and **left nullspace** directly from $U$ and $V$

The Eckart-Young theorem states that the best rank-$k$ approximation to $A$ in both the Frobenius and spectral norms is given by the truncated SVD $A_k = U_k \Sigma_k V_k^T$. This is the foundation of virtually all dimensionality reduction and compression techniques in data science.

## Historical Background

The SVD was developed over more than a century by several mathematicians:

- **Eugenio Beltrami** (1873) first derived the decomposition for real square matrices in his paper "Sulle funzioni bilineari."
- **Camille Jordan** (1874) independently discovered it and connected it to the theory of quadratic forms.
- **James Joseph Sylvester** (1889) extended the work to rectangular matrices in "On the reduction of a bilinear form."
- **Erhard Schmidt** (1907) developed the infinite-dimensional analogue (now called the Schmidt decomposition in quantum mechanics).
- **Hermann Weyl** (1912) and **Carl Eckart & Gale Young** (1936) proved the low-rank approximation theorem.
- **Gene Golub** and **William Kahan** (1965) published the first practical numerical algorithm for computing SVD, a milestone in numerical linear algebra.

The SVD became a cornerstone of numerical computing after Golub and Christian Reinsch published the widely-used Golub-Reinsch SVD algorithm in 1970.

## Real World Examples

1. **Image Compression**: A grayscale image is an $m \times n$ matrix of pixel intensities. By keeping only the top 10-100 singular values (out of potentially thousands), we can reconstruct a highly compressed but recognizable image. The storage drops from $m \times n$ to $k(m + n + 1)$. This is the principle behind early JPEG-like compression.

2. **Recommendation Systems**: In collaborative filtering (e.g., Netflix Prize), a user-item rating matrix $R$ is highly sparse. SVD-based matrix factorization finds low-rank approximations $R \approx U\Sigma V^T$ that predict missing ratings. Users and items are represented by $k$-dimensional latent vectors, and the predicted rating for user $i$ on item $j$ is $(\mathbf{u}_i)^T (\sigma) (\mathbf{v}_j)$.

3. **Google's PageRank**: While PageRank primarily uses eigenvector analysis, SVD variants (especially in related HITS algorithm by Jon Kleinberg) use singular vectors to identify authoritative pages and hubs.

4. **MRI Image Reconstruction**: SVD is used to denoise MRI images and to compress the large volumes of data generated by medical scanners.

5. **Genomics**: SVD (often called Principal Component Analysis in this context, since PCA = SVD on centred data) is used to identify patterns in gene expression data across thousands of genes and experimental conditions.

## AI/ML Relevance

SVD is deeply embedded in modern machine learning:

### Dimensionality Reduction

Given a data matrix $X \in \mathbb{R}^{n \times d}$ of $n$ samples and $d$ features, truncated SVD reduces the feature dimension from $d$ to $k$ while preserving as much variance as possible:

$$
X_{\text{reduced}} = X V_k \in \mathbb{R}^{n \times k}
$$

This is equivalent to PCA (when data is centred). It removes noise, speeds up downstream algorithms, and mitigates the curse of dimensionality.

### Recommendation Systems (Collaborative Filtering)

The Netflix prize-winning solution used SVD for matrix factorization. Given a sparse user-item matrix $R \in \mathbb{R}^{m \times n}$, we factorize:

$$
R \approx U_k \Sigma_k V_k^T
$$

Each user $i$ gets a $k$-dimensional latent vector $\mathbf{p}_i = \mathbf{u}_i \Sigma_k$. Each item $j$ gets $\mathbf{q}_j = \mathbf{v}_j$. Predicted rating: $\hat{r}_{ij} = \mathbf{p}_i^T \mathbf{q}_j$.

### Matrix Completion

Building on SVD, matrix completion algorithms (like those used in Netflix Prize) find a low-rank matrix that matches observed entries. The nuclear norm ($\|\cdot\|_* = \sum \sigma_i$) is used as a convex relaxation of rank.

### Image Compression

```python
# Pseudocode for SVD-based image compression
U, S, Vt = svd(image_matrix)
k = 50  # keep top 50 singular values
compressed = U[:, :k] @ diag(S[:k]) @ Vt[:k, :]
# Compression ratio: (m*n) / (k*(m + n + 1))
```

### Data Whitening (Sphering)

Whitening transforms data to have identity covariance. Using SVD of the centred data $X = U\Sigma V^T$, the whitened data is:

$$
Z = U \quad \text{or} \quad Z = X V \Sigma^{-1} = U \Sigma V^T V \Sigma^{-1} = U
$$

This decorrelates features and normalizes variance, a crucial preprocessing step for ICA and some neural network architectures.

### Pseudo-inverse for Ordinary Least Squares

For a linear system $A\mathbf{x} = \mathbf{b}$, the least-squares solution is $\mathbf{x} = A^+ \mathbf{b}$, where the Moore-Penrose pseudo-inverse is:

$$
A^+ = V \Sigma^+ U^T
$$

where $\Sigma^+$ is formed by taking the reciprocal of each non-zero singular value and transposing. This is numerically superior to solving the normal equations.

### Topic Modelling (LSI/LSA)

Latent Semantic Indexing (LSI) applies SVD to a term-document matrix. The left singular vectors represent "topics" — groups of related terms — and the right singular vectors represent documents expressed in terms of those topics.

## Mathematical Explanation

### Derivation from Eigenvalue Decomposition

Consider $A \in \mathbb{R}^{m \times n}$. The matrix $A^T A \in \mathbb{R}^{n \times n}$ is symmetric positive semi-definite (all eigenvalues $\ge 0$). By the Spectral Theorem, $A^T A$ admits an eigenvalue decomposition:

$$
A^T A = V \Lambda V^T
$$

where $V$ is orthogonal and $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_n)$ with $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n \ge 0$.

Define the singular values: $\sigma_i = \sqrt{\lambda_i}$ for $i = 1, \dots, n$.

Similarly, $AA^T \in \mathbb{R}^{m \times m}$ has eigenvalues $\lambda_1, \dots, \lambda_m$ (with $\lambda_i = 0$ for $i > \text{rank}(A)$) and eigenvectors $U$:

$$
AA^T = U \Lambda_m U^T
$$

Now, how are $U$ and $V$ related? For $i = 1, \dots, r$ where $r = \text{rank}(A)$:

$$
A V_i = \sigma_i U_i \quad \text{and} \quad A^T U_i = \sigma_i V_i
$$

Proof:
$$
A^T A V_i = \lambda_i V_i \quad \Rightarrow \quad V_i^T A^T A V_i = \lambda_i
$$
$$
\|A V_i\|^2 = \lambda_i \quad \Rightarrow \quad \|A V_i\| = \sigma_i
$$
Let $U_i = \frac{A V_i}{\sigma_i}$. Then $A V_i = \sigma_i U_i$.

For the remaining $m - r$ left singular vectors, extend $\{U_1, \dots, U_r\}$ to an orthonormal basis of $\mathbb{R}^m$.

### Full Decomposition

We can now write:

$$
A V = U \Sigma
$$

where $V = [V_1 | \dots | V_n]$, $U = [U_1 | \dots | U_m]$, and $\Sigma_{ii} = \sigma_i$ for $i \le p = \min(m,n)$, zero elsewhere.

Since $V$ is orthogonal, $V^{-1} = V^T$, giving:

$$
A = U \Sigma V^T
$$

## Formula(s)

### Full SVD
$$
A_{m \times n} = U_{m \times m} \Sigma_{m \times n} V_{n \times n}^T
$$

### Reduced (Thin) SVD ($m \ge n$)
$$
A_{m \times n} = U_{m \times n} \Sigma_{n \times n} V_{n \times n}^T
$$

### Rank-$k$ Truncated SVD
$$
A_k = U_{m \times k} \Sigma_{k \times k} V_{n \times k}^T
$$

### Singular Values
$$
\sigma_i = \sqrt{\lambda_i (A^T A)} = \sqrt{\lambda_i (AA^T)}
$$
$$
\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_p \ge 0, \quad p = \min(m, n)
$$

### Pseudo-inverse
$$
A^+ = V \Sigma^+ U^T
$$
where $\Sigma^+$ is $n \times m$, with $1/\sigma_i$ on the diagonal for $\sigma_i > 0$, zero elsewhere.

### Low-rank Approximation Error (Eckart-Young)
$$
\|A - A_k\|_2 = \sigma_{k+1}
$$
$$
\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}
$$

## Properties

1. **Existence**: Every real (or complex) matrix has an SVD.
2. **Uniqueness**: Singular values are unique. Singular vectors are unique up to sign (or multiplication by $-1$) when singular values are distinct.
3. **Rank Revealing**: $\text{rank}(A) =$ number of non-zero singular values.
4. **Norms**: $\|A\|_2 = \sigma_1$, $\|A\|_F = \sqrt{\sum \sigma_i^2}$, $\|A\|_* = \sum \sigma_i$ (nuclear norm).
5. **Condition Number**: $\kappa(A) = \sigma_1 / \sigma_r$ where $r = \text{rank}(A)$. Large ratio $\Rightarrow$ ill-conditioned.
6. **Range and Nullspace**:
   - Columns of $U$ corresponding to non-zero $\sigma_i$ span $\text{Col}(A)$.
   - Columns of $U$ corresponding to zero $\sigma_i$ span $\text{Null}(A^T)$.
   - Columns of $V$ corresponding to non-zero $\sigma_i$ span $\text{Row}(A)$.
   - Columns of $V$ corresponding to zero $\sigma_i$ span $\text{Null}(A)$.
7. **Invariance**: Singular values are invariant under orthogonal transformations: if $P$ and $Q$ are orthogonal, then $PAQ^T$ has the same singular values as $A$.
8. **Relationship to Eigenvalues**: For symmetric $A$, SVD = eigenvalue decomposition (up to signs of singular vectors).
9. **Submultiplicativity**: $\sigma_1(AB) \le \sigma_1(A) \sigma_1(B)$.
10. **Interlacing**: For a submatrix $B$ of $A$, $\sigma_i(B) \le \sigma_i(A)$.

## Step-by-Step Worked Examples

### Example 1: Full SVD of a $2 \times 2$ Matrix

Compute the full SVD of:

$$
A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}
$$

**Step 1: Compute $A^T A$**

$$
A^T A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}^T \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}
= \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix} \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}
= \begin{bmatrix} 10 & 6 \\ 6 & 10 \end{bmatrix}
$$

**Step 2: Find eigenvalues of $A^T A$**

Solve $\det(A^T A - \lambda I) = 0$:

$$
\det \begin{bmatrix} 10 - \lambda & 6 \\ 6 & 10 - \lambda \end{bmatrix} = (10 - \lambda)^2 - 36 = 0
$$

$$
(10 - \lambda)^2 = 36
$$

$$
10 - \lambda = \pm 6
$$

$$
\lambda_1 = 16, \quad \lambda_2 = 4
$$

**Step 3: Singular values**

$$
\sigma_1 = \sqrt{16} = 4, \quad \sigma_2 = \sqrt{4} = 2
$$

Note the singular values are positive and sorted: $\sigma_1 \ge \sigma_2$. The condition number is $\kappa = \sigma_1/\sigma_2 = 4/2 = 2$.

**Step 4: Find eigenvectors of $A^T A$ (columns of $V$)**

For $\lambda_1 = 16$:

$$
(A^T A - 16I) \mathbf{v} = \begin{bmatrix} -6 & 6 \\ 6 & -6 \end{bmatrix} \mathbf{v} = \mathbf{0}
$$

This gives $-6v_1 + 6v_2 = 0$, so $v_1 = v_2$. Eigenvector: $\begin{bmatrix} 1 \\ 1 \end{bmatrix}$. Normalize:

$$
V_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

For $\lambda_2 = 4$:

$$
(A^T A - 4I) \mathbf{v} = \begin{bmatrix} 6 & 6 \\ 6 & 6 \end{bmatrix} \mathbf{v} = \mathbf{0}
$$

$6v_1 + 6v_2 = 0$, so $v_2 = -v_1$. Eigenvector: $\begin{bmatrix} 1 \\ -1 \end{bmatrix}$. Normalize:

$$
V_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \end{bmatrix}
$$

Thus:

$$
V = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{bmatrix}
$$

**Step 5: Find $U$ using $U_i = \frac{A V_i}{\sigma_i}$**

$$
A V_1 = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix} \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}
= \frac{1}{\sqrt{2}} \begin{bmatrix} 3 \cdot 1 + 1 \cdot 1 \\ 1 \cdot 1 + 3 \cdot 1 \end{bmatrix}
= \frac{1}{\sqrt{2}} \begin{bmatrix} 4 \\ 4 \end{bmatrix}
= \begin{bmatrix} \frac{4}{\sqrt{2}} \\ \frac{4}{\sqrt{2}} \end{bmatrix}
$$

$$
U_1 = \frac{A V_1}{\sigma_1} = \frac{1}{4} \begin{bmatrix} \frac{4}{\sqrt{2}} \\ \frac{4}{\sqrt{2}} \end{bmatrix}
= \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix}
$$

$$
A V_2 = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix} \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \end{bmatrix}
= \frac{1}{\sqrt{2}} \begin{bmatrix} 3 \cdot 1 + 1 \cdot (-1) \\ 1 \cdot 1 + 3 \cdot (-1) \end{bmatrix}
= \frac{1}{\sqrt{2}} \begin{bmatrix} 2 \\ -2 \end{bmatrix}
= \begin{bmatrix} \frac{2}{\sqrt{2}} \\ -\frac{2}{\sqrt{2}} \end{bmatrix}
$$

$$
U_2 = \frac{A V_2}{\sigma_2} = \frac{1}{2} \begin{bmatrix} \frac{2}{\sqrt{2}} \\ -\frac{2}{\sqrt{2}} \end{bmatrix}
= \begin{bmatrix} \frac{1}{\sqrt{2}} \\ -\frac{1}{\sqrt{2}} \end{bmatrix}
$$

Since $m=n=2$, both singular vectors are found. Thus:

$$
U = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{bmatrix}, \quad
\Sigma = \begin{bmatrix} 4 & 0 \\ 0 & 2 \end{bmatrix}, \quad
V = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{bmatrix}
$$

**Step 6: Verify**

$$
U \Sigma V^T = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{bmatrix}
\begin{bmatrix} 4 & 0 \\ 0 & 2 \end{bmatrix}
\begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{bmatrix}^T
$$

$$
= \begin{bmatrix} \frac{4}{\sqrt{2}} & \frac{2}{\sqrt{2}} \\ \frac{4}{\sqrt{2}} & -\frac{2}{\sqrt{2}} \end{bmatrix}
\begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{bmatrix}
$$

$$
= \begin{bmatrix} \frac{4}{2} + \frac{2}{2} & \frac{4}{2} - \frac{2}{2} \\ \frac{4}{2} - \frac{2}{2} & \frac{4}{2} + \frac{2}{2} \end{bmatrix}
= \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix} = A \quad \checkmark
$$

Note that here $U = V$ because $A$ is symmetric.

### Example 2: SVD of a Rectangular $3 \times 2$ Matrix (Reduced SVD)

Compute the reduced SVD of:

$$
A = \begin{bmatrix} 1 & 2 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}
$$

**Step 1: Compute $A^T A$**

$$
A^T A = \begin{bmatrix} 1 & 0 & 0 \\ 2 & 0 & 0 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}
= \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}
$$

**Step 2: Eigenvalues of $A^T A$**

$$
\det \begin{bmatrix} 1-\lambda & 2 \\ 2 & 4-\lambda \end{bmatrix} = (1-\lambda)(4-\lambda) - 4 = \lambda^2 - 5\lambda + 4 - 4 = \lambda^2 - 5\lambda = \lambda(\lambda - 5)
$$

Eigenvalues: $\lambda_1 = 5$, $\lambda_2 = 0$.

**Step 3: Singular values**

$$
\sigma_1 = \sqrt{5} \approx 2.236, \quad \sigma_2 = 0
$$

Since $\sigma_2 = 0$, $\text{rank}(A) = 1$.

**Step 4: Eigenvectors of $A^T A$ ($V$)**

For $\lambda_1 = 5$:

$$
\begin{bmatrix} 1-5 & 2 \\ 2 & 4-5 \end{bmatrix} = \begin{bmatrix} -4 & 2 \\ 2 & -1 \end{bmatrix}
$$

$-4v_1 + 2v_2 = 0 \Rightarrow v_2 = 2v_1$. Eigenvector: $\begin{bmatrix} 1 \\ 2 \end{bmatrix}$. Normalize: $\| \begin{bmatrix} 1 \\ 2 \end{bmatrix} \| = \sqrt{1 + 4} = \sqrt{5}$.

$$
V_1 = \frac{1}{\sqrt{5}} \begin{bmatrix} 1 \\ 2 \end{bmatrix}
$$

For $\lambda_2 = 0$:

$$
\begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}
$$

$v_1 + 2v_2 = 0 \Rightarrow v_1 = -2v_2$. Eigenvector: $\begin{bmatrix} -2 \\ 1 \end{bmatrix}$. Normalize:

$$
V_2 = \frac{1}{\sqrt{5}} \begin{bmatrix} -2 \\ 1 \end{bmatrix}
$$

Thus:

$$
V = \begin{bmatrix} \frac{1}{\sqrt{5}} & -\frac{2}{\sqrt{5}} \\ \frac{2}{\sqrt{5}} & \frac{1}{\sqrt{5}} \end{bmatrix}
$$

**Step 5: Find $U$ (only $U_1$ since $\sigma_2 = 0$)**

$$
U_1 = \frac{A V_1}{\sigma_1} = \frac{1}{\sqrt{5}} \begin{bmatrix} 1 & 2 \\ 0 & 0 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} \cdot \frac{1}{\sqrt{5}}? 
$$

Let's compute carefully:

$$
A V_1 = \begin{bmatrix} 1 & 2 \\ 0 & 0 \\ 0 & 0 \end{bmatrix} \frac{1}{\sqrt{5}} \begin{bmatrix} 1 \\ 2 \end{bmatrix}
= \frac{1}{\sqrt{5}} \begin{bmatrix} 1\cdot 1 + 2 \cdot 2 \\ 0 \\ 0 \end{bmatrix}
= \frac{1}{\sqrt{5}} \begin{bmatrix} 5 \\ 0 \\ 0 \end{bmatrix}
$$

$$
U_1 = \frac{A V_1}{\sigma_1} = \frac{1}{\sqrt{5}} \begin{bmatrix} 5 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} \sqrt{5} \\ 0 \\ 0 \end{bmatrix}?
$$

Wait, let me recalculate. $\sigma_1 = \sqrt{5}$, so:

$$
U_1 = \frac{1}{\sqrt{5}} \cdot \frac{1}{\sqrt{5}} \begin{bmatrix} 5 \\ 0 \\ 0 \end{bmatrix} = \frac{1}{5} \begin{bmatrix} 5 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}
$$

We need to extend $U_1$ to an orthonormal basis of $\mathbb{R}^3$. We can pick any two vectors orthogonal to $U_1$:

$$
U_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, \quad U_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
$$

**Reduced SVD ($m=3, n=2$, keep $\min(m,n)=2$ columns):**

$$
A = U \Sigma V^T
$$

$$
U = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}, \quad
\Sigma = \begin{bmatrix} \sqrt{5} & 0 \\ 0 & 0 \end{bmatrix}, \quad
V = \begin{bmatrix} \frac{1}{\sqrt{5}} & -\frac{2}{\sqrt{5}} \\ \frac{2}{\sqrt{5}} & \frac{1}{\sqrt{5}} \end{bmatrix}
$$

**Verify:**

$$
U \Sigma V^T = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}
\begin{bmatrix} \sqrt{5} & 0 \\ 0 & 0 \end{bmatrix}
\begin{bmatrix} \frac{1}{\sqrt{5}} & \frac{2}{\sqrt{5}} \\ -\frac{2}{\sqrt{5}} & \frac{1}{\sqrt{5}} \end{bmatrix}
$$

First compute $U\Sigma$:

$$
U\Sigma = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}
\begin{bmatrix} \sqrt{5} & 0 \\ 0 & 0 \end{bmatrix}
= \begin{bmatrix} \sqrt{5} & 0 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}
$$

Then $U\Sigma V^T$:

$$
= \begin{bmatrix} \sqrt{5} & 0 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}
\begin{bmatrix} \frac{1}{\sqrt{5}} & \frac{2}{\sqrt{5}} \\ -\frac{2}{\sqrt{5}} & \frac{1}{\sqrt{5}} \end{bmatrix}
= \begin{bmatrix} \frac{\sqrt{5}}{\sqrt{5}} & \frac{2\sqrt{5}}{\sqrt{5}} \\ 0 & 0 \\ 0 & 0 \end{bmatrix}
= \begin{bmatrix} 1 & 2 \\ 0 & 0 \\ 0 & 0 \end{bmatrix} = A \quad \checkmark
$$

### Example 3: Low-Rank Approximation via Truncated SVD (Eckart-Young)

Given:

$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}
$$

Find the best rank-1 approximation $A_1$ and compute the approximation error.

**Step 1: Compute the full SVD (we'll compute key parts)**

First, $A^T A$:

$$
A^T A = \begin{bmatrix} 1 & 4 & 7 \\ 2 & 5 & 8 \\ 3 & 6 & 9 \end{bmatrix}
\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}
$$

$$
A^T A_{11} = 1\cdot 1 + 4\cdot 4 + 7\cdot 7 = 1 + 16 + 49 = 66
$$
$$
A^T A_{12} = 1\cdot 2 + 4\cdot 5 + 7\cdot 8 = 2 + 20 + 56 = 78
$$
$$
A^T A_{13} = 1\cdot 3 + 4\cdot 6 + 7\cdot 9 = 3 + 24 + 63 = 90
$$
$$
A^T A_{21} = 78, \quad A^T A_{22} = 4 + 25 + 64 = 93, \quad A^T A_{23} = 6 + 30 + 72 = 108
$$
$$
A^T A_{31} = 90, \quad A^T A_{32} = 108, \quad A^T A_{33} = 9 + 36 + 81 = 126
$$

$$
A^T A = \begin{bmatrix} 66 & 78 & 90 \\ 78 & 93 & 108 \\ 90 & 108 & 126 \end{bmatrix}
$$

**Step 2: Eigenvalues**

$A^T A$ has eigenvalues: $\lambda_1 \approx 283.19$, $\lambda_2 \approx 1.81$, $\lambda_3 \approx 0$ (actually exactly 0). We can see $\text{rank}(A) = 2$ because the rows are linearly dependent (row 3 - row 2 = row 2 - row 1).

Check: $[7, 8, 9] = 2[4, 5, 6] - [1, 2, 3]$, so rows are linearly dependent. The third row is a linear combination of the first two. Thus $\text{rank}(A) = 2$.

Eigenvalues: $\lambda_1 = \frac{1}{2}(285 + \sqrt{285^2 - 4\cdot 9 \cdot 9})$... Let's use the known result: for this particular matrix, the non-zero eigenvalues of $A^T A$ are approximately 283.19 and 1.81.

Singular values:
$$
\sigma_1 \approx \sqrt{283.19} \approx 16.83, \quad \sigma_2 \approx \sqrt{1.81} \approx 1.35, \quad \sigma_3 = 0
$$

**Step 3: Top singular vectors**

For $\lambda_1 \approx 283.19$:

The dominant eigenvector of $A^T A$ (corresponding to $\sigma_1$) is approximately:

$$
V_1 \approx \begin{bmatrix} 0.214 \\ 0.517 \\ 0.830 \end{bmatrix}
$$

(properly normalized)

$$
U_1 = \frac{A V_1}{\sigma_1}
$$

Compute $A V_1$:

$$
A V_1 \approx \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}
\begin{bmatrix} 0.214 \\ 0.517 \\ 0.830 \end{bmatrix}
$$

$$
= \begin{bmatrix} 1(0.214) + 2(0.517) + 3(0.830) \\ 4(0.214) + 5(0.517) + 6(0.830) \\ 7(0.214) + 8(0.517) + 9(0.830) \end{bmatrix}
$$

$$
= \begin{bmatrix} 0.214 + 1.034 + 2.490 \\ 0.856 + 2.585 + 4.980 \\ 1.498 + 4.136 + 7.470 \end{bmatrix}
= \begin{bmatrix} 3.738 \\ 8.421 \\ 13.104 \end{bmatrix}
$$

$$
U_1 = \frac{1}{16.83} \begin{bmatrix} 3.738 \\ 8.421 \\ 13.104 \end{bmatrix}
\approx \begin{bmatrix} 0.222 \\ 0.500 \\ 0.779 \end{bmatrix}
$$

**Step 4: Rank-1 approximation**

$$
A_1 = U_1 \sigma_1 V_1^T
$$

$$
A_1 \approx \begin{bmatrix} 0.222 \\ 0.500 \\ 0.779 \end{bmatrix}
16.83 \begin{bmatrix} 0.214 & 0.517 & 0.830 \end{bmatrix}
$$

$$
= 16.83 \begin{bmatrix} 0.222 \cdot 0.214 & 0.222 \cdot 0.517 & 0.222 \cdot 0.830 \\
0.500 \cdot 0.214 & 0.500 \cdot 0.517 & 0.500 \cdot 0.830 \\
0.779 \cdot 0.214 & 0.779 \cdot 0.517 & 0.779 \cdot 0.830 \end{bmatrix}
$$

$$
= 16.83 \begin{bmatrix} 0.0475 & 0.1148 & 0.1843 \\
0.1070 & 0.2585 & 0.4150 \\
0.1667 & 0.4027 & 0.6466 \end{bmatrix}
$$

$$
A_1 \approx \begin{bmatrix} 0.80 & 1.93 & 3.10 \\
1.80 & 4.35 & 6.98 \\
2.81 & 6.78 & 10.88 \end{bmatrix}
$$

**Step 5: Compute error**

$$
\|A - A_1\|_2 = \sigma_2 \approx 1.35
$$

$$
\|A - A_1\|_F = \sqrt{\sigma_2^2 + \sigma_3^2} = \sqrt{1.81} \approx 1.35
$$

Check one element: $A_{11} - (A_1)_{11} = 1 - 0.80 = 0.20$.

The rank-1 approximation captures the dominant trend (all entries increasing left-to-right and top-to-bottom) but misses the finer structure.

## Visual Interpretation

Geometrically, SVD tells us that a matrix $A$ maps the unit sphere in $\mathbb{R}^n$ to an ellipsoid in $\mathbb{R}^m$:

$$
\{A\mathbf{x} : \|\mathbf{x}\| = 1\} = \{\mathbf{y} : \sum_{i=1}^m \frac{(\mathbf{u}_i^T \mathbf{y})^2}{\sigma_i^2} = 1\}
$$

- The **directions** of the ellipsoid axes are the left singular vectors $\mathbf{u}_i$ (columns of $U$).
- The **lengths** of the axes are the singular values $\sigma_i$.
- The **pre-images** of the axes directions are the right singular vectors $\mathbf{v}_i$ (columns of $V$).

For a $2 \times 2$ matrix, this can be visualized as:
1. Start with the unit circle.
2. Apply $V^T$: rotate the circle (no shape change).
3. Apply $\Sigma$: stretch along axes by $\sigma_1$, $\sigma_2$ — the circle becomes an ellipse.
4. Apply $U$: rotate the ellipse to its final orientation.

When we truncate to rank $k$, we are approximating the ellipsoid by keeping only the $k$ longest axes — the most "important" directions.

## Common Mistakes

1. **Confusing left and right singular vectors**: $U$ (columns = eigenvectors of $AA^T$) is in the output space; $V$ (columns = eigenvectors of $A^T A$) is in the input space. Always verify dimensions: $U$ is $m \times m$, $V$ is $n \times n$.

2. **Forgetting to sort singular values**: SVD requires $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_p \ge 0$. The corresponding singular vectors must be reordered consistently.

3. **Assuming $U = V$**: This only happens when $A$ is symmetric (and even then, signs may differ). In general, $U$ and $V$ are completely different matrices.

4. **Confusing SVD with eigenvalue decomposition**: Eigenvalue decomposition requires a square, diagonalizable matrix. SVD works for any matrix. For symmetric positive definite matrices, they coincide, but this is a special case.

5. **Incorrectly computing the pseudo-inverse**: $\Sigma^+$ has reciprocals of *non-zero* singular values. Zero singular values remain zero. The transpose is also needed: if $\Sigma$ is $m \times n$, then $\Sigma^+$ is $n \times m$.

6. **Misunderstanding the truncated SVD error**: The Eckart-Young theorem says $\|A - A_k\|_2 = \sigma_{k+1}$, not $\sigma_k$. The error depends on the *next* singular value, not the $k$th one.

7. **Not centring data for PCA**: PCA via SVD requires the data matrix to be centred (mean-subtracted). Applying SVD directly to raw data yields a different decomposition.

8. **Treating SVD as unique**: Singular vectors can be multiplied by $-1$ and still be valid. When singular values repeat, any orthonormal basis of the corresponding subspace works.

9. **Ignoring numerical stability**: Computing $A^T A$ explicitly can lose precision. Numerical SVD algorithms (e.g., Golub-Reinsch) work directly on $A$.

10. **Assuming all singular values matter**: In practice, small singular values often represent noise. The "effective rank" is determined by a threshold on $\sigma_i$.

## Interview Questions

### Beginner

**Q1**: What is the Singular Value Decomposition of a matrix $A$?

**A1**: SVD factorizes any matrix $A \in \mathbb{R}^{m \times n}$ as $A = U\Sigma V^T$, where $U$ and $V$ are orthogonal matrices whose columns are left and right singular vectors respectively, and $\Sigma$ is a diagonal matrix containing the singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$.

**Q2**: How do you compute the singular values of a matrix $A$?

**A2**: Singular values are the square roots of the eigenvalues of $A^T A$ (or $AA^T$). Specifically, $\sigma_i = \sqrt{\lambda_i(A^T A)}$, where $\lambda_i$ are the eigenvalues of $A^T A$.

**Q3**: What is the relationship between the rank of a matrix and its singular values?

**A3**: The rank of $A$ equals the number of non-zero singular values. If $\sigma_r > 0$ and $\sigma_{r+1} = 0$, then $\text{rank}(A) = r$.

**Q4**: What is the reduced (thin) SVD?

**A4**: For $A \in \mathbb{R}^{m \times n}$ with $m \ge n$, the reduced SVD keeps only the $n$ columns of $U$ and the $n \times n$ part of $\Sigma$ (since remaining columns of $U$ multiply zero rows of $\Sigma$). The decomposition becomes $A = U_{m \times n} \Sigma_{n \times n} V_{n \times n}^T$.

**Q5**: What is the condition number of a matrix in terms of SVD?

**A5**: The condition number $\kappa(A) = \sigma_1 / \sigma_r$, where $\sigma_1$ is the largest singular value and $\sigma_r$ is the smallest non-zero singular value. A large condition number indicates an ill-conditioned (nearly singular) matrix.

### Intermediate

**Q1**: Prove that any matrix $A$ can be written as $A = U\Sigma V^T$.

**A1**: Consider $A^T A$, which is symmetric positive semi-definite, so it has an eigenvalue decomposition $A^T A = V\Lambda V^T$ with $\lambda_i \ge 0$. Define $\sigma_i = \sqrt{\lambda_i}$. For $i$ where $\sigma_i > 0$, define $U_i = A V_i / \sigma_i$. Then $U_i^T U_j = (V_i^T A^T A V_j)/(\sigma_i \sigma_j) = \sigma_j^2 \delta_{ij}/(\sigma_i \sigma_j) = \delta_{ij}$, so the $U_i$ are orthonormal. Extend to an orthonormal basis for $\mathbb{R}^m$. Then $AV = U\Sigma$ and $A = U\Sigma V^T$.

**Q2**: State and prove the Eckart-Young theorem.

**A2**: The Eckart-Young theorem states that for any matrix $A$ with SVD $A = U\Sigma V^T$, the best rank-$k$ approximation to $A$ in both the spectral norm $\|\cdot\|_2$ and Frobenius norm $\|\cdot\|_F$ is $A_k = U_k \Sigma_k V_k^T$, where only the top $k$ singular values and corresponding vectors are kept. The errors are $\|A - A_k\|_2 = \sigma_{k+1}$ and $\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}$.

**Q3**: How is SVD used to compute the Moore-Penrose pseudo-inverse?

**A3**: Given $A = U\Sigma V^T$, the pseudo-inverse is $A^+ = V \Sigma^+ U^T$, where $\Sigma^+$ is the transpose of $\Sigma$ with each non-zero singular value $\sigma_i$ replaced by $1/\sigma_i$. For a full-rank $m \times n$ matrix with $m > n$, $A^+ = (A^T A)^{-1} A^T$, and SVD provides a numerically stable way to compute this.

**Q4**: How does SVD relate to Principal Component Analysis (PCA)?

**A4**: PCA on a centred data matrix $X \in \mathbb{R}^{n \times d}$ (n samples, d features) can be performed via SVD: $X = U\Sigma V^T$. The principal components (loadings) are the columns of $V$. The principal component scores are the columns of $U\Sigma$. The variance explained by the $i$-th component is $\sigma_i^2 / (n-1)$.

**Q5**: What is the computational complexity of computing the full SVD of an $m \times n$ matrix?

**A5**: For a dense matrix, the full SVD costs $O(m n \min(m,n))$ flops. The reduced SVD costs $O(m n k)$ where $k$ is the target rank. Modern randomized SVD algorithms can compute a good rank-$k$ approximation in $O(m n \log k)$ time.

### Advanced

**Q1**: Derive the relationship between SVD of $X$ and the eigendecompositions of $X^T X$ and $X X^T$. Show that if $X = U\Sigma V^T$, then $X^T X = V \Sigma^T \Sigma V^T$ and $X X^T = U \Sigma \Sigma^T U^T$.

**A1**: Given $X = U\Sigma V^T$, compute $X^T = V \Sigma^T U^T$. Then:

$X^T X = (V \Sigma^T U^T)(U \Sigma V^T) = V \Sigma^T (U^T U) \Sigma V^T = V \Sigma^T \Sigma V^T = V \Lambda V^T$

where $\Lambda = \Sigma^T \Sigma$ is diagonal with entries $\lambda_i = \sigma_i^2$. Similarly:

$X X^T = (U \Sigma V^T)(V \Sigma^T U^T) = U \Sigma (V^T V) \Sigma^T U^T = U \Sigma \Sigma^T U^T = U \Lambda_m U^T$

where $\Lambda_m$ has $\sigma_i^2$ for $i \le \min(m,n)$ and zeros elsewhere.

Thus $V$ diagonalizes $X^T X$ and $U$ diagonalizes $X X^T$, with eigenvalues $\sigma_i^2$.

**Q2**: Explain the concept of "singular value hardening" in random matrix theory and its implications for high-dimensional data analysis.

**A2**: For a random matrix $X \in \mathbb{R}^{n \times d}$ with i.i.d. entries (mean 0, variance $\sigma^2$), as $n, d \to \infty$ with $d/n \to \gamma$, the singular values follow the Marchenko-Pastur distribution. The largest singular value converges to $\sigma(1 + \sqrt{\gamma})^2$, not to infinity. This means that in high dimensions, even noise produces seemingly "large" singular values. This phenomenon ("singular value hardening") implies that in PCA for high-dimensional data, the spectrum of the sample covariance matrix is a biased estimator of the population spectrum. The "Bulk" of singular values from noise must be identified and separated from signal. The Tracy-Widom law describes fluctuations of the largest eigenvalue. This has profound implications: one cannot simply look at a scree plot and assume all large singular values represent signal — a null model from random matrix theory must be used as a baseline.

**Q3**: Describe the randomized SVD algorithm and analyze its computational advantages.

**A3**: The randomized SVD (Halko, Martinsson, Tropp, 2011) computes a rank-$k$ approximation in two stages:

Stage 1 (Randomized Range Finder):
1. Generate an $n \times (k+p)$ random matrix $\Omega$ (Gaussian or structured).
2. Form $Y = A\Omega \in \mathbb{R}^{m \times (k+p)}$, where $p$ is an oversampling parameter (typically 5-10).
3. Compute an orthonormal basis $Q$ for $\text{Col}(Y)$ via QR decomposition.

Stage 2 (Deterministic SVD):
4. Form $B = Q^T A \in \mathbb{R}^{(k+p) \times n}$.
5. Compute the full SVD of the small matrix $B: B = \tilde{U} \Sigma V^T$.
6. Set $U = Q\tilde{U}$.

The output $U_k \Sigma_k V_k^T$ is a near-optimal rank-$k$ approximation. Complexity: $O(mn \log k)$ vs $O(mnk)$ for classical methods, with high probability of success. This is crucial for big-data applications where $m, n \gg 10^5$.

## Practice Problems

### Easy

1. Compute the SVD of $A = \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix}$.

2. Find the singular values of $A = \begin{bmatrix} 2 & 0 \\ 0 & 0 \end{bmatrix}$.

3. If $A$ has singular values $\sigma_1 = 5$, $\sigma_2 = 3$, $\sigma_3 = 1$, what is $\|A\|_F$?

4. What is the rank of a matrix with singular values $\sigma = [10, 0.001, 0]$?

5. For $A = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}$, find the non-zero singular value.

### Medium

1. Compute the full SVD of $A = \begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}$.

2. Find the best rank-1 approximation of $A = \begin{bmatrix} 2 & 2 \\ 2 & 2 \end{bmatrix}$ using SVD.

3. Given $A = \begin{bmatrix} 3 & 0 \\ 0 & -2 \end{bmatrix}$, find its SVD. (Hint: singular values are always non-negative.)

4. Let $A = \begin{bmatrix} 1 & 0 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}$. Compute the reduced SVD.

5. Compute the pseudo-inverse $A^+$ of $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ using SVD.

### Hard

1. Compute the full SVD of $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \\ 0 & 0 \end{bmatrix}$.

2. Prove that for any matrix $A$, $\|A\|_2 \le \|A\|_F \le \sqrt{r} \|A\|_2$ where $r = \text{rank}(A)$, using singular values.

3. The matrix $A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{bmatrix}$ has rank 1. Find its SVD and verify that $A_1 = A$.

## Solutions

### Easy Solutions

**Solution 1**: $A = \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix}$ is already diagonal. Its SVD has $U = I$, $V = I$, $\Sigma = \text{diag}(2, 1)$ (sorted singular values). Note: singular values must be sorted descending! So $\sigma_1 = 2$, $\sigma_2 = 1$.

$$
A = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}^T
$$

Actually, since $U$ and $V$ must align with sorted $\sigma$:
$\sigma_1 = 2$ corresponds to direction $(0,1)$ in output and $(0,1)$ in input.
$U = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$, $\Sigma = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$, $V = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$.

**Solution 2**: $A^T A = \begin{bmatrix} 4 & 0 \\ 0 & 0 \end{bmatrix}$. Eigenvalues: 4, 0. Singular values: $\sigma_1 = \sqrt{4} = 2$, $\sigma_2 = 0$.

**Solution 3**: $\|A\|_F = \sqrt{\sum \sigma_i^2} = \sqrt{25 + 9 + 1} = \sqrt{35} \approx 5.92$.

**Solution 4**: Rank = 2 (two non-zero singular values). The third singular value is 0.

**Solution 5**: $A^T A = \begin{bmatrix} 0 & 0 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}$. Eigenvalues: 1, 0. Non-zero singular value: $\sigma = 1$.

### Medium Solutions

**Solution 1**: $A = \begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}$.

$A^T A = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$. Eigenvalues: 2, 0. $\sigma_1 = \sqrt{2}$.

For $\lambda = 2$: $\begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix}\mathbf{v}=0 \Rightarrow v_1 = v_2$. $V_1 = \frac{1}{\sqrt{2}}[1,1]^T$.

For $\lambda = 0$: $\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}\mathbf{v}=0 \Rightarrow v_1 = -v_2$. $V_2 = \frac{1}{\sqrt{2}}[1,-1]^T$.

$U_1 = \frac{A V_1}{\sigma_1} = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix} \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix} = \frac{1}{2}\begin{bmatrix} 2 \\ 0 \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$.

$U_2$ extends: $\begin{bmatrix} 0 \\ 1 \end{bmatrix}$.

$$
U = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \Sigma = \begin{bmatrix} \sqrt{2} & 0 \\ 0 & 0 \end{bmatrix}, V = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
$$

**Solution 2**: $A = \begin{bmatrix} 2 & 2 \\ 2 & 2 \end{bmatrix}$.

$A^T A = \begin{bmatrix} 8 & 8 \\ 8 & 8 \end{bmatrix}$. Eigenvalues: 16, 0. $\sigma_1 = 4$, $\sigma_2 = 0$.

$V_1 = \frac{1}{\sqrt{2}}[1,1]^T$, $V_2 = \frac{1}{\sqrt{2}}[1,-1]^T$.

$U_1 = \frac{A V_1}{4} = \frac{1}{4}\begin{bmatrix} 2 & 2 \\ 2 & 2 \end{bmatrix} \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix} = \frac{1}{4\sqrt{2}}\begin{bmatrix} 4 \\ 4 \end{bmatrix} = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix}$.

$U_2 = \frac{1}{\sqrt{2}}[1,-1]^T$.

Rank-1 approximation: $A_1 = \sigma_1 U_1 V_1^T = 4 \cdot \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix} \cdot \frac{1}{\sqrt{2}}[1,1] = 4 \cdot \frac{1}{2}\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 2 \\ 2 & 2 \end{bmatrix} = A$. Since $A$ is rank-1 already, $A_1 = A$ exactly.

**Solution 3**: $A = \begin{bmatrix} 3 & 0 \\ 0 & -2 \end{bmatrix}$.

$A^T A = \begin{bmatrix} 9 & 0 \\ 0 & 4 \end{bmatrix}$. Eigenvalues: 9, 4. $\sigma_1 = 3$, $\sigma_2 = 2$.

$V_1 = [1,0]^T$, $V_2 = [0,1]^T$.

$U_1 = \frac{A V_1}{3} = \frac{1}{3}\begin{bmatrix} 3 \\ 0 \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$.

$U_2 = \frac{A V_2}{2} = \frac{1}{2}\begin{bmatrix} 0 \\ -2 \end{bmatrix} = \begin{bmatrix} 0 \\ -1 \end{bmatrix}$.

Thus $U = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$, $\Sigma = \begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix}$, $V = I$.

Check: $U\Sigma V^T = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 3 & 0 \\ 0 & -2 \end{bmatrix} = A$. $\checkmark$

(Singular values are always non-negative, achieved by absorbing signs into $U$.)

**Solution 4**: $A = \begin{bmatrix} 1 & 0 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}$.

$A^T A = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$. Eigenvalues: 1, 0. $\sigma_1 = 1$.

$V_1 = [1,0]^T$, $V_2 = [0,1]^T$.

$U_1 = \frac{A V_1}{1} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$.

Reduced SVD: $U = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$, $\Sigma = \begin{bmatrix} 1 & 0 \end{bmatrix}$, $V = I$.

**Solution 5**: $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$.

$A^T A = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} = \begin{bmatrix} 10 & 14 \\ 14 & 20 \end{bmatrix}$.

Eigenvalues: $\det \begin{bmatrix} 10-\lambda & 14 \\ 14 & 20-\lambda \end{bmatrix} = (10-\lambda)(20-\lambda) - 196 = \lambda^2 - 30\lambda + 200 - 196 = \lambda^2 - 30\lambda + 4 = 0$.

$\lambda = \frac{30 \pm \sqrt{900 - 16}}{2} = \frac{30 \pm \sqrt{884}}{2} = \frac{30 \pm 2\sqrt{221}}{2} = 15 \pm \sqrt{221}$.

$\lambda_1 = 15 + \sqrt{221} \approx 29.866$, $\lambda_2 = 15 - \sqrt{221} \approx 0.134$.

$\sigma_1 \approx \sqrt{29.866} \approx 5.465$, $\sigma_2 \approx \sqrt{0.134} \approx 0.366$.

For $\lambda_1$: $(10-29.866)v_1 + 14v_2 = 0 \Rightarrow -19.866v_1 + 14v_2 = 0 \Rightarrow v_2 = 1.419v_1$. $V_1 = \frac{1}{\sqrt{1 + 1.419^2}}[1, 1.419]^T \approx [0.576, 0.817]^T$.

$U_1 = \frac{A V_1}{\sigma_1} \approx \frac{1}{5.465} \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 0.576 \\ 0.817 \end{bmatrix} = \frac{1}{5.465} \begin{bmatrix} 0.576 + 1.634 \\ 1.728 + 3.268 \end{bmatrix} = \frac{1}{5.465} \begin{bmatrix} 2.210 \\ 4.996 \end{bmatrix} \approx [0.404, 0.914]^T$.

Similarly $V_2 \approx [-0.817, 0.576]^T$, $U_2 \approx [-0.914, 0.404]^T$.

$A^+ = V \Sigma^{-1} U^T \approx \begin{bmatrix} -2.00 & 1.00 \\ 1.50 & -0.50 \end{bmatrix}$.

### Hard Solutions

**Solution 1**: $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \\ 0 & 0 \end{bmatrix}$.

$A^T A = \begin{bmatrix} 2 & 1 & 0 \\ 1 & 2 & 0 \end{bmatrix} \begin{bmatrix} 2 & 1 \\ 1 & 2 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 5 & 4 \\ 4 & 5 \end{bmatrix}$.

Eigenvalues: $(5-\lambda)^2 - 16 = \lambda^2 - 10\lambda + 25 - 16 = \lambda^2 - 10\lambda + 9 = 0$. $\lambda = \frac{10 \pm \sqrt{100-36}}{2} = \frac{10 \pm 8}{2}$.

$\lambda_1 = 9$, $\lambda_2 = 1$. $\sigma_1 = 3$, $\sigma_2 = 1$.

For $\lambda_1 = 9$: $\begin{bmatrix} -4 & 4 \\ 4 & -4 \end{bmatrix} \Rightarrow v_1 = v_2$. $V_1 = \frac{1}{\sqrt{2}}[1,1]^T$.

For $\lambda_2 = 1$: $\begin{bmatrix} 4 & 4 \\ 4 & 4 \end{bmatrix} \Rightarrow v_1 = -v_2$. $V_2 = \frac{1}{\sqrt{2}}[1,-1]^T$.

$U_1 = \frac{A V_1}{3} = \frac{1}{3} \begin{bmatrix} 2 & 1 \\ 1 & 2 \\ 0 & 0 \end{bmatrix} \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \frac{1}{3\sqrt{2}} \begin{bmatrix} 3 \\ 3 \\ 0 \end{bmatrix} = \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \\ 0 \end{bmatrix}$.

$U_2 = \frac{A V_2}{1} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \\ 0 & 0 \end{bmatrix} \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \end{bmatrix} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \\ 0 \end{bmatrix}$.

$U_3$ must be orthogonal to $U_1, U_2$: choose $U_3 = [0,0,1]^T$.

Full SVD:
$$
U = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0 \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 0 \\ 0 & 0 & 1 \end{bmatrix}, \;
\Sigma = \begin{bmatrix} 3 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}, \;
V = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{bmatrix}
$$

**Solution 2**: Let singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$, $\sigma_{r+1} = \dots = 0$.

$\|A\|_2 = \sigma_1$. $\|A\|_F = \sqrt{\sum_{i=1}^r \sigma_i^2}$.

Lower bound: $\|A\|_F^2 = \sum \sigma_i^2 \ge \sigma_1^2 = \|A\|_2^2 \Rightarrow \|A\|_F \ge \|A\|_2$.

Upper bound: $\sigma_i \le \sigma_1 = \|A\|_2$ for all $i$, so $\sum \sigma_i^2 \le r \sigma_1^2$, giving $\|A\|_F \le \sqrt{r} \|A\|_2$.

Thus $\|A\|_2 \le \|A\|_F \le \sqrt{r} \|A\|_2$. $\square$

**Solution 3**: $A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{bmatrix}$.

Since row 2 = 2 $\times$ row 1, rank = 1.

$A^T A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \\ 3 & 6 \end{bmatrix} \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{bmatrix} = \begin{bmatrix} 5 & 10 & 15 \\ 10 & 20 & 30 \\ 15 & 30 & 45 \end{bmatrix}$.

Eigenvalues: $\lambda_1 = 70$, $\lambda_2 = 0$, $\lambda_3 = 0$.

$\sigma_1 = \sqrt{70}$.

$V_1$: Since rows are proportional, $A^T A (1,2,3)^T = \begin{bmatrix} 5+20+45 \\ 10+40+90 \\ 15+60+135 \end{bmatrix} = \begin{bmatrix} 70 \\ 140 \\ 210 \end{bmatrix} = 70 \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$.

$V_1 = \frac{1}{\sqrt{14}}[1,2,3]^T$.

$U_1 = \frac{A V_1}{\sigma_1} = \frac{1}{\sqrt{70}} \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{bmatrix} \frac{1}{\sqrt{14}} \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} = \frac{1}{\sqrt{980}} \begin{bmatrix} 1 + 4 + 9 \\ 2 + 8 + 18 \end{bmatrix} = \frac{1}{\sqrt{980}} \begin{bmatrix} 14 \\ 28 \end{bmatrix} = \begin{bmatrix} \frac{14}{\sqrt{980}} \\ \frac{28}{\sqrt{980}} \end{bmatrix}$.

$\sqrt{980} = \sqrt{4 \cdot 245} = 2\sqrt{245} = 2\sqrt{49 \cdot 5} = 14\sqrt{5}$. So $U_1 = \begin{bmatrix} \frac{14}{14\sqrt{5}} \\ \frac{28}{14\sqrt{5}} \end{bmatrix} = \begin{bmatrix} \frac{1}{\sqrt{5}} \\ \frac{2}{\sqrt{5}} \end{bmatrix}$.

$U_2$: $\begin{bmatrix} \frac{2}{\sqrt{5}} \\ -\frac{1}{\sqrt{5}} \end{bmatrix}$ (orthogonal to $U_1$).

Full SVD:
$$
U = \begin{bmatrix} \frac{1}{\sqrt{5}} & \frac{2}{\sqrt{5}} \\ \frac{2}{\sqrt{5}} & -\frac{1}{\sqrt{5}} \end{bmatrix}, \;
\Sigma = \begin{bmatrix} \sqrt{70} & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}, \;
V = \begin{bmatrix} \frac{1}{\sqrt{14}} & * & * \\ \frac{2}{\sqrt{14}} & * & * \\ \frac{3}{\sqrt{14}} & * & * \end{bmatrix}
$$

$A_1 = U_1 \sigma_1 V_1^T = \begin{bmatrix} \frac{1}{\sqrt{5}} \\ \frac{2}{\sqrt{5}} \end{bmatrix} \sqrt{70} \frac{1}{\sqrt{14}} \begin{bmatrix} 1 & 2 & 3 \end{bmatrix} = \frac{\sqrt{70}}{\sqrt{5}\sqrt{14}} \begin{bmatrix} 1 \\ 2 \end{bmatrix} \begin{bmatrix} 1 & 2 & 3 \end{bmatrix}$.

$\frac{\sqrt{70}}{\sqrt{5}\sqrt{14}} = \sqrt{\frac{70}{70}} = 1$.

$A_1 = \begin{bmatrix} 1 \\ 2 \end{bmatrix} \begin{bmatrix} 1 & 2 & 3 \end{bmatrix} = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{bmatrix} = A$. $\checkmark$

## Related Concepts

- **Eigenvalue Decomposition**: SVD generalizes eigenvalue decomposition to non-square matrices. For symmetric $A$, SVD = eigendecomposition.
- **Principal Component Analysis (PCA)**: PCA = SVD on centred data. The right singular vectors are principal component loadings.
- **QR Decomposition**: An alternative matrix factorization. SVD is more informative but more expensive.
- **Cholesky Decomposition**: For positive definite matrices; SVD is more general.
- **Moore-Penrose Pseudo-inverse**: SVD provides the most numerically stable way to compute it.
- **Low-rank Approximation**: SVD gives the optimal low-rank approximation (Eckart-Young theorem).
- **Nuclear Norm**: Sum of singular values, used as a convex surrogate for rank in optimization.
- **Frobenius Norm**: Root-sum-of-squares of singular values.

## Next Concepts

- **Principal Component Analysis (MATH-043)**: Builds directly on SVD for dimensionality reduction.
- **Matrix Completion**: Recovering missing entries under low-rank assumption; uses SVD iteratively.
- **Non-negative Matrix Factorization (NMF)**: An alternative low-rank factorization with non-negativity constraints.
- **Randomized Linear Algebra**: Scalable SVD algorithms for big data.
- **Tensor Decompositions**: SVD generalized to higher-order tensors (Tucker decomposition, CP decomposition).
- **Kernel PCA**: Nonlinear generalization of PCA using the kernel trick.

## Summary

The Singular Value Decomposition $A = U\Sigma V^T$ is a fundamental matrix factorization that exists for every matrix. It decomposes any linear transformation into a rotation/scaling/rotation sequence, revealing the rank, norm, range, nullspace, and condition number of the matrix. The singular values are the square roots of eigenvalues of $A^T A$, and they quantify the "importance" of each dimension. The Eckart-Young theorem guarantees that the truncated SVD provides the optimal low-rank approximation, making it the tool of choice for dimensionality reduction, compression, denoising, and matrix completion. SVD underlies PCA, LSA, collaborative filtering, and many other machine learning methods, and its pseudo-inverse provides the most numerically stable solution to least-squares problems.

## Key Takeaways

1. **Every matrix has an SVD** — it is universal and numerically stable.
2. **$A = U\Sigma V^T$** — $U$ and $V$ are orthogonal, $\Sigma$ is diagonal with non-negative entries.
3. **Singular values = $\sqrt{\text{eigenvalues of } A^T A}$**, sorted descending.
4. **Left singular vectors $U$** = eigenvectors of $AA^T$ span the column space.
5. **Right singular vectors $V$** = eigenvectors of $A^T A$ span the row space.
6. **Rank = number of non-zero singular values**.
7. **Eckart-Young theorem**: The best rank-$k$ approximation is $A_k = U_k \Sigma_k V_k^T$ with error $\sigma_{k+1}$.
8. **Pseudo-inverse**: $A^+ = V \Sigma^+ U^T$ solves least-squares problems.
9. **PCA = SVD on centred data**: Principal components = columns of $V$, scores = $U\Sigma$.
10. **SVD is everywhere in ML**: dimensionality reduction, recommendations, compression, whitening, feature extraction.
