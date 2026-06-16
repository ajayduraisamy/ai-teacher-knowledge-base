# Concept: NumPy Linear Algebra

## Concept ID

PYT-069

## Difficulty

Intermediate

## Domain

Python

## Module

NumPy

## Learning Objectives

- Compute matrix inverse with `np.linalg.inv`
- Calculate determinants with `np.linalg.det`
- Compute eigenvalues and eigenvectors with `np.linalg.eig`
- Perform Singular Value Decomposition (SVD) with `np.linalg.svd`
- Solve linear systems with `np.linalg.solve` and `np.linalg.lstsq`
- Compute matrix powers, norms, cross products, and outer products
- Apply linear algebra to machine learning problems (least squares, PCA)

## Prerequisites

- Basic linear algebra concepts (matrix multiplication, eigenvalues)
- NumPy array operations (PYT-068)
- Understanding of matrix shapes and dimensions

## Definition

The `np.linalg` module provides a comprehensive suite of linear algebra operations built on optimized BLAS and LAPACK libraries. It includes matrix decomposition, solving systems of equations, eigenvalue computations, norms, and specialized matrix operations essential for scientific computing and machine learning.

## Intuition

Linear algebra is the language of data science. Matrices represent datasets (rows = samples, columns = features), transformations (weight matrices in neural networks), or systems of equations. `np.linalg` functions are the computational backbone of these operations — finding how matrices stretch space (eigendecomposition), compressing data (SVD), or solving for unknown parameters (least squares). These operations are numerically stable and highly optimized.

## Why This Concept Matters

Nearly every machine learning algorithm relies on linear algebra under the hood. Linear regression is a least squares problem. PCA is an eigendecomposition or SVD of the covariance matrix. Neural networks are compositions of linear transformations and nonlinearities. Understanding and using `np.linalg` is essential for implementing ML algorithms from scratch and for understanding how libraries like scikit-learn work.

## Real World Examples

1. **Linear Regression:** `np.linalg.lstsq(X, y)` solves for optimal weights in one line.
2. **PCA for Dimensionality Reduction:** SVD of the data matrix gives principal components.
3. **Image Compression:** Truncated SVD stores a low-rank approximation of an image.
4. **Network Analysis:** Eigenvalues of the adjacency matrix reveal community structure.
5. **Robotics:** Matrix inverses and cross products compute 3D rotations and transformations.

## AI/ML Relevance

Linear algebra is the mathematical foundation of ML. Gradient descent updates involve matrix-vector products. PCA uses SVD or eigendecomposition. Regularization adds penalty terms to the normal equations. Neural network layers are matrix multiplications. Understanding SVD is critical for understanding model compression, feature extraction, and data whitening.

## Code Examples

### Example 1: Matrix Inverse and Determinant

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])

# Determinant
det = np.linalg.det(A)
print("A:\n", A)
print("det(A):", det)

# Inverse
A_inv = np.linalg.inv(A)
print("\nA^{-1}:\n", A_inv)

# Verify: A @ A^{-1} ≈ I
I = A @ A_inv
print("\nA @ A_inv:\n", I)
print("Is identity:", np.allclose(I, np.eye(2)))

# Singular matrix (det ≈ 0)
singular = np.array([[1, 2],
                     [2, 4]])
try:
    np.linalg.inv(singular)
except np.linalg.LinAlgError as e:
    print(f"\nSingular matrix error: {e}")
```
```
# Output:
# A:
#  [[1 2]
#  [3 4]]
# det(A): -2.0000000000000004
# 
# A^{-1}:
#  [[-2.   1. ]
#  [ 1.5 -0.5]]
# 
# A @ A_inv:
#  [[1.0000000e+00 0.0000000e+00]
#  [8.8817842e-16 1.0000000e+00]]
# Is identity: True
# 
# Singular matrix error: Singular matrix
```

### Example 2: Solving Linear Systems

```python
import numpy as np

# Solve Ax = b
A = np.array([[3, 1],
              [1, 2]])
b = np.array([9, 8])

# Exact solution (square matrix, full rank)
x = np.linalg.solve(A, b)
print("A:\n", A)
print("b:", b)
print("x:", x)
print("Verify A @ x:", A @ x)

# Least squares (overdetermined system)
A_ls = np.array([[1, 1],
                 [1, 2],
                 [1, 3]])
b_ls = np.array([2, 3, 4])

x_ls, residuals, rank, s = np.linalg.lstsq(A_ls, b_ls, rcond=None)
print(f"\nLeast squares solution: {x_ls}")
print(f"Residuals: {residuals}")
print(f"Rank: {rank}")
print(f"Singular values: {s}")
print(f"Predicted: {A_ls @ x_ls}")
print(f"Actual:    {b_ls}")
```
```
# Output:
# A:
#  [[3 1]
#  [1 2]]
# b: [9 8]
# x: [2. 3.]
# Verify A @ x: [9. 8.]
# 
# Least squares solution: [1.         1.        ]
# Residuals: [0.66666667]
# Rank: 2
# Singular values: [4.07914333 0.60049122]
# Predicted: [2. 3. 4.]
# Actual:    [2 3 4]
```

### Example 3: Eigenvalues and Eigenvectors

```python
import numpy as np

A = np.array([[4, -2],
              [1,  1]])

eigvals, eigvecs = np.linalg.eig(A)
print("A:\n", A)
print("Eigenvalues:", eigvals)
print("Eigenvectors (columns):\n", eigvecs)

# Verify: A @ v = λ * v
for i in range(len(eigvals)):
    v = eigvecs[:, i]
    lam = eigvals[i]
    lhs = A @ v
    rhs = lam * v
    print(f"\nλ{i} = {lam:.4f}")
    print(f"  A @ v = {lhs}")
    print(f"  λ * v = {rhs}")
    print(f"  Match: {np.allclose(lhs, rhs)}")

# Real symmetric matrix (guaranteed real eigenvalues)
sym = np.array([[3, 2],
                [2, 5]])
eigvals_sym, eigvecs_sym = np.linalg.eig(sym)
print(f"\nSymmetric matrix eigenvalues: {eigvals_sym}")
```
```
# Output:
# A:
#  [[ 4 -2]
#  [ 1  1]]
# Eigenvalues: [3. 2.]
# Eigenvectors (columns):
#  [[0.89442719 0.70710678]
#  [0.4472136  0.70710678]]
# 
# λ0 = 3.0000
#   A @ v = [2.68328157 1.34164079]
#   λ * v = [2.68328157 1.34164079]
#   Match: True
# 
# λ1 = 2.0000
#   A @ v = [1.41421356 1.41421356]
#   λ * v = [1.41421356 1.41421356]
#   Match: True
# 
# Symmetric matrix eigenvalues: [2.17157288 5.82842712]
```

### Example 4: Singular Value Decomposition (SVD)

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

U, s, Vt = np.linalg.svd(A)
print("A:\n", A)
print("U shape:", U.shape)
print("s (singular values):", s)
print("Vt shape:", Vt.shape)

# Reconstruct A (full SVD)
S_full = np.zeros_like(A, dtype=float)
np.fill_diagonal(S_full, s)
A_reconstructed = U @ S_full @ Vt
print(f"\nReconstruction error: {np.linalg.norm(A - A_reconstructed):.2e}")

# Low-rank approximation (rank-2)
k = 2
S_k = np.zeros((A.shape[0], k))
np.fill_diagonal(S_k, s[:k])
A_k = U[:, :k] @ S_k @ Vt[:k, :]
print(f"Rank-2 approximation error: {np.linalg.norm(A - A_k):.4f}")

# Economy SVD (more common usage)
U_econ, s_econ, Vt_econ = np.linalg.svd(A, full_matrices=False)
print(f"\nEconomy SVD shapes: U={U_econ.shape}, s={s_econ.shape}, Vt={Vt_econ.shape}")
```
```
# Output:
# A:
#  [[1 2 3]
#  [4 5 6]
#  [7 8 9]]
# U shape: (3, 3)
# s (singular values): [1.68481034e+01 1.06836951e+00 1.47280825e-16]
# Vt shape: (3, 3)
# 
# Reconstruction error: 1.58e-14
# 
# Rank-2 approximation error: 1.3807e-15
# 
# Economy SVD shapes: U=(3, 3), s=(3,), Vt=(3, 3)
```

### Example 5: Norms and Matrix Powers

```python
import numpy as np

v = np.array([3, -4])
print("Vector v:", v)
print("L2 norm:", np.linalg.norm(v))
print("L1 norm:", np.linalg.norm(v, ord=1))
print("L∞ norm:", np.linalg.norm(v, ord=np.inf))

# Matrix norms
M = np.array([[1, 2],
              [3, 4]])
print("\nFrobenius norm:", np.linalg.norm(M, ord='fro'))
print("Spectral norm (largest singular value):", np.linalg.norm(M, ord=2))

# Matrix powers
A = np.array([[1, 2],
              [3, 4]])
print("\nA:\n", A)
print("A^2:\n", np.linalg.matrix_power(A, 2))
print("A^3:\n", np.linalg.matrix_power(A, 3))
print("A^0 (identity):\n", np.linalg.matrix_power(A, 0))
print("A^{-1} (inverse):\n", np.linalg.matrix_power(A, -1))
```
```
# Output:
# Vector v: [ 3 -4]
# L2 norm: 5.0
# L1 norm: 7.0
# L∞ norm: 4.0
# 
# Frobenius norm: 5.477225575051661
# Spectral norm (largest singular value): 5.464985704219043
# 
# A:
#  [[1 2]
#  [3 4]]
# A^2:
#  [[ 7 10]
#  [15 22]]
# A^3:
#  [[ 37  54]
#  [ 81 118]]
# A^0 (identity):
#  [[1 0]
#  [0 1]]
# A^{-1} (inverse):
#  [[-2.   1. ]
#  [ 1.5 -0.5]]
```

### Example 6: Cross Product, Outer Product, and Trace

```python
import numpy as np

# Cross product (3D vectors only)
a = np.array([1, 0, 0])
b = np.array([0, 1, 0])
cross = np.cross(a, b)
print(f"{a} × {b} = {cross}")

# Cross product is orthogonal to both inputs
print("Dot with a:", np.dot(cross, a))
print("Dot with b:", np.dot(cross, b))

# Outer product
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
outer = np.outer(x, y)
print("\nOuter product:\n", outer)

# Trace (sum of diagonal)
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print("\nTrace:", np.trace(M))
```
```
# Output:
# [1 0 0] × [0 1 0] = [0 0 1]
# Dot with a: 0
# Dot with b: 0
# 
# Outer product:
#  [[ 4  5  6]
#  [ 8 10 12]
#  [12 15 18]]
# 
# Trace: 15
```

### Example 7: PCA via SVD

```python
import numpy as np

# Generate synthetic 2D data with correlation
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 2)
# Transform to create correlation
transform = np.array([[0.8, 0.6],
                     [-0.3, 0.9]])
X = X @ transform
X -= X.mean(axis=0)  # Center the data

# PCA via SVD
U, s, Vt = np.linalg.svd(X, full_matrices=False)

# Principal components (rows of Vt are the PCs)
print("Principal components:\n", Vt)

# Explained variance ratio
explained_var = s**2 / (s**2).sum()
print("Explained variance ratio:", explained_var)

# Project data onto first principal component
X_pca = X @ Vt.T
print("Projected data (first 5 rows):\n", X_pca[:5])

# Reconstruct with first PC only
X_reconstructed = X_pca[:, :1] @ Vt[:1, :]
print(f"Reconstruction error: {np.linalg.norm(X - X_reconstructed):.4f}")
```
```
# Output:
# Principal components:
#  [[ 0.79318863  0.6089811 ]
#  [ 0.6089811  -0.79318863]]
# Explained variance ratio: [0.88828069 0.11171931]
# Projected data (first 5 rows):
#  [[ 1.71912096 -0.4653087 ]
#  [ 1.03018658 -0.09834091]
#  [-0.54401828  0.1993353 ]
#  [ 0.05893707 -0.4650432 ]
#  [-1.12761533 -0.03358794]]
# Reconstruction error: 1.5865
```

## Common Mistakes

1. **Using `np.linalg.inv` Instead of `np.linalg.solve`:** To solve `Ax = b`, use `np.linalg.solve(A, b)` instead of `x = np.linalg.inv(A) @ b`. `solve` is faster and numerically more stable.

2. **Forgetting That `eig` Returns Complex Numbers for Non-Symmetric Matrices:** Eigenvalues of non-symmetric matrices can be complex. Use `eigvalsh` for real symmetric or Hermitian matrices.

3. **Not Checking `rcond` in `lstsq`:** Always pass `rcond=None` (or a small value) to `lstsq` to handle rank-deficient systems gracefully. Otherwise a FutureWarning is raised.

4. **Confusing SVD Output Shapes:** `np.linalg.svd(A, full_matrices=True)` returns `U (m,m)`, `s (k,)`, `Vt (n,n)` where `k = min(m,n)`. Use `full_matrices=False` for the economy version.

5. **Assuming `det` is Exact:** Determinants of large matrices computed via LU decomposition have floating-point errors. Check `abs(det) < 1e-10` for zero rather than exact equality.

6. **Using `matrix_power` with Non-Square Matrices:** `matrix_power` works only for square matrices. For non-square, use repeated `@` operations.

7. **Forgetting to Center Data for PCA:** PCA must be computed on centered data (subtract the mean). Failing to center means the first PC will capture the mean rather than variance.

## Interview Questions

### Beginner

1. **Q:** How do you compute the inverse of a matrix in NumPy?
   **A:** Use `np.linalg.inv(A)`. The matrix must be square and non-singular.

2. **Q:** What is the difference between `np.linalg.solve` and `np.linalg.inv`?
   **A:** Both solve `Ax = b`. `solve` is faster and numerically more stable, using LU decomposition. `inv` explicitly computes the inverse, which is slower and less stable.

3. **Q:** How do you compute the determinant of a matrix?
   **A:** Use `np.linalg.det(A)`.

4. **Q:** What does SVD return?
   **A:** `np.linalg.svd(A)` returns three arrays: `U` (left singular vectors), `s` (singular values as 1D array), and `Vt` (right singular vectors transposed). `A = U @ diag(s) @ Vt`.

5. **Q:** How do you compute the L2 norm of a vector?
   **A:** `np.linalg.norm(v)` or `np.sqrt(np.sum(v**2))`.

### Intermediate

1. **Q:** What is the difference between `np.linalg.eig` and `np.linalg.eigvals`?
   **A:** `eig` returns both eigenvalues and eigenvectors. `eigvals` returns only eigenvalues and is faster when you don't need eigenvectors.

2. **Q:** How do you solve a least squares problem `min ||Ax - b||²`?
   **A:** Use `np.linalg.lstsq(A, b, rcond=None)`. It returns the solution, residuals, rank, and singular values.

3. **Q:** What is the difference between full SVD and economy SVD?
   **A:** Full SVD on an `(m, n)` matrix returns `U (m,m)`, `s (min(m,n),)`, `Vt (n,n)`. Economy SVD (`full_matrices=False`) returns `U (m, k)`, `s (k,)`, `Vt (k, n)` where `k = min(m, n)`, which is usually sufficient and more memory-efficient.

4. **Q:** How do you compute `A³` without using `matrix_power`?
   **A:** `A @ A @ A` or `np.linalg.matrix_power(A, 3)`. The latter is more efficient for large powers as it uses exponentiation by squaring.

5. **Q:** What is the relationship between `np.cross(a, b)` and the determinant?
   **A:** For 2D vectors `a = [a1, a2]` and `b = [b1, b2]`, `np.cross(a, b) = a1*b2 - a2*b1`, which is the signed area of the parallelogram spanned by `a` and `b` (also the determinant of the matrix `[a, b]`).

### Advanced

1. **Q:** Explain how SVD is used for dimensionality reduction (PCA) and what the explained variance ratio represents.
   **A:** In PCA, the data matrix `X` (centered) is decomposed via SVD: `X = U S V^T`. The principal components are the rows of `V^T`. Projecting onto the first `k` components: `X_k = U_k S_k V_k^T`. The explained variance ratio of the `i`-th component is `s_i² / sum(s_j²)`, representing the fraction of total variance captured.

2. **Q:** What is the connection between `np.linalg.eig` on the covariance matrix and `np.linalg.svd` on the data matrix for PCA?
   **A:** For centered data `X`, the covariance matrix is `X^T X / (n-1)`. The right singular vectors `V` from SVD of `X` equal the eigenvectors of `X^T X`. The singular values `s` relate to eigenvalues `λ` by `λ_i = s_i² / (n-1)`. SVD is numerically more stable than forming the covariance matrix explicitly.

3. **Q:** Compare `np.linalg.solve` for square systems and `np.linalg.lstsq` for over/under-determined systems. When would you choose one over the other?
   **A:** For square full-rank systems, `solve` is faster and uses direct LU decomposition. For overdetermined (more equations than unknowns) or underdetermined (more unknowns than equations) systems, `lstsq` finds the minimum-norm least-squares solution. `lstsq` works for square systems too but `solve` is preferred for performance. Use `lstsq` when you also need residuals and rank information.

## Practice Problems

### Easy

1. Compute the determinant and inverse of `np.array([[2, 1], [1, 3]])`.

2. Solve `3x + y = 7`, `x + 2y = 4` using `np.linalg.solve`.

3. Compute the eigenvalues of `np.array([[1, 2], [2, 1]])`.

4. Find the L2 norm of `np.array([3, 4, 12])`.

5. Compute the cross product of `[1, 2, 3]` and `[4, 5, 6]`.

### Medium

1. Given `X` (100x5) and `y` (100,), compute the least squares weights for `Xw = y`. Then compute the R-squared score.

2. Perform PCA on the Iris dataset (4 features) using SVD and determine how many components explain 95% of the variance.

3. Compute the Moore-Penrose pseudoinverse of a 3x2 matrix using SVD, and verify `A^+ A ≈ I`.

4. Write a function that computes the condition number of a matrix using singular values.

5. Compute the cosine similarity between all pairs of 5 vectors in 3D space using `np.outer` and norms.

### Hard

1. Implement a function that computes the principal component analysis (PCA) using `np.linalg.eig` on the covariance matrix and using `np.linalg.svd` on the centered data. Compare the results.

2. Implement low-rank matrix approximation using truncated SVD. Apply it to a 50x50 random matrix with rank 10 (by construction) and verify the rank-`k` approximation error decreases with `k`.

3. Implement a function to solve a linear system `Ax = b` using LU decomposition without calling `solve` (use `scipy.linalg.lu` or implement your own). Compare with `np.linalg.solve`.

## Solutions

### Easy Solutions

```python
# 1
A = np.array([[2, 1], [1, 3]])
print("det:", np.linalg.det(A))
print("inv:\n", np.linalg.inv(A))

# 2
A = np.array([[3, 1], [1, 2]])
b = np.array([7, 4])
x = np.linalg.solve(A, b)
print("x:", x)  # [2, 1]

# 3
A = np.array([[1, 2], [2, 1]])
eigvals = np.linalg.eigvals(A)
print("eigenvalues:", eigvals)

# 4
v = np.array([3, 4, 12])
print("norm:", np.linalg.norm(v))

# 5
a, b = np.array([1, 2, 3]), np.array([4, 5, 6])
print("cross:", np.cross(a, b))
```

### Medium Solutions

```python
# 1 Least squares
np.random.seed(42)
X = np.random.randn(100, 5)
true_w = np.array([1.5, -2.0, 0.5, 3.0, -1.0])
y = X @ true_w + np.random.randn(100) * 0.1

w_hat, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
y_pred = X @ w_hat
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - y.mean())**2)
r2 = 1 - ss_res / ss_tot
print(f"R-squared: {r2:.4f}")
print(f"True w: {true_w}")
print(f"Estimated w: {w_hat.round(4)}")

# 2 PCA on Iris
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
explained_var = s**2 / (s**2).sum()
cumsum = np.cumsum(explained_var)
n_components = np.argmax(cumsum >= 0.95) + 1
print(f"Components for 95% variance: {n_components}")
print(f"Cumulative variance: {cumsum}")

# 3 Pseudoinverse via SVD
A = np.array([[1, 2], [3, 4], [5, 6]])
U, s, Vt = np.linalg.svd(A, full_matrices=False)
S_inv = np.diag(1/s)
A_pinv = Vt.T @ S_inv @ U.T
print("A^+ @ A:\n", A_pinv @ A)

# 4 Condition number
def condition_number(A):
    s = np.linalg.svd(A, compute_uv=False)
    return s.max() / s.min()

M = np.array([[1, 2], [3, 4]])
print(f"Condition number: {condition_number(M):.4f}")

# 5 Cosine similarity
vectors = np.random.randn(5, 3)
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
normalized = vectors / norms
cosine_sim = normalized @ normalized.T
print("Cosine similarity matrix:\n", cosine_sim.round(3))
```

### Hard Solutions

```python
# 1 PCA comparison
X = np.random.randn(100, 5)
X_centered = X - X.mean(axis=0)

# Method 1: eig on covariance
cov = X_centered.T @ X_centered / (X_centered.shape[0] - 1)
eigvals_cov, eigvecs_cov = np.linalg.eigh(cov)
idx = np.argsort(eigvals_cov)[::-1]
eigvecs_cov = eigvecs_cov[:, idx]

# Method 2: SVD on centered data
U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
print("PCs match:", np.allclose(np.abs(eigvecs_cov), np.abs(Vt.T)))
print("Eigenvalues from SVD:", s**2 / (X.shape[0] - 1))
print("Eigenvalues from cov:", eigvals_cov[idx])

# 2 Low-rank approximation
np.random.seed(42)
A = np.random.randn(50, 10) @ np.random.randn(10, 50)  # rank 10
for k in [1, 5, 10, 15]:
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    error = np.linalg.norm(A - A_k)
    print(f"Rank-{k} approximation error: {error:.4f}")

# 3 Solve via LU
def solve_lu(A, b):
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy().astype(float)
    for j in range(n):
        for i in range(j+1, n):
            factor = U[i, j] / U[j, j]
            L[i, j] = factor
            U[i, j:] -= factor * U[j, j:]
    # Forward substitution Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    # Back substitution Ux = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    return x

A_test = np.array([[3, 1], [1, 2]], dtype=float)
b_test = np.array([9, 8], dtype=float)
x_lu = solve_lu(A_test, b_test)
x_solve = np.linalg.solve(A_test, b_test)
print("LU solve:", x_lu)
print("np.linalg.solve:", x_solve)
print("Match:", np.allclose(x_lu, x_solve))
```

## Related Concepts

- NumPy array operations (PYT-068)
- Matrix multiplication in array operations
- Statistics and covariance (PYT-071)
- scipy.linalg for advanced linear algebra

## Next Concepts

- NumPy random module (PYT-070)
- NumPy statistics (PYT-071)
- Reshaping and transposition (PYT-072)

## Summary

`np.linalg` provides essential linear algebra operations: `inv`, `det`, `eig`, `svd`, `solve`, and `lstsq`. These functions are built on optimized BLAS/LAPACK libraries. SVD is central to PCA, matrix compression, and pseudoinverse computation. `solve` is preferred over `inv` for solving linear systems. Cross products, outer products, norms, and matrix powers fill out the toolkit.

## Key Takeaways

- `np.linalg.solve(A, b)` over `np.linalg.inv(A) @ b` for speed and stability
- SVD (`np.linalg.svd`) is the most numerically stable matrix decomposition
- PCA = SVD on centered data or eigendecomposition of covariance matrix
- `lstsq` solves over/under-determined systems with minimum norm
- `rcond=None` is required in modern NumPy for `lstsq`
- Economy SVD (`full_matrices=False`) saves memory for tall/skinny matrices
- Eigenvalues of symmetric matrices are always real; non-symmetric may be complex
- Condition number (ratio of extreme singular values) measures matrix stability
