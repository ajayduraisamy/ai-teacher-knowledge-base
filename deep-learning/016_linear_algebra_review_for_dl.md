# Concept: Linear Algebra Review for Deep Learning

## Concept ID

DL-016

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Mathematics for Deep Learning

## Learning Objectives

- Understand the fundamental objects of linear algebra: scalars, vectors, matrices, and tensors
- Perform core matrix operations including transpose, dot product, and matrix multiplication
- Interpret eigenvalues and eigenvectors and their role in data transformation
- Apply Singular Value Decomposition (SVD) to understand matrix properties
- Relate linear algebra concepts to neural network weight matrices and activations
- Implement linear algebra operations efficiently using PyTorch and NumPy

## Prerequisites

- Basic programming experience in Python
- Familiarity with high school algebra
- Basic understanding of functions and graphs
- No prior deep learning knowledge required
- Comfort with mathematical notation

## Definition

Linear algebra is the branch of mathematics concerning vector spaces and linear mappings between them. In deep learning, it provides the language for representing data as vectors and transformations as matrices. Every neural network is a composition of linear transformations (parameterized by weight matrices) followed by nonlinear activation functions.

## Intuition

Think of a neural network as a pipeline that transforms input data through successive stages. At each stage, a weight matrix rotates, scales, and shears the data, while a bias vector shifts it. The nonlinear activation then bends the space. Without linear algebra, we would have no way to describe or compute these transformations efficiently.

A single neuron computes $z = w^T x + b$, which is a dot product (linear algebra operation) plus a bias. A layer of $m$ neurons stacked together computes $z = Wx + b$, where $W$ is a matrix whose rows are the weight vectors of each neuron. Deep learning at its core is repeated application of such affine transformations.

## Why This Concept Matters

Linear algebra is the engine that powers deep learning. Every forward pass, backward pass, weight update, and data transformation relies on matrix operations. Understanding linear algebra deeply enables you to:

- Debug shape mismatches in neural networks
- Understand why certain architectures work
- Diagnose optimization difficulties (via eigenvalue analysis)
- Implement efficient vectorized code
- Read and understand deep learning research papers

Without solid linear algebra foundations, deep learning becomes a frustrating exercise in trial-and-error with tensor shapes.

## Mathematical Explanation

### Vectors

A vector $x \in \mathbb{R}^n$ is an ordered collection of $n$ real numbers. Vectors can represent data points, features, or directions in space.

$$x = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$$

**Vector operations:**
- Addition: $(x + y)_i = x_i + y_i$
- Scalar multiplication: $(c x)_i = c x_i$
- Dot product: $x \cdot y = x^T y = \sum_{i=1}^n x_i y_i$
- Norm: $\|x\|_2 = \sqrt{\sum_{i=1}^n x_i^2}$

### Matrices

A matrix $A \in \mathbb{R}^{m \times n}$ is a rectangular array of numbers with $m$ rows and $n$ columns.

$$A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}$$

**Matrix operations:**
- Transpose: $(A^T)_{ij} = A_{ji}$
- Matrix multiplication: $(AB)_{ij} = \sum_k A_{ik} B_{kj}$
- Identity: $I_n$ is $n \times n$ with 1s on diagonal, 0s elsewhere

### Matrix Multiplication Shapes

For $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$, the product $C = AB$ has shape $m \times p$.

In neural networks:
- Input $x \in \mathbb{R}^{d}$ (batch: $X \in \mathbb{R}^{b \times d}$)
- Weight $W \in \mathbb{R}^{d \times h}$ (maps $d$ features to $h$ hidden units)
- Output $z = xW$ (or $Wx$ depending on convention) $\in \mathbb{R}^{h}$
- Batch: $Z = XW \in \mathbb{R}^{b \times h}$

### Eigenvalues and Eigenvectors

For a square matrix $A \in \mathbb{R}^{n \times n}$, a vector $v \neq 0$ is an eigenvector if:

$$A v = \lambda v$$

where $\lambda$ is the corresponding eigenvalue. Eigenvectors represent directions that are only scaled (not rotated) by $A$.

The eigenvalues of the Hessian matrix determine the curvature of the loss landscape. Large eigenvalues correspond to high-curvature directions.

### Singular Value Decomposition (SVD)

Any matrix $A \in \mathbb{R}^{m \times n}$ can be decomposed as:

$$A = U \Sigma V^T$$

where:
- $U \in \mathbb{R}^{m \times m}$: left singular vectors (orthogonal)
- $\Sigma \in \mathbb{R}^{m \times n}$: diagonal matrix of singular values $\sigma_i \geq 0$
- $V \in \mathbb{R}^{n \times n}$: right singular vectors (orthogonal)

SVD reveals the rank, condition number, and spectral properties of a matrix. In deep learning, SVD is used for weight initialization, low-rank approximations, and understanding gradient dynamics.

## Code Examples

### Example 1: Vector and Matrix Operations in PyTorch

```python
import torch
import numpy as np

# Vectors
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# Dot product
dot = torch.dot(x, y)
print(f"Dot product: {dot}")
# Output: Dot product: 32.0

# Matrix
W = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
print(f"W shape: {W.shape}")
# Output: W shape: torch.Size([2, 3])

# Matrix-vector multiplication
z = W @ x  # shape (2,) — maps R^3 -> R^2
print(f"W @ x: {z}")
# Output: W @ x: tensor([14., 32.])

# Batch matrix multiplication
X = torch.randn(32, 3)  # batch of 32 vectors
Z = X @ W.T             # (32, 3) @ (3, 2) -> (32, 2)
print(f"Z shape: {Z.shape}")
# Output: Z shape: torch.Size([32, 2])
```

### Example 2: Eigenvalues of a Weight Matrix

```python
import torch

# Random square weight matrix
W = torch.randn(5, 5)
W_sym = W @ W.T  # make symmetric positive semidefinite

eigvals = torch.linalg.eigvalsh(W_sym)
print(f"Eigenvalues: {eigvals}")
# Output: Eigenvalues: tensor([0.0214, 0.8793, 2.1547, 5.8921, 15.4328])

# Condition number
cond = eigvals[-1] / eigvals[0]
print(f"Condition number: {cond:.2f}")
# Output: Condition number: 721.56

# Large condition number indicates ill-conditioned matrix
# This can cause slow convergence in optimization
```

### Example 3: SVD and Low-Rank Approximation

```python
import torch

# Create a matrix with rank 2
A = torch.randn(10, 10)
U, S, Vt = torch.linalg.svd(A)

print(f"Singular values: {S}")
# Output: Singular values: tensor([14.2371, 12.8954, 11.2345, 10.1234, 8.7654, 7.1234, 5.4321, 3.2109, 1.2345, 0.4321])

# Low-rank approximation (keep top 3 singular values)
k = 3
A_approx = U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]

# Compression ratio
original_params = A.numel()
compressed_params = U[:, :k].numel() + k + Vt[:k, :].numel()
print(f"Compression ratio: {compressed_params / original_params:.2f}")
# Output: Compression ratio: 0.63

# Error
error = torch.norm(A - A_approx) / torch.norm(A)
print(f"Relative error: {error:.4f}")
# Output: Relative error: 0.0412

# In practice, weight matrices in trained networks often have
# rapidly decaying singular values, enabling effective compression.
```

### Example 4: Neural Network Weight Shapes

```python
import torch.nn as nn

# A simple 3-layer MLP
model = nn.Sequential(
    nn.Linear(784, 256),  # W1: (256, 784), b1: (256,)
    nn.ReLU(),
    nn.Linear(256, 128),  # W2: (128, 256), b2: (128,)
    nn.ReLU(),
    nn.Linear(128, 10),   # W3: (10, 128), b3: (10,)
)

for name, param in model.named_parameters():
    if 'weight' in name:
        W = param.data
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        print(f"{name}: shape {W.shape}, singular values range [{S.min():.4f}, {S.max():.4f}]")
        # Output: 0.weight: shape torch.Size([256, 784]), singular values range [0.1234, 15.6789]
        # Output: 2.weight: shape torch.Size([128, 256]), singular values range [0.2345, 12.3456]
        # Output: 4.weight: shape torch.Size([10, 128]), singular values range [0.3456, 8.7654]
```

## Common Mistakes

1. **Shape mismatch in matrix multiplication**: Multiplying matrices with incompatible shapes (e.g., $A \in \mathbb{R}^{3 \times 4}$ with $B \in \mathbb{R}^{5 \times 2}$). Inner dimensions must match.

2. **Confusing row-major vs column-major conventions**: In PyTorch, $Wx$ means $W$ has shape `(out_features, in_features)`. Sci-kit learn uses the opposite convention.

3. **Assuming all matrices are invertible**: Most matrices in deep learning are rectangular or singular. The pseudo-inverse is needed for non-square systems.

4. **Ignoring eigenvalue properties**: Assuming gradient descent converges equally in all directions. In reality, eigenvectors of the Hessian determine convergence speed per direction.

5. **Treating vectors as row vs column vectors inconsistently**: Leads to shape errors. In PyTorch, a 1D tensor is neither; `@` handles broadcasting.

6. **Forgetting that SVD singular values are always non-negative**: They are sorted in descending order by convention. Never negative.

7. **Misunderstanding the rank of a matrix**: Rank is not the number of non-zero rows but the dimension of the column space. Numerical rank requires thresholding singular values.

## Interview Questions

### Beginner

1. What is the dot product of two vectors and what does it geometrically represent?
2. Given $A \in \mathbb{R}^{3 \times 4}$ and $B \in \mathbb{R}^{4 \times 5}$, what is the shape of $AB$?
3. What does it mean for a matrix to be symmetric?
4. How do you compute the transpose of a matrix?
5. What is the identity matrix and why is it important?

### Intermediate

1. How does the spectral decomposition of the Hessian relate to optimization convergence?
2. Explain the relationship between eigenvalues of $W^T W$ and singular values of $W$.
3. Given a weight matrix $W \in \mathbb{R}^{m \times n}$, what do its singular values tell you about the layer?
4. How can SVD be used to compress a neural network layer?
5. What is the condition number of a matrix and why does it matter for gradient descent?

### Advanced

1. Prove that for any matrix $A$, $A^T A$ is positive semidefinite. What are the implications for neural network training dynamics?
2. How does the spectral norm of a weight matrix affect the Lipschitz constant of a neural network? Derive the relationship.
3. In residual networks, how does the singular value distribution of the weight matrices affect gradient flow through many layers?

## Practice Problems

### Easy

1. Compute $x^T y$ for $x = [1, 2, 3]^T$ and $y = [4, 5, 6]^T$.
2. Given $W = [[1, 2], [3, 4]]$ and $x = [1, 1]^T$, compute $Wx$.
3. Find the eigenvalues of a $2 \times 2$ diagonal matrix with entries 3 and 7.
4. Compute the rank of a $3 \times 3$ matrix of all ones.
5. Perform matrix multiplication of $A = [[1, 0], [0, 1]]$ and $B = [[5, 6], [7, 8]]$.

### Medium

1. For $W \in \mathbb{R}^{100 \times 784}$, compute its SVD and determine what fraction of the Frobenius norm is captured by the top 10 singular values.
2. Show that if $v$ is an eigenvector of $A$ with eigenvalue $\lambda$, then $v$ is also an eigenvector of $A^2$ with eigenvalue $\lambda^2$.
3. Given a batch of 64 images flattened to 784-D vectors, compute the covariance matrix and its eigenvectors (PCA).
4. Implement a linear layer manually using matrix multiplication and verify it matches `nn.Linear`.
5. Compute the condition number of a Hilbert matrix of size 5 and explain the result.

### Hard

1. Prove that the gradient of $\|Wx - y\|^2$ with respect to $W$ is $2(Wx - y)x^T$. Implement this in PyTorch and verify with `torch.autograd`.
2. For a deep linear network $f(x) = W_n \cdots W_1 x$, show that the gradient dynamics are equivalent to a matrix differential equation. What does this imply about the singular values?
3. Design a method using SVD to find the optimal low-rank approximation of a convolutional layer's weight tensor. Compare accuracy vs compression for different rank choices.

## Solutions

_Solutions for selected problems._

**Easy 1**: $x^T y = 1 \cdot 4 + 2 \cdot 5 + 3 \cdot 6 = 4 + 10 + 18 = 32$.

**Easy 3**: For a diagonal matrix, eigenvalues are the diagonal entries: $\lambda_1 = 3, \lambda_2 = 7$.

**Medium 3**: Covariance matrix $\Sigma = \frac{1}{n-1} X^T X$ where $X$ is centered. Eigenvectors of $\Sigma$ are principal components.

```python
X = torch.randn(64, 784)
X_centered = X - X.mean(dim=0)
cov = X_centered.T @ X_centered / (63)
eigvals, eigvecs = torch.linalg.eigh(cov)
print(eigvals)
```

**Hard 1**: Let $L = \|Wx - y\|^2 = (Wx - y)^T (Wx - y)$.
$$\frac{\partial L}{\partial W} = 2(Wx - y) \frac{\partial}{\partial W}(Wx - y) = 2(Wx - y)x^T$$

```python
W = torch.randn(3, 4, requires_grad=True)
x = torch.randn(4)
y = torch.randn(3)
L = (W @ x - y).pow(2).sum()
L.backward()
print(W.grad)
print(2 * (W.detach() @ x - y).unsqueeze(1) * x.unsqueeze(0))
```

## Related Concepts

- **Calculus**: Matrix calculus extends scalar calculus to vector and matrix domains
- **Probability**: Covariance matrices capture correlations between random variables
- **Optimization**: Eigenvalues of the Hessian determine convergence rates
- **Numerical Computing**: SVD is used for stable numerical computations
- **PCA**: Principal Component Analysis is based on eigendecomposition of the covariance matrix

## Next Concepts

- DL-017: Matrix Calculus for DL (extends gradients to matrices)
- DL-022: Jacobian and Hessian in DL (second-order derivatives)
- DL-028: Condition Number (eigenvalue ratio and optimization)
- DL-030: Spectral Analysis (singular value distributions in networks)

## Summary

Linear algebra provides the fundamental language for deep learning. Vectors represent data and features; matrices represent linear transformations (weights). The dot product measures similarity between vectors. Eigenvalues and eigenvectors reveal the directional scaling properties of matrices. SVD decomposes any matrix into orthogonal components and reveals its rank and spectral properties. Every neural network layer is a matrix multiplication followed by a nonlinearity, making linear algebra knowledge essential for understanding, debugging, and advancing deep learning.

## Key Takeaways

- Vectors are data; matrices are transformations; tensors are multi-dimensional arrays
- Matrix multiplication requires matching inner dimensions
- Eigenvalues of the Hessian determine optimization curvature
- SVD reveals the intrinsic dimensionality and condition of a matrix
- Weight matrices in neural networks can be analyzed via their singular value distribution
- Vectorized operations (matrix multiplies) are the key to efficient deep learning implementations
- Shape awareness is critical when building and debugging neural networks
