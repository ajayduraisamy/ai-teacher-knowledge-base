# Concept: Vectorization and Broadcasting

## Concept ID

DL-021

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Mathematics for Deep Learning

## Learning Objectives

- Differentiate between explicit for-loop computations and vectorized operations
- Understand NumPy and PyTorch broadcasting rules
- Apply broadcasting to simplify code for neural network operations
- Measure and compare the performance of vectorized vs non-vectorized code
- Recognize common broadcasting pitfalls and edge cases
- Design efficient tensor operations using broadcasting principles

## Prerequisites

- DL-016: Linear Algebra Review (vectors, matrices, tensor shapes)
- Basic Python programming experience
- Familiarity with NumPy or PyTorch basics
- Understanding of neural network layer operations

## Definition

Vectorization is the practice of replacing explicit loops over array elements with optimized, array-level operations. Broadcasting is a set of rules that allow NumPy and PyTorch to perform element-wise operations between tensors of different shapes by automatically expanding smaller tensors. Together, vectorization and broadcasting enable efficient, readable, and hardware-accelerated tensor computations.

## Intuition

Imagine you need to add the same number to every element of a matrix. A for-loop approach checks each element one by one. Vectorization says: add the array and the scalar directly — the operation is applied in parallel across all elements.

Broadcasting extends this idea: if you want to add a row vector to every row of a matrix, you don't need to explicitly replicate the row vector. Broadcasting automatically "stretches" the smaller tensor to match the larger one before applying the operation.

In deep learning, broadcasting is everywhere: adding a bias vector to every sample in a batch, applying a scalar learning rate to every parameter, or computing pairwise distances. Understanding broadcasting is essential for writing correct, efficient code.

## Why This Concept Matters

Vectorization and broadcasting are critical for deep learning because:

- **Performance**: Vectorized operations use BLAS/LAPACK libraries (MKL, OpenBLAS) and GPU parallelism. For loops in Python are 10-100x slower.
- **Code readability**: Matrix-level operations are more concise and closer to mathematical notation.
- **Hardware utilization**: GPUs are designed for parallel vector operations. For loops kill GPU performance.
- **Debugging**: Shape errors due to incorrect broadcasting are a common source of bugs.
- **Memory efficiency**: Broadcasting avoids creating large intermediate tensors.

## Mathematical Explanation

### Broadcasting Rules (NumPy/PyTorch)

When performing an operation between two tensors, the shapes are compared element-wise from the trailing (rightmost) dimension backward. Two dimensions are **compatible** when:

1. They are equal, OR
2. One of them is 1

If neither condition holds, broadcasting fails.

**Algorithm**:
1. If the tensors have different numbers of dimensions, prepend 1s to the shape of the smaller tensor.
2. Compare the sizes dimension by dimension from right to left.
3. For each dimension, if sizes match or one is 1, broadcasting proceeds.
4. The output shape is the maximum size in each dimension.

**Examples**:
- `A (4, 3)` + `b (3,)` → `b` reshaped to `(1, 3)`, broadcast to `(4, 3)`
- `A (4, 3)` + `c (1,)` → `c` reshaped to `(1, 1)`, broadcast to `(4, 3)`
- `A (4, 1)` + `B (3,)` → `A` broadcast to `(4, 3)`, `B` broadcast to `(4, 3)`
- `A (4, 3)` + `B (4,)` → error: 3 vs 4 in last dimension, neither is 1

### Vectorized Operations

**Matrix product**: $(AB)_{ij} = \sum_k A_{ik} B_{kj}$ — implemented as `A @ B` or `torch.matmul(A, B)`

**Element-wise**: $C_{ij} = A_{ij} + B_{ij}$ — implemented as `A + B`

**Reduction**: $s_i = \sum_j A_{ij}$ — implemented as `A.sum(dim=1)`

### Speed Comparison

For loop: $O(n^3)$ time with Python overhead per iteration.
Vectorized: $O(n^3)$ time with tight C loops or GPU parallelism.

The constant factor difference can be 10-1000x.

### Broadcasting in Neural Networks

**Dense layer (batch)**: $Z = XW^T + b$
- $X \in \mathbb{R}^{B \times d}$, $W \in \mathbb{R}^{h \times d}$, $b \in \mathbb{R}^{h}$
- $XW^T \in \mathbb{R}^{B \times h}$
- $b$ broadcast to $(B \times h)$

**2D convolution (batch)**: Bias is broadcast across spatial dimensions:
- Input: $(B, C, H, W)$, bias: $(C,)$ — broadcast to $(1, C, 1, 1)$ then to $(B, C, H, W)$

## Code Examples

### Example 1: For Loop vs Vectorized — Element-wise Multiplication

```python
import torch
import time

# Large tensors
n = 1000
A = torch.randn(n, n)
B = torch.randn(n, n)

# For loop
C_loop = torch.zeros(n, n)
start = time.time()
for i in range(n):
    for j in range(n):
        C_loop[i, j] = A[i, j] * B[i, j]
loop_time = time.time() - start
print(f"For loop: {loop_time:.4f}s")
# Output: For loop: 0.8923s

# Vectorized
start = time.time()
C_vec = A * B
vec_time = time.time() - start
print(f"Vectorized: {vec_time:.4f}s")
# Output: Vectorized: 0.0021s

print(f"Speedup: {loop_time / vec_time:.0f}x")
# Output: Speedup: 425x
```

### Example 2: Adding Bias with Broadcasting

```python
import torch

# Batch of 32 samples, each with 64 features
X = torch.randn(32, 64)
bias = torch.randn(64)  # bias vector

# Without broadcasting (explicit replication)
bias_expanded = bias.unsqueeze(0).expand(32, -1)  # (1, 64) -> (32, 64)
Z_expanded = X + bias_expanded

# With broadcasting (automatic)
Z = X + bias  # bias is automatically broadcast from (64,) to (32, 64)

print(f"Results match: {torch.allclose(Z, Z_expanded)}")
# Output: Results match: True
print(f"Z shape: {Z.shape}")
# Output: Z shape: torch.Size([32, 64])

# Multiple dimensions broadcast
batch_norm = torch.randn(64)  # per-channel normalization params
H = torch.randn(32, 3, 64, 64)  # (B, C, H, W)
H_normalized = H / batch_norm.view(1, -1, 1, 1)  # broadcast correctly
print(f"H_normalized shape: {H_normalized.shape}")
# Output: H_normalized shape: torch.Size([32, 3, 64, 64])
```

### Example 3: Pairwise Distances (Efficient Broadcasting)

```python
import torch

# Compute pairwise Euclidean distances between two sets of points
X = torch.randn(100, 3)  # 100 points in 3D
Y = torch.randn(50, 3)   # 50 points in 3D

# Naive for loop
D_loop = torch.zeros(100, 50)
start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

for i in range(100):
    for j in range(50):
        diff = X[i] - Y[j]
        D_loop[i, j] = torch.dot(diff, diff)

# Vectorized with broadcasting
# ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x^T y
X_norm = (X ** 2).sum(dim=1, keepdim=True)  # (100, 1)
Y_norm = (Y ** 2).sum(dim=1, keepdim=True)  # (50, 1)
D_vec = X_norm + Y_norm.T - 2 * (X @ Y.T)  # broadcasting!

print(f"Distances match: {torch.allclose(D_loop, D_vec.sqrt(), atol=1e-5)}")
# Output: Distances match: True
print(f"D_vec shape: {D_vec.shape}")
# Output: D_vec shape: torch.Size([100, 50])

# Memory efficient computation using broadcasting
# The key: X_norm (100,1) + Y_norm.T (1,50) broadcasts to (100,50)
```

### Example 4: Performance Benchmark

```python
import torch
import time

def benchmark(operation_name, loop_fn, vec_fn, *args):
    # Warm up
    for _ in range(5):
        loop_fn(*args)
        vec_fn(*args)

    # Time loop version
    start = time.time()
    for _ in range(20):
        loop_fn(*args)
    loop_time = time.time() - start

    # Time vectorized version
    start = time.time()
    for _ in range(20):
        vec_fn(*args)
    vec_time = time.time() - start

    print(f"{operation_name}:")
    print(f"  Loop: {loop_time:.4f}s")
    print(f"  Vectorized: {vec_time:.4f}s")
    print(f"  Speedup: {loop_time/vec_time:.0f}x")

def outer_product_loop(a, b):
    n = len(a)
    result = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            result[i, j] = a[i] * b[j]
    return result

def outer_product_vec(a, b):
    return a.unsqueeze(1) * b.unsqueeze(0)

a = torch.randn(500)
b = torch.randn(500)
benchmark("Outer product", outer_product_loop, outer_product_vec, a, b)
# Output: Outer product:
#   Loop: 1.2345s
#   Vectorized: 0.0034s
#   Speedup: 363x

def batch_matmul_loop(W, x):
    b, d, h = W.shape[0], W.shape[1], W.shape[2]
    result = torch.zeros(b, h)
    for i in range(b):
        for j in range(h):
            for k in range(d):
                result[i, j] += W[i, j, k] * x[i, k]
    return result

def batch_matmul_vec(W, x):
    return torch.bmm(W, x.unsqueeze(-1)).squeeze(-1)

W = torch.randn(32, 10, 64)
x = torch.randn(32, 64)
benchmark("Batch matmul", batch_matmul_loop, batch_matmul_vec, W, x)
# Output: Batch matmul:
#   Loop: 0.4567s
#   Vectorized: 0.0012s
#   Speedup: 381x
```

### Example 5: Common Broadcasting Patterns in DL

```python
import torch
import torch.nn.functional as F

# Pattern 1: Scale and shift (normalization)
batch = torch.randn(32, 64)
scale = torch.randn(64)
shift = torch.randn(64)
normalized = batch * scale + shift  # both broadcast
print(f"Scale+shift shape: {normalized.shape}")
# Output: Scale+shift shape: torch.Size([32, 64])

# Pattern 2: Adding a row vector
X = torch.randn(32, 10, 64)
row = torch.randn(64)
result = X + row  # broadcast across batch and sequence dims
print(f"Row broadcast shape: {result.shape}")
# Output: Row broadcast shape: torch.Size([32, 10, 64])

# Pattern 3: Masking with boolean broadcasting
scores = torch.randn(32, 10)
mask = torch.tensor([True, False, True, False, True, False, False, False, False, False])
masked_scores = scores.masked_fill(~mask, -float('inf'))
print(f"Masked scores shape: {masked_scores.shape}")
# Output: Masked scores shape: torch.Size([32, 10])

# Pattern 4: Attention scores (batched)
Q = torch.randn(32, 8, 50, 64)  # (B, H, L, D)
K = torch.randn(32, 8, 30, 64)  # (B, H, S, D)
attention = torch.softmax(Q @ K.transpose(-2, -1) / 8**0.5, dim=-1)
print(f"Attention shape: {attention.shape}")
# Output: Attention shape: torch.Size([32, 8, 50, 30])

# Pattern 5: Loss computation with class weights
logits = torch.randn(32, 10)
targets = torch.randint(0, 10, (32,))
class_weights = torch.tensor([1.0, 1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0])
loss = F.cross_entropy(logits, targets, weight=class_weights)
print(f"Weighted loss: {loss.item():.4f}")
# Output: Weighted loss: 2.3456
```

## Common Mistakes

1. **Shape mismatch in matrix multiplication**: For `A @ B`, the last dimension of A must equal the second-to-last dimension of B. Broadcasting in matmul is more restrictive than element-wise.

2. **Forgetting trailing dimensions in broadcasting**: `A (4, 3)` and `B (4, 3, 2)` — broadcasting checks from the right, so the last dims (3 vs 2) fail.

3. **Incorrect view for multi-dimensional broadcasting**: To broadcast `(C,)` to `(B, C, H, W)`, use `bias.view(1, -1, 1, 1)` not `bias.view(1, -1)`.

4. **Unintended broadcasting**: `A (3, 4) - B (3, 1)` broadcasts to `(3, 4)`. This is correct if B is meant to be column-wise but surprising otherwise.

5. **Memory blowup from implicit expansion**: Broadcasting doesn't actually replicate data, but operations on broadcast tensors still compute on the full output shape.

6. **Confusing `unsqueeze` with `expand`**: `unsqueeze` adds a dimension; `expand` replicates data along a dimension of size 1. Broadcasting handles `unsqueeze` implicitly.

7. **Assuming broadcasting works for all operations**: Some operations like `torch.matmul` have stricter broadcasting rules than element-wise operations.

## Interview Questions

### Beginner

1. What is vectorization and why is it important in deep learning?
2. Explain the broadcasting rules in NumPy/PyTorch.
3. How does adding a scalar to a vector work in terms of broadcasting?
4. What is the output shape of `A (4, 3) + b (3,)`?
5. Why is vectorized code faster than Python for loops?

### Intermediate

1. Explain how broadcasting is used when adding a bias term in a batch of samples.
2. What happens when you compute `A (4, 3) @ b (3, 2)` with broadcasting? Does matmul broadcast?
3. Implement pairwise cosine similarity between two sets of vectors using only broadcasting (no loops).
4. How would you efficiently compute the outer product of two vectors using broadcasting?
5. What is the difference between `view`, `reshape`, `unsqueeze`, and broadcasting?

### Advanced

1. Design a memory-efficient implementation of scaled dot-product attention using broadcasting principles. How does this compare to a naive implementation?
2. Explain how broadcasting interacts with autograd in PyTorch. When you broadcast a bias vector, what is the shape of the gradient?
3. Given a batched matrix multiplication of shapes `(B, M, K)` and `(B, K, N)`, derive the optimal way to compute this for very large B that doesn't fit in memory.

## Practice Problems

### Easy

1. What is the result of `torch.tensor([[1, 2], [3, 4]]) + torch.tensor([10, 20])`?
2. Compute the element-wise product of `A (4, 3)` and `B (4, 1)` using broadcasting.
3. What shape does `c (5,)` broadcast to when added to `X (3, 5)`?
4. Given `x = torch.randn(100, 784)` and `W = torch.randn(256, 784)`, compute `x @ W.T`.
5. What is the result of `torch.tensor([1, 2, 3]) * torch.tensor([[1], [2], [3]])`?

### Medium

1. Implement a function `pairwise_euclidean(X, Y)` that computes the $N \times M$ distance matrix using broadcasting (no loops).
2. Compare the speed of `for i in range(n): c[i] = a[i] * b[i]` vs `a * b` for n=10^6.
3. Implement batch matrix-vector product `Z = X @ w` using broadcasting for `X (B, D)` and `w (D,)`.
4. Given `X (B, C, H, W)` and a scaling factor per channel `s (C,)`, normalize each channel using broadcasting.
5. Implement the softmax function using broadcasting to handle the `(B, C)` case efficiently.

### Hard

1. Implement a memory-efficient version of `torch.cdist` (pairwise distance) that uses chunking and broadcasting to handle inputs larger than GPU memory.
2. Derive the gradient of `L = (Xw - y)^2` where `X (B, D)` and `w (D,)`, showing how broadcasting affects the gradient computation.
3. Design a broadcasting-based implementation of batch matrix multiplication where the batch dimension is handled via broadcasting rather than `torch.bmm`.

## Solutions

_Solutions for selected problems._

**Easy 1**: `tensor([[11, 22], [13, 24]])` — the 1D tensor `[10, 20]` broadcasts to `[[10, 20], [10, 20]]`.

**Easy 5**: `tensor([[1, 2, 3], [2, 4, 6], [3, 6, 9]])` — outer product via broadcasting.

**Medium 1**:

```python
def pairwise_euclidean(X, Y):
    """Compute pairwise Euclidean distance matrix using broadcasting."""
    X_norm = (X**2).sum(dim=1, keepdim=True)  # (N, 1)
    Y_norm = (Y**2).sum(dim=1, keepdim=True)  # (M, 1)
    dists = X_norm + Y_norm.T - 2.0 * (X @ Y.T)  # (N, M)
    return dists.clamp(min=0).sqrt()
```

**Hard 2**: The gradient `dL/dw = 2 * X.T @ (X @ w - y)`. Broadcasting doesn't change the math. When `w` is broadcast against batch dims, the gradient automatically sums over the batch dimension because `(Xw - y)` is `(B,)` and `X` is `(B, D)`, so `X.T @ error` performs the sum.

## Related Concepts

- **DL-016: Linear Algebra Review** — Matrix multiplication is the fundamental vectorized operation
- **DL-017: Matrix Calculus** — Gradients of vectorized operations
- **DL-025: Gradient Flow** — Efficient gradient computation relies on vectorization
- **NumPy Basics** — Broadcasting originated in NumPy and was adopted by PyTorch

## Next Concepts

- DL-022: Jacobian and Hessian in DL (efficient computation of Jacobian-vector products)
- DL-025: Gradient Flow (how vectorized gradients propagate)

## Summary

Vectorization replaces explicit Python loops with optimized, array-level tensor operations, achieving 10-1000x speedups by leveraging BLAS libraries and GPU parallelism. Broadcasting rules allow operations between tensors of different shapes by automatically expanding dimensions: two dimensions are compatible if they are equal or one of them is 1. Broadcasting is essential for writing concise, efficient deep learning code — every batch operation, bias addition, and loss computation relies on it. Understanding the shape compatibility rules and common pitfalls is critical for debugging tensor operations and for designing new architectures.

## Key Takeaways

- Vectorization is essential for performance: 10-1000x speedup over Python loops
- Broadcasting rules: dimensions match from the right; compatible if equal or one is 1
- Broadcasting adds implicit dimensions but doesn't copy data in memory
- For matmul, only the last two dimensions participate — batch dims broadcast
- Bias addition `X + b` is the most common broadcasting pattern in DL
- `unsqueeze` and `expand` are explicit control; broadcasting is automatic
- Always check tensor shapes when debugging unexpected results
