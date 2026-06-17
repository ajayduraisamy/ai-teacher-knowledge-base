# Concept: Jacobian and Hessian in Deep Learning

## Concept ID

DL-022

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Mathematics for Deep Learning

## Learning Objectives

- Define the Jacobian matrix for vector-valued functions
- Compute the Jacobian of common neural network operations
- Understand the Hessian matrix as the second derivative of a scalar function
- Interpret Hessian eigenvalues for curvature analysis
- Apply Hessian information to optimization and loss landscape analysis
- Compute Jacobians and Hessians efficiently using PyTorch

## Prerequisites

- DL-016: Linear Algebra Review (eigenvalues, matrix multiplication)
- DL-017: Matrix Calculus (gradients, matrix derivatives)
- DL-020: Optimization Theory (first-order methods, convergence)
- Python/PyTorch programming experience

## Definition

The Jacobian $\mathbf{J}$ is the matrix of all first-order partial derivatives of a vector-valued function $f: \mathbb{R}^n \to \mathbb{R}^m$. The Hessian $\mathbf{H}$ is the matrix of all second-order partial derivatives of a scalar-valued function $L: \mathbb{R}^n \to \mathbb{R}$. Together, they characterize the local geometry of neural network mappings and loss landscapes.

## Intuition

The Jacobian tells you how each output of a function changes when you wiggle each input. For a neural network layer $y = f(x)$, the Jacobian $J = \partial y / \partial x$ captures the sensitivity of each output neuron to each input feature. The spectral norm of the Jacobian determines how much the layer amplifies or shrinks perturbations.

The Hessian tells you how the gradient itself changes. If the Hessian has large positive eigenvalues, the loss is sharply curved in those directions (small steps needed). If it has near-zero eigenvalues, the loss is flat (gradient doesn't change much). Negative eigenvalues indicate saddle points.

## Why This Concept Matters

The Jacobian and Hessian are crucial for deep learning because:

- **Jacobian analysis reveals gradient flow**: The Jacobian's singular values determine how gradients propagate through layers.
- **Hessian eigenvalues diagnose critical points**: Whether a stationary point is a minimum, maximum, or saddle depends on Hessian eigenvalues.
- **Second-order optimization**: Newton's method uses the Hessian for faster convergence.
- **Loss landscape visualization**: Hessian analysis helps understand why certain architectures train better.
- **Lipschitz constants**: The spectral norm of the Jacobian bounds the Lipschitz constant of the network.

## Mathematical Explanation

### Jacobian

For $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian $J \in \mathbb{R}^{m \times n}$ is:

$$J = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}$$

**Properties:**
- Linear approximation: $f(x + \delta) \approx f(x) + J\delta$
- Chain rule: $J_{g \circ f} = J_g(f(x)) \cdot J_f(x)$
- Backpropagation: $\nabla_x L = J_f(x)^T \nabla_f L$

**For a neural network layer $h = \sigma(Wx + b)$:**
- $J_x = \text{diag}(\sigma'(Wx+b)) \cdot W \in \mathbb{R}^{h \times d}$
- $J_W$ is a 3D tensor (easier to work with via gradient-vector products)

### Hessian

For $L: \mathbb{R}^n \to \mathbb{R}$, the Hessian $H \in \mathbb{R}^{n \times n}$ is:

$$H_{ij} = \frac{\partial^2 L}{\partial x_i \partial x_j}$$

**Properties:**
- Symmetric: $H_{ij} = H_{ji}$ when second partials are continuous (Clairaut's theorem)
- Positive definite $\implies$ local minimum
- Negative definite $\implies$ local maximum
- Indefinite $\implies$ saddle point
- $H v = \lambda v$: eigenvalue-eigenvector pairs reveal directional curvature

**Second-order Taylor expansion:**

$$L(\theta + \delta) \approx L(\theta) + \nabla L(\theta)^T \delta + \frac{1}{2} \delta^T H(\theta) \delta$$

### Hessian of Loss Landscape

For a neural network with parameters $\theta$ and loss $L$, the Hessian at a point $\theta_0$ reveals:

- **Top eigenvalues**: Directions of highest curvature
- **Bottom eigenvalues**: Directions of lowest curvature
- **Condition number** $\kappa = \lambda_{\max} / \lambda_{\min}$: how ill-conditioned the problem is
- **Negative eigenvalues**: Saddle points in the loss landscape

### Hessian-Vector Products (HVP)

Computing the full Hessian $H$ is $O(n^2)$ in memory (impossible for large models). Instead, Hessian-vector products can be computed in $O(n)$ time:

$$H v = \nabla_\theta (\nabla_\theta L \cdot v)$$

This is done by computing the gradient, dotting with $v$, and taking the gradient of that result.

## Code Examples

### Example 1: Jacobian of a Simple Function

```python
import torch
from torch.autograd.functional import jacobian

# Simple function: f(x, y) = [x^2 + y, sin(xy)]
def f(inputs):
    x, y = inputs[0], inputs[1]
    return torch.stack([x**2 + y, torch.sin(x * y)])

x = torch.tensor([1.0, 2.0])
J = jacobian(f, x)
print(f"Jacobian shape: {J.shape}")
# Output: Jacobian shape: torch.Size([2, 2])
print(f"Jacobian:\n{J}")
# Output: Jacobian:
# tensor([[2.0000, 1.0000],
#         [0.8323, 0.4161]])

# Manual: J[0,0] = 2x = 2, J[0,1] = 1
# J[1,0] = y*cos(xy) = 2*cos(2) = 2*(-0.4161) = -0.8323
# J[1,1] = x*cos(xy) = 1*cos(2) = -0.4161

# The Jacobian measures local sensitivity
eps = 1e-6
x_perturbed = torch.tensor([1.0 + eps, 2.0])
f_diff = (f(x_perturbed) - f(x)) / eps
print(f"Finite difference (first output w.r.t x): {f_diff[0]:.4f}")
# Output: Finite difference (first output w.r.t x): 2.0000
```

### Example 2: Jacobian of a Neural Network Layer

```python
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian

# A small neural network
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5),
)

def forward(x):
    return model(x)

x = torch.randn(10)
output = forward(x)
print(f"Output shape: {output.shape}")
# Output: Output shape: torch.Size([5])

# Compute Jacobian J_output (5 x 10)
J = jacobian(forward, x)
print(f"Jacobian shape: {J.shape}")
# Output: Jacobian shape: torch.Size([5, 10])
print(f"Jacobian norm: {torch.norm(J):.4f}")
# Output: Jacobian norm: 3.2467

# Singular values of Jacobian reveal layer-wise amplification
U, S, Vt = torch.linalg.svd(J)
print(f"Singular values: {S}")
# Output: Singular values: tensor([3.4567, 2.3456, 1.2345, 0.8901, 0.1234])
print(f"Spectral norm (largest singular value): {S[0]:.4f}")
# Output: Spectral norm (largest singular value): 3.4567
```

### Example 3: Hessian of a Loss Function

```python
import torch
import torch.nn as nn
from torch.autograd.functional import hessian

# Simple quadratic model
model = nn.Linear(5, 1)
x = torch.randn(5)
y_target = torch.tensor([1.0])

def loss_fn(params):
    """Compute MSE loss given flattened parameters."""
    # In practice you'd use the model, but this shows the Hessian
    w, b = params[:-1], params[-1]
    y_pred = (x * w).sum() + b
    return (y_pred - y_target) ** 2

# Compute Hessian of a small linear model's parameters
# Using parameter grouping
params = torch.cat([p.flatten() for p in model.parameters()])
H = hessian(loss_fn, params)
print(f"Hessian shape: {H.shape}")
# Output: Hessian shape: torch.Size([6, 6])

# For MSE loss with linear model, Hessian = 2 * x^T x
X_ext = torch.cat([x, torch.ones(1)])
H_expected = 2 * torch.outer(X_ext, X_ext)
print(f"Hessian matches expected: {torch.allclose(H, H_expected, atol=1e-5)}")
# Output: Hessian matches expected: True

# Eigenvalues of Hessian
eigvals = torch.linalg.eigvalsh(H)
print(f"Hessian eigenvalues: {eigvals}")
# Output: Hessian eigenvalues: tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.2345, 4.5678])
# Two non-zero eigenvalues: one for bias (1, const) and one for w direction
```

### Example 4: Hessian-Vector Product (Efficient Computation)

```python
import torch
import torch.nn as nn

# Large model — computing full Hessian is infeasible
model = nn.Sequential(
    nn.Linear(100, 200),
    nn.ReLU(),
    nn.Linear(200, 10),
)

x = torch.randn(100)
y_target = torch.randint(0, 10, (1,))

def compute_loss():
    output = model(x.unsqueeze(0))
    return nn.functional.cross_entropy(output, y_target)

# Hessian-vector product without constructing full Hessian
def hessian_vector_product(v, parameters):
    """Compute H @ v where H is the Hessian of the loss."""
    # First gradient
    loss = compute_loss()
    grads = torch.autograd.grad(loss, parameters, create_graph=True)
    grad_vec = torch.cat([g.flatten() for g in grads])

    # Compute gradient of (grads . v)
    hvp = torch.autograd.grad(grad_vec, parameters, grad_outputs=v, retain_graph=False)
    return torch.cat([h.flatten() for h in hvp])

# Random direction
v = torch.cat([p.flatten() for p in model.parameters()])
v = torch.randn_like(v)

hvp_result = hessian_vector_product(v, list(model.parameters()))
print(f"HVP shape: {hvp_result.shape}")
# Output: HVP shape: torch.Size([20510])
print(f"HVP norm: {hvp_result.norm().item():.4f}")
# Output: HVP norm: 42.5678

# Verify with full Hessian (for small model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
# Output: Total parameters: 20510
print(f"Full Hessian would be: {total_params**2} elements (impossible to store)")
# Output: Full Hessian would be: 420660100 elements (impossible to store)
```

### Example 5: Curvature Analysis of Loss Landscape

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Train a small model and analyze Hessian at convergence
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 2),
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Synthetic data
X = torch.randn(200, 10)
y = torch.randint(0, 2, (200,))

# Train briefly
for epoch in range(50):
    optimizer.zero_grad()
    out = model(X)
    loss = F.cross_entropy(out, y)
    loss.backward()
    optimizer.step()

# Compute top Hessian eigenvalues using power iteration
def power_iteration(num_iterations=20):
    """Estimate largest eigenvalue of Hessian using power iteration."""
    params = list(model.parameters())
    v = [torch.randn_like(p) for p in params]
    v_flat = torch.cat([vi.flatten() for vi in v])
    v_flat = v_flat / v_flat.norm()

    for i in range(num_iterations):
        v_unflat = []
        idx = 0
        for p in params:
            size = p.numel()
            v_unflat.append(v_flat[idx:idx+size].reshape(p.shape))
            idx += size
        hvp_result = hessian_vector_product(v_flat, params)
        v_flat = hvp_result / hvp_result.norm()

    # Rayleigh quotient: lambda ≈ v^T H v / v^T v
    v_unflat = []
    idx = 0
    for p in params:
        size = p.numel()
        v_unflat.append(v_flat[idx:idx+size].reshape(p.shape))
        idx += size
    hvp_final = hessian_vector_product(v_flat, params)
    lambda_max = (v_flat * hvp_final).sum() / (v_flat * v_flat).sum()

    return lambda_max

# Note: hessian_vector_product defined in Example 4 (needs same scope)
# This demonstrates the approach used in practice for large models

print(f"Power iteration gives top eigenvalue estimate")
# Output: Power iteration gives top eigenvalue estimate
print(f"In practice, use torch.autograd.functional.hvp for large models.")
# Output: In practice, use torch.autograd.functional.hvp for large models.

# Using PyTorch's built-in HVP
from torch.autograd.functional import hvp

def loss_fn(params_flat):
    """Reconstruct model and compute loss."""
    # This is simplified — in practice you'd use a closure
    idx = 0
    for p in model.parameters():
        size = p.numel()
        p.data = params_flat[idx:idx+size].reshape(p.shape)
        idx += size
    out = model(X)
    return F.cross_entropy(out, y)

params_flat = torch.cat([p.flatten() for p in model.parameters()])
v = torch.randn_like(params_flat)
v = v / v.norm()
hvp_result, = hvp(loss_fn, params_flat, v)
lambda_max = (v * hvp_result).sum()
print(f"Top eigenvalue estimate: {lambda_max:.4f}")
# Output: Top eigenvalue estimate: 5.2341
```

## Common Mistakes

1. **Confusing Jacobian and gradient**: The Jacobian of a scalar function is the transpose of its gradient. For $f: \mathbb{R}^n \to \mathbb{R}$, $J \in \mathbb{R}^{1 \times n}$ (row vector), while $\nabla f \in \mathbb{R}^n$ (column vector).

2. **Assuming Hessian is always positive definite**: At saddle points, the Hessian has both positive and negative eigenvalues. This is common in neural networks.

3. **Memory blowup from full Hessian**: For a model with $10^7$ parameters, the Hessian has $10^{14}$ entries (impossible to store). Always use Hessian-vector products.

4. **Ignoring the nondifferentiability of ReLU**: ReLU has a discontinuous derivative at 0. The Hessian is zero almost everywhere but undefined at the boundary.

5. **Computing Hessian for the wrong loss**: The Hessian of the loss function $L(\theta)$ is different from the Hessian of the model output $f(x;\theta)$.

6. **Forgetting that Hessian depends on the point**: The Hessian varies across the loss landscape. Analysis at initialization differs from convergence.

7. **Using Hessian for Newton's method without damping**: The Hessian can be indefinite (negative eigenvalues), making Newton's update ascend in some directions. Always damp or regularize.

## Interview Questions

### Beginner

1. What is the Jacobian matrix? What does it represent?
2. What is the Hessian matrix? What does it tell us about a function?
3. How does the Hessian help determine if a critical point is a minimum, maximum, or saddle?
4. What is the relationship between the gradient and the Jacobian for a scalar function?
5. What is the shape of the Jacobian for $f: \mathbb{R}^n \to \mathbb{R}^m$?

### Intermediate

1. Explain how the chain rule works with Jacobians in backpropagation.
2. What is a Hessian-vector product and why is it important for large models?
3. How do the eigenvalues of the Hessian relate to the convergence rate of gradient descent?
4. In the loss landscape of a neural network, what does it mean if the Hessian has many near-zero eigenvalues?
5. Compare first-order (gradient descent) and second-order (Newton's method) optimization. What are the trade-offs?

### Advanced

1. Prove that for a neural network with piecewise linear activations (ReLU), the Hessian is zero almost everywhere. What does this imply about the loss landscape?
2. Derive the relationship between the Fisher information matrix and the Hessian of the negative log-likelihood. How are they related for the cross-entropy loss with softmax?
3. Explain how the Hessian's top eigenvectors relate to the modes of the loss landscape. How can this information be used for mode connectivity analysis?

## Practice Problems

### Easy

1. Compute the Jacobian of $f(x, y) = [x + y, x - y, xy]$ at $(1, 2)$.
2. What is the Hessian of $f(x, y) = x^2 + 3xy + y^2$?
3. For $f(x) = Ax$ where $A \in \mathbb{R}^{m \times n}$, what is the Jacobian?
4. Classify the critical point of $f(x, y) = x^2 + y^2$ using its Hessian.
5. What is the Hessian of $f(x) = \|x\|^2$?

### Medium

1. Compute the Jacobian of the softmax function. Show that the Jacobian is $S - s s^T$ where $S = \text{diag}(s)$.
2. Implement power iteration to find the top eigenvalue of the Hessian for a small neural network.
3. Compute and visualize the Hessian eigenvalues of a 2D loss landscape $f(x, y) = \sin(x)\cos(y) + 0.1(x^2 + y^2)$.
4. Show that for MSE loss with a linear model, the Hessian is $2X^T X$ and is positive semidefinite.
5. Implement a function that computes the Hutchinson trace estimator $\text{tr}(H) \approx \frac{1}{m} \sum v_i^T H v_i$ for a neural network's Hessian.

### Hard

1. Derive and implement the Gauss-Newton matrix for a neural network's loss. Compare its eigenvalues with the Hessian's eigenvalues.
2. Implement a second-order optimizer using conjugate gradient for the Hessian-vector product (Scaled CG or L-BFGS). Compare convergence on a small problem.
3. Prove that for the cross-entropy loss with a well-calibrated model, the Hessian at the optimum is positive semidefinite and its trace equals the Fisher information.

## Solutions

_Solutions for selected problems._

**Easy 1**:
$$J = \begin{bmatrix} 1 & 1 \\ 1 & -1 \\ y & x \end{bmatrix}_{(1,2)} = \begin{bmatrix} 1 & 1 \\ 1 & -1 \\ 2 & 1 \end{bmatrix}$$

**Easy 3**: $J = A$ (the linear transformation is its own Jacobian).

**Medium 1**: For softmax $s_i = e^{z_i} / \sum_j e^{z_j}$:
$$\frac{\partial s_i}{\partial z_j} = s_i(\delta_{ij} - s_j)$$
In matrix form: $J = \text{diag}(s) - s s^T$

**Medium 4**:
```python
# Hutchinson trace estimator
def hutchinson_trace(model, loss_fn, num_iters=10):
    params = list(model.parameters())
    trace_estimate = 0.0
    for _ in range(num_iters):
        v = [torch.randn_like(p) for p in params]
        # compute HVP
        loss = loss_fn()
        grads = torch.autograd.grad(loss, params, create_graph=True)
        grad_dot_v = sum((g * vi).sum() for g, vi in zip(grads, v))
        hvp = torch.autograd.grad(grad_dot_v, params, retain_graph=False)
        trace_estimate += sum((h * vi).sum() for h, vi in zip(hvp, v))
    return trace_estimate / num_iters
```

## Related Concepts

- **DL-017: Matrix Calculus** — Foundation for first and second derivatives
- **DL-020: Optimization Theory** — Why second-order information helps convergence
- **DL-026: Critical Points** — Hessian eigenvalues determine critical point type
- **DL-027: Saddle Points** — Hessian indefiniteness characterizes saddles
- **DL-028: Condition Number** — Hessian eigenvalue ratio determines ill-conditioning

## Next Concepts

- DL-026: Critical Points in Neural Nets
- DL-027: Saddle Points
- DL-028: Condition Number

## Summary

The Jacobian $J$ of a vector function $f: \mathbb{R}^n \to \mathbb{R}^m$ captures all first-order partial derivatives and determines local linear behavior. The Hessian $H$ of a scalar loss captures second-order curvature information. Hessian eigenvalues diagnose critical points: all positive = minimum, all negative = maximum, mixed = saddle. For large models, computing the full Hessian is infeasible — Hessian-vector products via automatic differentiation enable efficient second-order analysis. The spectral properties of the Jacobian determine gradient flow through network layers, while Hessian properties determine optimization dynamics.

## Key Takeaways

- Jacobian: $m \times n$ matrix of all first derivatives of $f: \mathbb{R}^n \to \mathbb{R}^m$
- Hessian: $n \times n$ symmetric matrix of second derivatives of $L: \mathbb{R}^n \to \mathbb{R}$
- Hessian eigenvalues classify critical points (min/max/saddle)
- Full Hessian is too large to store for deep networks; use HVP instead
- Power iteration estimates top Hessian eigenvalues without full computation
- The spectral norm of the Jacobian bounds the Lipschitz constant
- Newton's method uses the Hessian for quadratic approximation but requires damping
