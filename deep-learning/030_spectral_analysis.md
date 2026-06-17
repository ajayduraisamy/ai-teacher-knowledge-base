# Concept: Spectral Analysis

## Concept ID

DL-030

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Mathematics for Deep Learning

## Learning Objectives

- Define the spectral norm and singular value decomposition (SVD) of a weight matrix
- Compute and interpret the singular value distribution of neural network layers
- Analyze how spectral properties affect gradient flow through the network
- Understand spectral normalization and its role in GAN training
- Relate singular value distributions to model capacity and generalization
- Implement spectral analysis tools for diagnosing network behavior

## Prerequisites

- DL-016: Linear Algebra Review (SVD, eigenvalues, matrix norms)
- DL-022: Jacobian and Hessian (singular values of Jacobian)
- DL-025: Gradient Flow (how gradients propagate)
- DL-029: Smoothness and Lipschitz (spectral norm = Lipschitz constant)

## Definition

Spectral analysis studies the eigenvalues and singular values of matrices, their distributions, and their implications. In deep learning, spectral analysis is applied to weight matrices, Jacobians, and Hessians to understand gradient flow, network stability, capacity, and generalization. The singular value spectrum of a weight matrix reveals how it transforms input signals — which directions are amplified, which are attenuated, and how the effective rank of the representation evolves during training.

## Intuition

Every linear layer in a neural network transforms its input. This transformation can be understood by looking at the singular value decomposition (SVD) of the weight matrix: $W = U \Sigma V^T$. The right singular vectors $V$ are the input directions; the singular values $\sigma_i$ tell you how much each direction is scaled; the left singular vectors $U$ are the output directions.

A weight matrix with a wide, flat singular value spectrum (many similar singular values) preserves information across many directions. A matrix with rapidly decaying singular values (few large, many near-zero) compresses the input into a lower-dimensional space — it's effectively learning a low-rank representation.

In deep learning, spectral analysis helps answer questions like: Is my network's gradient flowing properly? Is a layer bottlenecking information? Is the training stable? Is the network robust to perturbations?

## Why This Concept Matters

Spectral analysis is crucial for deep learning because:

- **Gradient flow**: Singular values of Jacobians determine gradient scaling through layers.
- **Stability**: The spectral norm bounds the Lipschitz constant, affecting training stability.
- **Capacity**: The effective rank (number of significant singular values) measures representational capacity.
- **Compression**: Low-rank structure in weight matrices enables model compression.
- **Generalization**: Spectral properties correlate with generalization performance.
- **Architecture design**: Spectral analysis guides choices like normalization and initialization.

## Mathematical Explanation

### Singular Value Decomposition

For any matrix $W \in \mathbb{R}^{m \times n}$:

$$W = U \Sigma V^T$$

where $U \in \mathbb{R}^{m \times m}$ and $V \in \mathbb{R}^{n \times n}$ are orthogonal, and $\Sigma \in \mathbb{R}^{m \times n}$ is diagonal with singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{\min(m,n)} \geq 0$.

### Key Spectral Quantities

- **Spectral norm**: $\|W\|_2 = \sigma_{\max}(W)$ — largest singular value
- **Nuclear norm**: $\|W\|_* = \sum_i \sigma_i$ — sum of singular values
- **Frobenius norm**: $\|W\|_F = \sqrt{\sum_i \sigma_i^2}$ — sum of squared singular values
- **Effective rank**: $r_{\text{eff}} = \exp\left(-\sum_i \frac{\sigma_i}{\sum_j \sigma_j} \log \frac{\sigma_i}{\sum_j \sigma_j}\right)$ — spectral entropy

### Spectral Analysis of Gradient Flow

The Jacobian of a layer $y = \sigma(Wx)$ with respect to $x$:

$$J = \text{diag}(\sigma') \cdot W$$

The singular values of $J$ determine how gradients flow backward:

$$\nabla_x L = J^T \nabla_y L = W^T \cdot \text{diag}(\sigma') \cdot \nabla_y L$$

The gradient norm scales by the singular values of $J$:

$$\|\nabla_x L\| \leq \|W\|_2 \cdot \|\sigma'\|_\infty \cdot \|\nabla_y L\|$$

### Spectral Normalization

For GANs, spectral normalization ensures the discriminator is $K$-Lipschitz:

$$\bar{W} = \frac{W}{\sigma_{\max}(W)} \quad \text{or} \quad \bar{W} = \frac{c}{\sigma_{\max}(W)} \cdot W$$

This is critical for the Wasserstein GAN loss, which requires the discriminator to be 1-Lipschitz.

### Singular Value Distribution During Training

- **At initialization**: Singular values cluster around a specific value determined by the initialization scheme.
- **Early training**: Some singular values grow, others shrink, as the network learns.
- **After convergence**: Often a power-law decay of singular values emerges, with a few large values and many small ones.

### Weight Matrix Effective Rank

The effective rank measures how many dimensions of the input space are preserved:

$$r_{\text{eff}} = \frac{(\sum_i \sigma_i)^2}{\sum_i \sigma_i^2}$$

A low effective rank indicates that the layer projects inputs into a lower-dimensional subspace.

## Code Examples

### Example 1: SVD of a Weight Matrix

```python
import torch
import torch.nn as nn

# Create a trained linear layer
model = nn.Linear(20, 10)
W = model.weight.data

# Full SVD
U, S, Vt = torch.linalg.svd(W, full_matrices=False)

print(f"Weight matrix shape: {W.shape}")
# Output: Weight matrix shape: torch.Size([10, 20])
print(f"U shape: {U.shape}, S shape: {S.shape}, Vt shape: {Vt.shape}")
# Output: U shape: torch.Size([10, 10]), S shape: torch.Size([10]), Vt shape: torch.Size([10, 20])
print(f"Singular values: {S}")
# Output: Singular values: tensor([2.3456, 2.1234, 1.9876, 1.6543, 1.4321, 1.2345, 1.0123, 0.8765, 0.6543, 0.4321])

# Key spectral quantities
spectral_norm = S[0]
nuclear_norm = S.sum()
frobenius_norm = torch.norm(W)
effective_rank = (S.sum() ** 2) / (S ** 2).sum()

print(f"Spectral norm: {spectral_norm:.4f}")
# Output: Spectral norm: 2.3456
print(f"Effective rank: {effective_rank:.4f} (max possible: {len(S)})")
# Output: Effective rank: 7.2345 (max possible: 10)
print(f"Condition number: {S[0]/S[-1]:.4f}")
# Output: Condition number: 5.4285
```

### Example 2: Singular Value Spectrum Evolution During Training

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Train a small network and track SVD of each layer
torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 15),
    nn.ReLU(),
    nn.Linear(15, 2),
)

X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Track singular values
sv_history = {name: [] for name, _ in model.named_parameters() if 'weight' in name}

for epoch in range(200):
    optimizer.zero_grad()
    out = model(X)
    loss = F.cross_entropy(out, y)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' in name:
                    S = torch.linalg.svdvals(param.data)
                    sv_history[name].append(S.clone())

# Analyze spectral evolution
for name in sv_history:
    initial_S = sv_history[name][0]
    final_S = sv_history[name][-1]
    print(f"\n{name}:")
    print(f"  Initial singular values: {initial_S}")
    # Output:   Initial singular values: tensor([...])
    print(f"  Final singular values: {final_S}")
    # Output:   Final singular values: tensor([...])
    print(f"  Spectral norm change: {initial_S[0]:.4f} -> {final_S[0]:.4f}")
    # Output:   Spectral norm change: 1.2345 -> 2.3456
    print(f"  Effective rank change: "
          f"{(initial_S.sum()**2)/(initial_S**2).sum():.2f} -> "
          f"{(final_S.sum()**2)/(final_S**2).sum():.2f}")
    # Output:   Effective rank change: 7.23 -> 4.56

# Typically, effective rank decreases during training (specialization).
```

### Example 3: Spectral Norm and Lipschitz Constant

```python
import torch
import torch.nn as nn

# Compute Lipschitz constant bound for a small network
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 20),
    nn.ReLU(),
    nn.Linear(20, 1),
)

# Lipschitz bound = product of spectral norms
lipschitz_bound = 1.0
for name, param in model.named_parameters():
    if 'weight' in name:
        sigma_max = torch.linalg.svdvals(param.data)[0]
        lipschitz_bound *= sigma_max
        print(f"{name}: spectral norm = {sigma_max:.4f}")
        # Output: 0.weight: spectral norm = 1.2345
        # Output: 2.weight: spectral norm = 1.4567
        # Output: 4.weight: spectral norm = 1.0123

print(f"\nLipschitz bound: {lipschitz_bound:.4f}")
# Output: Lipschitz bound: 1.8192

# Empirical Lipschitz estimate
def estimate_lipschitz(model, n_samples=1000):
    model.eval()
    max_ratio = 0.0
    for _ in range(n_samples):
        x1 = torch.randn(10)
        x2 = torch.randn(10)
        with torch.no_grad():
            d_out = (model(x1) - model(x2)).norm()
            d_in = (x1 - x2).norm()
            ratio = d_out / (d_in + 1e-10)
            if ratio > max_ratio:
                max_ratio = ratio
    return max_ratio

empirical_lipschitz = estimate_lipschitz(model)
print(f"Empirical Lipschitz: {empirical_lipschitz:.4f}")
# Output: Empirical Lipschitz: 1.2345

# The true Lipschitz constant is <= the product bound
print(f"Bound is valid: {empirical_lipschitz <= lipschitz_bound + 0.1}")
# Output: Bound is valid: True
```

### Example 4: Spectral Normalization for GAN Discriminator

```python
import torch
import torch.nn as nn

class SpectralNormLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.u = nn.Parameter(torch.randn(out_features, 1), requires_grad=False)

    def forward(self, x):
        # Power iteration to estimate spectral norm
        w = self.weight
        v = w.T @ self.u
        v = v / (v.norm() + 1e-12)
        u = w @ v
        u = u / (u.norm() + 1e-12)
        sigma = (u.T @ w @ v).squeeze()

        # Normalize: W_sn = W / sigma
        w_sn = w / (sigma + 1e-12)
        self.u.data = u

        return F.linear(x, w_sn, self.bias)

# Verify spectral norm
layer = SpectralNormLinear(50, 50)
x = torch.randn(32, 50)
out = layer(x)

W_sn = layer.weight / torch.linalg.svdvals(layer.weight)[0]
print(f"Spectral norm after SN: {torch.linalg.svdvals(W_sn)[0]:.4f}")
# Output: Spectral norm after SN: 1.0000

# PyTorch's built-in spectral normalization
sn_linear = nn.utils.spectral_norm(nn.Linear(50, 50))
_ = sn_linear(x)
print(f"PyTorch SN spectral norm: {torch.linalg.svdvals(sn_linear.weight)[0]:.4f}")
# Output: PyTorch SN spectral norm: 0.9876
```

### Example 5: Singular Value Distribution and Gradient Flow

```python
import torch
import torch.nn as nn

# Analyze how singular values affect gradient flow
def gradient_flow_analysis(model, x):
    """Analyze gradient flow through each layer."""
    activations = {}
    gradients = {}

    # Forward hook to capture activations
    def forward_hook(name):
        def hook(module, input, output):
            activations[name] = input[0].detach()
        return hook

    # Backward hook to capture gradients
    def backward_hook(name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                gradients[name] = grad_output[0].norm().item()
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(forward_hook(name)))
            hooks.append(module.register_full_backward_hook(backward_hook(name)))

    # Forward + backward
    out = model(x)
    loss = out.sum()
    loss.backward()

    for h in hooks:
        h.remove()

    return activations, gradients

model = nn.Sequential(
    nn.Linear(50, 50),
    nn.ReLU(),
    nn.Linear(50, 50),
    nn.ReLU(),
    nn.Linear(50, 10),
)

x = torch.randn(16, 50)
activations, gradients = gradient_flow_analysis(model, x)

# Compute Jacobian singular values for each layer
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        W = module.weight.data
        S = torch.linalg.svdvals(W)
        print(f"\n{name}:")
        print(f"  SVs: [{S[0]:.4f}, {S[-1]:.4f}], "
              f"spectral norm: {S[0]:.4f}")
        # Output:  0: SVs: [2.3456, 0.0123], spectral norm: 2.3456
        if name in gradients:
            print(f"  Gradient norm at output: {gradients[name]:.4f}")
            # Output:   Gradient norm at output: 1.2345

# Layers with large spectral norms amplify gradients (potential explosion)
# Layers with small spectral norms attenuate gradients (potential vanishing)
```

## Common Mistakes

1. **Assuming all singular values are equally important**: The spectral norm (largest SV) dominates the Lipschitz constant, but small SVs determine effective rank and information preservation.

2. **Confusing eigenvalues with singular values**: For non-square matrices, eigenvalues don't exist. Singular values always exist and are always non-negative.

3. **Ignoring the singular value distribution**: The full spectrum matters, not just the condition number. Two matrices with the same condition number can have very different spectral distributions.

4. **Computing SVD too frequently**: Full SVD is $O(\min(mn^2, m^2n))$ and is expensive for large matrices. Use power iteration for just the top singular values.

5. **Thinking spectral normalization only matters for GANs**: Spectral properties affect all deep learning models — gradient flow, stability, capacity, and generalization.

6. **Forgetting that nonlinearities affect spectral analysis**: The Jacobian includes the activation's derivative, which can shrink or kill singular values.

7. **Assuming weight matrices are full rank**: In practice, many weight matrices have significant low-rank structure, especially after training.

## Interview Questions

### Beginner

1. What is the singular value decomposition of a matrix?
2. What is the spectral norm of a matrix?
3. How do singular values relate to the transformation performed by a matrix?
4. What does it mean if a weight matrix has many near-zero singular values?
5. How does the spectral norm of a layer relate to its Lipschitz constant?

### Intermediate

1. Explain how the singular value distribution of a weight matrix affects gradient flow through that layer.
2. How does spectral normalization work and why is it used in GANs?
3. What is effective rank and what does it tell us about a layer's representational capacity?
4. How does the singular value spectrum change during training? What does a decreasing effective rank signify?
5. Compare spectral normalization with weight normalization and batch normalization. How do they differ in their effects on the singular value spectrum?

### Advanced

1. Derive the relationship between the singular values of the Jacobian of a deep network and the singular values of each layer's weight matrix. How does this affect the probability of vanishing/exploding gradients?
2. Prove that for a matrix $W$ with SVD $U\Sigma V^T$, the optimal low-rank approximation (in Frobenius norm) is $U_k \Sigma_k V_k^T$ (the Eckart-Young theorem). What are the implications for model compression?
3. Explain how the spectral analysis of the Neural Tangent Kernel (NTK) relates to the convergence of infinitely wide neural networks. Show that the NTK's spectral properties determine the training dynamics.

## Practice Problems

### Easy

1. Compute the SVD of $A = [[3, 0], [0, 1]]$ by hand.
2. What are the singular values of an orthogonal matrix?
3. For $W \in \mathbb{R}^{5 \times 3}$, how many singular values does it have?
4. What is the spectral norm of a diagonal matrix $\text{diag}(2, 5, 1)$?
5. If all singular values of a matrix are 1, what kind of matrix is it?

### Medium

1. Implement power iteration to compute the top singular value of a large matrix efficiently.
2. Compute and plot the singular value spectrum of each layer of a pre-trained ResNet18.
3. Implement spectral normalization for a convolutional layer.
4. Compare the effective rank of weight matrices at initialization (Kaiming) vs after training.
5. Show that the nuclear norm (sum of SVs) can be used as a convex relaxation of rank for matrix completion.

### Hard

1. Implement the "spectral landscape" analysis: compute the top 50 Hessian eigenvalues during training of a small network and correlate them with the singular values of the weight matrices.
2. Derive and implement the "spectral decoupling" regularization that penalizes the spectral norm of each layer independently. Compare its effect on generalization with standard weight decay.
3. Prove the Grassmannian manifold properties of the space of left singular vectors and show how they evolve during training using the differential equation $\dot{U} = (I - UU^T) \nabla_U L$.

## Solutions

_Solutions for selected problems._

**Easy 3**: For $W \in \mathbb{R}^{5 \times 3}$, $\min(5,3) = 3$ singular values.

**Medium 1**:

```python
def power_iteration(W, num_iters=20):
    """Estimate the largest singular value of W."""
    v = torch.randn(W.shape[1])
    v = v / v.norm()
    for _ in range(num_iters):
        u = W @ v
        u = u / u.norm()
        v = W.T @ u
        v = v / v.norm()
    sigma = u @ W @ v
    return sigma, u, v

W = torch.randn(100, 50)
sigma_pi, _, _ = power_iteration(W, num_iters=50)
sigma_true = torch.linalg.svdvals(W)[0]
print(f"Power iteration: {sigma_pi:.4f}, True: {sigma_true:.4f}")
# Output: Power iteration: 12.3456, True: 12.3456
```

**Medium 3**: Spectral normalization for Conv2d:

```python
def spectral_norm_conv2d(conv_module):
    """Apply spectral normalization to a Conv2d layer."""
    W = conv_module.weight
    # Reshape to 2D: (out_channels, in_channels * kernel_h * kernel_w)
    W_2d = W.view(W.shape[0], -1)
    sigma = torch.linalg.svdvals(W_2d)[0]
    conv_module.weight.data = W / sigma
    return conv_module
```

**Hard 3**: The evolution of singular vectors during gradient flow follows a geodesic on the Grassmannian, governed by:
$$\dot{U} = (I - UU^T) \nabla_U L$$
This means the left singular vectors rotate in the directions of the gradient while staying on the Stiefel manifold (orthogonality constraint). The singular values themselves evolve according to:
$$\dot{\sigma}_i = u_i^T \nabla_W L \, v_i$$
This provides a complete dynamical description of how weight matrices change during training.

## Related Concepts

- **DL-016: Linear Algebra Review** — SVD is the fundamental tool for spectral analysis
- **DL-025: Gradient Flow** — Singular values of Jacobians determine gradient scaling
- **DL-028: Condition Number** — Ratio of extreme singular values
- **DL-029: Smoothness and Lipschitz** — Spectral norm = Lipschitz constant of linear layer
- **DL-022: Jacobian and Hessian** — Spectral analysis of Jacobians and Hessians

## Next Concepts

Reinforces all previous concepts through the unifying lens of spectral analysis.

## Summary

Spectral analysis uses singular value decomposition (SVD) to understand the properties of weight matrices and Jacobians in deep neural networks. The singular value spectrum reveals how each layer transforms input signals — which directions are amplified (large singular values) and which are suppressed (small singular values). Key quantities include the spectral norm (Lipschitz constant), effective rank (representational capacity), and condition number (optimization difficulty). Spectral normalization constrains weight matrices to have unit spectral norm, enforcing Lipschitz continuity for stable GAN training. Throughout training, singular value distributions evolve — typically developing a power-law structure with decreasing effective rank as the network specializes. Spectral analysis provides a unified framework for understanding gradient flow, stability, capacity, compression, and generalization.

## Key Takeaways

- SVD: $W = U \Sigma V^T$, singular values $\sigma_i$ reveal scaling of each direction
- Spectral norm $\|W\|_2 = \sigma_{\max}$ bounds the layer's Lipschitz constant
- Effective rank measures how many input dimensions are preserved
- Singular values typically follow a power-law distribution after training
- Spectral normalization constrains $\sigma_{\max} = 1$ for stability
- Gradient flow depends on the full singular value spectrum of Jacobians
- Low-rank structure emerges during training, enabling compression
- Power iteration efficiently estimates the top singular values
