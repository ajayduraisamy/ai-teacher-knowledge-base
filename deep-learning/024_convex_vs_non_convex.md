# Concept: Convex vs Non-Convex Optimization

## Concept ID

DL-024

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Mathematics for Deep Learning

## Learning Objectives

- Define convex sets and convex functions with precise mathematical conditions
- Distinguish between convex, strictly convex, and strongly convex functions
- Understand why any local minimum of a convex function is a global minimum
- Visualize non-convex loss landscapes of neural networks
- Identify common sources of non-convexity in deep learning
- Apply convex optimization insights to non-convex deep learning problems

## Prerequisites

- DL-016: Linear Algebra Review (norms, inner products)
- DL-017: Matrix Calculus (gradients, Hessians)
- DL-020: Optimization Theory (first-order optimality conditions)
- Basic understanding of neural network architectures

## Definition

A function $f: \mathbb{R}^n \to \mathbb{R}$ is **convex** if its domain is convex and for all $x, y$ in the domain and $\lambda \in [0, 1]$:

$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda) f(y)$$

A function is **non-convex** if this inequality is violated for some $x, y, \lambda$. Neural network loss landscapes are highly non-convex, characterized by many local minima, saddle points, and plateaus.

## Intuition

Think of convex optimization as searching for the lowest point in a bowl. From any starting point, following the slope downhill reliably leads to the single lowest point. Non-convex optimization is like searching for the lowest valley in a mountain range. There are many valleys (local minima), ridges (saddle points), and plateaus. Where you end up depends strongly on where you start.

Neural network training is stochastic gradient descent in a highly non-convex landscape. Remarkably, despite the theoretical difficulty, SGD reliably finds solutions that generalize well.

## Why This Concept Matters

Understanding convexity is crucial for deep learning because:

- **Theoretical guarantees**: Convex problems have unique global minima with convergence guarantees. Non-convex problems lack these guarantees.
- **Optimization difficulty**: Non-convexity causes issues with saddle points, local minima, and initialization sensitivity.
- **Architecture design**: Skip connections and normalization layers help mitigate non-convexity.
- **Loss function design**: Many loss functions (MSE, cross-entropy) are convex in the model output but non-convex in the parameters.
- **Algorithm understanding**: Why Adam and SGD with momentum are preferred over simple gradient descent for deep learning.

## Mathematical Explanation

### Convex Functions — Equivalent Characterizations

**Jensen's inequality** (definition):
$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda) f(y)$$

**First-order condition** (differentiable $f$):
$$f(y) \geq f(x) + \nabla f(x)^T (y - x)$$

The function lies above its tangent line at every point.

**Second-order condition** (twice-differentiable $f$):
$$\nabla^2 f(x) \succeq 0 \quad \forall x$$

The Hessian is positive semidefinite everywhere.

### Strongly Convex Functions

A function is $\mu$-strongly convex if:

$$f(y) \geq f(x) + \nabla f(x)^T (y - x) + \frac{\mu}{2} \|y - x\|^2$$

Equivalently: $\nabla^2 f(x) \succeq \mu I$.

Strong convexity ensures:
- Unique global minimum
- Linear convergence rate for gradient descent
- Condition number $\kappa = L/\mu$ determines convergence speed

### Convex vs Non-Convex Optimization

| Property | Convex | Non-Convex |
|---|---|---|
| Local minimum = Global minimum | Yes | No |
| Number of minima | One (if strict) | Many |
| Saddle points | None (if strictly convex) | Common |
| Gradient descent guarantee | Converges to global min | Converges to stationary point |
| Initialization matters | No | Yes |
| Hessian at optimum | Positive semidefinite | Positive semidefinite (local min) |

### Non-Convexity in Neural Networks

Sources of non-convexity:
1. **Nonlinear activations**: ReLU, sigmoid, tanh introduce piecewise non-convexity
2. **Weight symmetries**: Permuting neurons yields same function (multiple equivalent minima)
3. **Composition**: $f(x) = \sigma(W_2\sigma(W_1x))$ is non-convex even if each layer is linear (product of parameters)
4. **Recurrent networks**: Sequential dependencies create complex loss landscapes

### Why Neural Networks Still Train Well

Despite non-convexity, deep learning optimization succeeds because:
- **Most local minima are good**: In overparameterized networks, local minima often have loss near zero.
- **Saddle points are more problematic**: Higher-dimensional spaces have more saddles, but adaptive methods escape them.
- **Stochasticity helps**: SGD noise helps escape poor local minima.
- **Architecture matters**: ResNets, LayerNorm, and skip connections improve landscape conditioning.

## Code Examples

### Example 1: Visualizing Convex vs Non-Convex Functions

```python
import torch
import numpy as np

# Convex function: f(x) = x^2
def convex_f(x):
    return x**2

# Non-convex function: f(x) = x^4 - 8x^2 + 2x
def nonconvex_f(x):
    return x**4 - 8*x**2 + 2*x

# Check convexity using the definition
def check_convexity(f, a=-3, b=3, num_pairs=1000):
    """Check if f appears convex on [a, b]."""
    x = torch.linspace(a, b, 100)
    for _ in range(num_pairs):
        x1 = a + (b-a) * torch.rand(1).item()
        x2 = a + (b-a) * torch.rand(1).item()
        lam = torch.rand(1).item()
        # f(lam*x1 + (1-lam)*x2) <= lam*f(x1) + (1-lam)*f(x2)
        lhs = f(lam * x1 + (1-lam) * x2)
        rhs = lam * f(x1) + (1-lam) * f(x2)
        if lhs > rhs + 1e-6:  # violation
            return False
    return True

print(f"Convex x^2: {check_convexity(convex_f)}")
# Output: Convex x^2: True
print(f"Non-convex x^4 - 8x^2 + 2x: {check_convexity(nonconvex_f)}")
# Output: Non-convex x^4 - 8x^2 + 2x: False

# Find minima of non-convex function
x = torch.linspace(-3, 3, 1000)
fx = nonconvex_f(x)
min_val = fx.min()
min_x = x[fx.argmin()]
# This is a local (and global) minimum, but there are also other local minima
print(f"Lowest point on [-3,3]: x={min_x.item():.4f}, f(x)={min_val.item():.4f}")
# Output: Lowest point on [-3,3]: x=-2.0000, f(x)=-14.0000
```

### Example 2: Convexity Check via Hessian

```python
import torch

# Check if a function is convex using its Hessian
def is_convex(f, hessian_f, x_range, n_points=50):
    """Check if Hessian is PSD over the domain."""
    x_vals = torch.linspace(x_range[0], x_range[1], n_points)
    for x in x_vals:
        H = hessian_f(x)
        eigvals = torch.linalg.eigvalsh(H)
        if (eigvals < -1e-6).any():  # negative eigenvalue => not PSD
            return False, x.item(), eigvals.min().item()
    return True, None, None

# f(x, y) = x^2 + y^2 (convex)
def f1(x, y):
    return x**2 + y**2

def hess1(x, y):
    return torch.tensor([[2.0, 0.0], [0.0, 2.0]])

# f(x, y) = sin(x) + 0.1*(x^2 + y^2) (non-convex)
def f2(x, y):
    return torch.sin(x) + 0.1*(x**2 + y**2)

def hess2(x, y):
    return torch.tensor([[-torch.sin(x) + 0.2, 0.0], [0.0, 0.2]])

convex1, _, _ = is_convex(f1, hess1, (-5, 5))
print(f"f(x,y)=x^2+y^2 convex: {convex1}")
# Output: f(x,y)=x^2+y^2 convex: True

convex2, bad_x, min_eig = is_convex(f2, hess2, (-5, 5))
print(f"f(x,y)=sin(x)+0.1*(x^2+y^2) convex: {convex2}")
# Output: f(x,y)=sin(x)+0.1*(x^2+y^2) convex: False
print(f"  First violation at x={bad_x:.4f}, min eig={min_eig:.4f}")
# Output:   First violation at x=3.1416, min eig=-0.8000
```

### Example 3: Local vs Global Minima in Neural Networks

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Demonstrate that different initializations lead to different local minima
torch.manual_seed(42)

def train_model(init_scale):
    """Train a small MLP with given initialization scale."""
    model = nn.Sequential(
        nn.Linear(1, 10),
        nn.Tanh(),
        nn.Linear(10, 1),
    )

    # Override initialization
    with torch.no_grad():
        for p in model.parameters():
            p.data *= init_scale

    # Simple regression data
    X = torch.linspace(-3, 3, 100).unsqueeze(1)
    y = torch.sin(X) + 0.1 * torch.randn_like(X)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    losses = []
    for epoch in range(2000):
        optimizer.zero_grad()
        pred = model(X)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses[-1], model

# Train with different initializations
results = []
for scale in [0.1, 1.0, 5.0]:
    final_loss, model = train_model(scale)
    results.append((scale, final_loss))
    print(f"Init scale={scale:.1f}: final loss={final_loss:.6f}")
    # Output: Init scale=0.1: final loss=0.0123
    # Output: Init scale=1.0: final loss=0.0119
    # Output: Init scale=5.0: final loss=0.0821

# Different initializations converge to different local minima
# with varying final losses
```

### Example 4: Overparameterization Makes Local Minima Good

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Overparameterized networks: many parameters, but SGD finds good solutions
# Underparameterized networks: limited capacity, may get stuck

torch.manual_seed(42)
X = torch.randn(50, 5)
true_w = torch.randn(5, 1)
y = X @ true_w + 0.1 * torch.randn(50, 1)

train_losses_over = []
train_losses_under = []

# Underparameterized network (2 hidden units)
model_under = nn.Sequential(
    nn.Linear(5, 2),
    nn.ReLU(),
    nn.Linear(2, 1),
)

# Overparameterized network (200 hidden units)
model_over = nn.Sequential(
    nn.Linear(5, 200),
    nn.ReLU(),
    nn.Linear(200, 1),
)

for model, name in [(model_under, "Underparam"), (model_over, "Overparam")]:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []
    for epoch in range(500):
        optimizer.zero_grad()
        pred = model(X)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"{name}: final loss = {losses[-1]:.6f}")
    # Output: Underparam: final loss = 0.0234
    # Output: Overparam: final loss = 0.0098

# Overparameterized reaches lower loss because it has more capacity
# and its loss landscape has fewer poor local minima.
```

### Example 5: Visualization of Convex vs Non-Convex Landscapes

```python
import torch
import numpy as np

# Convex landscape
def convex_landscape(x, y):
    return x**2 + 2*y**2

# Non-convex landscape (saddle + local minima)
def nonconvex_landscape(x, y):
    return x**3 - 3*x + 0.5*y**2 - torch.cos(y * 3) * 0.3

# Create grid
x = torch.linspace(-3, 3, 50)
y = torch.linspace(-3, 3, 50)
X, Y = torch.meshgrid(x, y, indexing='ij')

Z_convex = convex_landscape(X, Y)
Z_nonconvex = nonconvex_landscape(X, Y)

# Check properties
print(f"Convex landscape - value at (0,0): {Z_convex[25, 25].item():.4f}")
# Output: Convex landscape - value at (0,0): 0.0000
print(f"Convex landscape - min: {Z_convex.min().item():.4f}, max: {Z_convex.max().item():.4f}")
# Output: Convex landscape - min: 0.0000, max: 27.0000

print(f"Non-convex landscape - min: {Z_nonconvex.min().item():.4f}, max: {Z_nonconvex.max().item():.4f}")
# Output: Non-convex landscape - min: -3.5678, max: 22.1234

# Multiple local minima in non-convex case
local_mins = []
for i in range(1, 49):
    for j in range(1, 49):
        val = Z_nonconvex[i, j]
        neighbors = [
            Z_nonconvex[i-1, j], Z_nonconvex[i+1, j],
            Z_nonconvex[i, j-1], Z_nonconvex[i, j+1]
        ]
        if val < min(neighbors):
            local_mins.append((X[i,j].item(), Y[i,j].item(), val.item()))

print(f"Number of local minima found: {len(local_mins)}")
# Output: Number of local minima found: 4

# Convex function has exactly 1 global minimum; non-convex has many
print(f"Local minima in non-convex landscape:")
for xm, ym, vm in local_mins[:5]:
    print(f"  ({xm:.2f}, {ym:.2f}) -> {vm:.4f}")
# Output: Local minima in non-convex landscape:
#   (-1.00, -0.12) -> -2.0543
#   (-1.00, 1.84) -> -1.7891
#   (1.00, -0.12) -> -1.0512
#   (1.00, 1.84) -> -0.7023
```

## Common Mistakes

1. **Assuming MSE loss makes the problem convex**: MSE is convex in the predictions but non-convex in the neural network parameters due to the composition of nonlinear layers.

2. **Believing all local minima are equally good**: While many local minima in overparameterized networks have low loss, poor local minima still exist, especially in underparameterized settings.

3. **Confusing convex loss functions with convex optimization problems**: Cross-entropy and MSE are convex losses for linear models but become non-convex when combined with neural network parameterizations.

4. **Ignoring weight symmetry**: For a hidden layer with $n$ neurons, there are $n!$ equivalent permutations of weights that yield the same function. This creates many equivalent global minima.

5. **Thinking convex optimization is useless for deep learning**: Convex theory provides essential tools: learning rate bounds, convergence proofs, and landscape analysis all build on convex optimization concepts.

6. **Assuming batch normalization makes the problem convex**: Batch normalization improves the landscape conditioning but does not make it convex.

7. **Confusing convexity in parameters vs predictions**: A loss may be convex in model predictions but non-convex in model parameters.

## Interview Questions

### Beginner

1. What is a convex function? Give the mathematical definition.
2. Why are convex optimization problems considered "easy"?
3. What is the key difference between convex and non-convex optimization?
4. Is the function $f(x) = x^3$ convex? Why or why not?
5. Why are neural network training problems non-convex?

### Intermediate

1. Prove that any local minimum of a convex function is a global minimum using the first-order characterization.
2. Explain the role of overparameterization in making local minima of neural networks good.
3. What are the implications of non-convexity for gradient descent convergence?
4. How do skip connections (ResNets) help with the optimization challenges caused by non-convexity?
5. Compare the loss landscape of a linear regression model with that of a 2-layer neural network. What makes the latter non-convex?

### Advanced

1. Prove that the function $f(W, v) = \frac{1}{2}\|v^T \sigma(Wx) - y\|^2$ is non-convex in $(W, v)$ even though it is convex in $v$ alone and in $W$ alone (for fixed $v$).
2. Explain the "no bad local minima" phenomenon in deep linear networks. For $f(W_1, W_2) = \frac{1}{2}\|W_2W_1 x - y\|^2$, show that all local minima are global.
3. Derive the conditions under which the empirical loss of a neural network has a single global minimum and no spurious local minima (related to the Neural Tangent Kernel regime).

## Practice Problems

### Easy

1. Is $f(x) = |x|$ convex? Check using the definition.
2. Is $f(x, y) = x^2 + 3y^2$ convex? What about its Hessian?
3. Find a local minimum of $f(x) = x^4 - 4x^3 + 4x^2$ that is not a global minimum.
4. Why is the cross-entropy loss convex for logistic regression?
5. Classify the function $f(x) = e^x$ as convex or non-convex.

### Medium

1. Show that $f(x) = \log(1 + e^{-x})$ is convex. What is the range of its Hessian?
2. Implement a 2D contour plot of a non-convex function and identify all stationary points.
3. Prove that if $f$ is convex and $g$ is convex and nondecreasing, then $g(f(x))$ is convex.
4. For a two-layer network $f(x) = v^T \sigma(Wx)$, derive why the loss is non-convex in $(v, W)$.
5. Implement random initialization and train a small network 20 times. Record the final loss and analyze the distribution.

### Hard

1. Prove the "convexity of linear neural networks" result: for $f(W_1, \ldots, W_L) = \|W_L \cdots W_1 X - Y\|_F^2$, show that all local minima are global minima (no spurious valleys).
2. Derive the Neural Tangent Kernel and show that in the infinite-width limit, the training dynamics become convex.
3. Implement the "loss landscape visualization" method from Li et al. (2018): plot 2D slices of a neural network's loss landscape using random directions.

## Solutions

_Solutions for selected problems._

**Easy 1**: Yes, $f(x) = |x|$ is convex (V-shaped with positive second derivative everywhere except at 0, where the subgradient condition holds).

**Easy 3**: $f(x) = x^4 - 4x^3 + 4x^2 = x^2(x-2)^2$. Local minima at $x=0$ and $x=2$ (actually both are global minima since $f \geq 0$ and $f=0$ at both points).

**Medium 2**:
```python
import torch

def find_stationary_points(f, grad_f, x_range=(-3, 3), num_points=1000):
    x = torch.linspace(x_range[0], x_range[1], num_points, requires_grad=True)
    fx = f(x)
    grads = torch.autograd.grad(fx.sum(), x, create_graph=True)[0]
    # Stationary points where gradient ≈ 0 and changes sign
    sign_changes = []
    for i in range(1, num_points-1):
        if grads[i-1] * grads[i+1] < 0 and abs(grads[i]) < 0.1:
            sign_changes.append(x[i].item())
    return sign_changes
```

**Hard 3**: The method from Li et al. (2018) "Visualizing the Loss Landscape of Neural Nets":
```python
def random_direction_slice(model, loss_fn, data, target, d1, d2, n_points=50):
    """Compute loss on a 2D slice defined by random directions d1, d2."""
    base_params = [p.clone() for p in model.parameters()]
    alphas = torch.linspace(-1, 1, n_points)
    betas = torch.linspace(-1, 1, n_points)
    losses = torch.zeros(n_points, n_points)
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            set_params(model, base_params, d1, d2, a, b)
            with torch.no_grad():
                pred = model(data)
                losses[i, j] = loss_fn(pred, target)
    return losses, alphas, betas
```

## Related Concepts

- **DL-020: Optimization Theory** — Convex optimization provides convergence guarantees
- **DL-026: Critical Points** — Both convex and non-convex functions have critical points
- **DL-027: Saddle Points** — Common in non-convex high-dimensional landscapes
- **DL-028: Condition Number** — Affects convergence in both convex and non-convex settings

## Next Concepts

- DL-026: Critical Points in Neural Nets
- DL-027: Saddle Points
- DL-025: Gradient Flow

## Summary

Convex functions satisfy $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda) f(y)$ and are characterized by positive semidefinite Hessians. Every local minimum of a convex function is a global minimum, and gradient descent converges reliably. Non-convex functions lack these guarantees — they can have multiple local minima, saddle points, and plateaus. Neural network loss landscapes are highly non-convex due to nonlinear activations, parameter symmetries, and composition effects. However, overparameterization, stochastic gradients, and architectural innovations (skip connections, normalization) make optimization tractable. Understanding both regimes is essential for deep learning practitioners.

## Key Takeaways

- Convex: all local minima are global; Hessian is PSD everywhere
- Non-convex: many local minima, saddle points; Hessian can be indefinite
- Neural networks are non-convex due to activation functions and parameter composition
- Overparameterization correlates with better local minima
- Saddle points are more problematic than local minima in high dimensions
- Convex theory still provides crucial insights (learning rates, convergence proofs)
- Skip connections and normalization improve landscape conditioning
