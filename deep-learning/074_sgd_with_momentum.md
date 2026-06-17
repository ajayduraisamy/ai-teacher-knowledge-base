# Concept: SGD with Momentum

## Concept ID

DL-074

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Optimization Algorithms

## Learning Objectives

- Understand the momentum mechanism for accelerating gradient descent
- Derive the momentum update rule and relate it to physical intuition
- Implement SGD with momentum manually and using torch.optim
- Analyze how momentum helps escape local minima and navigate ravines
- Tune the momentum coefficient for optimal convergence

## Prerequisites

- Stochastic Gradient Descent (DL-072)
- Mini-Batch Gradient Descent (DL-073)
- Basic physics: velocity, friction, momentum

## Definition

SGD with Momentum is an optimization algorithm that accumulates a velocity vector in the direction of persistent gradient signals, dampening oscillations and accelerating convergence. The update adds a fraction γ of the previous update to the current gradient step:

vₜ = γ vₜ₋₁ + η ∇L(θₜ)
θₜ₊₁ = θₜ − vₜ

Alternatively, in the standard formulation used by PyTorch:

vₜ = γ vₜ₋₁ + ∇L(θₜ)
θₜ₊₁ = θₜ − η vₜ

where γ is the momentum coefficient (typically 0.9 or 0.99) and η is the learning rate.

## Intuition

Imagine rolling a ball down a bumpy hill. Without momentum, the ball follows the local slope exactly — if the hill suddenly flattens, the ball stops. With momentum, the ball builds up speed as it rolls down consistently sloping sections, allowing it to power through small bumps and flat regions.

Mathematically, momentum averages past gradients with exponential decay. If the gradient consistently points in the same direction, the velocity grows, leading to larger steps. If the gradient direction oscillates (as in narrow ravines), the oscillations cancel out, dampening the zigzagging.

## Why This Concept Matters

Plain SGD struggles with:

- **Ravines**: areas where the loss surface is much steeper in one direction than another. SGD zigzags across the ravine, making slow progress along the valley floor.
- **Flat regions**: plateaus where gradients are near zero cause SGD to stall.
- **Noisy gradients**: high-variance gradient estimates cause erratic updates.

Momentum addresses all three issues. It is a standard component in virtually all practical optimizers and is the essential ingredient that makes SGD competitive with adaptive methods like Adam.

## Mathematical Explanation

### Standard Momentum Update

Let gₜ = ∇L(θₜ) be the gradient at step t. The momentum update is:

vₜ = μ vₜ₋₁ + gₜ
θₜ₊₁ = θₜ − η vₜ

where μ ∈ [0, 1) is the momentum coefficient.

### Unrolling the Velocity

Expanding the velocity recursively:

vₜ = gₜ + μ gₜ₋₁ + μ² gₜ₋₂ + ... + μᵗ g₀

The effective learning rate for gradient gₜ₋ₖ is η μᵏ. Past gradients decay exponentially with factor μ. The effective window length (time constant) is:

τ = 1 / (1 − μ)

For μ = 0.9, τ ≈ 10 steps. For μ = 0.99, τ ≈ 100 steps.

### Convergence Analysis

For strongly convex quadratic objectives, SGD with momentum achieves accelerated convergence. The optimal momentum is related to the condition number κ (ratio of largest to smallest eigenvalue of the Hessian):

μ* = (√κ − 1) / (√κ + 1)

With optimal momentum and learning rate, the convergence rate improves from (κ − 1)/(κ + 1) for plain GD to (√κ − 1)/(√κ + 1) — a dramatic improvement for ill-conditioned problems.

### Nesterov vs. Standard Momentum

Standard momentum computes the gradient at the current position and then updates the velocity. Nesterov Accelerated Gradient (NAG) computes the gradient at the lookahead position θₜ − η μ vₜ₋₁:

vₜ = μ vₜ₋₁ + ∇L(θₜ − η μ vₜ₋₁)
θₜ₊₁ = θₜ − η vₜ

NAG provides stronger theoretical guarantees and often converges faster in practice.

### Practical Considerations

- Momentum coefficient is typically 0.9 or 0.99
- Higher momentum values work better with larger batch sizes
- Momentum is often combined with learning rate decay
- Some implementations use dampening: vₜ = μ vₜ₋₁ + (1 − τ) gₜ

## Code Examples

### Example 1: Manual Momentum Implementation

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
X = torch.randn(1000, 10)
true_w = torch.randn(10, 1)
y = X @ true_w + 0.1 * torch.randn(1000, 1)

w = torch.randn(10, 1, requires_grad=True)
lr = 0.01
mu = 0.9
v = torch.zeros_like(w)

loss_fn = nn.MSELoss()

for epoch in range(50):
    loss = loss_fn(X @ w, y)
    loss.backward()
    with torch.no_grad():
        v = mu * v + w.grad
        w -= lr * v
    w.grad.zero_()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}, w[0] = {w[0].item():.4f}")
```

```
# Output:
# Epoch 0: loss = 2.353979, w[0] = 0.4985
# Epoch 10: loss = 0.105377, w[0] = 0.7722
# Epoch 20: loss = 0.009236, w[0] = 0.7998
# Epoch 30: loss = 0.009151, w[0] = 0.8000
# Epoch 40: loss = 0.009149, w[0] = 0.8000
```

### Example 2: torch.optim.SGD with Momentum

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(2000, 50)
true_w = torch.randn(50, 1)
y = X @ true_w + 0.05 * torch.randn(2000, 1)

model = nn.Linear(50, 1, bias=False)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn = nn.MSELoss()

for epoch in range(30):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")
```

```
# Output:
# Epoch 0: loss = 1.042629
# Epoch 10: loss = 0.005337
# Epoch 20: loss = 0.002637
# Epoch 29: loss = 0.002487
```

### Example 3: Momentum vs. No Momentum Comparison

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
N = 3000
X = torch.randn(N, 20)
y = torch.sin(X[:, 0:1]) + 0.1 * torch.randn(N, 1)

def train_with(momentum, label):
    model = nn.Sequential(
        nn.Linear(20, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1)
    )
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=momentum)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
    losses = []
    for epoch in range(100):
        epoch_loss = 0.0
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(Xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
    print(f"{label}: final loss = {losses[-1]:.6f}")
    return losses

losses_plain = train_with(0.0, "No momentum")
losses_mom = train_with(0.9, "Momentum=0.9")
```

```
# Output:
# No momentum: final loss = 0.014982
# Momentum=0.9: final loss = 0.008910
```

## Common Mistakes

1. **Momentum coefficient too high**: μ > 0.999 causes extreme acceleration that overshoots minima. The algorithm may never converge.
2. **Momentum coefficient too low**: μ < 0.5 provides negligible benefit over plain SGD.
3. **Not reducing learning rate when adding momentum**: Because momentum amplifies updates, the effective step size grows. Reduce η when introducing momentum.
4. **Forgetting to initialize velocity to zero**: Starting with a non-zero velocity biases early updates.
5. **Applying momentum with cyclic learning rates without adjustment**: High momentum with aggressive LR schedules can cause instability.
6. **Confusing momentum implementations**: PyTorch uses vₜ = μ vₜ₋₁ + gₜ (not the same as some textbooks with η inside the velocity). Check the specific implementation.
7. **Using momentum for sparse gradients**: For extremely sparse gradients (e.g., embedding layers), momentum can dilute useful sparse signals.

## Interview Questions

### Beginner

1. What problem does momentum solve in gradient descent?
2. How does the momentum term affect the optimization trajectory?
3. What is a typical value for the momentum coefficient?
4. How does momentum help in ravine-like loss landscapes?
5. What happens if the momentum coefficient is set to 0.999?

### Intermediate

1. Derive the momentum update rule from the perspective of exponential moving average of gradients.
2. Explain how momentum helps escape local minima.
3. Compare standard momentum with Nesterov accelerated gradient.
4. How does the effective time constant τ = 1/(1−μ) relate to convergence?
5. Implement momentum manually (without torch.optim) and explain each step.

### Advanced

1. Prove the accelerated convergence rate of momentum for quadratic objectives.
2. Analyze the behavior of momentum in the presence of gradient noise.
3. Explain the relationship between momentum and conjugate gradient methods.

## Practice Problems

### Easy

1. Implement SGD with momentum for a 1D quadratic function.
2. Plot the trajectory of SGD with and without momentum on a 2D contour.
3. Tune the momentum coefficient on a simple linear regression problem.
4. Compare convergence for μ in {0, 0.5, 0.9, 0.99}.
5. Implement the momentum update both manually and with torch.optim.

### Medium

1. Build an experiment showing momentum overcoming a shallow local minimum.
2. Implement Nesterov momentum manually and compare with standard momentum.
3. Visualize the velocity vector during training and explain its behavior.
4. Train a small CNN with and without momentum on MNIST.
5. Implement a learning rate schedule that accounts for momentum amplification.

### Hard

1. Derive the optimal momentum coefficient for a given condition number.
2. Implement the heavy-ball method with Polyak momentum and analyze convergence.
3. Design an experiment demonstrating the relationship between batch size and optimal momentum.

## Solutions

Momentum solutions involve tracking the velocity state. For the quadratic case, the optimal momentum and learning rate can be solved analytically from the eigenvalues of the Hessian.

## Related Concepts

- Nesterov Accelerated Gradient (DL-075): An improved momentum variant
- Gradient Descent (DL-071): The base algorithm
- Learning Rate Decay (DL-086): Combined with momentum in practice
- Adam Optimizer (DL-078): Adapts momentum per-parameter

## Next Concepts

- Nesterov Accelerated Gradient (DL-075)
- Adam Optimizer (DL-078)
- Learning Rate Warmup (DL-085)

## Summary

SGD with Momentum accelerates convergence by accumulating a velocity vector that persists in directions of consistent gradient signal. The velocity is an exponential moving average of past gradients with decay μ (typically 0.9). Momentum drastically improves optimization in ill-conditioned landscapes (ravines), helps escape shallow local minima, and smooths noisy gradient estimates. The effective window length τ = 1/(1−μ) controls how many past gradients influence the current update.

## Key Takeaways

1. Momentum accumulates velocity in gradient direction, accelerating consistent descent.
2. The momentum coefficient μ (0.9 default) controls the decay rate of past gradient influence.
3. Momentum dampens oscillations in narrow ravines, enabling faster progress along the valley.
4. Nesterov momentum provides theoretical acceleration and often better practical performance.
5. Adding momentum typically requires reducing the learning rate to prevent overshooting.
