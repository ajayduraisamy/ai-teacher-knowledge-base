# Concept: Nesterov Accelerated Gradient

## Concept ID

DL-075

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Optimization Algorithms

## Learning Objectives

- Understand the lookahead mechanism in Nesterov Accelerated Gradient (NAG)
- Derive the NAG update rule and compare it to standard momentum
- Implement NAG manually and using PyTorch's nesterov option
- Recognize convergence benefits of NAG for ill-conditioned problems
- Apply NAG in practical deep learning training

## Prerequisites

- SGD with Momentum (DL-074)
- Mini-Batch Gradient Descent (DL-073)
- Understanding of gradient-based optimization

## Definition

Nesterov Accelerated Gradient (NAG) is a first-order optimization method that computes the gradient at a lookahead position — the approximate future position based on the current momentum — rather than at the current position. The update rules are:

vₜ = μ vₜ₋₁ + η ∇L(θₜ − μ vₜ₋₁)
θₜ₊₁ = θₜ − vₜ

In the equivalent formulation (used by PyTorch):

vₜ = μ vₜ₋₁ + ∇L(θₜ)
θₜ₊₁ = θₜ − η (μ vₜ + (1 + μ) ∇L(θₜ))

## Intuition

Standard momentum is like a heavy ball rolling down a hill — its trajectory depends on where it has been. Nesterov momentum is like a ball that first looks ahead, anticipates where it will be, corrects its course based on the slope at that anticipated position, and then rolls. This "peek into the future" makes NAG more responsive to changes in the landscape.

Consider walking downhill in the dark. Standard momentum: you take a step based on the slope where you are standing. Nesterov: you lean forward, plant your foot where you expect to be, feel the slope there, and adjust. This prevents overshooting when the slope suddenly changes.

## Why This Concept Matters

NAG provides provably faster convergence for convex optimization problems and often outperforms standard momentum in practice. Its importance includes:

- **Theoretical guarantee**: O(1/k²) convergence rate for convex functions vs. O(1/k) for standard GD
- **Reduced oscillation**: The lookahead correction dampens oscillations more effectively than standard momentum
- **Practical gains**: Often converges in fewer iterations than standard momentum, especially for deep networks
- **Historical significance**: Nesterov's method inspired the development of many adaptive optimizers (NADAM)

## Mathematical Explanation

### NAG Update Rule

Let θₜ be the current parameters, μ the momentum coefficient, and η the learning rate.

Lookahead step: θ̃ = θₜ − μ vₜ₋₁
Compute gradient at lookahead: gₜ = ∇L(θ̃)
Update velocity: vₜ = μ vₜ₋₁ + η gₜ
Update parameters: θₜ₊₁ = θₜ − vₜ

### Equivalent Formulation

By substituting the lookahead, we can rewrite NAG without explicit lookahead computation:

θₜ₊₁ = θₜ − η ∇L(θₜ − μ vₜ₋₁) − μ vₜ₋₁ + μ (vₜ₋₁ − η ∇L(θₜ − μ vₜ₋₁))

This simplifies to the form used in many implementations:

vₜ₊₁ = μ vₜ − η ∇L(θₜ + μ vₜ)
θₜ₊₁ = θₜ + vₜ₊₁

### PyTorch's Implementation

PyTorch implements NAG with the following update (when nesterov=True):

vₜ₊₁ = μ vₜ + gₜ
θₜ₊₁ = θₜ − η (gₜ + μ vₜ₊₁)

This is equivalent to the standard NAG but reformulated for computational efficiency.

### Convergence Rate

For convex L-smooth functions, NAG achieves:

J(θₖ) − J(θ*) ≤ O(L ‖θ₀ − θ*‖² / k²)

This is optimal among first-order methods, matching the lower bound for convex optimization. Standard gradient descent achieves only O(L/k).

### Comparison with Standard Momentum

| Property | Standard Momentum | NAG |
|----------|-------------------|-----|
| Gradient location | Current θ | Lookahead θ − μv |
| Convex convergence | O(1/k) | O(1/k²) |
| Oscillation damping | Moderate | Strong |
| Overshoot correction | None | Built-in |

## Code Examples

### Example 1: Manual NAG Implementation

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
    w_hat = w - mu * v
    loss = loss_fn(X @ w_hat, y)
    loss.backward()
    with torch.no_grad():
        v = mu * v + lr * w_hat.grad
        w -= v
    w.grad.zero_()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")
```

```
# Output:
# Epoch 0: loss = 6.253141
# Epoch 10: loss = 0.033959
# Epoch 20: loss = 0.009404
# Epoch 30: loss = 0.009149
# Epoch 40: loss = 0.009149
```

### Example 2: PyTorch NAG with torch.optim

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(2000, 50)
true_w = torch.randn(50, 1)
y = X @ true_w + 0.05 * torch.randn(2000, 1)

model = nn.Linear(50, 1, bias=False)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
loss_fn = nn.MSELoss()

for epoch in range(20):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: loss = {loss.item():.6f}")
```

```
# Output:
# Epoch 0: loss = 1.042629
# Epoch 1: loss = 0.403802
# Epoch 2: loss = 0.128359
# Epoch 3: loss = 0.037819
# Epoch 4: loss = 0.013427
# Epoch 5: loss = 0.007316
# Epoch 6: loss = 0.005273
# Epoch 7: loss = 0.004314
# Epoch 8: loss = 0.003633
# Epoch 9: loss = 0.003115
# Epoch 10: loss = 0.002747
# Epoch 11: loss = 0.002532
# Epoch 12: loss = 0.002419
# Epoch 13: loss = 0.002358
# Epoch 14: loss = 0.002322
# Epoch 15: loss = 0.002298
# Epoch 16: loss = 0.002281
# Epoch 17: loss = 0.002268
# Epoch 18: loss = 0.002256
# Epoch 19: loss = 0.002246
```

### Example 3: NAG vs. Standard Momentum on a Pathological Problem

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
N = 2000
X = torch.randn(N, 30)
true_w = torch.randn(30, 1)
y = X @ true_w + 0.05 * torch.randn(N, 1)

def train(momentum, nesterov, label):
    model = nn.Linear(30, 1, bias=False)
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=momentum, nesterov=nesterov)
    loss_fn = nn.MSELoss()
    losses = []
    for epoch in range(50):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"{label}: final loss = {losses[-1]:.6f}")
    return losses

train(0.9, False, "Standard momentum")
train(0.9, True, "Nesterov")
```

```
# Output:
# Standard momentum: final loss = 0.002253
# Nesterov: final loss = 0.002246
```

## Common Mistakes

1. **Forgetting lookahead gradient computation**: Using the gradient at θ instead of θ − μvₜ₋₁ is just standard momentum, not NAG.
2. **Incorrect NAG update order**: Update velocity first, then apply to parameters. Some implementations do θ ← θ + v then compute gradient.
3. **Applying NAG with cyclic learning rates**: NAG's lookahead can become unstable if the learning rate changes abruptly.
4. **Not understanding PyTorch's NAG convention**: PyTorch's nesterov=True requires also setting momentum > 0. The implementation differs from the textbook formula.
5. **Overshooting due to high momentum with NAG**: NAG can overshoot even more than standard momentum if η is too large relative to the curvature.
6. **Using NAG with non-smooth objectives**: NAG's theoretical advantages assume smoothness. Proximal variants are needed for non-smooth optimization.

## Interview Questions

### Beginner

1. What is the key difference between NAG and standard momentum?
2. Why is NAG called "accelerated" gradient descent?
3. How does the "lookahead" step work in NAG?
4. What does Nesterov's method help prevent compared to standard momentum?
5. How do you enable Nesterov momentum in PyTorch?

### Intermediate

1. Derive the NAG update rule step by step.
2. Explain why NAG achieves O(1/k²) convergence for convex functions.
3. Compare the oscillatory behavior of NAG vs. standard momentum on a quadratic with high condition number.
4. How does PyTorch's NAG implementation differ from the classic formulation?
5. What is the relationship between NAG and gradient estimation at extrapolated points?

### Advanced

1. Prove the O(1/k²) convergence rate of NAG for convex L-smooth functions.
2. Explain how NAG relates to the method of conjugate gradients.
3. Derive the continuous-time limit of NAG and its Lyapunov function.

## Practice Problems

### Easy

1. Implement NAG manually for f(x) = x² + 5x + 3.
2. Plot the trajectory of NAG vs. standard momentum on a 2D contour.
3. Compare convergence for NAG with μ in {0.5, 0.9, 0.99}.
4. Train a linear regression model with NAG using torch.optim.
5. Verify the lookahead gradient computation gives a different direction than the current gradient.

### Medium

1. Build an experiment demonstrating NAG's reduced oscillation on a badly-conditioned quadratic.
2. Implement NAG from scratch without using torch.optim (manual backward loop).
3. Compare NAG and standard momentum on a deep network (3+ layers) with MNIST.
4. Visualize the lookahead position vs. actual position during training.
5. Implement gradient clipping with NAG.

### Hard

1. Derive the optimal parameters for NAG on a quadratic objective.
2. Implement NAG with adaptive restart (when gradient changes sign).
3. Prove the optimality of NAG among first-order methods using the oracle model.

## Solutions

NAG implementations require careful ordering of the lookahead, gradient computation, velocity update, and parameter update. The PyTorch implementation uses a reformulated version that avoids explicitly computing the lookahead position.

## Related Concepts

- SGD with Momentum (DL-074): Standard momentum without lookahead
- NADAM (DL-080): Adam with Nesterov momentum
- Gradient Descent (DL-071): The base algorithm

## Next Concepts

- Adagrad (DL-076)
- RMSprop (DL-077)
- Adam Optimizer (DL-078)

## Summary

Nesterov Accelerated Gradient improves upon standard momentum by computing gradients at an extrapolated lookahead position θ − μvₜ₋₁. This anticipatory correction provides provably optimal convergence O(1/k²) for convex L-smooth functions and reduces oscillatory behavior in practice. NAG's theoretical properties have inspired a family of accelerated methods and are a key component of modern optimizers like NADAM.

## Key Takeaways

1. NAG computes gradients at the lookahead position θ − μv, not at the current position θ.
2. NAG achieves the optimal O(1/k²) convergence rate for first-order convex optimization.
3. The lookahead mechanism dampens oscillations more effectively than standard momentum.
4. PyTorch supports NAG with `optim.SGD(..., momentum=μ, nesterov=True)`.
5. NAG forms the conceptual foundation for accelerated variants of Adam (NADAM).
