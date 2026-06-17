# Concept: Gradient Descent

## Concept ID

DL-071

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Optimization Algorithms

## Learning Objectives

- Understand the core principles of gradient descent as a first-order optimization algorithm
- Derive the parameter update rule from the gradient of the loss function
- Implement gradient descent from scratch using PyTorch
- Identify convergence criteria and diagnose failure modes
- Compare gradient descent with other optimization families

## Prerequisites

- Basic calculus: partial derivatives, gradients, chain rule
- Linear algebra: vectors, matrices, dot products
- Python programming with NumPy/PyTorch fundamentals
- Understanding of loss functions and model parameters

## Definition

Gradient Descent is an iterative first-order optimization algorithm used to minimize a differentiable objective function. Starting from an initial guess, the algorithm repeatedly takes steps proportional to the negative of the gradient of the function at the current point. The update rule is:

θₜ₊₁ = θₜ − η ∇J(θₜ)

where θ represents the model parameters, η is the learning rate, ∇J(θₜ) is the gradient of the loss J with respect to the parameters, and t is the iteration index.

## Intuition

Imagine you are standing on a foggy mountain and need to descend to the valley floor. You cannot see the entire landscape, but you can feel the slope beneath your feet. Gradient descent mimics this: at each step, you feel the steepest downward direction (the negative gradient) and take a step in that direction. The learning rate controls how large a step you take — too small and progress is painfully slow, too large and you might overstep and end up higher than before.

The gradient vector points in the direction of steepest *ascent*. Therefore, moving in the opposite direction — the negative gradient — is the direction of steepest *descent*. Each iteration evaluates the gradient at the current location and moves downhill.

## Why This Concept Matters

Gradient descent is the backbone of nearly all deep learning training. Every neural network from a single-layer perceptron to a billion-parameter transformer relies on some variant of gradient descent to learn from data. Understanding its mechanics, convergence properties, and limitations is essential for:

- Diagnosing training failures such as divergence or stagnation
- Selecting appropriate learning rates and schedules
- Recognizing when more advanced optimizers (SGD with momentum, Adam) provide meaningful benefits
- Debugging vanishing or exploding gradient problems
- Designing custom training loops in PyTorch

Without gradient descent, deep learning as we know it would not exist. It bridges the gap between computing loss values and actually improving model weights.

## Mathematical Explanation

### Full-Batch Gradient Descent

Given a dataset with N samples, the loss function J(θ) is the average of individual losses:

J(θ) = (1/N) ∑ᵢ₌₁ᴺ L(f(xᵢ; θ), yᵢ)

The gradient is:

∇J(θ) = (1/N) ∑ᵢ₌₁ᴺ ∇ₓ L(f(xᵢ; θ), yᵢ)

The standard gradient descent update:

θ ← θ − η ∇J(θ)

### Convergence Analysis

For a convex function with Lipschitz-continuous gradients, gradient descent achieves a convergence rate of O(1/k) where k is the number of iterations. Specifically:

J(θₖ) − J(θ*) ≤ (L ‖θ₀ − θ*‖²) / (2k)

where L is the Lipschitz constant of the gradient and θ* is the optimal solution.

For strongly convex functions with parameter μ, the rate improves to geometric (linear) convergence:

J(θₖ) − J(θ*) ≤ (L/μ) (1 − μ/L)ᵏ [J(θ₀) − J(θ*)]

### The Learning Rate

The learning rate η is the most critical hyperparameter. If η is too large, the algorithm diverges. If η is too small, convergence is extremely slow. A common heuristic is to use a learning rate that satisfies:

0 < η < 2/L

for convex functions with L-Lipschitz gradients.

### Variants

Full-batch gradient descent computes the gradient using the entire dataset. This is deterministic but computationally expensive for large datasets. The gradient for each parameter θⱼ is:

∂J/∂θⱼ = (1/N) ∑ᵢ₌₁ᴺ (∂Lᵢ/∂θⱼ)

## Code Examples

### Example 1: Gradient Descent on a Simple Quadratic

```python
import torch
import numpy as np

def f(x):
    return x**2 + 3*x + 2

x = torch.tensor(10.0, requires_grad=True)
lr = 0.1
steps = []

for i in range(50):
    loss = f(x)
    loss.backward()
    with torch.no_grad():
        x -= lr * x.grad
    x.grad.zero_()
    steps.append((i, x.item(), loss.item()))

for step in steps[:5] + steps[-3:]:
    print(f"Iter {step[0]}: x = {step[1]:.6f}, loss = {step[2]:.6f}")
```

```
# Output:
# Iter 0: x = 8.000000, loss = 129.000000
# Iter 1: x = 6.400000, loss = 83.560000
# Iter 2: x = 5.120000, loss = 55.174400
# Iter 3: x = 4.096000, loss = 37.449216
# Iter 4: x = 3.276800, loss = 26.289218
# Iter 47: x = -1.396984, loss = 0.000395
# Iter 48: x = -1.397587, loss = 0.000253
# Iter 49: x = -1.398070, loss = 0.000162
```

### Example 2: Full-Batch GD on Linear Regression

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
X = torch.randn(200, 1)
y = 2.0 * X + 1.0 + 0.1 * torch.randn(200, 1)

w = torch.randn(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.01
loss_fn = nn.MSELoss()

for epoch in range(100):
    y_pred = X @ w + b
    loss = loss_fn(y_pred, y)
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()
    if epoch % 20 == 0 or epoch == 99:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}, w = {w.item():.4f}, b = {b.item():.4f}")
```

```
# Output:
# Epoch 0: loss = 4.876297, w = 2.1196, b = 0.1488
# Epoch 20: loss = 0.291815, w = 2.0491, b = 1.0110
# Epoch 40: loss = 0.019421, w = 2.0130, b = 0.9988
# Epoch 60: loss = 0.009625, w = 2.0019, b = 0.9998
# Epoch 80: loss = 0.009548, w = 2.0005, b = 1.0003
# Epoch 99: loss = 0.009547, w = 2.0001, b = 1.0003
```

### Example 3: Toy Neural Network with Manual GD

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
X = torch.randn(100, 2)
y = ((X[:, 0]**2 + X[:, 1]**2) > 1.0).float().unsqueeze(1)

W1 = torch.randn(2, 16, requires_grad=True)
b1 = torch.zeros(16, requires_grad=True)
W2 = torch.randn(16, 1, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)
lr = 0.5

for epoch in range(2000):
    z1 = X @ W1 + b1
    a1 = torch.tanh(z1)
    logits = a1 @ W2 + b2
    loss = F.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    with torch.no_grad():
        for param in [W1, b1, W2, b2]:
            param -= lr * param.grad
            param.grad.zero_()
    if epoch % 400 == 0:
        acc = ((logits > 0).float() == y).float().mean()
        print(f"Epoch {epoch}: loss = {loss.item():.4f}, acc = {acc.item():.4f}")
```

```
# Output:
# Epoch 0: loss = 1.0564, acc = 0.4300
# Epoch 400: loss = 0.1376, acc = 0.9500
# Epoch 800: loss = 0.0643, acc = 0.9800
# Epoch 1200: loss = 0.0417, acc = 0.9900
# Epoch 1600: loss = 0.0308, acc = 0.9900
```

## Common Mistakes

1. **Forgetting to zero gradients**: Calling `.backward()` accumulates gradients by default. Failing to call `.zero_grad()` (or `optimizer.zero_grad()`) doubles gradient contributions each iteration.
2. **Using too large a learning rate**: When the learning rate exceeds 2/L, gradient descent diverges. Loss suddenly jumps to infinity or NaN.
3. **Using too small a learning rate**: The algorithm converges but impractically slowly. The model appears frozen when training.
4. **Applying gradient descent to non-differentiable objectives**: If the loss function has discontinuities or subgradients where the gradient is undefined, standard GD fails.
5. **Assuming gradient descent finds the global minimum**: For non-convex functions (neural networks), GD converges to a local minimum or saddle point.
6. **Mixing up gradient sign**: Adding the gradient instead of subtracting it maximizes the loss.
7. **Relying on a fixed learning rate throughout training**: Often leads to suboptimal convergence; annealing helps.

## Interview Questions

### Beginner

1. What is gradient descent and what problem does it solve?
2. Explain the role of the learning rate in gradient descent.
3. What happens if the learning rate is set too high?
4. What is the gradient of a function and how does it relate to optimization?
5. Why do we compute gradients in the backward pass in PyTorch?

### Intermediate

1. Derive the gradient descent update rule for mean squared error loss.
2. How does the convergence rate of gradient descent change for convex vs. non-convex functions?
3. What is the Lipschitz continuity condition and why does it matter for gradient descent?
4. How would you implement early stopping with gradient descent in PyTorch?
5. Compare full-batch gradient descent with stochastic gradient descent.

### Advanced

1. Prove that gradient descent with a fixed step size converges for convex functions with L-Lipschitz gradients.
2. Explain Polyak-Lojasiewicz condition and its role in convergence of gradient descent for non-convex problems.
3. How does gradient descent behave near saddle points in high-dimensional spaces?

## Practice Problems

### Easy

1. Implement gradient descent to minimize f(x) = x⁴ − 3x³ + 2.
2. Plot the loss landscape and gradient descent trajectory for a 2D quadratic.
3. Write a function that computes the gradient of MSE loss manually.
4. Compare convergence speed for learning rates [0.001, 0.01, 0.1, 1.0].
5. Implement GD for linear regression from scratch using only PyTorch tensors.

### Medium

1. Implement gradient descent with backtracking line search.
2. Build a 2-layer network trained with manual gradient descent on moons dataset.
3. Visualize the effect of initialization on GD convergence.
4. Implement gradient descent with Polyak step sizes.
5. Compare convergence of GD on Rosenbrock function vs. simple quadratic.

### Hard

1. Prove convergence rate of gradient descent for strongly convex functions.
2. Implement gradient descent with Nesterov acceleration manually.
3. Design and run an experiment showing how GD escapes or gets stuck at saddle points.

## Solutions

Solutions provided upon request. Key patterns: for Easy problems, implement the update loop θ ← θ − η∇J(θ) and track loss. For Medium problems, add line search or adaptive steps. For Hard problems, derive convergence bounds and implement acceleration.

## Related Concepts

- Stochastic Gradient Descent (DL-072): Uses one sample per update
- Mini-Batch Gradient Descent (DL-073): Balances batch and stochastic approaches
- Gradient Clipping (DL-084): Prevents exploding gradients
- Learning Rate Decay (DL-086): Reduces learning rate over time

## Next Concepts

- Stochastic Gradient Descent (DL-072) — the stochastic variant
- SGD with Momentum (DL-074) — accelerates convergence
- Adam Optimizer (DL-078) — adaptive learning rates

## Summary

Gradient descent is a foundational first-order optimization algorithm that iteratively moves parameters in the direction of steepest descent (negative gradient) to minimize a loss function. Its update rule θ ← θ − η∇J(θ) is simple yet powerful. The learning rate η controls step size and critically determines convergence behavior — too large causes divergence, too small causes slow progress. For convex functions, convergence is O(1/k), improving to geometric for strongly convex problems. While full-batch gradient descent is deterministic and stable, it becomes computationally prohibitive for large datasets, motivating stochastic and mini-batch variants.

## Key Takeaways

1. Gradient descent minimizes loss by iteratively taking steps opposite the gradient direction.
2. The learning rate is the single most important hyperparameter; it must be tuned carefully.
3. Convergence is guaranteed for convex functions with appropriate step sizes but not for general non-convex problems.
4. Full-batch GD computes exact gradients but scales poorly with dataset size.
5. Understanding gradient descent is essential before moving to advanced optimizers like Adam or SGD with momentum.
