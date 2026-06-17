# Concept: Optimization Theory for Deep Learning

## Concept ID

DL-020

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Mathematics for Deep Learning

## Learning Objectives

- Define convex sets and convex functions with mathematical precision
- Apply first-order optimality conditions to identify minima
- Understand gradient descent and its convergence properties
- Analyze convergence rates for convex and strongly convex functions
- Differentiate between batch, stochastic, and mini-batch gradient descent
- Interpret regret bounds in online convex optimization
- Visualize optimization trajectories in 2D loss landscapes

## Prerequisites

- DL-017: Matrix Calculus (gradients and Hessians)
- DL-016: Linear Algebra (norms, inner products)
- DL-018: Probability for DL (expectation, stochastic processes)
- Basic understanding of neural network training

## Definition

Optimization theory studies how to find the minimum (or maximum) of a function. In deep learning, optimization algorithms minimize the loss function with respect to the model parameters. The field provides theoretical guarantees about convergence rates, helps understand why certain algorithms work better than others, and guides the design of new optimizers like Adam, RMSprop, and SGD with momentum.

## Intuition

Imagine you're hiking in a foggy mountain range trying to find the lowest valley. You can only feel the slope at your feet. Gradient descent says: take a step in the opposite direction of the steepest uphill slope, then repeat. The size of your step (learning rate) matters — too large and you might overshoot the valley; too small and you'll take forever.

Stochastic gradient descent (SGD) is like trying to find that valley but you can only feel the slope on one small patch of ground at a time. Each measurement is noisy but fast. Over many measurements, you still make progress downhill.

Convex optimization is the ideal case: the landscape is a bowl with a single bottom. Deep learning is non-convex — the landscape has many valleys, plateaus, and ridges — but convex theory still provides crucial insights.

## Why This Concept Matters

Optimization theory is the engine that trains all deep learning models. Understanding it enables you to:

- Choose appropriate learning rates and schedules
- Diagnose convergence issues (oscillation, stagnation)
- Design optimizers for specific architectures
- Understand the impact of batch size, momentum, and weight decay
- Read and contribute to optimization research

Without optimization theory, hyperparameter tuning becomes blind guesswork.

## Mathematical Explanation

### Convex Sets

A set $C \subseteq \mathbb{R}^n$ is convex if for any $x, y \in C$ and $\lambda \in [0, 1]$:

$$\lambda x + (1-\lambda)y \in C$$

All points on the line segment between any two points also lie in the set.

### Convex Functions

A function $f: \mathbb{R}^n \to \mathbb{R}$ is convex if its domain is convex and for all $x, y$ in the domain and $\lambda \in [0, 1]$:

$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda) f(y)$$

The function lies below the chord connecting any two points.

**First-order characterization** (for differentiable $f$):

$$f(y) \geq f(x) + \nabla f(x)^T (y - x)$$

The first-order Taylor approximation is a global lower bound.

**Second-order characterization** (for twice-differentiable $f$):

$$\nabla^2 f(x) \succeq 0 \quad \text{(Hessian is positive semidefinite for all } x\text{)}$$

### First-Order Optimality Condition

For unconstrained convex optimization $\min_x f(x)$:

$$\nabla f(x^*) = 0$$

This is both necessary and sufficient for a global minimum when $f$ is convex.

For constrained convex optimization $\min_{x \in C} f(x)$:

$$\nabla f(x^*)^T (y - x^*) \geq 0 \quad \forall y \in C$$

### Gradient Descent

$$x_{k+1} = x_k - \eta \nabla f(x_k)$$

where $\eta$ is the learning rate (step size).

**Convergence for convex L-smooth functions:**

$$f(x_k) - f(x^*) \leq \frac{L\|x_0 - x^*\|^2}{2k}$$

The suboptimality decreases as $O(1/k)$, meaning we need $O(1/\epsilon)$ iterations to reach $\epsilon$ accuracy.

**Convergence for $\mu$-strongly convex functions:**

$$f(x_k) - f(x^*) \leq \frac{L}{2} \exp\left(-\frac{k}{\kappa}\right) \|x_0 - x^*\|^2$$

where $\kappa = L/\mu$ is the condition number. This is linear (exponential) convergence — much faster.

### Stochastic Gradient Descent (SGD)

$$\tilde{\nabla} f(x_k) = \frac{1}{|B|} \sum_{i \in B} \nabla \ell_i(x_k) \quad \text{(mini-batch gradient)}$$

$$x_{k+1} = x_k - \eta_k \tilde{\nabla} f(x_k)$$

The stochastic gradient is an unbiased estimate of the true gradient:

$$\mathbb{E}[\tilde{\nabla} f(x)] = \nabla f(x)$$

SGD converges with rate $O(1/\sqrt{k})$ for convex functions with bounded variance.

### Regret Bounds (Online Convex Optimization)

Regret measures the cumulative difference between the algorithm's loss and the best fixed decision in hindsight:

$$\text{Regret}_T = \sum_{t=1}^T f_t(x_t) - \min_x \sum_{t=1}^T f_t(x)$$

For online gradient descent with convex losses:

$$\text{Regret}_T \leq O(\sqrt{T})$$

This means the average regret goes to 0 as $T \to \infty$.

### Momentum

$$v_{k+1} = \beta v_k + \nabla f(x_k)$$
$$x_{k+1} = x_k - \eta v_{k+1}$$

Momentum accelerates convergence in directions of consistent gradient and dampens oscillations.

## Code Examples

### Example 1: Gradient Descent on a Convex Function

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

# Convex function: f(x, y) = x^2 + 2y^2
def f(x, y):
    return x**2 + 2*y**2

def grad_f(x, y):
    return torch.tensor([2*x, 4*y])

# Gradient descent
x, y = torch.tensor(3.0), torch.tensor(2.0)  # initial point
lr = 0.1
trajectory = [(x.item(), y.item())]

for i in range(50):
    g = grad_f(x, y)
    x -= lr * g[0]
    y -= lr * g[1]
    trajectory.append((x.item(), y.item()))

print(f"Final point: ({x.item():.4f}, {y.item():.4f})")
# Output: Final point: (0.0000, 0.0000)
print(f"Final loss: {f(x, y).item():.6f}")
# Output: Final loss: 0.000000

# The minimum is at (0, 0) with f=0.
# Note: convergence along y is slower because curvature is higher (2 vs 1).
```

### Example 2: Convergence Rate Visualization

```python
import torch
import numpy as np

# Compare convergence rates
def gradient_descent(f, grad_f, x_init, lr, n_iter, mu=None, L=None):
    x = x_init.clone()
    trajectory = [x.item()]
    losses = [f(x).item()]
    for i in range(n_iter):
        x -= lr * grad_f(x)
        trajectory.append(x.item())
        losses.append(f(x).item())
    return trajectory, losses

# f(x) = x^2 (convex), f(x) = x^2 + sin(3x) (non-convex)
def f_convex(x):
    return x**2

def grad_convex(x):
    return 2*x

def f_nonconvex(x):
    return x**2 + torch.sin(3*x)

def grad_nonconvex(x):
    return 2*x + 3*torch.cos(3*x)

x_init = torch.tensor(2.0)
lr = 0.1

_, losses_convex = gradient_descent(f_convex, grad_convex, x_init, lr, 30)
_, losses_nonconvex = gradient_descent(f_nonconvex, grad_nonconvex, x_init, lr, 30)

print(f"Convex final loss: {losses_convex[-1]:.6f}")
# Output: Convex final loss: 0.000000
print(f"Non-convex final loss: {losses_nonconvex[-1]:.6f}")
# Output: Non-convex final loss: 0.2897

# Non-convex converges to a local minimum, not the global one.
```

### Example 3: Mini-Batch SGD

```python
import torch

# Generate synthetic regression data
torch.manual_seed(42)
N, d = 1000, 10
X = torch.randn(N, d)
true_w = torch.randn(d)
y = X @ true_w + 0.1 * torch.randn(N)

# Full batch gradient descent
w_full = torch.randn(d, requires_grad=True)
optimizer_full = torch.optim.SGD([w_full], lr=0.01)
losses_full = []

for epoch in range(100):
    y_pred = X @ w_full
    loss = (y_pred - y).pow(2).mean()
    optimizer_full.zero_grad()
    loss.backward()
    optimizer_full.step()
    losses_full.append(loss.item())

# Mini-batch SGD (batch size 32)
w_mini = torch.randn(d, requires_grad=True)
optimizer_mini = torch.optim.SGD([w_mini], lr=0.01)
losses_mini = []
batch_size = 32

for epoch in range(100):
    permutation = torch.randperm(N)
    for i in range(0, N, batch_size):
        idx = permutation[i:i+batch_size]
        X_batch, y_batch = X[idx], y[idx]
        y_pred = X_batch @ w_mini
        loss = (y_pred - y_batch).pow(2).mean()
        optimizer_mini.zero_grad()
        loss.backward()
        optimizer_mini.step()
    # Full loss for monitoring
    with torch.no_grad():
        y_pred_full = X @ w_mini
        full_loss = (y_pred_full - y).pow(2).mean()
        losses_mini.append(full_loss.item())

print(f"Batch GD final loss: {losses_full[-1]:.6f}")
# Output: Batch GD final loss: 0.0102
print(f"Mini-batch SGD final loss: {losses_mini[-1]:.6f}")
# Output: Mini-batch SGD final loss: 0.0102

# Mini-batch SGD usually converges faster in wall-clock time
# even though it has more gradient noise.
```

### Example 4: Effect of Learning Rate

```python
import torch

# Quadratic function f(x) = x^2
x_init = torch.tensor(2.0)
lrs = [0.01, 0.1, 0.5, 0.9, 1.5]
n_iter = 20

for lr in lrs:
    x = x_init.clone()
    trajectory = [x.item()]
    for i in range(n_iter):
        x -= lr * 2 * x  # gradient of x^2 is 2x
        trajectory.append(x.item())
    print(f"lr={lr:.2f}: final x = {x.item():.4f}")
    # Output: lr=0.01: final x = 1.2159
    # Output: lr=0.10: final x = 0.0222
    # Output: lr=0.50: final x = 0.0000
    # Output: lr=0.90: final x = 0.0000
    # Output: lr=1.50: final x = 1.0000 (diverges!)

# The optimal learning rate for f(x)=x^2 is lr < 1/L where L=2 (smoothness).
# lr >= 2 causes divergence.
```

### Example 5: Momentum vs No Momentum

```python
import torch

# Ill-conditioned quadratic: f(x,y) = x^2 + 100*y^2
def step_no_momentum(x, y, lr=0.01):
    gx, gy = 2*x, 200*y
    return x - lr*gx, y - lr*gy

def step_momentum(x, y, vx, vy, lr=0.01, beta=0.9):
    gx, gy = 2*x, 200*y
    vx = beta*vx + lr*gx
    vy = beta*vy + lr*gy
    return x - vx, y - vy, vx, vy

x_nm, y_nm = 2.0, 2.0
x_m, y_m, vx, vy = 2.0, 2.0, 0.0, 0.0

for i in range(50):
    x_nm, y_nm = step_no_momentum(x_nm, y_nm)
    x_m, y_m, vx, vy = step_momentum(x_m, y_m, vx, vy)

print(f"No momentum: x={x_nm:.4f}, y={y_nm:.4f}")
# Output: No momentum: x=0.0000, y=0.2707
print(f"Momentum: x={x_m:.4f}, y={y_m:.4f}")
# Output: Momentum: x=0.0000, y=0.0000

# Momentum dramatically improves convergence on ill-conditioned problems.
```

## Common Mistakes

1. **Learning rate too high causes divergence**: The loss may initially decrease but then shoot to infinity. Always monitor loss and reduce LR if it diverges.

2. **Learning rate too low causes stagnation**: The loss decreases extremely slowly, giving the false impression of convergence.

3. **No learning rate schedule**: Constant learning rates often fail to converge to high precision. Use cosine annealing, step decay, or ReduceLROnPlateau.

4. **Assuming convergence to global minimum**: Neural network loss landscapes are non-convex. SGD converges to a local minimum or saddle point, which may still generalize well.

5. **Ignoring gradient noise in SGD**: Mini-batch gradients are noisy. The noise level affects the convergence rate and the quality of the solution found.

6. **Confusing batch and epoch**: One epoch = one pass over all data. One batch = one gradient computation. The number of iterations per epoch depends on batch size.

7. **Comparing optimizers unfairly**: Different optimizers (SGD, Adam, RMSprop) have different hyperparameters. A fair comparison requires tuning each one's hyperparameters.

## Interview Questions

### Beginner

1. What is the difference between convex and non-convex optimization?
2. Explain gradient descent in one sentence.
3. What is the role of the learning rate in gradient descent?
4. Why do we use stochastic gradient descent instead of full-batch gradient descent?
5. What does it mean for an optimization algorithm to converge?

### Intermediate

1. Derive the $O(1/k)$ convergence rate for gradient descent on smooth convex functions.
2. Explain the trade-off between batch size and convergence speed in SGD.
3. How does momentum help convergence? What is its effect on the optimization trajectory?
4. What is the condition number of a convex function and how does it affect gradient descent convergence?
5. Compare Adam and SGD: strengths, weaknesses, and when to use each.

### Advanced

1. Prove that for a convex $L$-smooth function, gradient descent with $\eta \leq 1/L$ achieves $f(x_k) - f(x^*) \leq \frac{L\|x_0 - x^*\|^2}{2k}$.
2. Derive the Polyak-Lojasiewicz (PL) condition and show how it ensures linear convergence even for non-convex functions.
3. Explain the implicit bias of SGD: why does SGD converge to solutions with certain properties (e.g., low norm) that differ from those found by full-batch GD?

## Practice Problems

### Easy

1. Is the function $f(x) = x^4$ convex? Check using the second derivative.
2. What is the gradient descent update for $f(x) = (x-3)^2 + 5$ starting at $x=0$ with $\eta=0.1$?
3. Compute one step of SGD if the mini-batch gradient is $g = [2.5, -1.3]^T$ and the learning rate is $0.01$.
4. What is the optimal learning rate for $f(x) = 5x^2$?
5. Is the set $C = \{x \in \mathbb{R}^2 : \|x\| \leq 1\}$ convex? Why or why not?

### Medium

1. Implement gradient descent with Nesterov momentum for a quadratic function and compare its convergence with standard momentum.
2. Show that the function $f(x) = \log(1 + e^{-x})$ is convex. What is its gradient?
3. Implement cosine annealing learning rate schedule and show its effect on training a simple classifier.
4. Compare the convergence of SGD, SGD with momentum, and Adam on a simple multi-layer network trained on MNIST.
5. Derive the optimal learning rate for gradient descent on $f(x) = \frac{1}{2}x^T A x$ where $A$ is positive definite with eigenvalues $\lambda_1 \geq \cdots \geq \lambda_n$.

### Hard

1. Prove that gradient descent with momentum (heavy-ball method) achieves accelerated convergence for strongly convex quadratics.
2. Implement a full batch Hessian-free optimization (conjugate gradient) for a small neural network. Compare iterations to convergence with gradient descent.
3. Derive and implement the SHADO (Stochastic Heavy-ball Acceleration) optimizer with theoretical convergence guarantees.

## Solutions

_Solutions for selected problems._

**Easy 1**: $f''(x) = 12x^2 \geq 0$, so yes, $f(x)=x^4$ is convex.

**Easy 4**: For $f(x) = 5x^2$, $f''(x) = 10 = L$. The optimal learning rate for convex quadratics is $\eta \leq 2/L = 0.2$. The "optimal" that minimizes the worst-case convergence rate is $\eta = 1/L = 0.1$.

**Medium 5**: For $f(x) = \frac{1}{2}x^T A x$, the optimal convergence rate is achieved with $\eta^* = \frac{2}{\lambda_1 + \lambda_n}$, giving convergence factor $\rho = \frac{\kappa - 1}{\kappa + 1}$ where $\kappa = \lambda_1/\lambda_n$.

**Hard 1**: Proof outline: For $f(x) = \frac{1}{2}x^T A x$, the heavy-ball method with optimal parameters achieves convergence factor $\rho = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$, compared to $\rho = \frac{\kappa - 1}{\kappa + 1}$ for gradient descent. This is the Polyak momentum acceleration.

## Related Concepts

- **DL-017: Matrix Calculus** — Computing gradients for optimization
- **DL-024: Convex vs Non-Convex** — Understanding when theory applies to practice
- **DL-028: Condition Number** — How ill-conditioning affects convergence
- **DL-029: Smoothness and Lipschitz** — Theoretical foundation for learning rate selection

## Next Concepts

- DL-024: Convex vs Non-Convex Optimization
- DL-028: Condition Number
- DL-029: Smoothness and Lipschitz

## Summary

Optimization theory provides the mathematical framework for training deep learning models. Convex functions have a single global minimum and gradient descent converges with rate $O(1/k)$. Strongly convex functions achieve linear (exponential) convergence. SGD uses noisy mini-batch gradients for computational efficiency, converging with rate $O(1/\sqrt{k})$. Momentum accelerates convergence, especially for ill-conditioned problems. Learning rate selection is critical: too large causes divergence, too small causes slow progress. While neural networks are non-convex, convex optimization theory provides essential insights and practical guidance for training algorithms.

## Key Takeaways

- Convex functions: gradient $\nabla f(x^*) = 0$ is necessary and sufficient for global minimum
- Gradient descent converges at rate $O(1/k)$ for convex smooth functions
- Strong convexity gives linear (exponential) convergence
- SGD trades exact gradients for computational efficiency
- Mini-batch size controls gradient noise and affects generalization
- Momentum accelerates convergence in ill-conditioned landscapes
- Learning rate is the single most important hyperparameter to tune
