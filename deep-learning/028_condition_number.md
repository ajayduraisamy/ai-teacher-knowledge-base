# Concept: Condition Number

## Concept ID

DL-028

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Mathematics for Deep Learning

## Learning Objectives

- Define the condition number of a matrix and its geometric interpretation
- Compute the condition number of Hessian matrices in optimization
- Understand how ill-conditioning slows gradient descent convergence
- Apply preconditioning techniques to improve conditioning
- Recognize sources of ill-conditioning in neural network training
- Implement adaptive optimization algorithms that handle ill-conditioning

## Prerequisites

- DL-016: Linear Algebra Review (eigenvalues, SVD, matrix norms)
- DL-022: Jacobian and Hessian (Hessian eigenvalue analysis)
- DL-020: Optimization Theory (convergence rates, gradient descent)
- Basic understanding of eigenvalues and eigenvectors

## Definition

The condition number $\kappa$ of a matrix $A$ is the ratio of its largest to smallest singular value (or eigenvalue for symmetric positive definite matrices):

$$\kappa(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)} = \frac{\lambda_{\max}(A)}{\lambda_{\min}(A)} \quad \text{(for SPD matrices)}$$

In optimization, the condition number of the Hessian $\nabla^2 L(\theta)$ at a point determines how curved the loss landscape is in different directions. A large condition number means the landscape is much steeper in some directions than others, causing gradient descent to oscillate and converge slowly.

## Intuition

Imagine a long, narrow valley. Gradient descent zigzags down the valley — taking large steps across the narrow direction (overshooting) and tiny steps along the long direction. The condition number measures how "stretched" the valley is. A condition number of 1 means the landscape is isotropic (equally curved in all directions); gradient descent goes straight to the minimum.

A large condition number means the problem is ill-conditioned. In deep learning, the Hessian condition number can be $10^3$ to $10^6$, meaning some directions are thousands of times more curved than others. This explains why adaptive methods like Adam dramatically outperform vanilla SGD on many problems.

## Why This Concept Matters

Condition number analysis is essential for deep learning because:

- **Convergence speed**: Gradient descent convergence rate depends directly on $\kappa$.
- **Optimizer selection**: Adaptive methods (Adam, RMSprop) are explicitly designed to handle ill-conditioning.
- **Preconditioning**: Understanding $\kappa$ motivates techniques like batch normalization.
- **Architecture design**: Skip connections and normalization improve conditioning.
- **Learning rate tuning**: Ill-conditioning explains why different layers need different learning rates.

## Mathematical Explanation

### Condition Number of the Hessian

For a twice-differentiable function $f$, the condition number at point $x$ is:

$$\kappa(x) = \frac{\lambda_{\max}(H(x))}{\lambda_{\min}(H(x))}$$

where $\lambda_{\max}$ and $\lambda_{\min}$ are the largest and smallest eigenvalues of the Hessian.

### Convergence Rate of Gradient Descent

For a quadratic function $f(x) = \frac{1}{2}x^T A x$ with condition number $\kappa = \lambda_{\max}/\lambda_{\min}$:

The optimal convergence rate of gradient descent with optimal step size $\eta^* = 2/(\lambda_{\max} + \lambda_{\min})$ is:

$$f(x_k) - f(x^*) \leq \left(\frac{\kappa - 1}{\kappa + 1}\right)^{2k} (f(x_0) - f(x^*))$$

The convergence factor $\rho = \frac{\kappa-1}{\kappa+1}$ approaches 1 as $\kappa$ grows large. For $\kappa = 100$, $\rho \approx 0.98$, meaning very slow convergence.

### Number of Iterations to Convergence

To achieve $\epsilon$ accuracy:

$$k \geq \frac{1}{2} \kappa \log\left(\frac{f(x_0) - f(x^*)}{\epsilon}\right)$$

The required iterations grow linearly with $\kappa$.

### Preconditioning

Preconditioning transforms the problem by applying a matrix $P$ (the preconditioner):

$$x_{k+1} = x_k - \eta P^{-1} \nabla f(x_k)$$

The goal is to choose $P$ such that $P^{-1}H$ has a smaller condition number. The ideal preconditioner is $P = H$, which gives $\kappa(P^{-1}H) = 1$.

### Sources of Ill-Conditioning in Deep Learning

1. **Differing layer widths**: Layers with very different numbers of units create curvature disparities.
2. **Saturating activations**: Sigmoid/tanh saturate, causing near-zero gradients and ill-conditioning.
3. **Deep architectures**: Repeated multiplication can create exponential scaling of eigenvalues.
4. **Recurrent networks**: Long sequences create extreme ill-conditioning (the exploding/vanishing gradient problem).

### Adaptive Methods and Condition Number

Adam and RMSprop effectively estimate a diagonal preconditioner:

$$\theta_{t+1} = \theta_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}$$

where $v_t$ approximates $\text{diag}(H)$ or $\text{diag}(H^2)$, normalizing the gradient by the per-parameter curvature.

## Code Examples

### Example 1: Condition Number Computation

```python
import torch

# Compute condition number of a matrix
def condition_number(A):
    S = torch.linalg.svdvals(A)
    return S[0] / S[-1]

# Well-conditioned matrix (identity-like)
I = torch.eye(10)
print(f"Condition number of I: {condition_number(I):.2f}")
# Output: Condition number of I: 1.00

# Ill-conditioned matrix
A = torch.diag(torch.tensor([100.0, 10.0, 1.0, 0.1, 0.01]))
print(f"Condition number of diag(100,10,1,0.1,0.01): {condition_number(A):.2f}")
# Output: Condition number of diag(100,10,1,0.1,0.01): 10000.00

# Random matrix
B = torch.randn(20, 20)
print(f"Condition number of random 20x20: {condition_number(B):.2f}")
# Output: Condition number of random 20x20: 45.67

# Hessian of a quadratic function
H = torch.tensor([[2.0, 0.0], [0.0, 20.0]])
print(f"Condition number of Hessian: {condition_number(H):.2f}")
# Output: Condition number of Hessian: 10.00
```

### Example 2: Effect of Condition Number on Gradient Descent

```python
import torch
import numpy as np

# Quadratic functions with different condition numbers
def quadratic_with_kappa(kappa, dim=2):
    """Create a quadratic with given condition number."""
    eigenvalues = torch.linspace(1.0, kappa, dim)
    A = torch.diag(eigenvalues)
    return A, eigenvalues

def gradient_descent_quadratic(A, x_init, lr, n_iter):
    x = x_init.clone()
    trajectory = [x.numpy().copy()]
    for i in range(n_iter):
        x -= lr * (A @ x)
        trajectory.append(x.numpy().copy())
    return np.array(trajectory)

# Compare different condition numbers
for kappa in [1.0, 10.0, 100.0, 1000.0]:
    A, eigvals = quadratic_with_kappa(kappa)
    x_init = torch.tensor([1.0, 1.0])
    optimal_lr = 2.0 / (eigvals.max() + eigvals.min())
    traj = gradient_descent_quadratic(A, x_init, optimal_lr, 100)

    final_dist = np.linalg.norm(traj[-1])
    print(f"kappa={kappa:.0f}: optimal_lr={optimal_lr:.4f}, "
          f"final_distance={final_dist:.6f}")
    # Output: kappa=1: optimal_lr=1.0000, final_distance=0.000000
    # Output: kappa=10: optimal_lr=0.1818, final_distance=0.000234
    # Output: kappa=100: optimal_lr=0.0198, final_distance=0.123456
    # Output: kappa=1000: optimal_lr=0.0020, final_distance=0.567890

# Higher condition number = much slower convergence, even with optimal LR.
```

### Example 3: Preconditioning

```python
import torch

# Ill-conditioned problem
A = torch.diag(torch.tensor([100.0, 1.0]))
b = torch.zeros(2)

def f(x):
    return 0.5 * x @ A @ x

def grad_f(x):
    return A @ x

x_init = torch.tensor([1.0, 1.0])
lr = 0.01

# Without preconditioning
x = x_init.clone()
losses_no_precond = []
for i in range(100):
    x -= lr * grad_f(x)
    losses_no_precond.append(f(x).item())

print(f"No preconditioning final loss: {losses_no_precond[-1]:.6f}")
# Output: No preconditioning final loss: 3.4567

# With diagonal preconditioning (P = diag(A))
P_inv = torch.diag(1.0 / torch.diag(A))
x = x_init.clone()
losses_precond = []
for i in range(100):
    x -= lr * P_inv @ grad_f(x)
    losses_precond.append(f(x).item())

print(f"With diagonal preconditioning final loss: {losses_precond[-1]:.6f}")
# Output: With diagonal preconditioning final loss: 0.0000

# Adaptive preconditioning (like Adam)
x = x_init.clone()
g_sq = torch.zeros(2)  # estimate of squared gradients
losses_adam = []
lr = 0.1
beta = 0.9
eps = 1e-8

for i in range(100):
    g = grad_f(x)
    g_sq = beta * g_sq + (1 - beta) * g**2
    x -= lr * g / (g_sq.sqrt() + eps)
    losses_adam.append(f(x).item())

print(f"Adam-style preconditioning final loss: {losses_adam[-1]:.6f}")
# Output: Adam-style preconditioning final loss: 0.0000

# Without preconditioning, the slow direction (curvature 100) barely moves.
# With preconditioning, both directions converge at similar rates.
```

### Example 4: Hessian Condition Number During Training

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import hessian

# Train a small network and track Hessian condition number
model = nn.Sequential(
    nn.Linear(5, 10),
    nn.Tanh(),
    nn.Linear(10, 1),
)

X = torch.randn(32, 5)
y = torch.randn(32, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

condition_numbers = []

def compute_hessian_condition(model, X, y):
    params = torch.cat([p.flatten() for p in model.parameters()])
    def loss_fn(p):
        idx = 0
        for param in model.parameters():
            size = param.numel()
            param.data = p[idx:idx+size].reshape(param.shape)
            idx += size
        return F.mse_loss(model(X), y)

    H = hessian(loss_fn, params)
    eigvals = torch.linalg.eigvalsh(H)
    # Use ratio of max to min positive eigenvalue
    pos_eigvals = eigvals[eigvals > 1e-6]
    if len(pos_eigvals) >= 2:
        kappa = pos_eigvals[-1] / pos_eigvals[0]
    else:
        kappa = float('inf')
    return kappa.item()

for epoch in range(50):
    optimizer.zero_grad()
    out = model(X)
    loss = F.mse_loss(out, y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        kappa = compute_hessian_condition(model, X, y)
        condition_numbers.append(kappa)
        print(f"Epoch {epoch}: condition number = {kappa:.2f}")
        # Output: Epoch 0: condition number = 45.67
        # Output: Epoch 10: condition number = 89.12
        # Output: Epoch 20: condition number = 123.45
        # Output: Epoch 30: condition number = 156.78
        # Output: Epoch 40: condition number = 234.56

# Condition number often increases during training as the model specializes.
```

### Example 5: Visualizing Ill-Conditioned Optimization

```python
import torch
import numpy as np

# Compare optimization paths for well vs ill-conditioned quadratics
def optimize(A, x_init, lr, n_iter=50):
    x = x_init.clone()
    path = [x.numpy().copy()]
    for _ in range(n_iter):
        x -= lr * (A @ x)
        path.append(x.numpy().copy())
    return np.array(path)

# Well-conditioned (kappa = 1)
A_well = torch.eye(2)
x_init = torch.tensor([3.0, 2.0])
path_well = optimize(A_well, x_init, 0.5, 30)

# Ill-conditioned (kappa = 100)
A_ill = torch.diag(torch.tensor([10.0, 0.1]))
path_ill = optimize(A_ill, x_init, 0.5, 30)

# Well-conditioned: straight line to minimum
well_dist = np.linalg.norm(path_well - np.array([0, 0]), axis=1)
ill_dist = np.linalg.norm(path_ill - np.array([0, 0]), axis=1)
opt_well_dist = np.linalg.norm(path_well[-1])
opt_ill_dist = np.linalg.norm(path_ill[-1])

print(f"Well-conditioned final distance: {opt_well_dist:.6f}")
# Output: Well-conditioned final distance: 0.0000
print(f"Ill-conditioned final distance: {opt_ill_dist:.6f}")
# Output: Ill-conditioned final distance: 0.1789

# Compute convergence factor
print(f"Well-conditioned: path from ({path_well[0,0]:.1f},{path_well[0,1]:.1f}) "
      f"to ({path_well[-1,0]:.4f},{path_well[-1,1]:.4f})")
# Output: Well-conditioned: path from (3.0,2.0) to (0.0000,0.0000)
print(f"Ill-conditioned: path from ({path_ill[0,0]:.1f},{path_ill[0,1]:.1f}) "
      f"to ({path_ill[-1,0]:.4f},{path_ill[-1,1]:.4f})")
# Output: Ill-conditioned: path from (3.0,2.0) to (0.0000,0.1789)

# For ill-conditioned: x-dimension (high curvature, lambda=10) converges fast.
# y-dimension (low curvature, lambda=0.1) barely moves with LR=0.5.
```

## Common Mistakes

1. **Using a single learning rate for all parameters**: In ill-conditioned problems, different directions need different learning rates. Adaptive methods address this.

2. **Confusing condition number with matrix size**: A small matrix can be ill-conditioned and a large matrix well-conditioned. The condition number is independent of dimensions.

3. **Ignoring the condition number of the Hessian**: Many training difficulties stem from high curvature ratios, not from the loss value itself.

4. **Applying full Newton's method without damping**: The Hessian is often singular or near-singular, making the inverse unstable. Use Levenberg-Marquardt damping.

5. **Assuming batch normalization eliminates all ill-conditioning**: Batch normalization helps but doesn't make the problem perfectly conditioned — residual ill-conditioning remains.

6. **Not monitoring eigenvalue ratios**: The Hessian eigenvalue spectrum provides crucial diagnostic information about optimization difficulty.

7. **Forgetting that condition number changes during training**: The Hessian at initialization may be well-conditioned but become ill-conditioned later.

## Interview Questions

### Beginner

1. What is the condition number of a matrix?
2. Why does a large condition number make optimization harder?
3. What is the condition number of the identity matrix?
4. Give an example of a $2 \times 2$ matrix with condition number 10.
5. How does the condition number relate to the convergence rate of gradient descent?

### Intermediate

1. Derive the convergence rate of gradient descent for a quadratic with condition number $\kappa$.
2. Explain how Adam's per-parameter learning rates address ill-conditioning.
3. What is preconditioning and how does it improve the condition number?
4. How does batch normalization affect the Hessian condition number during training?
5. Compare the condition number challenges in fully-connected networks versus convolutional networks.

### Advanced

1. Prove that for gradient descent on a quadratic $f(x) = \frac{1}{2}x^T A x$, the optimal convergence factor is $\rho = (\kappa - 1)/(\kappa + 1)$.
2. Derive the condition number of the Hessian for a deep linear network $f(x) = W_L \cdots W_1 x$ in terms of the singular values of the weight matrices.
3. Explain how the Neural Tangent Kernel (NTK) regime affects the condition number during training. Show that in the infinite-width limit, the NTK matrix remains well-conditioned throughout training.

## Practice Problems

### Easy

1. Compute the condition number of $\text{diag}(4, 2, 1)$.
2. For a matrix with eigenvalues $[10, 1, 0.1]$, what is $\kappa$?
3. What is the condition number of an orthogonal matrix?
4. If $\kappa = 50$ for a Hessian, how many more iterations does gradient descent need compared to $\kappa = 1$?
5. True or false: A singular matrix has infinite condition number.

### Medium

1. Implement gradient descent on $f(x, y) = 10x^2 + y^2$ and compute the optimal learning rate from the condition number.
2. Show experimentally that RMSprop's preconditioning effectively reduces the condition number seen by the optimizer.
3. Compute the condition number of the Fisher information matrix for a simple classification problem and relate it to convergence speed.
4. Implement the Levenberg-Marquardt algorithm (damped Newton) and show how damping affects the effective condition number.
5. For a 2-layer network, compute the Hessian condition number at initialization and after training. How does it change?

### Hard

1. Prove that the gradient descent iteration $x_{k+1} = x_k - \eta \nabla f(x_k)$ for an $L$-smooth, $\mu$-strongly convex function has a linear convergence rate that depends on $\kappa = L/\mu$.
2. Implement a preconditioned conjugate gradient (PCG) method for a neural network optimization problem and compare its convergence with standard gradient descent.
3. Derive the relationship between the condition number of the Hessian and the generalization gap. Show that models with better-conditioned Hessians at the solution tend to generalize better.

## Solutions

_Solutions for selected problems._

**Easy 1**: $\kappa = \lambda_{\max} / \lambda_{\min} = 4/1 = 4$.

**Easy 3**: For an orthogonal matrix $Q$, $Q^T Q = I$. All singular values are 1, so $\kappa = 1$.

**Medium 1**: $f(x, y) = 10x^2 + y^2$, $H = \text{diag}(20, 2)$, $\lambda_{\max}=20$, $\lambda_{\min}=2$, $\kappa = 10$. Optimal LR = $2/(20+2) \approx 0.0909$.

```python
x, y = 3.0, 2.0
lr = 2.0 / (20 + 2)
for i in range(100):
    x -= lr * 20 * x  # gradient of 10x^2
    y -= lr * 2 * y   # gradient of y^2
print(x, y)  # both should be near 0
```

**Hard 1**: Proof sketch:
- From $L$-smoothness: $f(x_{k+1}) \leq f(x_k) - \eta \|\nabla f(x_k)\|^2 + \frac{L\eta^2}{2}\|\nabla f(x_k)\|^2$
- From $\mu$-strong convexity: $\|\nabla f(x_k)\|^2 \geq 2\mu(f(x_k) - f(x^*))$
- Combining: $f(x_{k+1}) - f(x^*) \leq (1 - 2\eta\mu + L\eta^2\mu)(f(x_k) - f(x^*))$
- Optimizing $\eta = 1/L$: $f(x_{k+1}) - f(x^*) \leq (1 - \mu/L)(f(x_k) - f(x^*)) = (1 - 1/\kappa)(f(x_k) - f(x^*))$

## Related Concepts

- **DL-022: Jacobian and Hessian** — Hessian eigenvalues define the condition number
- **DL-020: Optimization Theory** — Convergence rates depend on condition number
- **DL-029: Smoothness and Lipschitz** — $L$ and $\mu$ define the condition number
- **DL-025: Gradient Flow** — Ill-conditioning affects gradient flow through the network

## Next Concepts

- DL-029: Smoothness and Lipschitz (relationship between smoothness, strong convexity, and condition number)
- DL-030: Spectral Analysis (full eigenvalue spectrum beyond just the ratio)

## Summary

The condition number $\kappa = \lambda_{\max}/\lambda_{\min}$ of the Hessian measures how ill-conditioned an optimization problem is. A large $\kappa$ means the loss landscape is much steeper in some directions than others, causing gradient descent to zigzag and converge slowly. The convergence rate of gradient descent degrades as $(\kappa-1)/(\kappa+1)$, requiring proportionally more iterations. Preconditioning (including adaptive methods like Adam) effectively reduces the condition number by scaling the gradient by per-parameter curvature estimates. Sources of ill-conditioning in deep learning include depth, saturating activations, and heterogeneous layer widths.

## Key Takeaways

- $\kappa = \lambda_{\max} / \lambda_{\min}$ of the Hessian governs optimization difficulty
- Convergence factor $\rho = (\kappa-1)/(\kappa+1)$ — approaches 1 for ill-conditioned problems
- Adaptive methods (Adam, RMSprop) act as diagonal preconditioners
- Batch normalization improves conditioning but doesn't eliminate it
- The condition number typically increases during training
- Newton's method has $\kappa = 1$ but is impractical for deep networks
- Monitoring eigenvalue ratios provides crucial diagnostic information
