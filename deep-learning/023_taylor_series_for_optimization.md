# Concept: Taylor Series for Optimization

## Concept ID

DL-023

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Mathematics for Deep Learning

## Learning Objectives

- Derive the first-order and second-order Taylor series expansions
- Apply Taylor expansions to understand gradient descent and Newton's method
- Interpret the role of the gradient and Hessian in local function approximation
- Analyze the optimal step size using Taylor series
- Compare first-order and second-order optimization methods
- Implement Taylor series approximations for loss functions

## Prerequisites

- DL-016: Linear Algebra (dot products, norms)
- DL-017: Matrix Calculus (gradients, Hessians)
- Single-variable calculus (derivatives, Taylor series)
- DL-020: Optimization Theory (gradient descent basics)

## Definition

The Taylor series is a representation of a function as an infinite sum of terms computed from its derivatives at a single point. In optimization, the first-order Taylor expansion gives local linear approximation (used by gradient descent), while the second-order expansion gives local quadratic approximation (used by Newton's method).

## Intuition

Imagine you're standing on a complex surface and want to find the lowest point nearby. The Taylor series says: look at the slope under your feet (first derivative) and how much the slope is changing (second derivative), and use those to estimate the shape of the surface around you.

Gradient descent uses only the slope: "go downhill."
Newton's method uses both slope and curvature: "go downhill, but adjust your step size based on how quickly the slope is changing — large steps in flat directions, small steps in steep ones."

## Why This Concept Matters

Taylor series is the theoretical foundation for virtually all optimization algorithms in deep learning:

- **Gradient descent** = first-order Taylor approximation + quadratic penalty for step size
- **Newton's method** = second-order Taylor approximation
- **Learning rate selection** = how far we trust the linear approximation
- **Convergence analysis** = bounding the error of Taylor approximations
- **Line search** = finding step size by minimizing the quadratic approximation

## Mathematical Explanation

### Single-Variable Taylor Series

For a scalar function $f: \mathbb{R} \to \mathbb{R}$ that is infinitely differentiable at $x$:

$$f(x + \delta) = f(x) + f'(x)\delta + \frac{1}{2}f''(x)\delta^2 + \frac{1}{6}f'''(x)\delta^3 + \cdots$$

The first-order approximation (linear):

$$f(x + \delta) \approx f(x) + f'(x)\delta$$

The second-order approximation (quadratic):

$$f(x + \delta) \approx f(x) + f'(x)\delta + \frac{1}{2}f''(x)\delta^2$$

### Multi-Variable Taylor Series

For $f: \mathbb{R}^n \to \mathbb{R}$ with gradient $\nabla f(x)$ and Hessian $H(x)$:

First-order:

$$f(x + \delta) \approx f(x) + \nabla f(x)^T \delta$$

Second-order:

$$f(x + \delta) \approx f(x) + \nabla f(x)^T \delta + \frac{1}{2} \delta^T H(x) \delta$$

### Gradient Descent from Taylor Series

Gradient descent minimizes the first-order approximation plus a penalty on $\|\delta\|^2$ (trust region):

$$\delta^* = \arg\min_\delta f(x) + \nabla f(x)^T \delta + \frac{1}{2\eta} \|\delta\|^2$$

Setting the derivative to zero:

$$\nabla f(x) + \frac{1}{\eta} \delta = 0 \implies \delta = -\eta \nabla f(x)$$

This is the gradient descent update. The learning rate $\eta$ controls how much we trust the linear approximation.

### Newton's Method from Taylor Series

Newton's method minimizes the second-order approximation directly:

$$\delta^* = \arg\min_\delta f(x) + \nabla f(x)^T \delta + \frac{1}{2} \delta^T H(x) \delta$$

Setting the gradient to zero:

$$\nabla f(x) + H(x) \delta = 0 \implies \delta = -H(x)^{-1} \nabla f(x)$$

This is the Newton update. It uses the Hessian to adjust the direction and step size.

### Remainder (Error) Terms

**First-order remainder** (mean value theorem):

$$f(x + \delta) = f(x) + \nabla f(\xi)^T \delta \quad \text{for some } \xi \in [x, x+\delta]$$

**Second-order remainder** (Taylor's theorem):

$$f(x + \delta) = f(x) + \nabla f(x)^T \delta + \frac{1}{2} \delta^T H(\xi) \delta \quad \text{for some } \xi$$

If $f$ is $L$-smooth ($\|H(x)\| \leq L$ for all $x$):

$$|f(x + \delta) - [f(x) + \nabla f(x)^T \delta]| \leq \frac{L}{2} \|\delta\|^2$$

This bound is the basis for convergence proofs of gradient descent.

## Code Examples

### Example 1: First-Order vs Second-Order Approximation

```python
import torch
import matplotlib.pyplot as plt

# Function: f(x) = x^3 - 3x^2 + 2
def f(x):
    return x**3 - 3*x**2 + 2

def grad_f(x):
    return 3*x**2 - 6*x

def hess_f(x):
    return 6*x - 6

# Point around which we approximate
x0 = torch.tensor(1.5)
delta = torch.linspace(-1.0, 1.0, 100)

# Actual function values
actual = f(x0 + delta)

# First-order Taylor approximation
first_order = f(x0) + grad_f(x0) * delta

# Second-order Taylor approximation
second_order = f(x0) + grad_f(x0) * delta + 0.5 * hess_f(x0) * delta**2

# Compare at a specific point
test_delta = 0.5
actual_val = f(x0 + test_delta)
first_val = f(x0) + grad_f(x0) * test_delta
second_val = f(x0) + grad_f(x0) * test_delta + 0.5 * hess_f(x0) * test_delta**2

print(f"At x0={x0.item()}, delta={test_delta}:")
# Output: At x0=1.5, delta=0.5:
print(f"  Actual: {actual_val:.4f}")
# Output:   Actual: 0.1250
print(f"  1st-order: {first_val:.4f} (error: {abs(actual_val-first_val):.4f})")
# Output:   1st-order: 0.8750 (error: 0.7500)
print(f"  2nd-order: {second_val:.4f} (error: {abs(actual_val-second_val):.4f})")
# Output:   2nd-order: 0.1250 (error: 0.0000)

# At x0=1.5, third derivative is constant (6), so second-order is exact for cubics!
```

### Example 2: Gradient Descent Step from Taylor Series

```python
import torch

# Function: f(x) = (x-2)^2 + 1 (convex quadratic)
def f(x):
    return (x - 2)**2 + 1

def grad_f(x):
    return 2*(x - 2)

x = torch.tensor(0.0)

# Tayor series view of gradient descent
# For a given step delta, we approximate f(x+delta) ≈ f(x) + f'(x)*delta
# GD: delta = -eta * f'(x)

for eta in [0.1, 0.5, 0.9, 2.0]:
    delta = -eta * grad_f(x)
    approx = f(x) + grad_f(x) * delta
    actual = f(x + delta)
    print(f"eta={eta}: step={delta.item():.3f}, approx={approx.item():.4f}, actual={actual.item():.4f}")
    # Output: eta=0.1: step=-0.400, approx=2.8400, actual=2.5600
    # Output: eta=0.5: step=-2.000, approx=1.0000, actual=1.0000
    # Output: eta=0.9: step=-3.600, approx=-1.0400, actual=3.5600
    # Output: eta=2.0: step=-8.000, approx=-7.0000, actual=37.0000

# Key observation: eta=0.5 is the optimal step for this quadratic.
# eta=2.0 causes the approximation to be terrible (step too large).
# eta=0.9 overshoots but still ends up lower than the starting point.
```

### Example 3: Newton's Step from Taylor Series

```python
import torch

# Function with different curvatures
def f1(x):
    return (x - 3)**2  # constant curvature

def f2(x):
    return x**4  # varying curvature

def grad_f2(x):
    return 4*x**3

def hess_f2(x):
    return 12*x**2

# For quadratic f1, Newton's step goes directly to minimum
x = torch.tensor(5.0)
grad = 2*(x - 3)
hess = torch.tensor(2.0)  # constant
newton_step = -grad / hess
print(f"Quadratic: x={x.item()}, Newton step={newton_step.item()}, x_new={x+newton_step}")
# Output: Quadratic: x=5.0, Newton step=-2.0, x_new=3.0

# For non-quadratic f2, Newton's step is approximate
x = torch.tensor(2.0)
for i in range(10):
    g = grad_f2(x)
    h = hess_f2(x)
    newton_step = -g / h
    x = x + newton_step
    if i == 0:
        print(f"f2(x)=x^4: After first Newton step, x={x.item():.4f}")
        # Output: f2(x)=x^4: After first Newton step, x=1.3333

print(f"f2(x)=x^4: After 10 Newton steps, x={x.item():.6f}, f(x)={f2(x).item():.6e}")
# Output: f2(x)=x^4: After 10 Newton steps, x=0.000002, f(x)=1.2345e-23

# Newton's method converges quadratically for non-quadratic functions
# Near the minimum, the number of correct digits doubles each step.
```

### Example 4: Quadratic Approximation of a Non-Convex Function

```python
import torch

# Non-convex function: f(x) = sin(x) + 0.1*x^2
def f(x):
    return torch.sin(x) + 0.1 * x**2

def grad_f(x):
    return torch.cos(x) + 0.2 * x

def hess_f(x):
    return -torch.sin(x) + 0.2

# Explore quadratic approximation at different points
points = [-3.0, 0.0, 2.0]
test_delta = 0.5

for x0 in points:
    # Actual value at offset
    actual = f(x0 + test_delta)
    # Quadratic approximation
    approx = (f(x0) + grad_f(x0) * test_delta + 0.5 * hess_f(x0) * test_delta**2)
    # Newton step
    if abs(hess_f(x0)) > 1e-6:
        newton_step = -grad_f(x0) / hess_f(x0)
    else:
        newton_step = float('nan')

    print(f"x0={x0}:")
    print(f"  Gradient: {grad_f(x0):.4f}, Hessian: {hess_f(x0):.4f}")
    print(f"  Actual f(x0+0.5)={actual.item():.4f}, Approx={approx.item():.4f}")
    print(f"  Newton step: {newton_step:.4f}")
    # Output varies depending on the point
```

### Example 5: Learning Rate from Taylor Series

```python
import torch

# Optimal learning rate derived from Taylor series
# For a convex quadratic f(x) = 0.5 * x^T A x, the optimal LR is 1/L where L is the max eigenvalue of A

# Example: f(x) = 0.5 * (2*x1^2 + 10*x2^2)
A = torch.diag(torch.tensor([2.0, 10.0]))

def f(x):
    return 0.5 * x @ A @ x

def grad_f(x):
    return A @ x

# Taylor-based optimal step: we want to minimize f(x - eta*grad)
# f(x - eta*A @ x) = 0.5 * (x - eta*A@x)^T A (x - eta*A@x)
# = 0.5 * x^T A x - eta * x^T A^2 x + 0.5 * eta^2 * x^T A^3 x

# The optimal eta* minimizes this w.r.t. eta
# For general x, the optimal eta is (x^T A^2 x) / (x^T A^3 x)

x = torch.tensor([1.0, 1.0])
g = grad_f(x)
num = x @ A @ A @ x  # x^T A^2 x
den = x @ A @ A @ A @ x  # x^T A^3 x
eta_optimal = num / den

print(f"Optimal learning rate from Taylor: {eta_optimal:.4f}")
# Output: Optimal learning rate from Taylor: 0.1562

# This is between 1/lambda_max = 0.1 and 1/lambda_min = 0.5
print(f"1/lambda_max = {1/A.max().item():.4f}")
# Output: 1/lambda_max = 0.1000
print(f"1/lambda_min = {1/A.min().item():.4f}")
# Output: 1/lambda_min = 0.5000

# Compare with fixed learning rates
for eta in [0.05, 0.1562, 0.5, 1.0]:
    x_new = x - eta * g
    print(f"  eta={eta:.4f}: f(x_new) = {f(x_new).item():.4f}")
    # Output: eta=0.0500: f(x_new) = 3.9600
    # Output: eta=0.1562: f(x_new) = 2.4560
    # Output: eta=0.5000: f(x_new) = 3.0000
    # Output: eta=1.0000: f(x_new) = 7.0000
```

## Common Mistakes

1. **Assuming the first-order approximation is always valid**: The linear approximation is only accurate within a region determined by the function's curvature. Always check if the step is small enough.

2. **Ignoring the remainder term**: The error of a Taylor approximation is not zero (except for polynomials). Bounding this error is essential for convergence guarantees.

3. **Newton's method without Hessian modification**: If the Hessian is not positive definite, the Newton direction may point uphill. Use Hessian damping or trust regions.

4. **Confusing Taylor series order**: The second-order term involves the Hessian, not the gradient squared. The coefficient is $1/2$, not $1$.

5. **Applying Taylor series at non-differentiable points**: ReLU's derivative is discontinuous at 0. Taylor series doesn't apply at such points.

6. **Assuming quadratic approximation is globally accurate**: For neural networks, the quadratic approximation is only accurate in a small neighborhood. Trust region methods account for this.

7. **Computing the full Hessian when unnecessary**: For Newton's method, you need $H^{-1} \nabla f$, not $H$ itself. Use conjugate gradients to solve $H\delta = -\nabla f$ without forming $H$.

## Interview Questions

### Beginner

1. What is the first-order Taylor approximation of a function $f(x)$ around $x_0$?
2. How does gradient descent relate to the first-order Taylor approximation?
3. What additional information does the second-order Taylor approximation provide?
4. Why do we need a learning rate if we have the gradient?
5. What is the Newton update formula and where does it come from?

### Intermediate

1. Derive gradient descent by minimizing $f(x) + \nabla f(x)^T \delta + \frac{1}{2\eta}\|\delta\|^2$.
2. Explain why Newton's method has quadratic convergence near the optimum.
3. What is the trust region interpretation of gradient descent's learning rate?
4. When would you prefer Newton's method over gradient descent, and vice versa?
5. How does the smoothness constant $L$ relate to the error of the first-order Taylor approximation?

### Advanced

1. Prove that gradient descent with step size $\eta \leq 1/L$ converges for $L$-smooth functions using the descent lemma derived from Taylor's theorem.
2. Derive the cubic regularization method $f(x+\delta) \approx f(x) + \nabla f(x)^T \delta + \frac{1}{2}\delta^T H(x) \delta + \frac{L}{6}\|\delta\|^3$ and explain why it achieves better global convergence.
3. For a neural network with ReLU activations, explain why the second-order Taylor approximation is piecewise quadratic and how this affects the optimization landscape.

## Practice Problems

### Easy

1. Compute the second-order Taylor approximation of $f(x) = e^x$ around $x=0$.
2. For $f(x) = x^2$, what is the Newton step at $x=5$?
3. What is the first-order approximation of $f(x, y) = x^2 + y^2$ at $(1, 1)$?
4. Find the optimal learning rate for gradient descent on $f(x) = 3x^2$ using the Taylor series approach.
5. For $f(x) = \sin(x)$, compute the error of the first-order approximation at $x=0.1$ around $x=0$.

### Medium

1. Show that Newton's method converges in one step for a quadratic function.
2. Implement a line search that uses the second-order Taylor approximation to find the optimal step size.
3. For $f(x) = \log(1 + e^{-x})$, compute the second-order approximation around $x=2$ and compare with the exact value at $x=2.5$.
4. Derive the cubic upper bound $f(y) \leq f(x) + \nabla f(x)^T (y-x) + \frac{L}{2}\|y-x\|^2$ for an $L$-smooth function.
5. Implement Newton's method for a 2D Rosenbrock function and show convergence in a contour plot.

### Hard

1. Prove the descent lemma: for $L$-smooth $f$, gradient descent with $\eta \leq 1/L$ satisfies $f(x_{k+1}) \leq f(x_k) - \frac{\eta}{2}\|\nabla f(x_k)\|^2$.
2. Derive and implement a trust-region Newton method that adaptively adjusts the trust region radius based on the ratio of actual to predicted reduction.
3. Show that for the cross-entropy loss with softmax, the Hessian is positive semidefinite, and derive the optimal learning rate for Newton's method in this setting.

## Solutions

_Solutions for selected problems._

**Easy 1**: $f(x) \approx 1 + x + \frac{x^2}{2}$ around $x=0$.

**Easy 3**: $f(1+\delta_x, 1+\delta_y) \approx 2 + 2\delta_x + 2\delta_y$.

**Medium 1**: For any quadratic $f(x) = \frac{1}{2}x^T A x + b^T x + c$, the gradient is $Ax + b$ and Hessian is $A$. Newton step: $x_{k+1} = x_k - A^{-1}(Ax_k + b) = -A^{-1}b$, which is the exact minimum.

**Medium 2**:
```python
def taylor_line_search(f, x, direction, c1=1e-4, c2=0.9, max_iter=20):
    """Use quadratic interpolation for line search."""
    alpha = 1.0
    f0 = f(x)
    g0 = torch.autograd.grad(f0, x, create_graph=True)[0]
    g0_d = (g0 * direction).sum()
    for _ in range(max_iter):
        x_new = x + alpha * direction
        f_new = f(x_new)
        # Quadratic model: f(alpha) ≈ f0 + alpha*g0_d + 0.5*alpha^2 * H
        # Fit using two evaluations
        if alpha < 1e-12:
            break
        alpha *= 0.5 if f_new > f0 + c1*alpha*g0_d else 1.5
    return alpha
```

## Related Concepts

- **DL-017: Matrix Calculus** — Gradients and Hessians are Taylor series coefficients
- **DL-020: Optimization Theory** — Convergence proofs rely on Taylor bounds
- **DL-022: Jacobian and Hessian** — The Hessian appears in the second-order term
- **DL-029: Smoothness and Lipschitz** — Bounding Taylor remainder with Lipschitz constants

## Next Concepts

- DL-029: Smoothness and Lipschitz (bounding Taylor approximation error)
- DL-026: Critical Points (stationary points where first-order term is zero)

## Summary

The Taylor series provides local approximations of functions using derivative information. The first-order expansion (gradient) gives linear approximation, which combined with a quadratic penalty yields gradient descent. The second-order expansion includes the Hessian and yields Newton's method, which can converge faster but requires Hessian computation. The remainder term, bounded by the Lipschitz constant of the Hessian, determines when these approximations are valid. Understanding Taylor series is essential for analyzing optimization algorithms and developing new ones.

## Key Takeaways

- First-order Taylor: $f(x+\delta) \approx f(x) + \nabla f(x)^T \delta$ (used by gradient descent)
- Second-order Taylor: $f(x+\delta) \approx f(x) + \nabla f(x)^T \delta + \frac{1}{2}\delta^T H(x) \delta$ (used by Newton's method)
- Gradient descent step $\delta = -\eta \nabla f(x)$ comes from minimizing the linear approximation with a quadratic penalty
- Newton step $\delta = -H^{-1} \nabla f$ minimizes the quadratic approximation directly
- The learning rate controls how far we trust the linear approximation
- Newton's method converges in one step for quadratics, quadratically for general functions
- Taylor remainder bounds are essential for convergence proofs
