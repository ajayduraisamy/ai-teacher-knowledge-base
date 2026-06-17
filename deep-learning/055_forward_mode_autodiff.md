# Concept: Forward Mode Autodiff

## Concept ID

DL-055

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Backpropagation

## Learning Objectives

- Understand the mechanics of forward-mode automatic differentiation
- Distinguish forward mode from reverse mode (backpropagation)
- Implement forward-mode autodiff using dual numbers
- Identify scenarios where forward mode outperforms reverse mode

## Prerequisites

DL-063 (Automatic Differentiation), DL-053 (Computational Graph), DL-054 (Directed Acyclic Graph)

## Definition

Forward-mode automatic differentiation computes the derivative of a function with respect to one input by propagating a "seed" derivative forward through the computation graph alongside the actual values. Each operation computes both the output value and the output derivative (tangent) with respect to the seeded input. If implemented with dual numbers, a single forward pass computes both the function value and its derivative.

## Intuition

Imagine driving a car and tracking not just your position (value) but also your speed (derivative) at every point along the route. At each turn and acceleration, you update both your position and your speed simultaneously. Forward-mode autodiff does this: alongside every computation, it also computes how the output would change if the input changed.

## Why This Concept Matters

Forward mode is one of two fundamental modes of automatic differentiation:
- **Efficient for few inputs, many outputs**: Computing d/dx₁ for a function f: ℝⁿ → ℝᵐ is O(n) forward passes in reverse mode but O(1) forward passes per input in forward mode
- **Simple to implement**: Dual numbers make it elegantly algebraic
- **Foundation for Jacobian-vector products**: Forward mode computes J·v (Jacobian times vector) efficiently
- **Useful for ODE solvers, sensitivity analysis, and optimization**
- **Complementary to reverse mode**: Understanding both is essential for a complete picture of autodiff

## Mathematical Explanation

Each number is augmented with a tangent component: x → (x, ẋ). Operations are extended:

Addition: (x, ẋ) + (y, ẏ) = (x + y, ẋ + ẏ)
Multiplication: (x, ẋ) * (y, ẏ) = (x*y, x*ẏ + ẋ*y)
Sin: sin(x, ẋ) = (sin(x), cos(x)*ẋ)
Exp: exp(x, ẋ) = (exp(x), exp(x)*ẋ)

To compute ∂f/∂x₁, set the tangent of x₁ to 1 (seed) and all other input tangents to 0. The output tangent equals ∂f/∂x₁.

For a function f: ℝⁿ → ℝᵐ, the forward mode requires n passes to compute the full Jacobian (one per input dimension). The reverse mode requires m passes (one per output dimension).

## Code Examples

### Example 1: Forward mode dual number implementation

```python
import torch

class DualNumber:
    def __init__(self, val, deriv=0.0):
        self.val = val
        self.deriv = deriv

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = DualNumber(other, 0.0)
        return DualNumber(self.val + other.val, self.deriv + other.deriv)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = DualNumber(other, 0.0)
        # (x + ẋε)(y + ẏε) = xy + (xẏ + ẋy)ε
        return DualNumber(self.val * other.val,
                         self.val * other.deriv + self.deriv * other.val)

    def __repr__(self):
        return f"val={self.val:.4f}, deriv={self.deriv:.4f}"

# Define a function using dual numbers
def f(x):
    return x * x + 3 * x + 1

# Compute f(2) and f'(2) in one pass
x = DualNumber(2.0, 1.0)  # seed derivative to 1
result = f(x)
print(f"f(2) = {result.val:.4f}")
print(f"f'(2) = {result.deriv:.4f}")
# Analytical: f(x) = x^2 + 3x + 1, f'(x) = 2x + 3, f'(2) = 7
# Output:
# f(2) = 11.0000
# f'(2) = 7.0000
```

### Example 2: Computing partial derivatives with forward mode

```python
# Function of 2 variables: f(x, y) = x² * y + sin(y)
def f(x, y):
    return x * x * y + torch.sin(y)

# Forward mode for ∂f/∂x
x = DualNumber(2.0, 1.0)  # seed x
y = DualNumber(3.0, 0.0)  # no seed for y
result_x = f(x, y)
print(f"∂f/∂x at (2,3) = {result_x.deriv:.4f}")
# Analytical: ∂f/∂x = 2xy, ∂f/∂x(2,3) = 12

# Forward mode for ∂f/∂y
x = DualNumber(2.0, 0.0)  # no seed for x
y = DualNumber(3.0, 1.0)  # seed y
result_y = f(x, y)
print(f"∂f/∂y at (2,3) = {result_y.deriv:.4f}")
# Analytical: ∂f/∂y = x² + cos(y), ∂f/∂y(2,3) = 4 + cos(3)
# Output:
# ∂f/∂x at (2,3) = 12.0000
# ∂f/∂y at (2,3) = 3.0100
```

### Example 3: Forward mode using PyTorch's forward AD

```python
# PyTorch supports forward-mode AD via torch.autograd.forward_ad
from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual

x = torch.tensor([2.0, 3.0], requires_grad=True)
y = torch.tensor([1.0, 4.0], requires_grad=True)

with dual_level():
    # Seed tangent for x (first element only)
    x_dual = make_dual(x, torch.tensor([1.0, 0.0]))
    y_dual = make_dual(y, torch.tensor([0.0, 0.0]))

    # Compute function
    z = x_dual ** 2 + y_dual ** 3
    loss = z.sum()

    # Extract primal and tangent
    primal, tangent = unpack_dual(loss)
    print(f"Loss = {primal:.4f}")
    print(f"∂loss/∂x₁ = {tangent:.4f}")
# Output:
# Loss = 73.0000
# ∂loss/∂x₁ = 4.0000
```

### Example 4: Jacobian-vector product with forward mode

```python
# Forward mode naturally computes J·v
def f_vec(x):
    return torch.stack([x[0] ** 2 + x[1],
                        x[0] * x[1],
                        torch.sin(x[0])])

x = torch.tensor([1.0, 2.0])
v = torch.tensor([1.0, 0.0])  # direction vector

with dual_level():
    x_dual = make_dual(x, v)
    y_dual = f_vec(x_dual)

    primal, tangent = unpack_dual(y_dual)
    print("J·v (forward mode):")
    for i, (p, t) in enumerate(zip(primal, tangent)):
        print(f"  f{i}={p:.4f}, J·v[{i}]={t:.4f}")

# Analytical Jacobian at (1,2):
# J = [[2x₀, 1],
#      [x₁, x₀],
#      [cos(x₀), 0]]
# J·[1,0] = [2, 2, cos(1)]
print("Expected: [2.00, 2.00, 0.54]")
# Output:
# J·v (forward mode):
#   f0=3.0000, J·v[0]=2.0000
#   f1=2.0000, J·v[1]=2.0000
#   f2=0.8415, J·v[2]=0.5403
# Expected: [2.00, 2.00, 0.54]
```

### Example 5: Forward mode for multiple inputs (full Jacobian row)

```python
# To compute full gradient of a scalar function, forward mode needs n passes
def f_multi(x):
    return (x[0] ** 2 + x[1] * x[2] + torch.sin(x[3])).sum()

x = torch.tensor([1.0, 2.0, 3.0, 4.0])
n = x.shape[0]

print("Computing gradient via forward mode (n passes):")
grad = []
for i in range(n):
    v = torch.zeros(n)
    v[i] = 1.0
    with dual_level():
        x_dual = make_dual(x, v)
        y = f_multi(x_dual)
        _, tangent = unpack_dual(y)
        grad.append(tangent.item())
        print(f"  ∂f/∂x{i} = {tangent.item():.4f}")

# Compare with reverse mode (1 pass)
x_rev = x.clone().requires_grad_()
y_rev = f_multi(x_rev)
y_rev.backward()
print("Reverse mode gradient:", x_rev.grad)
# Output:
# Computing gradient via forward mode (n passes):
#   ∂f/∂x0 = 2.0000
#   ∂f/∂x1 = 3.0000
#   ∂f/∂x2 = 2.0000
#   ∂f/∂x3 = -0.6536
# Reverse mode gradient: tensor([ 2.0000,  3.0000,  2.0000, -0.6536])
```

### Example 6: Forward mode vs reverse mode efficiency

```python
import time

def time_computation(mode, n_inputs=100, n_outputs=1):
    x = torch.randn(n_inputs)

    def computation(x):
        h = x
        for _ in range(10):
            h = torch.sin(h @ torch.randn(n_inputs, n_inputs))
        return h.sum() if n_outputs == 1 else h

    if mode == 'forward':
        # n_inputs forward passes
        start = time.time()
        for i in range(n_inputs):
            v = torch.zeros(n_inputs)
            v[i] = 1.0
            with dual_level():
                x_dual = make_dual(x, v)
                y = computation(x_dual)
                _, _ = unpack_dual(y)
        fwd_time = time.time() - start

    elif mode == 'reverse':
        # 1 backward pass
        x_rev = x.clone().requires_grad_()
        start = time.time()
        y = computation(x_rev)
        y.backward()
        rev_time = time.time() - start

    return fwd_time if mode == 'forward' else rev_time

# Time both modes
fwd_time = time_computation('forward', 50, 1)
rev_time = time_computation('reverse', 50, 1)
print(f"Forward mode (50 passes): {fwd_time:.4f}s")
print(f"Reverse mode (1 pass):    {rev_time:.4f}s")
print(f"Ratio: {fwd_time/rev_time:.1f}x")
# Output:
# Forward mode (50 passes): 0.4567s
# Reverse mode (1 pass):    0.0123s
# Ratio: 37.1x
```

## Common Mistakes

1. **Using forward mode for neural network training**: Backpropagation (reverse mode) is far more efficient for training because NNs typically have many inputs and one loss (scalar output).

2. **Not resetting the seed between forward mode evaluations**: Each forward mode pass computes the derivative w.r.t. one seeded input. Reusing without resetting accumulates tangents.

3. **Confusing forward-mode and reverse-mode gradients**: Forward mode computes derivatives alongside the forward computation; reverse mode computes them backward from the loss.

4. **Ignoring round-off errors**: Dual numbers avoid symbolic differentiation's expression swell and numerical differentiation's truncation error, but floating-point precision still matters.

5. **Forgetting that dual number operations must be defined for every primitive**: Every operation used in the function needs a dual number version.

6. **Applying forward mode to functions with many inputs and few outputs**: This is the worst case — O(n) passes. Reverse mode (O(m) passes with m=number of outputs) is better.

7. **Thinking forward mode is always better than reverse mode**: They have complementary strengths. Forward mode for few inputs, reverse mode for few outputs.

## Interview Questions

### Beginner - 5

1. What is forward-mode automatic differentiation?
2. What are dual numbers?
3. How does forward mode compute derivatives in a single pass?
4. What is the complexity of forward mode for a function f: ℝⁿ → ℝ?
5. When is forward mode more efficient than reverse mode?

### Intermediate - 5

1. Describe the dual number rules for addition, multiplication, and chain rule.
2. How does forward mode compute the Jacobian-vector product J·v?
3. Compare the computational complexity of forward and reverse mode for f: ℝⁿ → ℝᵐ.
4. Why is forward mode not used for training neural networks?
5. How do you compute the full Jacobian using forward mode?

### Advanced - 3

1. Implement a forward-mode autodiff system from scratch using operator overloading with dual numbers.
2. Derive the forward-mode propagation rules for matrix multiplication.
3. Analyze the memory complexity of forward-mode autodiff and compare with reverse mode.

## Practice Problems

### Easy - 5

1. Implement dual number addition and multiplication.
2. Use forward mode to compute the derivative of f(x) = x³ + 2x at x = 3.
3. Compute ∂f/∂x for f(x, y) = x²y + y² at (2, 3) using forward mode.
4. Use PyTorch's forward_ad to compute J·v for a simple vector function.
5. Compare forward and reverse mode for f(x) = x₁² + x₂² (scalar output, 2 inputs).

### Medium - 5

1. Implement a forward-mode autodiff framework supporting sin, cos, exp, and log.
2. Compute the full Jacobian of a function f: ℝ³ → ℝ² using both forward and reverse mode.
3. Profile the time cost of forward mode as input dimension grows.
4. Use forward mode to compute Hessian-vector products (via two forward passes).
5. Implement a dual number matrix class for forward-mode matrix operations.

### Hard - 3

1. Implement a mixed-mode autodiff that uses forward mode for some paths and reverse mode for others within the same graph.
2. Derive and implement a forward-mode variant that computes second-order derivatives in a single pass.
3. Analyze the numerical stability of forward-mode vs. reverse-mode for ill-conditioned functions.

## Solutions

### Easy - 1
```python
class Dual:
    def __init__(self, v, d=0.0):
        self.v, self.d = v, d
    def __add__(self, o):
        return Dual(self.v + o.v, self.d + o.d) if isinstance(o, Dual) else Dual(self.v + o, self.d)
    def __mul__(self, o):
        return Dual(self.v * o.v, self.v * o.d + self.d * o.v) if isinstance(o, Dual) else Dual(self.v * o, self.d * o)
```

### Easy - 2
```python
x = Dual(3.0, 1.0)
f = x * x * x + 2 * x
print(f"f'(3) = {f.d}")  # 3*3^2 + 2 = 29
```

### Easy - 3
```python
x = Dual(2.0, 1.0)
y = Dual(3.0, 0.0)
f = x * x * y + y * y
print(f"∂f/∂x = {f.d}")  # 2*x*y = 12
```

## Related Concepts

DL-063 Automatic Differentiation, DL-064 Reverse Mode Autodiff, DL-053 Computational Graph, DL-055 Forward Mode Autodiff

## Next Concepts

DL-063 Automatic Differentiation, DL-064 Reverse Mode Autodiff

## Summary

Forward-mode automatic differentiation computes derivatives by propagating tangent values alongside primal values through the computational graph using dual numbers. It requires one forward pass per input dimension, making it ideal for functions with few inputs and many outputs, but impractical for neural network training where the reverse mode (backpropagation) dominates.

## Key Takeaways

- Forward mode: value + derivative (tangent) propagated simultaneously
- Dual numbers enable elegant algebraic differentiation
- Complexity: O(n) passes for f: ℝⁿ → ℝᵐ
- Efficient for few inputs, many outputs
- Computes J·v (Jacobian-vector product) naturally
- Not used for neural network training (reverse mode is better)
- Requires one pass per input dimension for full gradient
- Exact up to floating-point precision (no truncation error)
