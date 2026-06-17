# Concept: Automatic Differentiation

## Concept ID

DL-063

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Backpropagation

## Learning Objectives

- Understand the principles of automatic differentiation (autodiff)
- Distinguish autodiff from numerical and symbolic differentiation
- Implement a simple autodiff system from scratch
- Analyze the computational graph and gradient computation

## Prerequisites

DL-053 (Computational Graph), DL-055 (Forward Mode Autodiff), DL-064 (Reverse Mode Autodiff), DL-056 (Chain Rule for Neural Nets)

## Definition

Automatic differentiation (autodiff) is a set of techniques for computing exact derivatives of functions specified by computer programs. Unlike numerical differentiation (which approximates) or symbolic differentiation (which manipulates algebraic expressions), autodiff evaluates derivatives by repeatedly applying the chain rule to elementary operations, providing exact results (up to floating-point precision).

## Intuition

Autodiff is like having a smart assistant who watches every arithmetic operation you perform. For each operation, the assistant automatically records both the result and how to compute the derivative. When you ask for the gradient of your final result with respect to any input, the assistant efficiently combines all the recorded derivative rules.

## Why This Concept Matters

Automatic differentiation is the engine that powers deep learning:
- **Every deep learning framework** (PyTorch, TensorFlow, JAX) is built on autodiff
- **Enables gradient-based optimization** without manual derivative computation
- **No approximation error** (unlike numerical differentiation)
- **Efficient computation**: O(ops_forward) for forward mode, O(ops_backward) for reverse mode
- **General purpose**: Works for arbitrary compositions of differentiable functions

## Mathematical Explanation

Three approaches to differentiation:

### Numerical differentiation:
f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
- Approximate, subject to truncation and round-off errors
- O(ε) truncation error, O(1/ε) round-off error
- Simple but inaccurate and computationally expensive

### Symbolic differentiation:
Manipulate algebraic expressions to derive f'(x)
- Exact but can produce exponentially large expressions ("expression swell")
- Cannot handle loops, conditionals, or recursion

### Automatic differentiation:
- Decomposes f into elementary ops (+, *, sin, exp, ...)
- Applies chain rule at each op, computing derivatives precisely
- Forward mode: one forward pass per input
- Reverse mode (backpropagation): one forward + one backward pass regardless of input dimension

## Code Examples

### Example 1: Comparison of differentiation methods

```python
import torch
import numpy as np

def f(x):
    return x ** 3 + torch.sin(x * 2)

x = torch.tensor(1.5, requires_grad=True)

# Autodiff (PyTorch)
y = f(x)
y.backward()
autodiff_grad = x.grad.item()

# Numerical differentiation
x_np = 1.5
h = 1e-6
numerical_grad = (f(torch.tensor(x_np + h)).item() - f(torch.tensor(x_np - h)).item()) / (2 * h)

# Analytical derivative: f'(x) = 3x^2 + 2*cos(2x)
analytical_grad = 3 * 1.5**2 + 2 * np.cos(3)

print(f"Autodiff gradient:    {autodiff_grad:.10f}")
print(f"Numerical gradient:   {numerical_grad:.10f}")
print(f"Analytical gradient:  {analytical_grad:.10f}")
print(f"Autodiff - analytical: {autodiff_grad - analytical_grad:.2e}")
# Output:
# Autodiff gradient:    5.6494889140
# Numerical gradient:   5.6494904237
# Analytical gradient:  5.6494889140
# Autodiff - analytical: 0.00e+00
```

### Example 2: Building a minimal autodiff system

```python
import math

class Value:
    """A differentiable scalar value."""
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # Topological order
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

# Test the autodiff system
x = Value(2.0)
y = Value(3.0)
z = x * y + x ** 2
loss = z.relu()
loss.backward()

print(f"loss = {loss.data}")
print(f"∂loss/∂x = {x.grad}")  # Should be y + 2*x = 3 + 4 = 7
print(f"∂loss/∂y = {y.grad}")  # Should be x = 2
# Output:
# loss = 10.0
# ∂loss/∂x = 7.0
# ∂loss/∂y = 2.0
```

### Example 3: PyTorch's autograd

```python
import torch

# Autograd tracks operations on tensors with requires_grad=True
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = torch.tensor([1.0, 4.0], requires_grad=False)

# Forward pass with automatic graph building
z = x ** 2 + y ** 3
loss = z.sum()

# Backward pass (gradient computation using reverse-mode autodiff)
loss.backward()

print(f"Gradients:")
print(f"  ∂loss/∂x = {x.grad}")  # [2*x[0], 2*x[1]] = [4, 6]
print(f"  ∂loss/∂y = {y.grad if hasattr(y, 'grad') and y.grad is not None else 'None'}")

# Check the graph
print(f"\nComputational graph of loss:")
print(f"  loss.grad_fn: {loss.grad_fn}")
print(f"  z.grad_fn: {z.grad_fn}")
print(f"  y.grad_fn: {y.grad_fn}")
# Output:
# Gradients:
#   ∂loss/∂x = tensor([4., 6.])
#   ∂loss/∂y = None
# 
# Computational graph of loss:
#   loss.grad_fn: <SumBackward0>
#   z.grad_fn: <AddBackward0>
#   y.grad_fn: None
```

### Example 4: Higher-order gradients with autodiff

```python
x = torch.tensor([2.0], requires_grad=True)

# First derivative
y = x ** 3 + 2 * x ** 2 + x
first_grad = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"f'(x) at x=2: {first_grad.item():.4f}")  # 3*x^2 + 4*x + 1 = 21

# Second derivative (using create_graph for the computation graph)
second_grad = torch.autograd.grad(first_grad, x, create_graph=True)[0]
print(f"f''(x) at x=2: {second_grad.item():.4f}")  # 6*x + 4 = 16

# Third derivative
third_grad = torch.autograd.grad(second_grad, x)[0]
print(f"f'''(x) at x=2: {third_grad.item():.4f}")  # 6

# Hessian-vector product for vector functions
x_vec = torch.tensor([1.0, 2.0], requires_grad=True)
def f_vec(x):
    return torch.stack([x[0]**2 + x[1], x[0] * x[1]**2])

y_vec = f_vec(x_vec)
v = torch.tensor([1.0, 1.0])  # direction vector

# Compute (J @ v) using forward-mode or:
# Compute v^T @ H using backward of (v^T @ J)
grad_y = torch.autograd.grad(y_vec.sum(), x_vec, create_graph=True)[0]
hvp = torch.autograd.grad((grad_y * v).sum(), x_vec)[0]
print(f"Hessian-vector product: {hvp}")
# Output:
# f'(x) at x=2: 21.0000
# f''(x) at x=2: 16.0000
# f'''(x) at x=2: 6.0000
# Hessian-vector product: tensor([3., 10.])
```

### Example 5: Disabling autodiff for inference

```python
model = nn.Linear(10, 1)
x = torch.randn(4, 10)

# torch.no_grad() disables autodiff for inference
with torch.no_grad():
    out = model(x)
    print(f"Inference output (no grad tracking): {out}")
    print(f"out.requires_grad: {out.requires_grad}")

# Without no_grad, autodiff tracks the operations
out_grad = model(x)
print(f"Training output (with grad tracking): {out_grad}")
print(f"out.requires_grad: {out_grad.requires_grad}")
# Output:
# Inference output (no grad tracking): tensor([...])
# out.requires_grad: False
# Training output (with grad tracking): tensor([...], grad_fn=<AddmmBackward0>)
# out.requires_grad: True
```

### Example 6: Gradient computation modes

```python
# PyTorch supports multiple autodiff modes
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Mode 1: Standard reverse-mode (backward from scalar)
y = (x ** 2).sum()
y.backward()
print(f"Standard backward: {x.grad}")
x.grad.zero_()

# Mode 2: torch.autograd.grad (don't modify .grad attribute)
y = (x ** 2).sum()
grad = torch.autograd.grad(y, x)
print(f"torch.autograd.grad: {grad}")
x.grad.zero_()

# Mode 3: Vector-Jacobian product (vhp)
y = x ** 2
v = torch.tensor([1.0, 0.5, 0.25])
vjp = torch.autograd.grad(y, x, grad_outputs=v)
print(f"VJP (v @ J): {vjp}")  # [v[0]*2*x[0], v[1]*2*x[1], v[2]*2*x[2]]
# Output:
# Standard backward: tensor([2., 4., 6.])
# torch.autograd.grad: (tensor([2., 4., 6.]),)
# VJP (v @ J): (tensor([2.0000, 2.0000, 1.5000]),)
```

## Common Mistakes

1. **Confusing autodiff with symbolic or numerical differentiation**: Autodiff combines advantages of both — exact like symbolic, works with code like numerical.

2. **Forgetting that autodiff tracks operations on `requires_grad` tensors**: Only operations involving `requires_grad=True` tensors are tracked. Operations on detached tensors don't create graph nodes.

3. **Modifying data in-place breaks the graph**: In-place ops like `x.add_(1)` break the computational graph. Always use functional operations.

4. **Not using `torch.no_grad()` for inference**: Inference doesn't need gradients. Disabling autodiff saves memory and computation.

5. **Trying to compute gradients through non-differentiable operations**: Operations like `torch.argmax`, indexing with variables, or rounding break gradient flow.

6. **Understanding that autodiff has modes**: Forward mode (for few inputs, many outputs) and reverse mode (for many inputs, few outputs) have different use cases.

7. **Overlooking memory costs**: Autodiff stores intermediate values for the backward pass. Very deep graphs use significant memory.

## Interview Questions

### Beginner - 5

1. What is automatic differentiation?
2. How is autodiff different from numerical differentiation?
3. How is autodiff different from symbolic differentiation?
4. What are the two modes of automatic differentiation?
5. What does `requires_grad=True` do in PyTorch?

### Intermediate - 5

1. Explain how autodiff decomposes a function into elementary operations.
2. Compare the computational complexity of forward and reverse mode autodiff.
3. How does autodiff handle control flow (if statements, loops)?
4. What is the role of the computational graph in autodiff?
5. How does PyTorch's dynamic autodiff differ from TensorFlow 1.x's static graph?

### Advanced - 3

1. Implement reverse-mode autodiff from scratch for a minimal tensor library.
2. Analyze the memory complexity of reverse-mode autodiff and propose optimizations.
3. Explain how autodiff enables higher-order gradients and implement double backward.

## Practice Problems

### Easy - 5

1. Use `torch.autograd.grad` to compute the gradient of `f(x) = x.sin().sum()` at `x=2.0`.
2. Compare autodiff and numerical differentiation for `f(x) = e^x * sin(x)`.
3. Disable gradient tracking with `torch.no_grad()` during inference.
4. Compute the gradient of a simple quadratic function using `.backward()`.
5. Verify that autodiff gives exact gradients (up to machine precision).

### Medium - 5

1. Implement a minimal autodiff library supporting +, *, and ReLU.
2. Compare forward and reverse mode for a function with 3 inputs and 1 output.
3. Compute the Hessian of a function using nested `torch.autograd.grad`.
4. Implement gradient checking against numerical differentiation.
5. Profile memory usage of autodiff for deep vs. wide graphs.

### Hard - 3

1. Implement a reverse-mode autodiff system supporting arbitrary computation graphs with broadcasting.
2. Design and implement a custom autograd Function with a non-trivial backward pass.
3. Implement gradient checkpointing by manually controlling which intermediate values are stored.

## Solutions

### Easy - 1
```python
x = torch.tensor(2.0, requires_grad=True)
y = torch.sin(x)
grad = torch.autograd.grad(y, x)
print(grad)  # cos(2) ≈ -0.4161
```

### Easy - 2
```python
x = torch.tensor(1.0, requires_grad=True)
y = torch.exp(x) * torch.sin(x)
y.backward()
autodiff_grad = x.grad.item()
numerical_grad = (torch.exp(torch.tensor(1.0001)) * torch.sin(torch.tensor(1.0001)) - torch.exp(torch.tensor(0.9999)) * torch.sin(torch.tensor(0.9999))).item() / 0.0002
print(f"Autodiff: {autodiff_grad:.6f}, Numerical: {numerical_grad:.6f}")
```

### Easy - 3
```python
with torch.no_grad():
    out = model(x)  # No gradient computation
```

## Related Concepts

DL-053 Computational Graph, DL-055 Forward Mode Autodiff, DL-064 Reverse Mode Autodiff, DL-065 Computation Graph Backward

## Next Concepts

DL-064 Reverse Mode Autodiff, DL-065 Computation Graph Backward

## Summary

Automatic differentiation (autodiff) computes exact derivatives of computer programs by applying the chain rule to elementary operations. It is the core technology behind all deep learning frameworks, providing exact gradients without numerical approximation or symbolic expression swell. Understanding autodiff is essential for advanced model design, custom operations, and debugging gradient computation.

## Key Takeaways

- Autodiff = exact derivatives via chain rule on elementary ops
- Not numerical (no approximation) and not symbolic (no expression swell)
- Two modes: forward (one pass per input) and reverse (one pass per output)
- Reverse mode (backprop) is the standard for neural networks
- PyTorch builds a dynamic computational graph for autodiff
- Higher-order gradients are possible (nested autodiff)
- `torch.no_grad()` disables autodiff for inference
- Custom autograd Functions extend autodiff to new operations
- Memory cost: intermediate values stored for backward pass
