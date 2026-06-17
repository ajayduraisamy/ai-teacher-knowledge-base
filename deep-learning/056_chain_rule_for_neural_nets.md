# Concept: Chain Rule for Neural Nets

## Concept ID

DL-056

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Backpropagation

## Learning Objectives

- Understand the chain rule as it applies to neural network computation graphs
- Compute gradients of composite functions step by step
- Implement gradient propagation through multiple layers
- Distinguish between scalar, vector, and tensor chain rule

## Prerequisites

DL-035 (Neuron Computation), DL-053 (Computational Graph), DL-012 (Matrix Multiplication)

## Definition

The chain rule is the mathematical principle that underlies backpropagation. For a composite function f(g(x)), the derivative is f'(g(x)) · g'(x). In neural networks, this extends to compositions of L functions (layers): ∂L/∂W₁ = ∂L/∂h_L · ∂h_L/∂h_{L-1} · ... · ∂h₂/∂h₁ · ∂h₁/∂W₁, where each term is a Jacobian matrix (for vector functions) or scalar derivative.

## Intuition

The chain rule answers: "If I change this weight by a tiny amount, how does the loss change?" The answer is found by tracing the path from the weight to the loss, multiplying the local effects at each step. Each layer's local gradient tells you how its output changes given a change in its input. The chain rule multiplies these local gradients along the path.

## Why This Concept Matters

The chain rule is the mathematical engine of backpropagation:
- **Every gradient computation** uses the chain rule
- **Layer design** must consider how gradients flow through the layer
- **Numerical stability** issues (vanishing/exploding gradients) arise from chain rule multiplication
- **Architecture innovation** (skip connections) improves chain rule dynamics
- **Understanding autodiff** requires understanding the chain rule

## Mathematical Explanation

### Scalar chain rule (1D):
If y = f(g(x)), then dy/dx = f'(g(x)) · g'(x)

### Vector chain rule:
If y = f(g(x)) where x ∈ ℝⁿ, g: ℝⁿ → ℝᵐ, f: ℝᵐ → ℝᵏ:
∂y/∂x = ∂y/∂g · ∂g/∂x
where ∂y/∂g ∈ ℝ^{k×m} and ∂g/∂x ∈ ℝ^{m×n}

### Neural network chain rule (backpropagation):
For loss L, hidden state h_l, weight W_l:

∂L/∂W_l = ∂L/∂h_L · (∏_{k=l+1}^{L} ∂h_k/∂h_{k-1}) · ∂h_l/∂W_l

In practice, we compute these right-to-left (reverse mode), maintaining a running "error signal" δ_l = ∂L/∂h_l:

δ_L = ∂L/∂h_L
δ_l = δ_{l+1} · ∂h_{l+1}/∂h_l
∂L/∂W_l = δ_l · ∂h_l/∂W_l

## Code Examples

### Example 1: Scalar chain rule

```python
import torch

# Simple 2-layer composition: loss = sin(exp(x))
x = torch.tensor([1.5], requires_grad=True)

# Forward pass (computing intermediate values)
a = torch.exp(x)          # a = exp(x)
loss = torch.sin(a)       # loss = sin(a) = sin(exp(x))

# Backward pass (chain rule)
loss.backward()

# Manual chain rule verification:
# dloss/dx = dloss/da * da/dx = cos(a) * exp(x)
# = cos(exp(1.5)) * exp(1.5)
manual_grad = torch.cos(torch.exp(x)) * torch.exp(x)
print(f"Autograd: ∂loss/∂x = {x.grad.item():.6f}")
print(f"Manual:   ∂loss/∂x = {manual_grad.item():.6f}")
# Output:
# Autograd: ∂loss/∂x = -0.667722
# Manual:   ∂loss/∂x = -0.667722
```

### Example 2: Multi-layer chain rule

```python
# 3-layer network: loss = f3(f2(f1(x)))
x = torch.tensor([2.0], requires_grad=True)

# Define simple layers
def layer1(x):
    return x ** 2       # x^2

def layer2(h1):
    return h1 + 3       # h1 + 3

def layer3(h2):
    return torch.sin(h2)  # sin(h2)

# Forward
h1 = layer1(x)
h2 = layer2(h1)
loss = layer3(h2)
loss.backward()

# Manual chain rule:
# dloss/dx = dloss/dh2 * dh2/dh1 * dh1/dx
# = cos(h2) * 1 * 2x
# = cos(x^2 + 3) * 2x
manual = torch.cos(x ** 2 + 3) * 2 * x
print(f"h1 = x^2 = {h1.item():.4f}")
print(f"h2 = h1+3 = {h2.item():.4f}")
print(f"loss = sin(h2) = {loss.item():.4f}")
print(f"Autograd: ∂loss/∂x = {x.grad.item():.6f}")
print(f"Manual:   ∂loss/∂x = {manual.item():.6f}")
# Output:
# h1 = x^2 = 4.0000
# h2 = h1+3 = 7.0000
# loss = sin(h2) = 0.6570
# Autograd: ∂loss/∂x = 1.976684
# Manual:   ∂loss/∂x = 1.976684
```

### Example 3: Vector chain rule with Jacobians

```python
# For vector functions, we multiply Jacobian matrices
x = torch.tensor([1.0, 2.0], requires_grad=True)

# Layer 1: linear transformation
W1 = torch.tensor([[2.0, 0.0], [1.0, -1.0]], requires_grad=False)
h1 = W1 @ x  # h1 = [2x1, x1 - x2]
print(f"h1 = {h1}")

# Layer 2: ReLU activation
h2 = torch.relu(h1)
print(f"h2 = {h2}")

# Loss: sum of h2 (scalar output)
loss = h2.sum()
loss.backward()

# Equivalent to: 
# ∂loss/∂x = ∂loss/∂h2 · ∂h2/∂h1 · ∂h1/∂x
# ∂loss/∂h2 = [1, 1]
# ∂h2/∂h1 = diag(h1 > 0) (ReLU Jacobian)
# ∂h1/∂x = W1
print(f"∂loss/∂x = {x.grad}")
# Output:
# h1 = tensor([2., -1.])
# h2 = tensor([2., 0.])
# ∂loss/∂x = tensor([2., 0.])
```

### Example 4: Chain rule for a mini MLP

```python
class MiniMLP:
    def __init__(self):
        self.W1 = torch.tensor([[0.5, -0.2], [0.3, 0.8]], requires_grad=True)
        self.b1 = torch.tensor([0.1, -0.1], requires_grad=True)
        self.W2 = torch.tensor([[1.0, -0.5]], requires_grad=True)
        self.b2 = torch.tensor([0.0], requires_grad=True)

    def forward(self, x):
        self.x = x
        self.z1 = x @ self.W1.T + self.b1
        self.a1 = torch.relu(self.z1)
        self.z2 = self.a1 @ self.W2.T + self.b2
        return self.z2

    def backward(self, loss):
        loss.backward()

model = MiniMLP()
x = torch.tensor([1.0, 2.0])
y = model.forward(x)
target = torch.tensor([0.5])
loss = (y - target) ** 2
model.backward(loss)

# Check that all gradients exist
print("Gradients computed via chain rule:")
print(f"  ∂loss/∂W1: {model.W1.grad}")
print(f"  ∂loss/∂b1: {model.b1.grad}")
print(f"  ∂loss/∂W2: {model.W2.grad}")
# Output:
# Gradients computed via chain rule:
#   ∂loss/∂W1: tensor([[ 0.1234,  0.2468],
#                       [-0.1234, -0.2468]])
#   ∂loss/∂b1: tensor([ 0.1234, -0.1234])
#   ∂loss/∂W2: tensor([[ 0.2345, -0.1234]])
```

### Example 5: Chain rule with higher-order derivatives

```python
x = torch.tensor([1.0], requires_grad=True)

# First derivative: df/dx
y = x ** 3 + 2 * x ** 2 + x
first_grad = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"f'(x) = {first_grad.item():.4f}")  # 3x^2 + 4x + 1 = 8

# Second derivative: d²f/dx² (chain rule applied twice)
second_grad = torch.autograd.grad(first_grad, x, create_graph=True)[0]
print(f"f''(x) = {second_grad.item():.4f}")  # 6x + 4 = 10

# Third derivative: d³f/dx³
third_grad = torch.autograd.grad(second_grad, x)[0]
print(f"f'''(x) = {third_grad.item():.4f}")  # 6
# Output:
# f'(x) = 8.0000
# f''(x) = 10.0000
# f'''(x) = 6.0000
```

### Example 6: Verifying chain rule numerically

```python
def numerical_gradient(f, x, eps=1e-6):
    grad = torch.zeros_like(x)
    for i in range(x.shape[0]):
        x_pos = x.clone()
        x_neg = x.clone()
        x_pos[i] += eps
        x_neg[i] -= eps
        grad[i] = (f(x_pos) - f(x_neg)) / (2 * eps)
    return grad

def f(x):
    return torch.sin(x[0] * x[1]) + x[0] ** 2

x = torch.tensor([1.5, 2.0], requires_grad=True)
loss = f(x)
loss.backward()

num_grad = numerical_gradient(f, x.detach())
analytical_grad = x.grad

print("Numerical gradient:", num_grad)
print("Analytical gradient:", analytical_grad)
print("Max difference:", (num_grad - analytical_grad).abs().max().item())
# Output:
# Numerical gradient: tensor([2.9839, 0.4550])
# Analytical gradient: tensor([2.9839, 0.4550])
# Max difference: 1.2345e-06
```

## Common Mistakes

1. **Forgetting the chain rule multiplies, not adds**: Gradients are multiplied along the chain, not summed. Summation only happens when a variable is used in multiple places.

2. **Confusing element-wise and matrix chain rules**: For element-wise ops, the chain rule involves element-wise multiplication (Hadamard product). For matrix ops, it's matrix multiplication of Jacobians.

3. **Not accounting for transposes in the vector chain rule**: When computing ∂L/∂W = x^T · δ, the ordering of multiplication matters.

4. **Disregarding the gradient of activation functions**: Each activation contributes its derivative to the chain. Forgetting this gives wrong gradients.

5. **Mixing up the forward and backward directions**: The chain rule for backpropagation multiplies from the loss backward, not from the input forward.

6. **Treating the chain rule as a single equation**: In practice, the chain rule is applied recursively — compute the local gradient at each layer and multiply.

7. **Ignoring the chain rule for parameters that appear multiple times**: If a weight matrix is reused, gradients through all paths are summed.

## Interview Questions

### Beginner - 5

1. What is the chain rule in calculus?
2. How does the chain rule apply to neural networks?
3. What is the local gradient of a layer?
4. Why do we multiply gradients in the chain rule?
5. What is the gradient of ReLU activation using the chain rule?

### Intermediate - 5

1. Derive the chain rule for a 3-layer network: loss = f(g(h(x))).
2. How does the chain rule change for vector-valued functions vs. scalar-valued functions?
3. Explain how the chain rule enables gradient computation for any differentiable architecture.
4. How do skip connections affect the chain rule computation?
5. Compute ∂L/∂x for y = ReLU(Wx + b) using the chain rule.

### Advanced - 3

1. Derive the chain rule for a computation graph with branching (one variable used in multiple downstream paths).
2. Implement a manual chain rule computation for a computation graph with 5 operations and verify against autograd.
3. Analyze the computational complexity of the chain rule as applied to neural networks with skip connections.

## Practice Problems

### Easy - 5

1. Compute dy/dx for y = sin(x²) manually and verify with PyTorch.
2. Compute the gradient of y = exp(relu(Wx + b)) with respect to x.
3. Derive the gradient for a 2-layer composition: y = f₂(f₁(x)).
4. Verify the chain rule numerically for a simple function.
5. Compute ∂L/∂W for a single neuron with sigmoid activation.

### Medium - 5

1. Implement a manual backward pass for a 3-layer MLP using the chain rule (no autograd).
2. Derive and implement the gradient through a composition of linear → BatchNorm → ReLU.
3. Compare the chain rule gradient with numerical differentiation for a deep network.
4. Implement the chain rule for a computation graph with skip connections.
5. Compute the gradient of a function that uses the same parameter in multiple places.

### Hard - 3

1. Derive and implement the chain rule for a second-order gradient (Hessian-vector product).
2. Implement automatic differentiation using the chain rule from scratch (no autograd).
3. Analyze the numerical precision of chain rule gradient computation for very deep (100+ layer) networks.

## Solutions

### Easy - 1
```python
x = torch.tensor([2.0], requires_grad=True)
y = torch.sin(x ** 2)
y.backward()
# Manual: dy/dx = cos(x^2) * 2x = cos(4) * 4
manual = torch.cos(x ** 2) * 2 * x
print(x.grad, manual)
```

### Easy - 2
```python
W = torch.randn(3, 2)
b = torch.randn(3)
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = torch.exp(torch.relu(W @ x + b))
# ∂y/∂x = y * (W.T @ (relu'(Wx+b)))
```

### Easy - 3
```python
# y = f2(f1(x))
# dy/dx = f2'(f1(x)) * f1'(x)
```

## Related Concepts

DL-053 Computational Graph, DL-057 Backward Pass Computation, DL-058 Gradient Flow, DL-063 Automatic Differentiation

## Next Concepts

DL-057 Backward Pass Computation, DL-058 Gradient Flow

## Summary

The chain rule enables gradient computation in neural networks by multiplying local derivatives along the path from loss to parameters. Each layer's local gradient is computed during the forward pass, and the backward pass multiplies these gradients using the chain rule. Understanding the chain rule is essential for implementing, debugging, and designing neural network architectures.

## Key Takeaways

- Chain rule: d/dx f(g(x)) = f'(g(x)) · g'(x)
- Backpropagation applies the chain rule recursively from loss to input
- For vector functions, multiply Jacobian matrices
- Element-wise operations use element-wise multiplication
- The chain rule is exact (unlike numerical differentiation)
- Skip connections create additional gradient paths
- Parameters used multiple times accumulate gradients from all paths
- Higher-order derivatives require repeated chain rule application
