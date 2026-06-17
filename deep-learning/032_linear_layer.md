# Concept: Linear Layer

## Concept ID

DL-032

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Neural Network Building Blocks

## Learning Objectives

- Distinguish the linear layer from the dense/fully connected layer
- Implement linear transformations using PyTorch's `nn.Linear` and functional API
- Understand the mathematical properties of linear transformations
- Recognize when a linear-only (no activation) layer is appropriate

## Prerequisites

DL-031 (Dense / Fully Connected Layer), DL-012 (Matrix Multiplication), DL-033 (Bias Term)

## Definition

A linear layer is a neural network module that applies an affine transformation to the input: **y** = **xW**^T + **b**. In PyTorch, `nn.Linear` is the canonical implementation. The term "linear layer" is often used interchangeably with "dense layer" or "fully connected layer," though strictly speaking the linear layer performs an affine (linear + bias) transformation.

## Intuition

A linear layer is like a mathematical function that scales, rotates, and shifts the input space. Without any non-linear activation, a stack of linear layers is mathematically equivalent to a single linear layer. This is why non-linearities must be inserted between them. The linear layer alone cannot model curved decision boundaries, but it serves as the fundamental parameterized mapping from which all deeper networks are built.

## Why This Concept Matters

Linear layers are the core parameterized operation in neural networks. Every `nn.Linear` call is a learnable matrix multiplication. Understanding their properties — rank, spectral norm, conditioning — is essential for advanced topics like weight decay, initialization schemes, and optimization landscapes.

## Mathematical Explanation

Given input **x** ∈ ℝ^{batch × n} and weights **W** ∈ ℝ^{m × n}:

**y** = **xW**^T + **b**

Element-wise: y_{i,k} = Σ_{j=1}^{n} x_{i,j} W_{k,j} + b_k

Properties:
- Linearity: L(a**x** + **y**) = aL(**x**) + L(**y**)
- Without bias, a linear layer is a pure linear map represented by matrix **W**
- Composition of two linear layers without activation is another linear layer: L₂(L₁(x)) = (xW₁^T + b₁)W₂^T + b₂ = x(W₁W₂)^T + (b₁W₂^T + b₂)

## Code Examples

### Example 1: Basic linear transformation

```python
import torch
import torch.nn as nn

x = torch.tensor([[1.0, 2.0, 3.0]])
linear = nn.Linear(3, 2)
with torch.no_grad():
    linear.weight.copy_(torch.tensor([[1.0, 0.5, 0.0],
                                      [0.0, 0.5, 1.0]]))
    linear.bias.copy_(torch.tensor([0.1, -0.1]))

y = linear(x)
print("Input:", x)
print("Output:", y)
# Output:
# Input: tensor([[1., 2., 3.]])
# Output: tensor([[ 2.1000,  2.4000]], grad_fn=<AddmmBackward0>)
```

### Example 2: Stacking linear layers without activation (redundant)

```python
import torch.nn as nn

# Two linear layers stacked without activation
model = nn.Sequential(
    nn.Linear(4, 4),
    nn.Linear(4, 2)
)

# Mathematically equivalent single layer
merged = nn.Linear(4, 2)

# Copy weights: combined = W2 @ W1, bias = W2 @ b1 + b2
with torch.no_grad():
    W1, b1 = model[0].weight, model[0].bias
    W2, b2 = model[1].weight, model[1].bias
    merged.weight.copy_(W2 @ W1)
    merged.bias.copy_(W2 @ b1 + b2)

x = torch.randn(1, 4)
out_stacked = model(x)
out_merged = merged(x)
print("Difference:", (out_stacked - out_merged).abs().max().item())
# Output:
# Difference: 0.0
```

### Example 3: Using `F.linear` functional API

```python
import torch
import torch.nn.functional as F

x = torch.randn(2, 5)
weight = torch.randn(3, 5)
bias = torch.randn(3)

y = F.linear(x, weight, bias)
print("Functional linear output shape:", y.shape)
# Output:
# Functional linear output shape: torch.Size([2, 3])
```

### Example 4: Linear layer without bias

```python
x = torch.randn(4, 8)
linear_no_bias = nn.Linear(8, 4, bias=False)
y = linear_no_bias(x)
print("Output shape:", y.shape)
print("Bias exists:", linear_no_bias.bias is None)
# Output:
# Output shape: torch.Size([4, 4])
# Bias exists: True
```

### Example 5: Visualizing the linear transformation effect

```python
import torch
import torch.nn as nn

# Unit square corners
square = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
linear_map = nn.Linear(2, 2, bias=False)
with torch.no_grad():
    linear_map.weight.copy_(torch.tensor([[1.0, 0.5],
                                          [0.0, 1.0]]))  # shear

transformed = linear_map(square)
print("Original corners:\n", square)
print("Sheared corners:\n", transformed)
# Output:
# Original corners:
#  tensor([[0., 0.],
#          [1., 0.],
#          [1., 1.],
#          [0., 1.]])
# Sheared corners:
#  tensor([[0.0000, 0.0000],
#          [1.0000, 0.5000],
#          [1.0000, 1.5000],
#          [0.0000, 1.0000]])
```

## Common Mistakes

1. **Thinking linear layers can model non-linear functions**: Without activation functions, stacking linear layers is equivalent to a single linear layer. Always insert non-linearities between them.

2. **Confusing linear with dense/FC**: The terms are interchangeable in modern deep learning, though "linear" emphasizes the mathematical operation while "dense" emphasizes the connectivity pattern.

3. **Forgetting the transpose**: PyTorch's `nn.Linear` stores weight as (out_features, in_features) and computes `x @ W.T`. This is opposite to the convention in many textbooks.

4. **Using linear layers for sequential data without considering time dimension**: For sequences of length L with d features, you need (L, d) input, not (L,) or (d,).

5. **Not handling batch dimensions correctly**: If your input is (C,) instead of (1, C), you'll get an error. Always ensure at least 2D input.

6. **Assuming linear layers preserve vector norms**: Weight matrices can arbitrarily scale inputs. Weight initialization and regularization are needed to control this.

7. **Applying bias after batch normalization**: Using `nn.Linear` with `bias=True` followed by `nn.BatchNorm1d` creates redundant parameters because BatchNorm has its own shift parameter.

## Interview Questions

### Beginner - 5

1. What operation does a linear layer perform?
2. What is the shape of the weight matrix in a linear layer mapping 10 features to 5?
3. Can you use a linear layer without a bias term? How?
4. What is the difference between a linear layer and a dense layer?
5. Why doesn't stacking two linear layers help without an activation function?

### Intermediate - 5

1. Prove mathematically that two consecutive linear layers without activation collapse to one linear layer.
2. What is the rank of a linear layer's weight matrix, and why does it matter for expressivity?
3. How does `F.linear` differ from `nn.Linear` in terms of state management?
4. How would you implement a linear layer that applies different weights to different parts of the input (a blocked linear layer)?
5. What happens to the gradient through a linear layer if the weight matrix is orthogonal?

### Advanced - 3

1. Derive the backward pass of a linear layer and show how it relates to the forward pass of the transposed operation.
2. How can you use the SVD of a linear layer's weight matrix to compress a model? What are the trade-offs?
3. Explain the concept of "linear region" in the context of ReLU networks and how linear layers partition the input space.

## Practice Problems

### Easy - 5

1. Create a linear layer mapping 5 to 3 and print the weight matrix shape.
2. Implement a linear layer without bias and verify that `F.linear(x, W)` gives the same result.
3. Show that `nn.Linear(4, 4, bias=False)` with identity weights is the identity function.
4. Compute the output of a linear layer where all weights are 1 and input is [1,2,3].
5. Create two linear layers and show their composition is equivalent to a single linear layer.

### Medium - 5

1. Write a function that takes a list of `nn.Linear` layers and merges them into one equivalent layer (assuming no activations between them).
2. Implement an orthogonal linear layer whose weight matrix satisfies WW^T = I. Show that it preserves input norms.
3. Compare the computational efficiency of `nn.Linear` vs. a manual loop for a (1000, 1000) layer.
4. Implement spectral normalization for a linear layer: divide weights by the largest singular value.
5. Create a linear layer with block-diagonal structure and verify that it has fewer parameters than a full linear layer.

### Hard - 3

1. Implement a linear layer using the Fourier basis: learn coefficients in frequency space instead of pixel space.
2. Build a linear layer that adaptively masks weights based on input magnitude (a gated linear unit variant).
3. Derive and implement a method to measure the effective rank of a linear layer during training and relate it to model capacity.

## Solutions

### Easy - 1
```python
layer = nn.Linear(5, 3)
print(layer.weight.shape)  # (3, 5)
```

### Easy - 2
```python
W = torch.randn(3, 5)
layer = nn.Linear(5, 3, bias=False)
layer.weight = nn.Parameter(W)
x = torch.randn(2, 5)
assert torch.allclose(layer(x), F.linear(x, W))
```

### Easy - 3
```python
layer = nn.Linear(4, 4, bias=False)
nn.init.eye_(layer.weight)
x = torch.randn(3, 4)
assert torch.allclose(layer(x), x)
```

## Related Concepts

DL-031 Dense / Fully Connected Layer, DL-012 Matrix Multiplication, DL-033 Bias Term, DL-034 Weight Matrix, DL-035 Neuron Computation

## Next Concepts

DL-036 Layer Normalization, DL-037 Batch Normalization, DL-046 Forward Pass Computation

## Summary

A linear layer performs an affine transformation **y** = **xW**^T + **b**. It is the fundamental parameterized building block of neural networks. Without activation functions, multiple linear layers collapse into one. Linear layers are used everywhere — from simple MLPs to the QKV projections in transformers.

## Key Takeaways

- Linear layer = matrix multiplication + bias addition
- Stacking linear layers without activations is mathematically equivalent to a single layer
- PyTorch stores weights as (out_features, in_features) — the transpose convention
- Linear layers preserve the topology of the input space (they are continuous, differentiable maps)
- Use `bias=False` when the next layer normalizes (BatchNorm, LayerNorm) to avoid redundant parameters
