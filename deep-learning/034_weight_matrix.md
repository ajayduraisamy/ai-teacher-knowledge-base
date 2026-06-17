# Concept: Weight Matrix

## Concept ID

DL-034

## Difficulty

Beginner

## Domain

Deep Learning

## Module

Neural Network Building Blocks

## Learning Objectives

- Define the weight matrix and understand its role in a neural network layer
- Identify the shape and structure of weight matrices for different layer types
- Initialize weight matrices using PyTorch utilities
- Understand how weight matrices are updated during training

## Prerequisites

DL-001 (Perceptron), DL-012 (Matrix Multiplication), DL-031 (Dense / Fully Connected Layer)

## Definition

The weight matrix is the collection of learnable parameters connecting the inputs to the outputs of a neural network layer. For a dense layer mapping n inputs to m outputs, the weight matrix has shape (m, n) in PyTorch's convention. Each element W_{ij} represents the strength of the connection from input j to output i.

## Intuition

Think of the weight matrix as a transformation engine. Each column corresponds to an input feature, and each row corresponds to an output neuron. The value W_{ij} determines how much input feature j influences output neuron i. During training, these values are adjusted to minimize the loss function. A weight of zero means no connection, a large positive weight means strong excitatory influence, and a negative weight means inhibitory influence.

## Why This Concept Matters

The weight matrix is where all learning happens. Every parameter update during training modifies these weights. The structure, initialization, and dynamics of weight matrices determine whether a network trains successfully, generalizes well, or overfits. Understanding weight matrices is essential for grasping optimization, regularization, and model capacity.

## Mathematical Explanation

For a dense layer with input **x** ∈ ℝ^n and weights **W** ∈ ℝ^{m × n}:

**z** = **Wx** + **b**

Element-wise: z_i = Σ_{j=1}^{n} W_{ij} x_j + b_i

The weight matrix determines the linear transformation. Its properties include:
- **Rank**: Maximum number of linearly independent rows/columns. A low-rank weight matrix projects inputs into a lower-dimensional subspace.
- **Norm**: The magnitude of weights (e.g., Frobenius norm) affects gradient magnitudes and is regularized via weight decay.
- **Condition number**: The ratio of largest to smallest singular values affects optimization difficulty.

During backpropagation:

∂L/∂**W** = (∂L/∂**z**) **x**^T

This is an outer product of the output gradient and the input activation.

## Code Examples

### Example 1: Accessing weight matrix shape and values

```python
import torch
import torch.nn as nn

layer = nn.Linear(4, 3)
print("Weight shape:", layer.weight.shape)  # (3, 4)
print("Weight matrix:\n", layer.weight.data)
# Output:
# Weight shape: torch.Size([3, 4])
# Weight matrix:
#  tensor([[ 0.1234, -0.0567,  0.0890, -0.1123],
#          [ 0.2345,  0.3456, -0.4567,  0.5678],
#          [-0.1234,  0.2345, -0.3456, -0.4567]])
```

### Example 2: Manual matrix multiplication vs nn.Linear

```python
x = torch.randn(2, 4)
layer = nn.Linear(4, 3, bias=False)

y1 = layer(x)
y2 = torch.matmul(x, layer.weight.T)  # x @ W^T

print("Are they equal?", torch.allclose(y1, y2))
# Output:
# Are they equal? True
```

### Example 3: Weight initialization strategies

```python
import torch.nn as nn
import torch.nn.init as init

layer = nn.Linear(128, 64)

# Xavier/Glorot uniform (good for tanh/sigmoid)
init.xavier_uniform_(layer.weight)

# Kaiming/He uniform (good for ReLU)
init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')

# Orthogonal initialization (good for RNNs)
init.orthogonal_(layer.weight, gain=1.0)

# Sparse initialization
init.sparse_(layer.weight, sparsity=0.9, std=0.01)

print("Weight mean:", layer.weight.mean().item())
print("Weight std:", layer.weight.std().item())
# Output:
# Weight mean: -0.0012
# Weight std: 0.0876
```

### Example 4: Monitoring weight matrix norm during training

```python
import torch.nn as nn

model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001)

x = torch.randn(32, 10)
y = torch.randn(32, 1)

# Training step
loss = ((model(x) - y) ** 2).mean()
loss.backward()
optimizer.step()

# Check weight norms
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"{name} norm: {param.norm().item():.4f}")
# Output:
# 0.weight norm: 1.2345
# 1.weight norm: 0.6789
```

### Example 5: Visualizing weight patterns

```python
import torch.nn as nn
import matplotlib.pyplot as plt

# Simulate a learned weight matrix
layer = nn.Linear(20, 20)
with torch.no_grad():
    # Create a structured pattern: diagonal with noise
    layer.weight.copy_(torch.eye(20) * 0.9 + torch.randn(20, 20) * 0.1)

# Plot as heatmap
weights = layer.weight.detach().numpy()
print("Weight matrix shape:", weights.shape)
print("Max weight:", weights.max())
print("Min weight:", weights.min())
print("Sparsity (|w| < 0.01):", (abs(weights) < 0.01).mean())
# Output:
# Weight matrix shape: (20, 20)
# Max weight: 1.0912
# Min weight: -0.2134
# Sparsity (|w| < 0.01): 0.0025
```

## Common Mistakes

1. **Confusing weight matrix orientation**: PyTorch's `nn.Linear` stores weights as (out_features, in_features) and multiplies as `x @ W.T`. Many textbooks define W as (in_features, out_features).

2. **Initializing weights to zero**: If all weights are zero, every neuron computes the same output and gradients are identical — symmetry breaking never happens. Always use random initialization.

3. **Using the wrong initialization for the activation function**: Xavier/Glorot for tanh/sigmoid, Kaiming/He for ReLU. Using the wrong one can cause vanishing or exploding activations.

4. **Not normalizing weight updates**: Without normalization or careful learning rates, weight updates can cause the weight norm to grow or shrink unstably.

5. **Forgetting that weight matrices in conv layers are 4D**: Conv2d weight shape is (out_channels, in_channels, kernel_h, kernel_w), not a 2D matrix.

6. **Applying weight decay to bias parameters**: Bias should generally not be regularized. PyTorch's optimizers handle this when `weight_decay` is passed to parameter groups.

7. **Overlooking the weight gradient direction**: The gradient ∂L/∂W is an outer product. A common implementation bug is using element-wise multiplication instead of the outer product.

## Interview Questions

### Beginner - 5

1. What is the shape of the weight matrix in `nn.Linear(100, 200)`?
2. How does the weight matrix transform the input?
3. Why can't you initialize all weights to zero?
4. How many weight parameters does a dense layer from 50 to 30 have?
5. What is the difference between weights and biases?

### Intermediate - 5

1. Derive the gradient of the loss with respect to the weight matrix.
2. Explain why the gradient ∂L/∂W is an outer product of the upstream gradient and the input.
3. Compare Xavier and Kaiming initialization and explain when to use each.
4. What is weight decay and how does it affect the weight matrix during training?
5. How does the rank of the weight matrix affect the representational capacity of a layer?

### Advanced - 3

1. Explain the spectral norm of a weight matrix and how spectral normalization controls the Lipschitz constant of the network.
2. Derive the update rule for a weight matrix under natural gradient descent.
3. How can you impose orthogonality constraints on weight matrices during training, and why would you?

## Practice Problems

### Easy - 5

1. Create an `nn.Linear(5, 3)` and print the weight matrix shape and values.
2. Write code to manually compute the forward pass of a linear layer using `torch.matmul`.
3. Initialize the weight matrix of a linear layer with `torch.ones` and compute output for a random input.
4. Count the total weight parameters in a 3-layer MLP: 784->256->128->10.
5. Extract and print the weight matrix from a trained `nn.Linear` layer.

### Medium - 5

1. Implement a function that computes the effective rank (number of singular values above a threshold) of a weight matrix.
2. Compare training dynamics of a network initialized with Kaiming vs. Xavier init on MNIST classification.
3. Implement weight normalization (normalize each row of the weight matrix to unit norm) and compare with standard linear layer.
4. Write code to visualize the weight matrix as a heatmap before and after training.
5. Implement a weight matrix that is constrained to be lower-triangular (causal) and verify forward and backward passes.

### Hard - 3

1. Implement spectral normalization for a weight matrix and train a GAN discriminator with it.
2. Derive and implement a method to maintain the orthogonality of a weight matrix during SGD updates.
3. Build a linear layer where the weight matrix is parameterized as a product of low-rank matrices and learnable scaling factors. Compare parameter efficiency.

## Solutions

### Easy - 1
```python
layer = nn.Linear(5, 3)
print("Shape:", layer.weight.shape)  # (3, 5)
print("Weights:\n", layer.weight)
```

### Easy - 2
```python
x = torch.randn(1, 5)
W = torch.randn(3, 5)
b = torch.randn(3)
y = torch.matmul(x, W.T) + b
```

### Easy - 3
```python
layer = nn.Linear(5, 3)
nn.init.ones_(layer.weight)
nn.init.zeros_(layer.bias)
x = torch.randn(2, 5)
print(layer(x))
```

## Related Concepts

DL-031 Dense / Fully Connected Layer, DL-033 Bias Term, DL-012 Matrix Multiplication, DL-014 Parameter

## Next Concepts

DL-035 Neuron Computation, DL-046 Forward Pass Computation

## Summary

The weight matrix is the collection of all learnable connection strengths between the inputs and outputs of a neural network layer. Its shape, initialization, and update dynamics determine how the network learns and generalizes. Understanding weight matrices is fundamental to all of deep learning.

## Key Takeaways

- Weight matrix shape: (out_features, in_features) for `nn.Linear` in PyTorch
- Forward: **y** = **xW**^T + **b**
- Backward: ∂L/∂**W** = **x**^T ∂L/∂**y**
- Proper initialization is critical for trainability
- Weight norm and rank affect optimization and capacity
- Weight decay applies only to weights (not biases) in standard practice
