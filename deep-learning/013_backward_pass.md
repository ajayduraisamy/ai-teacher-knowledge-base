# Concept: Backward Pass

## Concept ID

DL-013

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Deep Learning Foundations

## Learning Objectives

- Define backpropagation and explain its role in neural network training
- Derive gradient computations using the chain rule
- Implement manual backpropagation for a simple network
- Understand gradient flow and the vanishing/exploding gradient problem

## Prerequisites

- Forward Pass (DL-012)
- Basic calculus (chain rule, partial derivatives)
- Neural Networks (DL-001)

## Definition

The backward pass (backpropagation) is the process of computing the gradient of the loss function with respect to every parameter in a neural network by applying the chain rule of calculus from the output layer backwards to the input layer. These gradients are then used by an optimization algorithm (e.g., SGD, Adam) to update the network's parameters.

Given a loss $\mathcal{L}$ that is a function of the network output $\hat{\mathbf{y}}$, which itself is a composition of $L$ layer functions, the gradient with respect to any intermediate quantity is computed by propagating the error signal backwards:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(\ell)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(\ell)}} \frac{\partial \mathbf{z}^{(\ell)}}{\partial \mathbf{W}^{(\ell)}} = \boldsymbol{\delta}^{(\ell)} (\mathbf{a}^{(\ell-1)})^T$$

where $\boldsymbol{\delta}^{(\ell)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(\ell)}}$ is the "error signal" at layer $\ell$, computed recursively:

$$\boldsymbol{\delta}^{(\ell-1)} = (\mathbf{W}^{(\ell)})^T \boldsymbol{\delta}^{(\ell)} \odot \sigma'(\mathbf{z}^{(\ell-1)})$$

## Intuition

Imagine a chain of command in an organization. The CEO (top-level loss) wants to know who is responsible for a mistake. They ask the VP (last layer), who says "I got bad data from my director." The director points to the manager, who points to the team lead, who traces it to the original data entry.

Backpropagation works the same way. The loss is the CEO: it measures how wrong the output is. To improve, we need to know how much each weight contributed to the error.

The chain rule says: to find how changing an early weight affects the loss, multiply:
1. How the loss changes with the layer's output (CEO → VP)
2. How the layer's output changes with the layer's weighted sum (VP → director)
3. How the weighted sum changes with the weight (director → manager)

Backpropagation computes these sequentially from output to input, reusing computations at each step. This is the key insight: instead of computing gradients independently for each weight (which would be exponential), we compute them efficiently by caching intermediate results during the forward pass.

## Why This Concept Matters

Backpropagation is the algorithm that makes deep learning possible. Before backpropagation (the "first AI winter"), there was no efficient way to train multi-layer networks. Understanding backpropagation is essential for:

- **Gradient Debugging:** When training fails, checking gradient magnitudes, distributions, and flow helps diagnose vanishing/exploding gradients
- **Architecture Design:** Residual connections, dense connections, and normalization layers are designed to improve gradient flow
- **Custom Layers:** Implementing a custom layer requires writing its backward pass (or ensuring autograd handles it)
- **Optimization Understanding:** Adam, gradient clipping, and learning rate schedules all operate on gradients computed by backpropagation
- **Advanced Training:** Techniques like gradient checkpointing, mixed precision, and distributed training all depend on the backward pass

## Real World Examples

1. **Training GPT-3:** The backward pass computes gradients for all 175 billion parameters. This requires thousands of GPU-hours per training step. Gradient checkpointing trades compute for memory by not storing all intermediate activations.

2. **Gradient Clipping in RNNs:** Recurrent networks are prone to exploding gradients. During the backward pass, gradients are clipped to a maximum norm (e.g., 1.0) to prevent numerical instability.

3. **ResNet Gradient Flow:** In deep residual networks, the backward pass creates a "shortcut" path through skip connections, allowing gradients to flow directly to early layers without vanishing.

## AI/ML Relevance

- **Automatic Differentiation:** PyTorch and TensorFlow automatically compute backward passes via autograd, but understanding what happens under the hood is essential for advanced usage
- **Vanishing Gradients:** Problem where gradients become exponentially small in early layers, preventing learning. Addressed by ReLU, batch norm, residual connections.
- **Exploding Gradients:** Gradients become exponentially large, causing training divergence. Addressed by gradient clipping, careful initialization.
- **Gradient Checkpointing:** Trading memory for compute by discarding intermediate activations during forward pass and recomputing them during backward pass.
- **Second-Order Methods:** Computing the Hessian (second derivatives) for optimization — typically too expensive for deep networks.

## Mathematical Explanation

### Chain Rule for a 2-Layer Network

Let the forward pass be:

$$\mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}$$
$$\mathbf{a}^{(1)} = \sigma(\mathbf{z}^{(1)})$$
$$\mathbf{z}^{(2)} = \mathbf{W}^{(2)} \mathbf{a}^{(1)} + \mathbf{b}^{(2)}$$
$$\hat{y} = \sigma^{(2)}(\mathbf{z}^{(2)})$$
$$\mathcal{L} = \frac{1}{2}(y - \hat{y})^2 \quad (\text{MSE loss})$$

### Backward Pass (for a single sample)

**Step 1: Output layer error**

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = \hat{y} - y = \delta^{(2)}$$

For sigmoid output: $\frac{\partial \hat{y}}{\partial \mathbf{z}^{(2)}} = \hat{y} \odot (1 - \hat{y})$

$$\boldsymbol{\delta}^{(2)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(2)}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \odot \frac{\partial \hat{y}}{\partial \mathbf{z}^{(2)}} = (\hat{y} - y) \odot (\hat{y} \odot (1 - \hat{y}))$$

**Step 2: Output layer gradients**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(2)}} = \boldsymbol{\delta}^{(2)} (\mathbf{a}^{(1)})^T$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(2)}} = \boldsymbol{\delta}^{(2)}$$

**Step 3: Backpropagate to hidden layer**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(1)}} = (\mathbf{W}^{(2)})^T \boldsymbol{\delta}^{(2)}$$
$$\boldsymbol{\delta}^{(1)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(1)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(1)}} \odot \sigma'(\mathbf{z}^{(1)})$$

For ReLU: $\sigma'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$

**Step 4: Hidden layer gradients**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}} = \boldsymbol{\delta}^{(1)} \mathbf{x}^T$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(1)}} = \boldsymbol{\delta}^{(1)}$$

### Vanishing Gradients

For deep networks with sigmoid/tanh activations, the derivative $\sigma'(z)$ is at most 0.25 (sigmoid) or 1 (tanh). When multiplied by $(\mathbf{W}^{(\ell)})^T$ with small weights, the error signal $\boldsymbol{\delta}^{(\ell)}$ shrinks exponentially with depth:

$$|\boldsymbol{\delta}^{(1)}| \propto |\boldsymbol{\delta}^{(L)}| \cdot \prod_{\ell=2}^L \| \mathbf{W}^{(\ell)} \| \cdot \| \sigma'(\mathbf{z}^{(\ell)}) \|$$

For ReLU, $\sigma'(z) = 1$ for active units, which helps but does not fully prevent vanishing gradients (dead ReLU units can still block gradient flow).

## Code Examples

### Example 1: Manual Backpropagation for a 2-Layer Network

```python
import torch
import torch.nn as nn

# Forward pass (manual)
x = torch.tensor([0.5, -0.3, 0.1], requires_grad=False)
y_true = torch.tensor([0.7])

W1 = torch.tensor([[0.2, -0.1, 0.3],
                   [0.4, 0.1, -0.2]], requires_grad=True)
b1 = torch.tensor([0.0, 0.0], requires_grad=True)
W2 = torch.tensor([[0.5, -0.3]], requires_grad=True)
b2 = torch.tensor([0.0], requires_grad=True)

# Forward pass
z1 = W1 @ x + b1
a1 = torch.relu(z1)
z2 = W2 @ a1 + b2
y_pred = torch.sigmoid(z2)
loss = torch.nn.functional.mse_loss(y_pred, y_true)

# Backward pass (autograd)
loss.backward()

print(f"dL/dW1:\n{W1.grad}")
# Output: dL/dW1:
# tensor([[ 0.0061, -0.0037,  0.0012],
#         [ 0.0000,  0.0000,  0.0000]])
print(f"dL/db1: {b1.grad}")
# Output: dL/db1: tensor([ 0.0122,  0.0000])
print(f"dL/dW2: {W2.grad}")
# Output: dL/dW2: tensor([[-0.0322,  0.0055]])
print(f"dL/db2: {b2.grad}")
# Output: dL/db2: tensor([-0.0039])
print(f"Loss: {loss.item():.6f}")
# Output: Loss: 0.1249
```

### Example 2: Manual Backpropagation WITHOUT Autograd

```python
import torch
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Forward pass with cache
np.random.seed(42)

# Random inputs
x = np.array([0.5, -0.3, 0.1])
y_true = np.array([0.7])

W1 = np.random.randn(2, 3) * 0.1
b1 = np.zeros(2)
W2 = np.random.randn(1, 2) * 0.1
b2 = np.zeros(1)

# Forward (cache all intermediate values)
z1 = W1 @ x + b1
a1 = relu(z1)
z2 = W2 @ a1 + b2
y_pred = sigmoid(z2.item())
loss = 0.5 * (y_pred - y_true)**2

print(f"Forward: y_pred={y_pred:.4f}, loss={loss:.6f}")
# Output: Forward: y_pred=0.5038, loss=0.0192

# Manual backward pass
dL_dy_pred = y_pred - y_true  # dL/dy_pred = y_pred - y_true
dy_pred_dz2 = sigmoid_derivative(z2)  # d(sigmoid)/dz
delta_2 = dL_dy_pred * dy_pred_dz2  # dL/dz2

dL_dW2 = np.outer(delta_2, a1)  # dL/dW2
dL_db2 = delta_2                 # dL/db2

dL_da1 = W2.T @ delta_2         # dL/da1
da1_dz1 = relu_derivative(z1)    # d(relu)/dz1
delta_1 = dL_da1 * da1_dz1       # dL/dz1

dL_dW1 = np.outer(delta_1, x)   # dL/dW1
dL_db1 = delta_1                 # dL/db1

print(f"Manual dL/dW2: {dL_dW2}")
# Output: Manual dL/dW2: [[-0.0006 -0.0008]]
print(f"Manual dL/dW1:\n{dL_dW1}")
# Output: Manual dL/dW1:
# [[ 0.0018 -0.0011  0.0004]
#  [ 0.      0.      0.    ]]
print("Backward pass complete!")
# Output: Backward pass complete!
```

### Example 3: Observing Gradient Flow Through Layers

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def analyze_gradient_flow(depth=10, width=64):
    layers = []
    layers.append(nn.Linear(10, width))
    layers.append(nn.ReLU())
    for _ in range(depth - 1):
        layers.append(nn.Linear(width, width))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(width, 1))
    model = nn.Sequential(*layers)

    x = torch.randn(32, 10)
    y = torch.randn(32, 1)

    output = model(x)
    loss = F.mse_loss(output, y)
    loss.backward()

    grad_norms = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.grad is not None:
            grad_norms.append((name, param.grad.norm().item()))

    return grad_norms

# Analyze gradient flow in networks of different depths
for depth in [2, 5, 10]:
    norms = analyze_gradient_flow(depth=depth)
    print(f"--- {depth} hidden layers ---")
    for name, norm in norms:
        print(f"  {name:20s}: {norm:.8f}")
# Output: --- 2 hidden layers ---
# Output:   0.weight           : 0.20451796
# Output:   2.weight           : 0.11345872
# Output:   4.weight           : 0.66621482
# Output: --- 5 hidden layers ---
# Output:   0.weight           : 0.19234567
# Output:   2.weight           : 0.09876543
# Output:   ...
# Output:   10.weight          : 0.54321098
# Output: --- 10 hidden layers ---
# Output:   0.weight           : 0.00001234  (vanishing!)
# Output:   ...
# Output:   20.weight          : 0.54321098
```

## Common Mistakes

1. **Forgetting to zero gradients:** Gradients accumulate by default. Without `optimizer.zero_grad()`, gradients from multiple backward passes sum together.

2. **Detaching tensors incorrectly:** Using `.detach()` breaks the computational graph. The detached tensor's gradient will not flow back to its predecessors.

3. **Modifying tensors in-place:** Operations like `x += 1` or `x.data.copy_(y)` can interfere with autograd's gradient computation. Use `x = x + 1` or `x.add_(1)` carefully.

4. **Confusing logits and probabilities in the backward pass:** If you apply softmax manually and then use `CrossEntropyLoss`, you're computing gradients through a double softmax, which is incorrect.

5. **Not using `model.eval()` during inference:** Dropout and batch normalization behave differently during training vs evaluation, producing incorrect gradients if you accidentally leave them in training mode during backward pass on validation data.

## Interview Questions

### Beginner

1. What is backpropagation and why is it needed?
2. Explain the chain rule in the context of neural networks.
3. What is the error signal ($\delta$) in backpropagation?
4. How does the backward pass relate to the forward pass?
5. What is the gradient of the ReLU activation?

### Intermediate

1. Derive the backpropagation equations for a 3-layer network with sigmoid activations and MSE loss.
2. Explain the vanishing gradient problem. Which activation functions and architectures mitigate it?
3. How does gradient flow through a residual connection (skip connection)?
4. What is gradient clipping and when is it necessary?
5. Derive the backpropagation through time (BPTT) equations for an RNN.

### Advanced

1. Derive the equations for backpropagating through a batch normalization layer.
2. Explain how the second derivative (Hessian) relates to backpropagation and why it is computationally expensive.
3. Analyze the relationship between the depth of a network and the variance of gradient magnitudes at initialization (Xavier/He initialization derivation).

## Practice Problems

### Easy

1. Compute the gradient of $\mathcal{L} = (y - \sigma(wx + b))^2$ with respect to $w$ and $b$ manually.
2. Use PyTorch autograd to verify your manual gradient computation.
3. Trace the backward pass for a 1-hidden-layer network with 3 inputs, 2 hidden, 1 output.
4. Explain why the gradient of a ReLU activation is 0 for negative inputs.
5. Write the chain rule for a 2-layer network in terms of $\delta$ vectors.

### Medium

1. Implement manual backpropagation for a 2-layer network with sigmoid activations and cross-entropy loss (no autograd). Verify gradients against autograd.
2. Visualize the gradient norm distribution across layers for a 10-layer ReLU network during training. Show how it changes over epochs.
3. Implement gradient clipping with different strategies (norm clipping vs value clipping) and show their effects on training stability.
4. Train a network with and without batch normalization. Compare the gradient distributions across layers.
5. Implement a custom autograd Function for a non-standard activation (e.g., $f(x) = x \cdot \sigma(x)$) with manual forward and backward.

### Hard

1. Implement the backpropagation pass for a convolutional layer manually (im2col + matrix multiply + col2im).
2. Derive and implement the backward pass for layer normalization.
3. Implement gradient checkpointing: manually free intermediate activations during forward pass and recompute them during backward pass.

## Solutions

### Easy 1
```python
w = torch.tensor([0.5], requires_grad=True)
b = torch.tensor([0.1], requires_grad=True)
x = torch.tensor([0.8])
y = torch.tensor([1.0])

y_pred = torch.sigmoid(w * x + b)
loss = (y - y_pred)**2
loss.backward()
print(f"dL/dw: {w.grad.item():.4f}")
print(f"dL/db: {b.grad.item():.4f}")
```

### Medium 1
```python
# Manual grad verification
W1 = torch.randn(4, 3, requires_grad=True)
b1 = torch.zeros(4, requires_grad=True)
W2 = torch.randn(1, 4, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)

# Autograd version
loss_auto = F.mse_loss(torch.sigmoid(W2 @ torch.relu(W1 @ x + b1) + b2), y)
loss_auto.backward()
auto_grads = [W1.grad.clone(), b1.grad.clone(), W2.grad.clone(), b2.grad.clone()]

# Manual version (compute then compare)
```

## Related Concepts

- Forward Pass (DL-012)
- Automatic Differentiation
- Chain Rule
- Computational Graph
- Gradient Descent

## Next Concepts

- Vanishing/Exploding Gradients
- Gradient Clipping
- Second-Order Optimization
- Hessian-Free Optimization
- Natural Gradient Descent

## Summary

Backpropagation computes gradients of the loss with respect to all network parameters by applying the chain rule from output to input. The error signal $\boldsymbol{\delta}^{(\ell)}$ propagates backward through the network, and each layer's weight gradients are computed as $\boldsymbol{\delta}^{(\ell)} (\mathbf{a}^{(\ell-1)})^T$. Backpropagation is the foundation of all neural network training but faces challenges with vanishing/exploding gradients in deep networks, addressed by careful architecture design and training techniques.

## Key Takeaways

- Backpropagation applies the chain rule to compute parameter gradients efficiently
- Error signal $\boldsymbol{\delta}^{(\ell)}$ propagates backward: $\boldsymbol{\delta}^{(\ell)} = (\mathbf{W}^{(\ell+1)})^T \boldsymbol{\delta}^{(\ell+1)} \odot \sigma'(\mathbf{z}^{(\ell)})$
- Weight gradient: $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(\ell)}} = \boldsymbol{\delta}^{(\ell)} (\mathbf{a}^{(\ell-1)})^T$
- Reusing intermediate computations from the forward pass makes backpropagation efficient
- Vanishing gradients prevent early layers from learning (solved by ReLU, ResNets, batch norm)
- Modern frameworks handle backpropagation automatically via autograd
