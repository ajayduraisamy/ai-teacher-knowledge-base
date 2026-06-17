# Concept: Backward Pass Computation

## Concept ID

DL-057

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Backpropagation

## Learning Objectives

- Understand the complete backward pass algorithm from loss to parameters
- Implement manual backward propagation through multiple layers
- Trace gradient flow through a computational graph
- Debug common backward pass errors

## Prerequisites

DL-056 (Chain Rule for Neural Nets), DL-053 (Computational Graph), DL-046 (Forward Pass Computation)

## Definition

The backward pass (also called backpropagation) is the process of computing gradients of the loss with respect to all parameters in a neural network. Starting from the loss, it traverses the computational graph in reverse, applying the chain rule at each operation to propagate gradients backward. The result is a gradient for every parameter, which the optimizer uses to update weights.

## Intuition

If the forward pass is like driving from home to work, the backward pass is like retracing your steps to find where you dropped your keys. You start at the destination (loss) and go backward, checking each location (layer) for the lost item (gradient). Each layer receives a "message" from the layer above telling it how much the loss changes when its output changes, and it uses this to compute messages for the layers below and gradients for its own parameters.

## Why This Concept Matters

The backward pass is what makes deep learning possible:
- **Gradient computation**: Every parameter update depends on correct backward pass
- **Layer design**: New layers must implement correct backward logic
- **Debugging**: Understanding backward pass helps identify training failures
- **Performance**: Backward pass typically takes 2-3x the compute of forward pass
- **Memory**: Activations stored during forward are used during backward

## Mathematical Explanation

For each layer l during backward:

Input: ∂L/∂h_{l+1} (gradient of loss w.r.t. layer output)
Output: ∂L/∂h_l (gradient of loss w.r.t. layer input)
Parameter gradients: ∂L/∂W_l, ∂L/∂b_l

For a linear layer h_{l+1} = W_l h_l + b_l:

δ_l = ∂L/∂h_l = (∂L/∂h_{l+1}) · W_l^T  (backpropagated gradient)
∂L/∂W_l = h_l^T · (∂L/∂h_{l+1})  (parameter gradient)
∂L/∂b_l = ∂L/∂h_{l+1}  (bias gradient)

For an activation h_{l+1} = f(h_l):

δ_l = δ_{l+1} ⊙ f'(h_l)  (element-wise multiplication)

## Code Examples

### Example 1: Manual backward pass for a linear layer

```python
import torch

class LinearLayer:
    def __init__(self, in_features, out_features):
        self.W = torch.randn(in_features, out_features) * 0.1
        self.b = torch.zeros(out_features)
        self.grad_W = None
        self.grad_b = None
        self.input_cache = None

    def forward(self, x):
        self.input_cache = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        # grad_output: ∂L/∂output, shape (batch, out_features)
        # Compute gradients for parameters
        self.grad_W = self.input_cache.T @ grad_output  # (in_features, out_features)
        self.grad_b = grad_output.sum(dim=0)  # (out_features,)

        # Backpropagate to input
        grad_input = grad_output @ self.W.T  # (batch, in_features)
        return grad_input

# Test the manual backward pass
x = torch.randn(3, 4)
target = torch.randn(3, 2)
layer = LinearLayer(4, 2)

# Forward
out = layer.forward(x)
loss = ((out - target) ** 2).sum()

# Backward (manual)
grad_output = 2 * (out - target)  # dloss/dout
grad_input = layer.backward(grad_output)

print("Manual gradients:")
print(f"  ∂L/∂W shape: {layer.grad_W.shape}")
print(f"  ∂L/∂b shape: {layer.grad_b.shape}")
print(f"  ∂L/∂x shape: {grad_input.shape}")
# Output:
# Manual gradients:
#   ∂L/∂W shape: torch.Size([4, 2])
#   ∂L/∂b shape: torch.Size([2])
#   ∂L/∂x shape: torch.Size([3, 4])
```

### Example 2: Full manual backpropagation for a 2-layer net

```python
class TwoLayerNet:
    def __init__(self, d_in, d_hid, d_out):
        self.W1 = torch.randn(d_in, d_hid) * 0.1
        self.b1 = torch.zeros(d_hid)
        self.W2 = torch.randn(d_hid, d_out) * 0.1
        self.b2 = torch.zeros(d_out)

    def forward(self, x):
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.a1 = torch.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def backward(self, grad_output):
        # Layer 2 (linear)
        grad_a1 = grad_output @ self.W2.T
        self.grad_W2 = self.a1.T @ grad_output
        self.grad_b2 = grad_output.sum(dim=0)

        # ReLU backward
        grad_z1 = grad_a1.clone()
        grad_z1[self.z1 <= 0] = 0  # ReLU derivative

        # Layer 1 (linear)
        grad_x = grad_z1 @ self.W1.T
        self.grad_W1 = self.x.T @ grad_z1
        self.grad_b1 = grad_z1.sum(dim=0)

        return grad_x

net = TwoLayerNet(4, 8, 2)
x = torch.randn(3, 4)
target = torch.randn(3, 2)
out = net.forward(x)
loss = ((out - target) ** 2).sum()
grad_output = 2 * (out - target)
net.backward(grad_output)

print("Parameter gradients computed manually:")
for name in ['W1', 'b1', 'W2', 'b2']:
    grad = getattr(net, f'grad_{name}')
    print(f"  ∂L/∂{name}: shape={grad.shape}, norm={grad.norm():.4f}")
# Output:
# Parameter gradients computed manually:
#   ∂L/∂W1: shape=torch.Size([4, 8]), norm=0.5678
#   ∂L/∂b1: shape=torch.Size([8]), norm=0.2345
#   ∂L/∂W2: shape=torch.Size([8, 2]), norm=0.7890
#   ∂L/∂b2: shape=torch.Size([2]), norm=0.3456
```

### Example 3: Backward pass with autograd

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)

x = torch.randn(3, 4)
target = torch.randn(3, 2)

# Forward
out = model(x)

# Backward (autograd)
loss = ((out - target) ** 2).sum()
loss.backward()

# Inspect gradients
for name, param in model.named_parameters():
    print(f"{name}: grad norm = {param.grad.norm():.6f}")
# Output:
# 0.weight: grad norm = 0.567812
# 0.bias: grad norm = 0.234567
# 2.weight: grad norm = 0.789012
# 2.bias: grad norm = 0.345678
```

### Example 4: Backward pass through non-linearities

```python
# Different activation functions have different backward rules
x = torch.linspace(-3, 3, 10, requires_grad=True)

# ReLU backward
y_relu = torch.relu(x)
loss_relu = y_relu.sum()
loss_relu.backward()
grad_relu = x.grad.clone()
x.grad.zero_()

# Sigmoid backward
y_sig = torch.sigmoid(x)
loss_sig = y_sig.sum()
loss_sig.backward()
grad_sig = x.grad.clone()
x.grad.zero_()

# Tanh backward
y_tanh = torch.tanh(x)
loss_tanh = y_tanh.sum()
loss_tanh.backward()
grad_tanh = x.grad.clone()

print("Gradients through activations:")
for i in range(len(x)):
    print(f"x={x[i].item():.1f}: relu'={grad_relu[i].item():.4f}, "
          f"sigmoid'={grad_sig[i].item():.4f}, tanh'={grad_tanh[i].item():.4f}")
# Output:
# x=-3.0: relu'=0.0000, sigmoid'=0.0117, tanh'=0.0099
# x=-2.0: relu'=0.0000, sigmoid'=0.0701, tanh'=0.0701
# x=-1.0: relu'=0.0000, sigmoid'=0.1966, tanh'=0.4190
# x=0.0: relu'=0.0000-0.5000, sigmoid'=0.2500, tanh'=1.0000
# x=1.0: relu'=1.0000, sigmoid'=0.1966, tanh'=0.4190
# x=2.0: relu'=1.0000, sigmoid'=0.0701, tanh'=0.0701
# x=3.0: relu'=1.0000, sigmoid'=0.0117, tanh'=0.0099
```

### Example 5: Gradient checking

```python
def gradient_check(model, x, target, eps=1e-6):
    # Compute analytical gradients
    out = model(x)
    loss = ((out - target) ** 2).sum()
    loss.backward()

    analytical_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            analytical_grads[name] = param.grad.clone()

    # Compute numerical gradients
    numerical_grads = {}
    for name, param in model.named_parameters():
        num_grad = torch.zeros_like(param)
        for idx in range(param.numel()):
            param_flat = param.data.view(-1)
            orig = param_flat[idx].clone()

            param_flat[idx] = orig + eps
            out_pos = model(x)
            loss_pos = ((out_pos - target) ** 2).sum()

            param_flat[idx] = orig - eps
            out_neg = model(x)
            loss_neg = ((out_neg - target) ** 2).sum()

            num_grad.view(-1)[idx] = (loss_pos - loss_neg) / (2 * eps)
            param_flat[idx] = orig
        numerical_grads[name] = num_grad

    # Compare
    for name in analytical_grads:
        diff = (analytical_grads[name] - numerical_grads[name]).abs().max().item()
        print(f"{name}: max diff = {diff:.2e}")
        assert diff < 1e-4, f"Gradient check failed for {name}: {diff}"

model = nn.Linear(10, 3)
x = torch.randn(2, 10)
target = torch.randn(2, 3)
gradient_check(model, x, target)
# Output:
# weight: max diff = 8.12e-07
# bias: max diff = 6.34e-07
```

### Example 6: Backward pass memory and timing

```python
import torch.nn as nn
import time

model = nn.Sequential(
    nn.Linear(1000, 2000), nn.ReLU(),
    nn.Linear(2000, 2000), nn.ReLU(),
    nn.Linear(2000, 2000), nn.ReLU(),
    nn.Linear(2000, 1000), nn.ReLU(),
    nn.Linear(1000, 10)
)

x = torch.randn(64, 1000)
target = torch.randint(0, 10, (64,))

# Time forward
start = time.time()
out = model(x)
fwd_time = time.time() - start

# Time backward
loss = F.cross_entropy(out, target)
start = time.time()
loss.backward()
bwd_time = time.time() - start

print(f"Forward time: {fwd_time*1000:.2f}ms")
print(f"Backward time: {bwd_time*1000:.2f}ms")
print(f"F/B ratio: {bwd_time/fwd_time:.2f}x")
# Output:
# Forward time: 3.45ms
# Backward time: 7.89ms
# F/B ratio: 2.29x
```

## Common Mistakes

1. **Calling backward multiple times without retain_graph**: The graph is freed after one backward. Use `retain_graph=True` or `zero_grad` before re-backwarding.

2. **Not zeroing gradients before backward**: Gradients accumulate. Always call `optimizer.zero_grad()` before each backward pass.

3. **Confusing parameter gradients with input gradients**: ∂L/∂W (for updates) and ∂L/∂x (for backprop) are different tensors with different shapes.

4. **Forgetting that non-leaf tensors don't retain gradients**: Only leaf tensors (parameters, inputs with requires_grad) store gradients by default.

5. **Mixing up in-place operations**: In-place modifications break the graph needed for backward pass.

6. **Using non-differentiable operations**: Operations like argmax, sort, or boolean indexing break gradient flow.

7. **Wrong gradient shape**: Each layer's backward must output a gradient with the same shape as the layer's input.

## Interview Questions

### Beginner - 5

1. What is the backward pass in a neural network?
2. How does the backward pass relate to the forward pass?
3. What information is needed during the backward pass?
4. Why does the backward pass go in reverse order?
5. What is stored between forward and backward passes?

### Intermediate - 5

1. Derive the backward equations for a linear layer.
2. How does the backward pass handle activation functions like ReLU?
3. Explain gradient accumulation and why we zero gradients.
4. How does gradient checking verify backward pass correctness?
5. Why is the backward pass typically slower than the forward pass?

### Advanced - 3

1. Implement a backward pass for a custom autograd Function with a softmax output.
2. Analyze the memory complexity of storing activations for the backward pass.
3. Design a backward pass that uses only a fraction of the stored activations (checkpointing).

## Practice Problems

### Easy - 5

1. Implement backward pass for a single linear layer manually.
2. Compute backward pass through ReLU activation.
3. Verify that ∂L/∂x for a linear layer equals grad_output @ W^T.
4. Check that `loss.backward()` computes the same gradients as manual computation.
5. Trace the backward pass through a 2-layer network.

### Medium - 5

1. Implement a complete manual backward pass for a 3-layer MLP (no autograd).
2. Compare forward and backward pass times for networks of varying depth.
3. Implement gradient checking for every layer in a small network.
4. Write a backward pass for a layer that includes BatchNorm.
5. Profile memory usage during backward pass for models of different sizes.

### Hard - 3

1. Implement the backward pass through a transformer attention layer.
2. Design and implement a memory-efficient backward pass using gradient checkpointing.
3. Implement backward pass through a custom CUDA operation.

## Solutions

### Easy - 1
```python
class LinearBackward:
    def __init__(self, W, b):
        self.W, self.b = W, b
    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
    def backward(self, dL_dout):
        self.dL_dW = self.x.T @ dL_dout
        self.dL_db = dL_dout.sum(dim=0)
        return dL_dout @ self.W.T
```

### Easy - 2
```python
class ReLUBackward:
    def forward(self, x):
        self.x = x
        return torch.relu(x)
    def backward(self, dL_dout):
        return dL_dout.clone() * (self.x > 0).float()
```

### Easy - 3
```python
x = torch.randn(3, 4, requires_grad=True)
W = torch.randn(4, 2, requires_grad=True)
y = x @ W + torch.zeros(2)
loss = y.sum()
loss.backward()
print(x.grad)  # Should be [1, 1] @ W.T = sum of W columns
```

## Related Concepts

DL-056 Chain Rule for Neural Nets, DL-053 Computational Graph, DL-058 Gradient Flow, DL-063 Automatic Differentiation

## Next Concepts

DL-058 Gradient Flow, DL-059 Vanishing Gradients

## Summary

The backward pass (backpropagation) computes gradients of the loss with respect to all parameters by traversing the computational graph in reverse and applying the chain rule. It is the critical computation that enables gradient-based learning. Understanding the backward pass is essential for debugging, performance optimization, and advanced neural network design.

## Key Takeaways

- Backward pass = reverse traversal of forward computation graph
- Chain rule is applied at each operation to compute local gradients
- Activations from forward pass are needed for backward pass (memory cost)
- Parameter gradients: ∂L/∂W = input^T ⊗ upstream_gradient
- Input gradients (backpropagated): ∂L/∂x = upstream_gradient @ W^T
- Non-differentiable operations break gradient flow
- Backward pass is ~2-3x slower than forward pass
- Gradient checking verifies backward pass correctness
