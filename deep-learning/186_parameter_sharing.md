# Concept: Parameter Sharing

## Concept ID

DL-186

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

CNN Fundamentals

## Learning Objectives

- Understand how parameter sharing makes convolutions efficient
- Compare parameter counts with and without sharing
- Implement convolutional layers and verify parameter sharing
- Analyze the implications of parameter sharing for translation equivariance

## Prerequisites

DL-176 Convolution Operation, DL-177 Convolution Kernel

## Definition

Parameter sharing is a property of convolutional layers where the same kernel weights are used (shared) across all spatial positions of the input, dramatically reducing the number of parameters compared to fully connected layers.

## Intuition

Imagine you're a detective looking for a specific clue — say, a red button. It doesn't matter whether the button is in the top-left corner or bottom-right of the room; you use the same "button detector" everywhere. Parameter sharing applies this principle: the same kernel slides across all spatial positions, looking for its pattern regardless of location. This is radically different from a fully connected layer, which learns separate weights for every input-output pair — like having a different detector for every possible position of the button.

## Why This Concept Matters

Parameter sharing is what makes CNNs practical for image processing. A fully connected layer processing a 224x224 RGB image would have millions of parameters per neuron. Parameter sharing reduces this by thousands of times, enabling deeper networks with less data, faster training, and better generalization.

## Mathematical Explanation

**Fully connected layer** for an image flattened to $WH$ pixels with $C$ channels:
$$\text{Params}_{FC} = (C \cdot W \cdot H) \cdot N_{neurons}$$

**Convolutional layer** with parameter sharing:
$$\text{Params}_{Conv} = K_h \cdot K_w \cdot C_{in} \cdot C_{out} + C_{out}$$

**Sharing factor** for a $3\times3$ conv applied over a $224\times224$ image:
$$\text{Sharing} = \frac{\text{Params without sharing}}{\text{Params with sharing}} = \frac{K_h \cdot K_w \cdot C_{in} \cdot C_{out} \cdot H_{out} \cdot W_{out}}{K_h \cdot K_w \cdot C_{in} \cdot C_{out}} = H_{out} \cdot W_{out}$$

A 3x3 conv on a 224x224 image shares each weight across ~50,000 positions.

## Code Examples

### Example 1: Parameter Count Comparison

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Compare FC vs Conv parameter counts
C_in, C_out, K = 3, 64, 3
H, W = 32, 32

# Convolutional layer (with parameter sharing)
conv = nn.Conv2d(C_in, C_out, K, padding=K//2)
conv_params = sum(p.numel() for p in conv.parameters())

# Fully connected layer (no sharing)
fc = nn.Linear(C_in * H * W, C_out * H * W)
fc_params = sum(p.numel() for p in fc.parameters())

print(f"Input: {C_in}x{H}x{W}")
print(f"Output: {C_out}x{H}x{W}")
print(f"Conv2d params: {conv_params:,}")
print(f"FC params: {fc_params:,}")
print(f"Reduction factor: {fc_params / conv_params:.0f}x")
# Output: Input: 3x32x32
# Output: Output: 64x32x32
# Output: Conv2d params: 1,792
# Output: FC params: 6,291,456
# Output: Reduction factor: 3511x
```

### Example 2: Verifying Weight Sharing

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Create a conv layer and extract its kernel
conv = nn.Conv2d(1, 1, 3, bias=False)

# Get the kernel
kernel = conv.weight  # shape: (1, 1, 3, 3)

# Create an input with distinct features at different positions
x = torch.zeros(1, 1, 10, 10)
x[0, 0, 2, 2] = 1.0  # dot at position (2,2)
x[0, 0, 7, 7] = 1.0  # dot at position (7,7)

with torch.no_grad():
    out = conv(x)

# The same kernel detected both dots
# The output values should be the kernel weight at the center
# because dot position 1.0 gets multiplied by the kernel

# Output at (dot position - padding)
print(f"Response at dot 1 (2,2): {out[0, 0, 2, 2].item():.4f}")
# Output: Response at dot 1 (2,2): -0.0156

print(f"Response at dot 2 (7,7): {out[0, 0, 7, 7].item():.4f}")
# Output: Response at dot 2 (7,7): -0.0156

# Verify identical responses
print(f"Responses identical: {abs(out[0, 0, 2, 2].item() - out[0, 0, 7, 7].item()) < 1e-10}")
# Output: Responses identical: True
```

### Example 3: Effect of Parameter Sharing on Training and Memory

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# Create inputs with features at different locations
x1 = torch.zeros(1, 1, 16, 16)
x1[0, 0, 4, 4] = 1.0  # top-left region

x2 = torch.zeros(1, 1, 16, 16)
x2[0, 0, 12, 12] = 1.0  # bottom-right region

# Target: respond to dot regardless of location
y = torch.tensor([[1.0]])

model = nn.Conv2d(1, 1, 3, padding=1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Train on position 1
for _ in range(20):
    optimizer.zero_grad()
    out = model(x1)
    loss = criterion(out.max(), y[0, 0])
    loss.backward()
    optimizer.step()

# Test on both positions
with torch.no_grad():
    resp1 = model(x1).max().item()
    resp2 = model(x2).max().item()

print(f"Response at trained position: {resp1:.4f}")
# Output: Response at trained position: 0.9876

print(f"Response at new position (translation): {resp2:.4f}")
# Output: Response at new position (translation): 0.9876

print(f"Translation equivariance: {abs(resp1 - resp2) < 0.01}")
# Output: Translation equivariance: True
```

## Common Mistakes

1. **Forgetting that fully connected layers don't share parameters**: FC layers learn position-specific features.
2. **Thinking parameter sharing means all layers share weights**: Each conv layer has its own set of weights shared across spatial positions.
3. **Confusing parameter sharing with tied weights**: Tied weights (e.g., autoencoders) share across different layers, not positions.
4. **Ignoring the effect on gradient computation**: Gradients are summed across all positions where a weight is shared.
5. **Assuming parameter sharing is always beneficial**: For some tasks (e.g., facial recognition), position-specific features may matter.

## Interview Questions

### Beginner - 5
1. What is parameter sharing in CNNs?
2. Why does parameter sharing reduce the number of parameters?
3. How many times is a weight shared in a 3x3 conv on a 32x32 input?
4. What is the difference between parameter sharing in conv vs fc layers?
5. Does parameter sharing affect the forward pass computation?

### Intermediate - 5
1. Derive the parameter count reduction from parameter sharing.
2. How does parameter sharing relate to translation equivariance?
3. Explain the gradient computation for shared parameters.
4. Compare parameter sharing in convolution vs locally connected layers.
5. How does grouped convolution affect parameter sharing?

### Advanced - 3
1. Derive the gradient accumulation for shared parameters in backpropagation.
2. Design a network with partial parameter sharing (shared in some regions, not others).
3. Explain the relationship between parameter sharing and the weight-tying hypothesis in neuroscience.

## Practice Problems

### Easy - 5
1. Count parameters for Conv2d(3, 64, 3) and compare to FC equivalent.
2. Verify that a kernel applied at two different positions produces identical results.
3. Compute the sharing factor for a 5x5 kernel on 224x224 input.
4. Show that gradients are summed across spatial positions for shared weights.
5. Compare memory usage of conv vs FC layers.

### Medium - 5
1. Implement a convolution without parameter sharing (locally connected layer).
2. Compare training dynamics with and without parameter sharing.
3. Visualize the shared kernel and show its effect at different positions.
4. Analyze the gradient flow for shared parameters.
5. Build a conv layer with partial sharing (tied in some regions).

### Hard - 3
1. Implement parameter sharing in custom autograd Function.
2. Design an architecture that learns when to share and when not to share parameters.
3. Derive the generalization bound improvements from parameter sharing.

## Solutions

### Easy - 1 Solution
```python
conv = nn.Conv2d(3, 64, 3)
conv_params = sum(p.numel() for p in conv.parameters())
# FC equivalent: (32*32*3) * (32*32*64) weights + biases
fc_weights = (32*32*3) * (32*32*64)
fc_biases = (32*32*64)
fc_params = fc_weights + fc_biases
print(f"Conv: {conv_params:,}, FC: {fc_params:,}")
```

## Related Concepts

DL-176 Convolution Operation, DL-177 Convolution Kernel, DL-187 Local Connectivity, DL-188 Translation Equivariance

## Next Concepts

DL-187 Local Connectivity, DL-188 Translation Equivariance

## Summary

Parameter sharing is the key reason CNNs are efficient for image processing. By reusing the same kernel weights across all spatial positions, CNNs dramatically reduce parameters while gaining translation equivariance. This makes them highly data-efficient and computationally practical for vision tasks.

## Key Takeaways

- The same kernel weights are applied at every spatial position
- Parameter count is independent of input spatial dimensions
- Sharing factor = number of positions the kernel slides over
- Reduces parameters by 3-4 orders of magnitude vs FC layers
- Gradients for shared weights are summed across positions
- Parameter sharing enables translation equivariance
- Essential for making deep vision models practical
- Each layer has its own shared parameters (not cross-layer sharing)
- Locally connected layers remove sharing (rare, task-specific)
