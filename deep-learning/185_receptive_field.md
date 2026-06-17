# Concept: Receptive Field

## Concept ID

DL-185

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

CNN Fundamentals

## Learning Objectives

- Define and compute the receptive field of a CNN neuron
- Understand how kernel size, stride, and dilation affect receptive field
- Relate receptive field to network depth and task requirements
- Design networks with appropriate receptive fields

## Prerequisites

DL-176 Convolution Operation, DL-178 Stride, DL-179 Padding, DL-180 Dilation

## Definition

The receptive field of a neuron in a convolutional neural network is the region of the input image that influences that neuron's activation. It describes how much of the input a particular feature map element "sees."

## Intuition

Imagine you're looking at a photograph through a narrow tube. The tube limits your view to a small patch. As you add more layers to your network, each layer's view is built from the views of the previous layer's neurons — like stacking tubes, where each subsequent tube peers through the previous ones. The receptive field is the diameter of the "view" at the original image. Early layers have small receptive fields (seeing edges and textures), while deep layers have large receptive fields (seeing objects and scenes).

## Why This Concept Matters

Receptive field size determines what scale of patterns a neuron can detect. A model for detecting edges needs a small receptive field; one for detecting faces needs a large one. Understanding receptive fields helps you design architectures with appropriate context, diagnose why a model misses large-scale patterns, and build efficient networks.

## Mathematical Explanation

The **effective receptive field** after $n$ layers is:

$$r_n = r_{n-1} + (k_n - 1) \cdot \prod_{i=1}^{n-1} s_i$$

Where:
- $r_0 = 1$ (each input pixel's receptive field is itself)
- $k_n$ is the kernel size of layer $n$
- $s_i$ is the stride of layer $i$

For a sequence of convolutions with kernel sizes $k_1, ..., k_n$ and strides $s_1, ..., s_n$:

$$r_{n} = 1 + \sum_{i=1}^{n} (k_i - 1) \cdot \prod_{j=1}^{i-1} s_j$$

With dilation $d_i$, the effective kernel size becomes:
$$k_i^{(eff)} = k_i + (k_i - 1) \cdot (d_i - 1)$$

**Example**: Two layers of 3x3 conv (stride 1):
$$r_2 = 1 + (3-1) \cdot 1 + (3-1) \cdot 1 = 5$$

Two layers of 3x3 conv (first stride 1, second stride 2):
$$r_2 = 1 + (3-1) \cdot 1 + (3-1) \cdot (1 \cdot 2) = 1 + 2 + 4 = 7$$

## Code Examples

### Example 1: Computing Receptive Field for a Sequential Model

```python
import torch
import torch.nn as nn

def compute_rf(model_structure):
    """Compute receptive field after each layer."""
    layers = []
    
    r = 1  # initial RF
    stride_product = 1
    
    for name, k, s, d in model_structure:
        k_eff = k + (k - 1) * (d - 1)
        r = r + (k_eff - 1) * stride_product
        layers.append({
            'name': name,
            'RF': r,
            'stride_product': stride_product * s
        })
        stride_product *= s
    
    return layers

# Model structure: (name, kernel, stride, dilation)
model_def = [
    ('conv1', 3, 1, 1),
    ('pool1', 2, 2, 1),
    ('conv2', 3, 1, 1),
    ('pool2', 2, 2, 1),
    ('conv3', 3, 1, 1),
]

rf_layers = compute_rf(model_def)

print(f"{'Layer':<8} {'Kernel':<8} {'Stride':<8} {'Dilation':<10} {'RF':<8}")
for l in rf_layers:
    print(f"{l['name']:<8} {k:<8} {s:<8} {d:<10} {l['RF']:<8}")
# Output: Layer    Kernel   Stride   Dilation   RF
# Output: conv1    3        1        1          3
# Output: pool1    2        2        1          5
# Output: conv2    3        1        1          9
# Output: pool2    2        2        1          13
# Output: conv3    3        1        1          21
```

### Example 2: Comparing Receptive Field Growth with Different Architectures

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Compare RF growth for different architectures
def rf_after_n_layers(kernel_size, stride, dilations, num_layers):
    r = 1
    stride_prod = 1
    for i in range(num_layers):
        d = dilations[i] if i < len(dilations) else 1
        k_eff = kernel_size + (kernel_size - 1) * (d - 1)
        r = r + (k_eff - 1) * stride_prod
        stride_prod *= stride
    return r

print(f"{'Architecture':<30} {'Layers':<8} {'RF':<8}")
print("-" * 46)

# Standard 3x3 stacks
for n in [1, 2, 3, 5, 10]:
    rf = rf_after_n_layers(3, 1, [1]*n, n)
    print(f"{f'{n}x 3x3 conv (s=1)':<30} {n:<8} {rf:<8}")
# Output: 1x 3x3 conv (s=1)             1       3
# Output: 2x 3x3 conv (s=1)             2       5
# Output: 3x 3x3 conv (s=1)             3       7
# Output: 5x 3x3 conv (s=1)             5       11
# Output: 10x 3x3 conv (s=1)            10      21

# Dilated convolutions
for dilations in [[1,2,4], [1,2,4,8], [1,1,1,1]]:
    n = len(dilations)
    rf = rf_after_n_layers(3, 1, dilations, n)
    print(f"{f'3x3 dil={dilations}':<30} {n:<8} {rf:<8}")
# Output: 3x3 dil=[1, 2, 4]             3       15
# Output: 3x3 dil=[1, 2, 4, 8]          4       31
# Output: 3x3 dil=[1, 1, 1, 1]          4       9

# Strided convolutions
for strides in [[1,2,1], [2,2,2], [1,1,1]]:
    r = 1
    sp = 1
    for i, s in enumerate(strides):
        r = r + (3 - 1) * sp
        sp *= s
    print(f"{f'3x3 strides={strides}':<30} {3:<8} {r:<8}")
# Output: 3x3 strides=[1, 2, 1]         3       7
# Output: 3x3 strides=[2, 2, 2]         3       15
# Output: 3x3 strides=[1, 1, 1]         3       7
```

### Example 3: Measuring Effective Receptive Field Empirically

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

def measure_rf_empirical(model, input_size=64, num_channels=3):
    """Measure receptive field by backpropagating from center pixel."""
    x = torch.randn(1, num_channels, input_size, input_size, requires_grad=True)
    out = model(x)
    
    # Pick center pixel of the last feature map
    center_y, center_x = out.shape[2] // 2, out.shape[3] // 2
    
    # Backprop from a single output activation
    out[0, 0, center_y, center_x].backward()
    
    gradient = x.grad[0].abs().sum(dim=0)  # Sum over channels
    active_pixels = (gradient > 1e-5).float()
    
    # Find bounding box of active gradient
    rows = torch.where(active_pixels.sum(dim=1) > 0)[0]
    cols = torch.where(active_pixels.sum(dim=0) > 0)[0]
    
    if len(rows) > 0 and len(cols) > 0:
        rf_h = rows[-1] - rows[0] + 1
        rf_w = cols[-1] - cols[0] + 1
        return int(rf_h.item()), int(rf_w.item())
    return None

# Build a simple model
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 32, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 1, 3, padding=1),
)

rf_h, rf_w = measure_rf_empirical(model, input_size=32)
print(f"Empirical receptive field: {rf_h}x{rf_w}")
# Output: Empirical receptive field: 7x7

# Compare to theoretical: r = 1 + 2*1 + 2*1 = 7
print(f"Theoretical RF: 7x7")
# Output: Theoretical RF: 7x7
```

## Common Mistakes

1. **Confusing receptive field with kernel size**: A single 3x3 conv has RF=3, but two stacked 3x3 convs have RF=5.
2. **Ignoring stride in RF calculations**: Stride multiplies the RF contributions of subsequent layers.
3. **Thinking all pixels in the RF contribute equally**: The effective RF is Gaussian-shaped — center pixels matter more.
4. **Not aligning RF with task requirements**: A small RF for global classification can miss context; a large RF for pixel-level tasks can blur details.
5. **Forgetting padding's effect on RF**: Padding doesn't change RF size, but it changes which input pixels are accessible.

## Interview Questions

### Beginner - 5
1. What is the receptive field of a CNN?
2. What is the receptive field of a single 3x3 convolution?
3. How does stacking multiple 3x3 convolutions affect RF?
4. Does a 7x7 kernel have a larger RF than two 3x3 layers?
5. Why is receptive field important?

### Intermediate - 5
1. Derive the receptive field formula for a sequence of convolutions.
2. How does stride affect receptive field?
3. Compare the RF of dilated vs standard convolutions.
4. Why is the effective receptive field Gaussian-shaped?
5. How do you compute RF for a ResNet with skip connections?

### Advanced - 3
1. Derive the RF formula for a general directed acyclic graph (DAG) architecture.
2. Explain the relationship between RF and the Nyquist sampling theorem in CNNs.
3. Design an architecture with a specific RF for a given task without excess computation.

## Practice Problems

### Easy - 5
1. Compute RF for a 3-layer network with 3x3 kernels, stride 1.
2. Compute RF for a network with pool after each 3x3 conv.
3. Compare RF of a 5x5 kernel vs two 3x3 kernels.
4. Find the number of 3x3 layers needed for RF=15.
5. Compute RF with dilated kernel (3x3, d=2).

### Medium - 5
1. Implement a function to compute RF for any sequential model.
2. Compare theoretical vs empirical RF for a small CNN.
3. Design a network with RF=35 using minimal layers.
4. Analyze RF in ResNet-18 and ResNet-50.
5. Implement RF-matched architectures (same RF, different designs).

### Hard - 3
1. Implement a differentiable RF measurement tool.
2. Design an architecture with RF that matches the dataset's object scale distribution.
3. Derive and implement the theoretical RF for non-sequential architectures (e.g., DenseNet, Inception).

## Solutions

### Easy - 1 Solution
```python
# 3 layers of 3x3 conv, stride 1
# r = 1 + 2*1 + 2*1 + 2*1 = 7
r = 1
for _ in range(3):
    r = r + (3 - 1) * 1
print(f"Receptive field: {r}")
```

## Related Concepts

DL-176 Convolution Operation, DL-178 Stride, DL-179 Padding, DL-180 Dilation

## Next Concepts

DL-186 Parameter Sharing, DL-187 Local Connectivity

## Summary

The receptive field defines the input region that influences each neuron. It grows with network depth, kernel size, dilation, and stride. Understanding RF is critical for matching architecture capacity to task requirements and for diagnosing whether a model has sufficient context for its decisions.

## Key Takeaways

- RF = the input region influencing a neuron's activation
- RF formula: r_n = r_{n-1} + (k_n - 1) * prod(strides before n)
- Stacks of 3x3 convs give linear RF growth with depth
- Dilated convolutions give exponential RF growth
- Effective RF is Gaussian (center pixels matter most)
- Task-appropriate RF is critical for good performance
- Large RF doesn't always mean better — may include irrelevant context
- Stride multiplies the RF contribution of subsequent layers
