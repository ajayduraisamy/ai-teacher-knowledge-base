# Concept: Stride

## Concept ID

DL-178

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

CNN Fundamentals

## Learning Objectives

- Understand how stride controls spatial downsampling in convolutions
- Compute output dimensions with different stride values
- Implement strided convolutions in PyTorch
- Analyze the trade-offs of different stride settings

## Prerequisites

DL-176 Convolution Operation, DL-177 Convolution Kernel

## Definition

Stride is the step size with which a convolution kernel moves across the input tensor. A stride of 1 moves the kernel one pixel at a time; a stride of 2 skips every other pixel, producing a spatially smaller output.

## Intuition

Imagine reading a book by moving your finger under each word (stride=1). You see every word, but it takes a while. Now imagine you skip every other word (stride=2). You read faster and get the gist, but miss some detail. Stride operates similarly in convolutions: smaller strides preserve spatial resolution (more detail), larger strides reduce resolution (less detail, less computation). A stride of 2 halves the output dimensions, which is commonly used instead of explicit pooling layers in modern architectures.

## Why This Concept Matters

Stride is the primary mechanism for spatial downsampling in many modern architectures (e.g., ResNet, DenseNet). It controls the trade-off between spatial resolution and computational cost. Understanding stride is essential for designing networks that correctly handle input dimensions and for connecting encoder-decoder architectures.

## Mathematical Explanation

Output size with stride $S$:

$$O_h = \left\lfloor \frac{W_h - K_h + 2P}{S} + 1 \right\rfloor$$
$$O_w = \left\lfloor \frac{W_w - K_w + 2P}{S} + 1 \right\rfloor$$

For a 3x3 kernel with stride 2 and padding 1:
$$O_h = \left\lfloor \frac{W_h - 3 + 2(1)}{2} + 1 \right\rfloor = \frac{W_h}{2}$$

This "same" padding with stride 2 produces exactly half the spatial dimensions.

**Computational cost scaling** with stride $S$:
$$\text{FLOPs} \propto \frac{1}{S^2}$$

Each stride increase of 2 reduces computation by 4x.

**Effective stride in sequential layers**: The overall downsampling factor after $N$ strided layers:
$$\text{Total Downsample} = \prod_{i=1}^{N} S_i$$

## Code Examples

### Example 1: Comparing Stride Values

```python
import torch
import torch.nn as nn

# Input: batch=1, channels=1, height=7, width=7
x = torch.randn(1, 1, 7, 7)

conv_stride_1 = nn.Conv2d(1, 1, 3, stride=1, padding=0)
conv_stride_2 = nn.Conv2d(1, 1, 3, stride=2, padding=0)
conv_stride_3 = nn.Conv2d(1, 1, 3, stride=3, padding=0)

out1 = conv_stride_1(x)
out2 = conv_stride_2(x)
out3 = conv_stride_3(x)

print(f"Input: {x.shape}")
# Output: Input: torch.Size([1, 1, 7, 7])

print(f"Stride 1 output: {out1.shape}")
# Output: Stride 1 output: torch.Size([1, 1, 5, 5])
# O = (7 - 3 + 0)/1 + 1 = 5

print(f"Stride 2 output: {out2.shape}")
# Output: Stride 2 output: torch.Size([1, 1, 3, 3])
# O = (7 - 3 + 0)/2 + 1 = 3

print(f"Stride 3 output: {out3.shape}")
# Output: Stride 3 output: torch.Size([1, 1, 2, 2])
# O = (7 - 3 + 0)/3 + 1 = 2.33 -> 2
```

### Example 2: Stride for Downsampling (Replacing Pooling)

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Modern architecture using stride-2 convolution for downsampling
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

x = torch.randn(1, 64, 32, 32)
block = DownsampleBlock(64, 128)
out = block(x)

print(f"Input: {x.shape}")
# Output: Input: torch.Size([1, 64, 32, 32])

print(f"Output (stride 2): {out.shape}")
# Output: Output (stride 2): torch.Size([1, 128, 16, 32])
```

### Example 3: Stride Behavior with Different Kernel Sizes

```python
import torch
import torch.nn.functional as F

x = torch.randn(1, 1, 10, 10)

# Test various combinations of kernel size and stride
configs = [(3, 1), (3, 2), (5, 1), (5, 2), (7, 1), (7, 2)]

print(f"{'Kernel':<8} {'Stride':<8} {'Padding':<8} {'Output Size':<12} {'Formula Check'}")
for k, s in configs:
    # Use padding to maintain size where possible
    p = k // 2  # same padding
    out = F.conv2d(x, torch.randn(1, 1, k, k), stride=s, padding=p)
    expected_h = (10 - k + 2*p) // s + 1
    actual_h = out.shape[2]
    match = "✓" if expected_h == actual_h else "✗"
    print(f"{k:<8} {s:<8} {p:<8} {str(out.shape):<12} O_h={expected_h} {match}")
# Output: Kernel   Stride   Padding  Output Size   Formula Check
# Output: 3        1        1        (1,1,10,10)   O_h=10 ✓
# Output: 3        2        1        (1,1,5,5)     O_h=5 ✓
# Output: 5        1        2        (1,1,10,10)   O_h=10 ✓
# Output: 5        2        2        (1,1,5,5)     O_h=5 ✓
# Output: 7        1        3        (1,1,10,10)   O_h=10 ✓
# Output: 7        2        3        (1,1,5,5)     O_h=5 ✓
```

## Common Mistakes

1. **Not updating padding when changing stride**: Stride affects output size dramatically; padding must be adjusted.
2. **Using non-integer stride that doesn't divide cleanly**: Fractional output sizes are floored, losing information.
3. **Stride larger than kernel size**: This skips input regions entirely, potentially missing important patterns.
4. **Assuming stride only affects spatial dimensions**: Stride also affects the effective receptive field.
5. **Using max pooling and stride-2 convolution together**: Double downsampling reduces spatial resolution too quickly.

## Interview Questions

### Beginner - 5
1. What is stride in convolution?
2. How does stride affect the output feature map size?
3. What is the difference between stride 1 and stride 2?
4. Can stride be a non-integer value?
5. How does stride affect computation time?

### Intermediate - 5
1. Derive the output size formula with stride.
2. Explain how stride-2 convolution can replace max pooling.
3. How does stride affect the effective receptive field?
4. What happens if stride > kernel size?
5. How do you maintain spatial dimensions with stride 2?

### Advanced - 3
1. Design a network with fractional strides for dense prediction tasks.
2. Explain the relationship between stride and aliasing in CNNs.
3. Derive the gradient propagation through a strided convolution.

## Practice Problems

### Easy - 5
1. Compute output size for input 28x28, kernel 5x5, stride 1.
2. Compute output size for input 28x28, kernel 5x5, stride 2.
3. Create a conv layer with stride 2 and check output half size.
4. Compare the number of FLOPs for stride 1 vs stride 2.
5. Show that stride 2 conv with padding halves the input.

### Medium - 5
1. Build a downsampling block using stride-2 convolutions.
2. Compare stride-2 conv vs max pooling for downsampling.
3. Implement a U-Net style encoder with varying strides.
4. Analyze grid artifacts from stride-2 convolutions.
5. Implement dilated convolution and compare stride vs dilation.

### Hard - 3
1. Design a learnable stride mechanism.
2. Implement sub-pixel convolution (stride < 1 for upsampling).
3. Analyze the spectral bias of strided convolutions.

## Solutions

### Easy - 1 Solution
```python
import torch
import torch.nn.functional as F
x = torch.randn(1, 1, 28, 28)
k = torch.randn(1, 1, 5, 5)
out = F.conv2d(x, k, stride=1, padding=0)
print(out.shape)  # (1, 1, 24, 24)
# O = (28 - 5 + 0)/1 + 1 = 24
```

## Related Concepts

DL-176 Convolution Operation, DL-179 Padding, DL-180 Dilation, DL-189 Max Pooling

## Next Concepts

DL-179 Padding, DL-180 Dilation

## Summary

Stride controls the step size of the convolution kernel, directly affecting output spatial dimensions and computational cost. Stride 2 halves dimensions and is commonly used for downsampling in modern architectures. Proper stride selection is crucial for dimension management and computational efficiency.

## Key Takeaways

- Stride determines how many pixels the kernel moves each step
- Larger stride = smaller output, less computation, less overlap
- Stride 2 is the standard for replacing pooling layers
- Output dimension formula: O = (W - K + 2P)/S + 1
- Stride affects effective receptive field and grid artifacts
- Padding must be adjusted when changing stride
- Modern CNNs use strided convs instead of pooling for downsampling
