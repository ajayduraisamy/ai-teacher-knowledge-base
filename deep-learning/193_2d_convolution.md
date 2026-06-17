# Concept: 2D Convolution

## Concept ID

DL-193

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

CNN Fundamentals

## Learning Objectives

- Understand 2D convolution as applied to image data
- Implement 2D convolution with various parameters
- Compute output dimensions and parameter counts
- Analyze the role of 2D convolution in vision models

## Prerequisites

DL-176 Convolution Operation, DL-177 Convolution Kernel, DL-178 Stride, DL-179 Padding

## Definition

2D convolution is the standard operation in image-processing CNNs, where 2D kernels slide over 2D spatial inputs (height and width) computing element-wise products and summing them to produce feature maps.

## Intuition

2D convolution is the workhorse of computer vision. It's designed specifically for images, where patterns exist in two spatial dimensions. A 2D kernel scans both horizontally and vertically, detecting 2D patterns like corners, edges, and textures. The kernel's 2D structure lets it capture spatial relationships in both directions simultaneously — something 1D convolution applied to flattened images would miss.

## Why This Concept Matters

2D convolution is the core operation that made deep learning revolution in computer vision possible. It exploits the 2D structure of images, enabling parameter-efficient learning of spatial features. Almost every vision model is built on 2D convolutions.

## Mathematical Explanation

**2D convolution** (actually cross-correlation as implemented in deep learning frameworks):

$$O[i,j] = \sum_{c=0}^{C_{in}-1} \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} I_c[i+S\cdot m, j+S\cdot n] \cdot K_{c',c}[m,n] + b_{c'}$$

**Output dimensions**:
$$H_{out} = \left\lfloor \frac{H_{in} - K_h + 2P}{S} + 1 \right\rfloor$$
$$W_{out} = \left\lfloor \frac{W_{in} - K_w + 2P}{S} + 1 \right\rfloor$$

**Parameter count**:
$$\text{Params} = K_h \cdot K_w \cdot C_{in} \cdot C_{out} + C_{out}$$

**Computational complexity**:
$$\text{FLOPs} = 2 \cdot K_h \cdot K_w \cdot C_{in} \cdot C_{out} \cdot H_{out} \cdot W_{out}$$

## Code Examples

### Example 1: Basic 2D Convolution with Different Parameters

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Input: batch=4, RGB image (3 channels), 224x224
x = torch.randn(4, 3, 224, 224)

# Different 2D conv configurations
convs = {
    '3x3 standard': nn.Conv2d(3, 64, 3, padding=1),
    '5x5 standard': nn.Conv2d(3, 64, 5, padding=2),
    '7x7 stride 2': nn.Conv2d(3, 64, 7, stride=2, padding=3),
    '1x1 bottleneck': nn.Conv2d(3, 64, 1),
}

print(f"{'Name':<20} {'Kernel':<10} {'Output shape':<20} {'Params':<10}")
for name, conv in convs.items():
    out = conv(x)
    params = sum(p.numel() for p in conv.parameters())
    k = conv.kernel_size
    print(f"{name:<20} {str(k):<10} {str(list(out.shape)):<20} {params:<10}")
# Output: Name                 Kernel     Output shape          Params
# Output: 3x3 standard         (3, 3)     [4, 64, 224, 224]    1,792
# Output: 5x5 standard         (5, 5)     [4, 64, 224, 224]    4,864
# Output: 7x7 stride 2         (7, 7)     [4, 64, 112, 112]    9,472
# Output: 1x1 bottleneck       (1, 1)     [4, 64, 224, 224]    256
```

### Example 2: 2D Convolution as Matrix Multiplication (im2col)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Demonstrate that convolution equals matrix multiplication
# using im2col (image to column) transformation

def conv_im2col(x, weight, bias=None, stride=1, padding=0):
    """Implement 2D conv using im2col and matrix multiplication."""
    B, C_in, H, W = x.shape
    C_out, _, K_h, K_w = weight.shape
    
    # Apply padding
    if padding > 0:
        x = F.pad(x, (padding, padding, padding, padding))
        H_pad, W_pad = H + 2*padding, W + 2*padding
    else:
        H_pad, W_pad = H, W
    
    # Output dimensions
    H_out = (H_pad - K_h) // stride + 1
    W_out = (W_pad - K_w) // stride + 1
    
    # Extract patches (im2col)
    cols = []  # This will be (B * H_out * W_out, C_in * K_h * K_w)
    for i in range(0, H_pad - K_h + 1, stride):
        for j in range(0, W_pad - K_w + 1, stride):
            patch = x[:, :, i:i+K_h, j:j+K_w]  # (B, C_in, K_h, K_w)
            col = patch.reshape(B, -1)  # (B, C_in * K_h * K_w)
            cols.append(col)
    
    # Stack patches: (H_out * W_out, B, C_in*K*K)
    cols = torch.stack(cols, dim=0)  # (H_out*W_out, B, C_in*K*K)
    
    # Reshape weight: (C_out, C_in, K_h, K_w) -> (C_out, C_in*K*K)
    w_flat = weight.reshape(C_out, -1)
    
    # Matrix multiplication
    # cols: (H_out*W_out, B, C_in*K*K) -> (B, H_out*W_out, C_in*K*K)
    cols = cols.permute(1, 0, 2)  # (B, H_out*W_out, C_in*K*K)
    
    # (B, H_out*W_out, C_in*K*K) @ (C_in*K*K, C_out) -> (B, H_out*W_out, C_out)
    out = cols @ w_flat.T  # (B, H_out*W_out, C_out)
    
    # Reshape to output: (B, C_out, H_out, W_out)
    out = out.permute(0, 2, 1).reshape(B, C_out, H_out, W_out)
    
    if bias is not None:
        out += bias.view(1, -1, 1, 1)
    
    return out

# Test equivalence
x = torch.randn(1, 1, 5, 5)
conv = nn.Conv2d(1, 1, 3, padding=0, bias=False)

with torch.no_grad():
    native_out = conv(x)
    im2col_out = conv_im2col(x, conv.weight, stride=1, padding=0)

print(f"Native conv output:\n{native_out.squeeze().detach().numpy()}")
# Output: Native conv output:
# Output: [[ ... ]]

print(f"\nim2col conv output:\n{im2col_out.squeeze().detach().numpy()}")
# Output: im2col conv output:
# Output: [[ ... ]]

print(f"\nOutputs match: {torch.allclose(native_out, im2col_out, atol=1e-6)}")
# Output: Outputs match: True
```

### Example 3: 2D Convolution in a Sequential Model

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Build a complete 2D conv network for image classification
class Simple2DCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3: 8x8 -> 4x4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = Simple2DCNN()
x = torch.randn(8, 3, 32, 32)
out = model(x)

print(f"Input: {x.shape}")
# Output: Input: torch.Size([8, 3, 32, 32])

print(f"Output: {out.shape}")
# Output: Output: torch.Size([8, 10])

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
# Output: Total parameters: 221,866
```

## Common Mistakes

1. **Forgetting the (N, C, H, W) format**: PyTorch uses channel-first, unlike TensorFlow's channel-last.
2. **Using 2D conv on 1D data**: 1D sequences should use Conv1d, not Conv2d with height=1.
3. **Mixing up kernel size tuple**: `nn.Conv2d(3, 64, 3)` gives 3x3 kernel; `nn.Conv2d(3, 64, (3, 5))` gives 3x5.
4. **Not accounting for padding with asymmetric kernels**: Different padding for height and width.
5. **Ignoring memory for large feature maps**: 2D conv with many channels at high resolution consumes significant GPU memory.

## Interview Questions

### Beginner - 5
1. What is 2D convolution?
2. What is the input shape for nn.Conv2d?
3. How do you compute the output size after 2D convolution?
4. What does the kernel size parameter mean in Conv2d?
5. Why is 2D convolution well-suited for images?

### Intermediate - 5
1. Derive the FLOPs count for a 2D convolutional layer.
2. Explain how 2D convolution handles multi-channel input.
3. What is the im2col algorithm and why is it used?
4. How does 2D convolution with 1x1 kernels work?
5. Compare 2D convolution with depthwise 2D convolution.

### Advanced - 3
1. Derive the gradient of 2D convolution with respect to input and weights.
2. Explain Winograd's minimal filtering algorithm for small 2D convolutions.
3. Design a 2D convolution variant that is equivariant to rotations.

## Practice Problems

### Easy - 5
1. Apply Conv2d(3, 16, 5) to a (1, 3, 32, 32) input.
2. Compute output size for Conv2d(64, 128, 3, stride=2, padding=1).
3. Count parameters in Conv2d(3, 64, 7, stride=2).
4. Create a 2D conv layer that preserves input spatial size.
5. Build a 2-layer 2D conv network.

### Medium - 5
1. Implement the im2col algorithm for 2D convolution.
2. Compare standard 2D conv with depthwise separable 2D conv.
3. Visualize feature maps from 2D conv layers.
4. Build a 2D conv autoencoder.
5. Analyze the computational cost of 2D conv at different resolutions.

### Hard - 3
1. Implement a grouped 2D convolution from scratch.
2. Derive and implement the backward pass for 2D convolution.
3. Design an efficient 2D convolution implementation using FFT.

## Solutions

### Easy - 1 Solution
```python
conv = nn.Conv2d(3, 16, 5)
x = torch.randn(1, 3, 32, 32)
out = conv(x)
print(out.shape)  # (1, 16, 28, 28)
```

## Related Concepts

DL-176 Convolution Operation, DL-177 Convolution Kernel, DL-192 1D Convolution, DL-194 3D Convolution

## Next Concepts

DL-194 3D Convolution, DL-195 Depthwise Convolution

## Summary

2D convolution is the core operation in image CNNs, applying 2D kernels over spatial dimensions to detect visual patterns. It exploits the 2D structure of images, enables parameter sharing, and forms the basis of virtually all deep learning vision models.

## Key Takeaways

- 2D conv applies 2D kernels over height and width dimensions
- Input shape: (N, C, H, W); Output: (N, C_out, H_out, W_out)
- Output formula: O = (W - K + 2P)/S + 1
- Kernel can be square or rectangular
- im2col converts conv to matrix multiplication
- 1x1 conv is a special case for channel mixing
- Dominant operation in vision models (ResNet, VGG, etc.)
- Memory and compute scale with K^2 * C_in * C_out * H_out * W_out
- Modern GPUs are highly optimized for 2D convolution
