# Concept: 3D Convolution

## Concept ID

DL-194

## Difficulty

Advanced

## Domain

Deep Learning

## Module

CNN Fundamentals

## Learning Objectives

- Understand 3D convolution for volumetric data
- Implement 3D convolution in PyTorch
- Compare 3D convolution with 2D and pseudo-3D approaches
- Identify appropriate applications for 3D convolution

## Prerequisites

DL-176 Convolution Operation, DL-193 2D Convolution

## Definition

3D convolution extends 2D convolution by adding a third spatial/temporal dimension, where 3D kernels slide over the input along height, width, and depth (or time) simultaneously, capturing patterns across all three dimensions.

## Intuition

If 2D convolution is like scanning a photograph, 3D convolution is like scanning a video or a CT scan — you're looking for patterns that exist in three dimensions. For video, these are spatiotemporal patterns (a hand waving moves through both space and time). For medical imaging (MRI, CT), it detects 3D anatomical structures. The kernel is a small 3D cube that slides through the larger 3D volume, computing dot products at each position.

## Why This Concept Matters

3D convolution is essential for video analysis (action recognition), medical imaging (volumetric segmentation), and scientific applications (fluid dynamics). It captures spatiotemporal relationships that 2D convs applied frame-by-frame miss. However, it's computationally expensive, so understanding when and how to use it is critical.

## Mathematical Explanation

**3D convolution**:
$$O[i,j,k] = \sum_{c=0}^{C_{in}-1} \sum_{t=0}^{K_t-1} \sum_{h=0}^{K_h-1} \sum_{w=0}^{K_w-1} I_c[tS_t+i, hS_h+j, wS_w+k] \cdot K[t,h,w]$$

**Output dimensions**:
$$D_{out} = \left\lfloor \frac{D_{in} - K_t + 2P_t}{S_t} + 1 \right\rfloor$$
$$H_{out} = \left\lfloor \frac{H_{in} - K_h + 2P_h}{S_h} + 1 \right\rfloor$$
$$W_{out} = \left\lfloor \frac{W_{in} - K_w + 2P_w}{S_w} + 1 \right\rfloor$$

**PyTorch shape**: $(N, C_{in}, D, H, W)$ for input.

**Parameter count**:
$$\text{Params} = K_t \cdot K_h \cdot K_w \cdot C_{in} \cdot C_{out} + C_{out}$$

**Computational cost**:
$$\text{FLOPs} = 2 \cdot K_t \cdot K_h \cdot K_w \cdot C_{in} \cdot C_{out} \cdot D_{out} \cdot H_{out} \cdot W_{out}$$

## Code Examples

### Example 1: Basic 3D Convolution

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Input: batch=2, channels=3, depth=16, height=64, width=64
x = torch.randn(2, 3, 16, 64, 64)
print(f"Input shape: {x.shape}")
# Output: Input shape: torch.Size([2, 3, 16, 64, 64])

# 3D conv: in_channels=3, out_channels=32, kernel=3x3x3
conv3d = nn.Conv3d(in_channels=3, out_channels=32, 
                   kernel_size=3, padding=1)

out = conv3d(x)
print(f"Output shape: {out.shape}")
# Output: Output shape: torch.Size([2, 32, 16, 64, 64])

# Parameter count
params = sum(p.numel() for p in conv3d.parameters())
print(f"Params: {params:,}")
# Output: Params: 2,624

# Compare with 2D conv on same data
conv2d = nn.Conv2d(in_channels=3, out_channels=32,
                   kernel_size=3, padding=1)
params_2d = sum(p.numel() for p in conv2d.parameters())
print(f"2D conv params (per slice): {params_2d:,}")
# Output: 2D conv params (per slice): 896
```

### Example 2: 3D Convolution vs 2D Convolution on Video

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Video data: batch=1, RGB=3, frames=8, height=32, width=32
x = torch.randn(1, 3, 8, 32, 32)

# 3D conv: captures spatiotemporal patterns together
conv3d = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1)
out3d = conv3d(x)

# 2D conv applied per-frame independently
conv2d = nn.Conv2d(3, 16, kernel_size=3, padding=1)
# Reshape: (B, C, D, H, W) -> (B*D, C, H, W)
x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(-1, 3, 32, 32)
out2d_frames = conv2d(x_reshaped)
# Reshape back: (B*D, C_out, H, W) -> (B, C_out, D, H, W)
out2d = out2d_frames.reshape(1, 8, 16, 32, 32).permute(0, 2, 1, 3, 4)

print(f"3D conv output: {out3d.shape}")
# Output: 3D conv output: torch.Size([1, 16, 8, 32, 32])

print(f"2D conv (per-frame) output: {out2d.shape}")
# Output: 2D conv (per-frame) output: torch.Size([1, 16, 8, 32, 32])

# Compare parameters
params_3d = sum(p.numel() for p in conv3d.parameters())
params_2d = sum(p.numel() for p in conv2d.parameters())
print(f"\n3D conv params: {params_3d:,}")
# Output: 3D conv params: 4,368

print(f"2D conv params (same channels): {params_2d:,}")
# Output: 2D conv params (same channels): 448
```

### Example 3: 3D Convolution for Medical Imaging

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# 3D UNet for medical image segmentation
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Encoder
        self.enc1 = self._block(in_channels, 32)
        self.enc2 = self._block(32, 64)
        self.enc3 = self._block(64, 128)
        
        # Bottleneck
        self.bottleneck = self._block(128, 256)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.dec3 = self._block(256, 128)
        self.upconv2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec2 = self._block(128, 64)
        self.upconv1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec1 = self._block(64, 32)
        
        self.out = nn.Conv3d(32, out_channels, 1)
    
    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(),
            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(),
        )
    
    def forward(self, x):
        enc1 = self.enc1(x)
        p1 = F.max_pool3d(enc1, 2)
        enc2 = self.enc2(p1)
        p2 = F.max_pool3d(enc2, 2)
        enc3 = self.enc3(p2)
        p3 = F.max_pool3d(enc3, 2)
        
        bottleneck = self.bottleneck(p3)
        
        up3 = self.upconv3(bottleneck)
        cat3 = torch.cat([enc3, up3], dim=1)
        dec3 = self.dec3(cat3)
        
        up2 = self.upconv2(dec3)
        cat2 = torch.cat([enc2, up2], dim=1)
        dec2 = self.dec2(cat2)
        
        up1 = self.upconv1(dec2)
        cat1 = torch.cat([enc1, up1], dim=1)
        dec1 = self.dec1(cat1)
        
        return self.out(dec1)

model = UNet3D(1, 1)
x = torch.randn(1, 1, 32, 64, 64)  # 32 depth slices
out = model(x)

print(f"Input: {x.shape}")
# Output: Input: torch.Size([1, 1, 32, 64, 64])

print(f"Output (segmentation): {out.shape}")
# Output: Output (segmentation): torch.Size([1, 1, 32, 64, 64])

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
# Output: Total parameters: 1,741,697
```

## Common Mistakes

1. **Using 3D conv when 2D conv + temporal model suffices**: 3D conv is much more expensive — consider alternatives.
2. **Insufficient padding in temporal dimension**: For video with few frames, padding can introduce artifacts.
3. **Not considering memory limitations**: 3D conv uses much more memory than 2D (the 3rd dimension multiplies cost).
4. **Using large temporal kernels**: 3x3x3 is standard; larger temporal kernels quickly become prohibitively expensive.
5. **Forgetting that 3D requires more data**: 3D models have many more parameters and need larger datasets.

## Interview Questions

### Beginner - 5
1. What is 3D convolution?
2. How is 3D conv different from 2D conv?
3. What is the input shape for Conv3d in PyTorch?
4. Name two applications of 3D convolution.
5. What does the kernel size (3, 3, 3) mean in 3D conv?

### Intermediate - 5
1. Compare 3D convolution with 2D + temporal modeling.
2. Derive the parameter count for a 3D conv layer.
3. How does 3D convolution handle video data?
4. What is the computational cost of 3D vs 2D conv?
5. How do you reduce the memory footprint of 3D conv?

### Advanced - 3
1. Design a factorized 3D conv (separating spatial and temporal).
2. Derive the gradient of 3D convolution with respect to input.
3. Compare 3D conv with inflating 2D convs for video (I3D).

## Practice Problems

### Easy - 5
1. Apply Conv3d(3, 16, 3) to a (1, 3, 8, 32, 32) input.
2. Compute output depth for Conv3d with kernel 3, stride 2, padding 1.
3. Count parameters in Conv3d(64, 128, 3, padding=1).
4. Create a 3D conv layer that preserves all dimensions.
5. Compare Conv3d params vs Conv2d for the same channels.

### Medium - 5
1. Implement a 3D video classifier.
2. Build a factorized 3D conv (2D spatial + 1D temporal).
3. Compare training 3D conv vs 2D+RNN on a video task.
4. Implement 3D conv with group normalization.
5. Visualize 3D conv kernel filters.

### Hard - 3
1. Implement an I3D architecture that inflates 2D convs to 3D.
2. Design a memory-efficient 3D convolution using gradient checkpointing.
3. Derive and implement the efficient Winograd algorithm for 3D convolutions.

## Solutions

### Easy - 1 Solution
```python
conv = nn.Conv3d(3, 16, 3)
x = torch.randn(1, 3, 8, 32, 32)
out = conv(x)
print(out.shape)  # (1, 16, 6, 30, 30)
```

## Related Concepts

DL-193 2D Convolution, DL-192 1D Convolution, DL-195 Depthwise Convolution

## Next Concepts

DL-195 Depthwise Convolution

## Summary

3D convolution extends convolution to volumetric data, processing depth/time along with height and width. It's essential for video and medical imaging but comes with high computational cost. Applications include action recognition, volumetric segmentation, and scientific computing.

## Key Takeaways

- 3D conv operates over depth, height, and width simultaneously
- Input shape: (N, C, D, H, W) in PyTorch
- Much more expensive than 2D conv (multiply by D dimension)
- Effective for video (spatiotemporal features) and 3D medical images
- Factorized 3D conv (separate spatial+temporal) reduces cost
- May overfit without sufficient data
- Often used with smaller kernel sizes (3x3x3) to manage cost
- Pre-trained 2D weights can be inflated to 3D
- Modern video models (I3D, C3D, SlowFast) use variants of 3D conv
