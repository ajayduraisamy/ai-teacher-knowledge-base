# Concept: Video as 3D Data

## Concept ID

DL-266

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Video Understanding

## Learning Objectives

## Prerequisites

- DL-201: Convolutional Neural Networks
- DL-221: Image Classification
- Understanding of tensors

## Definition

Video as 3D Data refers to the representation and processing of video as 3D spatiotemporal tensors. While images are 2D arrays (height × width) with a channel dimension, video adds a temporal dimension: (T × C × H × W) where T is the number of frames. This formulation enables 3D convolutions that jointly process spatial and temporal information, treating the video as a 3D volume (spatial x, spatial y, and time). This is the foundation for video understanding models including C3D, I3D, and video transformers.

## Intuition

A video is simply a stack of frames over time. Instead of processing each frame independently (2D CNN), we can process the entire volume with 3D convolutions that have filters spanning both space and time. A 3D convolution with kernel size 3×3×3 looks at a 3×3 spatial region across 3 consecutive frames, capturing both visual features and their temporal evolution. This allows the model to learn motion patterns, temporal dynamics, and spatiotemporal features directly from the raw video.

## Why This Concept Matters

Representing video as 3D data enables end-to-end learning of spatiotemporal features, which is essential for action recognition, video classification, object tracking, and temporal action detection. The choice of how to represent time (2D + temporal pooling, 3D convolutions, or two-stream) fundamentally shapes the model's ability to capture motion. Understanding video as 3D data is the prerequisite for all video understanding architectures.

## Mathematical Explanation

Video tensor: X ∈ R^{T × C × H × W}
- T: number of frames (temporal dimension)
- C: channels (RGB: 3)
- H, W: spatial dimensions

3D Convolution: Given weight tensor K ∈ R^{C_out × C_in × K_t × K_h × K_w}:
Y[t, c_out] = Σ_{c_in} Σ_{τ} Σ_{i} Σ_{j} X[t+τ, c_in, i, j] · K[c_out, c_in, τ, i, j] + bias

3D Pooling: MaxPool3d(kernel_size=(2, 2, 2), stride=2) reduces T×H×W by half.

Processing a 16-frame clip at 224×224: [N, 3, 16, 224, 224]
After 3D conv with 64 filters, kernel 3×3×3: [N, 64, 14, 222, 222] (assuming stride=1, no padding)

## Code Examples

### Example 1: Video Tensor Operations

```python
import torch
import torch.nn as nn

# Create a synthetic video tensor
video = torch.randn(2, 3, 16, 224, 224)  # [N, C, T, H, W]
print(f"Video tensor shape: {video.shape}")
print(f"Dimensions: {video.dim()}D")
print(f"Frames: {video.shape[2]}, Height: {video.shape[3]}, Width: {video.shape[4]}")
# Output:
# Video tensor shape: torch.Size([2, 3, 16, 224, 224])
# Dimensions: 5D
# Frames: 16, Height: 224, Width: 224

# Extract a single frame
single_frame = video[:, :, 0, :, :]  # First frame
print(f"Single frame: {single_frame.shape}")
# Output: Single frame: torch.Size([2, 3, 224, 224])

# Stack frames manually
frames = [torch.randn(2, 3, 224, 224) for _ in range(16)]
stacked = torch.stack(frames, dim=2)
print(f"Stacked video: {stacked.shape}")
# Output: Stacked video: torch.Size([2, 3, 16, 224, 224])
```

### Example 2: 3D Convolution vs. 2D Convolution

```python
import torch
import torch.nn as nn

# Compare 2D conv on each frame vs. 3D conv on video

# 2D conv applied per frame (shared weights)
conv2d = nn.Conv2d(3, 64, 3, padding=1)
video_2d = torch.randn(2, 3, 8, 224, 224)
# Process each frame independently
out_2d = torch.stack([conv2d(video_2d[:, :, t]) for t in range(8)], dim=2)
print(f"2D conv per frame: {out_2d.shape}")

# 3D conv processes all frames jointly
conv3d = nn.Conv3d(3, 64, (3, 3, 3), padding=(1, 1, 1))
out_3d = conv3d(video_2d)
print(f"3D conv video: {out_3d.shape}")
# Output:
# 2D conv per frame: torch.Size([2, 64, 8, 224, 224])
# 3D conv video: torch.Size([2, 64, 8, 224, 224])

# Parameter comparison
params_2d = sum(p.numel() for p in conv2d.parameters())
params_3d = sum(p.numel() for p in conv3d.parameters())
print(f"2D conv params: {params_2d}, 3D conv params: {params_3d}")
print(f"Ratio: {params_3d / params_2d:.1f}x")
# Output:
# 2D conv params: 1792, 3D conv params: 5376
# Ratio: 3.0x
```

### Example 3: Temporal Downsampling and Padding

```python
import torch
import torch.nn as nn

class VideoTemporalProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d = nn.Conv3d(3, 64, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2))
        # stride=(1,2,2) -> downsample spatial only, not temporal

    def forward(self, x):
        print(f"Input: {x.shape}")
        x = self.conv3d(x)
        print(f"After conv3d: {x.shape}")
        x = self.pool3d(x)
        print(f"After pool3d: {x.shape}")
        return x

# Different temporal strides
video_16 = torch.randn(1, 3, 16, 224, 224)
video_32 = torch.randn(1, 3, 32, 224, 224)

processor = VideoTemporalProcessor()
print("=== 16-frame clip ===")
processor(video_16)
print("\n=== 32-frame clip ===")
processor(video_32)
# Output:
# === 16-frame clip ===
# Input: torch.Size([1, 3, 16, 224, 224])
# After conv3d: torch.Size([1, 64, 16, 224, 224])
# After pool3d: torch.Size([1, 64, 16, 112, 112])
```

## Common Mistakes

1. **Confusing tensor dimension order**: Different frameworks use different dimension orders (T, C, H, W vs. C, T, H, W). PyTorch typically uses (N, C_in, T, H, W) for 3D convolutions. Always verify your framework's convention.

2. **Processing frames independently ignores motion**: 2D convolutions applied per frame cannot capture temporal patterns like motion. The model sees each frame as an independent image.

3. **Not handling variable-length videos**: Videos have different frame counts. Common solutions: sampling, padding+masking, or temporal pooling.

4. **Memory explosion with 3D convolutions**: 3D convolutions have K_t × K_h × K_w × C_in × C_out parameters. A 3×3×3 conv3d has 3x parameters of a 3×3 conv2d. Large clips strain GPU memory.

5. **Ignoring temporal receptive field**: The number of frames a model processes defines its temporal receptive field. A model with T=16 frames cannot capture patterns longer than 16 frames.

## Interview Questions

### Beginner - 5

1. What is the shape of a video tensor?
2. How does a 3D convolution differ from 2D convolution?
3. What are the advantages of processing video as 3D data?
4. What is the temporal dimension?
5. How many parameters does a 3D conv have vs. 2D conv?

### Intermediate - 5

1. Explain the memory implications of 3D convolutions.
2. How do you handle variable-length videos in a batch?
3. What is temporal pooling and why is it used?
4. How does the temporal receptive field affect video understanding?
5. Compare frame-wise processing vs. 3D convolution.

### Advanced - 3

1. Design an efficient video representation that balances temporal and spatial processing.
2. How would you implement a video model that processes both short and long temporal ranges?
3. Analyze the trade-offs between 3D convolutions and 2D convolutions with optical flow.

## Practice Problems

### Easy - 5

1. Create a random video tensor of shape [2, 3, 8, 224, 224].
2. Extract frames 4-8 from a video tensor.
3. Apply a 3D convolution with kernel 3×3×3.
4. Count the number of parameters in a Conv3d(3, 64, 3).
5. Implement temporal average pooling.

### Medium - 5

1. Implement a 3D convolutional block (conv + BN + ReLU + pool).
2. Compare memory usage between 2D and 3D convolutions.
3. Implement frame sampling: select 16 frames uniformly from a long video.
4. Build a simple 3D CNN for video classification.
5. Implement temporal padding for variable-length videos.

### Hard - 3

1. Implement a spatiotemporal feature pyramid.
2. Design a video model that combines 2D and 3D convolutions.
3. Implement a memory-efficient 3D convolution using separable convolutions.

## Solutions

Easy 1:
```python
video = torch.randn(2, 3, 8, 224, 224)
print(f"Video shape: {video.shape}")
print(f"Batch: {video.shape[0]}, Channels: {video.shape[1]}, "
      f"Frames: {video.shape[2]}, Height: {video.shape[3]}, Width: {video.shape[4]}")
# Output: Video shape: torch.Size([2, 3, 8, 224, 224])
# Batch: 2, Channels: 3, Frames: 8, Height: 224, Width: 224
```

Medium 1:
```python
class Conv3DBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))

block = Conv3DBlock(3, 64)
x = torch.randn(1, 3, 16, 112, 112)
print(f"3D block output: {block(x).shape}")
# Output: 3D block output: torch.Size([1, 64, 8, 56, 56])
```

## Related Concepts

- DL-267: Optical Flow
- DL-268: C3D
- DL-269: I3D

## Next Concepts

- DL-267: Optical Flow
- DL-268: C3D

## Summary

Video as 3D data extends image processing by adding a temporal dimension, enabling 3D convolutions that jointly model space and time. This representation captures motion patterns and temporal dynamics that are invisible to frame-wise processing. While 3D convolutions are more parameter-efficient per frame than independent 2D convolutions, they require significantly more memory. Understanding the video tensor representation is the foundation for all video understanding models.

## Key Takeaways

- Video tensor: [N, C, T, H, W] (PyTorch convention)
- 3D convolutions process space and time jointly
- 3D conv has 3× parameters of 2D conv (kernel 3×3×3 vs 3×3)
- Memory scales linearly with number of frames
- Temporal receptive field is a critical hyperparameter
- Variable-length video requires sampling or padding
- 3D models capture motion patterns naturally
- Foundation for action recognition, video classification, tracking
