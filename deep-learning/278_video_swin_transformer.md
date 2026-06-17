# Concept: Video Swin Transformer

## Concept ID

DL-278

## Difficulty

Expert

## Domain

Deep Learning

## Module

Video Understanding

## Learning Objectives

## Prerequisites

- DL-276: Video Transformers
- DL-277: TimeSformer
- DL-213: Swin Transformer

## Definition

Video Swin Transformer extends the Swin Transformer architecture from images to videos by introducing 3D shifted window attention. It builds hierarchical feature maps with local 3D windows that shift across both spatial and temporal dimensions, enabling efficient spatiotemporal self-attention. Video Swin achieves state-of-the-art results on Kinetics-400 (84.6% top-1 with Swin-L) and Kinetics-600 (86.1%), demonstrating strong temporal modeling while maintaining the inductive biases of locality and hierarchy.

## Intuition

TimeSformer applies attention across the full temporal dimension (each spatial position attends to all corresponding positions in all frames). Video Swin instead restricts attention to local 3D windows and uses shifted windows to enable cross-window connections. This mirrors how CNNs use local receptive fields but with the flexibility of attention. The result is a model that is more computationally efficient (especially for high-resolution videos) while building hierarchical representations.

## Why This Concept Matters

Video Swin Transformer bridges the gap between Swin (image) and video understanding, showing that the hierarchical, local-attention design generalizes naturally to video. It achieved state-of-the-art on multiple benchmarks and introduced design principles — 3D window attention, temporal downsampling, and shifted windows — that became standard in video transformers. It also demonstrated that local attention can match or exceed global attention for video tasks.

## Mathematical Explanation

Video Swin uses 3D windows of size P_t × P_h × P_w. For a feature tensor [T, H, W], windows partition it into non-overlapping cubes of shape (P_t, P_h, P_w). Self-attention is computed within each window.

3D shifted window attention alternates between two configurations:
- Regular: windows at (0, 0, 0)
- Shifted: windows at (P_t/2, P_h/2, P_w/2)

Patch merging in 3D concatenates 2×2×2 groups and projects to 2× the dimension, reducing spatiotemporal resolution.

Hierarchical stages:
- Stage 1: 4× downsampled (T/2, H/4, W/4)
- Stage 2: 8× downsampled (T/4, H/8, W/8)
- Stage 3: 16× downsampled (T/8, H/16, W/16)
- Stage 4: 32× downsampled (T/16, H/32, W/32)

## Code Examples

### Example 1: 3D Window Partition

```python
import torch
import torch.nn as nn

def window_partition_3d(x, window_size):
    # x: [B, T, H, W, C]
    B, T, H, W, C = x.shape
    w_t, w_h, w_w = window_size
    x = x.view(B, T // w_t, w_t, H // w_h, w_h, W // w_w, w_w, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7)
    windows = x.contiguous().view(-1, w_t, w_h, w_w, C)
    return windows

def window_reverse_3d(windows, window_size, B, T, H, W):
    w_t, w_h, w_w = window_size
    x = windows.view(B, T // w_t, H // w_h, W // w_w, w_t, w_h, w_w, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7)
    x = x.contiguous().view(B, T, H, W, -1)
    return x

x = torch.randn(2, 16, 56, 56, 96)
windows = window_partition_3d(x, (2, 7, 7))
print(f"3D windows shape: {windows.shape}")
# Output: 3D windows shape: torch.Size([512, 2, 7, 7, 96])
```

### Example 2: 3D Shifted Window Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowAttention3D(nn.Module):
    def __init__(self, dim=96, window_size=(2, 7, 7), num_heads=4):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )

    def forward(self, x, mask=None):
        B_, W, C = x.shape
        qkv = self.qkv(x).reshape(B_, W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.to(attn.dtype)
            attn = attn.view(B_ // mask.shape[0], mask.shape[0], self.num_heads, W, W) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, W, W)
        attn = F.softmax(attn, dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, W, C)
        x = self.proj(x)
        return x

attn = WindowAttention3D(dim=96, window_size=(2, 7, 7), num_heads=4)
windows = torch.randn(512, 2*7*7, 96)
out = attn(windows)
print(f"Window attention output: {out.shape}")
# Output: Window attention output: torch.Size([512, 98, 96])
```

### Example 3: 3D Patch Merging

```python
import torch
import torch.nn as nn

class PatchMerging3D(nn.Module):
    def __init__(self, dim=96):
        super().__init__()
        self.reduction = nn.Linear(dim * 8, dim * 2, bias=False)

    def forward(self, x, T, H, W):
        # x: [B, T, H, W, C]
        B = x.shape[0]
        x = x.view(B, T // 2, 2, H // 2, 2, W // 2, 2, -1)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(B, T // 2, H // 2, W // 2, -1)
        x = self.reduction(x)
        return x, T // 2, H // 2, W // 2

x = torch.randn(2, 16, 56, 56, 96)
merge = PatchMerging3D()
out, T2, H2, W2 = merge(x, 16, 56, 56)
print(f"After 3D patch merging: {out.shape}, T={T2}, H={H2}, W={W2}")
# Output: After 3D patch merging: torch.Size([2, 8, 28, 28, 192]), T=8, H=28, W=28
```

## Common Mistakes

1. **Incorrect 3D shifted window masking**: The cyclic shift creates non-contiguous windows. An attention mask must be applied to prevent unrelated patches from attending. Forgetting the mask breaks the model.

2. **Aspect ratio of 3D windows**: The temporal window size (P_t) matters. Too large P_t = too many temporal tokens per window (expensive). Too small = limited temporal receptive field.

3. **Batch processing of variable-length videos**: Video Swin assumes fixed spatiotemporal resolution. Handling variable-length videos requires padding and attention masking or interpolating positional biases.

4. **Ignoring temporal patch merging**: Unlike image Swin, Video Swin merges patches in the temporal dimension too. Omitting this prevents building hierarchical temporal features.

5. **Memory with large windows**: 3D windows are 3× the size of 2D windows. Window size choice significantly impacts memory. Default sizes are (2, 7, 7) or (8, 7, 7) for large models.

## Interview Questions

### Beginner - 5

1. What is the key difference between TimeSformer and Video Swin?
2. What are 3D shifted windows?
3. How does Video Swin create hierarchical features?
4. What is the purpose of patch merging in 3D?
5. How does Video Swin handle the temporal dimension?

### Intermediate - 5

1. Explain how 3D shifted window attention works.
2. How does Video Swin compare to TimeSformer in complexity?
3. What is the role of relative position bias in 3D windows?
4. Describe the four-stage hierarchy of Video Swin.
5. How does Video Swin handle cross-window connections?

### Advanced - 3

1. Analyze the trade-offs between local and global video attention.
2. Design a Video Swin variant for real-time video inference.
3. Compare Video Swin with MViT (Multiscale Vision Transformer).

## Practice Problems

### Easy - 5

1. Partition a video tensor into 3D windows.
2. Compute the number of windows for a given window size.
3. Merge 2×2×2 patches in 3D.
4. Apply cyclic shift to a 3D tensor.
5. Create a relative position bias table.

### Medium - 5

1. Implement 3D shifted window attention.
2. Build a Video Swin Transformer stage.
3. Implement the full 3D window partition/reverse cycle.
4. Write a 3D cyclic shift with attention mask.
5. Build the four-stage Video Swin backbone.

### Hard - 3

1. Implement a complete Video Swin Transformer.
2. Adapt Video Swin for video instance segmentation.
3. Design a Video Swin variant with adaptive window sizes.

## Solutions

Easy 1:
```python
def count_3d_windows(T, H, W, w_t, w_h, w_w):
    return (T // w_t) * (H // w_h) * (W // w_w)

num_windows = count_3d_windows(16, 56, 56, 2, 7, 7)
print(f"Number of 3D windows: {num_windows}")
# Output: Number of 3D windows: 512
```

Medium 1:
```python
class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim=96, num_heads=4, window_size=(2, 7, 7), shift=False):
        super().__init__()
        self.window_size = window_size
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))

    def forward(self, x, T, H, W):
        B, L, C = x.shape
        x = x.view(B, T, H, W, C)
        if self.shift:
            shifted = torch.roll(x, shifts=(-self.window_size[0]//2, -self.window_size[1]//2, -self.window_size[2]//2), dims=(1, 2, 3))
        else:
            shifted = x
        windows = window_partition_3d(shifted, self.window_size)
        windows = windows.view(-1, self.window_size[0]*self.window_size[1]*self.window_size[2], C)
        attn_windows = self.attn(self.norm1(windows))
        shifted = window_reverse_3d(attn_windows.view(-1, *self.window_size, C), self.window_size, B, T, H, W)
        if self.shift:
            x = x + torch.roll(shifted, shifts=(self.window_size[0]//2, self.window_size[1]//2, self.window_size[2]//2), dims=(1, 2, 3))
        else:
            x = x + shifted
        x = x + self.mlp(self.norm2(x))
        return x.view(B, T*H*W, C)

block = SwinTransformerBlock3D()
x = torch.randn(2, 16*56*56, 96)
out = block(x, 16, 56, 56)
print(f"3D Swin block output: {out.shape}")
# Output: 3D Swin block output: torch.Size([2, 50176, 96])
```

## Related Concepts

- DL-276: Video Transformers
- DL-277: TimeSformer
- DL-213: Swin Transformer

## Next Concepts

- DL-279: VideoMAE
- DL-280: Video Benchmarks

## Summary

Video Swin Transformer extends Swin to video with 3D shifted window attention, creating hierarchical spatiotemporal representations. It partitions the video into 3D windows for local attention, shifts windows across spatial and temporal dimensions for cross-window connections, and uses 3D patch merging for hierarchical downsampling. Video Swin achieves state-of-the-art results on Kinetics-400/600 and demonstrates that local attention scales effectively to video tasks.

## Key Takeaways

- 3D shifted window attention: local, efficient, hierarchical
- Four-stage architecture with 3D patch merging
- 84.6% top-1 on Kinetics-400 (Swin-L)
- Local attention reduces complexity vs. global attention
- 3D windows of size (P_t, P_h, P_w)
- Cyclic shift + attention mask for cross-window connections
- Hierarchical features from fine to coarse
- Scales to higher resolution videos than TimeSformer
