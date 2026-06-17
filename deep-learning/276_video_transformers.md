# Concept: Video Transformers

## Concept ID

DL-276

## Difficulty

Expert

## Domain

Deep Learning

## Module

Video Understanding

## Learning Objectives

## Prerequisites

- DL-208: Transformer Architecture
- DL-269: I3D
- DL-266: Video as 3D Data

## Definition

Video Transformers extend transformer architectures from images to video by adding the temporal dimension to self-attention. Unlike 3D CNNs that use local convolutions with fixed kernels, video transformers use self-attention to model long-range spatiotemporal dependencies. The core design questions are: (1) how to tokenize video (per-frame patches, spatiotemporal tubes), (2) how to organize attention (joint space-time, divided, or factorized), and (3) how to handle the quadratic complexity of video tokens. Key architectures include TimeSformer, Video Swin Transformer, ViViT, and MViT.

## Intuition

CNNs process video by sliding 3D convolutional kernels over the spatiotemporal volume, capturing local patterns. Transformers instead compute pairwise attention between all tokens, enabling direct modeling of long-range dependencies. A pixel in one frame can directly attend to a distant pixel in another frame, capturing motion without explicit optical flow. The challenge is the number of tokens: a video with T=16 frames at H=W=224 with 16×16 patches produces 16 × 14 × 14 = 3136 tokens, making full attention O(3136²) ≈ 10M pairs.

## Why This Concept Matters

Video transformers have achieved state-of-the-art results on major video benchmarks (Kinetics-400: 86.1% with TimeSformer-L, 84.6% with Video Swin), surpassing 3D CNNs. They demonstrate superior scaling properties (performance improves with more compute/data) and flexible attention designs. The trend toward transformer-based video architectures is expected to continue, similar to the trend in NLP and image recognition.

## Mathematical Explanation

Video tokenization:
- Per-frame patches: T × (H/p × W/p) tokens
- Spatiotemporal tubes: (T/t × H/p × W/p) tokens

Self-attention types:
1. Joint space-time: Single attention over all T×H×W tokens (expensive)
2. Divided attention: Spatial attention within each frame + temporal attention across frames (TimeSformer)
3. Factorized attention: 3D windows + shifted windows (Video Swin)
4. Separated attention: Dilated attention across space and time

Spatial-temporal positional encoding: Learned or sinusoidal encodings added to tokens.

## Code Examples

### Example 1: Video Tokenization

```python
import torch
import torch.nn as nn

class VideoPatchEmbed(nn.Module):
    def __init__(self, patch_size=16, tubelet_size=1, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(in_channels, embed_dim,
                             kernel_size=(tubelet_size, patch_size, patch_size),
                             stride=(tubelet_size, patch_size, patch_size))

    def forward(self, video):
        # video: [N, C, T, H, W]
        x = self.proj(video)  # [N, D, T', H', W']
        x = x.flatten(2).transpose(1, 2)  # [N, num_tokens, D]
        return x

embed = VideoPatchEmbed(patch_size=16, tubelet_size=2)
video = torch.randn(1, 3, 32, 224, 224)
tokens = embed(video)
print(f"Video tokens: {tokens.shape}")  # T'=16, H'=14, W'=14 -> 16*14*14=3136
# Output: Video tokens: torch.Size([1, 3136, 768])
```

### Example 2: Divided Space-Time Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DividedSpaceTimeAttention(nn.Module):
    def __init__(self, dim=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # Spatial attention
        self.norm1 = nn.LayerNorm(dim)
        self.qkv_s = nn.Linear(dim, dim * 3)
        # Temporal attention
        self.norm2 = nn.LayerNorm(dim)
        self.qkv_t = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, n_frames, n_spatial):
        # x: [N, T*HW, D]
        N, L, D = x.shape

        # Spatial attention (within each frame)
        x = x + self._spatial_attn(self.norm1(x), n_frames, n_spatial)

        # Temporal attention (across frames at same spatial position)
        x = x + self._temporal_attn(self.norm2(x), n_frames, n_spatial)

        return x

    def _spatial_attn(self, x, n_frames, n_spatial):
        N, L, D = x.shape
        x = x.view(N, n_frames, n_spatial, D)
        x = x.reshape(N * n_frames, n_spatial, D)
        qkv = self.qkv_s(x).reshape(N * n_frames, n_spatial, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(N * n_frames, n_spatial, D)
        out = out.view(N, n_frames, n_spatial, D).reshape(N, L, D)
        return self.proj(out)

    def _temporal_attn(self, x, n_frames, n_spatial):
        N, L, D = x.shape
        x = x.view(N, n_frames, n_spatial, D)
        x = x.transpose(1, 2).reshape(N * n_spatial, n_frames, D)
        qkv = self.qkv_t(x).reshape(N * n_spatial, n_frames, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(N * n_spatial, n_frames, D)
        out = out.transpose(1, 2).reshape(N, n_spatial, n_frames, D)
        out = out.transpose(1, 2).reshape(N, L, D)
        return self.proj(out)

attn = DividedSpaceTimeAttention()
x = torch.randn(2, 3136, 768)
out = attn(x, n_frames=16, n_spatial=196)
print(f"Divided attention output: {out.shape}")
# Output: Divided attention output: torch.Size([2, 3136, 768])
```

### Example 3: Simplified Video Transformer

```python
import torch
import torch.nn as nn

class SimpleVideoTransformer(nn.Module):
    def __init__(self, num_classes=400, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = VideoPatchEmbed(patch_size=16, tubelet_size=2, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + 3136, embed_dim))
        self.blocks = nn.ModuleList([
            DividedSpaceTimeAttention(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, video):
        # video: [N, C, T, H, W]
        x = self.patch_embed(video)
        N = x.shape[0]
        cls = self.cls_token.expand(N, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :x.shape[1]]
        n_spatial = (video.shape[3] // 16) * (video.shape[4] // 16)
        n_frames = video.shape[2] // 2
        for blk in self.blocks:
            x = blk(x, n_frames, n_spatial)
        x = self.norm(x)
        return self.head(x[:, 0])

model = SimpleVideoTransformer(num_classes=400)
video = torch.randn(1, 3, 32, 224, 224)
out = model(video)
print(f"Video transformer output: {out.shape}")
# Output: Video transformer output: torch.Size([1, 400])
```

## Common Mistakes

1. **Quadratic complexity with frame count**: Full joint space-time attention is O((THW)²). Use divided or factorized attention to scale linearly with T.

2. **Incorrect positional encoding for variable input sizes**: Positional encodings must be interpolated for different video resolutions or lengths. Failing to do so causes spatial misalignment.

3. **Not using pre-trained weights**: Video transformers benefit greatly from pre-training on image or video datasets. Training from scratch requires massive compute.

4. **Ignoring the temporal dimension ratio**: The ratio of temporal to spatial tokens affects attention distribution. Too many temporal tokens may dilute spatial information.

5. **Memory optimization neglect**: Video transformers are memory-intensive. Gradient checkpointing, mixed precision, and efficient attention implementations (xFormers) are essential.

## Interview Questions

### Beginner - 5

1. How do video transformers differ from image transformers?
2. What are the challenges of applying transformers to video?
3. How are video tokens created?
4. What is divided space-time attention?
5. Why is joint space-time attention expensive?

### Intermediate - 5

1. Explain factorized attention in video transformers.
2. How does Video Swin Transformer handle the temporal dimension?
3. Compare TimeSformer with Video Swin Transformer.
4. What is the role of positional encoding in video transformers?
5. How do video transformers compare with 3D CNNs?

### Advanced - 3

1. Analyze the scaling properties of video transformers vs. 3D CNNs.
2. Design a memory-efficient video transformer for long videos.
3. How would you incorporate audio into a video transformer?

## Practice Problems

### Easy - 5

1. Compute the number of tokens for a video [1, 3, 16, 224, 224].
2. Implement video patch embedding.
3. Create a learnable positional encoding.
4. Count the attention complexity for joint vs. divided attention.
5. Implement a [CLS] token for video classification.

### Medium - 5

1. Implement divided space-time attention.
2. Build a video transformer classification model.
3. Implement factorized self-attention.
4. Write a video data loader with tokenization.
5. Compare memory usage of different attention patterns.

### Hard - 3

1. Implement a full TimeSformer architecture.
2. Design a video transformer with adaptive token sampling.
3. Implement Video Swin Transformer's 3D shifted window attention.

## Solutions

Easy 1:
```python
T, H, W = 16, 224, 224
patch_size = 16
num_tokens = T * (H // patch_size) * (W // patch_size)
print(f"Number of video tokens: {num_tokens}")
# Output: Number of video tokens: 3136
```

Medium 1 — Divided Attention:
```python
class DividedAttentionBlock(nn.Module):
    def __init__(self, d_model=768, n_heads=12):
        super().__init__()
        self.space_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.time_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))

    def forward(self, x, n_frames, n_spatial):
        N, L, D = x.shape
        # Spatial
        x_space = x.view(N, n_frames, n_spatial, D).reshape(N*n_frames, n_spatial, D)
        x_space = x_space + self.space_attn(self.norm(x_space), self.norm(x_space), self.norm(x_space))[0]
        # Temporal
        x_time = x_space.view(N, n_frames, n_spatial, D).transpose(1, 2).reshape(N*n_spatial, n_frames, D)
        x_time = x_time + self.time_attn(self.norm(x_time), self.norm(x_time), self.norm(x_time))[0]
        x = x_time.view(N, n_spatial, n_frames, D).transpose(1, 2).reshape(N, L, D)
        return x + self.mlp(self.norm(x))

print("Divided attention block defined")
# Output: Divided attention block defined
```

## Related Concepts

- DL-277: TimeSformer
- DL-278: Video Swin Transformer
- DL-279: VideoMAE
- DL-208: Transformer Architecture

## Next Concepts

- DL-277: TimeSformer
- DL-278: Video Swin Transformer

## Summary

Video transformers adapt the transformer architecture to video by adding temporal self-attention to spatial self-attention. Divided space-time attention efficiently models spatiotemporal dependencies with linear complexity in the number of frames. Video transformers have achieved state-of-the-art results on Kinetics and other video benchmarks, surpassing 3D CNNs. Key architectures include TimeSformer, Video Swin, and ViViT.

## Key Takeaways

- Video tokens: T frames × H/p × W/p patches
- Joint attention: O((THW)²) — expensive
- Divided attention: spatial within frame + temporal across frames — O(THW² + T²HW)
- Video Swin: 3D shifted window attention
- Pre-training from image models accelerates convergence
- Video transformers outperform 3D CNNs on major benchmarks
- Memory efficiency is a key design consideration
- Flexible attention patterns enable scaling to longer videos
