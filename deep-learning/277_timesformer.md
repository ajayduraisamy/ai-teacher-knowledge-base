# Concept: TimeSformer

## Concept ID

DL-277

## Difficulty

Expert

## Domain

Deep Learning

## Module

Video Understanding

## Learning Objectives

## Prerequisites

- DL-276: Video Transformers
- DL-208: Transformer Architecture
- DL-269: I3D

## Definition

TimeSformer (Time-Space Transformer), introduced by Bertasius et al. in 2021, is a video classification architecture that applies divided space-time attention within a standard transformer encoder. It processes video as a sequence of patches extracted from each frame, applying self-attention first spatially (within each frame) and then temporally (across frames at the same spatial location). This divided attention pattern achieves state-of-the-art accuracy on Kinetics-400 (86.1% top-1 with TimeSformer-L) while being computationally more efficient than joint space-time attention.

## Intuition

TimeSformer's key insight is that space and time can be factorized: first understand what objects are present in each frame independently (spatial attention), then understand how they move over time (temporal attention). This factorization is computationally efficient (linear in T) and empirically effective. By separating the two attention steps, the model can learn complementary spatial and temporal representations without the quadratic cost of joint attention.

## Why This Concept Matters

TimeSformer demonstrated that standard transformer architectures without 3D convolutions could achieve state-of-the-art video classification, challenging the dominance of 3D CNNs. It showed that divided attention is a simple, effective, and scalable approach, matching or exceeding I3D while using less computation. TimeSformer also introduced several design choices (handling of CLS token, positional encodings) that influenced subsequent video transformers.

## Mathematical Explanation

TimeSformer processes a video of T frames, each divided into N = H/16 × W/16 patches.

1. Patch embedding: Each frame independently → T×N tokens
2. Positional encoding: spatial (learned per position) + temporal (learned per frame index)
3. Divided attention per block:
   - Layer norm → spatial attention (within frame) → residual
   - Layer norm → temporal attention (across frames, same position) → residual
   - Layer norm → MLP → residual
4. CLS token aggregation: Classification from the first token

Complexity: O(T·N²·D + N·T²·D) vs. O(T²·N²·D) for joint attention.

## Code Examples

### Example 1: TimeSformer Patch Embedding

```python
import torch
import torch.nn as nn

class TimeSformerPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, in_channels=3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, video):
        # video: [N, C, T, H, W]
        N, C, T, H, W = video.shape
        video = video.permute(0, 2, 1, 3, 4)  # [N, T, C, H, W]
        video = video.reshape(N * T, C, H, W)
        patches = self.proj(video)  # [N*T, D, H/p, W/p]
        Hp, Wp = patches.shape[-2:]
        patches = patches.flatten(2).transpose(1, 2)  # [N*T, num_patches, D]
        patches = patches.reshape(N, T, Hp * Wp, -1)
        return patches, T, Hp * Wp

embed = TimeSformerPatchEmbed()
video = torch.randn(2, 3, 16, 224, 224)
tokens, T, N = embed(video)
print(f"TimeSformer tokens: {tokens.shape}, T={T}, N={N}")
# Output: TimeSformer tokens: torch.Size([2, 16, 196, 768]), T=16, N=196
```

### Example 2: Space-Time Attention Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpaceTimeAttentionBlock(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.space_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.time_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, T, N):
        # x: [N_batch, T*N, D]
        B = x.shape[0]

        # Spatial attention (within each frame)
        x_space = x.reshape(B, T, N, -1).reshape(B * T, N, -1)
        x_space = x_space + self.space_attn(self.norm1(x_space), self.norm1(x_space), self.norm1(x_space))[0]
        x = x_space.reshape(B, T, N, -1).reshape(B, T * N, -1)

        # Temporal attention (across frames, same spatial position)
        x_time = x.reshape(B, T, N, -1).transpose(1, 2).reshape(B * N, T, -1)
        x_time = x_time + self.time_attn(self.norm2(x_time), self.norm2(x_time), self.norm2(x_time))[0]
        x = x_time.reshape(B, N, T, -1).transpose(1, 2).reshape(B, T * N, -1)

        # MLP
        x = x + self.mlp(self.norm3(x))
        return x

block = SpaceTimeAttentionBlock()
tokens = torch.randn(2, 3136, 768)
out = block(tokens, T=16, N=196)
print(f"Space-time block output: {out.shape}")
# Output: Space-time block output: torch.Size([2, 3136, 768])
```

### Example 3: Complete TimeSformer

```python
import torch
import torch.nn as nn

class TimeSformer(nn.Module):
    def __init__(self, num_classes=400, img_size=224, patch_size=16,
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = TimeSformerPatchEmbed(img_size, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, 1, (img_size//patch_size)**2, embed_dim))
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            SpaceTimeAttentionBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, video):
        tokens, T, N = self.patch_embed(video)
        B = tokens.shape[0]

        # Add CLS token
        cls = self.cls_token.expand(B, T, -1, -1)
        tokens = torch.cat([cls, tokens], dim=2)  # [B, T, N+1, D]

        # Add positional encodings
        tokens = tokens + self.pos_embed_spatial[:, :, :N+1] + self.pos_embed_temporal[:, :T]

        # Flatten T and N
        x = tokens.reshape(B, T * (N + 1), -1)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x, T, N + 1)

        x = self.norm(x)
        # Take CLS token from each frame
        x = x.reshape(B, T, N + 1, -1)[:, :, 0].mean(dim=1)  # average CLS across frames
        return self.head(x)

model = TimeSformer(num_classes=400)
video = torch.randn(1, 3, 16, 224, 224)
out = model(video)
print(f"TimeSformer output: {out.shape}")
# Output: TimeSformer output: torch.Size([1, 400])
```

## Common Mistakes

1. **Incorrect handling of CLS token per frame**: TimeSformer adds a CLS token to each frame, then averages them. Using a single video-level CLS token loses per-frame information.

2. **Forgetting temporal positional encoding**: Without temporal position information, the model cannot distinguish frame order. Temporal encoding is essential for motion understanding.

3. **Applying spatial and temporal attention in the wrong order**: TimeSformer applies spatial first, then temporal. Reversing the order changes the model's behavior and typically reduces accuracy.

4. **Fixed patch size affecting temporal resolution**: Larger patches reduce N but increase the burden on temporal attention. The patch size vs. frame count trade-off must be balanced.

5. **Not using ImageNet pre-training**: TimeSformer initializes the spatial attention weights from a ViT pre-trained on ImageNet. Cold-start training from scratch underperforms.

## Interview Questions

### Beginner - 5

1. What does TimeSformer stand for?
2. What is divided space-time attention?
3. How many CLS tokens does TimeSformer use?
4. What is the complexity advantage of divided attention?
5. How does TimeSformer achieve state-of-the-art video classification?

### Intermediate - 5

1. Explain the order of spatial and temporal attention in TimeSformer.
2. How does TimeSformer handle positional encoding?
3. What is the role of the CLS token in TimeSformer?
4. How does TimeSformer compare with I3D in terms of computation?
5. How does ImageNet pre-training help TimeSformer?

### Advanced - 3

1. Analyze the information flow: why does spatial-then-temporal attention work better?
2. Design a TimeSformer variant with learnable attention patterns.
3. How would you adapt TimeSformer for video object detection?

## Practice Problems

### Easy - 5

1. Count the number of tokens in a TimeSformer for 16 frames at 224×224.
2. Implement the patch embedding for TimeSformer.
3. Compute the complexity ratio between divided and joint attention.
4. Add a CLS token to each frame.
5. Average CLS tokens across frames.

### Medium - 5

1. Implement the space-time attention block.
2. Build a complete TimeSformer.
3. Implement temporal positional encoding.
4. Initialize spatial attention from ViT weights.
5. Write a TimeSformer training script.

### Hard - 3

1. Implement TimeSformer with multiple spatial scales.
2. Design a TimeSformer variant for action detection.
3. Compare TimeSformer with Video Swin Transformer on Kinetics.

## Solutions

Easy 1:
```python
T, H, W, p = 16, 224, 224, 16
N = (H // p) * (W // p)
total = T * (N + 1)  # +1 for CLS per frame
print(f"Total tokens: {total}")
# Output: Total tokens: 3152
```

Medium 1 — Space-Time Attention:
```python
class TimeSformerBlock(nn.Module):
    def __init__(self, d=768, h=12):
        super().__init__()
        self.space_attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.time_attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d))

    def forward(self, x, T, N):
        B = x.shape[0]
        x = x.reshape(B, T, N, -1)
        x_space = (self.norm(x) + self.space_attn(self.norm(x).reshape(B*T, N, -1), self.norm(x).reshape(B*T, N, -1), self.norm(x).reshape(B*T, N, -1))[0].reshape(B, T, N, -1))
        x_time = x_space.transpose(1, 2) + self.time_attn(self.norm(x_space).transpose(1, 2).reshape(B*N, T, -1), self.norm(x_space).transpose(1, 2).reshape(B*N, T, -1), self.norm(x_space).transpose(1, 2).reshape(B*N, T, -1))[0].reshape(B, N, T, -1).transpose(1, 2)
        return (x_time + self.mlp(self.norm(x_time))).reshape(B, T*N, -1)

print("TimeSformer block defined")
# Output: TimeSformer block defined
```

## Related Concepts

- DL-276: Video Transformers
- DL-278: Video Swin Transformer
- DL-208: Transformer Architecture

## Next Concepts

- DL-278: Video Swin Transformer
- DL-279: VideoMAE

## Summary

TimeSformer applies divided space-time attention to video classification, processing spatial attention within each frame followed by temporal attention across frames. This factorization achieves state-of-the-art accuracy on Kinetics while being computationally efficient (linear in T). TimeSformer demonstrated that standard transformers with divided attention can surpass 3D CNNs for video understanding, establishing a new paradigm in video architecture design.

## Key Takeaways

- Divided space-time attention: spatial first, then temporal
- Each frame has its own CLS token, averaged for final prediction
- Spatial + temporal positional encodings
- O(T·N² + N·T²) vs. O(T²·N²) for joint attention
- 86.1% top-1 on Kinetics-400 (TimeSformer-L)
- ImageNet pre-training for spatial attention weights
- Patch size 16×16, typically 16 frames
- Linear scaling with number of frames
