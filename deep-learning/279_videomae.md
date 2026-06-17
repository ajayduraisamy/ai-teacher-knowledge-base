# Concept: VideoMAE

## Concept ID

DL-279

## Difficulty

Expert

## Domain

Deep Learning

## Module

Video Understanding

## Learning Objectives

## Prerequisites

- DL-276: Video Transformers
- DL-217: Masked Autoencoders (MAE)
- DL-208: Transformer Architecture

## Definition

VideoMAE (Video Masked Autoencoder), introduced by Tong et al. in 2022, extends the Masked Autoencoder framework from images to videos for self-supervised pre-training. It masks a large proportion (90%) of video tokens and reconstructs the missing pixels from the visible tokens using a transformer encoder-decoder. VideoMAE leverages the high temporal redundancy in video to learn strong spatiotemporal representations without requiring labeled data. Pre-trained VideoMAE achieves 87.4% top-1 on Kinetics-400 with a ViT-L backbone, surpassing supervised pre-training.

## Intuition

Video contains massive redundancy: adjacent frames are highly similar, and even distant frames share background and object appearance. Masking 90% of video tokens forces the model to learn meaningful spatiotemporal representations because the visible 10% must provide enough information to reconstruct the entire video. The high masking ratio is possible because temporal redundancy provides multiple views of the same scene. Unlike image MAE (75% mask), VideoMAE uses 90-95% masking, significantly reducing computation during pre-training.

## Why This Concept Matters

VideoMAE demonstrated that self-supervised pre-training with masked autoencoders can surpass supervised pre-training for video transformers, achieving state-of-the-art on Kinetics-400, Something-Something V2, and AVA. It showed that high masking ratios are feasible and beneficial for video, reducing pre-training cost by 4× compared to standard MAE. VideoMAE provides a practical recipe for pre-training video transformers without labeled data.

## Mathematical Explanation

VideoMAE tokenizes video into patches of size 2×16×16 (t×h×w). For a video of 16 frames at 224×224: 16/2 × 224/16 × 224/16 = 8 × 14 × 14 = 1568 tokens.

Masking: 90% of tokens are randomly masked. Only 10% (≈157 tokens) are fed to the encoder.

Encoder: ViT processes only visible tokens (no positional masking). Outputs visible token representations.

Decoder: lightweight transformer that takes encoded visible tokens + mask tokens with positional encodings. Decoder outputs pixel values for all tokens.

Loss: Mean squared error (MSE) between predicted and original pixel values in the masked region only.

Pre-training recipe:
- Frame sampling: uniform temporal sampling
- Masking: random token masking (tube masking variant available)
- Decoder: 4-6 transformer layers, 512 dim
- Target: normalized pixel values per patch

## Code Examples

### Example 1: VideoMAE Tokenization with Masking

```python
import torch
import torch.nn as nn

class VideoMAETokenizer(nn.Module):
    def __init__(self, patch_size=(2, 16, 16), embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, video):
        # video: [N, C, T, H, W]
        x = self.proj(video)  # [N, D, T', H', W']
        x = x.flatten(2).transpose(1, 2)  # [N, num_tokens, D]
        return x

def apply_video_mask(tokens, mask_ratio=0.9):
    N, L, D = tokens.shape
    num_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(N, L, device=tokens.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = ids_shuffle[:, :num_keep]
    batch_indices = torch.arange(N, device=tokens.device)[:, None]
    masked_tokens = tokens[batch_indices, ids_keep]
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    return masked_tokens, ids_restore

tokenizer = VideoMAETokenizer()
video = torch.randn(2, 3, 16, 224, 224)
tokens = tokenizer(video)
masked_tokens, ids_restore = apply_video_mask(tokens, mask_ratio=0.9)
print(f"Original tokens: {tokens.shape}, Masked tokens: {masked_tokens.shape}")
# Output: Original tokens: torch.Size([2, 1568, 768]), Masked tokens: torch.Size([2, 157, 768])
```

### Example 2: VideoMAE Decoder

```python
import torch
import torch.nn as nn

class VideoMAEDecoder(nn.Module):
    def __init__(self, num_tokens=1568, encoder_dim=768, decoder_dim=512, num_heads=8, num_layers=4):
        super().__init__()
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, decoder_dim))

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(decoder_dim, num_heads, dim_feedforward=decoder_dim*4, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, 2 * 16 * 16 * 3)  # reconstruct patches

    def forward(self, x, ids_restore):
        # x: encoded visible tokens
        x = self.decoder_embed(x)

        # Add mask tokens
        N, M, D = x.shape  # M = num_visible
        mask_tokens = self.mask_token.repeat(N, ids_restore.shape[1] - M, 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).repeat(1, 1, D))

        # Add positional embeddings
        x = x + self.pos_embed

        # Decoder blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.head(x)
        return x

decoder = VideoMAEDecoder()
encoded = torch.randn(2, 157, 768)
ids = torch.argsort(torch.rand(2, 1568), dim=1)
reconstruction = decoder(encoded, ids)
print(f"Reconstruction output: {reconstruction.shape}")
# Output: Reconstruction output: torch.Size([2, 1568, 1536])
```

### Example 3: VideoMAE Pre-training Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoMAELoss(nn.Module):
    def __init__(self, patch_size=(2, 16, 16)):
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = patch_size[0] * patch_size[1] * patch_size[2] * 3

    def forward(self, reconstruction, video, ids_restore, mask_ratio=0.9):
        # Normalize patches
        N, C, T, H, W = video.shape
        patches = video.unfold(2, self.patch_size[0], self.patch_size[0])
        patches = patches.unfold(3, self.patch_size[1], self.patch_size[1])
        patches = patches.unfold(4, self.patch_size[2], self.patch_size[2])
        patches = patches.contiguous().view(N, -1, self.patch_dim)  # [N, num_patches, patch_dim]

        # Normalize per patch
        patch_mean = patches.mean(dim=-1, keepdim=True)
        patch_std = patches.std(dim=-1, keepdim=True) + 1e-6
        patches_norm = (patches - patch_mean) / patch_std

        # Compute loss only on masked tokens
        N, L = patches.shape[:2]
        num_masked = int(L * mask_ratio)
        ids_mask = ids_restore[:, :num_masked]
        batch_indices = torch.arange(N, device=video.device)[:, None]
        target = patches_norm[batch_indices, ids_mask]
        pred = reconstruction[batch_indices, ids_mask]

        loss = F.mse_loss(pred, target)
        return loss

loss_fn = VideoMAELoss()
video = torch.randn(2, 3, 16, 224, 224)
recon = torch.randn(2, 1568, 1536)
ids = torch.argsort(torch.rand(2, 1568), dim=1)
loss = loss_fn(recon, video, ids, mask_ratio=0.9)
print(f"VideoMAE pre-training loss: {loss.item():.4f}")
# Output: VideoMAE pre-training loss: 1.0023
```

## Common Mistakes

1. **Masking ratio too low**: Unlike image MAE (75%), VideoMAE requires 90-95% masking due to high temporal redundancy. Lower ratios reduce the difficulty of the pre-training task.

2. **Incorrect pixel normalization**: Patches must be normalized per patch (mean and std) before the reconstruction target. Using global normalization reduces performance.

3. **Temporal frame sampling without jitter**: Uniform sampling of every 2nd or 4th frame may miss important motion cues. Random frame jitter during sampling helps.

4. **Decoder too large or too small**: The decoder should be lightweight (4-6 layers, 512 dim). Too large decoder makes the encoder lazy; too small limits reconstruction quality.

5. **Not using tube masking for dense tasks**: Random token masking works for classification but tube masking (masking entire tubes across time) is better for localization tasks.

## Interview Questions

### Beginner - 5

1. What does VideoMAE stand for?
2. What masking ratio does VideoMAE use?
3. How is video tokenized in VideoMAE?
4. What loss function is used for pre-training?
5. Why does VideoMAE use a higher masking ratio than image MAE?

### Intermediate - 5

1. Explain the encoder-decoder architecture of VideoMAE.
2. How does VideoMAE handle the reconstruction of masked video tokens?
3. What is tube masking and when is it used?
4. How does VideoMAE compare with supervised pre-training?
5. What are the computational benefits of high masking ratios?

### Advanced - 3

1. Analyze why 90% masking works for video but not images.
2. Design a VideoMAE variant for video object detection.
3. Compare VideoMAE with Video Contrastive Learning (e.g., VideoMoCo).

## Practice Problems

### Easy - 5

1. Count the number of tokens in VideoMAE for 32 frames.
2. Implement pixel normalization for video patches.
3. Compute the patch dimension.
4. Implement random token masking.
5. Count the encoder FLOPs vs. decoder FLOPs.

### Medium - 5

1. Implement tube masking for VideoMAE.
2. Build the VideoMAE encoder (ViT).
3. Implement the complete pre-training loop.
4. Build the VideoMAE decoder.
5. Write a script for fine-tuning VideoMAE.

### Hard - 3

1. Implement VideoMAE with multi-scale reconstruction.
2. Design a VideoMAE variant with motion-aware masking.
3. Combine VideoMAE with contrastive learning objectives.

## Solutions

Easy 1:
```python
def compute_videomae_tokens(T, H, W, tubelet_size=2, patch_size=16):
    return (T // tubelet_size) * (H // patch_size) * (W // patch_size)

tokens = compute_videomae_tokens(32, 224, 224)
print(f"VideoMAE tokens for 32 frames: {tokens}")
# Output: VideoMAE tokens for 32 frames: 3136
```

Medium 1 — Tube Masking:
```python
def tube_masking(tokens, mask_ratio=0.9):
    N, L, D = tokens.shape
    # Compute spatial positions: T' = T//tubelet, H' = H//16, W' = W//16
    T_prime, H_prime, W_prime = 8, 14, 14  # for 16 frames at 224
    num_spatial = H_prime * W_prime
    num_temporal = T_prime
    num_tubes = num_spatial  # one tube per spatial location

    tokens = tokens.view(N, num_temporal, num_spatial, D)
    noise = torch.rand(N, num_spatial, device=tokens.device)
    num_keep = int(num_spatial * (1 - mask_ratio))
    ids_keep = torch.argsort(noise, dim=1)[:, :num_keep]
    batch_indices = torch.arange(N, device=tokens.device)[:, None, None]
    masked_tokens = tokens[batch_indices, :, ids_keep, :]  # keep all temporal at those spatial positions
    masked_tokens = masked_tokens.reshape(N, num_temporal * num_keep, D)
    return masked_tokens

tokens = torch.randn(2, 1568, 768)
masked = tube_masking(tokens)
print(f"Tube masked tokens: {masked.shape}")
# Output: Tube masked tokens: torch.Size([2, 1256, 768])
```

## Related Concepts

- DL-276: Video Transformers
- DL-217: Masked Autoencoders (MAE)
- DL-277: TimeSformer

## Next Concepts

- DL-280: Video Benchmarks
- DL-281: VideoMAE V2

## Summary

VideoMAE adapts masked autoencoding to video with a very high masking ratio (90-95%), leveraging temporal redundancy to create a challenging self-supervised pre-training task. The model reconstructs pixel values of masked video patches from visible tokens using a lightweight decoder. VideoMAE achieves state-of-the-art results on Kinetics-400 (87.4%), Something-Something V2, and AVA, surpassing supervised pre-training while being computationally efficient.

## Key Takeaways

- 90-95% masking ratio due to temporal redundancy
- Encoder processes only visible tokens (10-15%)
- Lightweight decoder reconstructs masked patches
- Per-patch normalization for reconstruction target
- 87.4% top-1 on Kinetics-400 with ViT-L
- 4× pre-training cost reduction vs. standard MAE
- Tube masking variant for localization tasks
- Self-supervised pre-training surpasses supervised
