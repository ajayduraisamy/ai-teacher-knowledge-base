# Concept: Mask2Former

## Concept ID

DL-262

## Difficulty

Expert

## Domain

Deep Learning

## Module

Image Segmentation

## Learning Objectives

- Understand the masked attention mechanism for segmentation
- Implement the Mask2Former architecture with transformer decoder
- Comprehend the improvements over MaskFormer for convergence and performance
- Analyze the multi-scale high-resolution features

## Prerequisites

- DL-261: MaskFormer
- DL-247: DETR
- DL-257: DeepLab

## Definition

Mask2Former, introduced by Cheng et al. in 2022, is a unified panoptic segmentation architecture that improves upon MaskFormer with three key innovations: (1) masked attention, where transformer decoder self-attention is constrained to regions predicted by the current mask prediction, (2) multi-scale high-resolution features processed efficiently with a transformer decoder, and (3) optimization improvements including more training iterations and better loss weighting. Mask2Former achieved 57.8% PQ on COCO panoptic, 56.1% mIoU on ADE20K semantic, and 50.1% mAP on COCO instance segmentation.

## Intuition

MaskFormer uses standard cross-attention where each query attends to all image features. This is computationally expensive and slow to converge because queries must learn which pixels belong to "their" region. Mask2Former introduces masked attention: each query only attends to pixels where its predicted mask has high values. This focuses attention on the relevant region, speeding convergence and improving results. As the mask prediction improves through decoder layers, the attention becomes more focused. The multi-scale features provide high-resolution details for fine boundaries.

## Why This Concept Matters

Mask2Former set a new state of the art across all three segmentation tasks (semantic, instance, panoptic) simultaneously, demonstrating that a single architecture can outperform task-specific designs. The masked attention mechanism is a significant innovation that has influenced subsequent transformer-based segmentation and detection models. Mask2Former's convergence speed (3x faster than MaskFormer) makes transformer-based segmentation more practical.

## Mathematical Explanation

Masked Attention: Instead of standard cross-attention where query q attends to all N keys:
Attn(q, K, V) = Softmax(qK^T/√d) V

Masked attention restricts to keys within the predicted mask region:
Attn_masked(q, K, V, M) = Softmax(qK^T/√d + log(M)) V

where M ∈ {0, 1}^N is the binary mask indicating which positions belong to the query's region.

Multi-scale features: Mask2Former processes feature maps from FPN levels P2, P3, P4, P5 (resolutions 1/4, 1/8, 1/16, 1/32) using a transformer decoder with pooling-based feature extraction per scale.

## Code Examples

### Example 1: Masked Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedCrossAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        # query: [N, Q, d], key/value: [N, HW, d]
        # mask: [N, Q, HW] binary mask (0 = attend, -inf = ignore)
        N, Q, _ = query.shape
        _, S, _ = key.shape

        # Project
        q = self.q_proj(query).view(N, Q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(N, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(N, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply mask
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            # Invert mask: 0 = ignore, 1 = attend
            attn_mask = (mask == 0).float() * -1e9
            attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(N, Q, -1)
        return self.out_proj(out)

# Compare standard vs. masked attention
query = torch.randn(2, 10, 256)
key_val = torch.randn(2, 100, 256)
mask = torch.randint(0, 2, (2, 10, 100)).float()  # 1 = attend, 0 = ignore

attn = MaskedCrossAttention()
out_standard = attn(query, key_val, key_val)  # no mask (all attend)
out_masked = attn(query, key_val, key_val, mask)
print(f"Standard: {out_standard.shape}, Masked: {out_masked.shape}")
# Output: Standard: torch.Size([2, 10, 256]), Masked: torch.Size([2, 10, 256])
```

### Example 2: Multi-Scale Feature Extraction

```python
import torch
import torch.nn as nn

class MultiScaleTransformerDecoder(nn.Module):
    def __init__(self, d_model=256, n_heads=8, num_layers=3, num_queries=100):
        super().__init__()
        self.num_layers = num_layers
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, n_heads, 2048, batch_first=True)
            for _ in range(num_layers)
        ])

    def forward(self, multi_scale_features, mask_predictions=None):
        # multi_scale_features: list of [N, C, H, W] at different scales
        N = multi_scale_features[0].shape[0]
        query = self.query_embed.weight.unsqueeze(0).repeat(N, 1, 1)

        for i, layer in enumerate(self.layers):
            # Process each scale (simplified: concatenate all scales)
            all_features = []
            for feat in multi_scale_features:
                all_features.append(feat.view(N, feat.shape[1], -1).transpose(1, 2))
            memory = torch.cat(all_features, dim=1)  # [N, total_HW, C]

            query = layer(query, memory)

        return query

decoder = MultiScaleTransformerDecoder()
features = [torch.randn(2, 256, 64, 64), torch.randn(2, 256, 32, 32),
            torch.randn(2, 256, 16, 16)]
output = decoder(features)
print(f"Decoder output: {output.shape}")
# Output: Decoder output: torch.Size([2, 100, 256])
```

### Example 3: Mask2Former Prediction Flow

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Mask2FormerOutput(nn.Module):
    def __init__(self, d_model=256, num_classes=80, num_queries=100):
        super().__init__()
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.mask_embed = nn.Linear(d_model, d_model)

    def forward(self, query_features, multi_scale_pixel_feats):
        # query_features: [N, Q, d]
        # multi_scale_pixel_feats: list of pixel feature maps

        # Class predictions
        class_logits = self.class_embed(query_features)  # [N, Q, C+1]

        # Mask embeddings
        mask_embeds = self.mask_embed(query_features)  # [N, Q, d]

        # Compute masks at the highest resolution
        pixel_feats = multi_scale_pixel_feats[-1]  # Highest res: [N, d, H, W]
        N, _, H, W = pixel_feats.shape
        pixel_feats_flat = pixel_feats.view(N, -1, H*W).transpose(1, 2)  # [N, HW, d]

        mask_logits = torch.matmul(mask_embeds.unsqueeze(2),
                                    pixel_feats_flat.unsqueeze(1))
        mask_logits = mask_logits.squeeze(2).view(N, -1, H, W)  # [N, Q, H, W]

        return class_logits, mask_logits

out = Mask2FormerOutput()
query = torch.randn(2, 100, 256)
pixel_feats = [torch.randn(2, 256, 64, 64), torch.randn(2, 256, 128, 128)]
cls_out, mask_out = out(query, pixel_feats)
print(f"Class: {cls_out.shape}, Mask: {mask_out.shape}")
# Output: Class: torch.Size([2, 100, 81]), Mask: torch.Size([2, 100, 128, 128])
```

## Common Mistakes

1. **Incorrect mask preparation for masked attention**: The mask for masked attention must be upsampled to match the feature resolution. Using the mask at its prediction resolution without upsampling ignores fine details.

2. **Not using multi-scale features properly**: Each decoder layer should process features at different resolutions, not just the highest resolution. The resolution assignment per layer matters.

3. **Ignoring the mask prediction feedback loop**: Masked attention uses the mask prediction from the previous decoder layer. The mask must be updated per layer and fed back.

4. **Memory consumption from multi-scale processing**: Processing all feature scales simultaneously increases memory. Use pooling-based sampling to control memory.

5. **Training instability from masked attention**: The mask prediction can be noisy early in training, causing masked attention to miss relevant regions. Start with full attention and gradually introduce masking.

## Interview Questions

### Beginner - 5

1. What is the main innovation of Mask2Former over MaskFormer?
2. What is masked attention?
3. How many feature scales does Mask2Former use?
4. What tasks can Mask2Former perform?
5. What is the role of multi-scale features?

### Intermediate - 5

1. Explain how masked attention speeds up convergence.
2. How does Mask2Former process multi-scale features?
3. What is the mask prediction feedback loop in Mask2Former?
4. Compare the transformer decoder in Mask2Former vs. MaskFormer.
5. How does Mask2Former achieve 3x faster training than MaskFormer?

### Advanced - 3

1. Derive the gradient flow through masked attention and analyze potential issues.
2. Analyze the trade-offs between using high-resolution features vs. deep features in the decoder.
3. How would you extend Mask2Former to video segmentation with temporal consistency?

## Practice Problems

### Easy - 5

1. Implement a binary mask for attention masking.
2. Compute the number of parameters in a MaskedCrossAttention layer.
3. Upsample a mask prediction to feature map resolution.
4. Implement multi-scale feature concatenation.
5. Compute output dimensions for multi-scale decoder.

### Medium - 5

1. Implement masked cross-attention.
2. Build the multi-scale transformer decoder.
3. Implement the Mask2Former prediction head.
4. Write a Mask2Former training iteration.
5. Compare MaskFormer vs. Mask2Former inference.

### Hard - 3

1. Implement a complete Mask2Former architecture.
2. Design a video Mask2Former with temporal masked attention.
3. Analyze the mask prediction evolution across decoder layers.

## Solutions

Easy 1:
```python
def create_attention_mask(mask_logits, threshold=0.0):
    # mask_logits: [N, Q, H, W]
    mask = torch.sigmoid(mask_logits) > threshold  # [N, Q, H, W]
    N, Q, H, W = mask.shape
    mask_flat = mask.view(N, Q, H * W).float()  # 1 = attend, 0 = ignore
    return mask_flat

logits = torch.randn(2, 100, 32, 32)
attn_mask = create_attention_mask(logits)
print(f"Attention mask: {attn_mask.shape}")
# Output: Attention mask: torch.Size([2, 100, 1024])
```

Medium 1 — Masked Attention:
```python
class MaskedAttn(nn.Module):
    def __init__(self, d=256, h=8):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.h = h

    def forward(self, q, kv, mask=None):
        N, Q, _ = q.shape
        q = self.q(q).view(N, Q, self.h, -1).transpose(1, 2)
        k = self.k(kv).view(N, -1, self.h, -1).transpose(1, 2)
        v = self.v(kv).view(N, -1, self.h, -1).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
        if mask is not None:
            m = F.interpolate(mask[:, None].float(), size=kv.shape[1:2], mode='nearest').squeeze(1)
            attn = attn + (m.unsqueeze(1) == 0).float() * -1e9
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(N, Q, -1)
        return self.out(out)

print("Masked attention defined")
# Output: Masked attention defined
```

## Related Concepts

- DL-261: MaskFormer
- DL-247: DETR
- DL-251: Semantic Segmentation
- DL-253: Panoptic Segmentation

## Next Concepts

- DL-263: Segmentation Metrics
- DL-264: Medical Image Segmentation

## Summary

Mask2Former improves upon MaskFormer with masked attention that focuses each query on its predicted region, multi-scale high-resolution features, and optimization improvements. Masked attention speeds convergence by 3x and improves both accuracy and efficiency. Mask2Former achieved state-of-the-art results across semantic, instance, and panoptic segmentation simultaneously, establishing the current best practice for transformer-based unified segmentation.

## Key Takeaways

- Masked attention: queries attend only to their predicted mask region
- 3x faster convergence than MaskFormer
- Multi-scale features processed efficiently with pooling sampling
- State-of-the-art on all three segmentation tasks
- 57.8% PQ on COCO panoptic segmentation
- Mask prediction feedback loop across decoder layers
- High-resolution features preserved for fine boundary details
- Per-pixel loss replaced with per-mask loss
