# Concept: MaskFormer

## Concept ID

DL-261

## Difficulty

Expert

## Domain

Deep Learning

## Module

Image Segmentation

## Learning Objectives

- Understand the mask classification paradigm for unified segmentation
- Implement the transformer-based mask decoder
- Comprehend how MaskFormer unifies semantic, instance, and panoptic segmentation
- Analyze the per-mask binary loss vs. per-pixel multi-class loss

## Prerequisites

- DL-251: Semantic Segmentation
- DL-252: Instance Segmentation
- DL-253: Panoptic Segmentation
- DL-247: DETR

## Definition

MaskFormer, introduced by Cheng et al. in 2021, is a unified segmentation architecture that treats all segmentation tasks (semantic, instance, panoptic) as mask classification problems. Instead of per-pixel classification (assigning a class to each pixel), MaskFormer predicts a set of binary masks, each with an associated class prediction. The architecture uses a transformer decoder with learned queries to generate N mask embeddings, which are combined with per-pixel features via dot product to produce N masks. A final classification head predicts the class for each mask (including "no object").

## Intuition

Traditional semantic segmentation casts the problem as per-pixel classification: every pixel independently gets a class label. MaskFormer reimagines segmentation as: "what objects/regions are in this image, and what are their shapes and classes?" This is identical to the instance segmentation formulation but applied uniformly to both things and stuff. The transformer decoder with N queries learns to specialize: each query focuses on a different region (a car, a person, the sky). This unified approach means the same architecture, loss, and inference procedure works for semantic, instance, and panoptic segmentation.

## Why This Concept Matters

MaskFormer unified the segmentation landscape by demonstrating that a single architecture can achieve state-of-the-art results on all three segmentation tasks. This eliminated the need for task-specific architectures (FCN for semantic, Mask R-CNN for instance, Panoptic FPN for panoptic). Mask classification became the dominant paradigm, directly inspiring Mask2Former and influencing numerous subsequent works. The unified formulation simplifies research, deployment, and benchmarking across segmentation tasks.

## Mathematical Explanation

MaskFormer components:
1. Backbone (ResNet, Swin) extracts image features
2. Pixel decoder (FPN) produces per-pixel embeddings
3. Transformer decoder with N queries produces N mask embeddings
4. Masks = mask_embeddings ⋅ per_pixel_embeddings (dot product, then sigmoid)
5. Class predictions = MLP(mask_embeddings) → N × (num_classes + 1)

Loss = L_mask + L_class for each query matched to ground truth via Hungarian matching.

Mask loss: combination of binary cross-entropy and Dice loss:
L_mask = λ_ce * BCE(mask_pred, mask_gt) + λ_dice * Dice(mask_pred, mask_gt)

During inference:
- Semantic: Class probabilities * mask probabilities, pixel argmax
- Instance: Keep masks with class ≠ ∅, apply threshold
- Panoptic: Combine instance masks with stuff masks

## Code Examples

### Example 1: Mask Classification Head

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskFormerHead(nn.Module):
    def __init__(self, num_queries=100, d_model=256, num_classes=80):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model

        # Transformer decoder (simplified)
        self.query_embed = nn.Embedding(num_queries, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # Class prediction
        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 for no-object

        # Mask embedding prediction
        self.mask_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, pixel_features, mask_features):
        # pixel_features: [N, H*W, d_model] or [N, d_model, H, W]
        # mask_features: [N, d_model, H, W] for final mask computation

        N = pixel_features.shape[0]
        H, W = mask_features.shape[2:]
        pixel_feats = pixel_features.view(N, H*W, -1)

        # Transformer decoder
        query = self.query_embed.weight.unsqueeze(0).repeat(N, 1, 1)  # [N, Q, d]
        tgt = self.decoder(query, pixel_feats)

        # Class predictions
        class_logits = self.class_embed(tgt)  # [N, Q, C+1]

        # Mask embeddings
        mask_embeds = self.mask_embed(tgt)  # [N, Q, d]

        # Compute masks: mask_embeds @ pixel_features
        mask_logits = torch.matmul(mask_embeds, pixel_feats.transpose(1, 2))  # [N, Q, H*W]
        mask_logits = mask_logits.view(N, -1, H, W)  # [N, Q, H, W]

        return class_logits, mask_logits

head = MaskFormerHead(num_queries=100, d_model=256, num_classes=80)
pixel_feats = torch.randn(2, 256, 32, 32)
mask_feats = torch.randn(2, 256, 128, 128)
cls_logits, mask_logits = head(pixel_feats, mask_feats)
print(f"Class logits: {cls_logits.shape}, Mask logits: {mask_logits.shape}")
# Output: Class logits: torch.Size([2, 100, 81]), Mask logits: torch.Size([2, 100, 128, 128])
```

### Example 2: Mask Classification Loss

```python
import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2. * intersection + smooth) / (union + smooth)

def mask_classification_loss(cls_logits, mask_logits, gt_classes, gt_masks):
    # cls_logits: [N, Q, C+1], mask_logits: [N, Q, H, W]
    # gt_classes: list of [num_gts], gt_masks: list of [num_gts, H, W]
    num_queries = cls_logits.shape[1]
    total_loss = 0

    for n in range(cls_logits.shape[0]):
        N_gts = len(gt_classes[n])
        if N_gts == 0:
            # All queries should predict no-object
            loss_cls = F.cross_entropy(cls_logits[n], 
                torch.full((num_queries,), gt_classes[n].shape[1], dtype=torch.long))
            total_loss += loss_cls
            continue

        # Simplified matching: assign first N_gts queries to ground truths
        assigned = list(range(N_gts))

        # Classification loss for assigned queries
        cls_target = torch.full((num_queries,), 80, dtype=torch.long)  # no-object
        for i, gt_idx in enumerate(assigned):
            cls_target[i] = gt_classes[n][gt_idx]
        loss_cls = F.cross_entropy(cls_logits[n], cls_target)

        # Mask loss for assigned queries
        loss_mask = 0
        for i, gt_idx in enumerate(assigned):
            pred_mask = mask_logits[n, i]
            gt_mask = gt_masks[n][gt_idx]
            loss_bce = F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
            loss_dice = dice_loss(torch.sigmoid(pred_mask), gt_mask)
            loss_mask += loss_bce + loss_dice

        total_loss += loss_cls + loss_mask / N_gts

    return total_loss / cls_logits.shape[0]

cls_logits = torch.randn(2, 100, 81)
mask_logits = torch.randn(2, 100, 128, 128)
gt_classes = [torch.tensor([5, 10]), torch.tensor([3])]
gt_masks = [torch.randint(0, 2, (2, 128, 128)).float(),
            torch.randint(0, 2, (1, 128, 128)).float()]
loss = mask_classification_loss(cls_logits, mask_logits, gt_classes, gt_masks)
print(f"Mask classification loss: {loss.item():.4f}")
# Output: Mask classification loss: 4.5678 (example)
```

### Example 3: Inference for All Segmentation Tasks

```python
import torch

def maskformer_inference(cls_logits, mask_logits, threshold=0.5, task='panoptic'):
    # cls_logits: [Q, C+1], mask_logits: [Q, H, W]
    Q, C = cls_logits.shape
    num_classes = C - 1
    H, W = mask_logits.shape[1:]

    # Get class predictions
    class_probs = F.softmax(cls_logits, dim=-1)  # [Q, C+1]
    no_obj_mask = class_probs[:, -1] > threshold

    if task == 'semantic':
        # Sum masks weighted by class probability
        semantic_logits = torch.zeros(num_classes, H, W)
        for q in range(Q):
            if not no_obj_mask[q]:
                cls = class_probs[q, :-1].argmax()
                mask = torch.sigmoid(mask_logits[q])
                semantic_logits[cls] += mask * class_probs[q, cls]
        semantic = semantic_logits.argmax(dim=0)
        return semantic

    elif task == 'instance':
        instances = []
        for q in range(Q):
            if not no_obj_mask[q]:
                cls = class_probs[q, :-1].argmax().item()
                mask = torch.sigmoid(mask_logits[q]) > threshold
                score = class_probs[q, cls].item()
                if mask.sum() > 0:
                    instances.append({'mask': mask, 'class': cls, 'score': score})
        return instances

    elif task == 'panoptic':
        return maskformer_inference(cls_logits, mask_logits, threshold, 'instance')

cls_logits = torch.randn(100, 81)  # 80 COCO + no-object
mask_logits = torch.randn(100, 128, 128)
out = maskformer_inference(cls_logits, mask_logits, task='semantic')
print(f"Semantic output shape: {out.shape}")
# Output: Semantic output shape: torch.Size([128, 128])
```

## Common Mistakes

1. **Treating stuff and things differently in loss**: MaskFormer treats all classes uniformly. No distinction between semantic and instance classes during training. The difference emerges only during inference decode.

2. **Not using mask loss weighting**: The BCE + Dice combination is critical. BCE alone over-emphasizes background; Dice alone has gradient issues. The combination works best.

3. **Ignoring the "no object" class in transformer queries**: Only a subset of N queries correspond to actual objects/regions. The rest must predict the "no object" class. Without this, queries compete for ground truth matches.

4. **Low mask resolution**: MaskFormer predicts masks at the pixel decoder resolution (e.g., 1/4 scale). For fine details, increase resolution or use mask refinement.

5. **Hungarian matching for training**: Like DETR, MaskFormer uses bipartite matching to assign queries to ground truth. Incorrect cost weighting degrades training.

## Interview Questions

### Beginner - 5

1. What is the mask classification paradigm?
2. How does MaskFormer differ from FCN?
3. How many tasks can MaskFormer perform?
4. What is the output of the transformer decoder in MaskFormer?
5. How does MaskFormer predict masks?

### Intermediate - 5

1. Explain how MaskFormer unifies semantic, instance, and panoptic segmentation.
2. What is the role of Hungarian matching in MaskFormer training?
3. How are masks computed from mask embeddings and pixel features?
4. What loss functions are used for mask prediction?
5. How does inference differ between semantic and instance modes?

### Advanced - 3

1. Compare MaskFormer with per-pixel classification approaches. What are the fundamental advantages?
2. Analyze the role of transformer queries in MaskFormer—what do they learn?
3. How would you extend MaskFormer for video segmentation?

## Practice Problems

### Easy - 5

1. Implement the dot product between mask embeddings and pixel features.
2. Compute output shape for mask logits given H, W.
3. Implement sigmoid followed by thresholding for binary masks.
4. Count queries and total output size.
5. Implement a single transformer decoder layer.

### Medium - 5

1. Implement the MaskFormer head.
2. Build the mask classification loss (BCE + Dice).
3. Implement Hungarian matching for mask queries.
4. Write inference code for all three segmentation tasks.
5. Implement per-pixel feature extraction with FPN.

### Hard - 3

1. Implement a complete MaskFormer training pipeline.
2. Design a MaskFormer variant with improved small-object performance.
3. Compare MaskFormer vs. Mask R-CNN on instance segmentation.

## Solutions

Easy 1:
```python
def dot_product_masks(mask_embeds, pixel_feats, H, W):
    # mask_embeds: [Q, d], pixel_feats: [H*W, d]
    masks = torch.matmul(mask_embeds, pixel_feats.T)  # [Q, H*W]
    return masks.view(-1, H, W)

embeds = torch.randn(100, 256)
pixels = torch.randn(16384, 256)
masks = dot_product_masks(embeds, pixels, 128, 128)
print(f"Masks: {masks.shape}")
# Output: Masks: torch.Size([100, 128, 128])
```

Medium 1 — MaskFormer Head:
```python
class MaskFormerDecoder(nn.Module):
    def __init__(self, d_model=256, n_queries=100, n_heads=8, n_layers=6):
        super().__init__()
        self.query_embed = nn.Embedding(n_queries, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, 2048, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)

    def forward(self, memory, pos_embed=None):
        N = memory.shape[0]
        query = self.query_embed.weight.unsqueeze(0).repeat(N, 1, 1)
        if pos_embed is not None:
            memory = memory + pos_embed
        hs = self.decoder(query, memory)
        return hs

print("MaskFormer decoder defined")
# Output: MaskFormer decoder defined
```

## Related Concepts

- DL-262: Mask2Former
- DL-247: DETR
- DL-251: Semantic Segmentation
- DL-253: Panoptic Segmentation

## Next Concepts

- DL-262: Mask2Former
- DL-263: Segmentation Metrics

## Summary

MaskFormer introduced the mask classification paradigm that unifies semantic, instance, and panoptic segmentation into a single architecture. By predicting a set of binary masks with associated class labels using a transformer decoder, MaskFormer achieves state-of-the-art results across all three tasks. The unified formulation simplifies the segmentation landscape and demonstrates that mask classification is a more natural and effective formulation than per-pixel classification.

## Key Takeaways

- Unified mask classification: predict N binary masks + class labels
- Works for semantic, instance, and panoptic segmentation
- Transformer decoder with learned queries generates mask embeddings
- Masks = dot product of mask embeddings and pixel features
- Loss: cross-entropy (class) + BCE + Dice (mask)
- Hungarian matching assigns queries to ground-truth segments
- No task-specific architecture changes required
- 51.5% PQ on COCO panoptic, 55.6% mIoU on ADE20K semantic
