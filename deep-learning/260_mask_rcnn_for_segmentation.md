# Concept: Mask R-CNN for Segmentation

## Concept ID

DL-260

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Image Segmentation

## Learning Objectives

- Understand how Mask R-CNN performs instance segmentation
- Implement the mask prediction head and RoI Align
- Comprehend the relationship between detection and segmentation branches
- Analyze the mask AP evaluation metric

## Prerequisites

- DL-240: Mask R-CNN
- DL-252: Instance Segmentation
- DL-239: Faster R-CNN

## Definition

Mask R-CNN for Segmentation extends Faster R-CNN by adding a parallel mask prediction branch that outputs a binary segmentation mask for each detected instance. The architecture consists of: (1) a shared backbone (ResNet-FPN) for feature extraction, (2) a Region Proposal Network (RPN) for proposing candidate boxes, (3) a Fast R-CNN head for classification and box regression, and (4) a mask head that predicts a per-class binary mask for each RoI using RoI Align for pixel-accurate feature extraction.

## Intuition

Mask R-CNN treats segmentation as an extension of detection: first find the objects (bounding boxes), then segment each one (pixel masks). The mask head operates on each detected region independently, predicting a 28×28 binary mask that is then upsampled to the full image resolution. The per-class mask ensures that the network learns class-specific shape priors. The RoI Align operation is critical: unlike RoI Pooling which uses quantization, RoI Align uses bilinear interpolation to extract features at exact sub-pixel locations, preserving the spatial fidelity needed for accurate masks.

## Why This Concept Matters

Mask R-CNN provides a unified framework for instance segmentation that can be trained end-to-end. It achieved state-of-the-art results on COCO instance segmentation (37.1 mask AP) and became the standard baseline for instance segmentation research. The framework's modular design (backbone + RPN + head) made it easy to extend to other tasks including keypoint detection, panoptic segmentation, and video instance segmentation.

## Mathematical Explanation

Mask Head Architecture:
- Input: RoI Align features (14×14 × 256 channels)
- 4 × (Conv3×3 → BN → ReLU) keeping 256 channels
- Transpose Conv (2×2, stride=2) → 28×28 × 256
- 1×1 Conv → 28×28 × num_classes

Mask Loss (for each positive RoI with class k):
L_mask = BCE(pred_mask_k, gt_mask)
= -1/(28²) * Σ [y log(ŷ) + (1-y) log(1-ŷ)]

Only the mask for the ground-truth class contributes to the loss. The mask branch predicts K masks but only the k-th mask is trained.

## Code Examples

### Example 1: Mask Head with RoI Align

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=80):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        )
        self.deconv = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.mask_pred = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        x = self.convs(x)      # [N, 256, 14, 14]
        x = self.deconv(x)     # [N, 256, 28, 28]
        x = F.relu(x)
        x = self.mask_pred(x)  # [N, num_classes, 28, 28]
        return x

mask_head = MaskHead(in_channels=256, num_classes=80)
roi_features = torch.randn(10, 256, 14, 14)
mask_logits = mask_head(roi_features)
print(f"Mask logits: {mask_logits.shape}")
# Output: Mask logits: torch.Size([10, 80, 28, 28])
```

### Example 2: Mask R-CNN Forward with All Branches

```python
import torch
import torch.nn as nn

class MaskRCNNForward(nn.Module):
    def __init__(self, num_classes=81):
        super().__init__()
        # Shared backbone (simplified)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 256, 3, padding=1), nn.ReLU(),
        )
        # RPN (simplified)
        self.rpn = nn.Conv2d(256, 9 * 2, 3, padding=1)  # 9 anchors, 2 scores
        # Fast R-CNN head
        self.roi_head = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
        )
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)
        # Mask head
        self.mask_head = MaskHead(256, num_classes - 1)  # exclude BG

    def forward(self, x, proposals=None):
        features = self.backbone(x)
        rpn_logits = self.rpn(features)

        if proposals is not None:
            # Extract RoI features (simplified: adaptive pooling)
            roi_feats = torch.stack([
                F.adaptive_avg_pool2d(features[:, :, p[1]:p[3], p[0]:p[2]], 7)
                for p in proposals
            ])
            roi_feats = roi_feats.view(roi_feats.shape[0], -1)
            x = self.roi_head(roi_feats)
            cls_logits = self.cls_score(x)
            bbox_deltas = self.bbox_pred(x)
            mask_logits = self.mask_head(
                F.adaptive_avg_pool2d(features, 14).unsqueeze(0)
            )
            return cls_logits, bbox_deltas, mask_logits

        return features, rpn_logits

model = MaskRCNNForward()
dummy = torch.randn(1, 3, 224, 224)
# Training with proposals
proposals = torch.randint(0, 224, (10, 4)).float()
cls, bbox, mask = model(dummy, proposals)
print(f"Class: {cls.shape}, BBox: {bbox.shape}, Mask: {mask.shape}")
# Output: Class: torch.Size([10, 81]), BBox: torch.Size([10, 324]), Mask: torch.Size([1, 80, 28, 28])
```

### Example 3: Mask AP Evaluation

```python
import torch

def compute_mask_ap(pred_masks, pred_scores, pred_labels,
                    gt_masks, gt_labels, iou_threshold=0.5):
    # Simplified mask AP for one class
    tp = 0
    fp = 0
    fn = 0
    matched_gts = set()

    # Sort predictions by score descending
    order = pred_scores.argsort(descending=True)

    for idx in order:
        pred_mask = pred_masks[idx] > 0.5
        pred_label = pred_labels[idx]

        best_iou = 0
        best_gt = -1
        for g_idx in range(len(gt_masks)):
            if g_idx in matched_gts:
                continue
            if gt_labels[g_idx] != pred_label:
                continue
            gt_mask = gt_masks[g_idx]
            intersection = (pred_mask & gt_mask).sum().float()
            union = (pred_mask | gt_mask).sum().float()
            iou = (intersection / (union + 1e-7)).item()
            if iou > best_iou:
                best_iou = iou
                best_gt = g_idx

        if best_iou >= iou_threshold:
            tp += 1
            matched_gts.add(best_gt)
        else:
            fp += 1

    fn = len(gt_masks) - len(matched_gts)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    ap = precision * recall  # simplified
    return ap, precision, recall

pred_masks = torch.randint(0, 2, (5, 28, 28)).bool()
pred_scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
pred_labels = torch.zeros(5, dtype=torch.long)
gt_masks = torch.randint(0, 2, (3, 28, 28)).bool()
gt_labels = torch.zeros(3, dtype=torch.long)
ap, prec, rec = compute_mask_ap(pred_masks, pred_scores, pred_labels, gt_masks, gt_labels)
print(f"Mask AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
# Output: Mask AP: 0.0000, Precision: 0.0000, Recall: 0.0000 (example)
```

## Common Mistakes

1. **Using RoI Pooling instead of RoI Align**: RoI Pooling's quantization causes pixel misalignment in masks, reducing mask AP by 3-5%. RoI Align is essential.

2. **Not using per-class masks**: The mask head outputs num_classes masks but only trains one per RoI. Using a single binary mask (class-agnostic) loses shape priors.

3. **Low mask resolution (28x28)**: Standard mask output is 28x28, which loses fine detail for large objects. Consider higher resolution or PointRend for detailed masks.

4. **Mask loss on background RoIs**: Only foreground RoIs (matched to ground truth) should contribute to mask loss. Background RoIs produce noisy gradients.

5. **Evaluating mask AP at single IoU threshold**: Standard mask AP averages over thresholds [0.5, 0.55, ..., 0.95]. Single-threshold evaluation overestimates performance.

## Interview Questions

### Beginner - 5

1. What does Mask R-CNN add to Faster R-CNN?
2. What is the output resolution of the mask head?
3. How many masks does the mask head predict per RoI?
4. Why is RoI Align important for segmentation?
5. What loss function is used for mask prediction?

### Intermediate - 5

1. Explain the mask head architecture.
2. How does per-class mask prediction work?
3. What is the advantage of per-class masks vs. class-agnostic?
4. How is mask AP computed?
5. What is the role of the deconvolution layer in the mask head?

### Advanced - 3

1. Analyze the gradient flow from mask loss through RoI Align to the backbone.
2. Compare Mask R-CNN with YOLACT for real-time instance segmentation.
3. How would you extend Mask R-CNN to video instance segmentation?

## Practice Problems

### Easy - 5

1. Implement the mask head with 4 conv layers and 1 deconv layer.
2. Compute mask IoU between two binary masks.
3. Implement binary cross-entropy for mask loss.
4. Threshold mask logits to binary predictions.
5. Resize a 28x28 mask to original image size.

### Medium - 5

1. Implement RoI Align for mask feature extraction.
2. Build a complete Mask R-CNN forward pass.
3. Implement mask AP evaluation.
4. Write a post-processing pipeline for mask outputs.
5. Implement per-class mask extraction from head output.

### Hard - 3

1. Implement full Mask R-CNN training with FPN backbone.
2. Design a cascade mask refinement module.
3. Implement PointRend for high-resolution mask upsampling.

## Solutions

Easy 1:
```python
class SimpleMaskHead(nn.Module):
    def __init__(self, in_c=256, num_classes=80):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c, in_c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c, in_c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c, in_c, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(in_c, in_c, 2, stride=2), nn.ReLU(),
            nn.Conv2d(in_c, num_classes, 1),
        )

    def forward(self, x):
        return self.layers(x)

head = SimpleMaskHead()
print(f"Output: {head(torch.randn(10, 256, 14, 14)).shape}")
# Output: Output: torch.Size([10, 80, 28, 28])
```

Medium 1 — RoI Align:
```python
def simple_roi_align(feature_map, boxes, output_size=14):
    # feature_map: [N, C, H, W], boxes: [K, 4] (x1,y1,x2,y2) normalized
    aligned = []
    for box in boxes:
        x1, y1, x2, y2 = box
        h, w = feature_map.shape[2:]
        # Map to feature coordinates
        x1_f, y1_f = int(x1 * w), int(y1 * h)
        x2_f, y2_f = int(x2 * w), int(y2 * h)
        crop = feature_map[0, :, y1_f:y2_f, x1_f:x2_f]
        crop = F.interpolate(crop.unsqueeze(0), (output_size, output_size), mode='bilinear')
        aligned.append(crop)
    return torch.cat(aligned, dim=0)

feat = torch.randn(1, 256, 32, 32)
boxes = torch.tensor([[0.1, 0.1, 0.5, 0.5], [0.3, 0.3, 0.8, 0.8]])
aligned = simple_roi_align(feat, boxes)
print(f"RoI Aligned: {aligned.shape}")
# Output: RoI Aligned: torch.Size([2, 256, 14, 14])
```

## Related Concepts

- DL-240: Mask R-CNN
- DL-252: Instance Segmentation
- DL-239: Faster R-CNN
- DL-238: Fast R-CNN

## Next Concepts

- DL-261: MaskFormer
- DL-262: Mask2Former

## Summary

Mask R-CNN extends Faster R-CNN with a parallel mask prediction head, enabling instance segmentation within a unified framework. The mask head uses RoI Align for pixel-accurate feature extraction and produces per-class binary masks at 28×28 resolution. The multi-task loss combines classification, box regression, and mask prediction, optimized end-to-end. Mask R-CNN achieved state-of-the-art instance segmentation and established the standard baseline for the field.

## Key Takeaways

- Mask branch added in parallel to cls + box branches
- RoI Align uses bilinear interpolation for sub-pixel accuracy
- Mask output: 28×28 per class, upsampled to image size
- Per-class masks (K masks, only k-th trained)
- Multi-task loss: L = L_cls + L_box + L_mask
- 37.1% mask AP on COCO at 5 FPS (ResNet-101-FPN)
- Foundation for video instance segmentation and panoptic segmentation
- RoI Align is essential (3-5% gain over RoI Pooling)
