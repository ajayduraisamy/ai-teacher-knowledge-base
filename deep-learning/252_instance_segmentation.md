# Concept: Instance Segmentation

## Concept ID

DL-252

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Image Segmentation

## Learning Objectives

- Understand how instance segmentation differs from semantic segmentation
- Implement the Mask R-CNN approach to instance segmentation
- Comprehend the challenges of separating overlapping instances
- Analyze the trade-offs between top-down and bottom-up approaches

## Prerequisites

- DL-251: Semantic Segmentation
- DL-240: Mask R-CNN
- DL-239: Faster R-CNN

## Definition

Instance Segmentation is the task of detecting and segmenting each individual object instance in an image, simultaneously producing a class label and a pixel-wise mask for every distinct object. Unlike semantic segmentation which treats all pixels of the same class identically, instance segmentation separates overlapping objects of the same class (e.g., distinguishing three individual cars). The output is a set of {class, mask} pairs where each mask is a binary map covering exactly one object instance.

## Intuition

Instance segmentation combines object detection (separating instances) with semantic segmentation (pixel-level labeling). Think of it as first finding each object with a bounding box (detection), then precisely outlining its shape within that box (segmentation). The key challenge is separating touching or overlapping instances of the same class—something semantic segmentation cannot do. The dominant approach (Mask R-CNN) uses a detect-then-segment pipeline: detect objects with bounding boxes, then segment each box region independently.

## Why This Concept Matters

Instance segmentation provides the most detailed scene understanding among detection-level tasks, answering "what objects are where, what are their shapes, and which pixels belong to which instance?" It's critical for autonomous driving (individual vehicle and pedestrian shapes), medical imaging (individual cell or tumor boundaries), robotic manipulation (grasping specific objects), and video editing (isolating specific foreground objects). Advances in instance segmentation directly power applications requiring precise object isolation.

## Mathematical Explanation

Instance segmentation can be formulated as:

For each ground-truth instance i with class c_i and mask m_i (binary H×W), the model predicts (ĉ_i, b̂_i, m̂_i) where b̂_i is the bounding box and m̂_i is the mask.

Mask R-CNN loss:
L = L_cls + L_box + L_mask

where the mask loss is per-pixel binary cross-entropy for each foreground RoI:
L_mask = -1/(K×K) Σ_{i,j} [y_ij log(ŷ_ij) + (1-y_ij) log(1-ŷ_ij)]

Key distinction from semantic segmentation: masks are predicted per-instance (binary) rather than per-pixel (multi-class). The mask branch predicts K masks for K classes, but only the mask for the ground-truth class contributes to the loss.

## Code Examples

### Example 1: Instance Segmentation with Mask R-CNN

```python
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn

model = maskrcnn_resnet50_fpn(pretrained=False)
model.eval()

dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(dummy)

print(f"Output keys: {output[0].keys()}")
print(f"Masks shape: {output[0]['masks'].shape}")
print(f"Boxes shape: {output[0]['boxes'].shape}")
# Output:
# Output keys: dict_keys(['boxes', 'labels', 'scores', 'masks'])
# Masks shape: torch.Size([100, 1, 224, 224])
# Boxes shape: torch.Size([100, 4])

# Extract first mask
mask = output[0]['masks'][0, 0]
print(f"Mask value range: [{mask.min().item():.3f}, {mask.max().item():.3f}]")
# Output: Mask value range: [0.000, 0.008] (example)
```

### Example 2: Converting Instance Masks to a Semantic Map

```python
import torch

def instance_to_semantic(boxes, masks, labels, scores, num_classes=80, threshold=0.5):
    # boxes: [N, 4]
    # masks: [N, 1, H, W]
    # labels: [N]
    N, _, H, W = masks.shape
    semantic = torch.zeros((H, W), dtype=torch.long)

    # Sort by score descending so higher-confidence objects overwrite lower ones
    order = scores.argsort(descending=True)

    for idx in order:
        label = labels[idx].item()
        mask = masks[idx, 0] > threshold
        semantic[mask] = label

    return semantic

masks = torch.sigmoid(torch.randn(5, 1, 224, 224))
boxes = torch.rand(5, 4)
labels = torch.randint(0, 20, (5,))
scores = torch.rand(5)
semantic = instance_to_semantic(boxes, masks, labels, scores)
print(f"Semantic map unique classes: {semantic.unique().tolist()}")
# Output: Semantic map unique classes: [0, 3, 5, 12, 17] (example)
```

### Example 3: Mask Post-processing

```python
import torch
import torch.nn.functional as F

def refine_mask(mask_logits, threshold=0.5):
    """Convert mask logits to binary mask with optional refinement"""
    mask = torch.sigmoid(mask_logits)
    binary_mask = (mask > threshold).float()

    # Remove small isolated regions (connected components)
    # Simplified: just thresholding
    return binary_mask

def mask_iou(pred_mask, gt_mask, threshold=0.5):
    pred_bin = (pred_mask > threshold).bool()
    gt_bin = gt_mask.bool()
    intersection = (pred_bin & gt_bin).sum().float()
    union = (pred_bin | gt_bin).sum().float()
    return (intersection / (union + 1e-7)).item()

pred = torch.sigmoid(torch.randn(28, 28))
gt = torch.randint(0, 2, (28, 28)).float()
print(f"Mask IoU: {mask_iou(pred, gt):.4f}")
# Output: Mask IoU: 0.1532 (example)
```

## Common Mistakes

1. **Training instance segmentation from scratch**: Always use pretrained detection weights. Training instance segmentation from scratch requires massive datasets and compute.

2. **Using class-agnostic masks**: Predicting a single mask per RoI instead of per-class masks loses the ability to encode class-specific shape priors.

3. **Not handling overlapping instances in evaluation**: When instances overlap, the order of mask rendering matters for pixel-wise metrics. Consistent evaluation requires sorted output (e.g., by score).

4. **Low mask resolution (28×28)**: Standard Mask R-CNN outputs 28×28 masks, which lose detail for large objects. Use higher resolution or PointRend for fine details.

5. **Ignoring mask quality vs. detection quality**: A model may have good detection mAP but poor mask mAP. Always evaluate both components separately.

## Interview Questions

### Beginner - 5

1. What is instance segmentation?
2. How is it different from semantic segmentation?
3. What is the dominant architecture for instance segmentation?
4. What is the output of an instance segmentation model?
5. How many masks does Mask R-CNN predict per proposal?

### Intermediate - 5

1. Explain the detect-then-segment paradigm.
2. How does Mask R-CNN handle overlapping instances of the same class?
3. What is the role of RoI Align in mask prediction?
4. How is mask loss computed differently from semantic segmentation loss?
5. How do you evaluate instance segmentation?

### Advanced - 3

1. Compare top-down (detect-then-segment) vs. bottom-up (cluster-then-classify) approaches for instance segmentation.
2. How does SOLO/YOLACT approach instance segmentation without bounding boxes?
3. What are the limitations of Mask R-CNN for instance segmentation in crowded scenes?

## Practice Problems

### Easy - 5

1. Implement binary cross-entropy loss for mask prediction.
2. Write a function to apply threshold to mask logits.
3. Compute mask IoU between two masks.
4. Count the number of predicted instances in an output.
5. Resize a predicted mask to original image size.

### Medium - 5

1. Build a simple instance segmentation head.
2. Implement the mask loss for Mask R-CNN.
3. Write an evaluation loop computing mask AP.
4. Convert instance masks to a semantic segmentation map.
5. Implement mask NMS (non-maximum suppression of masks).

### Hard - 3

1. Implement a bottom-up instance segmentation method (e.g., embedding-based clustering).
2. Implement PointRend for high-resolution mask refinement.
3. Design an instance segmentation method for video.

## Solutions

Easy 1:
```python
def mask_bce_loss(pred_mask, gt_mask):
    loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
    return loss

pred = torch.randn(10, 28, 28)
gt = torch.randint(0, 2, (10, 28, 28)).float()
print(f"Mask BCE loss: {mask_bce_loss(pred, gt):.4f}")
# Output: Mask BCE loss: 0.6938 (example)
```

Medium 1 — Simple Instance Head:
```python
class InstanceMaskHead(nn.Module):
    def __init__(self, in_channels=256, roi_size=14, num_classes=80):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 2, stride=2), nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x):
        return self.conv(x)

head = InstanceMaskHead()
x = torch.randn(10, 256, 14, 14)
print(f"Mask head output: {head(x).shape}")
# Output: Mask head output: torch.Size([10, 80, 28, 28])
```

## Related Concepts

- DL-251: Semantic Segmentation
- DL-240: Mask R-CNN
- DL-253: Panoptic Segmentation

## Next Concepts

- DL-253: Panoptic Segmentation
- DL-260: Mask R-CNN for Segmentation

## Summary

Instance segmentation combines detection and segmentation to produce per-instance pixel masks. The dominant Mask R-CNN approach uses a detect-then-segment pipeline with RoI Align for pixel-accurate mask extraction. The per-class mask branch and binary cross-entropy loss enable high-quality instance separation. Instance segmentation provides the most detailed object-level scene understanding and is critical for applications requiring precise object isolation.

## Key Takeaways

- Instance segmentation = detection + per-instance pixel mask
- Mask R-CNN: top-down detect-then-segment approach
- Masks are binary (per-instance), not multi-class
- RoI Align essential for pixel-accurate mask alignment
- Mask output: 28×28 per object, upsampled to image size
- Mask AP (average over IoU thresholds) is the evaluation metric
- Overlapping instances of same class are distinguished
- Challenges: crowded scenes, fine details, inference speed
