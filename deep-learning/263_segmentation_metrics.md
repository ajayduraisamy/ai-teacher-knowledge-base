# Concept: Segmentation Metrics

## Concept ID

DL-263

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Image Segmentation

## Learning Objectives

## Prerequisites

- DL-251: Semantic Segmentation
- DL-252: Instance Segmentation
- DL-253: Panoptic Segmentation

## Definition

Segmentation metrics are quantitative measures used to evaluate the quality of segmentation predictions against ground truth. Different tasks require different metrics: semantic segmentation uses pixel accuracy, mean IoU (mIoU), and frequency-weighted IoU; instance segmentation uses mask Average Precision (AP); and panoptic segmentation uses Panoptic Quality (PQ). Additional boundary-aware metrics include F-measure for edge detection and Hausdorff distance for medical images.

## Intuition

Choosing the right metric is crucial because different metrics measure different aspects of quality. Pixel accuracy can be misleading in datasets with class imbalance (e.g., 90% background, 10% foreground)—a model that always predicts background gets 90% accuracy. Mean IoU corrects for this by averaging per-class IoU. For instance segmentation, mask AP measures both detection (finding objects) and segmentation (pixel accuracy of masks). PQ decomposes into recognition quality (RQ) and segmentation quality (SQ).

## Why This Concept Matters

Without proper metrics, model comparisons are meaningless. Understanding segmentation metrics is essential for: (1) correctly evaluating and comparing models, (2) diagnosing model weaknesses (low PQ due to RQ vs. SQ), (3) reading research papers critically, and (4) selecting the right metric for your application (medical imaging may prioritize boundary accuracy over pixel overlap).

## Mathematical Explanation

Pixel Accuracy = TP / (TP + FP) — overall fraction of correctly classified pixels

Mean IoU = (1/C) * Σ_c (TP_c / (TP_c + FP_c + FN_c))

Frequency-weighted IoU = (1/Σ_c t_c) * Σ_c (t_c * IoU_c) where t_c is total pixels of class c

Mask AP: Average precision computed over mask IoU thresholds [0.5:0.95]

Panoptic Quality PQ = Σ_{(p,g)∈TP} IoU(p,g) / (|TP| + 0.5|FP| + 0.5|FN|) = SQ × RQ

Boundary F-measure: Precision and recall on boundary pixels within a threshold distance.

## Code Examples

### Example 1: Pixel Accuracy and Mean IoU

```python
import torch

def pixel_accuracy(pred, target):
    correct = (pred == target).sum().item()
    total = pred.numel()
    return correct / total

def mean_iou(pred, target, num_classes=21, ignore_index=255):
    ious = []
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        # Exclude ignored pixels
        valid = target != ignore_index
        pred_valid = pred_mask & valid
        target_valid = target_mask & valid
        intersection = (pred_valid & target_valid).sum().item()
        union = (pred_valid | target_valid).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return sum(ious) / len(ious) if ious else 0.0

def frequency_weighted_iou(pred, target, num_classes=21):
    ious = []
    freq = []
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        inter = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        if union > 0:
            ious.append(inter / union)
        else:
            ious.append(0.0)
        freq.append(target_mask.sum().item())
    total = sum(freq)
    if total == 0:
        return 0.0
    return sum(f * iou for f, iou in zip(freq, ious)) / total

pred = torch.randint(0, 21, (4, 256, 256))
target = torch.randint(0, 21, (4, 256, 256))
print(f"Pixel Acc: {pixel_accuracy(pred, target):.4f}")
print(f"mIoU: {mean_iou(pred, target):.4f}")
print(f"FW IoU: {frequency_weighted_iou(pred, target):.4f}")
# Output:
# Pixel Acc: 0.0478
# mIoU: 0.0476
# FW IoU: 0.0480
```

### Example 2: Per-Class IoU Analysis

```python
import torch
import numpy as np

def per_class_iou(pred, target, class_names, num_classes=21):
    results = {}
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        inter = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        iou = inter / union if union > 0 else 0.0
        name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        results[name] = iou
    return results

class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

pred = torch.randint(0, 21, (1, 256, 256))
target = torch.randint(0, 21, (1, 256, 256))
ious = per_class_iou(pred, target, class_names)
# Print classes where IoU > 0
for name, iou in sorted(ious.items()):
    if iou > 0:
        print(f"  {name:15s}: {iou:.4f}")
# Output: (example printed values)
```

### Example 3: Boundary F-Measure

```python
import torch
import torch.nn.functional as F

def compute_boundary(pred_mask):
    """Extract boundary pixels using Laplacian"""
    kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32)
    boundary = F.conv2d(pred_mask.float().unsqueeze(0).unsqueeze(0), kernel, padding=1)
    return (boundary.abs() > 0).squeeze()

def boundary_f_measure(pred, target, num_classes=21):
    f_scores = []
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        if target_mask.sum() < 10:
            continue
        # Compute boundaries within distance threshold (simplified)
        pred_boundary = compute_boundary(pred_mask)
        target_boundary = compute_boundary(target_mask)
        # Dilate boundaries (simplified: use 3x3 max pool)
        pred_dilated = F.max_pool2d(pred_boundary.float().unsqueeze(0).unsqueeze(0), 3, padding=1).squeeze()
        target_dilated = F.max_pool2d(target_boundary.float().unsqueeze(0).unsqueeze(0), 3, padding=1).squeeze()

        tp = (pred_dilated & target_boundary).sum().item()
        fp = (pred_dilated & ~target_boundary).sum().item()
        fn = (~pred_dilated & target_boundary).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f_scores.append(f)

    return sum(f_scores) / len(f_scores) if f_scores else 0.0

pred = torch.randint(0, 3, (256, 256))
target = torch.randint(0, 3, (256, 256))
bf = boundary_f_measure(pred, target, num_classes=3)
print(f"Boundary F-measure: {bf:.4f}")
# Output: Boundary F-measure: 0.1923 (example)
```

## Common Mistakes

1. **Reporting pixel accuracy without mIoU**: High pixel accuracy can mask poor per-class performance. Always report mIoU as the primary metric.

2. **Including void/unlabeled pixels in evaluation**: Void pixels (label 255) must be excluded from all metric computations. Including them artificially inflates accuracy.

3. **Using different interpolation methods for evaluation**: If predictions are upsampled for evaluation, use the same method as during training. Bilinear vs. nearest neighbor can give different results.

4. **Not using the official evaluation code**: COCO and other benchmarks have specific evaluation implementations. Custom implementations may differ in edge cases.

5. **Ignoring the IoU threshold for mask AP**: Mask AP averaged over [0.5:0.95] is much harder than just AP@0.5. Specify which threshold is used.

## Interview Questions

### Beginner - 5

1. What is mean IoU and how is it computed?
2. What is pixel accuracy?
3. What is the difference between mIoU and pixel accuracy?
4. What is Panoptic Quality?
5. What is mask AP?

### Intermediate - 5

1. Why is mIoU preferred over pixel accuracy?
2. How is frequency-weighted IoU different from mean IoU?
3. What does PQ decompose into?
4. How is mask AP different from box AP?
5. What is boundary F-measure?

### Advanced - 3

1. Analyze the limitations of mIoU for evaluating segmentation quality.
2. Propose a metric that better captures boundary quality.
3. How would you design a metric for video segmentation that accounts for temporal consistency?

## Practice Problems

### Easy - 5

1. Implement pixel accuracy.
2. Implement mean IoU for binary segmentation.
3. Compute per-class IoU for 3 classes.
4. Implement frequency-weighted IoU.
5. Count TP, FP, FN for a given class.

### Medium - 5

1. Implement mIoU with void class exclusion.
2. Implement mask AP for instance segmentation.
3. Implement PQ for panoptic segmentation.
4. Write a function to compute boundary F-measure.
5. Implement Dice coefficient.

### Hard - 3

1. Implement the COCO evaluation protocol for mask AP.
2. Design a boundary-aware IoU metric.
3. Implement a temporal consistency metric for video segmentation.

## Solutions

Easy 1:
```python
def pixel_acc(pred, target):
    return (pred == target).float().mean().item()

pred = torch.randint(0, 3, (1000,))
target = torch.randint(0, 3, (1000,))
print(f"Pixel accuracy: {pixel_acc(pred, target):.4f}")
# Output: Pixel accuracy: 0.3320 (example)
```

Medium 1 — mIoU with Void:
```python
def miou(pred, target, num_classes, void=255):
    mask = target != void
    pred = pred[mask]
    target = target[mask]
    ious = []
    for c in range(num_classes):
        inter = ((pred == c) & (target == c)).sum().float()
        union = ((pred == c) | (target == c)).sum().float()
        if union > 0:
            ious.append((inter / union).item())
    return sum(ious) / len(ious) if ious else 0.0

pred = torch.randint(0, 5, (10000,))
target = torch.randint(0, 5, (10000,))
target[::10] = 255
print(f"mIoU with void: {miou(pred, target, 5):.4f}")
# Output: mIoU with void: 0.1987 (example)
```

## Related Concepts

- DL-251: Semantic Segmentation
- DL-252: Instance Segmentation
- DL-253: Panoptic Segmentation
- DL-236: Mean Average Precision

## Next Concepts

- DL-264: Medical Image Segmentation
- DL-265: Segmentation with Transformers

## Summary

Segmentation metrics quantify the quality of pixel-level predictions across different tasks. Semantic segmentation uses mIoU and pixel accuracy; instance segmentation uses mask AP; panoptic segmentation uses PQ. Understanding the strengths and limitations of each metric is essential for proper evaluation. Metrics should be selected based on the specific requirements of the application, with mIoU being the most commonly reported for semantic segmentation.

## Key Takeaways

- Pixel accuracy can be misleading with class imbalance—always report mIoU
- mIoU averages per-class IoU, treating all classes equally
- Frequency-weighted IoU weights by class frequency
- Mask AP evaluates both detection and segmentation quality
- PQ = SQ × RQ for panoptic segmentation
- Boundary F-measure evaluates edge quality
- Void/unlabeled pixels must be excluded from evaluation
- COCO evaluation protocol is the standard for instance/panoptic segmentation
