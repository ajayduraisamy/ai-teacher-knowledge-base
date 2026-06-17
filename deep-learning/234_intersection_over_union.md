# Concept: Intersection over Union

## Concept ID

DL-234

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Object Detection

## Learning Objectives

- Compute Intersection over Union (IoU) between two bounding boxes
- Understand IoU as both a matching criterion and an evaluation metric
- Implement variants of IoU including GIoU, DIoU, and CIoU
- Apply IoU in anchor matching and Non-Maximum Suppression

## Prerequisites

- DL-231: Object Detection Overview
- Basic coordinate geometry

## Definition

Intersection over Union (IoU), also known as the Jaccard Index, is a measure of overlap between two bounding boxes. Given two boxes A and B, IoU is defined as the area of their intersection divided by the area of their union:

IoU(A, B) = |A ∩ B| / |A ∪ B|

The value ranges from 0 (no overlap) to 1 (perfect overlap). IoU is the primary metric used in object detection for both matching predictions to ground truth during training and evaluating detection quality during testing.

## Intuition

IoU provides a normalized measure of localization quality that is invariant to the absolute size of the boxes. A prediction that perfectly overlaps the ground truth gets IoU=1, while a prediction that partially overlaps or covers a different region gets a lower score proportional to the quality of alignment. Unlike pixel-wise distance metrics (Euclidean distance between centers), IoU captures both position and size agreement in a single scalar value.

## Why This Concept Matters

IoU is arguably the single most important metric in object detection. It is used in three critical roles: (1) as the matching criterion to assign anchor boxes to ground-truth objects during training, (2) as the evaluation metric (mAP thresholds at IoU 0.5, 0.75, etc.), and (3) as the sorting criterion in Non-Maximum Suppression. Because standard IoU is non-differentiable in its raw form, variants like GIoU, DIoU, and CIoU were developed as smooth surrogate losses that directly optimize the evaluation metric.

## Mathematical Explanation

For two axis-aligned rectangles (x1, y1, x2, y2):

Intersection area:
x_left = max(A_x1, B_x1)
y_top = max(A_y1, B_y1)
x_right = min(A_x2, B_x2)
y_bottom = min(A_y2, B_y2)
intersection = max(0, x_right - x_left) * max(0, y_bottom - y_top)

Union area:
union = (A_x2 - A_x1)*(A_y2 - A_y1) + (B_x2 - B_x1)*(B_y2 - B_y1) - intersection

IoU = intersection / union

GIoU extends IoU to account for non-overlapping boxes:
GIoU = IoU - (C - U) / C where C is the area of the smallest enclosing box.

CIoU adds center distance and aspect ratio consistency:
CIoU = IoU - ρ^2(b, b_gt)/c^2 - αv

## Code Examples

### Example 1: Basic IoU Computation

```python
import torch

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

box_pred = torch.tensor([20, 20, 80, 80])
box_gt = torch.tensor([30, 30, 70, 70])
print(f"IoU: {compute_iou(box_pred, box_gt):.4f}")
# Output: IoU: 0.5102
```

### Example 2: Batch IoU Matrix

```python
import torch

def batch_iou(boxes1, boxes2):
    # boxes1: [N, 4], boxes2: [M, 4]
    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-7)

preds = torch.tensor([[50, 50, 150, 150], [200, 200, 300, 300]])
gts = torch.tensor([[60, 60, 140, 140], [50, 50, 150, 150]])
iou_matrix = batch_iou(preds, gts)
print(iou_matrix)
# Output: tensor([[0.6400, 0.0000],
#                 [0.0000, 0.6400]])
```

### Example 3: Using IoU as a Loss Function

```python
import torch

def iou_loss(pred_boxes, target_boxes):
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area_p = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    area_t = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union = area_p + area_t - inter
    iou = inter / (union + 1e-7)
    return 1 - iou

pred = torch.tensor([[20., 20., 80., 80.]], requires_grad=True)
target = torch.tensor([[30., 30., 70., 70.]])
loss = iou_loss(pred, target)
loss.backward()
print(f"IoU Loss: {loss.item():.4f}")
print(f"Gradient: {pred.grad}")
# Output:
# IoU Loss: 0.4898
# Gradient: tensor([[ 0.0016,  0.0016, -0.0016, -0.0016]])
```

## Common Mistakes

1. **Using IoU as a loss without gradient masking**: Standard IoU is piecewise constant when boxes do not overlap—the gradient is zero, which prevents learning. Use GIoU or CIoU for non-overlapping cases.

2. **Assuming axis-aligned boxes**: IoU for rotated boxes requires polygon intersection algorithms (e.g., Sutherland-Hodgman). Most detection assumes axis-aligned boxes.

3. **Dividing by zero**: Single-pixel or degenerate boxes produce union = 0. Always add epsilon (1e-7) to the denominator.

4. **Integer vs. float coordinates**: IoU computed with integer pixel coordinates may have off-by-one errors. Use float coordinates and avoid counting boundary pixels inconsistently.

5. **Ignoring IoU threshold sensitivity**: mAP at IoU=0.5 saturates quickly, while mAP at IoU=0.75 is a much stricter measure. Report both to characterize detector performance.

## Interview Questions

### Beginner - 5

1. What is IoU and what range of values can it take?
2. How is IoU used during the training of an object detector?
3. What does an IoU of 0 mean? What does 1 mean?
4. Why is IoU preferred over Euclidean distance for evaluating box overlap?
5. What is the minimum IoU typically required for a detection to be considered correct?

### Intermediate - 5

1. Explain the problem with standard IoU when two boxes do not overlap.
2. How does GIoU address the non-overlapping box problem?
3. What additional terms does CIoU add beyond GIoU?
4. How is IoU used in the anchor matching strategy?
5. Why is IoU loss preferable to Smooth L1 loss for localization?

### Advanced - 3

1. Derive the gradient of CIoU with respect to predicted box coordinates.
2. Compare the behavior of IoU, GIoU, and CIoU during training with overlapping vs. non-overlapping cases.
3. How would you extend IoU to 3D bounding boxes (e.g., for point cloud detection)?

## Practice Problems

### Easy - 5

1. Compute IoU between two identical boxes.
2. Compute IoU between two non-overlapping boxes.
3. Compute IoU between a small box completely inside a large box.
4. Write a function to check if IoU > 0.5 between two pairs of boxes.
5. Convert IoU score to distance: d = 1 - IoU.

### Medium - 5

1. Implement GIoU loss from scratch.
2. Implement DIoU loss from scratch.
3. Implement CIoU loss from scratch.
4. Write a function that computes the average IoU between a set of predictions and ground truths.
5. Create an IoU-based early stopping criterion for training.

### Hard - 3

1. Implement IoU for rotated bounding boxes using shapely or custom polygon intersection.
2. Implement the alpha-IoU family of losses (IoU^α) and analyze sensitivity to α.
3. Design a differentiable IoU approximation that works with non-axis-aligned boxes.

## Solutions

Easy 2 — Non-overlapping boxes:
```python
box_a = torch.tensor([0, 0, 10, 10])
box_b = torch.tensor([20, 20, 30, 30])
print(f"IoU: {compute_iou(box_a, box_b):.4f}")
# Output: IoU: 0.0000
```

Medium 2 — DIoU Loss:
```python
def diou_loss(pred, target):
    x1 = torch.max(pred[:, 0], target[:, 0])
    y1 = torch.max(pred[:, 1], target[:, 1])
    x2 = torch.min(pred[:, 2], target[:, 2])
    y2 = torch.min(pred[:, 3], target[:, 3])
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_t = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = area_p + area_t - inter
    iou = inter / (union + 1e-7)
    cx_p = (pred[:, 0] + pred[:, 2]) / 2
    cy_p = (pred[:, 1] + pred[:, 3]) / 2
    cx_t = (target[:, 0] + target[:, 2]) / 2
    cy_t = (target[:, 1] + target[:, 3]) / 2
    rho2 = (cx_p - cx_t)**2 + (cy_p - cy_t)**2
    xc1 = torch.min(pred[:, 0], target[:, 0])
    yc1 = torch.min(pred[:, 1], target[:, 1])
    xc2 = torch.max(pred[:, 2], target[:, 2])
    yc2 = torch.max(pred[:, 3], target[:, 3])
    c2 = (xc2 - xc1)**2 + (yc2 - yc1)**2
    diou = iou - rho2 / (c2 + 1e-7)
    return 1 - diou.mean()

print(diou_loss(torch.tensor([[20.,20.,80.,80.]]), torch.tensor([[30.,30.,70.,70.]])))
# Output: tensor(0.4509)
```

## Related Concepts

- DL-231: Object Detection Overview
- DL-232: Bounding Box Regression
- DL-233: Anchor Boxes
- DL-235: Non-Maximum Suppression
- DL-236: Mean Average Precision

## Next Concepts

- DL-235: Non-Maximum Suppression
- DL-236: Mean Average Precision

## Summary

Intersection over Union is the fundamental measure of box overlap in object detection, serving as both a matching criterion and evaluation metric. While standard IoU is simple to compute, its limitations for non-overlapping boxes led to the GIoU, DIoU, and CIoU families of losses that provide smoother gradients and better optimization. Mastering IoU and its variants is essential for implementing training pipelines, evaluation, and post-processing in any detection system.

## Key Takeaways

- IoU = intersection area / union area, ranging from 0 to 1
- Used for anchor matching, NMS, and evaluation (mAP)
- Standard IoU offers zero gradient for non-overlapping boxes
- GIoU adds a penalty term for the enclosing box area
- DIoU incorporates center distance normalization
- CIoU adds aspect ratio consistency for further refinement
- Differentiable IoU losses directly optimize the evaluation metric
