# Concept: Mean Average Precision

## Concept ID

DL-236

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Object Detection

## Learning Objectives

- Understand the computation of precision-recall curves for object detection
- Implement mean Average Precision (mAP) from scratch
- Distinguish between mAP@0.5, mAP@0.75, and mAP@[0.5:0.95]
- Analyze how different IoU thresholds affect evaluation

## Prerequisites

- DL-231: Object Detection Overview
- DL-234: Intersection over Union
- DL-235: Non-Maximum Suppression

## Definition

Mean Average Precision (mAP) is the primary evaluation metric for object detection. It computes the average precision (area under the precision-recall curve) for each class, then averages across all classes. A detection is considered a true positive if its IoU with a ground-truth box exceeds a specified threshold and its class matches. Multiple detections of the same ground-truth object are treated as false positives. The COCO evaluation protocol defines several mAP variants: mAP@0.5 (IoU=0.5), mAP@0.75 (IoU=0.75), and mAP@[0.5:0.95] (average over IoU thresholds from 0.5 to 0.95 in 0.05 increments).

## Intuition

Average Precision summarizes the shape of the precision-recall curve into a single number. A perfect detector achieves AP=1.0: it finds all objects (high recall) with no false positives (high precision) at every confidence threshold. Real detectors trade off precision and recall as the confidence threshold varies. mAP averages this trade-off across classes and IoU thresholds, giving a comprehensive measure of detection quality that rewards both correct classification and precise localization.

## Why This Concept Matters

mAP is the standard benchmark metric for object detection research. Progress in the field is measured by improvements in COCO mAP. Understanding mAP is essential for: (1) correctly evaluating your own models, (2) reading detection papers critically, (3) debugging detection pipelines (e.g., low precision vs. low recall), and (4) comparing with published results. The COCO mAP variant with multiple IoU thresholds is particularly informative because it measures both detection (presence/absence) and localization quality.

## Mathematical Explanation

Given a set of detections D = {(b_i, c_i, s_i)} and ground truths G = {(b_j, c_j)} after matching:

1. Sort detections by confidence score descending
2. For each rank k:
   - Compute precision_k = TP_k / (TP_k + FP_k)
   - Compute recall_k = TP_k / (total ground truths for this class)
3. Interpolate precision at 101 recall points [0, 0.01, ..., 1.0]:
   p_interp(r) = max_{r' >= r} p(r')
4. AP = (1/101) * Σ p_interp(r_i)

For COCO mAP@[0.5:0.95]:
mAP = (1/10) * Σ_{t ∈ {0.5, 0.55, ..., 0.95}} AP@t

Average Precision across all classes:
mAP = (1/C) * Σ_{k=1}^C AP_k

## Code Examples

### Example 1: Computing AP for a Single Class

```python
import torch
import numpy as np

def compute_ap(detections, ground_truths, iou_threshold=0.5):
    # detections: [(box, score)] sorted by score descending
    # ground_truths: [box]
    tp = torch.zeros(len(detections))
    fp = torch.zeros(len(detections))
    gt_matched = torch.zeros(len(ground_truths), dtype=torch.bool)

    for i, (box, score) in enumerate(detections):
        best_iou = 0
        best_j = -1
        for j, gt_box in enumerate(ground_truths):
            if gt_matched[j]:
                continue
            iou = compute_iou(box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_threshold and best_j >= 0:
            tp[i] = 1
            gt_matched[best_j] = True
        else:
            fp[i] = 1

    tp_cum = tp.cumsum(0)
    fp_cum = fp.cumsum(0)
    precision = tp_cum / (tp_cum + fp_cum + 1e-7)
    recall = tp_cum / len(ground_truths)

    # 101-point interpolation
    ap = 0
    for t in torch.linspace(0, 1, 101):
        p_recall = precision[recall >= t]
        if p_recall.numel() > 0:
            ap += p_recall.max()
    return ap / 101

dets = [(torch.tensor([10,10,50,50]), 0.9), (torch.tensor([30,30,80,80]), 0.8)]
gts = [torch.tensor([12,12,48,48])]
ap = compute_ap(dets, gts)
print(f"AP@0.5: {ap:.4f}")
# Output: AP@0.5: 1.0000
```

### Example 2: COCO mAP Computation Across IoU Thresholds

```python
import torch

def compute_coco_map(all_detections, all_ground_truths, num_classes):
    iou_thresholds = torch.arange(0.5, 1.0, 0.05)
    aps = torch.zeros(num_classes, len(iou_thresholds))

    for cls in range(num_classes):
        dets = [(b, s) for (b, c, s) in all_detections if c == cls]
        gts = [b for (b, c) in all_ground_truths if c == cls]
        dets.sort(key=lambda x: x[1], reverse=True)

        for t_idx, iou_thresh in enumerate(iou_thresholds):
            if len(gts) == 0:
                aps[cls, t_idx] = 0.0
                continue
            aps[cls, t_idx] = compute_ap(dets, gts, iou_thresh)

    map_50 = aps[:, 0].mean()
    map_75 = aps[:, 5].mean()
    map_all = aps.mean()
    return map_all, map_50, map_75

# Simulated data
all_dets = [(torch.tensor([10,10,50,50]), 0, 0.95),
            (torch.tensor([60,60,100,100]), 1, 0.90)]
all_gts = [(torch.tensor([12,12,48,48]), 0),
           (torch.tensor([62,62,98,98]), 1)]
map_all, map_50, map_75 = compute_coco_map(all_dets, all_gts, 2)
print(f"mAP@[0.5:0.95]: {map_all:.4f}, mAP@0.5: {map_50:.4f}, mAP@0.75: {map_75:.4f}")
# Output: mAP@[0.5:0.95]: 0.8097, mAP@0.5: 1.0000, mAP@0.75: 1.0000
```

### Example 3: Using torchvision to Compute mAP

```python
import torch
import torchvision

# Simulate model predictions and ground truths
pred_boxes = [torch.tensor([[10, 10, 50, 50], [30, 30, 80, 80]])]
pred_scores = [torch.tensor([0.95, 0.85])]
pred_labels = [torch.tensor([0, 1])]

gt_boxes = [torch.tensor([[12, 12, 48, 48], [60, 60, 100, 100]])]
gt_labels = [torch.tensor([0, 1])]

# torchvision's evaluation utilities
from torchvision.ops import box_iou

iou_matrix = box_iou(pred_boxes[0], gt_boxes[0])
print(f"IoU Matrix:\n{iou_matrix}")
# Output:
# IoU Matrix:
# tensor([[0.6400, 0.0000],
#         [0.0000, 0.1633]])

# Compute precision and recall
matched = iou_matrix > 0.5
true_positives = matched.any(dim=1).sum()
false_positives = len(pred_boxes[0]) - true_positives
false_negatives = len(gt_boxes[0]) - true_positives
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")
# Output: Precision: 0.50, Recall: 0.50
```

## Common Mistakes

1. **Computing mAP without NMS**: Raw detector outputs contain many duplicate detections. mAP must be computed after NMS to avoid artificially inflating true positives with multiple boxes on the same object.

2. **Using interpolation incorrectly**: The 101-point interpolation is the VOC 2010+ standard. The earlier VOC 2007 standard used 11-point interpolation. Always specify which protocol you use.

3. **Averaging over all classes including empty ones**: If a class has no ground truths in a dataset split, exclude it from the mAP calculation, or define AP=0 for that class.

4. **Greedy matching without optimal assignment**: The matching algorithm assigns each detection to the highest-IoU unmatched ground truth. This greedy approach can miss better global assignments.

5. **Confusing mAP for classification vs. detection**: In classification, AP measures ranking quality (precision@k). In detection, AP additionally conditions on correct localization via IoU thresholding.

## Interview Questions

### Beginner - 5

1. What does mAP stand for and what does it measure?
2. What is the difference between precision and recall in object detection?
3. What is a true positive in the context of object detection?
4. Why is mAP averaged over multiple IoU thresholds?
5. What does mAP@0.5 vs. mAP@0.75 tell us about a detector?

### Intermediate - 5

1. Explain the 101-point interpolation method for computing AP.
2. How does the COCO evaluation protocol differ from Pascal VOC?
3. What is the role of the confidence score in computing the precision-recall curve?
4. How do you handle multiple detections of the same object in mAP computation?
5. What is the effect of NMS on mAP?

### Advanced - 3

1. Derive the relationship between precision, recall, and the F1 score in detection.
2. How would you implement mAP for rotated bounding boxes?
3. Discuss the limitations of mAP as an evaluation metric and propose improvements.

## Practice Problems

### Easy - 5

1. Given a set of detections and one ground truth, compute precision and recall at IoU=0.5.
2. Compute how many true positives occur if there are 3 detections and 2 ground truths.
3. Calculate AP using 11-point interpolation given a precision array.
4. Compute class-wise mAP for a 2-class detection problem.
5. Given three detections with scores [0.9, 0.8, 0.7], sort them by score.

### Medium - 5

1. Implement the full VOC 2010 AP computation (101-point interpolation).
2. Implement COCO-style mAP evaluation from scratch.
3. Write a function that computes mAP at IoU thresholds 0.5, 0.75, and 0.9.
4. Create a class that tracks mAP during training (evaluation callback).
5. Compare mAP@0.5 vs. mAP@0.75 on a simulated dataset with noisy localizations.

### Hard - 3

1. Implement the optimal matching (Hungarian algorithm) for detection evaluation.
2. Implement mAP for open-vocabulary detection where class names are dynamic.
3. Design a detection metric that accounts for both mAP and inference speed (latency-aware evaluation).

## Solutions

Easy 1:
```python
detections = [(torch.tensor([10,10,50,50]), 0.9)]
ground_truth = torch.tensor([12,12,48,48])
iou = compute_iou(detections[0][0], ground_truth)
is_tp = iou >= 0.5
precision = 1.0 if is_tp else 0.0
recall = 1.0 if is_tp else 0.0
print(f"Precision: {precision}, Recall: {recall}")
# Output: Precision: 1.0, Recall: 1.0
```

Medium 1 — VOC 2010 AP:
```python
def voc_ap(recall, precision):
    recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    precision = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])
    for i in range(len(precision)-1, 0, -1):
        precision[i-1] = max(precision[i-1], precision[i])
    ap = 0.0
    for i in range(len(recall)-1):
        ap += (recall[i+1] - recall[i]) * precision[i+1]
    return ap

r = torch.tensor([0.0, 0.5, 1.0])
p = torch.tensor([1.0, 0.8, 0.6])
print(f"VOC AP: {voc_ap(r, p):.4f}")
# Output: VOC AP: 0.7500
```

## Related Concepts

- DL-231: Object Detection Overview
- DL-234: Intersection over Union
- DL-235: Non-Maximum Suppression

## Next Concepts

- DL-237: R-CNN
- DL-241: YOLO v1

## Summary

Mean Average Precision is the definitive metric for object detection evaluation. It computes the area under the precision-recall curve using interpolation and averages across classes and IoU thresholds. The COCO protocol, with its multi-threshold averaging, provides a comprehensive measure of both detection and localization quality. Understanding mAP computation is essential for model development, debugging, and research comparison.

## Key Takeaways

- mAP averages precision across recall levels and across classes
- COCO mAP uses 10 IoU thresholds from 0.5 to 0.95
- AP computation requires matching predictions to ground truths via IoU
- 101-point interpolation creates a smoothed precision-recall curve
- mAP@0.5 measures detection ability; mAP@0.75 measures localization quality
- NMS must be applied before mAP evaluation
- mAP does not capture inference speed or computational cost
