# Concept: Non-Maximum Suppression

## Concept ID

DL-235

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Object Detection

## Learning Objectives

- Understand why NMS is required in object detection pipelines
- Implement greedy NMS from scratch in PyTorch
- Compare Soft-NMS, DIoU-NMS, and other NMS variants
- Analyze the trade-off between precision and recall introduced by NMS

## Prerequisites

- DL-231: Object Detection Overview
- DL-234: Intersection over Union

## Definition

Non-Maximum Suppression (NMS) is a post-processing algorithm that eliminates duplicate detections of the same object. Object detectors typically produce multiple overlapping bounding boxes around each instance. NMS greedily selects the highest-scoring box, suppresses all other boxes that have IoU above a threshold with it, and iterates until all boxes are processed. The output is a clean set of detections where each object is represented by exactly one box.

## Intuition

Imagine detecting a cat in an image. The model might output 20 highly overlapping boxes all around the cat, each with slightly different scores. NMS acts as a cleanup step: it keeps the box the model is most confident about, then removes all boxes that substantially overlap with it (assuming they refer to the same cat). This process repeats for remaining boxes until every object has exactly one box. Without NMS, the detector output would be unusably cluttered.

## Why This Concept Matters

NMS is a critical component in virtually every object detector. The choice of NMS algorithm and its IoU threshold directly affects the final precision and recall. A threshold too low suppresses nearby true positives; a threshold too high leaves false positives. NMS also presents a bottleneck for end-to-end differentiable pipelines, and recent work explores NMS-free architectures (e.g., DETR). Understanding NMS variants is essential for deploying detectors in production where clean outputs are mandatory.

## Mathematical Explanation

Greedy NMS algorithm:
Input: Set of detections D = {(b_i, s_i)} where b_i is box and s_i is score
Output: Filtered set F

1. Sort D by score descending
2. Initialize F = {d_1} (highest-scoring detection)
3. For each remaining detection d_i in order:
   a. Compute IoU(d_i, f) for all f ∈ F
   b. If max IoU < τ (threshold), add d_i to F
   c. Otherwise, discard d_i
4. Return F

Soft-NMS modifies step 3c to decay rather than discard:
s_i = s_i * (1 - IoU) if IoU > τ  [linear decay]
s_i = s_i * exp(-IoU^2 / σ)      [Gaussian decay]

DIoU-NMS uses DIoU instead of IoU as the suppression metric, which better handles occluded objects by penalizing center distance.

## Code Examples

### Example 1: Greedy NMS Implementation

```python
import torch

def nms(boxes, scores, iou_threshold=0.5):
    keep = []
    order = scores.argsort(descending=True)
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = torch.zeros(rest.numel())
        for j, idx in enumerate(rest):
            ious[j] = compute_iou(boxes[i], boxes[idx])
        mask = ious <= iou_threshold
        order = rest[mask]
    return torch.tensor(keep)

boxes = torch.tensor([[10,10,50,50],[15,15,55,55],[100,100,150,150]])
scores = torch.tensor([0.9, 0.8, 0.7])
keep = nms(boxes, scores, 0.5)
print(f"Indices kept: {keep.tolist()}")
# Output: Indices kept: [0, 2]
```

### Example 2: Soft-NMS Implementation

```python
import torch

def soft_nms(boxes, scores, sigma=0.5, score_threshold=0.1):
    dets = torch.cat([boxes, scores[:, None]], dim=1)
    for i in range(len(dets)):
        max_idx = dets[i:, 4].argmax() + i
        dets[[i, max_idx]] = dets[[max_idx, i]]
        ious = torch.tensor([compute_iou(dets[i, :4], dets[j, :4])
                            for j in range(i+1, len(dets))])
        weights = torch.exp(-ious ** 2 / sigma)
        dets[i+1:, 4] *= weights
    mask = dets[:, 4] > score_threshold
    return dets[mask]

boxes = torch.tensor([[10.,10.,50.,50.],[12.,12.,48.,48.],[100.,100.,150.,150.]])
scores = torch.tensor([0.9, 0.8, 0.7])
result = soft_nms(boxes, scores, sigma=0.5)
print(f"Remaining detections:\n{result}")
# Output: Remaining detections:
# tensor([[ 10.,  10.,  50.,  50., 0.9000],
#         [100., 100., 150., 150., 0.7000]])
```

### Example 3: Batch NMS with torchvision

```python
import torch
import torchvision

boxes = torch.tensor([
    [10, 10, 50, 50],
    [15, 15, 55, 55],
    [60, 60, 100, 100],
    [65, 65, 95, 95],
    [200, 200, 300, 300]
], dtype=torch.float32)

scores = torch.tensor([0.95, 0.75, 0.90, 0.80, 0.85])

# Use torchvision's batched_nms for multi-class
labels = torch.tensor([0, 0, 1, 1, 0])
keep = torchvision.ops.batched_nms(boxes, scores, labels, iou_threshold=0.5)
print(f"Kept indices: {keep.tolist()}")
# Output: Kept indices: [0, 2, 4]

# Visualization of NMS filtering
print(f"Before NMS: {len(boxes)} boxes")
print(f"After NMS: {len(keep)} boxes")
# Output:
# Before NMS: 5 boxes
# After NMS: 3 boxes
```

## Common Mistakes

1. **Applying NMS independently per class without cross-class suppression**: Different classes can overlap in the same region (e.g., a person holding a phone). Usually NMS is applied per class, but for some applications cross-class NMS is needed.

2. **Using a fixed IoU threshold for all scenarios**: The optimal NMS threshold varies by dataset, object density, and application. Crowded scenes (e.g., pedestrian detection) need lower thresholds.

3. **Applying NMS before score thresholding**: Always apply score thresholding first to remove low-confidence predictions, then NMS. This reduces the number of boxes NMS must process.

4. **Not tuning the NMS threshold for the metric**: For COCO mAP@0.5, a higher NMS threshold (0.6-0.7) works well. For mAP@0.75, a stricter NMS (0.5) is better.

5. **Using NMS in end-to-end training**: Standard NMS is non-differentiable. Some approaches use Soft-NMS or learnable NMS to allow gradient flow through the suppression step.

## Interview Questions

### Beginner - 5

1. Why is NMS needed in object detection?
2. What is the input and output of the NMS algorithm?
3. What happens if the NMS IoU threshold is set too high?
4. What happens if the NMS IoU threshold is set too low?
5. How does NMS interact with confidence scores?

### Intermediate - 5

1. Explain the Soft-NMS algorithm and how it differs from greedy NMS.
2. What is the time complexity of standard greedy NMS?
3. How does batched NMS handle multi-class detection?
4. What is DIoU-NMS and when is it beneficial?
5. How would you adapt NMS for rotated bounding boxes?

### Advanced - 3

1. Compare the effectiveness of NMS vs. learned NMS vs. NMS-free architectures (DETR).
2. How does NMS impact the evaluation of mean Average Precision?
3. Propose a differentiable approximation to NMS that allows end-to-end training.

## Practice Problems

### Easy - 5

1. Implement a function that applies score thresholding (keep boxes with score > 0.5).
2. Write a function to count how many boxes remain after NMS with different thresholds.
3. Sort a tensor of confidence scores in descending order and return indices.
4. Given two identical boxes with scores 0.9 and 0.8, what does NMS return?
5. Extract the highest-scoring box from a set of detections.

### Medium - 5

1. Implement greedy NMS without using loops (vectorized).
2. Implement Soft-NMS with Gaussian decay.
3. Implement DIoU-NMS.
4. Write a function that computes recall after NMS (fraction of ground-truth boxes with at least one surviving detection).
5. Benchmark NMS runtime with different numbers of input boxes.

### Hard - 3

1. Implement a learnable NMS module using a small neural network.
2. Implement Cluster-NMS (batched matrix operation-based NMS).
3. Design an adaptive NMS that adjusts the IoU threshold based on object density.

## Solutions

Easy 1 — Score thresholding:
```python
def score_filter(boxes, scores, threshold=0.5):
    mask = scores > threshold
    return boxes[mask], scores[mask]

boxes = torch.tensor([[10,10,50,50],[60,60,100,100]])
scores = torch.tensor([0.9, 0.3])
filtered_boxes, filtered_scores = score_filter(boxes, scores)
print(filtered_boxes, filtered_scores)
# Output: tensor([[10, 10, 50, 50]]) tensor([0.9000])
```

Medium 1 — Vectorized NMS:
```python
def nms_vectorized(boxes, scores, iou_threshold=0.5):
    order = scores.argsort(descending=True)
    boxes = boxes[order]
    keep = torch.ones(len(boxes), dtype=torch.bool)
    for i in range(len(boxes)):
        if keep[i]:
            rest = boxes[i+1:]
            if rest.numel() == 0:
                break
            x1 = torch.max(boxes[i, 0], rest[:, 0])
            y1 = torch.max(boxes[i, 1], rest[:, 1])
            x2 = torch.min(boxes[i, 2], rest[:, 2])
            y2 = torch.min(boxes[i, 3], rest[:, 3])
            inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_r = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
            union = area_i + area_r - inter
            ious = inter / (union + 1e-7)
            keep_indices = i + 1 + torch.where(ious > iou_threshold)[0]
            keep[keep_indices] = False
    return order[keep]

print(nms_vectorized(boxes, scores))
# Output: tensor([0, 2])
```

## Related Concepts

- DL-231: Object Detection Overview
- DL-234: Intersection over Union
- DL-236: Mean Average Precision

## Next Concepts

- DL-237: R-CNN
- DL-241: YOLO v1

## Summary

Non-Maximum Suppression is the standard post-processing step in object detection that converts raw model outputs into clean, non-redundant detections. Greedy NMS iteratively selects the highest-scoring box and suppresses overlapping neighbors based on IoU. Variants like Soft-NMS and DIoU-NMS improve recall in crowded scenes. While NMS is effective, it is non-differentiable and poses challenges for end-to-end learning, motivating NMS-free architectures.

## Key Takeaways

- NMS removes duplicate detections by suppressing overlapping boxes
- Greedy NMS selects the highest-scoring box, suppresses neighbors above IoU threshold
- The IoU threshold controls the precision-recall trade-off
- Soft-NMS decays scores instead of hard-suppressing, improving recall
- DIoU-NMS considers center distance for better occlusion handling
- NMS is applied per-class in multi-class detection
- DETR and other transformer-based detectors eliminate NMS entirely
