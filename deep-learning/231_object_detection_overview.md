# Concept: Object Detection Overview

## Concept ID

DL-231

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Object Detection

## Learning Objectives

- Understand the problem formulation of object detection and how it differs from image classification
- Identify the key components of an object detection pipeline
- Differentiate between two-stage and one-stage detection paradigms
- Recognize common evaluation metrics and datasets used in object detection

## Prerequisites

- DL-101: Neural Networks Basics
- DL-201: Convolutional Neural Networks
- DL-221: Image Classification

## Definition

Object Detection is a computer vision task that involves identifying and localizing multiple objects within an image. Unlike image classification which assigns a single label to an entire image, object detection outputs a set of bounding boxes with associated class labels, enabling both recognition and precise spatial localization. Formally, given an input image I, an object detector produces a set of predictions {(b_i, c_i, s_i)} where b_i is a bounding box defined by four coordinates (x_center, y_center, width, height) or (x_min, y_min, x_max, y_max), c_i is the class label from a predefined set of categories, and s_i is a confidence score indicating the likelihood the prediction is correct.

## Intuition

Think of object detection as simultaneously answering two questions for every object in an image: "What is it?" and "Where is it?" This is fundamentally harder than classification because the number of objects varies per image, objects can overlap, appear at different scales, and the model must distinguish between foreground objects and background regions. A good detector needs to be invariant to changes in lighting, pose, occlusion, and object appearance while maintaining precise localization.

## Why This Concept Matters

Object detection is the backbone of numerous real-world applications including autonomous driving (detecting pedestrians, vehicles, traffic signs), medical imaging (locating tumors or lesions), surveillance, robotics, and retail analytics. Advances in detection directly power higher-level tasks like instance segmentation, object tracking, and activity recognition. Modern deep learning-based detectors have achieved human-level performance on several benchmarks, making this one of the most impactful areas of computer vision.

## Mathematical Explanation

The object detection task can be formalized as learning a function f: R^(H×W×3) → P([0,1]^4 × C) that maps an input image to a set of predictions. The loss function typically combines classification and regression components:

L = L_cls + λ * L_reg

where L_cls is typically cross-entropy loss for class predictions and L_reg is Smooth L1 loss (also called Huber loss) for bounding box coordinates:

Smooth L1(x) = { 0.5 * x^2 if |x| < 1, |x| - 0.5 otherwise }

The Smooth L1 loss is preferred over standard L2 because it is less sensitive to outliers. For two-stage detectors, the classification loss often uses binary cross-entropy per class with focal loss variants to handle class imbalance between foreground and background.

Mean Average Precision (mAP) is the standard evaluation metric. It computes the area under the precision-recall curve for each class and averages across classes. A prediction is considered a true positive if its Intersection over Union (IoU) with a ground truth box exceeds a threshold (typically 0.5 or 0.75).

## Code Examples

### Example 1: Loading a Pre-trained Detection Model

```python
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    predictions = model(dummy_input)

print(predictions[0].keys())
# Output: dict_keys(['boxes', 'labels', 'scores'])
print(predictions[0]['boxes'].shape)
# Output: torch.Size([100, 4])
```

### Example 2: Computing IoU Between Predictions and Ground Truth

```python
import torch

def compute_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

pred = torch.tensor([50, 50, 150, 150])
gt = torch.tensor([60, 60, 140, 140])
print(f"IoU: {compute_iou(pred, gt):.4f}")
# Output: IoU: 0.6400
```

### Example 3: Visualizing Detection Outputs

```python
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_detections(image, boxes, labels, scores, threshold=0.5):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image.permute(1, 2, 0).numpy())

    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(x1, y1, f"Class {label}: {score:.2f}",
                bbox=dict(facecolor='white', alpha=0.8))

    plt.show()

# Usage (assuming image tensor and model outputs)
# visualize_detections(image, preds[0]['boxes'], preds[0]['labels'], preds[0]['scores'])
print("Visualization function defined successfully")
# Output: Visualization function defined successfully
```

## Common Mistakes

1. **Ignoring aspect ratio when resizing**: Resizing images without preserving aspect ratio distorts bounding box coordinates and degrades detection performance. Always use letterboxing or pad images appropriately.

2. **Using IoU threshold too low or too high**: A low threshold (e.g., 0.3) during evaluation inflates mAP unfairly, while a high threshold (e.g., 0.95) may penalize reasonable but imprecise detections.

3. **Confusing coordinate formats**: Mixing center-based (x_center, y_center, width, height) with corner-based (x1, y1, x2, y2) formats is a common source of bugs. Always verify the format expected by your loss function and evaluation code.

4. **Applying softmax across all classes including background**: Many detection architectures treat background as a separate class rather than using softmax over all classes. Using softmax can suppress foreground scores incorrectly.

5. **Not accounting for duplicate predictions**: Without Non-Maximum Suppression (NMS), models often produce multiple boxes around the same object, leading to artificially inflated false positive counts.

## Interview Questions

### Beginner - 5

1. What is the difference between image classification and object detection?
2. What is a bounding box and what are common formats for representing it?
3. What is Intersection over Union (IoU) and why is it used?
4. Name two popular datasets for object detection.
5. What is the purpose of a confidence score in object detection?

### Intermediate - 5

1. Explain the difference between one-stage and two-stage object detectors. What are the trade-offs?
2. How does Non-Maximum Suppression (NMS) work and why is it needed?
3. What is the role of anchor boxes in modern detectors?
4. How is mean Average Precision (mAP) calculated?
5. What is the "class imbalance" problem in object detection and how is it addressed?

### Advanced - 3

1. Discuss the impact of Feature Pyramid Networks (FPN) on multi-scale object detection.
2. Compare and contrast the loss functions used in YOLO vs. Faster R-CNN.
3. How have transformers changed the object detection landscape since 2020?

## Practice Problems

### Easy - 5

1. Implement a function that converts bounding box format from [x1, y1, x2, y2] to [cx, cy, w, h].
2. Write code to compute the area of a bounding box given its corner coordinates.
3. Given two bounding boxes, determine if they overlap.
4. Rescale bounding box coordinates from a 224x224 image to a 448x448 image.
5. Filter out predictions with confidence scores below 0.5.

### Medium - 5

1. Implement Non-Maximum Suppression from scratch.
2. Implement mean Average Precision at IoU=0.5 for a single class.
3. Write a function that applies RandomHorizontalFlip to both an image and its bounding boxes.
4. Build a simple sliding window detector using a pre-trained classifier.
5. Implement the Smooth L1 loss function used in bounding box regression.

### Hard - 3

1. Implement Focal Loss from scratch and compare training dynamics with cross-entropy loss.
2. Build a minimal two-stage detector: region proposal + classification network.
3. Implement Complete IoU (CIoU) loss that accounts for aspect ratio and center distance.

## Solutions

Solutions for Easy 1:
```python
def xyxy_to_cxcywh(box):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]

print(xyxy_to_cxcywh([10, 20, 50, 80]))
# Output: [30.0, 50.0, 40, 60]
```

Solutions for Medium 1:
```python
def nms(boxes, scores, iou_threshold=0.5):
    keep = []
    indices = scores.argsort(descending=True)
    while len(indices) > 0:
        current = indices[0]
        keep.append(current.item())
        if len(indices) == 1:
            break
        rest = indices[1:]
        ious = torch.tensor([compute_iou(boxes[current], boxes[i]) for i in rest])
        mask = ious <= iou_threshold
        indices = rest[mask]
    return keep

boxes = torch.tensor([[10,10,50,50],[15,15,55,55],[100,100,150,150]])
scores = torch.tensor([0.9, 0.8, 0.7])
keep = nms(boxes, scores)
print(keep)
# Output: [tensor(0), tensor(2)]
```

## Related Concepts

- DL-201: Convolutional Neural Networks
- DL-221: Image Classification
- DL-234: Intersection over Union
- DL-235: Non-Maximum Suppression
- DL-236: Mean Average Precision

## Next Concepts

- DL-232: Bounding Box Regression
- DL-233: Anchor Boxes
- DL-237: R-CNN

## Summary

Object detection combines classification and localization to identify and locate multiple objects in images. Modern approaches fall into one-stage (YOLO, SSD) and two-stage (Faster R-CNN) paradigms, trading speed for accuracy. The task requires handling variable numbers of objects, scale variation, and class imbalance. Evaluation uses mean Average Precision computed across IoU thresholds. Understanding detection fundamentals is essential for advanced vision tasks including segmentation, tracking, and video understanding.

## Key Takeaways

- Object detection answers both "what" and "where" for every object in an image
- Two-stage detectors offer higher accuracy; one-stage detectors offer faster inference
- IoU is the core metric for measuring localization quality
- NMS is essential for removing duplicate predictions
- mAP remains the standard evaluation metric across detection benchmarks
- Modern detectors leverage feature pyramids, anchor boxes, and attention mechanisms
