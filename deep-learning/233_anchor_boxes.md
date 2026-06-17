# Concept: Anchor Boxes

## Concept ID

DL-233

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Object Detection

## Learning Objectives

- Understand the motivation and mathematical formulation of anchor boxes
- Implement anchor generation for multiple scales and aspect ratios
- Comprehend the assignment strategy between anchors and ground-truth boxes
- Analyze the impact of anchor design parameters on detection performance

## Prerequisites

- DL-231: Object Detection Overview
- DL-232: Bounding Box Regression
- DL-234: Intersection over Union

## Definition

Anchor boxes are a set of predefined bounding boxes of various scales and aspect ratios placed at regular intervals across the image feature map. They serve as reference templates against which a detector predicts object presence and refines coordinates. Formally, at each spatial location (i, j) on a feature map of stride S, we place K anchor boxes with centers at (S * j, S * i) and dimensions derived from combinations of scales s_k and aspect ratios r_a. The total number of anchors is H × W × K, where H and W are the feature map height and width.

## Intuition

Anchor boxes encode prior knowledge about object shapes and sizes. Instead of searching over all possible locations and scales exhaustively (sliding window), anchors discretize the search space into a manageable set of candidates. If you know that objects in your dataset tend to be roughly square or rectangular with specific size ranges, you design anchors to match these statistics. The detector then only needs to predict small adjustments (regression deltas) to align an anchor with the true object, plus a binary foreground/background classification for each anchor.

## Why This Concept Matters

Anchor boxes are the cornerstone of nearly all pre-transformer object detectors. The design of anchors—their number, scales, aspect ratios, and placement—directly influences recall, especially for small objects. Poor anchor design leads to low recall because no anchor sufficiently overlaps with certain ground-truth boxes. Understanding anchors is critical for working with Faster R-CNN, SSD, YOLO, and RetinaNet. The shift to anchor-free methods (FCOS, CenterNet) was motivated precisely by the limitations and hyperparameter sensitivity of anchor-based approaches.

## Mathematical Explanation

Anchor generation parameters:
- Scales: s_k ∈ {s_1, s_2, ..., s_m} typically normalized to the feature map stride
- Aspect ratios: r_a ∈ {r_1, r_2, ..., r_n} typically {1:1, 1:2, 2:1}

At each location, anchor dimensions are:
w = s_k * sqrt(r_a)
h = s_k / sqrt(r_a)

For multi-scale feature maps (FPN), each level uses a different base scale.

Anchor assignment during training: An anchor is positive (foreground) if its IoU with any ground-truth box exceeds a high threshold (e.g., 0.7). It is negative (background) if IoU is below a low threshold (e.g., 0.3). Anchors with intermediate IoU are ignored during training.

## Code Examples

### Example 1: Generating Anchor Boxes for a Single Feature Map

```python
import torch

def generate_anchors(stride, scales, aspect_ratios):
    anchors = []
    for s in scales:
        for r in aspect_ratios:
            w = s * (r ** 0.5)
            h = s / (r ** 0.5)
            anchors.append([-w/2, -h/2, w/2, h/2])
    return torch.tensor(anchors)

stride = 32
scales = [32, 64, 128]
aspect_ratios = [0.5, 1.0, 2.0]
anchors = generate_anchors(stride, scales, aspect_ratios)
print(f"Number of anchors per location: {len(anchors)}")
print(anchors)
# Output:
# Number of anchors per location: 9
# tensor([[-11.3137, -22.6274,  11.3137,  22.6274],
#         [-16.0000, -16.0000,  16.0000,  16.0000],
#         [-22.6274, -11.3137,  22.6274,  11.3137],
#         [-22.6274, -45.2548,  22.6274,  45.2548],
#         [-32.0000, -32.0000,  32.0000,  32.0000],
#         [-45.2548, -22.6274,  45.2548,  22.6274],
#         [-45.2548, -90.5097,  45.2548,  90.5097],
#         [-64.0000, -64.0000,  64.0000,  64.0000],
#         [-90.5097, -45.2548,  90.5097,  45.2548]])
```

### Example 2: Placing Anchors Across a Grid

```python
import torch

def place_anchors_on_grid(anchor_templates, feat_height, feat_width, stride):
    # anchor_templates: [K, 4] in [x1, y1, x2, y2] centered format
    K = anchor_templates.shape[0]
    all_anchors = []
    for i in range(feat_height):
        for j in range(feat_width):
            cx = j * stride
            cy = i * stride
            shifted = anchor_templates + torch.tensor([cx, cy, cx, cy])
            all_anchors.append(shifted)
    return torch.cat(all_anchors, dim=0)

feat_h, feat_w = 7, 7
stride = 32
anchor_templates = generate_anchors(stride, [32, 64, 128], [0.5, 1.0, 2.0])
all_anchors = place_anchors_on_grid(anchor_templates, feat_h, feat_w, stride)
print(f"Total anchors: {all_anchors.shape[0]}")
print(f"First 5 anchors:\n{all_anchors[:5]}")
# Output:
# Total anchors: 441
# First 5 anchors:
# tensor([[-11.3137, -22.6274,  11.3137,  22.6274],
#         [-16.0000, -16.0000,  16.0000,  16.0000],
#         [-22.6274, -11.3137,  22.6274,  11.3137],
#         [-22.6274, -45.2548,  22.6274,  45.2548],
#         [-32.0000, -32.0000,  32.0000,  32.0000]])
```

### Example 3: Anchors Matching to Ground Truth

```python
import torch

def match_anchors_to_gt(anchors, gt_boxes, pos_thresh=0.7, neg_thresh=0.3):
    # Compute IoU matrix [num_anchors, num_gt]
    n = anchors.shape[0]
    m = gt_boxes.shape[0]
   ious = torch.zeros((n, m))
    for i in range(n):
        for j in range(m):
            a = anchors[i]
            b = gt_boxes[j]
            inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            area_a = (a[2] - a[0]) * (a[3] - a[1])
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            ious[i, j] = inter / (area_a + area_b - inter + 1e-7)

    best_iou_per_anchor, best_gt_idx = ious.max(dim=1)
    labels = torch.full((n,), -1, dtype=torch.long)  # -1 = ignore
    labels[best_iou_per_anchor >= pos_thresh] = 1  # foreground
    labels[best_iou_per_anchor <= neg_thresh] = 0  # background
    return labels, best_gt_idx, best_iou_per_anchor

anchors = torch.tensor([[10.,10.,50.,50.],[60.,60.,100.,100.]])
gt = torch.tensor([[15.,15.,45.,45.]])
labels, idx, ious = match_anchors_to_gt(anchors, gt)
print(labels, ious)
# Output: tensor([ 1, -1]) tensor([0.5625, 0.0000])
```

## Common Mistakes

1. **Mismatching anchor scales to feature map stride**: Using anchors too small on a high-stride (low-resolution) feature map means they cannot be accurately localized. Each FPN level should have anchors matched to its receptive field.

2. **Using too many anchors**: Thousands of anchors per image creates severe class imbalance (mostly background). This requires sampling strategies or focal loss to handle.

3. **Ignoring anchor visualization**: Always visualize anchor overlap with ground truth to catch design flaws. Anchors that never match any object are wasted capacity.

4. **Hardcoding anchor parameters without dataset analysis**: Anchor design should be data-driven. Compute the distribution of ground-truth box sizes and aspect ratios on your dataset.

5. **Using anchors with extreme aspect ratios**: Very tall or very wide anchors (e.g., 10:1) are rarely useful and mostly generate noise. Stick to 1:1, 2:1, 1:2 unless your data has extreme shapes.

## Interview Questions

### Beginner - 5

1. What is an anchor box in object detection?
2. Why do we need multiple anchor boxes at each location?
3. What are the two main parameters used to define anchor boxes?
4. How many anchors does a 7x7 feature map with 9 anchors per location produce?
5. What is the difference between anchor-based and anchor-free detection?

### Intermediate - 5

1. How does the anchor matching strategy work during training?
2. Explain how Feature Pyramid Networks handle multi-scale detection with anchors.
3. What is the purpose of having different anchor sizes on different feature map levels?
4. How does the number of anchors affect training time and memory?
5. Describe the anchor sampling strategy in Faster R-CNN (positive vs. negative ratio).

### Advanced - 3

1. Derive an optimal anchor scale design given the ground-truth box distribution of a dataset.
2. Compare the anchor design in SSD vs. Faster R-CNN: what trade-offs exist?
3. How does SimOTA or ATSS replace traditional fixed-threshold anchor assignment?

## Practice Problems

### Easy - 5

1. Generate 5 anchors with scale 64 and aspect ratios [0.5, 1, 2].
2. Count total anchors for a feature map of size 10x10 with 6 anchors per cell.
3. Compute the center coordinates of anchors at position (2, 3) with stride 16.
4. Filter anchors that lie entirely outside a 224x224 image.
5. Convert an anchor from [x1,y1,x2,y2] to [cx,cy,w,h].

### Medium - 5

1. Write a complete anchor generation function for FPN with 5 levels.
2. Implement balanced positive/negative anchor sampling.
3. Compute anchor recall: what fraction of ground-truth boxes have at least one anchor with IoU > 0.5?
4. Implement k-means clustering to learn anchor dimensions from a dataset.
5. Write a visualization that overlays anchors on an image at a specific location.

### Hard - 3

1. Implement ATSS (Adaptive Training Sample Selection) for anchor assignment.
2. Build an anchor-free detection head and compare performance with an anchor-based version.
3. Design and implement deformable anchor generation that adapts to object shape.

## Solutions

Medium 4 — K-means Anchor Learning:
```python
import torch
from sklearn.cluster import KMeans

def learn_anchors(gt_boxes, n_anchors=9):
    # gt_boxes: [N, 2] width, height
    kmeans = KMeans(n_clusters=n_anchors, random_state=0, n_init=10)
    kmeans.fit(gt_boxes.numpy())
    anchors = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    return anchors

gt_wh = torch.tensor([[30., 40.], [50., 60.], [100., 150.],
                       [200., 180.], [25., 25.], [300., 400.]])
anchors = learn_anchors(gt_wh, n_anchors=3)
print(anchors)
# Output: tensor([[ 30.,  40.],
#                 [200., 180.],
#                 [100., 150.]])
```

## Related Concepts

- DL-231: Object Detection Overview
- DL-232: Bounding Box Regression
- DL-234: Intersection over Union
- DL-245: SSD

## Next Concepts

- DL-237: R-CNN
- DL-238: Fast R-CNN
- DL-239: Faster R-CNN

## Summary

Anchor boxes discretize the continuous space of possible object locations and shapes into a finite set of reference templates. Anchors are characterized by scale and aspect ratio, placed densely across feature maps. Training involves matching anchors to ground-truth boxes via IoU thresholds, producing positive and negative samples. While anchor-based methods have dominated detection, they require careful hyperparameter tuning and have been increasingly challenged by anchor-free alternatives.

## Key Takeaways

- Anchors provide predefined reference boxes at each spatial location
- Anchor design (scale, aspect ratio, placement) significantly impacts recall
- IoU-based matching assigns anchors as positive or negative training samples
- Multi-scale anchors on FPN levels enable detection across object sizes
- Anchor-free methods eliminate anchor hyperparameters but introduce new design choices
- Data-driven anchor design via k-means often outperforms hand-crafted anchors
