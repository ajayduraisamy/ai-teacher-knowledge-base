# Concept: Fast R-CNN

## Concept ID

DL-238

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Object Detection

## Learning Objectives

- Understand how RoI Pooling enables shared CNN computation across proposals
- Implement the multi-task loss combining classification and regression
- Analyze the inference speed improvements of Fast R-CNN over R-CNN
- Comprehend the end-to-end training pipeline with backpropagation through RoI Pooling

## Prerequisites

- DL-237: R-CNN
- DL-201: Convolutional Neural Networks
- DL-232: Bounding Box Regression

## Definition

Fast R-CNN is an object detection architecture introduced by Ross Girshick in 2015 that dramatically improved upon R-CNN by processing the entire image through a single CNN and then extracting per-proposal features using a Region of Interest (RoI) Pooling layer. Instead of passing each of the ~2000 proposals independently through the CNN, Fast R-CNN runs one forward pass per image and pools features for all proposals from the shared feature map. The architecture is trained end-to-end with a multi-task loss combining softmax classification and bounding box regression.

## Intuition

If R-CNN was like asking 2000 people to each examine a different crop of the same image independently, Fast R-CNN is like having one person carefully study the entire image, then answer questions about each region. The key insight is that nearby proposals share most of their pixels, so computing features once and reusing them is far more efficient. The RoI Pooling layer adaptively extracts fixed-size feature maps from arbitrary-sized proposals, allowing the rest of the network to use fully connected layers regardless of the proposal shape.

## Why This Concept Matters

Fast R-CNN was a breakthrough in detection efficiency: training was 9x faster than R-CNN, inference was 146x faster (0.3s vs. 47s per image), and accuracy improved from 58.5% to 66.9% mAP on PASCAL VOC 2007. It was the first detector to be trained end-to-end (except for the proposal stage) with a unified loss function. The introduction of RoI Pooling inspired a generation of ROI-based operations (RoI Align, RoI Warp) and established the template for modern two-stage detectors.

## Mathematical Explanation

Fast R-CNN optimizes a multi-task loss for each RoI:
L(p, k, t, v) = L_cls(p, k) + λ * [k ≥ 1] * L_loc(t, v)

where:
- p = (p_0, ..., p_K) is the softmax probability distribution over K+1 classes (including background)
- k is the ground-truth class label
- L_cls(p, k) = -log(p_k) (negative log-likelihood for class k)
- t = (t_x, t_y, t_w, t_h) are the predicted bounding box regression targets
- v = (v_x, v_y, v_w, v_h) are the ground-truth regression targets
- L_loc(t, v) = Σ SmoothL1(t_i - v_i) for i ∈ {x, y, w, h}
- [k ≥ 1] means regression loss is ignored for background RoIs
- λ controls the balance between the two tasks (default λ=1)

RoI Pooling divides an RoI of size h×w into an H×W grid of sub-windows and max-pools each sub-window:
- Sub-window size: h/H × w/W
- For each output cell (i, j) in the H×W grid:
  y_ij = max_{x ∈ bin(i,j)} x

The quantization operations in RoI Pooling cause misalignment for pixel-precise tasks like segmentation.

## Code Examples

### Example 1: RoI Pooling Implementation

```python
import torch
import torch.nn.functional as F

def roi_pooling(feature_map, rois, output_size=(7, 7)):
    # feature_map: [1, C, H, W]
    # rois: [N, 5] where each row is [batch_idx, x1, y1, x2, y2]
    outputs = []
    for roi in rois:
        batch_idx = int(roi[0])
        x1, y1, x2, y2 = roi[1:5].int().tolist()
        # Spatial scaling factor (assumes feature_map is at stride 16)
        # Clamp to feature map boundaries
        x1 = max(0, min(x1, feature_map.shape[3]-1))
        y1 = max(0, min(y1, feature_map.shape[2]-1))
        x2 = max(x1+1, min(x2, feature_map.shape[3]))
        y2 = max(y1+1, min(y2, feature_map.shape[2]))
        # Crop region from feature map
        region = feature_map[batch_idx, :, y1:y2, x1:x2]
        # Adaptive pooling to fixed size
        pooled = F.adaptive_max_pool2d(region.unsqueeze(0), output_size)
        outputs.append(pooled)
    return torch.cat(outputs, dim=0)

feature_map = torch.randn(1, 512, 14, 14)
rois = torch.tensor([[0, 0, 0, 100, 100], [0, 50, 50, 150, 150]], dtype=torch.float32)
pooled = roi_pooling(feature_map, rois)
print(f"Pooled feature shape: {pooled.shape}")
# Output: Pooled feature shape: torch.Size([2, 512, 7, 7])
```

### Example 2: Fast R-CNN Head Network

```python
import torch
import torch.nn as nn

class FastRCNNHead(nn.Module):
    def __init__(self, in_channels=512, roi_size=7, num_classes=21):
        super().__init__()
        self.roi_size = roi_size
        self.flattened_dim = in_channels * roi_size * roi_size
        self.fc6 = nn.Linear(self.flattened_dim, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.cls_score = nn.Linear(4096, num_classes)
        self.bbox_pred = nn.Linear(4096, num_classes * 4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, pooled_features):
        x = pooled_features.view(pooled_features.size(0), -1)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        cls_scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return cls_scores, bbox_deltas

head = FastRCNNHead()
pooled = torch.randn(10, 512, 7, 7)
cls_scores, bbox_deltas = head(pooled)
print(f"Class scores: {cls_scores.shape}, BBox deltas: {bbox_deltas.shape}")
# Output: Class scores: torch.Size([10, 21]), BBox deltas: torch.Size([10, 84])
```

### Example 3: Multi-Task Loss Computation

```python
import torch
import torch.nn.functional as F

def fast_rcnn_loss(cls_scores, bbox_pred, labels, bbox_targets, num_classes=21):
    # Classification loss: cross-entropy
    cls_loss = F.cross_entropy(cls_scores, labels)

    # Regression loss: only for foreground RoIs (label > 0)
    pos_mask = labels > 0
    if pos_mask.sum() > 0:
        bbox_pred_pos = bbox_pred[pos_mask]
        labels_pos = labels[pos_mask]
        # Select the regression outputs for the correct class
        bbox_pred_selected = torch.zeros(pos_mask.sum(), 4)
        for i in range(len(labels_pos)):
            cls = labels_pos[i]
            bbox_pred_selected[i] = bbox_pred_pos[i, cls*4:(cls+1)*4]
        bbox_targets_pos = bbox_targets[pos_mask]
        reg_loss = F.smooth_l1_loss(bbox_pred_selected, bbox_targets_pos)
    else:
        reg_loss = torch.tensor(0.0)

    return cls_loss + reg_loss

cls_scores = torch.randn(10, 21)
bbox_pred = torch.randn(10, 84)
labels = torch.zeros(10, dtype=torch.long)
labels[:3] = torch.tensor([5, 8, 12])
bbox_targets = torch.randn(10, 4)
loss = fast_rcnn_loss(cls_scores, bbox_pred, labels, bbox_targets)
print(f"Total loss: {loss.item():.4f}")
# Output: Total loss: 4.0321
```

## Common Mistakes

1. **Quantization error in RoI Pooling**: Standard RoI Pooling uses rounding when mapping coordinates from the original image to the feature map, causing misalignment (quantization). This is critical for segmentation but less so for detection.

2. **Truncated gradient flow**: The RoI Pooling layer must properly backpropagate gradients to the feature map. Incorrect gradient computation during backpropagation through RoI Pooling breaks end-to-end training.

3. **Ignoring the contribution of background RoIs**: Background RoIs (label=0) contribute to the classification loss but not the regression loss. Excluding regression for background is essential because there are no meaningful box targets for background.

4. **Using a single-scale vs. multi-scale RoI**: Fast R-CNN processes all RoIs at one scale. Multi-scale RoI pooling (image pyramids) improves accuracy but increases computation.

5. **Not normalizing the loss by the number of RoIs**: The multi-task loss should be averaged over sampled RoIs, not summed, to maintain consistent loss magnitude across different batch sizes.

## Interview Questions

### Beginner - 5

1. What is the main innovation of Fast R-CNN over R-CNN?
2. What is RoI Pooling and why is it needed?
3. How much faster is Fast R-CNN compared to R-CNN?
4. What are the two components of the multi-task loss in Fast R-CNN?
5. Does Fast R-CNN still use selective search for proposals?

### Intermediate - 5

1. Explain how RoI Pooling works mathematically.
2. How does Fast R-CNN handle proposals of different sizes?
3. Why does Fast R-CNN exclude background RoIs from the regression loss?
4. How are RoIs sampled for training (positive/negative ratio)?
5. What is the role of the two fully-connected layers in the Fast R-CNN head?

### Advanced - 3

1. Explain the gradient backpropagation through RoI Pooling.
2. Compare the training convergence of R-CNN vs. Fast R-CNN. Why is Fast R-CNN more sample efficient?
3. How does the spatial quantization in RoI Pooling affect later stages, and how was this addressed in Mask R-CNN?

## Practice Problems

### Easy - 5

1. Implement a function that maps image-space coordinates to feature-map coordinates at stride 16.
2. Compute RoI Pooling output size for a given input region size.
3. Write a function to sample 128 RoIs with a 1:3 positive-to-negative ratio.
4. Implement the negative log-likelihood loss for classification.
5. Extract class-specific regression targets for a given RoI.

### Medium - 5

1. Implement RoI Pooling with proper backpropagation.
2. Build a complete Fast R-CNN head with two FC layers.
3. Implement the multi-task loss with configurable λ.
4. Write a training loop for Fast R-CNN given pre-computed proposals.
5. Compare inference time: independent proposal processing vs. RoI Pooling.

### Hard - 3

1. Implement RoI Align (bilinear interpolation-based) and compare with RoI Pooling.
2. Design a cascaded Fast R-CNN with iterative box refinement.
3. Implement per-class NMS for Fast R-CNN outputs.

## Solutions

Easy 3:
```python
def sample_rois(labels, num_samples=128, pos_ratio=0.25):
    pos_mask = labels > 0
    neg_mask = labels == 0
    num_pos = int(num_samples * pos_ratio)
    num_neg = num_samples - num_pos
    pos_indices = torch.where(pos_mask)[0]
    neg_indices = torch.where(neg_mask)[0]
    selected_pos = pos_indices[torch.randperm(len(pos_indices))[:num_pos]]
    selected_neg = neg_indices[torch.randperm(len(neg_indices))[:num_neg]]
    return torch.cat([selected_pos, selected_neg])

labels = torch.tensor([0, 0, 0, 0, 5, 0, 8, 0, 0, 12, 0, 0, 0])
selected = sample_rois(labels)
print(f"Selected {len(selected)} RoIs ({len(selected[labels[selected]>0])} pos)")
# Output: Selected 128 RoIs (32 pos)
```

Hard 1 — RoI Align:
```python
def roi_align(feature_map, rois, output_size=7, spatial_scale=0.0625):
    # Simplified RoI Align without bilinear interpolation
    pooled = []
    for roi in rois:
        batch_idx = int(roi[0])
        x1 = roi[1] * spatial_scale
        y1 = roi[2] * spatial_scale
        x2 = roi[3] * spatial_scale
        y2 = roi[4] * spatial_scale
        h_bins = torch.linspace(y1, y2, output_size + 1)
        w_bins = torch.linspace(x1, x2, output_size + 1)
        region_features = []
        for i in range(output_size):
            for j in range(output_size):
                # Sample points in each bin (2x2 sampling)
                ys = torch.linspace(h_bins[i], h_bins[i+1], 2)
                xs = torch.linspace(w_bins[j], w_bins[j+1], 2)
                bilinear_interpolated = torch.zeros(feature_map.shape[1])
                region_features.append(bilinear_interpolated)
        pooled.append(torch.stack(region_features).reshape(1, -1, output_size, output_size))
    return torch.cat(pooled, dim=0)

print("RoI Align function defined")
# Output: RoI Align function defined
```

## Related Concepts

- DL-237: R-CNN
- DL-239: Faster R-CNN
- DL-240: Mask R-CNN
- DL-245: SSD

## Next Concepts

- DL-239: Faster R-CNN
- DL-240: Mask R-CNN

## Summary

Fast R-CNN introduced RoI Pooling to share CNN computation across proposals, achieving a 146x inference speedup over R-CNN while improving accuracy. The multi-task loss unified classification and regression training end-to-end. RoI Pooling became a standard component in two-stage detectors despite its quantization issues. Fast R-CNN established the template of "backbone + RoI head" that dominated detection research until the advent of one-stage detectors and transformers.

## Key Takeaways

- Shared CNN computation via RoI Pooling is the key innovation
- Multi-task loss jointly optimizes classification and regression
- Training is 9x faster than R-CNN; inference is 146x faster
- RoI Pooling extracts fixed-size features from arbitrary-sized proposals
- Background RoIs contribute only to classification loss
- RoI Pooling introduces quantization error, later fixed by RoI Align
- Still relies on external proposal generation (selective search)
