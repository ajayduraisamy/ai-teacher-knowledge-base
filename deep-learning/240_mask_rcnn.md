# Concept: Mask R-CNN

## Concept ID

DL-240

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Object Detection

## Learning Objectives

- Understand how Mask R-CNN extends Faster R-CNN with a segmentation branch
- Implement RoI Align as a replacement for RoI Pooling
- Comprehend the multi-task loss combining detection and segmentation
- Analyze the relationship between bounding boxes and instance masks

## Prerequisites

- DL-239: Faster R-CNN
- DL-238: Fast R-CNN
- DL-234: Intersection over Union

## Definition

Mask R-CNN, introduced by He et al. in 2017, is an instance segmentation architecture that extends Faster R-CNN by adding a branch that predicts a binary segmentation mask for each Region of Interest (RoI), in parallel with the existing classification and bounding box regression branches. The mask branch is a small FCN applied to each RoI, predicting a pixel-wise binary mask in a fixed resolution (e.g., 28x28). Mask R-CNN also introduces RoI Align, which uses bilinear interpolation to eliminate the quantization errors of RoI Pooling, enabling pixel-accurate mask prediction.

## Intuition

Mask R-CNN asks the detector to not only locate objects but also to outline their exact shape. The key insight is that detection and segmentation are complementary tasks that benefit from shared features. By adding a small FCN head that outputs a binary mask per class, the model learns fine-grained shape information. The per-class mask branch ensures that masks do not compete across classes: the mask for class "dog" is only trained on dog RoIs. RoI Align ensures that the pixel-level mask predictions are precisely aligned with the input image, avoiding the coarse quantization of RoI Pooling.

## Why This Concept Matters

Mask R-CNN was a landmark achievement: it simplified the instance segmentation pipeline by extending Faster R-CNN with minimal overhead, achieving 37.1% mask AP on COCO at 5 fps. It set the state-of-the-art for instance segmentation and served as the backbone for numerous extensions (Cascade Mask R-CNN, PointRend, etc.). The RoI Align operation became a standard component in two-stage architectures. Mask R-CNN also demonstrated strong results on human pose estimation (Keypoint R-CNN variant), showing the flexibility of the framework.

## Mathematical Explanation

Mask R-CNN loss:
L = L_cls + L_box + L_mask

where L_cls and L_box are identical to Faster R-CNN. The mask loss is:

L_mask = -1/m Σ Σ [y_ij * log(p_ij^k) + (1 - y_ij) * log(1 - p_ij^k)]

where:
- m is the number of pixels in the mask (e.g., 28x28 = 784)
- y_ij is the ground-truth label at pixel (i, j) (0 or 1)
- p_ij^k is the predicted probability at pixel (i, j) for class k

Crucially, the mask branch outputs K masks (one per class), but only the mask corresponding to the ground-truth class contributes to the loss. This decouples mask prediction from class competition.

RoI Align removes quantization by computing feature values at regular sample points using bilinear interpolation, then aggregating (max or average). For each RoI bin:
- Sample 4 regular points
- Compute feature value at each point via bilinear interpolation
- Aggregate (max or average) the 4 values for the bin output

## Code Examples

### Example 1: RoI Align Implementation

```python
import torch
import torch.nn.functional as F

def bilinear_interpolate(feature_map, x, y):
    # feature_map: [C, H, W], x, y: normalized coordinates [0, 1]
    H, W = feature_map.shape[1], feature_map.shape[2]
    x = x * (W - 1)
    y = y * (H - 1)
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1
    x0 = torch.clamp(x0, 0, W - 1)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 1)
    y1 = torch.clamp(y1, 0, H - 1)
    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x - x0.float()) * (y1.float() - y)
    wc = (x1.float() - x) * (y - y0.float())
    wd = (x - x0.float()) * (y - y0.float())
    result = (wa * feature_map[:, y0, x0] + wb * feature_map[:, y0, x1] +
              wc * feature_map[:, y1, x0] + wd * feature_map[:, y1, x1])
    return result

def roi_align(feature_map, rois, output_size=14, spatial_scale=0.0625, sampling_ratio=2):
    outputs = []
    for roi in rois:
        batch_idx = int(roi[0])
        x1, y1, x2, y2 = roi[1:] * spatial_scale
        # Generate sampling grid
        h_bins = torch.linspace(y1, y2, output_size * sampling_ratio + 1)
        w_bins = torch.linspace(x1, x2, output_size * sampling_ratio + 1)
        bin_h = (y2 - y1) / output_size
        bin_w = (x2 - x1) / output_size
        mask = torch.zeros(feature_map.shape[1], output_size, output_size)
        for i in range(output_size):
            for j in range(output_size):
                y_center = y1 + (i + 0.5) * bin_h
                x_center = x1 + (j + 0.5) * bin_w
                y_center = y_center / feature_map.shape[2]  # normalize
                x_center = x_center / feature_map.shape[3]
                val = bilinear_interpolate(feature_map[batch_idx], x_center, y_center)
                mask[:, i, j] = val
        outputs.append(mask.unsqueeze(0))
    return torch.cat(outputs, dim=0)

feature_map = torch.randn(1, 256, 32, 32)
rois = torch.tensor([[0, 0, 0, 100, 100], [0, 50, 50, 150, 150]], dtype=torch.float32)
aligned = roi_align(feature_map, rois, output_size=7)
print(f"RoI Align output: {aligned.shape}")
# Output: RoI Align output: torch.Size([2, 256, 7, 7])
```

### Example 2: Mask Head Implementation

```python
import torch
import torch.nn as nn

class MaskHead(nn.Module):
    def __init__(self, in_channels=256, roi_size=14, num_classes=80):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.deconv = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.mask_fcn = nn.Conv2d(256, num_classes, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.deconv(x))
        mask_logits = self.mask_fcn(x)
        return mask_logits

mask_head = MaskHead(in_channels=256, num_classes=80)
roi_features = torch.randn(10, 256, 14, 14)
mask_logits = mask_head(roi_features)
print(f"Mask logits shape: {mask_logits.shape}")
# Output: Mask logits shape: torch.Size([10, 80, 28, 28])
```

### Example 3: Mask R-CNN Loss Computation

```python
import torch
import torch.nn.functional as F

def mask_rcnn_loss(mask_logits, labels, mask_targets):
    # mask_logits: [N, num_classes, 28, 28]
    # labels: [N] ground-truth class labels
    # mask_targets: [N, 28, 28] ground-truth masks
    loss = 0.0
    for i in range(len(labels)):
        if labels[i] > 0:  # foreground
            class_mask_logits = mask_logits[i, labels[i]]  # [28, 28]
            loss += F.binary_cross_entropy_with_logits(
                class_mask_logits, mask_targets[i]
            )
    return loss / max(labels.gt(0).sum(), 1)

# Simulate data
mask_logits = torch.randn(4, 80, 28, 28)
labels = torch.tensor([0, 5, 12, 8])  # 1 background, 3 foreground
mask_targets = torch.randint(0, 2, (4, 28, 28)).float()
loss = mask_rcnn_loss(mask_logits, labels, mask_targets)
print(f"Mask loss: {loss.item():.4f}")
# Output: Mask loss: 0.6931
```

## Common Mistakes

1. **Training masks for all classes simultaneously**: The mask branch outputs K class-specific masks, but only the mask corresponding to the ground-truth class has its loss computed. Using softmax across classes for masks is incorrect.

2. **Applying mask loss to background RoIs**: Background RoIs (label=0) do not have mask targets. The mask loss is computed only for positive RoIs. Including background in mask loss adds noise.

3. **Using RoI Pooling instead of RoI Align**: RoI Pooling's quantization error causes mask misalignment (a few pixels shift), significantly reducing mask AP. RoI Align is essential for pixel-accurate masks.

4. **Low mask resolution (e.g., 14x14)**: The standard 28x28 mask output is sufficient for COCO but may under-segment fine details. For high-resolution segmentation, increase the mask resolution or use PointRend.

5. **Not using per-class mask heads**: A single binary mask head (class-agnostic) loses the ability to distinguish shape variations across classes. Per-class masks capture class-specific shape priors.

## Interview Questions

### Beginner - 5

1. What additional task does Mask R-CNN perform compared to Faster R-CNN?
2. What is RoI Align and why was it introduced?
3. What is the output resolution of the mask head in Mask R-CNN?
4. How many masks does the mask head predict per RoI?
5. Does Mask R-CNN use the mask predictions for detection?

### Intermediate - 5

1. Explain how RoI Align differs from RoI Pooling and why this matters for segmentation.
2. How is the mask loss computed in Mask R-CNN?
3. What is the role of the per-class mask branch?
4. How does the mask head architecture look (conv layers)?
5. Why does the classification branch still use bounding boxes even with mask predictions?

### Advanced - 3

1. Analyze the gradient flow through RoI Align compared to RoI Pooling.
2. Compare Mask R-CNN with direct instance segmentation approaches like YOLACT or SOLO.
3. How would you extend Mask R-CNN to panoptic segmentation?

## Practice Problems

### Easy - 5

1. Compute the output size after a ConvTranspose2d with kernel=2, stride=2.
2. Implement binary cross-entropy loss for mask predictions.
3. Extract the mask logits for a specific class from the mask head output.
4. Count the number of parameters in a MaskHead with 256 channels.
5. Write a function to visualize a predicted mask overlayed on an image.

### Medium - 5

1. Implement RoI Align with proper bilinear interpolation.
2. Build a complete Mask R-CNN head combining box and mask branches.
3. Implement the multi-task loss for Mask R-CNN.
4. Write a function to compute mask IoU (between predicted and ground-truth masks).
5. Implement mask post-processing: thresholding and extracting contours.

### Hard - 3

1. Implement Cascade Mask R-CNN with multi-stage refinement.
2. Implement PointRend for high-resolution mask refinement.
3. Design a video instance segmentation extension of Mask R-CNN.

## Solutions

Easy 1:
```python
input_size = 14
deconv = nn.ConvTranspose2d(256, 256, 2, stride=2)
out = deconv(torch.randn(1, 256, input_size, input_size))
print(f"Input {input_size}x{input_size} -> Output {out.shape[2]}x{out.shape[3]}")
# Output: Input 14x14 -> Output 28x28
```

Medium 1 — RoI Align:
```python
def compute_mask_iou(pred_mask, gt_mask, threshold=0.5):
    pred_binary = (pred_mask > threshold).float()
    intersection = (pred_binary * gt_mask).sum()
    union = (pred_binary + gt_mask).clamp(0, 1).sum()
    return intersection / (union + 1e-7)

pred = torch.sigmoid(torch.randn(28, 28))
gt = torch.randint(0, 2, (28, 28)).float()
print(f"Mask IoU: {compute_mask_iou(pred, gt):.4f}")
# Output: Mask IoU: 0.3521
```

## Related Concepts

- DL-239: Faster R-CNN
- DL-252: Instance Segmentation
- DL-260: Mask R-CNN for Segmentation
- DL-256: Skip Connections in U-Net

## Next Concepts

- DL-241: YOLO v1
- DL-245: SSD
- DL-261: MaskFormer

## Summary

Mask R-CNN extends Faster R-CNN with a parallel mask prediction branch, enabling instance segmentation with minimal computational overhead. The introduction of RoI Align eliminates quantization errors, enabling pixel-accurate mask predictions. The per-class mask branch decouples shape prediction from class competition. Mask R-CNN achieves state-of-the-art instance segmentation, demonstrating that detection and segmentation benefit from shared feature learning.

## Key Takeaways

- Mask R-CNN adds a mask prediction branch to Faster R-CNN
- RoI Align replaces RoI Pooling for pixel-accurate alignment
- Mask loss is computed per-class, only for the ground-truth class
- Mask output is 28x28, upsampled from 14x14 features
- RoI Align uses bilinear interpolation for sub-pixel precision
- Achieved 37.1% mask AP on COCO at 5 fps
- Framework extensible to keypoint detection and panoptic segmentation
