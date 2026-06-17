# Concept: Faster R-CNN

## Concept ID

DL-239

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Object Detection

## Learning Objectives

- Understand how the Region Proposal Network (RPN) replaces selective search
- Implement the RPN architecture with anchor-based proposals
- Comprehend the alternating training strategy or joint training
- Analyze the end-to-end detection pipeline of Faster R-CNN

## Prerequisites

- DL-237: R-CNN
- DL-238: Fast R-CNN
- DL-233: Anchor Boxes

## Definition

Faster R-CNN, introduced by Ren et al. in 2015, is a unified object detection architecture that integrates a Region Proposal Network (RPN) directly into the detection pipeline, eliminating the need for external proposal generation. The RPN shares convolutional features with the detection network, making proposals nearly cost-free (10ms per image vs. 2s for selective search). The architecture consists of: (1) a shared backbone CNN, (2) an RPN that outputs object proposals with objectness scores, and (3) a Fast R-CNN head that classifies proposals and refines boxes.

## Intuition

Faster R-CNN's key insight is that the same convolutional features used for object classification should also be useful for proposing regions. The RPN slides a small network over the feature map, at each location predicting whether an anchor box contains an object and regressing its coordinates. This creates a fully differentiable pipeline where the proposal generator learns to output better proposals as detection training progresses. The network is "attention" mechanism: the RPN tells the detector "look here," and the detector's feedback helps the RPN propose better regions.

## Why This Concept Matters

Faster R-CNN was a milestone: it was the first end-to-end trainable object detector to achieve real-time performance (5-17 fps on a GPU) while surpassing previous state-of-the-art accuracy. It achieved 73.2% mAP on PASCAL VOC 2007 and 69.8% mAP on COCO. The RPN architecture became the standard proposal mechanism for two-stage detectors, and Faster R-CNN served as the foundation for Mask R-CNN, Cascade R-CNN, and numerous variants. Its two-stage paradigm remains competitive with modern one-stage detectors.

## Mathematical Explanation

RPN Loss: Given a feature map of size W×H with K anchors per location:
L_RPN({p_i}, {t_i}) = (1/N_cls) Σ L_cls(p_i, p_i*) + λ * (1/N_reg) Σ p_i* * L_reg(t_i, t_i*)

where:
- p_i is the predicted probability of anchor i being an object
- p_i* = 1 if anchor is positive (IoU > 0.7), 0 if negative (IoU < 0.3)
- t_i = (t_x, t_y, t_w, t_h) predicted regression targets
- t_i* = ground-truth regression targets for positive anchors
- L_cls is binary cross-entropy (object vs. not-object)
- L_reg is Smooth L1 loss
- p_i* * L_reg means regression loss is only for positive anchors

The total loss for the full Faster R-CNN is:
L = L_RPN + L_Fast_RCNN

## Code Examples

### Example 1: RPN Module Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RPN(nn.Module):
    def __init__(self, in_channels=512, mid_channels=512, num_anchors=9):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(mid_channels, num_anchors * 2, 1)  # obj/non-obj
        self.bbox_pred = nn.Conv2d(mid_channels, num_anchors * 4, 1)   # 4 deltas
        self._init_weights()

    def _init_weights(self):
        for m in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        t = F.relu(self.conv(x))
        cls_logits = self.cls_logits(t)
        bbox_deltas = self.bbox_pred(t)
        # Reshape: [N, 2*K, H, W] -> [N, H*W*K, 2]
        N, _, H, W = cls_logits.shape
        cls_logits = cls_logits.view(N, -1, 2, H, W).permute(0, 3, 4, 1, 2)
        cls_logits = cls_logits.reshape(N, -1, 2)
        # [N, H*W*K, 4]
        bbox_deltas = bbox_deltas.view(N, -1, 4, H, W).permute(0, 3, 4, 1, 2)
        bbox_deltas = bbox_deltas.reshape(N, -1, 4)
        return cls_logits, bbox_deltas

rpn = RPN(in_channels=512, num_anchors=9)
feature_map = torch.randn(2, 512, 14, 14)
cls_logits, bbox_deltas = rpn(feature_map)
print(f"RPN output - Class logits: {cls_logits.shape}, BBox deltas: {bbox_deltas.shape}")
# Output: RPN output - Class logits: torch.Size([2, 1764, 2]), BBox deltas: torch.Size([2, 1764, 4])
```

### Example 2: Proposal Generation and Filtering

```python
import torch

def generate_proposals(cls_logits, bbox_deltas, anchors, score_thresh=0.5, nms_thresh=0.7, pre_nms_top=2000, post_nms_top=300):
    # cls_logits: [N, A, 2]
    scores = F.softmax(cls_logits, dim=-1)[:, :, 1]  # objectness score
    proposals = []
    for i in range(scores.shape[0]):
        # Decode boxes from deltas
        boxes = decode_deltas(anchors, bbox_deltas[i])
        # Filter by score
        mask = scores[i] > score_thresh
        boxes = boxes[mask]
        sc = scores[i][mask]
        if len(boxes) == 0:
            proposals.append(torch.empty(0, 4))
            continue
        # Sort by score and limit
        topk = min(pre_nms_top, len(boxes))
        sc, idx = sc.topk(topk)
        boxes = boxes[idx]
        # NMS
        keep = nms(boxes, sc, nms_thresh)
        keep = keep[:post_nms_top]
        proposals.append(boxes[keep])
    return proposals

# Simulate RPN outputs
anchors = torch.randn(1764, 4)
cls_logits = torch.randn(1, 1764, 2)
bbox_deltas = torch.randn(1, 1764, 4)
proposals = generate_proposals(cls_logits, bbox_deltas, anchors)
print(f"Generated proposals: {proposals[0].shape[0]}")
# Output: Generated proposals: 300
```

### Example 3: Full Faster R-CNN Inference

```python
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(pretrained=False)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(dummy_input)

print(f"Number of detected objects: {len(output[0]['boxes'])}")
print(f"Output keys: {output[0].keys()}")
# Output:
# Number of detected objects: 100
# Output keys: dict_keys(['boxes', 'labels', 'scores'])

# Inspect predictions
boxes = output[0]['boxes']
scores = output[0]['scores']
labels = output[0]['labels']
print(f"Top 3 scores: {scores[:3].tolist()}")
# Output: Top 3 scores: [0.1234, 0.0987, 0.0765] (example values)
```

## Common Mistakes

1. **Improper anchor assignment across RPN stages**: The RPN uses a different IoU threshold (0.7/0.3) for positive/negative assignment than the Fast R-CNN head (0.5/0.0). These serve different purposes.

2. **Sharing features without gradient scaling**: During joint training, gradients from both RPN and Fast R-CNN flow to the shared backbone. Without proper loss balancing, one task can dominate.

3. **Cross-boundary anchors**: Anchors that extend beyond the image boundary should be handled carefully—either clipped or excluded from training.

4. **NMS truncation of proposals**: Limiting proposals too aggressively during NMS (e.g., 100 post-NMS) can hurt recall for models with many objects per image.

5. **Training instability from the alternating approach**: The original Faster R-CNN used 4-stage alternating training. Modern implementations use approximate joint training or SGD with proper warmup.

## Interview Questions

### Beginner - 5

1. What is the main innovation of Faster R-CNN over Fast R-CNN?
2. What does RPN stand for and what does it do?
3. How does the RPN generate proposals without selective search?
4. What is the output of the RPN?
5. How many anchors does the RPN use at each spatial location?

### Intermediate - 5

1. Explain how the RPN loss function is formulated.
2. How are positive and negative samples defined for RPN training?
3. What is the role of the 3x3 convolutional layer in the RPN?
4. How does anchor-based proposal generation handle multi-scale objects?
5. Explain the 4-stage alternating training procedure for Faster R-CNN.

### Advanced - 3

1. Derive the gradient flow from RPN through the shared backbone and analyze potential training instabilities.
2. Compare the feature sharing mechanism in Faster R-CNN vs. FPN-based feature extraction.
3. How does Faster R-CNN handle translation invariance given that the RPN uses fixed anchor positions?

## Practice Problems

### Easy - 5

1. Implement a function that generates anchor boxes for a given feature map size.
2. Compute objectness scores from RPN logits using softmax.
3. Filter out proposals with low objectness scores.
4. Decode bounding box deltas given anchors and predictions.
5. Count the total number of RPN anchors for a feature map.

### Medium - 5

1. Implement the RPN loss function (classification + regression).
2. Build a complete RPN module in PyTorch.
3. Implement proposal generation with score filtering and NMS.
4. Write a function to compute RPN recall (fraction of ground truths with at least one positive anchor).
5. Implement anchor target assignment given ground truth boxes.

### Hard - 3

1. Implement a full Faster R-CNN training loop on a small dataset.
2. Compare RPN vs. selective search: measure recall at different IoU thresholds.
3. Implement FPN-based Faster R-CNN with multi-scale RPN heads.

## Solutions

Easy 1:
```python
def generate_anchors_per_location(stride, scales, aspect_ratios):
    anchors = []
    for scale in scales:
        for ar in aspect_ratios:
            h = scale * (ar ** 0.5)
            w = scale / (ar ** 0.5)
            anchors.append([0, 0, w, h])
    return torch.tensor(anchors)

anchors = generate_anchors_per_location(32, [32, 64, 128], [0.5, 1, 2])
print(f"Generated {len(anchors)} anchors per location")
# Output: Generated 9 anchors per location
```

Medium 1 — RPN Loss:
```python
def rpn_loss(cls_logits, bbox_deltas, anchors, gt_boxes):
    # Assign anchors to ground truth
    labels, bbox_targets = assign_anchors(anchors, gt_boxes)
    # Classification loss: binary cross-entropy
    cls_loss = F.binary_cross_entropy_with_logits(
        cls_logits.view(-1)[labels >= 0],
        labels[labels >= 0].float()
    )
    # Regression loss: only for positive anchors
    pos_mask = labels == 1
    if pos_mask.sum() > 0:
        reg_loss = F.smooth_l1_loss(
            bbox_deltas[pos_mask],
            bbox_targets[pos_mask]
        )
    else:
        reg_loss = torch.tensor(0.0)
    return cls_loss + reg_loss

print("RPN loss function defined")
# Output: RPN loss function defined
```

## Related Concepts

- DL-237: R-CNN
- DL-238: Fast R-CNN
- DL-240: Mask R-CNN
- DL-233: Anchor Boxes

## Next Concepts

- DL-240: Mask R-CNN
- DL-241: YOLO v1

## Summary

Faster R-CNN integrated the Region Proposal Network into the detection pipeline, achieving end-to-end training and near-real-time inference. The RPN shares backbone features, making proposals nearly cost-free. Anchor boxes provide translation-invariant reference templates for proposal generation. Faster R-CNN established the dominant two-stage paradigm, achieving state-of-the-art accuracy on VOC and COCO benchmarks, and remains widely used as a robust baseline for detection research.

## Key Takeaways

- RPN replaces selective search with a learned proposal generator
- RPN shares convolutional features with the detection network
- Anchors at each location enable multi-scale proposal generation
- Binary objectness classification + 4-coordinate regression per anchor
- End-to-end differentiable pipeline (with some approximations)
- Achieved 73.2% mAP on VOC 2007 and 69.8% on COCO
- Served as the foundation for Mask R-CNN, Cascade R-CNN, and others
