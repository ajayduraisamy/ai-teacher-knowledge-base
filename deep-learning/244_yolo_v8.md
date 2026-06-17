# Concept: YOLO v8

## Concept ID

DL-244

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Object Detection

## Learning Objectives

- Understand the anchor-free design in YOLO v8
- Implement the C2f module and decoupled head
- Comprehend the task-aligned sample assignment
- Analyze the improvements over YOLO v5

## Prerequisites

- DL-243: YOLO v5
- DL-242: YOLO v3
- DL-235: Non-Maximum Suppression

## Definition

YOLO v8, released by Ultralytics in 2023, is the latest major version of the YOLO family, succeeding YOLO v5. It introduces an anchor-free detection head, a C2f (Cross-Stage Partial with 2 convolutions and f) module replacing the C3 module, a decoupled classification and regression head, and task-aligned sample assignment (similar to TaskAlignedAssigner). YOLO v8 also supports multiple vision tasks including detection, segmentation, pose estimation, and classification. YOLO v8x achieves 53.9% mAP on COCO at 42 FPS on T4 GPU.

## Intuition

YOLO v8 represents a paradigm shift from anchor-based to anchor-free detection, simplifying the architecture and reducing hyperparameter tuning. Instead of predicting offsets from predefined anchor boxes, each grid cell directly predicts whether an object's center falls within it and regresses the box boundaries. The C2f module enriches gradient flow by concatenating all bottleneck outputs rather than just the last one. The decoupled head separates classification and regression into different branches, following the observation that these tasks benefit from different feature representations.

## Why This Concept Matters

YOLO v8 sets the current state-of-the-art for real-time detection, offering superior accuracy to YOLO v5 at similar speeds. Its anchor-free design eliminates the need for anchor clustering and tuning, simplifying deployment. The unified framework for detection, segmentation, and pose estimation makes it a versatile tool. YOLO v8's architectural innovations (C2f, decoupled head, task-aligned assignment) represent the culmination of years of detection research refinement.

## Mathematical Explanation

Task-aligned sample assignment: Computes a task-aligned score for each anchor-instance pair:
t = s^α * u^β

where s is the classification score, u is the IoU between predicted and ground-truth box, and α, β are weighting hyperparameters (typically α=1, β=6). The top k predictions with highest t scores are selected as positive samples.

Decoupled head: Separate branches for classification (Cls) and regression (Reg), each with 2 Conv layers:
Cls: F → Conv → Conv → channels×num_classes
Reg: F → Conv → Conv → 4×reg_max

where reg_max=16 uses distributional focal loss to predict discretized box coordinates.

## Code Examples

### Example 1: C2f Module

```python
import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=None):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, autopad(k, p), bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def autopad(k, p=None):
    return k // 2 if p is None else p

class Bottleneck(nn.Module):
    def __init__(self, in_c, out_c, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(out_c * e)
        self.cv1 = Conv(in_c, c_, 1, 1)
        self.cv2 = Conv(c_, out_c, 3, 1)
        self.shortcut = shortcut and in_c == out_c

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))

class C2f(nn.Module):
    def __init__(self, in_c, out_c, n=1, shortcut=True, e=0.5):
        super().__init__()
        self.c = int(out_c * e)
        self.cv1 = Conv(in_c, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, out_c, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

c2f = C2f(256, 256, n=3)
x = torch.randn(1, 256, 64, 64)
print(f"C2f output: {c2f(x).shape}")
# Output: C2f output: torch.Size([1, 256, 64, 64])
```

### Example 2: Decoupled Detection Head

```python
import torch
import torch.nn as nn

class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data.reshape(-1, 1, 1)[:] = nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, c // 4, a).transpose(2, 1).softmax(1)).view(b, 4, a)

class DetectHead(nn.Module):
    def __init__(self, nc=80, filters=(256, 512, 1024), reg_max=16):
        super().__init__()
        self.nc = nc
        self.nl = len(filters)
        self.reg_max = reg_max
        # Decoupled layers per scale
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(f, f, 3), Conv(f, f, 3), nn.Conv2d(f, 4 * reg_max, 1))
            for f in filters)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(f, f, 3), Conv(f, f, 3), nn.Conv2d(f, nc, 1))
            for f in filters)
        self.dfl = DFL(reg_max)

    def forward(self, x):
        outputs = []
        for i in range(self.nl):
            box = self.cv2[i](x[i])
            cls = self.cv3[i](x[i])
            if not self.training:
                N, _, H, W = box.shape
                box = box.view(N, 4, self.reg_max, H, W).permute(0, 1, 3, 4, 2)
                box = self.dfl(box.flatten(2)).view(N, 4, H, W)
                box = box.sigmoid()
            outputs.append(torch.cat((box, cls.sigmoid()), dim=1))
        return torch.cat([o.view(o.shape[0], -1, o.shape[-2], o.shape[-1]) for o in outputs], dim=2) if not self.training else outputs

head = DetectHead(nc=80, filters=[256, 512, 1024])
feats = [torch.randn(1, f, 64 // (2**i), 64 // (2**i)) for i, f in enumerate([256, 512, 1024])]
out = head(feats)
print(f"Inference output shape: {out.shape}")
# Output: Inference output shape: torch.Size([1, 84, 64, 64])
```

### Example 3: Task-Aligned Sample Assignment

```python
import torch

def task_aligned_assigner(cls_scores, bbox_preds, gt_labels, gt_bboxes, topk=10, alpha=1.0, beta=6.0):
    """Task-aligned sample assignment for YOLO v8"""
    # cls_scores: [N, num_anchors, num_classes]
    # bbox_preds: [N, num_anchors, 4]
    # gt_labels: [N, num_gts]
    # gt_bboxes: [N, num_gts, 4]
    num_gts = gt_labels.shape[1]
    if num_gts == 0:
        return None, None

    # Compute alignment score for each anchor-gt pair
    # t = s^alpha * IoU^beta
    # Simplified: compute classification score and IoU
    N, A, _ = cls_scores.shape
    num_classes = cls_scores.shape[2]

    aligned_scores = torch.zeros(N, A, num_gts)
    for n in range(N):
        for g in range(num_gts):
            gt_cls = gt_labels[n, g]
            if gt_cls >= 0:
                cls_score = cls_scores[n, :, gt_cls]
                # Compute IoU (simplified)
                iou = compute_batch_iou(bbox_preds[n], gt_bboxes[n, g].unsqueeze(0))
                t = (cls_score ** alpha) * (iou.squeeze(-1) ** beta)
                aligned_scores[n, :, g] = t

    # Select top-k anchors per GT
    pos_mask = torch.zeros(N, A, dtype=torch.bool)
    for n in range(N):
        for g in range(num_gts):
            if gt_labels[n, g] >= 0:
                _, topk_idx = aligned_scores[n, :, g].topk(min(topk, A))
                pos_mask[n, topk_idx] = True

    return pos_mask, aligned_scores

def compute_batch_iou(boxes1, boxes2):
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - inter
    return inter / (union + 1e-7)

cls_scores = torch.randn(1, 100, 80)
bbox_preds = torch.randn(1, 100, 4)
gt_labels = torch.tensor([[0, 1, -1]])
gt_bboxes = torch.randn(1, 3, 4)
mask, scores = task_aligned_assigner(cls_scores, bbox_preds, gt_labels, gt_bboxes)
print(f"Positive anchors: {mask.sum().item()}")
# Output: Positive anchors: 20 (example)
```

## Common Mistakes

1. **Confusing anchor-free with keypoint-based**: YOLO v8 is still grid-based but predicts box boundaries directly rather than offsets from anchors. It is NOT a keypoint-based detector.

2. **Using incorrect output format**: YOLO v8's output format differs from YOLO v5. The decoupled head produces separate classification and regression features that need different post-processing.

3. **Ignoring the task-aligned assigner's hyperparameters**: The α and β in the alignment score significantly affect sample quality. Tuning these per dataset can improve performance.

4. **Applying YOLO v5 post-processing to YOLO v8**: The anchor-free output requires different decoding logic. Using YOLO v5's anchor-based decoding produces incorrect results.

5. **Not leveraging mixed precision training**: YOLO v8 benefits significantly from automatic mixed precision (AMP). Training without it on modern GPUs is slower and may reduce batch size.

## Interview Questions

### Beginner - 5

1. What is the key architectural difference between YOLO v8 and YOLO v5?
2. What does anchor-free detection mean?
3. What is the C2f module?
4. How many tasks does YOLO v8 support?
5. What is a decoupled detection head?

### Intermediate - 5

1. Explain the task-aligned sample assignment mechanism.
2. How does the decoupled head improve detection performance?
3. What is distributional focal loss (DFL) and why is it used?
4. How does the C2f module differ from the C3 module in YOLO v5?
5. What are the benefits of anchor-free detection?

### Advanced - 3

1. Compare YOLO v8's anchor-free design with FCOS and other anchor-free detectors.
2. Analyze the gradient flow in the C2f module vs. C3.
3. How does YOLO v8 handle multi-scale features compared to YOLO v5?

## Practice Problems

### Easy - 5

1. Implement a SiLU activation function.
2. Count the number of output channels from a C2f module.
3. Write a function to convert YOLO v8 output to bounding boxes.
4. Compute the number of positive samples per image in task-aligned assignment.
5. Implement the DFL module.

### Medium - 5

1. Implement the complete C2f module.
2. Build a decoupled detection head.
3. Implement task-aligned sample assignment.
4. Write a post-processing pipeline for YOLO v8.
5. Implement the distributional focal loss.

### Hard - 3

1. Implement a complete YOLO v8 training pipeline.
2. Design and implement an instance segmentation version of YOLO v8.
3. Compare YOLO v5 vs. YOLO v8 on COCO metrics.

## Solutions

Easy 1:
```python
def silu(x):
    return x * torch.sigmoid(x)

x = torch.tensor([-1., 0., 1.])
print(f"SiLU: {silu(x)}")
# Output: SiLU: tensor([-0.2689,  0.0000,  0.7311])
```

Medium 2 — Decoupled Head:
```python
class SimpleDecoupledHead(nn.Module):
    def __init__(self, in_c=256, num_classes=80):
        super().__init__()
        self.cls_branch = nn.Sequential(
            Conv(in_c, in_c, 3),
            Conv(in_c, in_c, 3),
            nn.Conv2d(in_c, num_classes, 1)
        )
        self.reg_branch = nn.Sequential(
            Conv(in_c, in_c, 3),
            Conv(in_c, in_c, 3),
            nn.Conv2d(in_c, 4, 1)
        )

    def forward(self, x):
        cls_out = self.cls_branch(x)
        reg_out = self.reg_branch(x)
        return cls_out, reg_out

head = SimpleDecoupledHead()
feat = torch.randn(1, 256, 64, 64)
cls_out, reg_out = head(feat)
print(f"Classification: {cls_out.shape}, Regression: {reg_out.shape}")
# Output: Classification: torch.Size([1, 80, 64, 64]), Regression: torch.Size([1, 4, 64, 64])
```

## Related Concepts

- DL-243: YOLO v5
- DL-242: YOLO v3
- DL-249: YOLO NAS
- DL-247: DETR

## Next Concepts

- DL-245: SSD
- DL-249: YOLO NAS

## Summary

YOLO v8 represents the state of the art in real-time detection from the YOLO family, introducing anchor-free detection, C2f modules, and a decoupled head with task-aligned sampling. It achieves 53.9% COCO mAP at 42 FPS on a T4 GPU, surpassing YOLO v5 while simplifying the architecture. The unified task framework supports detection, segmentation, pose, and classification, making it versatile for production deployment.

## Key Takeaways

- Anchor-free design eliminates anchor tuning
- C2f module concatenates all bottleneck outputs for richer gradient flow
- Decoupled head separates classification and regression
- Task-aligned sample assignment selects positive samples based on alignment score
- Distributional Focal Loss for precise box coordinate prediction
- Multi-task framework: detection, segmentation, pose estimation
- Surpasses YOLO v5 in accuracy at equivalent speeds
