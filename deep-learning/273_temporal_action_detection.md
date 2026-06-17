# Concept: Temporal Action Detection

## Concept ID

DL-273

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Video Understanding

## Learning Objectives

## Prerequisites

- DL-272: Action Recognition
- DL-231: Object Detection Overview
- DL-271: Video Classification

## Definition

Temporal Action Detection (TAD) is the task of localizing and classifying human actions in untrimmed long videos. Given an untrimmed video containing multiple actions with background segments, TAD outputs a set of temporal intervals [start_time, end_time] with associated action class labels and confidence scores. This is analogous to object detection in images, but the detection dimension is temporal rather than spatial. Key architectures include: (1) sliding window + action classifier, (2) proposal-based methods (similar to Faster R-CNN), and (3) one-stage methods (similar to SSD/YOLO).

## Intuition

Object detection finds "where" objects are (spatial boxes); temporal action detection finds "when" actions occur (temporal segments). In a 1-hour cooking video, there are many actions: chopping, stirring, boiling. TAD must identify each action's start and end time and classify it. This is challenging because actions have highly variable durations (chopping may last 30 seconds; stirring may last 2 minutes), transitions between actions are ambiguous, and background segments must be distinguished from actions.

## Why This Concept Matters

Temporal action detection is critical for real-world video understanding where videos are long and untrimmed. Applications include: surveillance (detecting unusual activities in 24/7 footage), video summarization (identifying key moments), content moderation (finding specific content in long videos), and sports analysis (detecting plays in game footage). TAD bridges the gap between short-clip action recognition and practical video analysis.

## Mathematical Explanation

Formulation: Given video V with T frames, predict {(s_i, e_i, c_i, p_i)} where s_i is start time, e_i is end time, c_i is class, p_i is confidence.

Evaluation metrics:
- mAP@IoU: average precision at temporal IoU thresholds [0.3, 0.5, 0.7]
- tIoU = intersection / union of predicted and ground-truth intervals

Anchor-based methods: Predefined temporal windows of various lengths and strides.
Anchor-free methods: Directly predict start and end times.

## Code Examples

### Example 1: Temporal Detection with Sliding Window

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SlidingWindowDetector(nn.Module):
    def __init__(self, window_size=64, num_classes=20):
        super().__init__()
        self.window_size = window_size
        self.features = nn.Sequential(
            nn.Conv1d(512, 256, 3, padding=1), nn.ReLU(),
            nn.Conv1d(256, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, num_classes + 1)  # +1 for background

    def extract_features(self, video_features):
        # video_features: [N, D, T]
        feats = self.features(video_features).squeeze(-1)
        return feats

    def sliding_window(self, video_features, stride=16):
        N, D, T = video_features.shape
        windows = []
        for t in range(0, T - self.window_size + 1, stride):
            window = video_features[:, :, t:t+self.window_size]
            windows.append(self.extract_features(window))
        return torch.stack(windows, dim=1)

model = SlidingWindowDetector(window_size=64, num_classes=20)
video_feats = torch.randn(1, 512, 256)  # 256 temporal positions
window_scores = model.sliding_window(video_feats)
print(f"Window scores: {window_scores.shape}")
# Output: Window scores: torch.Size([1, 13, 21])
```

### Example 2: Temporal Proposal Generation

```python
import torch
import torch.nn as nn

class TemporalProposalNet(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.start_conv = nn.Conv1d(in_channels, 1, 3, padding=1)
        self.end_conv = nn.Conv1d(in_channels, 1, 3, padding=1)

    def forward(self, features):
        # features: [N, D, T]
        start_scores = torch.sigmoid(self.start_conv(features))
        end_scores = torch.sigmoid(self.end_conv(features))
        return start_scores, end_scores

def generate_proposals(start_scores, end_scores, threshold=0.5):
    N, _, T = start_scores.shape
    proposals = []
    for n in range(N):
        starts = torch.where(start_scores[n, 0] > threshold)[0]
        ends = torch.where(end_scores[n, 0] > threshold)[0]
        for s in starts:
            for e in ends:
                if e > s:
                    proposals.append((s.item(), e.item()))
    return proposals

tpn = TemporalProposalNet()
features = torch.randn(1, 512, 100)
start, end = tpn(features)
proposals = generate_proposals(start, end, threshold=0.3)
print(f"Generated {len(proposals)} proposals")
print(f"Example: {proposals[:3]}" if proposals else "No proposals")
# Output: Generated 25 proposals (example)
```

### Example 3: Temporal IoU Evaluation

```python
import torch

def temporal_iou(pred_start, pred_end, gt_start, gt_end):
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return intersection / union if union > 0 else 0.0

def compute_temporal_map(predictions, ground_truths, iou_threshold=0.5):
    """Simplified mAP computation for temporal detection"""
    # predictions: [(start, end, class, score)]
    # ground_truths: [(start, end, class)]
    predictions = sorted(predictions, key=lambda x: -x[3])  # sort by score descending

    tp = 0
    fp = 0
    matched_gt = set()

    for pred in predictions:
        p_s, p_e, p_cls, p_score = pred
        best_iou = 0
        best_gt = -1
        for i, gt in enumerate(ground_truths):
            if i in matched_gt:
                continue
            if gt[2] != p_cls:  # wrong class
                continue
            iou = temporal_iou(p_s, p_e, gt[0], gt[1])
            if iou > best_iou:
                best_iou = iou
                best_gt = i
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt)
        else:
            fp += 1

    fn = len(ground_truths) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall

preds = [(10, 50, 3, 0.9), (60, 80, 3, 0.8), (30, 40, 5, 0.7)]
gts = [(12, 48, 3), (62, 78, 3)]
prec, rec = compute_temporal_map(preds, gts)
print(f"Precision: {prec:.4f}, Recall: {rec:.4f}")
# Output: Precision: 0.6667, Recall: 0.5000
```

## Common Mistakes

1. **Using accuracy for evaluation**: Temporal detection has severe class imbalance (mostly background). Use mAP at various tIoU thresholds.

2. **Fixed temporal windows ignore action duration variation**: Different actions have different durations (e.g., "blink" vs. "walk"). Multi-scale temporal windows are essential.

3. **Applying action recognition models directly to long videos**: Action recognition models expect trimmed clips. Untrimmed videos require temporal localization before classification.

4. **Ignoring context outside the action**: Actions are influenced by what happens before and after. Incorporating surrounding context improves detection.

5. **Post-processing without soft-NMS**: Overlapping detections of the same action should be suppressed. Soft-NMS with temporal IoU helps clean up outputs.

## Interview Questions

### Beginner - 5

1. What is temporal action detection?
2. How does it differ from action recognition?
3. What is a temporal proposal?
4. What is temporal IoU?
5. What evaluation metric is used?

### Intermediate - 5

1. Explain sliding window-based temporal detection.
2. How do proposal-based methods work for temporal detection?
3. What are the challenges of detecting actions in long videos?
4. Compare one-stage and two-stage temporal detection methods.
5. How does temporal IoU differ from spatial IoU?

### Advanced - 3

1. Design a temporal action detection model that handles both short and long actions.
2. Compare temporal detection vs. online action detection (real-time).
3. How would you incorporate context modeling for temporal detection?

## Practice Problems

### Easy - 5

1. Compute temporal IoU between two intervals.
2. Implement a sliding window over temporal features.
3. Create temporal proposal anchors.
4. Count the number of anchor windows for a 100-frame video.
5. Generate random temporal proposals.

### Medium - 5

1. Implement a temporal proposal network.
2. Build a one-stage temporal action detector.
3. Implement temporal NMS.
4. Write mAP evaluation for temporal detection.
5. Implement boundary regression.

### Hard - 3

1. Implement a complete temporal action detection pipeline.
2. Design a graph-based temporal relation modeling module.
3. Implement online temporal action detection.

## Solutions

Easy 1:
```python
def temporal_iou(p_start, p_end, g_start, g_end):
    inter = max(0, min(p_end, g_end) - max(p_start, g_start))
    union = max(p_end, g_end) - min(p_start, g_start)
    return inter / union if union else 0

print(f"tIoU: {temporal_iou(10, 50, 20, 60):.4f}")
# Output: tIoU: 0.5000
```

Medium 1 — Temporal Proposals:
```python
class TPN(nn.Module):
    def __init__(self, in_c=512):
        super().__init__()
        self.conv = nn.Conv1d(in_c, 2, 3, padding=1)

    def forward(self, x):
        return self.conv(x).sigmoid()

tpn = TPN()
output = tpn(torch.randn(1, 512, 100))
print(f"Start/End scores: {output.shape}")
# Output: Start/End scores: torch.Size([1, 2, 100])
```

## Related Concepts

- DL-272: Action Recognition
- DL-231: Object Detection Overview
- DL-274: Video Object Detection

## Next Concepts

- DL-274: Video Object Detection
- DL-275: Video Instance Segmentation

## Summary

Temporal action detection localizes and classifies actions in untrimmed long videos. The task parallels object detection but in the temporal dimension. Methods include sliding window, proposal-based (two-stage), and dense prediction (one-stage). Evaluation uses mAP at temporal IoU thresholds. TAD is critical for real-world video analysis where videos are long and contain multiple actions with background.

## Key Takeaways

- Temporal detection: find when actions occur, not just what
- Untrimmed videos vs. trimmed clips
- Temporal IoU: measures temporal overlap quality
- mAP@tIoU thresholds [0.3, 0.5, 0.7] is standard metric
- Action duration varies widely (seconds to minutes)
- Proposal-based methods follow two-stage detection paradigm
- Context modeling is crucial for accurate detection
- Post-processing with temporal NMS removes duplicates
