# Concept: Video Object Detection

## Concept ID

DL-274

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Video Understanding

## Learning Objectives

## Prerequisites

- DL-231: Object Detection Overview
- DL-266: Video as 3D Data
- DL-271: Video Classification

## Definition

Video Object Detection is the task of detecting and localizing objects in each frame of a video. Unlike static image detection, video object detection can leverage temporal context across frames to improve detection accuracy, handle motion blur, and reduce false positives. Key challenges include: motion blur making objects unrecognizable in individual frames, occlusions lasting several frames, large appearance variations due to deformation, and rare viewpoints. State-of-the-art methods use temporal feature aggregation, tubelet linking, or sequence-level optimization.

## Intuition

If an object is hard to recognize in one frame due to blur or occlusion, nearby frames may show it clearly. Video object detection exploits this temporal redundancy. The model can aggregate features from multiple frames to build a stronger representation. For example, a car partly occluded by a tree in frame 10 may be fully visible in frames 8-9 and 11-12. By linking detections across frames and refining them with temporal context, video object detection achieves higher accuracy than per-frame detection.

## Why This Concept Matters

Video object detection is essential for autonomous driving (detecting pedestrians across frames), video surveillance (tracking people and vehicles), robotics (real-time object detection in dynamic scenes), and video editing. The temporal dimension provides valuable information unavailable in static images, enabling more robust and accurate detection. Methods like FGFA (Flow-Guided Feature Aggregation) and SELSA (Sequence Level Semantics Aggregation) have significantly advanced the field.

## Mathematical Explanation

Per-frame baseline: Apply image detector independently to each frame.

Temporal feature aggregation:
F'_t = Aggregation({F_{t-τ}, ..., F_{t+τ}})

where F_t are frame-level features and aggregation uses:
- Weighted average: F'_t = Σ w_i * F_i (optical flow guided)
- Attention: F'_t = Attn(F_t, {F_i})
- Memory: F'_t = GRU(F_t, h_{t-1})

Evaluation: Video mAP (standard image mAP averaged over all frames), or tubelet-based metrics.

## Code Examples

### Example 1: Frame-Independent Baseline

```python
import torch
import torch.nn as nn

class FrameIndependentDetector(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.bbox_head = nn.Conv2d(128, num_classes * 4, 1)
        self.cls_head = nn.Conv2d(128, num_classes, 1)

    def forward(self, frames):
        # frames: [N, T, C, H, W] -> process per frame
        N, T, C, H, W = frames.shape
        frames = frames.view(N * T, C, H, W)
        feats = self.backbone(frames)
        cls = self.cls_head(feats)
        bbox = self.bbox_head(feats)
        return cls.view(N, T, -1), bbox.view(N, T, -1)

model = FrameIndependentDetector()
video = torch.randn(2, 8, 3, 224, 224)
cls, bbox = model(video)
print(f"Per-frame cls: {cls.shape}, bbox: {bbox.shape}")
# Output: Per-frame cls: torch.Size([2, 8, 80]), bbox: torch.Size([2, 8, 320])
```

### Example 2: Temporal Feature Aggregation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalFeatureAggregation(nn.Module):
    def __init__(self, in_channels=512, num_frames=8):
        super().__init__()
        self.num_frames = num_frames
        self.aggregator = nn.Conv3d(in_channels, in_channels, (num_frames, 1, 1), padding=0)

    def forward(self, frame_features):
        # frame_features: [N, T, C, H, W]
        N, T, C, H, W = frame_features.shape
        if T < self.num_frames:
            frame_features = F.pad(frame_features, (0,0,0,0,0,0,0, self.num_frames - T))
        # Slide a temporal window (simplified: average all frames)
        aggregated = frame_features.mean(dim=1)
        return aggregated

aggregator = TemporalFeatureAggregation()
features = torch.randn(2, 8, 512, 14, 14)
aggregated = aggregator(features)
print(f"Aggregated features: {aggregated.shape}")
# Output: Aggregated features: torch.Size([2, 512, 14, 14])
```

### Example 3: Flow-Guided Feature Aggregation (FGFA)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FGFA(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.flow_net = nn.Sequential(
            nn.Conv2d(in_channels * 2, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 2, 3, padding=1),  # output: 2-channel flow
        )

    def warp(self, features, flow):
        # Warp features using optical flow
        N, C, H, W = features.shape
        grid = F.affine_grid(torch.eye(2, 3).unsqueeze(0).repeat(N, 1, 1).to(features.device),
                            features.shape, align_corners=False)
        flow_grid = grid + flow.permute(0, 2, 3, 1)
        warped = F.grid_sample(features, flow_grid, align_corners=False)
        return warped

    def aggregate(self, key_frame, support_frames):
        # Simple adaptive weighting
        aggregated = key_frame.clone()
        for f in support_frames:
            # Estimate flow between key and support (simplified: direct subtraction)
            flow = self.flow_net(torch.cat([key_frame, f], dim=1))
            warped = self.warp(f, flow)
            # Weight by feature similarity
            similarity = torch.sigmoid((key_frame * warped).mean(dim=1, keepdim=True))
            aggregated += warped * similarity
        return aggregated / (1 + len(support_frames))

fgfa = FGFA()
key = torch.randn(1, 512, 14, 14)
supports = [torch.randn(1, 512, 14, 14) for _ in range(3)]
aggregated = fgfa.aggregate(key, supports)
print(f"FGFA aggregated: {aggregated.shape}")
# Output: FGFA aggregated: torch.Size([1, 512, 14, 14])
```

## Common Mistakes

1. **Treating frames independently**: Per-frame detection ignores temporal consistency and fails on blurred or occluded frames. Temporal aggregation is essential.

2. **Ignoring object motion**: Objects move between frames. Simple averaging blurs features. Flow-guided warping aligns features before aggregation.

3. **Processing all frames at full resolution**: Video detection at high resolution is computationally expensive. Use sparse keyframe detection + tracking.

4. **Not handling occlusions**: Occlusion patterns vary across frames. Temporal attention can learn to down-weight occluded regions.

5. **Evaluation on keyframes only**: Some methods only evaluate on every few frames, inflating speed metrics. Report per-frame speed and accuracy.

## Interview Questions

### Beginner - 5

1. What is video object detection?
2. How does it differ from image object detection?
3. What challenges exist in video detection?
4. What is temporal feature aggregation?
5. Why is motion blur problematic?

### Intermediate - 5

1. Explain flow-guided feature aggregation (FGFA).
2. How does temporal context improve detection accuracy?
3. What are tubelets and how are they used?
4. Compare per-frame detection vs. sequence-level detection.
5. How do you handle occlusions in video detection?

### Advanced - 3

1. Design a video object detector that balances accuracy and speed.
2. Compare offline (batch) vs. online (streaming) video object detection.
3. How would you adapt a transformer-based detector (DETR) for video?

## Practice Problems

### Easy - 5

1. Apply a detection model to each frame of a video independently.
2. Compute average precision across video frames.
3. Detect objects in a 16-frame video clip.
4. Count the total number of detections in a video.
5. Visualize detections on sampled frames.

### Medium - 5

1. Implement temporal feature averaging.
2. Build a flow-guided feature aggregation module.
3. Implement tubelet linking across frames.
4. Write an evaluation script for video mAP.
5. Compare per-frame vs. temporally aggregated detection.

### Hard - 3

1. Implement a complete FGFA-based video detector.
2. Design a memory-efficient video detection pipeline.
3. Implement a transformer-based video object detector.

## Solutions

Easy 1:
```python
def detect_per_frame(model, frames):
    detections = []
    for t in range(frames.shape[1]):
        frame = frames[:, t]
        det = model(frame)  # returns boxes, scores, labels
        detections.append(det)
    return detections

frames = torch.randn(1, 8, 3, 224, 224)
print(f"Frame-independent detection over {frames.shape[1]} frames")
# Output: Frame-independent detection over 8 frames
```

Medium 1 — Temporal Feature Aggregation:
```python
def aggregate_features(features, method='mean'):
    if method == 'mean':
        return features.mean(dim=1)
    elif method == 'max':
        return features.max(dim=1).values
    elif method == 'attention':
        weights = F.softmax(features.mean(dim=(2,3,4)), dim=1)
        return (features * weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum(dim=1)

feats = torch.randn(2, 8, 256, 14, 14)
print(f"Mean agg: {aggregate_features(feats, 'mean').shape}")
# Output: Mean agg: torch.Size([2, 256, 14, 14])
```

## Related Concepts

- DL-231: Object Detection Overview
- DL-266: Video as 3D Data
- DL-275: Video Instance Segmentation

## Next Concepts

- DL-275: Video Instance Segmentation
- DL-277: TimeSformer

## Summary

Video object detection extends image detection to video by leveraging temporal context. Temporal feature aggregation, optical-flow guidance, and tubelet linking improve detection accuracy on challenging frames. The task requires balancing per-frame accuracy with computational efficiency and maintaining temporal consistency. Video detection is critical for autonomous driving, surveillance, and robotics.

## Key Takeaways

- Temporal context improves detection on blurred/occluded frames
- Per-frame detection is the baseline but misses temporal information
- FGFA: flow-guided feature aggregation
- Tubelets: linking detections across frames
- Evaluation: video mAP (per-frame) or tubelet-based metrics
- Key challenges: motion blur, occlusion, deformation
- Online vs. offline detection trade-offs
- Transformer-based video detectors extend DETR with temporal attention
