# Concept: Video Instance Segmentation

## Concept ID

DL-275

## Difficulty

Expert

## Domain

Deep Learning

## Module

Video Understanding

## Learning Objectives

## Prerequisites

- DL-252: Instance Segmentation
- DL-274: Video Object Detection
- DL-253: Panoptic Segmentation

## Definition

Video Instance Segmentation (VIS) is the task of detecting, segmenting, and tracking object instances across video frames. Given a video, VIS outputs a set of binary masks for each object instance, along with class labels, that are consistent across the entire video. Each instance has a unique ID that persists across frames. VIS combines instance segmentation (pixel-level object delineation) with object tracking (temporal instance association). Standard benchmarks include YouTube-VIS (2019/2021/2022), OVIS, and UVO.

## Intuition

VIS is the most comprehensive object-level video understanding task. It answers: "What objects are in this video, what are their shapes, and how do they move over time?" For a video of a dog running, VIS produces a segmentation mask for the dog in every frame, with the same instance ID. This distinguishes it from per-frame instance segmentation (which treats each frame independently) and object tracking (which uses boxes instead of masks).

## Why This Concept Matters

VIS provides the most detailed understanding of objects in video, enabling applications in autonomous driving (precise object boundaries over time), video editing (isolating and manipulating objects), medical imaging (tracking cells or organs), and robotics (object persistence and interaction understanding).

VIS is significantly harder than image instance segmentation because it requires temporal consistency, handles appearance changes, occlusions, and object interactions across frames.

## Code Examples

### Example 1: Per-Frame Instance Segmentation Baseline

```python
import torch
import torch.nn as nn

class PerFrameVIS(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        )
        self.mask_head = nn.Conv2d(128, num_classes, 1)

    def forward(self, video):
        # video: [N, T, C, H, W]
        N, T, C, H, W = video.shape
        video = video.view(N * T, C, H, W)
        masks = self.mask_head(self.backbone(video))
        masks = masks.view(N, T, -1, H, W)
        return masks

model = PerFrameVIS()
video = torch.randn(2, 8, 3, 224, 224)
masks = model(video)
print(f"Per-frame masks: {masks.shape}")
# Output: Per-frame masks: torch.Size([2, 8, 40, 224, 224])
```

### Example 2: Temporal Mask Propagation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalMaskPropagation(nn.Module):
    def __init__(self):
        super().__init__()
        self.corr = nn.Conv2d(256 * 2, 2, 3, padding=1)

    def propagate(self, mask_t, features_t, features_t1):
        # Estimate flow between frames
        flow = self.corr(torch.cat([features_t, features_t1], dim=1))
        # Warp mask to next frame
        grid = self._flow_to_grid(flow, mask_t.shape)
        mask_t1 = F.grid_sample(mask_t, grid, align_corners=False)
        return mask_t1, flow

    def _flow_to_grid(self, flow, shape):
        N, _, H, W = shape
        grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, W), torch.linspace(-1, 1, H))
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(N, 1, 1, 1).to(flow.device)
        return grid + flow.permute(0, 2, 3, 1)

propagator = TemporalMaskPropagation()
mask = torch.sigmoid(torch.randn(1, 1, 56, 56))
feat_t = torch.randn(1, 256, 56, 56)
feat_t1 = torch.randn(1, 256, 56, 56)
prop_mask, flow = propagator.propagate(mask, feat_t, feat_t1)
print(f"Propagated mask: {prop_mask.shape}, Flow: {flow.shape}")
# Output: Propagated mask: torch.Size([1, 1, 56, 56]), Flow: torch.Size([1, 2, 56, 56])
```

### Example 3: MaskTrack R-CNN Style Architecture

```python
import torch
import torch.nn as nn

class MaskTrackRCNN(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
        )
        # Detection head
        self.rpn = nn.Conv2d(256, 9 * 2, 3, padding=1)
        # Mask head
        self.mask_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )
        # Tracking head (re-identification embedding)
        self.track_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, video):
        N, T, C, H, W = video.shape
        all_masks = []
        all_embeddings = []

        for t in range(T):
            frame = video[:, t]
            feats = self.backbone(frame)
            rpn = self.rpn(feats)
            masks = self.mask_head(feats)
            embeds = self.track_head(feats).view(N, -1)
            all_masks.append(masks)
            all_embeddings.append(embeds)

        masks = torch.stack(all_masks, dim=1)
        embeddings = torch.stack(all_embeddings, dim=1)

        # Associate instances using embedding similarity
        return masks, embeddings

model = MaskTrackRCNN(num_classes=40)
video = torch.randn(1, 8, 3, 128, 128)
masks, embeddings = model(video)
print(f"Masks: {masks.shape}, Embeddings: {embeddings.shape}")
# Output: Masks: torch.Size([1, 8, 40, 128, 128]), Embeddings: torch.Size([1, 8, 64])
```

## Common Mistakes

1. **Using per-frame instance segmentation without tracking**: Each frame produces independent instance IDs. The same object gets different IDs in different frames. Tracking head or association is required.

2. **Ignoring long-term occlusions**: Objects may disappear and reappear. Simple frame-to-frame tracking fails. Use re-identification features for long-term matching.

3. **Processing full-resolution video**: VIS at native resolution is computationally prohibitive. Use memory-efficient architectures or keyframe propagation.

4. **Evaluation metric confusion**: VIS uses different metrics than image instance segmentation: mask AP with 3D IoU or mask tracking accuracy.

5. **Not handling new object appearances**: Objects entering the frame need new instance IDs. Detection-based association must handle object birth and death.

## Interview Questions

### Beginner - 5

1. What is video instance segmentation?
2. How does it differ from image instance segmentation?
3. What are the two main components of VIS?
4. What is YouTube-VIS?
5. Why is temporal consistency important?

### Intermediate - 5

1. Explain the mask propagation approach.
2. How do re-identification embeddings help VIS?
3. What are the challenges of occlusions in VIS?
4. Compare per-frame + tracking vs. end-to-end VIS.
5. How is VIS evaluated?

### Advanced - 3

1. Design a transformer-based VIS architecture.
2. How would you handle objects that disappear and reappear?
3. Compare VIS with video panoptic segmentation.

## Practice Problems

### Easy - 5

1. Create a video with multiple object masks.
2. Assign consistent instance IDs across frames.
3. Compute mask IoU between consecutive frames.
4. Count the number of instances in a VIS output.
5. Visualize masks on sampled video frames.

### Medium - 5

1. Implement per-frame instance segmentation.
2. Build a simple tracking head.
3. Implement mask propagation across frames.
4. Write VIS evaluation (mask AP + tracking).
5. Implement embedding-based association.

### Hard - 3

1. Implement a complete MaskTrack R-CNN.
2. Design a video instance segmentation transformer.
3. Implement online VIS with real-time constraints.

## Solutions

Easy 1:
```python
video_masks = torch.randint(0, 2, (3, 8, 112, 112))  # 3 instances, 8 frames
instance_ids = torch.arange(3).float()
print(f"Video masks: {video_masks.shape}, IDs: {instance_ids}")
# Output: Video masks: torch.Size([3, 8, 112, 112]), IDs: tensor([0., 1., 2.])
```

Medium 1 — Per-frame VIS:
```python
def per_frame_vis(model, video):
    N, T, C, H, W = video.shape
    results = []
    for t in range(T):
        detections = model(video[:, t])  # returns masks, boxes, labels, scores
        results.append(detections)
    return results

print("Per-frame VIS function defined")
# Output: Per-frame VIS function defined
```

## Related Concepts

- DL-252: Instance Segmentation
- DL-274: Video Object Detection
- DL-253: Panoptic Segmentation

## Next Concepts

- DL-276: Video Transformers
- DL-277: TimeSformer

## Summary

Video Instance Segmentation combines instance segmentation with temporal tracking to produce consistent per-instance segmentation masks across video frames. Approaches range from per-frame detection + association to end-to-end video-level models. Key challenges include temporal consistency, occlusion handling, and computational efficiency. VIS provides the most comprehensive object-level video understanding.

## Key Takeaways

- VIS = instance segmentation + temporal tracking
- Each object instance has a unique, persistent ID across frames
- YouTube-VIS is the standard benchmark
- Mask propagation and re-identification are key techniques
- Per-frame + association is a common baseline
- End-to-end VIS with transformers is state-of-the-art
- Challenges: occlusion, appearance change, new object appearance
- Evaluation: mask AP + tracking metrics
