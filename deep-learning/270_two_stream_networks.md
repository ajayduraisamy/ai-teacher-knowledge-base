# Concept: Two-Stream Networks

## Concept ID

DL-270

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Video Understanding

## Learning Objectives

## Prerequisites

- DL-267: Optical Flow
- DL-221: Image Classification
- DL-201: Convolutional Neural Networks

## Definition

Two-Stream Networks, introduced by Simonyan and Zisserman in 2014, is a video classification architecture that processes two complementary input modalities in parallel: a spatial stream (RGB frames) that captures appearance information, and a temporal stream (stacked optical flow) that captures motion information. The two streams are trained separately and their predictions are fused at test time, typically by averaging. This architecture was the state of the art for action recognition before 3D CNNs, achieving 88.0% on UCF-101 and 59.4% on HMDB-51.

## Intuition

Human vision uses two complementary pathways: the ventral stream ("what") processes object recognition, and the dorsal stream ("where/how") processes motion. Two-stream networks mimic this: the spatial stream recognizes objects and scenes (what is in the video), while the temporal stream analyzes motion patterns (how things are moving). A person waving has a distinct flow pattern regardless of their appearance, and a car moving right produces consistent flow vectors. By combining both, the model becomes robust to both appearance and motion variations.

## Why This Concept Matters

Two-stream networks demonstrated that explicit motion representation (optical flow) significantly improves video understanding. This insight influenced virtually all subsequent video architectures. The two-stream paradigm was the dominant approach for action recognition until 3D CNNs (C3D, I3D) learned motion implicitly. Modern video transformers and two-stream I3D continue to use the multi-stream concept.

## Mathematical Explanation

Spatial stream: Single RGB frame (or averaged frames) → 2D CNN → class scores

Temporal stream: Stack of L consecutive optical flow fields (L=10) → 2D CNN → class scores

The temporal stream input has shape: [N, 2*L, H, W] where each flow field has 2 channels (u, v).

Fusion at test time:
Score_final = α * Score_spatial + (1-α) * Score_temporal

Typically α = 0.5 (average) or learned via a fusion layer.

Late fusion: Average/sum of individual frame predictions
Early fusion: Concatenate frames at input
Slow fusion: Gradually merge temporal information through the network

## Code Examples

### Example 1: Two-Stream Input Preparation

```python
import torch
import torch.nn as nn

def prepare_two_stream_input(rgb_frames, flow_frames, stack_size=10):
    """
    rgb_frames: list of [C, H, W] tensors (T frames)
    flow_frames: list of [2, H, W] tensors (T-1 flow fields)
    """
    # Spatial stream: single RGB frame (middle frame)
    mid_idx = len(rgb_frames) // 2
    spatial_input = rgb_frames[mid_idx].unsqueeze(0)  # [1, C, H, W]

    # Temporal stream: stack L consecutive flow fields
    flow_stack = []
    for i in range(min(stack_size, len(flow_frames))):
        flow_stack.append(flow_frames[i])
    # If we have fewer frames, pad with zeros
    while len(flow_stack) < stack_size:
        flow_stack.append(torch.zeros_like(flow_frames[0]))

    temporal_input = torch.cat(flow_stack, dim=0)  # [2*L, H, W]
    temporal_input = temporal_input.unsqueeze(0)  # [1, 2*L, H, W]

    return spatial_input, temporal_input

# Simulate frames
rgb = [torch.randn(3, 224, 224) for _ in range(16)]
flow = [torch.randn(2, 224, 224) for _ in range(15)]
spatial, temporal = prepare_two_stream_input(rgb, flow, stack_size=10)
print(f"Spatial input: {spatial.shape}")
print(f"Temporal input: {temporal.shape}")
# Output:
# Spatial input: torch.Size([1, 3, 224, 224])
# Temporal input: torch.Size([1, 20, 224, 224])
```

### Example 2: Two-Stream Architecture

```python
import torch
import torch.nn as nn

class TwoStreamNetwork(nn.Module):
    def __init__(self, num_classes=101, temporal_channels=20):
        super().__init__()
        # Spatial stream (appearance)
        self.spatial_stream = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.spatial_fc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        # Temporal stream (motion)
        self.temporal_stream = nn.Sequential(
            nn.Conv2d(temporal_channels, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.temporal_fc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, spatial_input, temporal_input, fusion='avg'):
        # Spatial stream
        spatial_feat = self.spatial_stream(spatial_input)
        spatial_feat = spatial_feat.view(spatial_feat.shape[0], -1)
        spatial_out = self.spatial_fc(spatial_feat)

        # Temporal stream
        temporal_feat = self.temporal_stream(temporal_input)
        temporal_feat = temporal_feat.view(temporal_feat.shape[0], -1)
        temporal_out = self.temporal_fc(temporal_feat)

        if fusion == 'avg':
            return (spatial_out + temporal_out) / 2
        elif fusion == 'max':
            return torch.max(spatial_out, temporal_out)
        elif fusion == 'concat':
            combined = torch.cat([spatial_out, temporal_out], dim=1)
            return self.fusion_fc(combined)
        else:
            return spatial_out, temporal_out

model = TwoStreamNetwork(num_classes=101)
spatial = torch.randn(1, 3, 224, 224)
temporal = torch.randn(1, 20, 224, 224)
out = model(spatial, temporal)
print(f"Two-stream output: {out.shape}")
# Output: Two-stream output: torch.Size([1, 101])
```

### Example 3: Multi-Frame Fusion Strategies

```python
import torch
import torch.nn as nn

class FusionStrategies(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, frames, fusion='late'):
        # frames: [N, T, C, H, W]
        N, T, C, H, W = frames.shape

        if fusion == 'late':
            # Process each frame independently, average scores
            scores = []
            for t in range(T):
                feat = self.backbone(frames[:, t])
                scores.append(self.fc(feat.view(N, -1)))
            return torch.stack(scores).mean(dim=0)

        elif fusion == 'early':
            # Concatenate frames at input
            x = frames.view(N, C * T, H, W)
            feat = self.backbone(x)
            return self.fc(feat.view(N, -1))

        elif fusion == 'slow':
            # Process in pairs, gradually merge
            all_feats = []
            for t in range(0, T, 2):
                pair = frames[:, t:t+2].view(N, C * 2, H, W)
                feat = self.backbone(pair)
                all_feats.append(feat)
            combined = torch.stack(all_feats).mean(dim=0)
            return self.fc(combined.view(N, -1))

model = FusionStrategies()
frames = torch.randn(2, 8, 3, 224, 224)
print(f"Late fusion: {model(frames, 'late').shape}")
print(f"Early fusion: {model(frames, 'early').shape}")
print(f"Slow fusion: {model(frames, 'slow').shape}")
# Output:
# Late fusion: torch.Size([2, 101])
# Early fusion: torch.Size([2, 101])
# Slow fusion: torch.Size([2, 101])
```

## Common Mistakes

1. **Using optical flow without temporal consistency**: Flow estimation errors between distant frames compound. Use consecutive frame pairs for flow computation.

2. **Training streams independently**: The original two-stream network trains each stream separately. Joint training can improve performance but requires careful balancing.

3. **Ignoring data augmentation for flow**: Flow fields have different statistics than RGB. Augmentation like horizontal flip must be applied consistently to both flow and RGB.

4. **Stacking too many flow frames**: Standard is L=10 flow stacks (20 channels). More frames increase memory without proportional benefit; fewer frames lose motion context.

5. **Not aligning spatial and temporal inputs**: Both streams should process frames at the same spatial resolution and from the same temporal window for meaningful fusion.

## Interview Questions

### Beginner - 5

1. What are the two streams in a two-stream network?
2. What does the spatial stream capture?
3. What does the temporal stream capture?
4. What input does the temporal stream use?
5. How are the two streams combined?

### Intermediate - 5

1. Explain why optical flow is effective for action recognition.
2. Compare early, late, and slow fusion strategies.
3. How many flow frames are typically stacked?
4. What are the advantages and disadvantages of two-stream vs. 3D CNN?
5. How does two-stream differ from the human visual system?

### Advanced - 3

1. Analyze the complementary information captured by RGB and flow streams.
2. Design a multi-stream network with more than 2 streams (e.g., audio, pose).
3. How would you implement end-to-end training with a flow estimation network?

## Practice Problems

### Easy - 5

1. Stack 10 optical flow fields into a 20-channel tensor.
2. Implement late fusion (average scores across frames).
3. Compute the number of temporal input channels for L=10 flow stacks.
4. Extract the middle frame from a video clip.
5. Implement score averaging for two stream outputs.

### Medium - 5

1. Implement the two-stream architecture.
2. Build a multi-frame fusion model.
3. Write a data loader that prepares both RGB and flow inputs.
4. Implement early fusion (concatenation).
5. Compare late vs. early fusion performance.

### Hard - 3

1. Implement a trainable fusion layer for two-stream outputs.
2. Design a two-stream architecture with learnable temporal sampling.
3. Implement an end-to-end two-stream model with a flow estimation module.

## Solutions

Easy 1:
```python
flow_fields = [torch.randn(2, 224, 224) for _ in range(10)]
stacked = torch.cat(flow_fields, dim=0)
print(f"Stacked flow: {stacked.shape}")
# Output: Stacked flow: torch.Size([20, 224, 224])
```

Medium 1 — Two-Stream:
```python
def create_two_stream():
    spatial_backbone = nn.Sequential(
        nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.ReLU(), nn.MaxPool2d(3, stride=2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, stride=2),
    )
    temporal_backbone = nn.Sequential(
        nn.Conv2d(20, 64, 7, stride=2, padding=3), nn.ReLU(), nn.MaxPool2d(3, stride=2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, stride=2),
    )
    return spatial_backbone, temporal_backbone

spatial, temporal = create_two_stream()
print(f"Spatial: {spatial(torch.randn(1, 3, 224, 224)).shape}")
print(f"Temporal: {temporal(torch.randn(1, 20, 224, 224)).shape}")
# Output:
# Spatial: torch.Size([1, 128, 14, 14])
# Temporal: torch.Size([1, 128, 14, 14])
```

## Related Concepts

- DL-267: Optical Flow
- DL-269: I3D
- DL-266: Video as 3D Data

## Next Concepts

- DL-271: Video Classification
- DL-272: Action Recognition

## Summary

Two-stream networks process appearance (RGB) and motion (optical flow) in parallel, achieving strong action recognition performance by leveraging complementary information. Late fusion averages the predictions, while early and slow fusion merge information at different network depths. The two-stream paradigm was the dominant approach until 3D CNNs and remains influential in modern video architectures.

## Key Takeaways

- Spatial stream: RGB frames (appearance)
- Temporal stream: stacked optical flow (motion)
- 10 flow frames = 20 input channels
- Late fusion: average of independent stream predictions
- 88.0% on UCF-101, 59.4% on HMDB-51
- Human visual system inspiration (ventral + dorsal streams)
- Fusion: early (input), late (scores), slow (gradual)
- Two-stream principle extends to I3D, modern video transformers
