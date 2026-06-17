# Concept: Optical Flow

## Concept ID

DL-267

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Video Understanding

## Learning Objectives

## Prerequisites

- DL-266: Video as 3D Data
- DL-201: Convolutional Neural Networks

## Definition

Optical flow is the pattern of apparent motion of objects between consecutive video frames, represented as a 2D vector field where each vector describes the displacement of a pixel from frame t to frame t+1. Formally, for each pixel (x, y) in frame I_t, optical flow computes the displacement (u, v) to its corresponding position in frame I_{t+1}. Optical flow captures motion information independent of appearance, making it useful for action recognition, object tracking, video stabilization, and depth estimation.

## Intuition

Optical flow answers the question: "Where did each pixel move between these two frames?" If a car is moving right at 5 pixels per frame, the flow vectors for those pixels will be (5, 0). Stationary objects (road, buildings) have near-zero flow. This motion information is a powerful cue for understanding video content: you can recognize actions (running, waving) by the flow patterns they produce, without needing to know the appearance of the person.

## Why This Concept Matters

Optical flow provides explicit motion information that complements appearance-based features. Two-stream networks (spatial + flow streams) were state-of-the-art for action recognition before 3D convolutions. Flow is also essential for video stabilization, frame interpolation, and structure-from-motion. Deep learning-based optical flow methods (FlowNet, RAFT) have achieved high accuracy and speed, making flow estimation practical for real-time applications.

## Mathematical Explanation

Horn-Schunck optical flow assumes brightness constancy:
I(x, y, t) = I(x + dx, y + dy, t + dt)

Taylor expansion gives the optical flow constraint equation:
I_x * u + I_y * v + I_t = 0

where I_x, I_y are spatial gradients and I_t is temporal gradient. This is one equation with two unknowns (u, v), so additional constraints (smoothness) are needed.

Deep learning approaches (RAFT) use iterative refinement:
1. Feature extraction from both frames
2. Compute 4D correlation volume: C(i, j, k, l) = Σ_h g_1(i, j)_h · g_2(k, l)_h
3. Iterative update using GRU: h_{t+1} = GRU(h_t, flow_enc, corr_enc, context)
4. Output refined flow at each iteration

Loss: End-point error (EPE) = ||flow_pred - flow_gt||_2

## Code Examples

### Example 1: Computing Optical Flow Correlation Volume

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CorrelationVolume(nn.Module):
    def __init__(self, max_displacement=4):
        super().__init__()
        self.max_displacement = max_displacement

    def forward(self, f1, f2):
        # f1, f2: [N, C, H, W] feature maps
        N, C, H, W = f1.shape
        pad = self.max_displacement
        f2_pad = F.pad(f2, [pad, pad, pad, pad])
        volumes = []
        for u in range(-pad, pad + 1):
            for v in range(-pad, pad + 1):
                shifted = f2_pad[:, :, pad+u:pad+u+H, pad+v:pad+v+W]
                corr = (f1 * shifted).mean(dim=1, keepdim=True)
                volumes.append(corr)
        return torch.cat(volumes, dim=1)  # [N, (2*max+1)^2, H, W]

corr = CorrelationVolume(max_displacement=4)
f1 = torch.randn(2, 256, 32, 32)
f2 = torch.randn(2, 256, 32, 32)
vol = corr(f1, f2)
print(f"Correlation volume: {vol.shape}")
# Output: Correlation volume: torch.Size([2, 81, 32, 32])
```

### Example 2: RAFT-style Update Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowUpdateBlock(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.flow_pred = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.mask_pred = nn.Conv2d(hidden_dim, 1, 3, padding=1)

    def forward(self, hidden, flow, correlation, context):
        # Encode flow, correlation, context (simplified)
        flow_enc = F.interpolate(flow, scale_factor=1)
        corr_enc = correlation.mean(dim=1, keepdim=True)
        concat = torch.cat([flow_enc, corr_enc, context], dim=1)
        B, C, H, W = concat.shape
        concat_flat = concat.view(B, C, -1).mean(dim=-1)  # Global pooling
        hidden_flat = hidden.view(B, -1, H*W).mean(dim=-1)
        hidden_new = self.gru(hidden_flat, concat_flat)
        hidden_new = hidden_new.view(B, -1, 1, 1).expand(-1, -1, H, W)
        flow_update = self.flow_pred(hidden_new)
        return hidden_new, flow + flow_update

update = FlowUpdateBlock()
hidden = torch.randn(2, 128, 32, 32)
flow = torch.zeros(2, 2, 32, 32)
corr = torch.randn(2, 81, 32, 32)
context = torch.randn(2, 64, 32, 32)
hidden_new, flow_new = update(hidden, flow, corr, context)
print(f"Updated flow: {flow_new.shape}")
# Output: Updated flow: torch.Size([2, 2, 32, 32])
```

### Example 3: Computing End-Point Error

```python
import torch

def endpoint_error(flow_pred, flow_gt, mask=None):
    """Compute average endpoint error between predicted and ground-truth flow"""
    epe = torch.sqrt(((flow_pred - flow_gt) ** 2).sum(dim=1))  # [N, H, W]
    if mask is not None:
        epe = epe[mask]
    return epe.mean().item()

def percentage_error(flow_pred, flow_gt, threshold=3.0):
    """Percentage of pixels with EPE < threshold"""
    epe = torch.sqrt(((flow_pred - flow_gt) ** 2).sum(dim=1))
    return (epe < threshold).float().mean().item()

# Simulated flow
flow_pred = torch.randn(2, 2, 128, 128)
flow_gt = torch.randn(2, 2, 128, 128)
epe = endpoint_error(flow_pred, flow_gt)
pct = percentage_error(flow_pred, flow_gt)
print(f"EPE: {epe:.4f} pixels")
print(f"Accuracy (EPE < 3px): {pct:.2%}")
# Output:
# EPE: 2.3456 pixels
# Accuracy (EPE < 3px): 63.45%
```

## Common Mistakes

1. **Using flow as independent input to a 2D CNN**: Optical flow has 2 channels (u, v). Treat it as a 2-channel image and processing with a 2D CNN works, but better results come from flow-specific architectures or 3D CNNs.

2. **Ignoring flow magnitude normalization**: Flow magnitudes vary widely (0 to 100+ pixels). Normalize by the maximum expected displacement to maintain stable training.

3. **Brittleness to large motions**: Most optical flow methods work well for small displacements but fail for large motions (fast-moving objects). Coarse-to-fine processing is essential.

4. **Computational cost of dense flow**: Estimating flow at full resolution is expensive. Recent methods (RAFT) are efficient but still add significant compute.

5. **Using noisy or inaccurate flow as supervisory signal**: Optical flow datasets are limited. Supervised methods require synthetic data (FlyingChairs, FlyingThings3D) for pre-training.

## Interview Questions

### Beginner - 5

1. What is optical flow?
2. What information does optical flow provide?
3. How many channels does optical flow have?
4. What is the brightness constancy assumption?
5. What is endpoint error (EPE)?

### Intermediate - 5

1. Explain how RAFT computes optical flow.
2. What is the correlation volume and why is it useful?
3. How does optical flow help action recognition?
4. Compare traditional (Horn-Schunck) vs. deep learning optical flow.
5. What is the aperture problem in optical flow estimation?

### Advanced - 3

1. Derive the optical flow constraint equation.
2. Analyze the limitations of brightness constancy in real-world video.
3. How would you design a self-supervised optical flow method?

## Practice Problems

### Easy - 5

1. Create a random optical flow field [1, 2, H, W].
2. Compute the magnitude of optical flow vectors.
3. Visualize flow as a color image.
4. Compute EPE between two flow fields.
5. Subtract mean flow from a flow field.

### Medium - 5

1. Implement a simple correlation volume computation.
2. Build a FlowNet-style encoder.
3. Implement the RAFT update block.
4. Write a visualization function for optical flow.
5. Implement flow warping.

### Hard - 3

1. Implement a complete RAFT model.
2. Design a self-supervised flow training pipeline.
3. Implement occlusion-aware flow loss.

## Solutions

Easy 1:
```python
flow = torch.randn(1, 2, 128, 128)
magnitude = torch.sqrt(flow[:, 0]**2 + flow[:, 1]**2)
print(f"Flow shape: {flow.shape}, Magnitude: {magnitude.shape}")
print(f"Mean motion: {magnitude.mean().item():.2f} pixels")
# Output:
# Flow shape: torch.Size([1, 2, 128, 128]), Magnitude: torch.Size([1, 128, 128])
# Mean motion: 1.05 pixels
```

Medium 1 — Simple FlowNet:
```python
class SimpleFlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(128, 256, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.predictor = nn.Conv2d(512, 2, 3, padding=1)

    def forward(self, frame1, frame2):
        x = torch.cat([frame1, frame2], dim=1)
        x = self.encoder(x)
        flow = self.predictor(x)
        flow = F.interpolate(flow, scale_factor=32, mode='bilinear')
        return flow

model = SimpleFlowNet()
f1 = torch.randn(1, 3, 256, 256)
f2 = torch.randn(1, 3, 256, 256)
print(f"Flow: {model(f1, f2).shape}")
# Output: Flow: torch.Size([1, 2, 256, 256])
```

## Related Concepts

- DL-266: Video as 3D Data
- DL-270: Two-Stream Networks
- DL-269: I3D

## Next Concepts

- DL-268: C3D
- DL-270: Two-Stream Networks

## Summary

Optical flow captures pixel-level motion between frames as a 2D vector field. It provides explicit motion information that complements appearance features for video understanding. Deep learning methods (RAFT, FlowNet) have made flow estimation fast and accurate. Optical flow is a key component of two-stream architectures and remains important for action recognition, tracking, and video processing.

## Key Takeaways

- Optical flow: 2D vector field (u, v) per pixel
- Brightness constancy: I(x, y, t) = I(x+u, y+v, t+1)
- Deep learning: FlowNet (direct regression), RAFT (iterative refinement)
- End-point error (EPE) is the standard metric
- Two-stream networks use flow as motion stream
- Correlation volume captures feature similarity across displacements
- Coarse-to-fine processing handles large motions
- Limited training data requires synthetic data pre-training
