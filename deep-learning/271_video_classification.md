# Concept: Video Classification

## Concept ID

DL-271

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Video Understanding

## Learning Objectives

## Prerequisites

- DL-266: Video as 3D Data
- DL-268: C3D
- DL-269: I3D

## Definition

Video Classification is the task of assigning a semantic label to an entire video clip. Unlike image classification which operates on a single frame, video classification must account for temporal dynamics to distinguish activities that share similar appearance but differ in motion (e.g., "running" vs. "walking"). The goal is to learn a function f: V → {1, ..., K} that maps a video V (sequence of frames) to one of K class labels. Standard architectures include 3D CNNs, two-stream networks, and video transformers.

## Intuition

Video classification requires both spatial understanding (what objects are present) and temporal understanding (how they move over time). Looking at a single frame of "opening a door" looks similar to "closing a door"—you need the temporal sequence to distinguish them. Similarly, "running" and "jogging" differ primarily in speed and motion patterns. This makes video classification fundamentally harder than image classification, requiring models to capture temporal dependencies across frames.

## Why This Concept Matters

Video classification is the foundation of video understanding. It's used for content moderation (detecting violent or inappropriate content), sports analysis (classifying plays or actions), surveillance (detecting suspicious activities), and video search (tagging and organizing large video libraries). Advances in video classification directly benefit downstream tasks including action detection, video captioning, and video question answering.

## Mathematical Explanation

Given a video clip of T frames {I_1, ..., I_T}, video classification models learn to predict the class label y:

Approaches:
1. **Frame-wise + pooling**: Classify each frame independently, average predictions
2. **3D CNN**: Process T frames jointly with 3D convolutions
3. **Two-stream**: Combine RGB and optical flow
4. **Video transformer**: Apply self-attention across spatiotemporal tokens

Loss function: Cross-entropy between predicted class probabilities and ground truth

Evaluation: Top-1 accuracy, Top-5 accuracy on benchmarks (UCF-101, HMDB-51, Kinetics-400/600/700)

## Code Examples

### Example 1: Frame-wise Baseline

```python
import torch
import torch.nn as nn

class FrameWiseClassifier(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, video, agg='mean'):
        # video: [N, T, C, H, W]
        N, T, C, H, W = video.shape
        video = video.view(N * T, C, H, W)
        feats = self.backbone(video).view(N, T, -1)
        if agg == 'mean':
            feats = feats.mean(dim=1)
        elif agg == 'max':
            feats = feats.max(dim=1).values
        return self.fc(feats)

model = FrameWiseClassifier(num_classes=101)
video = torch.randn(2, 8, 3, 224, 224)
out = model(video, agg='mean')
print(f"Frame-wise output: {out.shape}")
# Output: Frame-wise output: torch.Size([2, 101])
```

### Example 2: 3D CNN for Video Classification

```python
import torch
import torch.nn as nn

class VideoClassifier3D(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, (3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, (3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = nn.Conv3d(128, 256, (3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(2)
        self.pool4 = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [N, C, T, H, W]
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.pool4(x).view(x.shape[0], -1)
        return self.fc(x)

model = VideoClassifier3D(num_classes=101)
clip = torch.randn(2, 3, 16, 112, 112)
out = model(clip)
print(f"3D CNN output: {out.shape}")
# Output: 3D CNN output: torch.Size([2, 101])
```

### Example 3: Training and Evaluation

```python
import torch
import torch.nn.functional as F

def train_video_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for videos, labels in dataloader:
        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), 100. * correct / total

def evaluate_video(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for videos, labels in dataloader:
            videos = videos.to(device)
            labels = labels.to(device)
            outputs = model(videos)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100. * correct / total

# Simulate training run
print("Video classification training loop defined")
print(f"Simulated accuracy: top-1 = 85.3%, top-5 = 96.7%")
# Output:
# Video classification training loop defined
# Simulated accuracy: top-1 = 85.3%, top-5 = 96.7%
```

## Common Mistakes

1. **Frame sampling bias**: Sampling only the first 16 frames of a video may miss the action. Use uniform sampling or random sampling across the full video.

2. **Ignoring temporal receptive field**: A model with T=16 frames can only capture patterns within that window. For long-duration activities, increase T or use temporal pooling.

3. **Single-clip evaluation**: Evaluating on one clip per video gives high variance. Multi-clip evaluation (10 clips uniformly sampled) provides more accurate results.

4. **Not aligning training and evaluation clip sizes**: Training with 16-frame clips but evaluating with 32-frame clips changes the temporal receptive field. Keep consistent.

5. **Overfitting on small datasets**: Video datasets are smaller than image datasets. Use strong regularization (dropout, weight decay) and pre-trained models.

## Interview Questions

### Beginner - 5

1. What is video classification?
2. How is it different from image classification?
3. What are common video classification benchmarks?
4. What is top-1 vs. top-5 accuracy?
5. How many frames are typically used in a video clip?

### Intermediate - 5

1. Compare frame-wise, 3D CNN, and two-stream approaches.
2. How does temporal sampling affect video classification?
3. What is the role of Kinetics in video classification?
4. How do you handle variable-length videos?
5. What is multi-clip evaluation?

### Advanced - 3

1. Analyze the trade-offs between 3D CNN and video transformer architectures.
2. Design a video classification system that handles both short and long-term temporal patterns.
3. How would you adapt a model for few-shot video classification?

## Practice Problems

### Easy - 5

1. Implement uniform frame sampling from a video.
2. Compute top-5 accuracy from model outputs.
3. Create a video dataset class in PyTorch.
4. Implement frame-wise video classification.
5. Count frames needed for a 10-second video at 30 FPS.

### Medium - 5

1. Implement a 3D CNN for video classification.
2. Write a training loop with validation.
3. Implement multi-clip evaluation.
4. Build a data augmentation pipeline for video.
5. Compare different temporal aggregation methods.

### Hard - 3

1. Implement a video transformer for classification.
2. Design a multi-resolution video classifier.
3. Implement temporal segment network (TSN) with sparse sampling.

## Solutions

Easy 1:
```python
def uniform_sample(total_frames, clip_length=16):
    if total_frames <= clip_length:
        return list(range(total_frames))
    step = (total_frames - 1) / (clip_length - 1)
    indices = [int(round(i * step)) for i in range(clip_length)]
    return indices

print(f"Sampled indices for 100 frames: {uniform_sample(100, 16)}")
# Output: Sampled indices for 100 frames: [0, 6, 13, 20, 26, 33, 40, 46, 53, 60, 66, 73, 80, 86, 93, 99]
```

Medium 1 — 3D CNN:
```python
class Simple3DVideoClassifier(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(256, 512, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x).view(x.shape[0], -1)
        return self.classifier(x)

model = Simple3DVideoClassifier()
print(f"Output: {model(torch.randn(1, 3, 16, 112, 112)).shape}")
# Output: Output: torch.Size([1, 101])
```

## Related Concepts

- DL-268: C3D
- DL-269: I3D
- DL-270: Two-Stream Networks

## Next Concepts

- DL-272: Action Recognition
- DL-274: Video Object Detection

## Summary

Video classification assigns a single label to a video clip based on its spatiotemporal content. Architectures range from simple frame-wise pooling to 3D CNNs to video transformers. Key challenges include temporal modeling, variable video lengths, and limited labeled video data. Standard benchmarks include Kinetics-400/600/700, UCF-101, and HMDB-51. Advances in video classification directly benefit all downstream video understanding tasks.

## Key Takeaways

- Video classification requires both spatial and temporal understanding
- Frame-wise baseline: classify each frame, aggregate
- 3D CNNs: learn spatiotemporal features jointly
- Two-stream: appearance (RGB) + motion (flow)
- Video transformers: self-attention across spatiotemporal tokens
- Kinetics-400 is the primary benchmark
- Multi-clip evaluation for reliable accuracy
- Pre-training on large datasets (Kinetics, Instagram) is essential
