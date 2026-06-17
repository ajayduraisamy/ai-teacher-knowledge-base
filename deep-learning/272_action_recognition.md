# Concept: Action Recognition

## Concept ID

DL-272

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Video Understanding

## Learning Objectives

## Prerequisites

- DL-271: Video Classification
- DL-267: Optical Flow
- DL-269: I3D

## Definition

Action Recognition is the task of identifying human actions in video, typically classifying short video clips into predefined action categories (e.g., "running," "jumping," "swinging a bat"). While closely related to video classification, action recognition specifically focuses on human-centric activities and often involves understanding the interaction between humans and objects. Standard benchmarks include UCF-101 (101 action classes), HMDB-51 (51 actions), and Kinetics (400/600/700 action classes).

## Intuition

Action recognition requires understanding both what a person is doing and how they're doing it. "Swinging a golf club" involves recognizing the person, the club, and the specific motion pattern. Actions can be characterized by pose dynamics (body movements), object interactions (holding, manipulating), and scene context (on a golf course vs. in a living room). Unlike generic video classification, action recognition often benefits from explicit pose estimation or body-part tracking.

## Why This Concept Matters

Action recognition is central to human behavior understanding with applications in surveillance (detecting falls, fights), sports analytics (classifying plays), healthcare (monitoring patient activities), human-robot interaction, and video search. It's also the foundation for temporal action detection (locating actions in long videos) and action anticipation (predicting future actions).

## Mathematical Explanation

Action recognition is typically formulated as video classification with human-centric class labels. The key differences from generic video classification:

1. **Fine-grained discrimination**: Actions like "sprint" vs. "jog" differ subtly.
2. **Temporal structure**: Actions have characteristic temporal signatures (e.g., the backswing and follow-through of a golf swing).
3. **Spatial focus**: The action typically involves specific body parts or objects.

Multi-modal approaches:
- RGB stream: appearance
- Flow stream: motion
- Pose stream: skeleton keypoints
- Object stream: interacted objects

## Code Examples

### Example 1: Action Recognition with Skeleton Poses

```python
import torch
import torch.nn as nn

class SkeletonActionModel(nn.Module):
    def __init__(self, num_joints=17, num_frames=32, num_classes=51):
        super().__init__()
        # Process skeleton sequences: [N, T, J, C] where C=2 (x,y) or C=3 (x,y,confidence)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(num_joints * 2, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, skeleton):
        # skeleton: [N, T, J, C] -> [N, J*C, T]
        N, T, J, C = skeleton.shape
        skeleton = skeleton.permute(0, 2, 3, 1).reshape(N, J * C, T)
        feats = self.temporal_conv(skeleton).view(N, -1)
        return self.fc(feats)

# Simulate skeleton data (17 joints, 32 frames, x,y coordinates)
skeleton = torch.randn(2, 32, 17, 2)
model = SkeletonActionModel()
out = model(skeleton)
print(f"Skeleton action recognition: {out.shape}")
# Output: Skeleton action recognition: torch.Size([2, 51])
```

### Example 2: Multi-Stream Action Recognition

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiStreamActionRecognition(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        # RGB stream
        self.rgb_stream = nn.Sequential(
            nn.Conv3d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool3d(1),
        )
        # Flow stream
        self.flow_stream = nn.Sequential(
            nn.Conv3d(2, 64, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool3d(1),
        )
        # Fusion classifier
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, rgb, flow):
        rgb_feat = self.rgb_stream(rgb).view(rgb.shape[0], -1)
        flow_feat = self.flow_stream(flow).view(flow.shape[0], -1)
        fused = torch.cat([rgb_feat, flow_feat], dim=1)
        return self.classifier(fused)

model = MultiStreamActionRecognition(num_classes=101)
rgb = torch.randn(2, 3, 16, 112, 112)
flow = torch.randn(2, 2, 16, 112, 112)
out = model(rgb, flow)
print(f"Multi-stream output: {out.shape}")
# Output: Multi-stream output: torch.Size([2, 101])
```

### Example 3: Action Recognition Top-K Accuracy

```python
import torch

def topk_accuracy(output, target, k=(1, 5)):
    """Compute top-k accuracy for action recognition"""
    maxk = max(k)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k_val in k:
        correct_k = correct[:k_val].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.item() * 100.0 / batch_size)
    return res

# Simulated outputs
output = torch.randn(32, 101)
target = torch.randint(0, 101, (32,))
acc1, acc5 = topk_accuracy(output, target)
print(f"Top-1 accuracy: {acc1:.2f}%, Top-5 accuracy: {acc5:.2f}%")
# Output: Top-1 accuracy: 3.12%, Top-5 accuracy: 15.62% (example)
```

## Common Mistakes

1. **Confusing action recognition with video classification**: Action recognition is a subset of video classification focusing on human activities. Not all video classification datasets are action recognition datasets.

2. **Ignoring temporal ordering**: Flipping temporal order changes "opening" to "closing". Models must be sensitive to temporal direction. Using bidirectional temporal modeling preserves order.

3. **Scene bias in datasets**: Many action datasets have strong scene correlations (swimming always in water). Models may learn scene cues instead of motion. Use scene-diverse datasets or augmentation.

4. **Limited temporal resolution**: Using T=8 frames may miss the full action. Recognize that different actions have different durations. Adaptive temporal sampling helps.

5. **Not using human-centric attention**: Actions are typically centered on people. Using human detection or attention mechanisms to focus on the person improves accuracy.

## Interview Questions

### Beginner - 5

1. What is action recognition?
2. How does it differ from video classification?
3. What are common action recognition datasets?
4. What modalities are used for action recognition?
5. What is the standard evaluation metric?

### Intermediate - 5

1. Explain the role of optical flow in action recognition.
2. How do skeleton-based approaches work?
3. What is scene bias and how do you address it?
4. Compare 3D CNNs and two-stream networks for action recognition.
5. How does action recognition handle actions of different durations?

### Advanced - 3

1. Analyze the complementary information in RGB, flow, pose, and object streams.
2. Design an action recognition system that works with camera motion.
3. How would you approach few-shot action recognition with limited examples?

## Practice Problems

### Easy - 5

1. Implement top-1 accuracy for action recognition.
2. Count the number of action classes in UCF-101 and HMDB-51.
3. Create a skeleton sequence tensor [N, T, J, C].
4. Implement a single RGB stream for action recognition.
5. Compute prediction confidence from logits.

### Medium - 5

1. Implement a multi-stream action recognition model.
2. Build a skeleton-based action recognition model.
3. Write a data augmentation function for video actions.
4. Implement temporal segment sampling for long videos.
5. Evaluate a model on UCF-101.

### Hard - 3

1. Implement a pose-based action recognition pipeline.
2. Design a model that combines RGB, flow, and pose.
3. Implement online action recognition (streaming video).

## Solutions

Easy 1:
```python
def accuracy(output, target):
    _, pred = output.max(1)
    correct = pred.eq(target).sum().item()
    return correct / target.size(0) * 100

out = torch.randn(32, 101)
target = torch.randint(0, 101, (32,))
print(f"Accuracy: {accuracy(out, target):.2f}%")
# Output: Accuracy: 1.12% (example)
```

Medium 1 — Simple Action Recognition:
```python
class ActionRecognitionModel(nn.Module):
    def __init__(self, num_classes=51):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(128, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.classifier(self.backbone(x).view(x.shape[0], -1))

model = ActionRecognitionModel()
print(f"Output: {model(torch.randn(1, 3, 16, 112, 112)).shape}")
# Output: Output: torch.Size([1, 51])
```

## Related Concepts

- DL-271: Video Classification
- DL-273: Temporal Action Detection
- DL-267: Optical Flow

## Next Concepts

- DL-273: Temporal Action Detection
- DL-274: Video Object Detection

## Summary

Action recognition classifies human activities in video clips, typically using RGB, optical flow, and/or skeleton pose inputs. It requires understanding both spatial context and temporal dynamics. Standard benchmarks include UCF-101, HMDB-51, and Kinetics. Multi-stream architectures that combine appearance and motion cues achieve the best results, and modern video transformers have pushed accuracy to near-saturation on established benchmarks.

## Key Takeaways

- Action recognition = human-centric video classification
- Multi-modal: RGB, flow, pose, objects
- UCF-101 (101 classes), HMDB-51 (51 classes), Kinetics (400-700 classes)
- Scene bias is a common pitfall
- Top-1 and Top-5 accuracy are standard metrics
- Skeleton-based models are lightweight and privacy-preserving
- Temporal ordering matters: opening ≠ closing
- Video transformers achieve 97%+ on UCF-101
