# Concept: Video Benchmarks

## Concept ID

DL-280

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Video Understanding

## Learning Objectives

## Prerequisites

- DL-266: Video as 3D Data
- DL-267: Optical Flow
- DL-271: Video Classification
- DL-272: Action Recognition

## Definition

Video benchmarks are standardized datasets and evaluation protocols used to measure and compare the performance of video understanding models. Major benchmarks include Kinetics-400/600/700 (action classification), Something-Something V2 (fine-grained temporal modeling), UCF-101 and HMDB-51 (legacy benchmarks), AVA (spatiotemporal action detection), and ActivityNet (temporal action localization). Benchmarks define the task, data splits, evaluation metrics, and reporting conventions that enable fair comparison across methods.

## Intuition

Benchmarks serve as the common ground for measuring progress in video understanding. Kinetics focuses on recognizing diverse human actions in web videos. Something-Something V2 tests temporal reasoning (e.g., "pushing something from left to right"). UCF-101 and HMDB-51 are smaller datasets used for ablation studies. Each benchmark emphasizes different aspects of video understanding — appearance bias, temporal modeling, or spatial localization.

## Why This Concept Matters

Understanding video benchmarks is essential for:
- Comparing model performance fairly
- Identifying which tasks a model excels at
- Understanding dataset biases and limitations
- Choosing the right benchmark for evaluation
- Interpreting results in the context of dataset characteristics

## Mathematical Explanation

### Key Benchmarks

| Dataset | Classes | Clips | Task | Metric |
|---------|---------|-------|------|--------|
| Kinetics-400 | 400 | 306k | Classification | Top-1 / Top-5 Acc |
| Kinetics-600 | 600 | 480k | Classification | Top-1 / Top-5 Acc |
| Kinetics-700 | 700 | 650k | Classification | Top-1 / Top-5 Acc |
| Something-Something V2 | 174 | 220k | Classification | Top-1 / Top-5 Acc |
| UCF-101 | 101 | 13k | Classification | Top-1 Acc |
| HMDB-51 | 51 | 7k | Classification | Top-1 Acc |
| AVA | 80 | 430k | Detection | mAP @ IoU=0.5 |
| ActivityNet v1.3 | 200 | 20k | Detection | mAP @ tIoU |
| Diving48 | 48 | 16k | Classification | Top-1 Acc |

### Evaluation Metrics

1. Top-1 Accuracy: fraction of correctly classified clips
2. Top-5 Accuracy: fraction where correct class is in top-5 predictions
3. Mean Average Precision (mAP): average precision across classes at a given IoU threshold
4. Center-crop vs. Ten-crop: evaluation augmentation strategy
5. Clip-level vs. Video-level: averaging predictions across uniformly sampled clips

## Code Examples

### Example 1: Kinetics Dataset Loading

```python
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import json

class KineticsDataset(Dataset):
    def __init__(self, root_dir, annotation_file, frames_per_clip=16, frame_size=224):
        self.root_dir = root_dir
        self.frames_per_clip = frames_per_clip
        self.frame_size = frame_size
        with open(annotation_file) as f:
            self.annotations = json.load(f)
        self.classes = sorted(set(a['label'] for a in self.annotations))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        video_path = os.path.join(self.root_dir, ann['video'])
        frames = sorted(os.listdir(video_path))[:self.frames_per_clip]
        # Pad if not enough frames
        if len(frames) < self.frames_per_clip:
            frames = frames + [frames[-1]] * (self.frames_per_clip - len(frames))
        clip = []
        for frame in frames:
            img = Image.open(os.path.join(video_path, frame)).resize((self.frame_size, self.frame_size))
            clip.append(torch.tensor(list(img.getdata())).reshape(self.frame_size, self.frame_size, 3).permute(2, 0, 1).float() / 255.0)
        clip = torch.stack(clip, dim=1)  # [C, T, H, W]
        label = self.class_to_idx[ann['label']]
        return clip, label

# Usage example
ds = KineticsDataset('/data/kinetics/train', '/data/kinetics/train.json')
print(f"Kinetics classes: {len(ds.classes)}, samples: {len(ds)}")
# Output: Kinetics classes: 400, samples: 240000
```

### Example 2: Benchmark Evaluation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def evaluate_top1_top5(model, dataloader, num_classes=400):
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for videos, labels in dataloader:
            # videos: [N, C, T, H, W]
            outputs = model(videos)  # [N, num_classes]
            _, preds = outputs.topk(5, dim=1)
            labels = labels.unsqueeze(1)

            top1_correct += (preds[:, :1] == labels).sum().item()
            top5_correct += (preds == labels).sum().item()
            total += labels.size(0)

    top1_acc = 100.0 * top1_correct / total
    top5_acc = 100.0 * top5_correct / total
    return top1_acc, top5_acc

# Simulated evaluation
logits = torch.randn(1000, 400)
labels = torch.randint(0, 400, (1000,))
_, preds = logits.topk(5, dim=1)
top1 = (preds[:, :1] == labels.unsqueeze(1)).float().mean().item()
top5 = (preds == labels.unsqueeze(1)).float().sum(dim=1).mean().item()
print(f"Simulated Kinetics results - Top-1: {top1*100:.1f}%, Top-5: {top5*100:.1f}%")
# Output: Simulated Kinetics results - Top-1: 0.2%, Top-5: 1.6%
```

### Example 3: Something-Something V2 Evaluation

```python
import torch
import torch.nn as nn

def evaluate_something_something(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for videos, labels in dataloader:
            # videos: [N, C, T, H, W]
            outputs = model(videos)

            # Something-Something requires temporal reasoning
            # Test both forward and backward clips
            outputs_fwd = outputs
            outputs_bwd = model(torch.flip(videos, dims=[2]))
            outputs_avg = (outputs_fwd + outputs_bwd) / 2.0

            _, preds = outputs_avg.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100.0 * correct / total
    return acc

print("Something-Something V2 evaluation function defined")
# Output: Something-Something V2 evaluation function defined
```

## Common Mistakes

1. **Using different evaluation protocols**: Some papers report center-crop results, others report ten-crop or 3-crop. Comparing models without knowing the protocol is misleading.

2. **Not accounting for Kinetics class overlap**: Kinetics-600 and 700 contain all classes from Kinetics-400. Fine-tuning on Kinetics-400 and testing on Kinetics-600 gives an unfair advantage.

3. **Ignoring dataset bias**: Kinetics has strong appearance bias (objects, scenes). A model may recognize archery by the bow (appearance) rather than the motion. Something-Something removes objects to test pure temporal reasoning.

4. **Training on the validation set**: Many papers accidentally or intentionally train models using validation data. This overestimates performance and is not reproducible.

5. **Inconsistent frame sampling**: Some methods sample uniformly from the video, others sample random clips. Frame rate and sampling strategy significantly affect results.

## Interview Questions

### Beginner - 5

1. What is the Kinetics dataset?
2. What is the difference between UCF-101 and HMDB-51?
3. How is top-1 accuracy computed?
4. What is Something-Something V2 designed to test?
5. How many clips are in Kinetics-400?

### Intermediate - 5

1. Compare Kinetics-400 with Something-Something V2.
2. What is the AVA benchmark and what task does it evaluate?
3. How does ActivityNet define temporal action detection?
4. Why are center-crop and ten-crop evaluations different?
5. What is the Diving48 benchmark?

### Advanced - 3

1. Analyze the bias in Kinetics vs. Something-Something V2.
2. Design a new benchmark that addresses limitations of Kinetics.
3. How would you evaluate spatiotemporal localization models?

## Practice Problems

### Easy - 5

1. Download and inspect the Kinetics dataset.
2. Count the number of classes in UCF-101.
3. Implement top-1 and top-5 accuracy.
4. Compute the average clip length in Kinetics.
5. List all available Kinetics versions.

### Medium - 5

1. Write a dataloader for Kinetics with frame sampling.
2. Implement Something-Something V2 evaluation.
3. Compute mAP for AVA detection.
4. Create training/validation splits for UCF-101.
5. Implement ten-crop evaluation.

### Hard - 3

1. Design a new video benchmark for fine-grained action understanding.
2. Implement the ActivityNet evaluation metric.
3. Build a unified evaluation framework for multiple video benchmarks.

## Solutions

Easy 1:
```python
# Kinetics-400: 400 action classes, ~306k video clips
# Each clip is approximately 10 seconds
# Download from https://deepmind.com/research/open-source/kinetics

print("Kinetics-400 statistics:")
print("Classes: 400")
print("Training clips: ~240,000")
print("Validation clips: ~20,000")
print("Test clips: ~40,000")
print("Clip duration: 10 seconds")
# Output: Kinetics-400 statistics:
# Output: Classes: 400
# Output: Training clips: ~240,000
# Output: Validation clips: ~20,000
# Output: Test clips: ~40,000
# Output: Clip duration: 10 seconds
```

Medium 1 — Kinetics Dataloader:
```python
class KineticsClipSampler:
    def __init__(self, num_frames=16, temporal_stride=2):
        self.num_frames = num_frames
        self.stride = temporal_stride

    def sample_clip(self, video_frames):
        total = len(video_frames)
        # Segment-based sampling (uniform)
        segment_size = total / self.num_frames
        indices = []
        for i in range(self.num_frames):
            start = int(segment_size * i)
            end = int(segment_size * (i + 1))
            idx = torch.randint(start, max(end, start + 1), (1,)).item()
            indices.append(idx)
        return [video_frames[i] for i in indices]

sampler = KineticsClipSampler(16, 2)
dummy_frames = list(range(300))  # 300 frames = 10s at 30fps
clip = sampler.sample_clip(dummy_frames)
print(f"Sampled {len(clip)} frames from {len(dummy_frames)} total")
# Output: Sampled 16 frames from 300 total
```

## Related Concepts

- DL-271: Video Classification
- DL-272: Action Recognition
- DL-280: Video Benchmarks

## Next Concepts

- DL-281: VideoMAE V2

## Summary

Video benchmarks provide standardized datasets and evaluation protocols for measuring video understanding performance. Kinetics-400/600/700 are the most widely used benchmarks for action classification. Something-Something V2 tests temporal reasoning. Smaller benchmarks (UCF-101, HMDB-51) are used for ablations. AVA and ActivityNet evaluate spatiotemporal and temporal action detection. Understanding benchmark characteristics, biases, and evaluation protocols is essential for fair comparison and meaningful progress.

## Key Takeaways

- Kinetics-400: 400 classes, 306k clips, top-1 accuracy
- Something-Something V2: 174 classes, temporal reasoning focus
- UCF-101 / HMDB-51: small benchmarks for ablations
- AVA: spatiotemporal action detection with mAP
- ActivityNet: temporal action localization with tIoU
- Center-crop vs. ten-crop affects reported numbers
- Dataset bias (appearance vs. motion) influences results
- Consistent evaluation protocol is critical for fair comparison
- Current SOTA on Kinetics-400: ~88% top-1 (VideoMAE, InternVideo)
