# Concept: YOLO v3

## Concept ID

DL-242

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Object Detection

## Learning Objectives

- Understand the multi-scale detection mechanism in YOLO v3
- Implement the Darknet-53 backbone and its residual connections
- Comprehend the use of logistic classifiers instead of softmax
- Analyze how YOLO v3 improved upon YOLO v1 and v2

## Prerequisites

- DL-241: YOLO v1
- DL-201: Convolutional Neural Networks
- DL-233: Anchor Boxes

## Definition

YOLO v3, introduced by Redmon and Farhadi in 2018, is a significant evolution of the YOLO family that incorporates multi-scale predictions, a deeper Darknet-53 backbone with residual connections, and logistic classifiers for multi-label classification. It predicts bounding boxes at three different scales using feature pyramid concepts, uses 9 anchor boxes (3 per scale), and outputs a tensor of size N×N×(3×(4+1+80)) for each scale. YOLO v3 achieved 57.9% mAP@50 on COCO at 20 FPS, offering an excellent speed-accuracy trade-off.

## Intuition

YOLO v3 addresses YOLO v1's fundamental weakness: detecting objects at multiple scales. By making predictions at three different feature map resolutions (large, medium, small), the model can detect both large and small objects effectively. The Darknet-53 backbone uses residual blocks (inspired by ResNet) to build a deeper network without vanishing gradients. Each scale combines features from the current level via upsampling from the previous level, forming a simple Feature Pyramid Network. Multi-label classification with independent logistic classifiers enables detecting overlapping objects (e.g., "person" and "woman").

## Why This Concept Matters

YOLO v3 was a major milestone that made YOLO competitive with state-of-the-art detectors while maintaining real-time speed. Its innovations—multi-scale prediction, residual backbone, logistic classification—set the template for all subsequent YOLO versions. YOLO v3 became one of the most widely deployed detectors in industry due to its excellent speed-accuracy balance. The Darknet-53 backbone with FPN-style multi-scale predictions influenced numerous other architectures.

## Mathematical Explanation

YOLO v3 predicts at 3 scales with strides 32, 16, and 8. At each scale:
- Feature map size: S_i × S_i where S_i ∈ {13, 26, 52} for 416×416 input
- Each cell predicts 3 bounding boxes
- For each box: (t_x, t_y, t_w, t_h, objectness) + 80 class scores

Box decoding:
b_x = σ(t_x) + c_x
b_y = σ(t_y) + c_y
b_w = p_w * exp(t_w)
b_h = p_h * exp(t_h)

where σ is sigmoid, (c_x, c_y) is grid cell offset, and (p_w, p_h) is anchor dimension.

Loss: Binary cross-entropy for class and objectness scores, MSE or GIoU for box coordinates.

## Code Examples

### Example 1: Darknet-53 Building Block

```python
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels // 2, 1)
        self.conv2 = ConvBlock(channels // 2, channels, 3)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(3, 32, 3)
        self.layer1 = self._make_layer(32, 64, 1)    # 1 res block
        self.layer2 = self._make_layer(64, 128, 2)   # 2 res blocks
        self.layer3 = self._make_layer(128, 256, 8)  # 8 res blocks
        self.layer4 = self._make_layer(256, 512, 8)  # 8 res blocks
        self.layer5 = self._make_layer(512, 1024, 4) # 4 res blocks

    def _make_layer(self, in_c, out_c, num_blocks):
        layers = [ConvBlock(in_c, out_c, 3, stride=2)]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(out_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)   # scale 1 (large objects)
        x4 = self.layer4(x3)  # scale 2 (medium objects)
        x5 = self.layer5(x4)  # scale 3 (small objects)
        return x3, x4, x5

backbone = Darknet53()
dummy = torch.randn(1, 3, 416, 416)
f3, f4, f5 = backbone(dummy)
print(f"Multi-scale features: {f3.shape}, {f4.shape}, {f5.shape}")
# Output: Multi-scale features: torch.Size([1, 256, 52, 52]), torch.Size([1, 512, 26, 26]), torch.Size([1, 1024, 13, 13])
```

### Example 2: YOLO v3 Detection Head

```python
import torch
import torch.nn as nn

class YOLOv3Head(nn.Module):
    def __init__(self, in_channels, num_classes=80, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv = nn.Sequential(
            ConvBlock(in_channels, in_channels // 2, 1),
            ConvBlock(in_channels // 2, in_channels, 3),
            ConvBlock(in_channels, in_channels // 2, 1),
            ConvBlock(in_channels // 2, in_channels, 3),
        )
        self.out = nn.Conv2d(in_channels, num_anchors * (5 + num_classes), 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.out(x)
        # Reshape: [N, A*(5+C), H, W] -> [N, A, H, W, 5+C]
        N, _, H, W = x.shape
        x = x.view(N, self.num_anchors, 5 + self.num_classes, H, W)
        x = x.permute(0, 1, 3, 4, 2)  # [N, A, H, W, 5+C]
        return x

# Three heads for three scales
head_small = YOLOv3Head(1024)  # stride 32
head_medium = YOLOv3Head(512)  # stride 16
head_large = YOLOv3Head(256)   # stride 8

f3, f4, f5 = backbone(dummy)
o1 = head_large(f3)    # large objects
o2 = head_medium(f4)   # medium objects
o3 = head_small(f5)    # small objects
print(f"Output shapes: {o1.shape}, {o2.shape}, {o3.shape}")
# Output: Output shapes: torch.Size([1, 3, 52, 52, 85]), torch.Size([1, 3, 26, 26, 85]), torch.Size([1, 3, 13, 13, 85])
```

### Example 3: Decoding YOLO v3 Predictions

```python
import torch

def decode_yolo_v3(output, anchors, stride, num_classes=80):
    # output: [N, A, H, W, 5+C]
    # anchors: [A, 2] (w, h)
    N, A, H, W, C = output.shape
    device = output.device

    # Grid cell coordinates
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device),
                                     torch.arange(W, device=device))
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, H, W, 2)

    box_xy = torch.sigmoid(output[..., 0:2]) + grid
    box_wh = torch.exp(output[..., 2:4]) * anchors.view(1, A, 1, 1, 2) / stride
    objectness = torch.sigmoid(output[..., 4:5])
    class_probs = torch.sigmoid(output[..., 5:])

    boxes = torch.cat([box_xy, box_wh], dim=-1)
    scores = objectness * class_probs

    return boxes, scores

output = torch.randn(1, 3, 13, 13, 85)
anchors = torch.tensor([[116, 90], [156, 198], [373, 326]], dtype=torch.float32)
boxes, scores = decode_yolo_v3(output, anchors, stride=32)
print(f"Decoded boxes shape: {boxes.shape}")
print(f"Scores shape: {scores.shape}")
# Output:
# Decoded boxes shape: torch.Size([1, 3, 13, 13, 4])
# Scores shape: torch.Size([1, 3, 13, 13, 80])
```

## Common Mistakes

1. **Softmax instead of logistic classifiers**: YOLO v3 uses independent sigmoid per class, not softmax. Softmax assumes classes are mutually exclusive, which is false for multi-label cases like "person" + "woman".

2. **Incorrect anchor assignment across scales**: The 9 anchors are distributed across 3 scales. Using anchors meant for the large-scale head on the small-scale head significantly harms detection.

3. **Forgetting upsampling connections**: The FPN connections from deeper to shallower scales are critical. Without them, each scale lacks context from other scales.

4. **Objectness target assignment**: The objectness score should target IoU (like YOLO v1), not a binary 0/1. Using binary targets penalizes good but imperfect boxes unfairly.

5. **Scale-specific NMS threshold**: Different scales may benefit from different NMS thresholds. Using a single threshold across all scales can over-suppress small objects.

## Interview Questions

### Beginner - 5

1. How many scales does YOLO v3 predict at?
2. What backbone does YOLO v3 use?
3. How many anchor boxes does YOLO v3 use?
4. What activation does YOLO v3 use for class predictions?
5. What is the input resolution typically used in YOLO v3?

### Intermediate - 5

1. Explain the multi-scale prediction mechanism in YOLO v3.
2. How does the FPN-style architecture work in YOLO v3?
3. Why does YOLO v3 use sigmoid instead of softmax for class predictions?
4. How are anchors assigned to different prediction scales?
5. What is the Darknet-53 architecture and why is it effective?

### Advanced - 3

1. Compare the loss function of YOLO v3 with YOLO v1. What changed and why?
2. Analyze the impact of multi-scale predictions on small object detection performance.
3. How does YOLO v3 balance speed and accuracy compared to contemporary detectors like RetinaNet?

## Practice Problems

### Easy - 5

1. Count the number of anchors used at each scale.
2. Compute output tensor size for a 416×416 input.
3. Implement sigmoid activation for class predictions.
4. Write a function to compute anchor dimensions for each scale.
5. Calculate the total number of predictions across all scales.

### Medium - 5

1. Implement the YOLO v3 decoding function.
2. Build the Darknet-53 backbone in PyTorch.
3. Implement the YOLO v3 loss function.
4. Write a function to perform NMS after decoding all scales.
5. Implement the FPN upsampling connections.

### Hard - 3

1. Implement full YOLO v3 training pipeline.
2. Compare YOLO v3 vs. YOLO v4 vs. YOLO v5 on a benchmark.
3. Implement a YOLO v3 variant with attention mechanisms.

## Solutions

Easy 3:
```python
import torch
import torch.nn.functional as F

logits = torch.randn(3, 80)
probs = torch.sigmoid(logits)
print(f"Sigmoid probabilities: {probs.shape}")
# Output: Sigmoid probabilities: torch.Size([3, 80])
```

Medium 1 — Full Decoding with NMS:
```python
def yolo_v3_postprocess(outputs, anchors_list, strides, conf_thresh=0.5, nms_thresh=0.5):
    all_boxes = []
    all_scores = []
    all_labels = []
    for output, anchors, stride in zip(outputs, anchors_list, strides):
        boxes, scores = decode_yolo_v3(output, anchors, stride)
        for b in range(boxes.shape[0]):
            for i in range(boxes.shape[1]):
                for j in range(boxes.shape[2]):
                    for k in range(boxes.shape[3]):
                        score, label = scores[b, i, j, k].max(dim=0)
                        if score > conf_thresh:
                            all_boxes.append(boxes[b, i, j, k])
                            all_scores.append(score)
                            all_labels.append(label)
    if len(all_boxes) > 0:
        boxes_t = torch.stack(all_boxes)
        scores_t = torch.stack(all_scores)
        labels_t = torch.tensor(all_labels)
        keep = torchvision.ops.batched_nms(boxes_t, scores_t, labels_t, nms_thresh)
        return boxes_t[keep], scores_t[keep], labels_t[keep]
    return [], [], []

print("Post-processing function defined")
# Output: Post-processing function defined
```

## Related Concepts

- DL-241: YOLO v1
- DL-243: YOLO v5
- DL-244: YOLO v8
- DL-245: SSD

## Next Concepts

- DL-243: YOLO v5
- DL-244: YOLO v8

## Summary

YOLO v3 significantly advanced the YOLO family with multi-scale predictions, a deeper Darknet-53 residual backbone, logistic classifiers for multi-label detection, and 9 anchor boxes across 3 scales. It achieved competitive accuracy (57.9% COCO mAP@50) at real-time speeds (20-30 FPS), establishing the standard architecture for subsequent YOLO variants. The FPN-style multi-scale predictions and independent logistic classifiers became standard design choices in modern one-stage detectors.

## Key Takeaways

- Three-scale predictions (strides 32, 16, 8) for multi-scale detection
- Darknet-53 backbone with residual blocks
- Independent logistic classifiers (sigmoid) replace softmax
- 9 anchor boxes, 3 per scale
- FPN-style upsampling connections between scales
- Competitive accuracy at real-time inference speeds
- Set the template for all subsequent YOLO versions
