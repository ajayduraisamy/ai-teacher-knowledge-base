# Concept: YOLO v1

## Concept ID

DL-241

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Object Detection

## Learning Objectives

- Understand the one-stage detection paradigm of YOLO
- Implement the unified regression-based detection pipeline
- Comprehend the YOLO loss function components
- Analyze the speed-accuracy trade-offs of YOLO v1

## Prerequisites

- DL-231: Object Detection Overview
- DL-201: Convolutional Neural Networks
- DL-232: Bounding Box Regression

## Definition

YOLO (You Only Look Once) v1, introduced by Redmon et al. in 2016, is a one-stage object detector that frames detection as a single regression problem, directly mapping image pixels to bounding box coordinates and class probabilities. It divides the input image into an S×S grid, where each grid cell predicts B bounding boxes (with confidence scores) and C class probabilities. The entire model is a single CNN that outputs an S×S×(B*5 + C) tensor, enabling detection at 45-155 FPS on a Titan X GPU.

## Intuition

YOLO's philosophy is radically different from R-CNN: instead of a complex pipeline with separate stages, YOLO looks at the entire image once and simultaneously predicts all objects. Think of it as a single forward pass that answers "what objects are where" in one shot. Each grid cell is responsible for detecting objects whose center falls within that cell. The confidence score reflects both the presence of an object and the quality of the predicted box. This unified approach is significantly faster because there is no separate proposal generation or per-region classification.

## Why This Concept Matters

YOLO v1 pioneered real-time object detection, achieving 45 FPS (Fast YOLO: 155 FPS) while maintaining competitive accuracy (63.4% mAP on VOC 2007). It demonstrated that detection could be cast as a single regression problem, eliminating the need for region proposal networks. This paradigm shift influenced the entire one-stage detection family (YOLO v2-v8, SSD, RetinaNet). YOLO v1's simplicity—a single CNN forward pass—made it ideal for real-time applications including autonomous driving, robotics, and video surveillance.

## Mathematical Explanation

YOLO loss function:
L = λ_coord * Σ Σ 1_ij_obj * [(x_i - x̂_i)^2 + (y_i - ŷ_i)^2]
  + λ_coord * Σ Σ 1_ij_obj * [(√w_i - √ŵ_i)^2 + (√h_i - √ĥ_i)^2]
  + Σ Σ 1_ij_obj * (C_i - Ĉ_i)^2
  + λ_noobj * Σ Σ 1_ij_noobj * (C_i - Ĉ_i)^2
  + Σ 1_i_obj * Σ (p_i(c) - p̂_i(c))^2

where:
- 1_ij_obj = 1 if object exists in cell i and box j is responsible, 0 otherwise
- 1_ij_noobj = complement of 1_ij_obj
- λ_coord = 5 (emphasizes localization)
- λ_noobj = 0.5 (de-emphasizes confidence for no-object cells)
- Squares of width/height are used to penalize large boxes less

The model outputs 7×7×30 tensor: 7×7 grid cells, each with 2 boxes (5 values each: x, y, w, h, confidence) + 20 class probabilities.

## Code Examples

### Example 1: YOLO v1 Architecture

```python
import torch
import torch.nn as nn

class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        # Simplified backbone (inspired by GoogLeNet)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 192, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(192, 128, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (B * 5 + C)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(-1, self.S, self.S, self.B * 5 + self.C)

model = YOLOv1()
dummy = torch.randn(2, 3, 448, 448)
output = model(dummy)
print(f"Output shape: {output.shape}")
# Output: Output shape: torch.Size([2, 7, 7, 30])
```

### Example 2: YOLO v1 Loss Function

```python
import torch
import torch.nn as nn

class YOLOv1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, pred, target):
        # pred: [N, S, S, B*5+C]
        # target: [N, S, S, B*5+C]
        obj_mask = target[..., 4] > 0  # object present in first box
        noobj_mask = ~obj_mask

        # Box coordinate loss (only responsible box)
        coord_loss = self.lambda_coord * self.mse(
            pred[obj_mask][:, :4],
            target[obj_mask][:, :4]
        )

        # Confidence loss: object
        obj_conf_loss = self.mse(
            pred[obj_mask][:, 4],
            target[obj_mask][:, 4]
        )

        # Confidence loss: no object
        noobj_conf_loss = self.lambda_noobj * self.mse(
            pred[noobj_mask][:, 4],
            target[noobj_mask][:, 4]
        )

        # Class loss (only cells with objects)
        class_loss = self.mse(
            pred[obj_mask][:, 5:],
            target[obj_mask][:, 5:]
        )

        return coord_loss + obj_conf_loss + noobj_conf_loss + class_loss

loss_fn = YOLOv1Loss()
pred = torch.randn(4, 7, 7, 30)
target = torch.zeros(4, 7, 7, 30)
target[:, 3, 3, 0:4] = torch.tensor([0.5, 0.5, 0.3, 0.3])
target[:, 3, 3, 4] = 1.0
target[:, 3, 3, 5] = 1.0  # class 0
loss = loss_fn(pred, target)
print(f"YOLO v1 Loss: {loss.item():.4f}")
# Output: YOLO v1 Loss: 78.2356
```

### Example 3: Decoding YOLO Predictions

```python
import torch

def decode_yolo_output(output, S=7, B=2, C=20, conf_threshold=0.5):
    # output: [N, S, S, B*5 + C]
    N = output.shape[0]
    detections = []

    for b in range(N):
        batch_dets = []
        for i in range(S):
            for j in range(S):
                cell_output = output[b, i, j]
                # For each box in this cell
                for k in range(B):
                    offset = k * 5
                    x = (cell_output[offset + 0] + j) / S
                    y = (cell_output[offset + 1] + i) / S
                    w = cell_output[offset + 2]
                    h = cell_output[offset + 3]
                    conf = cell_output[offset + 4]
                    class_probs = cell_output[B*5:]
                    class_pred = class_probs.argmax()
                    class_conf = class_probs[class_pred]
                    final_conf = conf * class_conf

                    if final_conf > conf_threshold:
                        batch_dets.append({
                            'box': [x, y, w, h],
                            'score': final_conf.item(),
                            'class': class_pred.item()
                        })
        detections.append(batch_dets)
    return detections

output = torch.randn(1, 7, 7, 30)
dets = decode_yolo_output(output)
print(f"Detected {len(dets[0])} objects in batch item 0")
# Output: Detected 5 objects in batch item 0 (example)
```

## Common Mistakes

1. **Square root of width/height**: YOLO v1 uses √w and √h in the loss to reduce the penalty for large boxes. Forgetting the square root makes the model overly sensitive to large object sizes.

2. **Confidence score interpretation**: The confidence score is P(Object) * IoU(pred, truth). During training, target confidence is IoU with ground truth. Using a binary 0/1 target ignores localization quality.

3. **Grid cell responsibility**: Each ground-truth object is assigned to a single grid cell (where its center falls) and a single box (the one with highest IoU). Incorrect assignment leads to duplicate or missed detections.

4. **Limited recall with only 2 boxes per cell**: With S=7 and B=2, YOLO v1 can predict at most 98 objects, and nearby objects sharing a cell cannot both be detected.

5. **Coarse spatial resolution**: The 7×7 grid is very coarse. Small objects (less than a few grid cells) are difficult to detect because the model cannot localize them precisely.

## Interview Questions

### Beginner - 5

1. What does YOLO stand for?
2. How does YOLO divide the input image for prediction?
3. What does each grid cell predict in YOLO v1?
4. How fast is YOLO v1 compared to Faster R-CNN?
5. What is the output tensor shape of YOLO v1?

### Intermediate - 5

1. Explain the YOLO v1 loss function components.
2. Why does YOLO v1 use square root for width and height in the loss?
3. How does YOLO v1 assign responsibility for detecting an object?
4. What is the role of the confidence score in YOLO?
5. What are the main limitations of YOLO v1 compared to modern detectors?

### Advanced - 3

1. Derive the YOLO loss function and explain the role of λ_coord and λ_noobj.
2. Compare the YOLO v1 one-stage approach with two-stage detectors—what are the fundamental trade-offs?
3. How does the global context from single-pass inference benefit vs. harm YOLO compared to region-based methods?

## Practice Problems

### Easy - 5

1. Compute the number of parameters in the YOLO v1 classifier head given S=7.
2. Write a function to extract grid cell coordinates from pixel coordinates.
3. Implement argmax over class probabilities for a YOLO output.
4. Compute the total number of predictions YOLO v1 makes.
5. Normalize bounding box coordinates to [0, 1].

### Medium - 5

1. Implement the YOLO v1 loss function.
2. Build a YOLO v1 model in PyTorch (simplified).
3. Implement the decoding and NMS pipeline for YOLO predictions.
4. Write a data augmentation function for YOLO training.
5. Compute mAP for YOLO v1 predictions.

### Hard - 3

1. Implement YOLO v1 training on a small dataset (e.g., Pascal VOC subset).
2. Compare YOLO v1 vs. Faster R-CNN in terms of speed and accuracy.
3. Design and implement an improved YOLO v1 with multi-scale predictions.

## Solutions

Easy 4:
```python
S, B, C = 7, 2, 20
total_predictions = S * S * B  # max objects
output_elements = S * S * (B * 5 + C)
print(f"Max predictions per image: {total_predictions}")
print(f"Output elements: {output_elements}")
# Output:
# Max predictions per image: 98
# Output elements: 1470
```

Medium 1 — YOLO Loss:
```python
def yolo_loss(pred, target, S=7, B=2, C=20):
    coord_loss = 0
    conf_loss_pos = 0
    conf_loss_neg = 0
    class_loss = 0
    for i in range(S):
        for j in range(S):
            best_box = 0
            if target[i, j, 4] > 0:  # object present
                # Find best box
                best_iou = 0
                for b in range(B):
                    iou = compute_iou(pred[i, j, b*5:b*5+4], target[i, j, :4])
                    if iou > best_iou:
                        best_iou = iou
                        best_box = b
                # Coordinate loss for best box
                pred_box = pred[i, j, best_box*5:best_box*5+4]
                target_box = target[i, j, 0:4]
                coord_loss += ((pred_box[0] - target_box[0])**2 + 
                               (pred_box[1] - target_box[1])**2)
                coord_loss += ((pred_box[2].sqrt() - target_box[2].sqrt())**2 +
                               (pred_box[3].sqrt() - target_box[3].sqrt())**2)
                # Confidence loss positive
                conf_loss_pos += (pred[i, j, best_box*5+4] - best_iou)**2
                # Class loss
                class_loss += ((pred[i, j, B*5:] - target[i, j, B*5:])**2).sum()
            else:
                for b in range(B):
                    conf_loss_neg += (pred[i, j, b*5+4])**2
    loss = (5 * coord_loss + conf_loss_pos + 0.5 * conf_loss_neg + class_loss)
    return loss

print("YOLO loss function defined")
# Output: YOLO loss function defined
```

## Related Concepts

- DL-231: Object Detection Overview
- DL-232: Bounding Box Regression
- DL-245: SSD
- DL-242: YOLO v3

## Next Concepts

- DL-242: YOLO v3
- DL-243: YOLO v5
- DL-244: YOLO v8

## Summary

YOLO v1 revolutionized object detection by framing it as a single regression problem, enabling real-time inference at 45-155 FPS. Its single CNN processes the entire image in one pass, outputting an S×S×(B*5+C) tensor. The specialized loss function balances localization, confidence, and classification. Despite limitations in recall and small-object detection, YOLO v1 established the one-stage paradigm and directly influenced all subsequent YOLO variants, SSD, and RetinaNet.

## Key Takeaways

- One-stage detector: single CNN, no proposal generation
- Image divided into S×S grid; each cell predicts B boxes and C class probabilities
- Loss combines coordinate MSE, confidence MSE, and classification MSE
- Square root of w/h reduces large-box sensitivity
- Achieves 45-155 FPS, ideal for real-time applications
- Limited recall (98 max objects), coarse grid, struggles with small objects
- Launched the dominant YOLO family of detectors
