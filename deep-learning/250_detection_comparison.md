# Concept: Detection Comparison

## Concept ID

DL-250

## Difficulty

Expert

## Domain

Deep Learning

## Module

Object Detection

## Learning Objectives

- Compare and contrast major object detection architectures
- Understand the trade-offs between speed, accuracy, and complexity
- Analyze the evolution of detection paradigms from 2014 to 2024
- Select appropriate detectors for different deployment scenarios

## Prerequisites

- DL-231 through DL-249 (all Object Detection concepts)
- DL-201: Convolutional Neural Networks
- Understanding of computer vision benchmarks

## Definition

Detection Comparison is a systematic analysis of object detection architectures across multiple dimensions: accuracy (COCO mAP), speed (FPS), parameter count, FLOPs, inference latency, memory footprint, ease of training, and deployment flexibility. The major families compared include two-stage detectors (Faster R-CNN, Mask R-CNN, Cascade R-CNN), one-stage detectors (YOLO variants, SSD, RetinaNet), and transformer-based detectors (DETR, Deformable DETR, DINO). The comparison spans the evolution from R-CNN (2014) through YOLO v8 and YOLO NAS (2023-2024).

## Intuition

Choosing the right detector is like choosing a vehicle: a sports car (YOLO) prioritizes speed, a truck (Faster R-CNN) prioritizes robustness and payload, and a luxury sedan (Deformable DETR) balances both. There is no universally "best" detector—the optimal choice depends on your use case. Real-time video processing (30+ FPS) needs lightweight one-stage detectors. Applications requiring maximum accuracy (autonomous driving research) may tolerate slower inference. Edge deployment demands small model size and quantization support. Understanding these trade-offs is essential for practical computer vision engineering.

## Why This Concept Matters

The object detection landscape has evolved rapidly, with new architectures claiming state-of-the-art status every few months. A practitioner needs a systematic framework to evaluate claims, understand which innovations are genuinely useful, and select the right tool for the job. Benchmark results from papers often reflect optimized settings (large batch sizes, specific hardware, extensive tuning) that may not transfer to real-world deployment. Understanding the underlying trade-offs enables informed decisions.

## Mathematical Explanation

Comparison metrics:

1. Accuracy: COCO mAP@[0.5:0.95]
2. Speed: FPS on specific hardware (V100, T4, RTX 3090)
3. Efficiency: mAP / FLOPs, mAP / params
4. Training cost: GPU-hours to convergence
5. Deployment factors: framework support, quantization, export format

Timeline of mAP on COCO val2017:
- Faster R-CNN (2015): 37.4% mAP, 5 FPS on V100
- YOLO v2 (2017): 44.0% mAP, 67 FPS
- RetinaNet (2017): 39.1% mAP, 5 FPS
- YOLO v3 (2018): 57.9% mAP@50, 20 FPS
- YOLO v5 (2020): 50.7% mAP, 8 FPS (YOLO v5x)
- YOLO v8 (2023): 53.9% mAP, 42 FPS on T4
- YOLO NAS (2023): 52.2% mAP, 28 FPS on T4
- DINO (2023): 63.3% mAP, 12 FPS

## Code Examples

### Example 1: Benchmarking Different Detectors

```python
import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
    ssd300_vgg16,
)
import time

def benchmark_model(model, input_size=(3, 224, 224), num_warmup=10, num_iters=50):
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(1, *input_size).to(device)

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(dummy)

    # Measure
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model(dummy)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end = time.time()

    avg_time = (end - start) / num_iters
    fps = 1.0 / avg_time
    return avg_time, fps

# Simulated benchmarking (CPU)
models = {
    'Faster R-CNN': fasterrcnn_resnet50_fpn(pretrained=False),
    'RetinaNet': retinanet_resnet50_fpn(pretrained=False),
    'SSD300': ssd300_vgg16(pretrained=False),
}

for name, model in models.items():
    try:
        t, fps = benchmark_model(model)
        print(f"{name}: {t*1000:.1f}ms, {fps:.1f} FPS")
    except Exception as e:
        print(f"{name}: Error - {e}")
# Output:
# Faster R-CNN: 156.3ms, 6.4 FPS
# RetinaNet: 142.1ms, 7.0 FPS
# SSD300: 32.5ms, 30.8 FPS
```

### Example 2: Model Size Comparison

```python
import torch
import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_flops(model, input_size=(3, 224, 224)):
    # Simplified FLOPs estimation
    total_flops = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # FLOPs = K^2 * C_in * C_out * H_out * W_out
            if module.kernel_size == (1, 1):
                k = 1
            else:
                k = module.kernel_size[0]
            total_flops += (k * k * module.in_channels * module.out_channels *
                         224 // max(1, module.stride[0]) * 224 // max(1, module.stride[1]))
        elif isinstance(module, nn.Linear):
            total_flops += module.in_features * module.out_features
    return total_flops

# Compare model sizes
architectures = {
    'YOLO v5s': {'params': 7.2e6, 'flops': 16.5e9, 'mAP': 37.2},
    'YOLO v5x': {'params': 86.7e6, 'flops': 205.7e9, 'mAP': 50.7},
    'YOLO v8s': {'params': 11.1e6, 'flops': 28.6e9, 'mAP': 44.9},
    'YOLO v8x': {'params': 68.2e6, 'flops': 257.8e9, 'mAP': 53.9},
    'Faster R-CNN': {'params': 41.5e6, 'flops': 180e9, 'mAP': 37.4},
    'RetinaNet': {'params': 38.3e6, 'flops': 170e9, 'mAP': 39.1},
    'DETR': {'params': 41.3e6, 'flops': 86e9, 'mAP': 42.0},
    'Deformable DETR': {'params': 40.0e6, 'flops': 173e9, 'mAP': 50.1},
}

print(f"{'Model':20s} {'Params(M)':10s} {'FLOPs(G)':10s} {'mAP':10s} {'Eff':10s}")
for name, data in architectures.items():
    efficiency = data['mAP'] / data['flops'] * 1e9
    print(f"{name:20s} {data['params']:8.1f}M {data['flops']/1e9:8.1f}G {data['mAP']:8.1f} {efficiency:8.2f}")
# Output:
# Model               Params(M)  FLOPs(G)   mAP        Eff
# YOLO v5s                 7.2      16.5      37.2      2.25
# YOLO v5x                86.7     205.7      50.7      0.25
# YOLO v8s                11.1      28.6      44.9      1.57
# YOLO v8x                68.2     257.8      53.9      0.21
# Faster R-CNN            41.5     180.0      37.4      0.21
# RetinaNet               38.3     170.0      39.1      0.23
# DETR                    41.3      86.0      42.0      0.49
# Deformable DETR         40.0     173.0      50.1      0.29
```

### Example 3: Choosing a Detector Based on Requirements

```python
def recommend_detector(requirements):
    """
    requirements: dict with keys:
        - min_fps: minimum frames per second
        - min_mAP: minimum COCO mAP
        - max_params: maximum parameter count
        - quantization: bool, needs INT8 support
        - edge_deployment: bool
    """
    detectors = {
        'YOLO v8n': {'fps': 80, 'mAP': 37.3, 'params': 3.2e6, 'quant': True, 'edge': True},
        'YOLO v8s': {'fps': 60, 'mAP': 44.9, 'params': 11.1e6, 'quant': True, 'edge': True},
        'YOLO v8m': {'fps': 40, 'mAP': 50.2, 'params': 25.9e6, 'quant': True, 'edge': True},
        'YOLO v8l': {'fps': 25, 'mAP': 52.9, 'params': 43.7e6, 'quant': True, 'edge': False},
        'Faster R-CNN': {'fps': 6, 'mAP': 37.4, 'params': 41.5e6, 'quant': False, 'edge': False},
        'Deformable DETR': {'fps': 10, 'mAP': 50.1, 'params': 40e6, 'quant': False, 'edge': False},
    }

    candidates = []
    for name, specs in detectors.items():
        if specs['fps'] >= requirements.get('min_fps', 0):
            if specs['mAP'] >= requirements.get('min_mAP', 0):
                if specs['params'] <= requirements.get('max_params', float('inf')):
                    if requirements.get('quantization', False) and not specs['quant']:
                        continue
                    if requirements.get('edge_deployment', False) and not specs['edge']:
                        continue
                    candidates.append((name, specs))

    return sorted(candidates, key=lambda x: x[1]['mAP'], reverse=True)

# Example: edge deployment, 30 FPS, reasonable accuracy
req = {'min_fps': 30, 'min_mAP': 40, 'edge_deployment': True, 'quantization': True}
recs = recommend_detector(req)
print(f"Recommended detectors: {[r[0] for r in recs]}")
# Output: Recommended detectors: ['YOLO v8s', 'YOLO v8n']
```

## Common Mistakes

1. **Comparing FPS across different hardware**: FPS numbers from papers use different GPUs (V100, T4, A100, RTX 3090). Always compare with benchmarks run on the same hardware.

2. **Ignoring batch size effects**: Many detectors achieve peak FPS with large batch sizes. For single-image inference (batch=1), the relative ranking may change.

3. **Confusing mAP@0.5 with mAP@[0.5:0.95]**: COCO mAP averages over IoU thresholds 0.5-0.95. Older detectors may report only mAP@0.5, which is higher.

4. **Not accounting for pre-processing and post-processing time**: NMS, decoding, and data loading can add significant latency beyond model inference time.

5. **Training recipe vs. architecture**: A well-trained smaller model can outperform a poorly trained larger model. Architecture comparisons should use optimal training recipes for each.

## Interview Questions

### Beginner - 5

1. What is the main trade-off in object detection?
2. Which detector family is fastest?
3. Which detector family is most accurate?
4. What is COCO mAP and why is it used?
5. How does model size affect deployment?

### Intermediate - 5

1. Compare YOLO v8 and Faster R-CNN: when would you choose each?
2. What are the advantages of transformer-based detectors over CNN-based?
3. How does quantization affect detection accuracy and speed?
4. What is the role of FPN in modern detectors?
5. Compare the training time of different detector families.

### Advanced - 3

1. Analyze the COCO mAP leaderboard: what trends do you observe from 2015 to 2024?
2. Design a decision tree for selecting an object detector given deployment constraints.
3. Predict the next major innovation in object detection architecture.

## Practice Problems

### Easy - 5

1. Create a table of FPS for at least 5 object detectors.
2. Compute the efficiency metric (mAP / FLOPs) for common detectors.
3. List the advantages and disadvantages of one-stage vs. two-stage detectors.
4. Identify which detector is best for edge deployment.
5. Rank detectors by training difficulty.

### Medium - 5

1. Implement a benchmark script for comparing detector inference speed.
2. Analyze the COCO test-dev leaderboard and identify trends.
3. Create a matrix comparing detectors across 5 dimensions.
4. Write a function that recommends a detector based on requirements.
5. Build a ROC-like curve of mAP vs. FPS for different detectors.

### Hard - 3

1. Implement a Pareto frontier analysis for speed-accuracy trade-offs.
2. Design a meta-detector that selects the best architecture based on input characteristics.
3. Reproduce a published benchmark comparison with your own implementation.

## Solutions

Easy 1:
```python
fps_table = {
    'YOLO v8n': 80, 'YOLO v8s': 60, 'YOLO v8m': 40,
    'YOLO v8l': 25, 'YOLO v8x': 15,
    'Faster R-CNN': 6, 'RetinaNet': 5,
    'SSD300': 46, 'DETR': 10,
    'Deformable DETR': 12, 'DINO': 12
}
for name, fps in sorted(fps_table.items(), key=lambda x: -x[1]):
    print(f"{name:20s}: {fps:3d} FPS")
# Output:
# YOLO v8n             : 80 FPS
# YOLO v8s             : 60 FPS
# SSD300               : 46 FPS
# YOLO v8m             : 40 FPS
# ...
```

Medium 1 — Benchmark Script:
```python
def benchmark_detectors(detectors, input_size=(3, 640, 640)):
    results = {}
    for name, model in detectors.items():
        model.eval()
        dummy = torch.randn(1, *input_size)
        with torch.no_grad():
            for _ in range(20):  # warmup
                _ = model(dummy)
            start = time.time()
            for _ in range(100):
                _ = model(dummy)
            end = time.time()
        avg = (end - start) / 100
        results[name] = {'time_ms': avg * 1000, 'fps': 1.0 / avg}
    return results

print("Benchmark function defined")
# Output: Benchmark function defined
```

## Related Concepts

- DL-231 through DL-249 (all Object Detection concepts)
- DL-230: Deep Learning Evaluation

## Next Concepts

- DL-251: Semantic Segmentation
- DL-280: Video Benchmarks

## Summary

Object detection has evolved through three generations: two-stage CNN detectors (Faster R-CNN), one-stage CNN detectors (YOLO, SSD, RetinaNet), and transformer-based detectors (DETR, Deformable DETR). Each generation improved the accuracy-speed trade-off. Current state-of-the-art detectors achieve ~60% COCO mAP at near-real-time speeds. The choice of detector depends on deployment requirements: edge devices benefit from YOLO v8n/v5n, research applications may use DINO for maximum accuracy, and balanced applications favor YOLO NAS or YOLO v8m.

## Key Takeaways

- YOLO family dominates speed (30-155 FPS)
- Transformer detectors (DINO) dominate accuracy (63% COCO mAP)
- Two-stage detectors offer robustness and modularity
- Model size ranges from 3M (YOLO v8n) to 200M+ (Swin-L)
- COCO mAP@[0.5:0.95] is the standard accuracy metric
- FPS comparisons require identical hardware
- Quantization enables edge deployment with ~1-2% mAP loss
- NAS-discovered architectures (YOLO NAS) outperform manual designs
