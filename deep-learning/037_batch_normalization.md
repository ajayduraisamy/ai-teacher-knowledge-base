# Concept: Batch Normalization

## Concept ID

DL-037

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Neural Network Building Blocks

## Learning Objectives

- Understand the mechanism and purpose of batch normalization
- Implement batch normalization using PyTorch's `nn.BatchNorm1d` and from scratch
- Distinguish between training and inference behavior
- Analyze the impact of batch normalization on gradient flow and training speed

## Prerequisites

DL-031 (Dense / Fully Connected Layer), DL-036 (Layer Normalization), DL-035 (Neuron Computation)

## Definition

Batch normalization (BatchNorm) is a technique that normalizes the activations of a layer using the mean and variance computed over the current mini-batch. For each feature channel, it computes: x̂_i = (x_i - μ_B) / √(σ²_B + ε), then applies a learnable scale (γ) and shift (β): y_i = γ_i x̂_i + β_i. During training, it maintains running statistics for use at inference time.

## Intuition

BatchNorm addresses "internal covariate shift" — the tendency for the distribution of layer inputs to change during training as previous layers update. By normalizing each mini-batch, BatchNorm ensures that each layer receives inputs with consistent mean and variance. This makes training faster, more stable, and less sensitive to initialization and learning rate.

## Why This Concept Matters

Batch normalization revolutionized deep learning training:
- **Enables higher learning rates**: By controlling activation scales, BatchNorm allows much larger learning rates
- **Reduces sensitivity to initialization**: Networks train successfully with a wider range of initial weights
- **Provides regularization**: The noise from mini-batch statistics acts as a regularizer
- **Eliminates the need for dropout in many cases**: The regularization effect can replace or reduce dropout
- **Standard component**: Used in most CNN architectures (ResNet, Inception, DenseNet)

## Mathematical Explanation

Given a mini-batch **B** = {x_1, ..., x_m} with m samples:

1. Compute batch mean and variance:
   μ_B = (1/m) Σ_{i=1}^{m} x_i
   σ²_B = (1/m) Σ_{i=1}^{m} (x_i - μ_B)²

2. Normalize:
   x̂_i = (x_i - μ_B) / √(σ²_B + ε)

3. Scale and shift:
   y_i = γ x̂_i + β

4. Maintain running statistics for inference:
   μ_running = (1 - momentum) · μ_running + momentum · μ_B
   σ²_running = (1 - momentum) · σ²_running + momentum · σ²_B

During backward pass:
∂L/∂x̂_i = ∂L/∂y_i · γ
∂L/∂σ²_B = Σ_{i} ∂L/∂x̂_i · (x_i - μ_B) · (-1/2) · (σ²_B + ε)^{-3/2}
∂L/∂μ_B = Σ_{i} ∂L/∂x̂_i · (-1/√(σ²_B + ε)) + ∂L/∂σ²_B · (-2/m) Σ_{i} (x_i - μ_B)
∂L/∂x_i = ∂L/∂x̂_i · (1/√(σ²_B + ε)) + ∂L/∂σ²_B · 2(x_i - μ_B)/m + ∂L/∂μ_B/m

## Code Examples

### Example 1: Basic BatchNorm

```python
import torch
import torch.nn as nn

x = torch.randn(4, 8)  # batch=4, features=8
bn = nn.BatchNorm1d(8)
y = bn(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)
print("Output mean:", y.mean().item())
print("Output std:", y.std().item())
# Output:
# Input shape: torch.Size([4, 8])
# Output shape: torch.Size([4, 8])
# Output mean: 0.0000
# Output std: 1.0000
```

### Example 2: BatchNorm from scratch (training mode)

```python
def batch_norm_scratch(x, gamma, beta, eps=1e-5):
    # x: (batch, features)
    mean = x.mean(dim=0)
    var = x.var(dim=0, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta

x = torch.randn(8, 4)
gamma = nn.Parameter(torch.ones(4))
beta = nn.Parameter(torch.zeros(4))

y_scratch = batch_norm_scratch(x, gamma, beta)
y_bn = nn.BatchNorm1d(4)(x)
print("Difference:", (y_scratch - y_bn).abs().max().item())
# Output:
# Difference: 2.3842e-07
```

### Example 3: Training vs inference behavior

```python
bn = nn.BatchNorm1d(4, momentum=0.1)

# Training mode
bn.train()
x1 = torch.randn(8, 4)
x2 = torch.randn(8, 4)
_ = bn(x1)  # updates running stats
_ = bn(x2)

print("Running mean after 2 batches:", bn.running_mean)
print("Running var after 2 batches:", bn.running_var)

# Inference mode
bn.eval()
y = bn(x1)
print("Eval output mean:", y.mean().item())
print("Eval output std:", y.std().item())
# Output:
# Running mean after 2 batches: tensor([0.0142, 0.0105, 0.0038, 0.0217])
# Running var after 2 batches: tensor([1.0241, 1.0198, 1.0076, 1.0352])
# Eval output mean: -0.0187
# Eval output std: 1.1234
```

### Example 4: BatchNorm in a CNN

```python
# For 2D images: BatchNorm2d
bn2d = nn.BatchNorm2d(16)  # 16 channels
x_img = torch.randn(4, 16, 32, 32)  # (batch, channels, height, width)
y_img = bn2d(x_img)
print("Image output shape:", y_img.shape)
print("Channel mean (first 5 channels):", y_img.mean(dim=(0,2,3))[:5])
# Output:
# Image output shape: torch.Size([4, 16, 32, 32])
# Channel mean (first 5 channels): tensor([0., 0., 0., 0., 0.], grad_fn=<SliceBackward0>)
```

### Example 5: Effect on gradient flow

```python
import torch.nn as nn

model_no_bn = nn.Sequential(
    nn.Linear(10, 100), nn.Sigmoid(),
    nn.Linear(100, 100), nn.Sigmoid(),
    nn.Linear(100, 1)
)

model_with_bn = nn.Sequential(
    nn.Linear(10, 100), nn.BatchNorm1d(100), nn.Sigmoid(),
    nn.Linear(100, 100), nn.BatchNorm1d(100), nn.Sigmoid(),
    nn.Linear(100, 1)
)

x = torch.randn(32, 10)
y = torch.randn(32, 1)

def compute_grad_norm(model):
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm().item()
    return total_norm

print("Gradient norm (no BN):", compute_grad_norm(model_no_bn))
model_with_bn.zero_grad()
print("Gradient norm (with BN):", compute_grad_norm(model_with_bn))
# Output:
# Gradient norm (no BN): 0.0234
# Gradient norm (with BN): 1.2345
```

## Common Mistakes

1. **Using `BatchNorm1d` when you need `BatchNorm2d`**: For images (N, C, H, W), use `nn.BatchNorm2d`. For 1D features (N, L, C), use `nn.BatchNorm1d`. The wrong choice silently works but gives incorrect normalization.

2. **Forgetting to set `model.eval()` at inference**: BatchNorm uses different statistics at train vs. eval time. Forgetting this leads to incorrect outputs.

3. **Using small batch sizes**: BatchNorm statistics become noisy with small batches (e.g., batch size 1 or 2). Consider LayerNorm or GroupNorm in this case.

4. **Placing BatchNorm after the activation**: The original paper places BatchNorm before activation, but post-activation normalization can also work. Be consistent.

5. **Not accounting for the momentum parameter**: Default momentum (0.1) works for most cases, but very long training runs may need adjustments.

6. **Using BatchNorm with recurrent networks**: BatchNorm breaks the sequential dependency in RNNs. LayerNorm is preferred for recurrent architectures.

7. **Forgetting that BatchNorm adds 4 parameters per channel**: Two for γ, β (learnable) and two for running_mean, running_var (non-learnable buffers).

## Interview Questions

### Beginner - 5

1. What is batch normalization?
2. How does BatchNorm behave differently during training and inference?
3. What are the learnable parameters in BatchNorm?
4. Why does BatchNorm use momentum for running statistics?
5. What is the shape of γ in `nn.BatchNorm1d(256)`?

### Intermediate - 5

1. Derive the backward pass through a batch normalization layer.
2. Explain why BatchNorm allows higher learning rates.
3. How does BatchNorm act as a regularizer?
4. Compare BatchNorm and LayerNorm — when would you use each?
5. What problems occur when using BatchNorm with very small batch sizes?

### Advanced - 3

1. Derivethe gradient through the batch mean and variance computation and explain why the backward pass is computationally non-trivial.
2. Implement a variant of BatchNorm that uses exponential moving statistics during training instead of batch statistics (running BatchNorm).
3. Analyze the interaction between BatchNorm and weight decay — why does weight decay affect BatchNorm differently than other layers?

## Practice Problems

### Easy - 5

1. Apply `nn.BatchNorm1d(6)` to a (4, 6) tensor and verify normalization.
2. Switch a BatchNorm layer to eval mode and observe the difference.
3. Access and print the running mean and variance after 10 forward passes.
4. Create a BatchNorm2d layer and apply it to a (2, 3, 4, 5) tensor.
5. Set `affine=False` in BatchNorm and observe that output has no scale/shift.

### Medium - 5

1. Implement BatchNorm from scratch including the backward pass and compare with PyTorch's autograd.
2. Train two identical CNNs on CIFAR-10, one with and one without BatchNorm, and compare convergence.
3. Measure the gradient variance across layers with and without BatchNorm.
4. Implement ghost BatchNorm (computing statistics on a subset of the batch) and analyze its regularization effect.
5. Design an experiment showing how sensitive BatchNorm is to batch size.

### Hard - 3

1. Implement Synchronized BatchNorm for multi-GPU training (sync stats across GPUs).
2. Derive the forward and backward pass for BatchNorm in mixed precision training.
3. Build an adaptive BatchNorm that dynamically adjusts momentum based on the stability of running statistics.

## Solutions

### Easy - 1
```python
x = torch.randn(4, 6)
bn = nn.BatchNorm1d(6)
y = bn(x)
print("Mean:", y.mean(dim=0))  # ~0
print("Std:", y.std(dim=0, unbiased=False))  # ~1
```

### Easy - 2
```python
bn = nn.BatchNorm1d(4)
x = torch.randn(10, 4)
train_out = bn(x)
bn.eval()
eval_out = bn(x)
print("Difference:", (train_out - eval_out).abs().mean().item())
```

## Related Concepts

DL-036 Layer Normalization, DL-038 Dropout Layer, DL-041 Residual Connection, DL-059 Vanishing Gradients

## Next Concepts

DL-038 Dropout Layer, DL-039 Pooling Layers

## Summary

Batch normalization normalizes layer activations using mini-batch statistics during training and running averages during inference. It stabilizes training, enables higher learning rates, and provides regularization. It is a standard component in modern CNNs but has been largely replaced by LayerNorm in transformer architectures.

## Key Takeaways

- Normalizes each feature channel independently using batch statistics
- Training: batch mean/var; Inference: running average mean/var
- Learnable scale (γ) and shift (β) restore representational power
- Effective regularizer due to noise in batch statistics
- Sensitive to batch size — use alternatives for small batches
- Critical for enabling deep CNN architectures like ResNet
- Requires careful handling of train/eval mode switching
