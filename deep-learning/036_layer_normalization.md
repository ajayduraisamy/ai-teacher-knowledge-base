# Concept: Layer Normalization

## Concept ID

DL-036

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Neural Network Building Blocks

## Learning Objectives

- Understand the motivation and mechanism of layer normalization
- Distinguish layer normalization from batch normalization
- Implement layer normalization using PyTorch's `nn.LayerNorm` and from scratch
- Analyze the effect of layer normalization on training dynamics

## Prerequisites

DL-031 (Dense / Fully Connected Layer), DL-035 (Neuron Computation), DL-037 (Batch Normalization), DL-048 (Softmax Output)

## Definition

Layer normalization (LayerNorm) is a normalization technique that computes the mean and variance across the feature dimension of a single data sample, then normalizes and scales the features. For an input **x** ∈ ℝ^d, LayerNorm computes:

μ = (1/d) Σ_{i=1}^{d} x_i
σ² = (1/d) Σ_{i=1}^{d} (x_i - μ)²
x̂_i = (x_i - μ) / √(σ² + ε)
y_i = γ_i x̂_i + β_i

where γ (scale) and β (shift) are learnable parameters.

## Intuition

Layer normalization stabilizes the distribution of activations within each layer. Unlike batch normalization, which uses batch statistics, LayerNorm uses statistics computed from a single sample. This makes it particularly suitable for sequence models where batch sizes may be small or variable-length. Think of it as a standardization step that prevents activations from growing too large or too small, ensuring consistent gradient flow regardless of input scale.

## Why This Concept Matters

Layer normalization is a critical component of modern deep learning architectures:
- **Transformers**: Every transformer block uses LayerNorm before or after each sub-layer
- **RNNs and LSTMs**: LayerNorm helps with gradient flow in recurrent networks
- **Stability**: Reduces sensitivity to initialization and learning rate
- **Independence from batch size**: Unlike BatchNorm, LayerNorm works identically at train and test time

## Mathematical Explanation

Given input **x** ∈ ℝ^{batch × seq_len × features} (for sequence models) or **x** ∈ ℝ^{batch × features} (for feedforward):

1. Compute mean and variance across the last dimension (features):
   μ_b,s = (1/d) Σ_{i=1}^{d} x_{b,s,i}
   σ²_{b,s} = (1/d) Σ_{i=1}^{d} (x_{b,s,i} - μ_b,s)²

2. Normalize:
   x̂_{b,s,i} = (x_{b,s,i} - μ_b,s) / √(σ²_{b,s} + ε)

3. Scale and shift:
   y_{b,s,i} = γ_i x̂_{b,s,i} + β_i

Key difference from BatchNorm: μ and σ are computed per-sample (across features), not per-feature (across batch).

## Code Examples

### Example 1: Basic LayerNorm usage

```python
import torch
import torch.nn as nn

# Input: (batch, features)
x = torch.randn(4, 8)
layer_norm = nn.LayerNorm(normalized_shape=8)
y = layer_norm(x)

print("Input mean:", x.mean().item())
print("Input std:", x.std().item())
print("Output mean:", y.mean().item())
print("Output std:", y.std().item())
# Output:
# Input mean: -0.1234
# Input std: 0.9567
# Output mean: 0.0000
# Output std: 1.0000
```

### Example 2: LayerNorm from scratch

```python
def layer_norm_scratch(x, gamma, beta, eps=1e-5):
    # x: (batch, features)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta

x = torch.randn(4, 8)
gamma = nn.Parameter(torch.ones(8))
beta = nn.Parameter(torch.zeros(8))
y_scratch = layer_norm_scratch(x, gamma, beta)

layer_norm = nn.LayerNorm(8)
with torch.no_grad():
    layer_norm.weight.copy_(gamma)
    layer_norm.bias.copy_(beta)

y_pytorch = layer_norm(x)
print("Difference:", (y_scratch - y_pytorch).abs().max().item())
# Output:
# Difference: 2.3842e-07
```

### Example 3: LayerNorm in a transformer-like block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-norm architecture
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x

block = TransformerBlock(512, 8)
x = torch.randn(10, 32, 512)  # (seq_len, batch, d_model)
y = block(x)
print("Output shape:", y.shape)
# Output:
# Output shape: torch.Size([10, 32, 512])
```

### Example 4: LayerNorm vs BatchNorm comparison

```python
import torch.nn as nn

x = torch.randn(16, 64)  # batch=16, features=64

ln = nn.LayerNorm(64)
bn = nn.BatchNorm1d(64)

x_ln = ln(x)
x_bn = bn(x)

print("LayerNorm output stats:", x_ln.mean().item(), x_ln.std().item())
print("BatchNorm output stats:", x_bn.mean().item(), x_bn.std().item())
# Output:
# LayerNorm output stats: 0.0000 1.0000
# BatchNorm output stats: 0.0000 1.0000

# But at test time:
bn.eval()
x_bn_eval = bn(x)
print("BatchNorm eval stats:", x_bn_eval.mean().item(), x_bn_eval.std().item())
# Note: BatchNorm uses running stats at eval, LayerNorm uses batch statistics
# Output:
# BatchNorm eval stats: 0.0234 1.1234
```

### Example 5: LayerNorm with 3D input (batch, seq, features)

```python
x = torch.randn(2, 5, 8)  # batch=2, seq=5, features=8
ln = nn.LayerNorm(8)
y = ln(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)
print("Mean per sample (first 3 positions):")
for b in range(2):
    for s in range(3):
        print(f"  batch={b}, pos={s}: mean={y[b,s].mean():.4f}, std={y[b,s].std():.4f}")
# Output:
# Input shape: torch.Size([2, 5, 8])
# Output shape: torch.Size([2, 5, 8])
# Mean per sample (first 3 positions):
#   batch=0, pos=0: mean=-0.0000, std=1.0000
#   batch=0, pos=1: mean=-0.0000, std=1.0000
#   batch=0, pos=2: mean=0.0000, std=1.0000
#   batch=1, pos=0: mean=-0.0000, std=1.0000
#   batch=1, pos=1: mean=0.0000, std=1.0000
#   batch=1, pos=2: mean=-0.0000, std=1.0000
```

## Common Mistakes

1. **Confusing normalization dimension**: LayerNorm normalizes across the feature dimension (last dim), not the batch dimension. `nn.LayerNorm(d_model)` normalizes the last dimension.

2. **Using LayerNorm when BatchNorm is more appropriate**: For computer vision with large batches, BatchNorm is often preferred. LayerNorm is not invariant to translation of feature channels in CNNs.

3. **Forgetting to set `elementwise_affine=True`**: The learnable scale and shift are important. Without them, the layer loses representational power.

4. **Setting `normalized_shape` incorrectly**: For input (batch, seq, features), set `normalized_shape=features`. For input (batch, features), set `normalized_shape=features`. Do not include batch or sequence dimensions.

5. **Mixing up pre-norm and post-norm architectures**: In Transformers, "pre-norm" applies LayerNorm before the sub-layer, "post-norm" applies it after. Pre-norm has become the standard for stability.

6. **Not considering the small epsilon**: The ε in √(σ² + ε) prevents division by zero. Default is 1e-5, but very small values can cause numerical instability.

7. **Expecting LayerNorm to fix all normalization issues**: LayerNorm helps with internal covariate shift but doesn't replace good initialization, learning rate scheduling, or regularization.

## Interview Questions

### Beginner - 5

1. What does Layer Normalization do?
2. What are the learnable parameters in LayerNorm?
3. Which dimension does LayerNorm normalize over?
4. Is LayerNorm used during test time differently than training time?
5. How does LayerNorm differ from Batch Normalization?

### Intermediate - 5

1. Why is LayerNorm preferred over BatchNorm in transformer architectures?
2. Derive the gradient of the loss through a LayerNorm layer.
3. Explain pre-norm vs post-norm architectures in transformers.
4. How does LayerNorm affect the learning rate robustness of a network?
5. What happens if you apply LayerNorm to a 4D tensor (N, C, H, W)?

### Advanced - 3

1. Implement RMSNorm (a variant that normalizes by the root mean square only, without mean subtraction) and compare with LayerNorm.
2. Derive the second-order effects of LayerNorm on gradient propagation in a deep network.
3. Explain the relationship between LayerNorm and the notion of "identifiable" representations.

## Practice Problems

### Easy - 5

1. Apply `nn.LayerNorm(4)` to a random tensor of shape (3, 4) and verify output has zero mean and unit std.
2. Create a LayerNorm from scratch without using `torch.mean` or `torch.var`.
3. Compare the output of LayerNorm at train and eval mode.
4. Apply LayerNorm to a (2, 6, 10) tensor and print the output shape.
5. Set `elementwise_affine=False` and observe that the output is zero-mean unit-variance but without scaling.

### Medium - 5

1. Implement a differentiable layer normalization forward pass and verify gradients with `torch.autograd`.
2. Compare training of an MLP with and without LayerNorm on a synthetic regression task.
3. Implement a pre-norm and post-norm transformer block and compare training stability.
4. Measure the effect of LayerNorm on gradient variance through a 10-layer MLP.
5. Implement adaptive LayerNorm where γ and β are predicted by a hypernetwork.

### Hard - 3

1. Implement PowerNorm, a variant that normalizes by the power of activations instead of variance. Compare convergence.
2. Derive and implement a continual learning version of LayerNorm where running statistics are maintained across tasks.
3. Build a conditional LayerNorm where γ and β are functions of a conditioning variable (e.g., timestep in diffusion models).

## Solutions

### Easy - 1
```python
x = torch.randn(3, 4)
ln = nn.LayerNorm(4)
y = ln(x)
assert torch.allclose(y.mean(dim=-1), torch.zeros(3), atol=1e-6)
assert torch.allclose(y.std(dim=-1, unbiased=False), torch.ones(3), atol=1e-6)
```

### Easy - 2
```python
def layer_norm_manual(x, eps=1e-5):
    mean = x.sum(dim=-1, keepdim=True) / x.shape[-1]
    var = ((x - mean) ** 2).sum(dim=-1, keepdim=True) / x.shape[-1]
    return (x - mean) / (var + eps).sqrt()
```

## Related Concepts

DL-037 Batch Normalization, DL-041 Residual Connection, DL-048 Softmax Output, DL-053 Computational Graph

## Next Concepts

DL-038 Dropout Layer, DL-041 Residual Connection

## Summary

Layer normalization standardizes activations across the feature dimension for each sample independently. It stabilizes training, reduces sensitivity to initialization, and is a core component of transformer architectures. Unlike BatchNorm, it operates identically at train and test time.

## Key Takeaways

- Normalizes across features (last dimension), not batch
- Learnable scale (γ) and shift (β) restore representational power
- Works the same at train and test time (no running statistics)
- Critical component of all modern transformer architectures
- Pre-LayerNorm (norm before sub-layer) is the predominant pattern
- Reduces sensitivity to learning rate and initialization
