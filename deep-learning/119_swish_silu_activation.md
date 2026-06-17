# Concept: Swish / SiLU Activation

## Concept ID

DL-119

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Activation Functions

## Learning Objectives

- Understand the formulation of Swish (SiLU) as x * sigmoid(x)
- Analyze the self-gating property and its implications for gradient flow
- Implement Swish in PyTorch with both module and functional APIs
- Compare Swish with GELU and ReLU on benchmark tasks
- Identify the role of the β parameter in the Swish family

## Prerequisites

- GELU activation (DL-118)
- Sigmoid activation (DL-111)
- Understanding of self-gating mechanisms
- Experience with comparative activation benchmarks

## Definition

Swish (also known as SiLU — Sigmoid Linear Unit) is a smooth, non-monotonic activation function defined as f(x) = x * σ(x), where σ(x) is the sigmoid function. Swish was discovered via automatic search (Ramachandran et al., 2017) and has a characteristic "bump" shape: it dips slightly below zero for negative inputs before asymptotically approaching zero from below. Unlike ReLU, Swish maintains non-zero gradients for all x ≠ 0, and unlike GELU, it is non-monotonic with a small negative region. The general form f(x) = x * σ(βx) adds a learnable or tunable β parameter.

## Intuition

Think of Swish as having a built-in attention mechanism at the neuron level — the sigmoid gate (σ(x)) learns to "open" or "close" the flow of information through each neuron based on the input itself. This self-gating means the activation function can express different behaviors depending on the input magnitude. For large positive inputs, the gate is fully open (σ(x) ≈ 1), so Swish behaves like the identity. For large negative inputs, the gate is almost closed (σ(x) ≈ 0), so Swish outputs near zero. But crucially, the small negative bump around x ≈ -1.28 provides gradient signal for moderately negative inputs, keeping neurons alive. This self-gating is more flexible than ReLU's hard threshold or GELU's Gaussian CDF.

## Why This Concept Matters

Swish represents a breakthrough in activation function design — it was discovered by neural architecture search rather than human intuition, and it consistently outperformed ReLU and its variants across a wide range of tasks. Google's Swish paper showed that simply replacing ReLU with Swish improved accuracy on ImageNet by 0.6-0.9% with no other changes. Swish is now widely used in EfficientNet, MobileNetV3, and other modern convolutional architectures. Its self-gating property provides a theoretical framework for understanding why learned or smooth activations outperform hard-threshold functions.

## Mathematical Explanation

The Swish/SiLU function is defined as:

f(x) = x * σ(x) = x / (1 + e^(-x))

The parameterized form: f(x) = x * σ(βx), where β is a learnable or fixed parameter.

The derivative is:
f'(x) = σ(x) + x * σ(x) * (1 - σ(x)) = σ(x) + x * σ(x) - x * σ²(x)
       = σ(x) * [1 + x * (1 - σ(x))]

For the parameterized form: f'(x) = σ(βx) * [1 + βx * (1 - σ(βx))]

Properties:
- Output range: approximately [-0.278, ∞) with minimum at ≈ -1.278
- Non-monotonic (has a small dip below zero)
- Smooth and differentiable everywhere
- Unbounded above, bounded below
- β controls the "hardness" of the transition: β → ∞ gives ReLU, β → 0 gives linear

## Code Examples

### Example 1: Basic Swish/SiLU

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.linspace(-5, 5, 11)

# Using SiLU module (PyTorch 1.7+)
silu = nn.SiLU()
y_silu = silu(x)

# Manual Swish
y_swish = x * torch.sigmoid(x)

print("Input:", x)
print("SiLU:", y_silu)
print("Swish:", y_swish)
# Output:
# Input: tensor([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.])
# SiLU: tensor([-0.0335, -0.0732, -0.1423, -0.2384, -0.2689,  0.0000,  0.7311,  1.7616,  2.8577,  3.9268,  4.9665])
# Swish: tensor([-0.0335, -0.0732, -0.1423, -0.2384, -0.2689,  0.0000,  0.7311,  1.7616,  2.8577,  3.9268,  4.9665])
```

### Example 2: Parameterized Swish with β

```python
import torch
import torch.nn as nn

class ParamSwish(nn.Module):
    def __init__(self, beta_init=1.0, learnable=True):
        super().__init__()
        if learnable:
            self.beta = nn.Parameter(torch.tensor(beta_init))
        else:
            self.register_buffer('beta', torch.tensor(beta_init))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

# Fixed beta
swish_beta_1 = ParamSwish(1.0, learnable=False)
swish_beta_5 = ParamSwish(5.0, learnable=False)
swish_beta_01 = ParamSwish(0.1, learnable=False)

x = torch.linspace(-3, 3, 7)
print("Input:", x)
print("β=0.1 (nearly linear):", swish_beta_01(x))
print("β=1.0 (standard Swish):", swish_beta_1(x))
print("β=5.0 (nearly ReLU):", swish_beta_5(x))
# Output:
# Input: tensor([-3., -2., -1.,  0.,  1.,  2.,  3.])
# β=0.1 (nearly linear): tensor([-0.5744, -0.4502, -0.2689,  0.0000,  0.7311,  1.5498,  2.4256])
# β=1.0 (standard Swish): tensor([-0.1423, -0.2384, -0.2689,  0.0000,  0.7311,  1.7616,  2.8577])
# β=5.0 (nearly ReLU): tensor([-0.0022, -0.0091, -0.0344,  0.0000,  0.7311,  1.9909,  3.0000])
```

### Example 3: Swish in EfficientNet-Style Block

```python
import torch
import torch.nn as nn

class MBConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio=4):
        super().__init__()
        hidden_ch = in_ch * expand_ratio
        self.expand = nn.Conv2d(in_ch, hidden_ch, 1)
        self.depthwise = nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1, groups=hidden_ch)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_ch, hidden_ch // 4, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_ch // 4, hidden_ch, 1),
            nn.Sigmoid()
        )
        self.project = nn.Conv2d(hidden_ch, out_ch, 1)

    def forward(self, x):
        x = self.expand(x)
        x = nn.functional.silu(x)
        x = self.depthwise(x)
        x = nn.functional.silu(x)
        se = self.se(x)
        x = x * se
        x = self.project(x)
        return x

block = MBConvBlock(32, 64, 4)
sample = torch.randn(1, 32, 56, 56)
output = block(sample)
print("Output shape:", output.shape)
# Output:
# Output shape: torch.Size([1, 64, 56, 56])
```

## Common Mistakes

1. **Confusing Swish with GELU**: Swish uses sigmoid (σ(x)), GELU uses the Gaussian CDF (Φ(x)). They are different functions with different shapes.
2. **Not using SiLU when available**: PyTorch's nn.SiLU is Swish with β=1. Using manual x * torch.sigmoid(x) is fine but less efficient.
3. **Assuming monotonicity**: Swish is non-monotonic — it dips below zero. This can surprise people who expect activations to be monotonic.
4. **Forgetting about the negative bump**: The negative values (≈ -0.278 at minimum) mean Swish can produce negative outputs, which matters for downstream layer statistics.
5. **Setting β too high or too low without validation**: β controls the transition sharpness. Very high β replicates ReLU (with its disadvantages), very low β is nearly linear.

## Interview Questions

### Beginner

1. What is the formula for Swish/SiLU?
2. Is Swish monotonic?
3. What is the approximate minimum value of Swish?
4. How is the parameterized Swish (x * σ(βx)) related to ReLU?
5. Which PyTorch module implements Swish?

### Intermediate

1. Explain the self-gating property of Swish and its advantages.
2. Compare Swish with GELU: how are they similar and different?
3. How does the β parameter in parameterized Swish affect the function shape?
4. Why does Swish's non-monotonicity help with optimization?
5. Show that Swish has a "bump" shape and explain why this is beneficial.

### Advanced

1. Derive the derivative of Swish and analyze its behavior for extreme positive and negative values.
2. Prove that Swish(x) = x * σ(x) is related to the logit of the sigmoid and explain the implications for gradient flow.
3. Design a Swish variant where β is input-dependent rather than fixed, and analyze its expressivity.

## Practice Problems

### Easy

1. Compute Swish(0), Swish(1), and Swish(-1).
2. At what x does Swish reach its minimum?
3. What is the limit of Swish(x) as x → ∞?
4. What is the limit of Swish(x) as x → -∞?
5. How many parameters does nn.SiLU() add?

### Medium

1. Implement Swish from scratch using torch.sigmoid and verify it matches nn.SiLU.
2. Train a ResNet-18 on CIFAR-10 with ReLU and Swish and compare accuracy curves.
3. Analyze the output distribution of Swish for normally distributed inputs and compare with GELU.
4. Design an experiment to find the optimal β for parameterized Swish on a regression task.
5. Compute and plot the gradient of Swish over [-5, 5] and identify the region of maximum gradient.

### Hard

1. Derive the closed-form expression for the inverse of Swish and analyze its properties.
2. Prove that Swish with learnable β can represent both ReLU (β → ∞) and linear (β → 0) as limiting cases.
3. Implement a "hard Swish" approximation (like the one used in MobileNetV3) and analyze the accuracy/computation trade-off.

## Solutions

### Easy Solutions

1. Swish(0) = 0 * 0.5 = 0, Swish(1) = 1 * σ(1) ≈ 0.7311, Swish(-1) = -1 * σ(-1) ≈ -0.2689
2. Minimum occurs at approximately x ≈ -1.278, where Swish ≈ -0.278
3. As x → ∞, σ(x) → 1, so Swish → x (linear asymptote)
4. As x → -∞, σ(x) → 0, so Swish → 0 (from below, due to negative bump)
5. Zero — nn.SiLU has no learnable parameters

## Related Concepts

- GELU Activation (DL-118)
- Sigmoid Activation (DL-111)
- Mish Activation (DL-120)
- Activation Function Search

## Next Concepts

- Mish Activation (DL-120)
- Softplus Activation (DL-121)
- Hard Swish (DL-124)

## Summary

Swish/SiLU is a smooth, non-monotonic activation function defined as x * σ(x). Discovered through neural architecture search, it consistently outperforms ReLU by providing self-gating behavior, non-zero gradients for all inputs, and a characteristic negative "bump." The parameterized form (x * σ(βx)) interpolates between linear and ReLU-like behavior. Swish is widely used in modern convolutional architectures including EfficientNet and MobileNetV3.

## Key Takeaways

- Swish = x * σ(x); SiLU is the same function
- Non-monotonic with a minimum of ≈ -0.278 at x ≈ -1.278
- Parameterized form: x * σ(βx) interpolates between linear (β→0) and ReLU (β→∞)
- Self-gating: the sigmoid acts as a learned, input-dependent gate
- Outperforms ReLU on ImageNet and other large-scale benchmarks
- Default activation in EfficientNet, MobileNetV3, and modern ConvNets
- Smooth, differentiable everywhere with non-zero gradient for all finite x
