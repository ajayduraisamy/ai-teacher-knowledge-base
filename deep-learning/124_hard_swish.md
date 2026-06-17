# Concept: Hard Swish

## Concept ID

DL-124

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Activation Functions

## Learning Objectives

- Understand the formulation of hard Swish as a piecewise linear approximation
- Implement hard Swish in PyTorch and compare with standard Swish
- Analyze the computational benefits for mobile and edge deployment
- Evaluate the accuracy trade-off in MobileNetV3 and efficient architectures
- Identify scenarios where hard Swish is appropriate

## Prerequisites

- Swish/SiLU activation (DL-119)
- Hard sigmoid (DL-123)
- Understanding of mobile deployment constraints
- Knowledge of ReLU6

## Definition

Hard Swish is a piecewise linear approximation of the Swish activation function, introduced in MobileNetV3 (Howard et al., 2019). It replaces the sigmoid gate in Swish with the hard sigmoid: hard_swish(x) = x * hard_sigmoid(x) = x * clamp((x + 3) / 6, 0, 1). This eliminates all exponential operations while preserving Swish's key properties: the non-monotonic bump in the negative region and the linear asymptote for positive values. Hard Swish was found to be nearly as accurate as Swish while being significantly faster on mobile hardware.

## Intuition

Hard Swish takes the "self-gating" idea of Swish and replaces the smooth sigmoid gate with a hard sigmoid gate. Instead of a smooth, curved gate, you get a straight-line ramp that goes from fully closed (0) to fully open (1) between x = -3 and x = 3. For x < -3, the gate is closed (0), and for x > 3, the gate is open (1). The result is a function that looks very similar to Swish but can be computed using only addition, multiplication, and clamping — no exponentials. This makes it ideal for mobile CPUs that lack dedicated exponential hardware.

## Why This Concept Matters

Hard Swish represents a key innovation in efficient neural architecture design. MobileNetV3 demonstrated that careful replacement of activation functions with piecewise linear approximations can maintain accuracy while dramatically reducing latency on mobile devices. This insight has influenced quantization techniques, hardware-specific optimizations, and the broader field of efficient deep learning. Understanding hard Swish is essential for practitioners deploying models on resource-constrained platforms, and it illustrates the general principle that activation functions can be optimized for specific hardware backends.

## Mathematical Explanation

Hard Swish is defined as:

hard_swish(x) = 0 if x <= -3
                x * (x + 3) / 6 if -3 < x < 3
                x if x >= 3

Or equivalently: x * clamp((x + 3) / 6, 0, 1)

The derivative is:
d/dx hard_swish(x) = 0 if x <= -3
                      (2x + 3) / 6 if -3 < x < 3
                      1 if x >= 3
                      undefined at x = -3 and x = 3

Properties:
- Output range: approximately [-0.375, infinity) with minimum at x = -1.5
- Non-monotonic (small negative dip)
- Piecewise quadratic (product of two linear functions)
- Not differentiable at x = -3 and x = 3
- Requires only arithmetic operations
- Equivalent to Swish for large |x|

## Code Examples

### Example 1: Basic Hard Swish

`python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.linspace(-5, 5, 11)

# PyTorch's built-in Hardswish
hard_swish = nn.Hardswish()
y_hard = hard_swish(x)

# Standard Swish for comparison
y_swish = x * torch.sigmoid(x)

print("Input:", x)
print("Hard Swish:", y_hard)
print("Standard Swish:", y_swish)
# Output:
# Input: tensor([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.])
# Hard Swish: tensor([-0.0000, -0.0000, -0.0000, -0.3333, -0.3333,  0.0000,  0.6667,  1.6667,  3.0000,  4.0000,  5.0000])
# Standard Swish: tensor([-0.0335, -0.0732, -0.1423, -0.2384, -0.2689,  0.0000,  0.7311,  1.7616,  2.8577,  3.9268,  4.9665])
`

### Example 2: Manual Implementation

`python
import torch

def hard_swish_manual(x):
    return x * torch.clamp((x + 3.0) / 6.0, 0.0, 1.0)

def hard_swish_relu6(x):
    return x * F.relu6(x + 3.0) / 6.0

x = torch.tensor([-4.0, -2.0, 0.0, 2.0, 4.0])
print("Manual:", hard_swish_manual(x))
print("ReLU6:", hard_swish_relu6(x))
print("Built-in:", nn.Hardswish()(x))
# Output:
# Manual: tensor([-0.0000, -0.3333,  0.0000,  1.6667,  4.0000])
# ReLU6: tensor([-0.0000, -0.3333,  0.0000,  1.6667,  4.0000])
# Built-in: tensor([-0.0000, -0.3333,  0.0000,  1.6667,  4.0000])
`

### Example 3: MobileNetV3-Style Block

`python
import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidualV3(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden_ch = in_ch * expand_ratio
        self.use_res_connect = stride == 1 and in_ch == out_ch

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_ch, hidden_ch, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_ch))
            layers.append(nn.Hardswish())

        layers.extend([
            nn.Conv2d(hidden_ch, hidden_ch, 3, stride, 1,
                     groups=hidden_ch, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.Hardswish(),
        ])

        layers.extend([
            nn.Conv2d(hidden_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

block = InvertedResidualV3(16, 32, 2, 4)
sample = torch.randn(1, 16, 28, 28)
output = block(sample)
print("Output shape:", output.shape)
# Output:
# Output shape: torch.Size([1, 32, 14, 14])
`

## Common Mistakes

1. **Using hard Swish during training when soft Swish is feasible**: Train with standard Swish for better gradient flow, then convert to hard Swish for inference.
2. **Not profiling performance on the target device**: Hard Swish benefits are hardware-specific. Always benchmark on the target platform.
3. **Assuming hard Swish matches Swish accuracy for all tasks**: The approximation introduces noise that may hurt performance on fine-grained tasks.
4. **Forgetting the negative dip**: Hard Swish has a minimum of -0.375 at x = -1.5. This negative value matters for downstream layers.
5. **Using hard Swish with very deep or wide networks**: The piecewise gradient can cause optimization issues in very large models.

## Interview Questions

### Beginner

1. What is the formula for hard Swish?
2. How does hard Swish differ from Swish?
3. What is the output range of hard Swish?
4. At what x does hard Swish reach its minimum?
5. Which architecture introduced hard Swish?

### Intermediate

1. Explain why hard Swish is more efficient than Swish on mobile CPUs.
2. Compare hard Swish with standard Swish in terms of gradient properties.
3. What is the advantage of using ReLU6 to implement hard Swish?
4. How does the negative bump of hard Swish differ from standard Swish?
5. When would you choose hard Swish over hard sigmoid?

### Advanced

1. Derive the derivative of hard Swish and identify its discontinuities.
2. Design a training scheme that progressively converts soft Swish to hard Swish.
3. Analyze the quantization properties of hard Swish compared to standard Swish.

## Practice Problems

### Easy

1. Compute hard_swish(-4), hard_swish(0), hard_swish(4).
2. What is the minimum value of hard Swish?
3. How many parameters does nn.Hardswish() have?
4. Is hard Swish monotonic?
5. What is hard_swish(-1.5)?

### Medium

1. Implement hard Swish using ReLU6 instead of clamp.
2. Profile the forward pass of hard Swish vs standard Swish on CPU.
3. Train a model on CIFAR-10 with hard Swish and Swish and compare accuracy.
4. Analyze the gradient sparsity of hard Swish compared to Swish.
5. Build a training pipeline that anneals from Swish to hard Swish.

### Hard

1. Implement a quantized hard Swish that uses only int8 arithmetic.
2. Prove that the maximum absolute error between hard Swish and Swish occurs at a specific x and find it.
3. Design an adaptive piecewise Swish with variable knot points and analyze its expressivity.

## Solutions

### Easy Solutions

1. hard_swish(-4) = 0, hard_swish(0) = 0, hard_swish(4) = 4
2. Minimum is -0.375 at x = -1.5
3. Zero — no learnable parameters
4. No, hard Swish is non-monotonic (has a negative dip)
5. hard_swish(-1.5) = -1.5 * clamp((1.5)/6, 0, 1) = -1.5 * 0.25 = -0.375

## Related Concepts

- Swish/SiLU (DL-119)
- Hard Sigmoid (DL-123)
- MobileNetV3 Architecture
- ReLU6

## Next Concepts

- Maxout Activation (DL-125)
- Piecewise Linear Activation (DL-126)
- Activation Function Comparison (DL-127)

## Summary

Hard Swish approximates Swish by replacing the sigmoid gate with hard sigmoid, enabling computation with only arithmetic operations. Introduced in MobileNetV3, it achieves nearly identical accuracy to Swish while being significantly faster on mobile hardware. Hard Swish is piecewise quadratic (for -3 < x < 3) and transitions to linear outside this range.

## Key Takeaways

- Hard Swish = x * clamp((x+3)/6, 0, 1)
- Piecewise approximation of Swish using hard sigmoid
- Non-monotonic with minimum -0.375 at x = -1.5
- Requires only arithmetic operations (no exponentials)
- Standard activation in MobileNetV3 and efficient architectures
- Best used with post-training conversion from Swish for training stability
- Hardware-dependent speedup — always benchmark on target platform
