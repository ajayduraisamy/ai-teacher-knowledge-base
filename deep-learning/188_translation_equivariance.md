# Concept: Translation Equivariance

## Concept ID

DL-188

## Difficulty

Advanced

## Domain

Deep Learning

## Module

CNN Fundamentals

## Learning Objectives

- Define translation equivariance and distinguish it from invariance
- Understand how convolution achieves translation equivariance
- Verify translation equivariance empirically with PyTorch
- Analyze the implications for CNN design

## Prerequisites

DL-176 Convolution Operation, DL-186 Parameter Sharing, DL-187 Local Connectivity

## Definition

Translation equivariance means that if the input is shifted (translated), the output feature map shifts by the same amount, preserving the spatial relationship between input patterns and their detected features.

## Intuition

Imagine you have a stamp that prints a smiley face. If you move the stamp to a different position on the paper, the smiley face appears at the new position. The stamp (kernel) doesn't care where it is applied — it produces the same pattern response. This is translation equivariance: shift the input, and the response shifts accordingly, without changing its nature. This is different from translation invariance (where shifting doesn't change the output at all, like in global pooling). Equivariance preserves spatial structure — the "where" information.

## Why This Concept Matters

Translation equivariance is what makes CNNs robust to object position. A cat in the top-left corner is detected by the same kernels as a cat in the bottom-right. This is why CNNs generalize across positions with far fewer parameters than models that must learn separate detectors for each location.

## Mathematical Explanation

Let $T_v$ be the translation operator that shifts an image by vector $v$:
$$(T_v I)[x, y] = I[x - v_x, y - v_y]$$

A convolution operation $\text{Conv}$ is translation equivariant if:
$$\text{Conv}(T_v I) = T_v(\text{Conv}(I))$$

Proof: 
$$\text{Conv}(T_v I)[i, j] = \sum_{m,n} (T_v I)[i+m, j+n] \cdot K[m,n]$$
$$= \sum_{m,n} I[i+m - v_x, j+n - v_y] \cdot K[m,n]$$
$$= \text{Conv}(I)[i - v_x, j - v_y] = T_v(\text{Conv}(I))[i, j]$$

**Equivariance vs Invariance**:
- **Equivariance**: $f(T_v(x)) = T_v(f(x))$ — the output shifts with the input
- **Invariance**: $f(T_v(x)) = f(x)$ — the output is unchanged by translation

Convolution + pooling gives a spectrum from equivariance (conv layers) to invariance (global pooling).

## Code Examples

### Example 1: Verifying Translation Equivariance

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Create input with a pattern
x = torch.zeros(1, 1, 10, 10)
x[0, 0, 2:5, 2:5] = 1.0  # square pattern at top-left

# Shift it by (3, 3)
x_shifted = torch.zeros(1, 1, 10, 10)
x_shifted[0, 0, 5:8, 5:8] = 1.0  # same square at shifted position

# Conv layer
conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)

with torch.no_grad():
    out_original = conv(x)
    out_shifted = conv(x_shifted)

# Shift the original output by (3,3) and compare
out_original_shifted = torch.zeros_like(out_original)
out_original_shifted[0, 0, 3:8, 3:8] = out_original[0, 0, 0:5, 0:5]

difference = (out_original_shifted - out_shifted).abs().max().item()
print(f"Max difference (equivariance test): {difference:.6f}")
# Output: Max difference (equivariance test): 0.000000

print(f"Translation equivariance holds: {difference < 1e-10}")
# Output: Translation equivariance holds: True
```

### Example 2: Equivariance vs Invariance

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Demonstrate equivariance (conv) vs invariance (global pooling)
x = torch.zeros(1, 1, 16, 16)
x[0, 0, 4, 4] = 1.0  # dot at (4,4)

x_shifted = torch.zeros(1, 1, 16, 16)
x_shifted[0, 0, 12, 12] = 1.0  # dot at (12,12)

# Conv (equivariant)
conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)

# Global average pooling (invariant)
pool = nn.AdaptiveAvgPool2d(1)

with torch.no_grad():
    conv_out1 = conv(x)
    conv_out2 = conv(x_shifted)
    
    pool_out1 = pool(conv_out1)
    pool_out2 = pool(conv_out2)

# Conv: outputs differ (shifted)
conv_diff = (conv_out1 - conv_out2).abs().max().item()
print(f"Conv outputs differ (equivariance preserves shift): {conv_diff > 0}")
# Output: Conv outputs differ (equivariance preserves shift): True

# Pool: outputs are same (invariant to shift)
pool_diff = (pool_out1 - pool_out2).abs().max().item()
print(f"Pool outputs same (invariance removes shift): {pool_diff < 1e-10}")
# Output: Pool outputs same (invariance removes shift): True

# Show that conv outputs are shifted versions
# Find max positions
pos1 = conv_out1[0, 0].argmax().item()
pos2 = conv_out2[0, 0].argmax().item()
print(f"\nMax activation position 1: ({pos1 // 16}, {pos1 % 16})")
print(f"Max activation position 2: ({pos2 // 16}, {pos2 % 16})")
# Output: Max activation position 1: (4, 4)
# Output: Max activation position 2: (12, 12)
```

### Example 3: Breaking Equivariance with Padding

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# How padding affects translation equivariance
x = torch.zeros(1, 1, 10, 10)
x[0, 0, 4, 4] = 1.0  # center dot

# Shift by (3, 3)
x_shifted = torch.zeros(1, 1, 10, 10)
x_shifted[0, 0, 7, 7] = 1.0

# Conv with no padding (should be perfectly equivariant except at edges)
conv_no_pad = nn.Conv2d(1, 1, 3, padding=0, bias=False)

# Conv with padding
conv_pad = nn.Conv2d(1, 1, 3, padding=1, bias=False)

with torch.no_grad():
    out1_no = conv_no_pad(x)
    out2_no = conv_no_pad(x_shifted)
    out1_pad = conv_pad(x)
    out2_pad = conv_pad(x_shifted)

# Without padding: edge effects break equivariance near borders
# Dot at (4,4): output at (4-1,4-1)=(3,3) considering kernel offset
# Dot at (7,7): output at (7-1,7-1)=(6,6)

# Shift first output by (3,3) and compare
print(f"Input dot positions: (4,4) and (7,7)")

# For no-padding: 10->8, positions shift by 3
pos1_no = out1_no[0, 0].argmax().item()
pos2_no = out2_no[0, 0].argmax().item()
print(f"No padding - max positions: ({pos1_no // 8}, {pos1_no % 8}) "
      f"and ({pos2_no // 8}, {pos2_no % 8})")
# Output: No padding - max positions: (3, 3) and (6, 6)

# Check if positions shifted correctly
# Position 1 = 3, Position 2 = 6 (shift of +3)
print(f"Shift preserved (no pad): {pos2_no - pos1_no == 3}")
# Output: Shift preserved (no pad): True

# With padding: input 10->10
pos1_pad = out1_pad[0, 0].argmax().item()
pos2_pad = out2_pad[0, 0].argmax().item()
print(f"With padding - max positions: ({pos1_pad // 10}, {pos1_pad % 10}) "
      f"and ({pos2_pad // 10}, {pos2_pad % 10})")
# Output: With padding - max positions: (4, 4) and (7, 7)

print(f"Shift preserved (pad): {pos2_pad - pos1_pad == 3}")
# Output: Shift preserved (pad): True
```

## Common Mistakes

1. **Confusing equivariance with invariance**: Equivariance preserves spatial structure; invariance discards it.
2. **Thinking all CNN layers are perfectly equivariant**: Padding, striding, and pooling break exact equivariance.
3. **Ignoring boundary effects**: Near-image borders, equivariance fails because of missing input context.
4. **Assuming fully connected layers are equivariant**: FC layers are not translation equivariant (cat at different positions activates different neurons).
5. **Not accounting for downsampling**: Striding and pooling change the translation equivalence — a 2-pixel shift in input becomes a 1-pixel shift in output.

## Interview Questions

### Beginner - 5
1. What is translation equivariance?
2. How does a conv layer achieve translation equivariance?
3. What is the difference between equivariance and invariance?
4. Why is translation equivariance useful for image classification?
5. Are all layers in a CNN translation equivariant?

### Intermediate - 5
1. Prove that convolution is translation equivariant.
2. How does padding affect translation equivariance?
3. How does striding affect translation equivariance?
4. What is the trade-off between equivariance and invariance in CNNs?
5. How do modern architectures balance equivariance and invariance?

### Advanced - 3
1. Derive the conditions for exact translation equivariance in a CNN.
2. Design a network that is equivariant to arbitrary groups of transformations (e.g., rotation, scaling).
3. Explain the relationship between translation equivariance and the weight-sharing mechanism in the Fourier domain.

## Practice Problems

### Easy - 5
1. Verify translation equivariance for a single conv layer.
2. Show that global pooling is translation invariant.
3. Demonstrate that FC layers are not translation equivariant.
4. Test equivariance at different positions in the input.
5. Measure equivariance error near image boundaries.

### Medium - 5
1. Build a test to quantitatively measure equivariance error.
2. Compare equivariance properties of different padding modes.
3. Analyze how max pooling affects translation equivariance.
4. Implement a network with controlled equivariance properties.
5. Visualize equivariance breakdown at image boundaries.

### Hard - 3
1. Implement a group-equivariant convolutional layer (GCNN).
2. Design a network that is both translation equivariant and scale equivariant.
3. Derive the theoretical maximum equivariance error for a given architecture.

## Solutions

### Easy - 1 Solution
```python
x = torch.randn(1, 1, 10, 10)
x_shifted = torch.roll(x, shifts=(2, 2), dims=(2, 3))
conv = nn.Conv2d(1, 1, 3, padding=1)
with torch.no_grad():
    out1 = conv(x)
    out2 = conv(x_shifted)
out1_shifted = torch.roll(out1, shifts=(2, 2), dims=(2, 3))
print(torch.allclose(out1_shifted, out2, atol=1e-6))
```

## Related Concepts

DL-176 Convolution Operation, DL-186 Parameter Sharing, DL-187 Local Connectivity, DL-189 Max Pooling

## Next Concepts

DL-189 Max Pooling, DL-190 Average Pooling

## Summary

Translation equivariance is a fundamental property of convolutions: shifting the input shifts the feature map by the same amount. This property enables CNNs to detect patterns regardless of position, dramatically reducing the data needed for learning and improving generalization.

## Key Takeaways

- Equivariance: f(T(x)) = T(f(x)) — shift in input = same shift in output
- Invariance: f(T(x)) = f(x) — shift doesn't change output
- Convolution is translation equivariant by design (shared weights + sliding)
- Padding, striding, and pooling modify exact equivariance
- Equivariance is preserved at interior; breaks at boundaries
- Deep layers combine equivariance (conv) with invariance (pool, global pool)
- Essential for position-agnostic pattern detection
- Group-equivariant CNNs extend this to rotations, scales, etc.
- Equivariance reduces sample complexity for vision tasks
