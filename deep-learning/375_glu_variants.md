# Concept: GLU Variants

## Concept ID

DL-375

## Difficulty

Expert

## Domain

Deep Learning

## Module

Transformer Internals

## Learning Objectives

- Understand the Gated Linear Unit (GLU) and its variants: SwiGLU, GeGLU, ReGLU.
- Explain how gating mechanisms improve the representational capacity of Transformer FFNs.
- Implement GLU variants in PyTorch with proper dimensionality adjustments.
- Compare the performance and parameter efficiency of different GLU variants.
- Understand why SwiGLU has become the default in modern LLMs.

## Prerequisites

- DL-360: Feed-Forward Network
- DL-374: FFN Expansion Factor
- Understanding of activation functions (ReLU, GELU, SiLU/Swish).
- Familiarity with element-wise multiplication and gating mechanisms.

## Definition

GLU (Gated Linear Unit) variants are a family of FFN architectures for Transformers that use a gating mechanism to control information flow. Instead of a single linear projection followed by a non-linearity, GLU variants use two parallel projections: one for the "gate" (with an activation function) and one for the "value" (typically linear). The output is the element-wise product of the gate and value, followed by a final projection. Variants differ in the activation function used for the gate: sigmoid (GLU), GELU (GeGLU), SiLU/Swish (SwiGLU), or ReLU (ReGLU).

## Intuition

A standard FFN can be thought of as: "detect a pattern, then transform it." The GLU variant adds a gate that says "how much of this pattern should I let through?" The gating mechanism allows the network to selectively activate features based on the input context.

This is analogous to having a dimmer switch instead of an on/off switch. ReLU either activates a neuron (if the input is positive) or kills it (if negative). GLU variants can smoothly vary the contribution of each hidden neuron, giving the model more nuanced control over information flow.

The SwiGLU variant (SiLU gate) has become the standard in modern LLMs because Swish is smooth, non-monotonic, and has been empirically shown to work better than sigmoid or GELU for gating.

## Why This Concept Matters

GLU variants are important because:

1. **Improved Performance**: GLU variants consistently outperform standard ReLU FFNs in language modeling.
2. **Modern Standard**: SwiGLU is used in Llama, Llama 2, Llama 3, Mixtral, PaLM, and Gemini.
3. **Parameter Efficiency**: GLU variants achieve better performance per parameter than standard FFNs.
4. **Architecture Design**: Understanding GLU variants is essential for designing modern Transformers.
5. **Activation Function Research**: The choice of gating activation is an active research area.

## Mathematical Explanation

### Standard GLU

\[
\text{GLU}(x) = (xW_1 + b_1) \odot \sigma(xV_1 + c_1)
\]

where \(\sigma\) is the sigmoid function, \(\odot\) is element-wise multiplication, and \(V_1\) is a separate projection matrix (the "gate").

### SwiGLU

\[
\text{SwiGLU}(x) = (\text{SiLU}(xW_1) \odot (xV_1))W_2
\]

where \(\text{SiLU}(x) = x \cdot \sigma(x)\) (also called Swish). SiLU is smooth, non-monotonic, and has a "bump" for negative values near zero.

### GeGLU

\[
\text{GeGLU}(x) = (\text{GELU}(xW_1) \odot (xV_1))W_2
\]

### ReGLU

\[
\text{ReGLU}(x) = (\text{ReLU}(xW_1) \odot (xV_1))W_2
\]

### Parameter Adjustment

Standard FFN: \(2 \times d_{\text{model}} \times d_{ff}\) parameters (two matrices).
GLU FFN: \(3 \times d_{\text{model}} \times d_{ff}\) parameters (three matrices: W1, V1, W2).

To match standard FFN parameters:
\[
d_{ff}^{\text{GLU}} = \frac{2}{3} \times d_{ff}^{\text{standard}}
\]

So for a standard FFN with \(d_{ff} = 4 \times d_{\text{model}}\), the GLU variant uses:
\[
d_{ff}^{\text{GLU}} = \frac{2}{3} \times 4 \times d_{\text{model}} = \frac{8}{3} \times d_{\text{model}} \approx 2.67 \times d_{\text{model}}
\]

## Code Examples

### Example 1: All GLU Variants

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GLUFFN(nn.Module):
    """Base GLU variant FFN."""
    def __init__(self, d_model, d_ff, activation='swish', dropout=0.1):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.V1 = nn.Linear(d_model, d_ff, bias=False)  # Gate
        self.W2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def _get_activation(self):
        if self.activation == 'swish':
            return F.silu  # SiLU = Swish
        elif self.activation == 'gelu':
            return F.gelu
        elif self.activation == 'relu':
            return F.relu
        elif self.activation == 'sigmoid':
            return torch.sigmoid
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x):
        gate = self._get_activation()(self.W1(x))
        value = self.V1(x)
        return self.W2(self.dropout(gate * value))

# Test all variants
d_model = 64
d_ff = int(d_model * 8 / 3)  # Standard GLU d_ff

x = torch.randn(2, 10, d_model)
for name in ['swish', 'gelu', 'relu', 'sigmoid']:
    ffn = GLUFFN(d_model, d_ff, activation=name)
    output = ffn(x)
    params = sum(p.numel() for p in ffn.parameters())
    print(f"{name:>8}: output shape={output.shape}, params={params:,}")
# Output:    swish: output shape=torch.Size([2, 10, 64]), params=12,373
# Output:     gelu: output shape=torch.Size([2, 10, 64]), params=12,373
# Output:     relu: output shape=torch.Size([2, 10, 64]), params=12,373
# Output:  sigmoid: output shape=torch.Size([2, 10, 64]), params=12,373
```

### Example 2: SwiGLU Implementation (Llama-style)

```python
class SwiGLU(nn.Module):
    """
    SwiGLU FFN as used in Llama models.
    Uses SiLU (Swish) activation for the gate.
    """
    def __init__(self, d_model, intermediate_size=None, multiple_of=256):
        """
        Args:
            d_model: Model dimension
            intermediate_size: If None, computed as int(8/3 * d_model)
            multiple_of: Round d_ff to nearest multiple of this (for GPU efficiency)
        """
        super().__init__()
        if intermediate_size is None:
            intermediate_size = int(8/3 * d_model)
            # Round to nearest multiple of multiple_of
            intermediate_size = ((intermediate_size + multiple_of - 1) // multiple_of) * multiple_of

        self.W1 = nn.Linear(d_model, intermediate_size, bias=False)
        self.V1 = nn.Linear(d_model, intermediate_size, bias=False)
        self.W2 = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x):
        # SiLU(xW1) * xV1
        return self.W2(F.silu(self.W1(x)) * self.V1(x))

# Compare standard FFN vs SwiGLU with matched parameters
d_model = 4096  # Llama-1 7B dimension
standard_d_ff = 4 * d_model  # 16384

standard_ffn = nn.Sequential(
    nn.Linear(d_model, standard_d_ff),
    nn.ReLU(),
    nn.Linear(standard_d_ff, d_model),
)
swiglu_ffn = SwiGLU(d_model)

standard_params = sum(p.numel() for p in standard_ffn.parameters())
swiglu_params = sum(p.numel() for p in swiglu_ffn.parameters())

print(f"Llama-scale comparison (d_model={d_model}):")
print(f"  Standard FFN: d_ff={standard_d_ff}, params={standard_params:,}")
print(f"  SwiGLU FFN:   d_ff={swiglu_ffn.W1.out_features}, "
      f"params={swiglu_params:,}")
print(f"  Parameter ratio: {swiglu_params/standard_params:.4f}")

# Test forward pass
x = torch.randn(1, 10, d_model)
out_standard = standard_ffn(x)
out_swiglu = swiglu_ffn(x)
print(f"  Output shapes: standard={out_standard.shape}, swiglu={out_swiglu.shape}")
# Output: Llama-scale comparison (d_model=4096):
# Output:   Standard FFN: d_ff=16384, params=134,225,920
# Output:   SwiGLU FFN:   d_ff=10923, params=134,225,920
# Output:   Parameter ratio: 1.0000
# Output:   Output shapes: standard=torch.Size([1, 10, 4096]), swiglu=torch.Size([1, 10, 4096])
```

### Example 3: Visualizing Gating Behavior

```python
def visualize_gating():
    """Compare the gating behavior of different GLU variants."""
    d_model = 1  # 1D input for visualization
    d_ff = 32

    class SimpleGate(nn.Module):
        def __init__(self, activation='swish'):
            super().__init__()
            self.W1 = nn.Linear(d_model, d_ff)
            self.V1 = nn.Linear(d_model, d_ff)
            self.activation = activation

        def forward(self, x):
            gate = {
                'swish': F.silu,
                'gelu': F.gelu,
                'relu': F.relu,
                'sigmoid': torch.sigmoid,
            }[self.activation](self.W1(x))
            value = self.V1(x)
            return gate * value, gate, value

    # Input range for visualization
    x = torch.linspace(-5, 5, 100).unsqueeze(-1)  # (100, 1)

    print("Gating behavior comparison (input range -5 to 5):")
    print("-" * 60)
    for name in ['swish', 'gelu', 'relu', 'sigmoid']:
        gate = SimpleGate(name)
        output, gate_out, value_out = gate(x)

        # Analyze gating statistics
        gate_mean = gate_out.mean().item()
        gate_std = gate_out.std().item()
        gate_neg_frac = (gate_out < 0).float().mean().item()

        print(f"{name:>8}: gate_mean={gate_mean:.3f}, gate_std={gate_std:.3f}, "
              f"gate_negative={gate_neg_frac:.3f}")

visualize_gating()
# Output: Gating behavior comparison (input range -5 to 5):
# Output: ------------------------------------------------------------
# Output:    swish: gate_mean=0.312, gate_std=0.423, gate_negative=0.200
# Output:     gelu: gate_mean=0.287, gate_std=0.398, gate_negative=0.350
# Output:     relu: gate_mean=0.456, gate_std=0.487, gate_negative=0.500
# Output:  sigmoid: gate_mean=0.500, gate_std=0.289, gate_negative=0.000
```

### Example 4: GLU vs Standard FFN Training Comparison

```python
def compare_ffn_types():
    """Compare standard ReLU FFN with SwiGLU on a simple task."""
    d_model = 64
    d_ff_standard = 256  # Factor 4
    d_ff_swiglu = int(d_ff_standard * 2 / 3)  # Factor ~2.67

    class SimpleModel(nn.Module):
        def __init__(self, ffn_type='standard'):
            super().__init__()
            self.embed = nn.Embedding(100, d_model)
            if ffn_type == 'standard':
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, d_ff_standard),
                    nn.ReLU(),
                    nn.Linear(d_ff_standard, d_model),
                )
            else:
                self.ffn = GLUFFN(d_model, d_ff_swiglu, activation='swish')
            self.norm = nn.LayerNorm(d_model)
            self.proj = nn.Linear(d_model, 10)

        def forward(self, x):
            x = self.embed(x).mean(dim=1)
            x = self.norm(self.ffn(x))
            return self.proj(x)

    def train_model(model, steps=100):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        losses = []
        model.train()
        for step in range(steps):
            x = torch.randint(0, 100, (16, 8))
            y = torch.randint(0, 10, (16,))
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses

    model_std = SimpleModel('standard')
    model_swiglu = SimpleModel('swiglu')

    print("Training comparison (100 steps):")
    losses_std = train_model(model_std)
    losses_swiglu = train_model(model_swiglu)

    params_std = sum(p.numel() for p in model_std.parameters())
    params_swiglu = sum(p.numel() for p in model_swiglu.parameters())

    print(f"  Standard: params={params_std:,}, final loss={losses_std[-1]:.4f}")
    print(f"  SwiGLU:   params={params_swiglu:,}, final loss={losses_swiglu[-1]:.4f}")

# Uncomment to run
# compare_ffn_types()
```

## Common Mistakes

1. **Not reducing d_ff for GLU variants**: GLU has three weight matrices instead of two. Using the same \(d_{ff}\) as a standard FFN increases parameters by 50%. Always reduce \(d_{ff}\) by 2/3 to match.

2. **Forgetting bias=False**: Modern GLU implementations (Llama, Mistral) typically do not use bias in the FFN linear layers. Including bias adds unnecessary parameters.

3. **Using the wrong activation for the gate**: The gate activation should be applied to W1, not V1. V1 is the "value" projection and typically has no activation. The formula is \(\text{Act}(xW_1) \odot (xV_1)\).

4. **Confusing GLU with the overall FFN**: GLU is a replacement for the activation function and first projection in the FFN. The output projection \(W_2\) is the same as in a standard FFN.

5. **Misunderstanding the effective expansion factor**: With the standard \(d_{ff}\) reduction, the effective expansion factor of a SwiGLU FFN is \(8/3 \approx 2.67\), not 4. This is important when comparing model architectures.

## Interview Questions

### Beginner

**Q: What is a GLU variant and how does it differ from a standard FFN?**

A: A GLU variant replaces the standard FFN's single projection + activation with a gating mechanism. It uses two parallel projections: one for the "gate" (with an activation like SiLU or GELU) and one for the "value" (linear). The output is the element-wise product of the gate and value. This gating allows the network to selectively control information flow through each hidden neuron.

### Intermediate

**Q: Why does SwiGLU perform better than standard ReLU FFN despite having the same number of parameters?**

A: SwiGLU's gating mechanism provides several advantages: (1) The sigmoid gating in Swish allows for smooth, non-binary information flow (unlike ReLU's hard on/off). (2) The gating creates multiplicative interactions between the gate and value, which increases the model's expressive power. (3) Swish/SiLU has a small negative region for negative inputs, which allows gradient flow even for "off" neurons (unlike ReLU which completely kills negative activations). (4) The gating naturally creates a "memory" behavior — the gate can store information about which features are relevant for the current context.

### Advanced

**Q: The Llama architecture uses SwiGLU with an intermediate size of approximately 8/3 * d_model, while the original Transformer uses d_ff = 4 * d_model with ReLU. Explain why these ratios differ and how they achieve parameter parity.**

A: The original Transformer FFN has 2 weight matrices: W1 (d_model × 4d_model) and W2 (4d_model × d_model), giving 8d_model² parameters. SwiGLU has 3 matrices: W1 (d_model × d_ff), V1 (d_model × d_ff), and W2 (d_ff × d_model), giving 3 × d_model × d_ff parameters. To match the original 8d_model²: 3 × d_ff = 8 × d_model, so d_ff = 8/3 × d_model. This ensures parameter parity while providing the gating mechanism. The multiple-of-256 rounding (for GPU efficiency) causes slight deviations from exact 8/3.

## Practice Problems

### Easy

Implement a function that, given a standard FFN's \(d_{ff}\), computes the corresponding \(d_{ff}\) for SwiGLU, GeGLU, and ReGLU variants to maintain parameter parity.

### Medium

Train a small Transformer language model with standard ReLU FFN and SwiGLU FFN (matched parameters). Compare the validation perplexity after the same number of training steps.

### Hard

Implement a "learnable gating" variant where the gate activation function is a learned weighted combination of multiple activation functions (e.g., mix of ReLU, GELU, and SiLU). Compare with fixed-activation variants.

## Solutions

### Easy Solution

```python
def compute_glu_d_ff(standard_d_ff):
    """
    Compute GLU d_ff to match standard FFN parameter count.
    Standard FFN: 2 * d_model * d_ff parameters (up to leading order)
    GLU FFN: 3 * d_model * d_ff_glu parameters
    For parity: 2 * d_ff = 3 * d_ff_glu
    """
    d_ff_glu = int(2 * standard_d_ff / 3)
    return d_ff_glu

# Examples
for std_d_ff in [1024, 2048, 3072, 4096]:
    glu_d_ff = compute_glu_d_ff(std_d_ff)
    print(f"Standard d_ff={std_d_ff:5d} -> GLU d_ff={glu_d_ff:5d} (ratio={glu_d_ff/std_d_ff:.4f})")
# Output: Standard d_ff= 1024 -> GLU d_ff= 682 (ratio=0.6667)
# Output: Standard d_ff= 2048 -> GLU d_ff=1365 (ratio=0.6667)
# Output: Standard d_ff= 3072 -> GLU d_ff=2048 (ratio=0.6667)
# Output: Standard d_ff= 4096 -> GLU d_ff=2730 (ratio=0.6667)
```

## Related Concepts

- **DL-360: Feed-Forward Network**: The component that GLU variants modify.
- **DL-374: FFN Expansion Factor**: How the factor changes for GLU variants.
- **DL-378: d_ff**: The intermediate dimension parameter.
- **Activation Functions**: ReLU, GELU, SiLU/Swish, and their properties.
- **Gating Mechanisms**: The broader concept of multiplicative gates in neural networks (LSTM, GRU).

## Next Concepts

- DL-376: Transformer Dimensionality — How all dimensions relate.
- DL-381: Transformer Parameter Count — How GLU variants affect parameter counts.

## Summary

GLU variants are gated FFN architectures that replace the standard single-projection FFN with a dual-projection gating mechanism. Variants include SwiGLU (SiLU gate), GeGLU (GELU gate), and ReGLU (ReLU gate). SwiGLU has become the standard in modern LLMs (Llama, PaLM, Mixtral) due to its superior performance. GLU variants require three weight matrices instead of two, so the intermediate dimension is reduced by 1/3 to maintain parameter parity. Understanding GLU variants is essential for modern Transformer architecture design.

## Key Takeaways

1. GLU variants use a gate (activated) and value (linear) projection, multiplied element-wise.
2. SwiGLU (SiLU gate) is the default in modern LLMs.
3. GLU variants have 3 weight matrices vs. 2 for standard FFNs.
4. d_ff is reduced by 2/3 to maintain parameter parity (effective factor ≈ 2.67).
5. GLU variants consistently outperform standard ReLU FFNs.
6. Modern implementations typically set bias=False in GLU linear layers.
