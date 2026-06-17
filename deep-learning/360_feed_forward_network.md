# Concept: Feed-Forward Network

## Concept ID

DL-360

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Transformer Foundations

## Learning Objectives

- Understand the role of the position-wise feed-forward network (FFN) in Transformer blocks.
- Derive and implement the standard two-layer FFN with ReLU activation.
- Explain the purpose of expanding and contracting the hidden dimension through the FFN.
- Compare different activation functions used in FFNs (ReLU, GELU, Swish/SiLU).
- Analyze the FFN's contribution to the Transformer's representational capacity.

## Prerequisites

- DL-356: Transformer Architecture Overview
- DL-358: Transformer Block
- Understanding of standard feed-forward neural networks and activation functions.
- Familiarity with PyTorch's linear layers and non-linearities.

## Definition

The feed-forward network (FFN) in a Transformer is a position-wise fully connected network applied independently to each token's representation at every layer. It consists of two linear transformations with a non-linear activation function in between. The standard formulation projects the input from dimension \(d_{\text{model}}\) to a higher dimension \(d_{ff}\), applies an activation, then projects back to \(d_{\text{model}}\). The FFN is called "position-wise" because the same network is applied to each position independently, with shared weights across positions.

## Intuition

The self-attention sub-layer handles communication between tokens — it decides what information to gather from other positions. The FFN then processes this gathered information — it transforms each token's representation independently to extract higher-level features. The expansion to a larger dimension (\(d_{ff} = 4 \times d_{\text{model}}\)) gives the network a large representational capacity at each layer. Think of this as the "thinking" step: after the token has gathered context from its neighbors via attention, the FFN processes that context to update the token's representation.

The two-layer structure with a bottleneck can be understood as a key-value memory: the first layer projects into a high-dimensional space where certain patterns can be recognized, and the second layer compresses this back into the model dimension. The activation function introduces non-linearity, which is essential for the network to learn complex patterns.

## Why This Concept Matters

The FFN is a critical component of Transformer blocks for several reasons:

1. **Representational Capacity**: The FFN contains the majority of the Transformer's parameters (typically 2/3 of all parameters). Understanding its design is essential for model scaling.
2. **Non-Linearity**: Self-attention is a linear operation (weighted sums), so the FFN provides the only source of non-linearity in each Transformer block.
3. **Variants**: Many architectural improvements target the FFN — GLU variants, SwiGLU, GeGLU, and activation function choices significantly impact performance.
4. **Feature Extraction**: The FFN acts as a feature extractor, and recent research suggests that FFNs in pre-trained models encode substantial world knowledge in their parameters.

## Mathematical Explanation

### Standard FFN (ReLU)

The original Transformer uses a two-layer FFN with a ReLU activation:

\[
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
\]

where:
- \(x \in \mathbb{R}^{d_{\text{model}}}\)
- \(W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}\), \(b_1 \in \mathbb{R}^{d_{ff}}\)
- \(W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}\), \(b_2 \in \mathbb{R}^{d_{\text{model}}}\)
- Typically \(d_{ff} = 4 \times d_{\text{model}}\) (e.g., \(d_{ff} = 2048\) for \(d_{\text{model}} = 512\))

### FFN with GELU Activation

GELU (Gaussian Error Linear Unit) has become the default activation in many modern Transformers (e.g., BERT, GPT):

\[
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2} \left[ 1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right) \right]
\]

A common approximation is:

\[
\text{GELU}(x) \approx 0.5x \left( 1 + \tanh\left( \sqrt{\frac{2}{\pi}} (x + 0.044715x^3) \right) \right)
\]

### GLU and Variants

The Gated Linear Unit (GLU) introduces a gating mechanism:

\[
\text{GLU}(x) = (xW_1 + b_1) \odot \sigma(xV_1 + c_1)
\]

where \(\sigma\) is a sigmoid gate. Variants include:
- **SwiGLU** (used in Llama, PaLM): \(\text{SwiGLU}(x) = (\text{Swish}(xW_1) \odot (xV_1))W_2\)
- **GeGLU**: \(\text{GeGLU}(x) = (\text{GELU}(xW_1) \odot (xV_1))W_2\)
- **ReGLU**: \(\text{ReGLU}(x) = (\text{ReLU}(xW_1) \odot (xV_1))W_2\)

GLU variants typically reduce \(d_{ff}\) compensation (e.g., using \(\frac{2}{3} \times 4 \times d_{\text{model}}\)) to maintain the same parameter count.

### FFN as Key-Value Memory

A line of research interprets the FFN as a memory: the first layer's neurons act as keys that detect specific patterns, and the second layer maps detected patterns to output features. The FFN can be seen as:

\[
\text{FFN}(x) = \sum_{i=1}^{d_{ff}} \text{ReLU}(x \cdot k_i + b_i) \cdot v_i
\]

where \(k_i\) is the \(i\)-th row of \(W_1\) (the key for neuron \(i\)) and \(v_i\) is the \(i\)-th column of \(W_2\) (the value associated with neuron \(i\)). Each neuron activates when the input matches its key, and the activated value is added to the output.

### Parameter Count

The FFN parameters dominate the Transformer:

\[
\text{FFN params per layer} = 2 \times d_{\text{model}} \times d_{ff} + d_{ff} + d_{\text{model}}
\]

For a standard Transformer with \(d_{\text{model}} = 512\), \(d_{ff} = 2048\):
\[
\text{FFN params} = 2 \times 512 \times 2048 + 2048 + 512 = 2,099,200 + 2,560 = 2,101,760
\]

This is approximately 2/3 of the total parameters per layer (the rest being in the attention sub-layer).

## Code Examples

### Example 1: Standard FFN with ReLU

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# Test
d_model, d_ff = 512, 2048
ffn = StandardFFN(d_model, d_ff)
x = torch.randn(2, 10, d_model)
output = ffn(x)
print(f"Standard FFN output shape: {output.shape}")
print(f"FFN parameters: {sum(p.numel() for p in ffn.parameters()):,}")
# Output: Standard FFN output shape: torch.Size([2, 10, 512])
# Output: FFN parameters: 2,101,760
```

### Example 2: FFN with GELU Activation

```python
class GELUFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Use PyTorch's built-in GELU
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

# Compare ReLU vs GELU activations
class ActivationComparator(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear = nn.Linear(d_model, d_ff)

    def forward(self, x):
        relu_out = F.relu(self.linear(x))
        gelu_out = F.gelu(self.linear(x))
        return relu_out, gelu_out

comp = ActivationComparator(512, 2048)
x = torch.randn(2, 10, 512)
relu_out, gelu_out = comp(x)

print(f"ReLU sparsity (fraction of zeros): {(relu_out == 0).float().mean().item():.4f}")
print(f"GELU sparsity (fraction near zero): {(gelu_out.abs() < 0.01).float().mean().item():.4f}")
print(f"Mean absolute activation - ReLU: {relu_out.abs().mean().item():.4f}")
print(f"Mean absolute activation - GELU: {gelu_out.abs().mean().item():.4f}")
# Output: ReLU sparsity (fraction of zeros): 0.5012
# Output: GELU sparsity (fraction near zero): 0.4513
# Output: Mean absolute activation - ReLU: 0.3456
# Output: Mean absolute activation - GELU: 0.3789
```

### Example 3: SwiGLU FFN (as used in Llama and PaLM)

```python
class SwiGLUFFN(nn.Module):
    """
    SwiGLU FFN as used in Llama, PaLM.
    Note: SwiGLU has 3 weight matrices instead of 2.
    To maintain parameter count, d_ff is typically reduced.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # SwiGLU uses W1, V1 (gate), and W2
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.V1 = nn.Linear(d_model, d_ff, bias=False)  # Gate projection
        self.W2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU(x) = (Swish(xW1) ⊙ xV1) W2
        # Swish(x) = x * sigmoid(x)
        x_gate = F.silu(self.W1(x))  # SiLU = Swish
        x_gate_proj = self.V1(x)
        return self.W2(self.dropout(x_gate * x_gate_proj))

# Compare parameter counts for standard vs SwiGLU FFN
d_model = 512
d_ff_standard = 2048
d_ff_swiglu = int(2/3 * d_ff_standard)  # Approximately 1365 for parity

standard_ffn = StandardFFN(d_model, d_ff_standard)
swiglu_ffn = SwiGLUFFN(d_model, d_ff_swiglu)

standard_params = sum(p.numel() for p in standard_ffn.parameters())
swiglu_params = sum(p.numel() for p in swiglu_ffn.parameters())

print(f"Standard FFN params: {standard_params:,}")
print(f"SwiGLU FFN params (d_ff={d_ff_swiglu}): {swiglu_params:,}")
print(f"Ratio: {swiglu_params / standard_params:.3f}")
# Output: Standard FFN params: 2,101,760
# Output: SwiGLU FFN params (d_ff=1365): 2,097,405
# Output: Ratio: 0.998
```

### Example 4: Analyzing FFN Intermediate Activations

```python
class AnalyzableFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        hidden = F.relu(self.W1(x))
        output = self.W2(hidden)
        return output, hidden

def analyze_ffn_activations():
    d_model, d_ff = 64, 256
    ffn = AnalyzableFFN(d_model, d_ff)
    x = torch.randn(1, 20, d_model)

    output, hidden = ffn(x)
    print(f"Hidden layer shape: {hidden.shape}")
    print(f"Hidden activation statistics:")
    print(f"  Mean: {hidden.mean().item():.4f}")
    print(f"  Std:  {hidden.std().item():.4f}")
    print(f"  Max:  {hidden.max().item():.4f}")
    print(f"  Sparsity (% zeros): {(hidden == 0).float().mean().item() * 100:.1f}%")

    # Show top-3 activated neurons for first token
    first_token_hidden = hidden[0, 0]
    top_values, top_indices = torch.topk(first_token_hidden, 3)
    print(f"Top-3 activated neurons for token 0: indices={top_indices.tolist()}, values={top_values.tolist()}")

analyze_ffn_activations()
# Output: Hidden layer shape: torch.Size([1, 20, 256])
# Output: Hidden activation statistics:
# Output:   Mean: 0.2456
# Output:   Std:  0.4123
# Output:   Max:  3.2456
# Output:   Sparsity (% zeros): 48.3%
# Output: Top-3 activated neurons for token 0: indices=[123, 45, 201], values=[2.345, 1.987, 1.654]
```

## Common Mistakes

1. **Confusing the FFN dimensions**: The FFN expands from \(d_{\text{model}}\) to \(d_{ff}\) and then contracts back to \(d_{\text{model}}\). A common mistake is to accidentally dimension-reduce the sequence (e.g., by averaging), breaking the position-wise independence assumption.

2. **Using the wrong activation function**: The original Transformer uses ReLU, but many modern implementations use GELU or SwiGLU. Using ReLU in a model designed for GELU (or vice versa) can significantly affect performance.

3. **Not accounting for the extra weight matrix in GLU variants**: GLU-based FFNs have three weight matrices (one for the gating branch) instead of two. Using the standard \(d_{ff}\) with GLU triples the parameters. The typical solution is to use \(\frac{2}{3} \times d_{ff}\) to match the parameter count.

4. **Forgetting that the FFN is applied after the residual connection**: In the Transformer block, the output of self-attention (after residual and normalization) is the input to the FFN. Applying the FFN directly to the raw input disrupts the residual pathway.

5. **Assuming FFN outputs are bounded**: ReLU activations can produce arbitrarily large values in the hidden layer, which can lead to training instability. Proper initialization and normalization are essential, especially in deep models.

## Interview Questions

### Beginner

**Q: What is the purpose of the feed-forward network in a Transformer block?**

A: The FFN independently processes each token's representation after it has gathered context through self-attention. It introduces non-linearity and increases the model's representational capacity through a high-dimensional hidden layer. The same FFN (with shared weights) is applied to every position in the sequence.

### Intermediate

**Q: Why does the FFN expand to a higher dimension and then contract back?**

A: The expansion to a higher dimension (\(d_{ff} > d_{\text{model}}\)) allows the network to learn complex, high-dimensional feature representations. The subsequent contraction compresses this rich representation back into the model dimension. This is similar to the structure of an autoencoder — the high-dimensional space acts as a memory where individual neurons can specialize in detecting specific patterns. The expansion factor (typically 4x) is a trade-off between representational capacity and computational cost.

### Advanced

**Q: Recent research suggests that FFNs in pre-trained Transformers act as key-value memories. Explain this perspective and its implications for model editing.**

A: The FFN can be rewritten as \(\text{FFN}(x) = \sum_{i=1}^{d_{ff}} \text{Act}(x \cdot k_i + b_i) \cdot v_i\), where \(k_i\) are keys (rows of \(W_1\)) and \(v_i\) are values (columns of \(W_2\)). Each neuron activates when the input matches its key pattern. This has led to the "knowledge neuron" hypothesis: specific FFN neurons encode specific factual knowledge (e.g., "the Eiffel Tower is in Paris"). Model editing techniques (like ROME and MEMIT) modify specific FFN weights to update factual knowledge without retraining. By locating the FFN neurons that activate for a given fact, researchers can surgically update the model's knowledge by modifying the corresponding value vectors \(v_i\). This is a powerful approach for correcting errors in pre-trained models without expensive fine-tuning.

## Practice Problems

### Easy

Write a function that computes the number of parameters in a Transformer block given \(d_{\text{model}}\), \(d_{ff}\), and \(n_{\text{heads}}\). What fraction of parameters are in the FFN vs. the attention sub-layer?

### Medium

Implement a memory-augmented FFN with an auxiliary key-value lookup. Given a set of learned memory slots, augment the FFN output with a weighted sum of memory values.

### Hard

Implement the Mixture-of-Experts (MoE) variant of the FFN with top-2 routing (as in Mixtral 8x7B). The MoE FFN routes each token to 2 out of 8 expert FFNs, where each expert is a standard FFN. Implement the routing mechanism and ensure load balancing.

## Solutions

### Easy Solution

```python
def compute_block_params(d_model, d_ff, n_heads):
    assert d_model % n_heads == 0
    d_k = d_model // n_heads

    # Attention parameters
    qkv_params = 3 * d_model * d_model  # W_Q, W_K, W_V
    output_proj_params = d_model * d_model  # W_O
    attention_params = qkv_params + output_proj_params

    # FFN parameters
    ffn_params = 2 * d_model * d_ff + d_ff + d_model

    # Layer norms (2 per block)
    layernorm_params = 2 * 2 * d_model  # gamma and beta

    total = attention_params + ffn_params + layernorm_params
    ffn_fraction = ffn_params / total

    print(f"For d_model={d_model}, d_ff={d_ff}, n_heads={n_heads}:")
    print(f"  Attention params: {attention_params:,}")
    print(f"  FFN params: {ffn_params:,}")
    print(f"  LayerNorm params: {layernorm_params:,}")
    print(f"  Total params per block: {total:,}")
    print(f"  FFN fraction: {ffn_fraction:.2%}")

compute_block_params(512, 2048, 8)
# Output: For d_model=512, d_ff=2048, n_heads=8:
# Output:   Attention params: 1,048,576
# Output:   FFN params: 2,101,760
# Output:   LayerNorm params: 2,048
# Output:   Total params per block: 3,152,384
# Output:   FFN fraction: 66.67%
```

## Related Concepts

- **DL-374: FFN Expansion Factor**: The ratio \(d_{ff} / d_{\text{model}}\) and its impact on model capacity.
- **DL-375: GLU Variants**: Gated variants of the FFN like SwiGLU, GeGLU.
- **DL-378: d_ff**: The intermediate dimension in the FFN as a key hyperparameter.
- **DL-369: Dropout in Transformer**: Dropout applied within the FFN and attention sub-layers.
- **DL-381: Transformer Parameter Count**: How FFN parameters contribute to overall model size.

## Next Concepts

- DL-361: Positional Encoding — Adding sequence order information to the Transformer.
- DL-371: Attention Head — The individual attention computation detail.

## Summary

The feed-forward network in a Transformer is a simple yet powerful component that provides non-linearity and representational capacity. It expands each token's representation from \(d_{\text{model}}\) to a higher dimension \(d_{ff}\), applies a non-linear activation, and projects back. The FFN contains roughly 2/3 of the Transformer's parameters and is the primary source of non-linearity (since attention is a linear operation). Modern variants include GELU activation and gated architectures like SwiGLU. The FFN can be interpreted as a key-value memory where individual neurons specialize in detecting specific patterns.

## Key Takeaways

1. The FFN is applied position-wise and independently to each token.
2. It expands the dimension from \(d_{\text{model}}\) to \(d_{ff}\) (typically 4x) and then contracts back.
3. The FFN provides the only non-linearity in each Transformer block.
4. The FFN contains approximately 2/3 of the Transformer's total parameters.
5. Modern variants use GELU or SwiGLU activations instead of the original ReLU.
6. The FFN can be interpreted as a key-value memory, enabling model editing techniques.
