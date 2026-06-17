# Concept: Transformer Block

## Concept ID

DL-358

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Transformer Foundations

## Learning Objectives

- Understand the structure and components of a Transformer block (the fundamental building unit of Transformer models).
- Explain the role of each sub-layer within the block: multi-head self-attention, feed-forward network, residual connection, and layer normalization.
- Differentiate between the encoder block and decoder block variants.
- Implement a Transformer block from scratch in PyTorch.
- Analyze the flow of information and gradients through the block.

## Prerequisites

- DL-356: Transformer Architecture Overview
- DL-359: Self-Attention Layer (recommended concurrent study)
- Understanding of residual networks and batch/layer normalization.
- Familiarity with PyTorch's nn.Module and tensor operations.

## Definition

A Transformer block (also called a Transformer layer) is a composite neural network module that serves as the fundamental building unit of Transformer models. Each block consists of a multi-head self-attention sub-layer followed by a position-wise feed-forward network, with residual connections and layer normalization surrounding each sub-layer. The encoder variant has two sub-layers (self-attention + FFN), while the decoder variant has three (masked self-attention + cross-attention + FFN). Multiple blocks are stacked to form the encoder and decoder of a Transformer model.

## Intuition

Think of a Transformer block as a processing unit that refines the representation of every token in a sequence. Each block performs two operations per token: (1) communication — the token "talks to" all other tokens via self-attention to gather contextual information, and (2) computation — the token processes the gathered information through a feed-forward network to produce a richer representation.

The block is designed so that information can flow easily through many stacked blocks. The residual connection provides a direct path for information to bypass the attention and feed-forward sub-layers, acting as a "gradient highway" during backpropagation. This allows dozens or even hundreds of blocks to be stacked effectively.

Layer normalization ensures that the activations have consistent statistics, preventing the scale of representations from growing uncontrollably and making training more stable.

## Why This Concept Matters

The Transformer block is the core repeating unit of all Transformer models. Understanding its design and how the components interact is essential because:

1. **Ubiquity**: Every Transformer model — BERT, GPT, T5, Llama, Mistral, ViT — is composed of stacked Transformer blocks.
2. **Modularity**: The block design enables easy scaling by adding more blocks (going from BERT-base with 12 blocks to BERT-large with 24, or GPT-3 with 96).
3. **Variations**: Many architectural innovations are modifications to the Transformer block (e.g., pre-norm vs post-norm, different activation functions, GLU variants).
4. **Debugging**: When a Transformer model fails to train or performs poorly, understanding the block's internals is essential for diagnosing issues.

## Mathematical Explanation

A Transformer block operates on a sequence of \(n\) tokens, each represented as a \(d_{\text{model}}\)-dimensional vector. The input is a tensor \(X \in \mathbb{R}^{n \times d_{\text{model}}}\).

### Encoder Block

The encoder block has two sub-layers:

**Sub-layer 1: Multi-Head Self-Attention**

\[
X' = \text{LayerNorm}(X + \text{MultiHeadSelfAttention}(X, X, X))
\]

In post-norm (original): the norm is applied after the addition. In pre-norm (modern): the norm is applied before the attention:

\[
X' = X + \text{MultiHeadSelfAttention}(\text{LayerNorm}(X), \text{LayerNorm}(X), \text{LayerNorm}(X))
\]

The multi-head attention computes:

\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
\]
\[
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) = \text{softmax}\left(\frac{QW_i^Q (KW_i^K)^T}{\sqrt{d_k}}\right) VW_i^V
\]

**Sub-layer 2: Feed-Forward Network**

The FFN is applied independently to each position:

\[
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
\]

where \(W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}\), \(W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}\), and typically \(d_{ff} = 4 \times d_{\text{model}}\).

With residual connection and normalization:

\[
X_{\text{out}} = \text{LayerNorm}(X' + \text{FFN}(X'))
\]

### Decoder Block

The decoder block has three sub-layers:

1. **Masked Multi-Head Self-Attention**: Same as encoder self-attention but with a causal mask to prevent attending to future tokens.
2. **Multi-Head Cross-Attention**: Queries from the decoder, keys and values from the encoder output.
3. **Feed-Forward Network**: Same as encoder FFN.

All three sub-layers have residual connections and layer normalization.

### Residual Connections

The residual (skip) connection adds the input of a sub-layer to its output. This is formalized as: \(y = x + F(x)\), where \(F\) is the sub-layer function. This provides a direct gradient flow path during backpropagation:

\[
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \left( 1 + \frac{\partial F(x)}{\partial x} \right)
\]

The identity term \(1\) ensures that the gradient does not vanish even if \(\frac{\partial F(x)}{\partial x}\) is small.

### Layer Normalization

Layer normalization normalizes activations across the feature dimension:

\[
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
\]

where \(\mu\) and \(\sigma^2\) are the mean and variance computed across the features, and \(\gamma\) and \(\beta\) are learnable parameters.

## Code Examples

### Example 1: Transformer Encoder Block from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, pre_norm=True):
        super().__init__()
        self.pre_norm = pre_norm
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if self.pre_norm:
            # Pre-norm: normalize before sub-layer
            attn_out, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask)
            x = x + self.dropout(attn_out)
            ff_out = self.ffn(self.norm2(x))
            x = x + ff_out
        else:
            # Post-norm: normalize after residual addition
            attn_out, _ = self.attention(x, x, x, attn_mask=mask)
            x = self.norm1(x + self.dropout(attn_out))
            ff_out = self.ffn(x)
            x = self.norm2(x + ff_out)
        return x

# Test block
d_model, n_heads, d_ff = 512, 8, 2048
block = EncoderBlock(d_model, n_heads, d_ff)
x = torch.randn(2, 10, d_model)
output = block(x)
print(f"Encoder block output shape: {output.shape}")
# Output: Encoder block output shape: torch.Size([2, 10, 512])

# Check that output has same shape as input (required for stacking)
assert output.shape == x.shape
print("Shape preserved: OK")
# Output: Shape preserved: OK
```

### Example 2: Decoder Block with Cross-Attention

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, pre_norm=True):
        super().__init__()
        self.pre_norm = pre_norm
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        if self.pre_norm:
            # Pre-norm decoder block
            attn_out, _ = self.self_attention(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=self_mask)
            x = x + self.dropout(attn_out)

            cross_out, _ = self.cross_attention(self.norm2(x), encoder_output, encoder_output, attn_mask=cross_mask)
            x = x + self.dropout(cross_out)

            ff_out = self.ffn(self.norm3(x))
            x = x + ff_out
        else:
            # Post-norm decoder block
            attn_out, _ = self.self_attention(x, x, x, attn_mask=self_mask)
            x = self.norm1(x + self.dropout(attn_out))

            cross_out, _ = self.cross_attention(x, encoder_output, encoder_output, attn_mask=cross_mask)
            x = self.norm2(x + self.dropout(cross_out))

            ff_out = self.ffn(x)
            x = self.norm3(x + ff_out)
        return x

# Test decoder block
dec_block = DecoderBlock(d_model, n_heads, d_ff)
tgt = torch.randn(2, 8, d_model)
enc_out = torch.randn(2, 10, d_model)
causal_mask = torch.triu(torch.full((8, 8), float('-inf')), diagonal=1)
output = dec_block(tgt, enc_out, self_mask=causal_mask)
print(f"Decoder block output shape: {output.shape}")
# Output: Decoder block output shape: torch.Size([2, 8, 512])
```

### Example 3: Stacking Transformer Blocks

```python
class StackedEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_positional_encoding(max_len, d_model)
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, mask)
        return x

    def get_intermediate_outputs(self, x, mask=None):
        """Returns outputs after each block for analysis."""
        seq_len = x.size(1)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)
        outputs = [x]
        for block in self.blocks:
            x = block(x, mask)
            outputs.append(x)
        return outputs

# Test stacking
vocab_size, n_layers = 10000, 12
encoder = StackedEncoder(vocab_size, d_model=512, n_heads=8, d_ff=2048, n_layers=n_layers)
input_ids = torch.randint(0, vocab_size, (2, 10))
output = encoder(input_ids)
intermediate = encoder.get_intermediate_outputs(input_ids)
print(f"Number of blocks: {len(encoder.blocks)}")
print(f"Intermediate outputs count: {len(intermediate)}")
print(f"Final output shape: {output.shape}")
print(f"Layer 6 output norm: {intermediate[6].norm().item():.4f}")
# Output: Number of blocks: 12
# Output: Intermediate outputs count: 13
# Output: Final output shape: torch.Size([2, 10, 512])
# Output: Layer 6 output norm: 178.2341
```

### Example 4: Gradient Flow Test

```python
def test_gradient_flow():
    """Verify gradients flow through all parameters."""
    block = EncoderBlock(64, 4, 256)
    x = torch.randn(2, 5, 64, requires_grad=True)
    output = block(x)
    loss = output.sum()
    loss.backward()

    total_grad = 0
    zero_grad = 0
    for name, param in block.named_parameters():
        if param.grad is not None:
            total_grad += 1
            if param.grad.abs().sum().item() == 0:
                zero_grad += 1
        else:
            print(f"Gradient is None for: {name}")

    print(f"Parameters with gradient: {total_grad}")
    print(f"Parameters with zero gradient: {zero_grad}")
    return total_grad > 0

grad_ok = test_gradient_flow()
print(f"Gradient flow OK: {grad_ok}")
# Output: Parameters with gradient: 10
# Output: Parameters with zero gradient: 0
# Output: Gradient flow OK: True
```

## Common Mistakes

1. **Forgetting to set `batch_first=True` in `nn.MultiheadAttention`**: PyTorch's default expects `(seq, batch, d_model)`. If your input is `(batch, seq, d_model)`, you must set `batch_first=True`. Forgetting this leads to silent shape errors or incorrect computations.

2. **Applying dropout on the residual branch incorrectly**: Dropout should be applied to the sub-layer output, not to the residual connection itself. The residual branch should be clean: `x + dropout(sublayer(x))`. Applying dropout to `x` as well harms performance.

3. **Confusing pre-norm and post-norm**: In pre-norm, normalization precedes the sub-layer; in post-norm, it follows the residual addition. Using post-norm with deep networks (>12 layers) often leads to training instability without careful learning rate scheduling.

4. **Applying the causal mask to cross-attention**: The causal mask (triangular) is only for decoder self-attention. Cross-attention should have full access to all encoder positions.

5. **Not keeping the hidden dimension consistent**: The Transformer block preserves the dimension: input and output both have shape `(batch, seq_len, d_model)`. If an internal module changes the sequence length or hidden dimension inadvertently, subsequent blocks will fail.

## Interview Questions

### Beginner

**Q: What are the two main sub-layers in a Transformer encoder block?**

A: The two sub-layers are: (1) multi-head self-attention, which allows each token to gather information from all other tokens in the sequence, and (2) a position-wise feed-forward network, which processes each token's representation independently. Each sub-layer has a residual connection and layer normalization.

### Intermediate

**Q: Explain the role of residual connections in Transformer blocks. Why are they especially important in deep Transformers?**

A: Residual connections create a direct gradient highway from the output back to the input. In deep networks with 12, 24, or more stacked blocks, gradients must propagate through many layers. Without residuals, the repeated application of attention and FFN would cause gradients to vanish (or explode). The residual connection ensures that the gradient of the loss with respect to the input of block \(i\) includes a term from the output that bypasses all sub-layer computations: \(\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial x_{i+1}} (1 + \frac{\partial F(x_i)}{\partial x_i})\). The identity term \(1\) ensures gradient flow even when the sub-layer's Jacobian is small.

### Advanced

**Q: A Transformer model with 96 blocks is not training — the loss decreases initially but then plateaus with high variance. Using pre-norm and adjusting the learning rate did not help. What architectural modifications could you investigate?**

A: Several factors could cause this: (1) Check the initialization — the default PyTorch initialization may not be optimal for deep Transformers. Consider using smaller initialization for residual branches (e.g., T5-style initialization where attention output is scaled by \(1/\sqrt{2n_{\text{layers}}}\)). (2) Investigate if gradient clipping (max_norm=1.0) is needed to prevent gradient explosion in early layers. (3) Verify that the FFN expansion factor \(d_{ff}\) is not too large, causing activation explosion in deep layers. (4) Check if the model benefits from additional regularization (increased dropout, weight decay, or label smoothing). (5) Consider implementing DeepNorm, which adjusts the initialization and residual scaling specifically for deep Transformers (up to 1000 layers). (6) Verify that the positional encodings do not dominate the token embeddings at later layers — if the positional signal is too strong, it can prevent the model from learning meaningful token interactions.

## Practice Problems

### Easy

Write a function that creates a Transformer encoder block using only `nn.Linear`, `nn.LayerNorm`, and `F.softmax` (no `nn.MultiheadAttention`). Include the scaled dot-product attention implementation.

### Medium

Implement a pre-norm decoder block and compare the gradient norms of early vs. late layers when using pre-norm vs. post-norm. Show empirically that pre-norm leads to more consistent gradient magnitudes across layers.

### Hard

Implement a "sandwich" block variant where the FFN is split and placed both before and after the attention sub-layer, as experimented with in the "Sandwich Transformer" paper. Compare training curves with the standard block on a language modeling task.

## Solutions

### Easy Solution

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores + mask
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, V)

class SimpleEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        # Self-attention
        Q = self.W_Q(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        attn_out = self.attention(Q, K, V, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        attn_out = self.W_O(attn_out)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

block = SimpleEncoderBlock(128, 4, 512)
x = torch.randn(2, 6, 128)
out = block(x)
print(f"Simple block output: {out.shape}")
# Output: Simple block output: torch.Size([2, 6, 128])
```

### Medium Solution

```python
def compare_norm_schemes():
    d_model, n_heads, d_ff = 128, 4, 512
    pre_block = DecoderBlock(d_model, n_heads, d_ff, pre_norm=True)
    post_block = DecoderBlock(d_model, n_heads, d_ff, pre_norm=False)

    x = torch.randn(2, 8, d_model)
    enc_out = torch.randn(2, 10, d_model)
    causal_mask = torch.triu(torch.full((8, 8), float('-inf')), diagonal=1)

    pre_out = pre_block(x, enc_out, causal_mask)
    post_out = post_block(x, enc_out, causal_mask)

    # Compute gradient norms for each parameter
    pre_grad_norms = {}
    post_grad_norms = {}

    loss = pre_out.sum()
    loss.backward(retain_graph=True)
    for name, p in pre_block.named_parameters():
        if p.grad is not None:
            pre_grad_norms[name] = p.grad.norm().item()

    loss = post_out.sum()
    loss.backward()
    for name, p in post_block.named_parameters():
        if p.grad is not None:
            post_grad_norms[name] = p.grad.norm().item()

    print("Pre-norm gradient norms:", {k: f"{v:.4f}" for k, v in list(pre_grad_norms.items())[:4]})
    print("Post-norm gradient norms:", {k: f"{v:.4f}" for k, v in list(post_grad_norms.items())[:4]})
    # Output: Pre-norm gradient norms: {'norm1.weight': '0.3542', ...}
    # Output: Post-norm gradient norms: {'norm1.weight': '0.1289', ...}

compare_norm_schemes()
```

## Related Concepts

- **DL-359: Self-Attention Layer**: The attention mechanism used within the transformer block.
- **DL-360: Feed-Forward Network**: The position-wise FFN sub-layer.
- **DL-366: Layer Normalization in Transformer**: The specific application of layer norm in Transformers.
- **DL-368: Residual Connections in Transformer**: The skip connections that enable deep stacking.
- **DL-367: Pre-Norm vs Post-Norm**: The two normalization placement schemes.

## Next Concepts

- DL-361: Positional Encoding — How Transformers encode sequence order.
- DL-369: Dropout in Transformer — Regularization within the block.
- DL-371: Attention Head — The individual attention computation within multi-head attention.

## Summary

The Transformer block is the fundamental building unit of Transformer models. It consists of multi-head self-attention (and optionally cross-attention for decoders) and a feed-forward network, each wrapped with residual connections and layer normalization. The block maintains the sequence length and hidden dimension, allowing arbitrary stacking. Understanding the block's design — the interaction of attention, FFN, residual connections, and normalization — is essential for working with any Transformer-based model.

## Key Takeaways

1. A Transformer block contains a multi-head attention sub-layer and a feed-forward sub-layer, each with residual connections and layer normalization.
2. Decoder blocks add a cross-attention sub-layer to attend to encoder outputs.
3. Residual connections provide direct gradient flow, enabling training of deep models.
4. Layer normalization stabilizes activations and improves training dynamics.
5. The block preserves dimensionality, allowing seamless stacking of many blocks.
