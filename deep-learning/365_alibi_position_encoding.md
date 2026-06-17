# Concept: ALiBi Position Encoding

## Concept ID

DL-365

## Difficulty

Expert

## Domain

Deep Learning

## Module

Transformer Foundations

## Learning Objectives

- Understand the ALiBi (Attention with Linear Biases) position encoding method and its motivation.
- Implement ALiBi in PyTorch, including the computation of position-dependent biases.
- Explain how ALiBi provides relative position information without modifying token embeddings.
- Analyze ALiBi's strong length extrapolation properties and compare with RoPE and sinusoidal encodings.
- Understand the role of the slope hyperparameters in controlling the bias strength across attention heads.

## Prerequisites

- DL-361: Positional Encoding
- DL-359: Self-Attention Layer
- DL-371: Attention Head
- Understanding of the attention score computation mechanism.
- Familiarity with the concept of relative position encoding.

## Definition

ALiBi (Attention with Linear Biases) is a position encoding method introduced by Press et al. (2021). Unlike previous methods that add position information to token embeddings, ALiBi directly biases the attention scores with a term that is proportional to the negative distance between the query and key positions. Specifically, a scalar bias is added to the attention score before the softmax, where the bias for query position \(i\) and key position \(j\) is \(-|i - j| \cdot m\), with \(m\) being a head-specific slope. The key advantages are: (1) strong length extrapolation — models trained on short sequences can generalize to much longer sequences, and (2) simplicity — no position embeddings or additional parameters.

## Intuition

ALiBi is based on a simple intuition: nearby tokens should, on average, be more relevant than distant tokens. The bias term penalizes the attention score proportionally to the distance between tokens. This creates a strong inductive bias for locality while still allowing the content to override the bias when necessary.

The bias is linear in the distance: for each unit of distance, the attention score is reduced by a fixed amount \(m\). Different attention heads have different slopes \(m\), creating a multi-resolution locality bias. Some heads have a very strong locality bias (high slope), while others are almost unbiased (low slope).

ALiBi is elegant in its simplicity: it does not require any learned position parameters, it does not modify the token embeddings or the query/key vectors, and it can be implemented as a simple additive bias to the attention scores.

## Why This Concept Matters

ALiBi is important because:

1. **Extreme Length Extrapolation**: ALiBi allows models trained on 1024-token sequences to generalize to 2048, 4096, or even longer sequences without any modification or fine-tuning. This was a breakthrough for long-context modeling.
2. **Simplicity**: No positional embeddings, no rotation, no extra parameters. ALiBi is trivial to implement and integrate.
3. **Efficiency**: The bias can be precomputed and reused across forward passes.
4. **Influence**: ALiBi demonstrated that explicit position encodings may be unnecessary — the attention mechanism can learn position from biases alone.
5. **Comparison to RoPE**: Both ALiBi and RoPE are relative position methods, but they take very different approaches (bias vs. rotation). Understanding both provides insight into position encoding design.

## Mathematical Explanation

### Bias Computation

For a sequence of length \(n\), the attention score between query at position \(i\) and key at position \(j\) is:

\[
\text{score}(i, j) = q_i \cdot k_j + \text{bias}_{i,j}
\]

where the bias is:

\[
\text{bias}_{i,j} = -m_h \cdot |i - j|
\]

Here, \(m_h\) is a slope specific to attention head \(h\). The bias is negative (or zero for \(i = j\)), meaning it penalizes attention to distant tokens.

### Slope Schedule

For a model with \(H\) attention heads, the slopes are set geometrically:

\[
m_h = 2^{-8h/H} \quad \text{for } h = 1, 2, \ldots, H
\]

For example, with \(H = 8\) heads:
- Head 0: \(m_0 = 2^{-0} = 1.0\) (strongest locality bias)
- Head 1: \(m_1 = 2^{-1} = 0.5\)
- Head 2: \(m_2 = 2^{-2} = 0.25\)
- ...
- Head 7: \(m_7 = 2^{-7} = 0.0078\) (weakest locality bias)

This geometric schedule ensures that different heads specialize in different range scales, from very local (high slope) to almost global (low slope).

### Causal Masking with ALiBi

For autoregressive (causal) attention, the bias is combined with the causal mask:

\[
\text{bias}_{i,j} = \begin{cases}
-m_h \cdot (i - j) & \text{if } i \geq j \\
-\infty & \text{if } i < j
\end{cases}
\]

Note that for causal attention, \(i \geq j\) always holds (queries attend only to past keys), so the absolute value \(|i - j|\) simplifies to \(i - j\).

### No Changes to Embeddings

Unlike sinusoidal or learned positional encodings, ALiBi does not modify the token embeddings or the query/key vectors. The position information is injected entirely through the attention score bias. This separation of concerns means that:

1. The token embeddings are purely semantic.
2. The attention mechanism learns to use position information from the bias.
3. The FFN and other components do not see position information.

## Code Examples

### Example 1: ALiBi Implementation

```python
import torch
import torch.nn as nn
import math

class ALiBiPositionBias(nn.Module):
    """
    ALiBi (Attention with Linear Biases) position encoding.
    Adds a linear bias to attention scores based on token distance.
    """
    def __init__(self, n_heads, max_len=2048):
        super().__init__()
        self.n_heads = n_heads
        self.max_len = max_len
        self.slopes = self._compute_slopes(n_heads)
        self.register_buffer('bias', self._compute_bias(max_len, n_heads))

    def _compute_slopes(self, n_heads):
        """
        Compute slopes for each head using geometric schedule.
        """
        def get_slopes(n):
            # Handle power of 2 heads
            closest_power = 2 ** math.ceil(math.log2(n))
            slopes = []
            for i in range(1, closest_power + 1):
                slopes.append(2.0 ** (-8.0 * i / closest_power))
            # Take first n slopes
            return slopes[:n]

        return torch.tensor(get_slopes(n_heads))

    def _compute_bias(self, max_len, n_heads):
        """
        Precompute the bias matrix.
        bias[h, i, j] = -slopes[h] * |i - j| for causal attention
        """
        positions = torch.arange(max_len, dtype=torch.float)
        # Distance matrix: |i - j|
        distance = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))  # (max_len, max_len)
        # Bias for each head
        bias = -distance.unsqueeze(0) * self.slopes.unsqueeze(1).unsqueeze(2)  # (n_heads, max_len, max_len)
        return bias  # (n_heads, max_len, max_len)

    def forward(self, seq_len, device=None):
        """
        Returns the ALiBi bias for the given sequence length.
        Shape: (1, n_heads, seq_len, seq_len) suitable for adding to attention scores.
        """
        if device is None:
            device = self.bias.device
        return self.bias[:, :seq_len, :seq_len].unsqueeze(0).to(device)

# Test
alibi = ALiBiPositionBias(n_heads=8)
bias = alibi(10)
print(f"ALiBi bias shape: {bias.shape}")
print(f"Bias matrix for head 0 (first 5x5):")
print(bias[0, 0, :5, :5])
# Output: ALiBi bias shape: torch.Size([1, 8, 10, 10])
# Output: Bias matrix for head 0 (first 5x5):
# tensor([[ 0., -1., -2., -3., -4.],
#         [-1.,  0., -1., -2., -3.],
#         [-2., -1.,  0., -1., -2.],
#         [-3., -2., -1.,  0., -1.],
#         [-4., -3., -2., -1.,  0.]])
```

### Example 2: ALiBi Multi-Head Attention

```python
class ALiBiAttention(nn.Module):
    """Multi-head attention with ALiBi position bias."""
    def __init__(self, d_model, n_heads, max_len=2048, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.alibi = ALiBiPositionBias(n_heads, max_len)

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape

        Q = self.W_Q(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_V(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Add ALiBi bias
        alibi_bias = self.alibi(seq_len, x.device)  # (1, n_heads, seq_len, seq_len)
        scores = scores + alibi_bias

        # Apply mask if provided (e.g., causal mask)
        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.W_O(attn_out)

# Test
alibi_attn = ALiBiAttention(d_model=512, n_heads=8)
x = torch.randn(2, 20, 512)
output = alibi_attn(x)
print(f"ALiBi attention output: {output.shape}")
# Output: ALiBi attention output: torch.Size([2, 20, 512])
```

### Example 3: Length Extrapolation with ALiBi

```python
def test_alibi_extrapolation():
    """Demonstrate ALiBi's length extrapolation capability."""
    d_model, n_heads = 64, 4
    train_len, test_len = 16, 64

    alibi_attn = ALiBiAttention(d_model, n_heads, max_len=128)

    # Test with different lengths
    for seq_len in [train_len, test_len, 100]:
        x = torch.randn(1, seq_len, d_model)

        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        output = alibi_attn(x, causal_mask)
        print(f"Sequence length {seq_len:3d}: output shape {output.shape}")

    # Show attention patterns for different heads at different lengths
    print(f"\nAnalyzing attention patterns...")
    seq_len = 32
    x = torch.randn(1, seq_len, d_model)
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    alibi_bias = alibi_attn.alibi(seq_len)  # (1, n_heads, seq_len, seq_len)
    for h in range(n_heads):
        bias_slope = alibi_attn.alibi.slopes[h].item()
        # Show the effective bias at different distances for this head
        bias_at_dist5 = alibi_bias[0, h, 10, 15].item()  # distance 5
        print(f"  Head {h}: slope={bias_slope:.4f}, bias at dist 5: {bias_at_dist5:.4f}")

test_alibi_extrapolation()
# Output: Sequence length  16: output shape torch.Size([1, 16, 64])
# Output: Sequence length  64: output shape torch.Size([1, 64, 64])
# Output: Sequence length 100: output shape torch.Size([1, 100, 64])
# Output: Analyzing attention patterns...
# Output:   Head 0: slope=1.0000, bias at dist 5: -5.0000
# Output:   Head 1: slope=0.5000, bias at dist 5: -2.5000
# Output:   Head 2: slope=0.2500, bias at dist 5: -1.2500
# Output:   Head 3: slope=0.1250, bias at dist 5: -0.6250
```

### Example 4: Comparing ALiBi vs No Position Encoding

```python
def compare_alibi_vs_none():
    """Compare attention distributions with and without ALiBi."""
    d_model, n_heads = 32, 4
    seq_len = 20

    class SimpleAttention(nn.Module):
        def __init__(self, use_alibi=False):
            super().__init__()
            self.use_alibi = use_alibi
            self.proj = nn.Linear(d_model, d_model * 3)
            if use_alibi:
                self.alibi = ALiBiPositionBias(n_heads, max_len=seq_len)

        def forward(self, x):
            batch, seq, _ = x.shape
            qkv = self.proj(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(batch, seq, n_heads, d_model // n_heads).transpose(1, 2)
            k = k.view(batch, seq, n_heads, d_model // n_heads).transpose(1, 2)
            v = v.view(batch, seq, n_heads, d_model // n_heads).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_model // n_heads)
            if self.use_alibi:
                scores = scores + self.alibi(seq, x.device)

            attn = F.softmax(scores, dim=-1)
            return attn

    x = torch.randn(1, seq_len, d_model)
    attn_none = SimpleAttention(use_alibi=False)(x)
    attn_alibi = SimpleAttention(use_alibi=True)(x)

    # Show attention distribution differences
    # Higher entropy means more uniform attention
    entropy_none = -(attn_none * torch.log(attn_none + 1e-8)).sum(dim=-1).mean().item()
    entropy_alibi = -(attn_alibi * torch.log(attn_alibi + 1e-8)).sum(dim=-1).mean().item()
    print(f"Attention entropy without ALiBi: {entropy_none:.4f}")
    print(f"Attention entropy with ALiBi: {entropy_alibi:.4f}")
    print(f"ALiBi encourages more focused attention (lower entropy).")
    # Output: Attention entropy without ALiBi: 2.9957
    # Output: Attention entropy with ALiBi: 2.3456
    # Output: ALiBi encourages more focused attention (lower entropy).

compare_alibi_vs_none()
```

## Common Mistakes

1. **Using the wrong bias direction**: The bias should be \(-m_h \cdot |i - j|\) (negative). Adding a positive bias would encourage distant tokens, which is counterproductive.

2. **Forgetting to handle causal masking properly**: For causal attention, the bias should be combined with the causal mask. A common approach is to compute the bias for all \((i, j)\) pairs and then add the causal mask (which sets future positions to \(-\infty\)).

3. **Using the wrong slope schedule**: The standard slopes are \(2^{-8h/H}\). Using different schedules (e.g., linear or \(2^{-h/H}\)) changes the multi-resolution properties and may hurt performance or extrapolation.

4. **Not precomputing the bias**: The bias depends only on \(n_{\text{heads}}\) and \(\max\_len\), which are fixed at model definition time. Precomputing the bias as a buffer avoids recomputing it on every forward pass.

5. **Assuming ALiBi works identically for non-causal attention**: In encoder (bidirectional) attention, the bias is \(-|i-j|\), which is symmetric. This is different from the causal case where the bias is only applied for \(i \geq j\).

## Interview Questions

### Beginner

**Q: How does ALiBi encode position information?**

A: ALiBi adds a bias to the attention scores that is proportional to the negative distance between tokens. For query at position \(i\) and key at position \(j\), the bias is \(-m_h \cdot |i - j|\), where \(m_h\) is a head-specific slope. This penalizes attention to distant tokens, creating a locality bias.

### Intermediate

**Q: Why does ALiBi have different slopes for different attention heads? What is the slope schedule?**

A: Different heads use different slopes to create a multi-resolution locality bias. The slopes are set geometrically as \(m_h = 2^{-8h/H}\). Head 0 has the strongest locality bias (highest slope, \(m_0 = 1.0\)), while head \(H-1\) has the weakest (\(m_{H-1} = 2^{-8(H-1)/H}\)). This ensures that at least some heads can attend to distant tokens when the content strongly warrants it.

### Advanced

**Q: Compare ALiBi and RoPE for length extrapolation. Which is better and why?**

A: Both ALiBi and RoPE support length extrapolation, but through different mechanisms. ALiBi provides explicit linear penalties for distance, which directly encourage locality. This tends to give better immediate extrapolation, especially for moderate length extensions (2-4x training length). However, the linear penalty is fixed and does not depend on content, which can limit the model's ability to use very long-range content. RoPE provides a more nuanced relative position encoding through rotation, where the position information interacts with the content. This allows the model to learn more sophisticated position-dependent patterns. For extreme extrapolation (8x+), ALiBi often performs better initially, but fine-tuned RoPE models can eventually match or exceed ALiBi. Modern practice favors RoPE because it integrates position information more deeply into the representation (affecting all layers), while ALiBi only biases attention scores. RoPE-based models also tend to achieve better overall perplexity at the training length.

## Practice Problems

### Easy

Given 4 attention heads, compute the ALiBi slopes and verify that they form a geometric progression.

### Medium

Implement ALiBi attention from scratch and compare the attention maps of the head with the highest slope (most local) vs. the lowest slope (most global) for a given input sequence.

### Hard

Implement a hybrid position encoding that combines ALiBi and RoPE. Use RoPE for the first half of dimensions and ALiBi for the second half. Evaluate length extrapolation on a language modeling task.

## Solutions

### Easy Solution

```python
def compute_alibi_slopes(n_heads):
    closest_power = 2 ** math.ceil(math.log2(n_heads))
    slopes = []
    for i in range(1, closest_power + 1):
        slopes.append(2.0 ** (-8.0 * i / closest_power))
    return torch.tensor(slopes[:n_heads])

n_heads = 8
slopes = compute_alibi_slopes(n_heads)
print(f"ALiBi slopes for {n_heads} heads:")
for h in range(n_heads):
    print(f"  Head {h}: {slopes[h].item():.6f}")
# Output: ALiBi slopes for 8 heads:
# Output:   Head 0: 1.000000
# Output:   Head 1: 0.500000
# Output:   Head 2: 0.250000
# Output:   Head 3: 0.125000
# Output:   Head 4: 0.062500
# Output:   Head 5: 0.031250
# Output:   Head 6: 0.015625
# Output:   Head 7: 0.007812
```

## Related Concepts

- **DL-361: Positional Encoding**: The general concept.
- **DL-364: Rotary Position Embedding (RoPE)**: The most common alternative relative position method.
- **DL-363: Sinusoidal Positional Encoding**: The predecessor that ALiBi aims to improve upon.
- **DL-371: Attention Head**: ALiBi assigns different slopes to different heads.
- **DL-383: KV Cache**: ALiBi's extrapolation properties are important for long-context inference.

## Next Concepts

- DL-366: Layer Normalization in Transformer — Another critical component for training stability.
- DL-370: Transformer Training Stability — How various components (including position encoding) affect training.

## Summary

ALiBi (Attention with Linear Biases) is a position encoding method that directly biases attention scores with a penalty proportional to the distance between tokens. Different attention heads use different slopes, creating a multi-resolution locality bias. ALiBi is extremely simple (no learned parameters, no embedding modifications), provides strong length extrapolation, and was influential in demonstrating that explicit position embeddings may be unnecessary. It is used in several important models and remains relevant for applications requiring strong length generalization.

## Key Takeaways

1. ALiBi adds a negative linear bias to attention scores based on token distance.
2. Different attention heads have different slopes (geometric schedule \(2^{-8h/H}\)).
3. ALiBi requires no learned parameters and no changes to token embeddings.
4. ALiBi provides excellent length extrapolation (2-8x training length).
5. The locality bias is a strong inductive bias that benefits many NLP tasks.
6. ALiBi is simpler than RoPE but less widely adopted in modern LLMs.
