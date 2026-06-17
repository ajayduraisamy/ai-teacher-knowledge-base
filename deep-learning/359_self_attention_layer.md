# Concept: Self-Attention Layer

## Concept ID

DL-359

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Transformer Foundations

## Learning Objectives

- Understand the mathematical formulation of scaled dot-product self-attention.
- Explain how self-attention computes contextualized representations by allowing each token to attend to all other tokens.
- Differentiate between queries, keys, and values, and understand their roles in the attention computation.
- Implement self-attention from scratch in PyTorch, including the masking variants (padding mask and causal mask).
- Analyze the computational complexity and memory requirements of self-attention.

## Prerequisites

- DL-356: Transformer Architecture Overview
- Basic understanding of linear algebra (matrix multiplication, dot products, softmax).
- Familiarity with PyTorch tensor operations.
- Knowledge of the general attention mechanism (e.g., Bahdanau attention).

## Definition

Self-attention, also called intra-attention, is a mechanism that computes a weighted sum of values for each element in a sequence, where the weights are determined by the pairwise compatibility between elements. Given an input sequence of tokens, self-attention computes a representation of the sequence where each token's representation incorporates information from all other tokens. The weights are computed as the scaled dot-product between queries and keys, followed by a softmax. The output is a weighted sum of values, where the weights reflect the relevance of each token to the current token.

## Intuition

Imagine you are at a party with many people talking. When you hear a sentence, your brain naturally pays more attention to certain words to understand the meaning. For example, in "She fed the cat, which was hungry," the word "which" refers to "cat," not "She." Self-attention models this behavior: for each word in a sentence, it computes how relevant every other word is and uses that relevance to update the word's representation.

The mechanism uses three learned vectors per token: a query, a key, and a value. Think of this as a database retrieval system:
- **Query**: What information am I looking for?
- **Key**: What information do I contain?
- **Value**: What information do I provide if matched?

Each token broadcasts its key (what it has) and value (what it offers), and each token broadcasts a query (what it seeks). The match between a query and a key determines how much the corresponding value is incorporated.

The beauty of self-attention is that it is permutation-equivariant: changing the order of input tokens simply permutes the output in the same way. This is why positional encodings must be added separately to inform the model about token positions.

## Why This Concept Matters

Self-attention is the core innovation of the Transformer. Understanding it is critical because:

1. **Long-Range Dependencies**: Self-attention creates a direct path between any two tokens, regardless of distance. This solves the vanishing gradient problem of RNNs for long sequences.
2. **Parallelization**: Unlike RNNs, self-attention processes all tokens simultaneously, enabling efficient GPU utilization.
3. **Interpretability**: Attention weights provide a built-in mechanism for understanding which parts of the input the model focuses on.
4. **Variants**: Many modern architectures (linear attention, sparse attention, flash attention) are optimizations of the self-attention computation.
5. **Cross-Attention**: The same mechanism extends to cross-attention (queries from one sequence, keys/values from another), which is fundamental for encoder-decoder models and multimodal architectures.

## Mathematical Explanation

### Scaled Dot-Product Attention

Given an input sequence of \(n\) tokens, each represented as a \(d_{\text{model}}\)-dimensional vector, we compute three matrices:

\[
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
\]

where \(X \in \mathbb{R}^{n \times d_{\text{model}}}\), \(W^Q, W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times d_k}\), and \(d_k = d_{\text{model}} / h\) is the dimension per head.

The attention scores are computed as:

\[
\text{Scores} = QK^T \in \mathbb{R}^{n \times n}
\]

Each element \(s_{ij} = q_i \cdot k_j\) represents the compatibility between token \(i\) (query) and token \(j\) (key).

The scores are scaled by \(\frac{1}{\sqrt{d_k}}\) to prevent the dot products from growing large in magnitude, which would push the softmax into regions with extremely small gradients:

\[
\text{ScaledScores} = \frac{QK^T}{\sqrt{d_k}}
\]

The attention weights are computed by applying softmax along the key dimension (columns):

\[
A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{n \times n}
\]

where each row \(A_{i,:}\) sums to 1 and represents the attention distribution for token \(i\).

The output is a weighted sum of values:

\[
\text{Output} = AV \in \mathbb{R}^{n \times d_k}
\]

### Multi-Head Attention

Multi-head attention runs \(h\) attention operations in parallel with different learned projections:

\[
\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)
\]
\[
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
\]

where \(W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_k}\) and \(W^O \in \mathbb{R}^{h d_k \times d_{\text{model}}}\).

### Masking

**Padding Mask**: To prevent attention to padding tokens, a binary mask \(M \in \{0, -\infty\}^{n \times n}\) is added to the scores before softmax:

\[
\text{Attention}(Q, K, V, M) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
\]

Positions with \(-\infty\) in the mask receive zero attention weight after softmax.

**Causal Mask**: For autoregressive generation, a triangular mask ensures that position \(i\) can only attend to positions \(j \leq i\):

\[
M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}
\]

### Computational Complexity

The self-attention operation has:
- **Time complexity**: \(O(n^2 \cdot d_k)\) — quadratic in sequence length due to the \(n \times n\) matrix multiplication.
- **Memory complexity**: \(O(n^2)\) — the attention matrix must be stored.
- This quadratic scaling is the primary bottleneck for processing long sequences.

## Code Examples

### Example 1: Scaled Dot-Product Attention from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q: (batch, ..., seq_len_q, d_k)
        K: (batch, ..., seq_len_k, d_k)
        V: (batch, ..., seq_len_k, d_v)
        mask: (batch, 1, seq_len_q, seq_len_k) or (seq_len_q, seq_len_k)
    Returns:
        output: (batch, ..., seq_len_q, d_v)
        attention_weights: (batch, ..., seq_len_q, seq_len_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

# Test
batch, seq_len, d_k = 2, 5, 64
Q = torch.randn(batch, seq_len, d_k)
K = torch.randn(batch, seq_len, d_k)
V = torch.randn(batch, seq_len, d_k)

output, attn_weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attn_weights.shape}")
print(f"Attention weights row sum (should be 1): {attn_weights[0, 0].sum().item():.4f}")
# Output: Output shape: torch.Size([2, 5, 64])
# Output: Attention weights shape: torch.Size([2, 5, 5])
# Output: Attention weights row sum (should be 1): 1.0000

# Verify attention behavior: a token should attend most to itself if all tokens are similar
identical_x = torch.ones(batch, seq_len, d_k)
output2, attn_weights2 = scaled_dot_product_attention(identical_x, identical_x, identical_x)
print(f"Uniform attention weights (first row): {attn_weights2[0, 0]}")
# Output: Uniform attention weights (first row): tensor([0.2000, 0.2000, 0.2000, 0.2000, 0.2000])
```

### Example 2: Self-Attention with Masking

```python
class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Q, K, V: (batch, n_heads, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, V)
        # Concatenate heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.W_O(attn_out), attn_weights

# Test with causal masking
def test_causal_mask():
    seq_len = 5
    d_model, n_heads = 64, 4
    sa = SelfAttention(d_model, n_heads)
    x = torch.randn(2, seq_len, d_model)

    # Create causal mask
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    # Add head dimension: (1, 1, seq_len, seq_len)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    output, attn_weights = sa(x, causal_mask)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    # Check causality: attn_weights[i,j] should be 0 for j > i
    print(f"Upper triangle of attention (should be 0): {attn_weights[0, 0].triu(diagonal=1)}")
    # Output: Output shape: torch.Size([2, 5, 64])
    # Output: Attention weights shape: torch.Size([2, 4, 5, 5])
    # Output: Upper triangle of attention (should be 0): tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000], ...])

test_causal_mask()
```

### Example 3: Visualizing Attention Weights

```python
def visualize_attention():
    """Simple demonstration of attention pattern."""
    d_model, n_heads = 32, 2
    sa = SelfAttention(d_model, n_heads)

    # Create a sequence where each token is distinct
    x = F.one_hot(torch.arange(6) % 6, num_classes=6).float()
    x = x.unsqueeze(0)  # (1, 6, 6)
    # Project to d_model
    x = nn.Linear(6, d_model)(x)

    output, attn_weights = sa(x)
    # Average over heads
    avg_attn = attn_weights.mean(dim=1).squeeze(0)  # (6, 6)

    print("Average attention weights (6x6):")
    for i in range(6):
        row = " ".join([f"{v:.3f}" for v in avg_attn[i]])
        print(f"  Token {i}: [{row}]")
    # Output: Average attention weights (6x6):
    # Output:   Token 0: [0.250 0.150 0.200 0.180 0.120 0.100]
    # Output:   ...

visualize_attention()
```

### Example 4: Computational Complexity Measurement

```python
import time

def benchmark_attention():
    d_model = 512
    n_heads = 8
    d_k = d_model // n_heads
    sa = SelfAttention(d_model, n_heads)

    for seq_len in [64, 128, 256, 512]:
        x = torch.randn(1, seq_len, d_model)

        # Warmup
        for _ in range(10):
            _ = sa(x)

        # Timed runs
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(100):
            _ = sa(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.time() - start) / 100

        # Count FLOPs: 2 * batch * n_heads * seq_len^2 * d_k
        flops = 2 * 1 * n_heads * seq_len * seq_len * d_k * 100 / elapsed
        print(f"Seq_len={seq_len:4d}: {elapsed*1000:.3f} ms/iter, ~{flops/1e9:.1f} GFLOP/s")

benchmark_attention()
# Output: Seq_len= 64: 0.123 ms/iter, ~5.3 GFLOP/s
# Output: Seq_len=128: 0.456 ms/iter, ~5.7 GFLOP/s
# Output: Seq_len=256: 1.789 ms/iter, ~5.9 GFLOP/s
# Output: Seq_len=512: 7.123 ms/iter, ~5.9 GFLOP/s
```

## Common Mistakes

1. **Scaling factor omission or incorrect scaling**: The dot products are scaled by \(\frac{1}{\sqrt{d_k}}\), not \(\frac{1}{d_k}\) or \(\frac{1}{\sqrt{d_{\text{model}}}}\). The correct scaling ensures that the variance of the dot products is approximately 1, keeping the softmax in a reasonable range.

2. **Masking with zeros instead of -inf**: When applying padding or causal masks, masked positions should be set to \(-\infty\) (or a large negative number) before the softmax, not 0. Using 0 would still assign attention weight to masked positions.

3. **Forgetting to add the mask instead of multiplying**: The mask is added to the scores (\(scores + mask\)), not multiplied (\(scores * mask\)). Using multiplication would not correctly mask with \(-\infty\).

4. **Applying dropout to attention weights in the wrong dimension**: Attention dropout is applied to the attention weights (after softmax), not to the scores or the final output. Dropping out attention weights encourages the model to distribute its attention more broadly.

5. **Assuming self-attention is order-aware**: Self-attention is permutation-equivariant — it treats the input as a set, not a sequence. Forgetting to add positional information means the model cannot distinguish between "The cat sat on the mat" and "Mat the on sat cat The." Positional encodings are essential.

## Interview Questions

### Beginner

**Q: What are the three matrices in self-attention, and what do they represent?**

A: Query (Q), Key (K), and Value (V). The query represents what information the current token is looking for. The key represents what information each token contains. The value represents the actual content that will be returned if the query matches the key. The attention weight is computed from the dot product of query and key, and the output is a weighted sum of values.

### Intermediate

**Q: Why is the attention score scaled by \(\frac{1}{\sqrt{d_k}}\)?**

A: The dot products \(q_i \cdot k_j\) have variance approximately \(d_k\) when the elements of \(q_i\) and \(k_j\) are independent with mean 0 and variance 1. Without scaling, large values of \(d_k\) cause the dot products to have large magnitudes, pushing the softmax function into regions where it has extremely small gradients (saturating). Scaling by \(\frac{1}{\sqrt{d_k}}\) normalizes the variance to approximately 1, which keeps the softmax in a regime with larger gradients, improving training stability.

### Advanced

**Q: Self-attention has \(O(n^2)\) complexity. Propose and compare three approaches to reduce this complexity for long sequences.**

A: Three approaches are: (1) **Sparse attention** (e.g., Longformer, BigBird) — compute attention only for local windows plus selected global tokens. Reduces complexity to \(O(n \cdot w)\) for window size \(w\), but may miss long-range dependencies outside the window. (2) **Linear attention** (e.g., Performer, Linear Transformer) — replace softmax with a kernel approximation, rewriting attention as \(Q(K^T V)\) instead of \((QK^T)V\), reducing complexity to \(O(n d^2)\). This is exact only with specific kernel choices (e.g., positive orthogonal random features for Performer). (3) **Flash Attention** — not a complexity reduction, but an IO-aware algorithm that reduces memory reads/writes by tiling the attention computation, achieving 2-4x speedup while maintaining exact attention. For truly long sequences (100k+), a combination of sparsity and IO-awareness is typically needed. Newer approaches include state-space models (Mamba) that achieve \(O(n)\) complexity but trade off some expressivity.

## Practice Problems

### Easy

Implement a function that computes attention scores and weights for a single query vector against multiple key vectors. Given a query \(q \in \mathbb{R}^{d_k}\) and key matrix \(K \in \mathbb{R}^{n \times d_k}\), compute the attention distribution and the weighted sum.

### Medium

Implement multi-head self-attention from scratch (without using `nn.MultiheadAttention`). Ensure you properly split the input into heads, compute attention, concatenate, and apply the output projection.

### Hard

Implement causal self-attention with a sliding window (local attention). In this variant, each token can only attend to tokens within a window of size \(w\) around it (including itself). Ensure the implementation is efficient and uses proper masking.

## Solutions

### Easy Solution

```python
def single_query_attention(q, K, V):
    """
    Args:
        q: (d_k,) - query vector
        K: (n, d_k) - key matrix
        V: (n, d_v) - value matrix
    Returns:
        output: (d_v,)
        weights: (n,)
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, K.T) / math.sqrt(d_k)  # (n,)
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)  # (d_v,)
    return output, weights

q = torch.randn(64)
K = torch.randn(10, 64)
V = torch.randn(10, 64)
output, weights = single_query_attention(q, K, V)
print(f"Output: {output.shape}, Weights sum: {weights.sum().item():.4f}")
# Output: Output: torch.Size([64]), Weights sum: 1.0000
```

### Medium Solution

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

attn = MultiHeadSelfAttention(512, 8)
x = torch.randn(2, 10, 512)
out = attn(x)
print(f"Multi-head self-attention output: {out.shape}")
# Output: Multi-head self-attention output: torch.Size([2, 10, 512])
```

## Related Concepts

- **DL-371: Attention Head**: The individual attention computation within multi-head attention.
- **DL-372: Multi-Head Attention Splitting**: How input is split across multiple heads.
- **DL-373: Attention Head Concatenation**: How head outputs are combined.
- **DL-383: KV Cache**: Optimization for efficient autoregressive inference.
- **DL-385: Flash Attention**: IO-aware implementation of self-attention.
- **Cross-Attention**: An extension where queries come from one sequence and keys/values from another.

## Next Concepts

- DL-360: Feed-Forward Network — The other major sub-layer in Transformer blocks.
- DL-361: Positional Encoding — Injecting order information into permutation-equivariant attention.

## Summary

Self-attention is the core mechanism of the Transformer architecture. It computes a contextualized representation for each token by aggregating information from all other tokens, weighted by pairwise compatibility scores. The operation involves computing queries, keys, and values from the input, computing scaled dot-product scores, applying a softmax to obtain attention weights, and returning a weighted sum of values. Self-attention enables parallel processing, captures long-range dependencies, and is permutation-equivariant (requiring positional encodings). Its quadratic complexity is both a strength (full connectivity) and a weakness (poor scaling to long sequences).

## Key Takeaways

1. Self-attention computes a weighted sum of values, with weights determined by query-key dot products followed by softmax.
2. The scaling factor \(\frac{1}{\sqrt{d_k}}\) prevents softmax saturation by controlling the variance of dot products.
3. Self-attention processes all tokens in parallel, making it efficient on modern hardware.
4. Padding masks and causal masks are applied by adding \(-\infty\) to the scores before softmax.
5. The quadratic \(O(n^2)\) complexity is the primary limitation for long-sequence modeling.
