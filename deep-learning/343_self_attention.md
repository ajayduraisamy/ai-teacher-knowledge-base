# Concept: Self-Attention

## Concept ID

DL-343

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Attention Mechanisms

## Learning Objectives

- Define self-attention as the mechanism where each token in a sequence attends to all tokens in the same sequence.
- Understand how self-attention captures contextual relationships between all pairs of tokens.
- Implement self-attention in PyTorch and analyze its behavior.
- Differentiate self-attention from cross-attention and encoder-decoder attention.
- Recognize self-attention as the core building block of transformer architectures.

## Prerequisites

- Understanding of the general attention mechanism (query, key, value).
- Familiarity with matrix operations and softmax normalization.
- Knowledge of seq2seq models and cross-attention.
- Basic understanding of transformers and the transformer architecture.

## Definition

Self-attention, also known as intra-attention, is an attention mechanism that relates different positions within a single sequence to compute a contextualized representation of the sequence. Given an input sequence X = (x_1, ..., x_T), self-attention computes:

Q = X W^Q, K = X W^K, V = X W^V
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

where W^Q, W^K, W^V are learned projection matrices. Each output position is a weighted sum of all input positions, where the weights depend on the pairwise compatibility between positions. Unlike cross-attention (where queries come from one sequence and keys/values from another), in self-attention all three matrices come from the same sequence. This allows each token to gather context from the entire sequence, capturing dependencies regardless of distance. Self-attention is the core mechanism of transformer encoders and decoders, replacing recurrence and convolution for sequence processing.

## Intuition

Imagine you are reading a sentence and trying to understand the meaning of each word. To understand "bank" in "He went to the bank to deposit money," you need to look at the surrounding words: "deposit" and "money" tell you it's a financial bank, not a river bank. Self-attention does exactly this: for each word (the query), it looks at all other words (the keys), decides which ones are relevant (attention weights), and blends their meanings together (the weighted sum of values). This produces a contextualized representation where each word's meaning is refined by its context. In the first layer of a transformer, self-attention captures local syntax (like subject-verb agreement). In deeper layers, it captures higher-level semantics (like coreference resolution — linking "he" to "John" earlier in the text).

## Why This Concept Matters

Self-attention is the foundational innovation of the transformer architecture. It replaced recurrent neural networks for sequence processing, enabling three key advantages: (1) Parallelization: Unlike RNNs which process tokens sequentially, self-attention computes all pairwise interactions in parallel, dramatically speeding up training. (2) Long-range dependencies: Self-attention has O(1) path length between any two positions (compared to O(T) in RNNs), making it much better at capturing long-range relationships. (3) Interpretability: Self-attention weights provide direct insight into which tokens influence each other. Self-attention is the core building block of all modern large language models (BERT, GPT, T5, LLaMA, Claude) and is increasingly used in computer vision (ViT), speech recognition, and scientific applications. Understanding self-attention is essential for working with any modern deep learning model.

## Mathematical Explanation

### Self-Attention Computation

Given input X in R^{batch x T x d_model}:

1. Compute Q, K, V:
   Q = X W^Q, W^Q in R^{d_model x d_k}
   K = X W^K, W^K in R^{d_model x d_k}
   V = X W^V, W^V in R^{d_model x d_v}

2. Compute attention scores:
   S = Q K^T / sqrt(d_k), S in R^{batch x T x T}

3. Apply softmax:
   A = softmax(S, dim=-1), A in R^{batch x T x T}

4. Weighted sum of values:
   O = A V, O in R^{batch x T x d_v}

5. Output projection (if d_v != d_model):
   O' = O W^O, W^O in R^{d_v x d_model}

### Multi-Head Self-Attention

h heads compute self-attention in parallel:

head_i = Attention(X W_i^Q, X W_i^K, X W_i^V)
MultiHead(X) = Concat(head_1, ..., head_h) W^O

### Positional Encoding

Since self-attention is permutation-invariant (no inherent notion of order), positional encodings are added to input:

X' = X + PE

where PE can be sinusoidal (fixed) or learned.

### Self-Attention vs. Recurrence

| Aspect | Self-Attention | RNN |
|--------|---------------|-----|
| Path length | O(1) | O(T) |
| Parallelization | Full | Sequential |
| Memory | O(T^2) | O(T) |
| Long-range dependencies | Excellent | Poor |

## Code Examples

### Example 1: Basic Self-Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)
        self.scale = math.sqrt(d_k)

    def forward(self, x, mask=None):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

d_model, d_k, d_v = 16, 8, 8
sa = SelfAttention(d_model, d_k, d_v)
x = torch.randn(2, 6, d_model)
output, weights = sa(x)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"Self-attention matrix (batch 0):\n{weights[0].round(decimals=2)}")
# Output: Output shape: torch.Size([2, 6, 8])
# Output: Attention weights shape: torch.Size([2, 6, 6])
# Output: Self-attention matrix (batch 0):
# Output: tensor([[0.20, 0.15, 0.18, 0.22, 0.13, 0.12],
# Output:         [0.12, 0.25, 0.18, 0.15, 0.17, 0.13],
# Output:         [0.15, 0.18, 0.22, 0.14, 0.19, 0.12],
# Output:         [0.13, 0.16, 0.14, 0.30, 0.12, 0.15],
# Output:         [0.14, 0.17, 0.19, 0.12, 0.23, 0.15],
# Output:         [0.11, 0.13, 0.12, 0.15, 0.14, 0.35]])
```

### Example 2: Multi-Head Self-Attention

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch, seq_len = x.shape[0], x.shape[1]
        Q = self.W_q(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.W_o(context), attn

mha = MultiHeadSelfAttention(d_model=64, n_heads=8)
x = torch.randn(2, 10, 64)
output, attn = mha(x)
print(f"Multi-head output: {output.shape}")
print(f"Attention shape (batch, heads, seq, seq): {attn.shape}")
# Output: Multi-head output: torch.Size([2, 10, 64])
# Output: Attention shape (batch, heads, seq, seq): torch.Size([2, 8, 10, 10])
```

### Example 3: Self-Attention with Positional Encoding

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

d_model, n_heads, d_ff = 32, 4, 128
pe = PositionalEncoding(d_model)
layer = TransformerEncoderLayer(d_model, n_heads, d_ff)
x = torch.randn(2, 8, d_model)
x = pe(x)
output = layer(x)
print(f"Transformer encoder layer output: {output.shape}")
# Output: Transformer encoder layer output: torch.Size([2, 8, 32])
```

## Common Mistakes

1. **Forgetting positional encoding**: Self-attention is permutation-invariant — without positional encoding, the model cannot distinguish "cat sat on mat" from "mat sat on cat." Positional encodings are essential for sequence tasks.

2. **Not masking future tokens in decoder self-attention**: In autoregressive models, decoder self-attention must use causal masking to prevent each token from attending to future tokens. This is essential for maintaining the autoregressive property.

3. **Using the same d_k for all heads without proper division**: In multi-head attention, d_k = d_model / n_heads must be an integer. Each head operates in a lower-dimensional space, and the total computation equals the single-head case.

4. **Ignoring the quadratic memory cost**: Self-attention stores a T x T attention matrix. For long sequences (e.g., 100K tokens), this is prohibitively expensive. Sparse attention patterns or linear attention mechanisms are needed.

5. **Applying softmax on the wrong dimension**: Softmax should be applied over the key dimension (dim=-1) so that each query attends to all keys with weights summing to 1. Applying softmax over the wrong dimension breaks the attention mechanism.

## Interview Questions

### Beginner

Q: What is self-attention and how does it differ from cross-attention?

A: Self-attention computes attention over the same sequence: queries, keys, and values all come from the same source. Cross-attention computes attention between two different sequences: queries come from one sequence, keys and values from another. Self-attention is used for contextualizing tokens within a sequence; cross-attention is used for relating two sequences.

### Intermediate

Q: Why are positional encodings necessary in self-attention when they weren't needed in RNNs?

A: Self-attention is permutation-invariant — it treats the input as a set, not a sequence. Without positional encodings, permuting the input tokens produces the same output, which is incorrect for most sequence tasks. RNNs inherently process tokens in order (sequential computation), so they naturally encode position. Transformers add positional encodings to inject order information explicitly.

### Advanced

Q: Self-attention has O(T^2) time and memory complexity. Describe at least three approaches to reduce this complexity for long sequences, including their trade-offs.

A: (1) Sparse attention patterns (e.g., Longformer, BigBird): use fixed patterns like sliding window, dilated sliding window, and global tokens. Reduces complexity to O(T * w) where w is window size. Trade-off: may miss long-range dependencies outside the pattern. (2) Linear attention (e.g., Performer, Linformer): approximate the softmax attention kernel using kernel methods or low-rank projections, reducing complexity to O(T). Trade-off: approximation error can reduce quality for fine-grained attention. (3) Hierarchical attention (e.g., Reformer): use locality-sensitive hashing to group tokens into buckets, computing attention only within buckets. Trade-off: approximation that may not capture cross-bucket dependencies. (4) Memory compression (e.g., Compressive Transformer): compress past memories into compressed representations, attending to both detailed and compressed memories. Trade-off: loss of detail in compressed representations.

## Practice Problems

### Easy

Implement a single-head self-attention module. Given an input tensor of shape (batch=4, seq=8, d_model=32), compute the self-attention output and verify the output shape.

### Medium

Implement multi-head self-attention with 4 heads. Show that the output has the same dimension as the input (d_model). Compare the attention patterns across different heads and show they capture different relationships.

### Hard

Implement a causal self-attention module for autoregressive decoding. The attention mask should prevent each position from attending to future positions. Compare the outputs of causal vs. full self-attention on a simple sequence.

## Solutions

### Easy Solution

```python
def single_head_self_attention(x):
    d_k = x.shape[-1]
    W_q = nn.Linear(d_k, d_k, bias=False)
    W_k = nn.Linear(d_k, d_k, bias=False)
    W_v = nn.Linear(d_k, d_k, bias=False)
    Q, K, V = W_q(x), W_k(x), W_v(x)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V)

x = torch.randn(4, 8, 32)
out = single_head_self_attention(x)
print(f"Output shape: {out.shape}")
# Output: Output shape: torch.Size([4, 8, 32])
```

## Related Concepts

- Multi-Head Attention
- Positional Encoding
- Transformer Architecture
- Causal Masked Attention
- Cross-Attention

## Next Concepts

- DL-344: Cross-Attention
- DL-345: Causal Masked Attention
- DL-346: Multi-Head Attention

## Summary

Self-attention is a mechanism that relates different positions within a single sequence to compute contextualized representations. It computes pairwise attention between all tokens in the sequence, allowing each token to gather context from the entire sequence regardless of distance. Self-attention is the core building block of transformers, replacing recurrence and convolution for sequence processing. Its advantages include full parallelization, excellent long-range dependency modeling, and interpretable attention weights. Its main limitation is quadratic complexity in sequence length. Self-attention with positional encoding, multi-head computation, and proper masking forms the foundation of all modern large language models.

## Key Takeaways

- Self-attention relates all positions within a single sequence to compute contextualized representations.
- Queries, keys, and values all come from the same input sequence.
- Self-attention has O(1) path length between any two positions (excellent for long-range dependencies).
- It can be fully parallelized across positions (unlike RNNs).
- Positional encodings are required because self-attention is permutation-invariant.
- Multi-head self-attention captures different types of relationships in different heads.
- Quadratic memory cost (O(T^2)) is the main limitation for long sequences.
