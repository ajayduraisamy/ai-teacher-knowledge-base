# Concept: Multi-Head Attention

## Concept ID

DL-346

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Attention Mechanisms

## Learning Objectives

- Define the multi-head attention mechanism and explain why multiple heads are beneficial.
- Implement multi-head attention in PyTorch, including the projection and concatenation steps.
- Understand how different attention heads can learn to focus on different types of relationships.
- Analyze the computational properties of multi-head vs. single-head attention.
- Apply multi-head attention in transformer encoder and decoder layers.

## Prerequisites

- Solid understanding of scaled dot-product attention.
- Familiarity with the query-key-value framework.
- Knowledge of linear projections and matrix multiplication.
- Understanding of the transformer architecture basics.

## Definition

Multi-head attention is an extension of the attention mechanism that runs multiple attention operations (heads) in parallel, each with different learned linear projections of the queries, keys, and values. Given Q, K, V in R^{d_model}, multi-head attention computes:

head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

where:
- W_i^Q in R^{d_model x d_k}, W_i^K in R^{d_model x d_k}, W_i^V in R^{d_model x d_v}
- W^O in R^{h * d_v x d_model}
- h is the number of heads
- Typically d_k = d_v = d_model / h

Each head operates in a lower-dimensional space (d_k < d_model) and focuses on different aspects of the input. The concatenation of all heads is projected back to d_model, allowing the model to jointly attend to information from different representation subspaces. Multi-head attention was introduced in the Transformer paper (Vaswani et al., 2017) as a key component that improved performance over single-head attention.

## Intuition

Imagine you are analyzing a sentence from multiple perspectives simultaneously. One "head" might focus on syntactic relationships (subject-verb agreement), another on semantic roles (who did what to whom), another on positional relationships (nearby words vs. distant words), and yet another on discourse relationships (pronoun resolution). Each head looks at the sentence through a different lens (different linear projection), allowing them to specialize in different patterns. Multi-head attention is like having a team of analysts working in parallel, each with their own expertise, and then combining their insights to form a comprehensive understanding. The number of heads controls how many different relationship types the model can capture simultaneously. In practice, models with 8, 12, or 16 heads are common, and different heads indeed learn to focus on different linguistic phenomena.

## Why This Concept Matters

Multi-head attention is a critical component of the transformer architecture and is used in virtually all modern large language models. It provides several key benefits: (1) It allows the model to attend to information from different representation subspaces, capturing diverse relationship types. (2) It increases model capacity without a proportional increase in computational cost (since each head works in a lower-dimensional space). (3) It stabilizes training by averaging across multiple attention distributions, reducing the variance of individual heads. (4) It provides interpretability through analyzing which heads specialize in which patterns. Understanding multi-head attention is essential for implementing transformers, debugging attention-related issues, and designing efficient attention variants.

## Mathematical Explanation

### Single Head vs. Multi-Head

Single head: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Multi-head: splits the computation into h heads:

For i = 1, ..., h:
  Q_i = Q W_i^Q, where W_i^Q in R^{d_model x d_k}
  K_i = K W_i^K, where W_i^K in R^{d_model x d_k}
  V_i = V W_i^V, where W_i^V in R^{d_model x d_v}
  head_i = Attention(Q_i, K_i, V_i) in R^{batch x seq x d_v}

MultiHead = Concat(head_1, ..., head_h) W^O, where W^O in R^{h*d_v x d_model}

### Parameter Count

Single head: 3 * d_model * d_k + d_model * d_v (Q, K, V, output projections)
Multi-head (h heads): h * (3 * d_model * d_k) + d_model * (h * d_v)

With d_k = d_v = d_model / h, the total parameter count is the same as a single head with d_k = d_v = d_model.

Total params = h * 3 * d_model * (d_model/h) + d_model * d_model = 3 * d_model^2 + d_model^2 = 4 * d_model^2

### Computational Cost

Multi-head attention with d_k = d_model / h has the same FLOPs as a single-head attention with d_k = d_model. The projections and concatenation add negligible overhead.

### Head Specialization

Empirically, different heads learn different patterns:
- Local heads: focus on nearby tokens (syntax, n-gram patterns)
- Global heads: focus on content words across distances
- Specialized heads: focus on specific relationships (subject-verb, pronoun resolution)

## Code Examples

### Example 1: Multi-Head Attention Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch = q.shape[0]
        Q = self.W_q(q).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        return self.W_o(context), attn

d_model, n_heads = 64, 8
mha = MultiHeadAttention(d_model, n_heads)
q = torch.randn(2, 10, d_model)
k = torch.randn(2, 12, d_model)
v = torch.randn(2, 12, d_model)
output, attn = mha(q, k, v)
print(f"Multi-head output: {output.shape}")
print(f"Attention shape (batch, heads, q_len, k_len): {attn.shape}")
# Output: Multi-head output: torch.Size([2, 10, 64])
# Output: Attention shape (batch, heads, q_len, k_len): torch.Size([2, 8, 10, 12])
```

### Example 2: Analyzing Head Specialization

```python
def analyze_heads(attn_matrix):
    n_heads = attn_matrix.shape[1]
    head_stats = {}
    for h in range(n_heads):
        head_w = attn_matrix[:, h]
        entropy = -(head_w * torch.log(head_w + 1e-8)).sum(-1).mean().item()
        max_attn = head_w.max(-1).values.mean().item()
        diag_attn = torch.diagonal(head_w[0], dim1=-2, dim2=-1).mean().item() if head_w.shape[-1] == head_w.shape[-2] else 0
        head_stats[h] = {'entropy': entropy, 'max_attn': max_attn}
    return head_stats

mha.eval()
with torch.no_grad():
    x = torch.randn(4, 20, 64)
    _, attn = mha(x, x, x)
    stats = analyze_heads(attn)
    for h, s in stats.items():
        print(f"Head {h}: entropy={s['entropy']:.3f}, max_attn={s['max_attn']:.3f}")
# Output: Head 0: entropy=2.341, max_attn=0.123
# Output: Head 1: entropy=1.892, max_attn=0.312
# Output: Head 2: entropy=2.101, max_attn=0.175
# Output: Head 3: entropy=1.234, max_attn=0.523
# Output: Head 4: entropy=2.401, max_attn=0.098
# Output: Head 5: entropy=1.567, max_attn=0.401
# Output: Head 6: entropy=2.212, max_attn=0.134
# Output: Head 7: entropy=1.998, max_attn=0.234
```

### Example 3: Multi-Head Attention in Transformer Encoder

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
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
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model))
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        seq_len = x.shape[1]
        x = self.dropout(self.embedding(x) + self.pos_encoding[:, :seq_len])
        for layer in self.layers:
            x = layer(x, mask)
        return x

encoder = TransformerEncoder(
    vocab_size=1000, d_model=64, n_heads=4,
    d_ff=128, n_layers=3, max_len=50
)
x = torch.randint(0, 1000, (2, 20))
output = encoder(x)
print(f"Encoder output: {output.shape}")
# Output: Encoder output: torch.Size([2, 20, 64])
```

## Common Mistakes

1. **Incorrectly reshaping the tensor for multi-head attention**: The reshape from (batch, seq, d_model) to (batch, seq, n_heads, d_k) must be followed by transposing to (batch, n_heads, seq, d_k). Common mistakes include transposing the wrong axes or forgetting to transpose entirely.

2. **Using d_k that doesn't divide d_model evenly**: d_model must be divisible by n_heads. If not, tensor reshaping will fail. Always assert d_model % n_heads == 0.

3. **Forgetting the output projection W^O**: After concatenating heads, the result is projected back to d_model. Forgetting this projection leaves the output with dimension h * d_v instead of d_model.

4. **Sharing the same mask across all heads incorrectly**: While the mask shape is (batch, seq) and broadcasted across heads, different heads should NOT have different masks in standard implementations. However, some architectures use head-specific masking.

5. **Assuming all heads learn useful patterns**: In practice, some heads may become "dead" (uniform attention) or redundant. Analyzing head utilization and pruning uninformative heads can improve efficiency.

## Interview Questions

### Beginner

Q: What is multi-head attention and why is it better than single-head attention?

A: Multi-head attention runs multiple attention operations in parallel, each with different learned projections of queries, keys, and values. This allows the model to attend to information from different representation subspaces simultaneously, capturing diverse relationship types (syntactic, semantic, positional).

### Intermediate

Q: How does multi-head attention maintain the same computational cost as single-head attention?

A: In multi-head attention, each head operates in a lower-dimensional space: d_k = d_model / n_heads. The total computation across all heads equals the computation of a single head with dimension d_model: n_heads * (d_model/n_heads)^2 = d_model^2. The projections and concatenation add minimal overhead. So multi-head attention has approximately the same FLOPs as single-head attention.

### Advanced

Q: How would you determine the optimal number of heads for a given task? What factors influence this choice?

A: The optimal number of heads depends on: (1) Model size: larger d_model can support more heads. A common heuristic is d_model = 64 * n_heads (each head gets 64 dimensions). (2) Task complexity: tasks requiring diverse relationship types (e.g., translation with syntax and semantics) benefit from more heads. (3) Sequence length: longer sequences benefit from more heads because each head can specialize in a different positional range. (4) Computational budget: more heads increase the output projection cost and memory. Ablation studies show diminishing returns beyond 16-32 heads. Recent work shows that learned head pruning can identify and remove redundant heads, suggesting that the optimal number depends on the data and can be smaller than the maximum.

## Practice Problems

### Easy

Implement multi-head attention with 4 heads, d_model=32. Verify that the output has the same shape as the input (batch, seq, d_model).

### Medium

Analyze the attention patterns of different heads in a pre-trained transformer on a text classification task. Show that some heads focus on local context while others focus on global context.

### Hard

Implement a head-pruning variant of multi-head attention where heads with consistently uniform attention are identified and removed during training. Compare the efficiency and performance of the pruned model with the full model.

## Solutions

### Easy Solution

```python
class SimpleMultiHead(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)

    def forward(self, x):
        out, attn = self.mha(x, x, x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
        return out

mh = SimpleMultiHead(32, 4)
x = torch.randn(2, 8, 32)
out = mh(x)
print(f"Output matches input shape: {out.shape}")
# Output: Output matches input shape: torch.Size([2, 8, 32])
```

## Related Concepts

- Scaled Dot-Product Attention
- Transformer Architecture
- Head Specialization
- Attention Pruning
- Grouped Query Attention

## Next Concepts

- DL-347: Scaled Dot-Product Attention
- DL-348: Attention Temperature
- DL-352: Attention Is All You Need

## Summary

Multi-head attention is a key component of the transformer architecture that runs multiple attention operations in parallel, each with different learned linear projections. This allows the model to jointly attend to information from different representation subspaces, capturing diverse types of relationships. Despite having multiple heads, the computational cost is approximately equal to single-head attention because each head works in a lower-dimensional space. Multi-head attention improves model capacity, training stability, and provides interpretability through head-specific attention patterns.

## Key Takeaways

- Multi-head attention runs h parallel attention operations with different projections.
- d_k = d_model / n_heads, keeping total computation comparable to single-head attention.
- Different heads learn to focus on different types of relationships (syntax, semantics, position).
- Multi-head attention was introduced in the Transformer paper and is used in all modern LLMs.
- The output of all heads is concatenated and projected back to d_model.
- Head pruning can identify and remove redundant heads for efficiency.
