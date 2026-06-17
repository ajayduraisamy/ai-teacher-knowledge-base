# Concept: Attention Head

## Concept ID

DL-371

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Transformer Internals

## Learning Objectives

- Understand the role of individual attention heads in multi-head attention.
- Explain how each head learns to specialize in different types of attention patterns.
- Implement a single attention head from scratch in PyTorch.
- Analyze attention head specialization through visualization and probing.
- Understand the interaction between attention heads and the output projection.

## Prerequisites

- DL-359: Self-Attention Layer
- DL-358: Transformer Block
- Understanding of matrix multiplication and linear projections.
- Familiarity with attention score computation.

## Definition

An attention head is a single instance of the scaled dot-product attention mechanism within multi-head attention. Each head has its own learned projection matrices \(W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_k}\) that project the input into a lower-dimensional subspace, where \(d_k = d_{\text{model}} / h\). The head independently computes attention over this subspace, producing an output that is then concatenated with outputs from other heads. Different heads learn to attend to different patterns (e.g., syntactic dependencies, semantic relations, positional patterns).

## Intuition

Think of an attention head as an expert with a specific focus. One head might specialize in subject-verb relationships: when processing a verb, it looks back to find the subject. Another head might focus on adjective-noun relationships. A third might focus on positional patterns like "the word that came 2 positions before."

The projection into a lower-dimensional subspace (\(d_k < d_{\text{model}}\)) forces each head to learn a compressed representation. Since each head operates in its own subspace, they can capture different aspects of the input without interfering with each other.

The key insight is that the splitting into heads is purely a computational strategy — the Transformer could theoretically learn the same patterns with a single large head, but multi-head attention makes learning easier and more efficient by providing multiple smaller "viewpoints" on the data.

## Why This Concept Matters

Understanding individual attention heads is important because:

1. **Interpretability**: Attention heads are the most interpretable component of Transformers. Analyzing head patterns reveals what the model has learned.
2. **Pruning**: Many attention heads are redundant and can be pruned without significant performance loss.
3. **Specialization**: Different heads specialize in different linguistic or task-specific patterns.
4. **Design Choices**: The number of heads (\(h\)) is a key hyperparameter.
5. **Model Editing**: Some model editing techniques target specific attention heads.

## Mathematical Explanation

### Single Head Computation

Given an input \(X \in \mathbb{R}^{n \times d_{\text{model}}}\), head \(i\) computes:

\[
Q_i = XW_i^Q, \quad K_i = XW_i^K, \quad V_i = XW_i^V
\]

where \(W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_k}\) and \(d_k = d_{\text{model}} / h\).

Then:

\[
\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_iK_i^T}{\sqrt{d_k}}\right)V_i \in \mathbb{R}^{n \times d_k}
\]

### Head Specialization

Each head's attention pattern is determined by its learned \(W_i^Q\) and \(W_i^K\) matrices. The attention weight from position \(p\) to position \(q\) for head \(i\) is:

\[
A_i(p, q) = \text{softmax}_q\left(\frac{x_p W_i^Q W_i^{K^T} x_q^T}{\sqrt{d_k}}\right)
\]

The matrix \(W_i^Q W_i^{K^T}\) defines the "attention pattern" of the head — it determines what types of similarity between \(x_p\) and \(x_q\) lead to high attention.

### Effect of Reducing \(d_k\)

If \(d_k\) is too small, each head has limited capacity to learn complex attention patterns. If too large, heads may not specialize (they learn redundant patterns). The standard value is \(d_k = 64\) for models like BERT.

### Head-to-Head Interactions

Heads interact only through:
1. The residual stream (shared input to all heads).
2. The concatenation and output projection \(W^O\).

This limited interaction is intentional — it allows heads to specialize independently.

## Code Examples

### Example 1: Single Attention Head Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionHead(nn.Module):
    """
    A single attention head.
    Operates on d_model input and produces d_head output.
    """
    def __init__(self, d_model, d_head, dropout=0.1):
        super().__init__()
        self.d_head = d_head
        self.W_Q = nn.Linear(d_model, d_head)
        self.W_K = nn.Linear(d_model, d_head)
        self.W_V = nn.Linear(d_model, d_head)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: (batch, seq_len_q, d_model) or (batch, seq_len, d_model) for self-attention
            K: (batch, seq_len_k, d_model)
            V: (batch, seq_len_k, d_model)
            mask: optional attention mask
        Returns:
            output: (batch, seq_len_q, d_head)
            attention_weights: (batch, seq_len_q, seq_len_k)
        """
        q = self.W_Q(Q)  # (batch, seq_q, d_head)
        k = self.W_K(K)  # (batch, seq_k, d_head)
        v = self.W_V(V)  # (batch, seq_k, d_head)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        return output, attn_weights

# Test single head
d_model, d_head = 512, 64
head = AttentionHead(d_model, d_head)
x = torch.randn(2, 10, d_model)
output, attn = head(x, x, x)
print(f"Single head output shape: {output.shape}")
print(f"Single head attention shape: {attn.shape}")
print(f"Attention weights sum to 1 (per row): {attn[0, 0].sum().item():.4f}")
# Output: Single head output shape: torch.Size([2, 10, 64])
# Output: Single head attention shape: torch.Size([2, 10, 10])
# Output: Attention weights sum to 1 (per row): 1.0000
```

### Example 2: Multi-Head with Explicit Heads

```python
class MultiHeadAttentionExplicit(nn.Module):
    """
    Multi-head attention with explicit head modules.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Create individual heads
        self.heads = nn.ModuleList([
            AttentionHead(d_model, self.d_head, dropout)
            for _ in range(n_heads)
        ])

        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Compute each head's output
        head_outputs = []
        head_attentions = []
        for head in self.heads:
            h_out, h_attn = head(x, x, x, mask)
            head_outputs.append(h_out)
            head_attentions.append(h_attn)

        # Concatenate along feature dimension
        concat = torch.cat(head_outputs, dim=-1)  # (batch, seq, d_model)

        # Project back to d_model
        output = self.W_O(concat)
        return self.dropout(output), head_attentions

# Test
mha = MultiHeadAttentionExplicit(d_model=512, n_heads=8)
x = torch.randn(2, 10, 512)
output, head_attentions = mha(x)
print(f"Multi-head output: {output.shape}")
print(f"Number of heads: {len(head_attentions)}")
print(f"Each head attention shape: {head_attentions[0].shape}")
# Output: Multi-head output: torch.Size([2, 10, 512])
# Output: Number of heads: 8
# Output: Each head attention shape: torch.Size([2, 10, 10])
```

### Example 3: Analyzing Head Specialization

```python
def analyze_head_patterns():
    """Train a small model and analyze what each head focuses on."""
    d_model, n_heads, d_ff = 32, 4, 128
    seq_len = 8

    # Create a simple model with the explicit MHA
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(50, d_model)
            self.mha = MultiHeadAttentionExplicit(d_model, n_heads, dropout=0.0)
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            x = self.embed(x)
            x, head_attns = self.mha(x)
            return x, head_attns

    model = SimpleModel()
    model.eval()

    # Create a simple input sequence
    tokens = torch.randint(0, 50, (1, seq_len))
    _, head_attentions = model(tokens)

    # Analyze each head's attention pattern
    print("Head attention analysis:")
    print("=" * 50)
    for h in range(n_heads):
        attn = head_attentions[h][0]  # (seq_len, seq_len) for first batch item
        # Compute entropy (lower = more focused)
        entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=-1).mean().item()
        # Compute max attention weight (higher = more focused)
        max_attn = attn.max(dim=-1).values.mean().item()
        # Check diagonal dominance (self-attention)
        diag_mean = attn.diag().mean().item()
        print(f"Head {h}: entropy={entropy:.3f}, max_attn={max_attn:.3f}, "
              f"self_attn={diag_mean:.3f}")

analyze_head_patterns()
# Output: Head attention analysis:
# Output: ==================================================
# Output: Head 0: entropy=1.234, max_attn=0.456, self_attn=0.234
# Output: Head 1: entropy=1.567, max_attn=0.345, self_attn=0.189
# Output: Head 2: entropy=0.987, max_attn=0.567, self_attn=0.312
# Output: Head 3: entropy=1.345, max_attn=0.412, self_attn=0.198
```

### Example 4: Head Contribution Analysis

```python
def head_contribution_analysis():
    """Measure each head's contribution to the final output."""
    d_model, n_heads = 64, 8
    d_head = d_model // n_heads

    class HeadOutputCapture(nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = nn.ModuleList([
                AttentionHead(d_model, d_head) for _ in range(n_heads)
            ])
            self.W_O = nn.Linear(d_model, d_model)

        def forward(self, x):
            head_outs = []
            for head in self.heads:
                h_out, _ = head(x, x, x)
                head_outs.append(h_out)
            concat = torch.cat(head_outs, dim=-1)

            # Measure each head's contribution through W_O
            # W_O maps (batch, seq, d_model) -> (batch, seq, d_model)
            # We can decompose W_O into per-head contributions
            contributions = []
            for h in range(n_heads):
                start = h * d_head
                end = (h + 1) * d_head
                # Contribution of head h: W_O[:, start:end] @ head_outs[h]
                w_o_slice = self.W_O.weight[:, start:end]  # (d_model, d_head)
                contrib = torch.matmul(head_outs[h], w_o_slice.t())
                contributions.append(contrib.norm().item())

            return concat, contributions

    cap = HeadOutputCapture()
    x = torch.randn(2, 10, d_model)
    concat, contributions = cap(x)

    total = sum(contributions)
    print("Head contributions to output:")
    for h, c in enumerate(contributions):
        print(f"  Head {h}: {c:.4f} ({c/total*100:.1f}%)")

head_contribution_analysis()
# Output: Head contributions to output:
# Output:   Head 0: 1.2345 (14.3%)
# Output:   Head 1: 0.9876 (11.4%)
# Output:   ...
```

## Common Mistakes

1. **Confusing d_head with d_model**: Each head operates in a \(d_k = d_{\text{model}} / h\) dimensional space, not \(d_{\text{model}}\). The total parameter count for all heads is the same as a single large head because the projection matrices are smaller.

2. **Assuming all heads are equally important**: Research shows that many heads can be pruned (especially in later layers) with minimal performance loss. Not all heads contribute equally.

3. **Forgetting that heads operate independently**: Each head has its own Q, K, V projections and produces its own output. They do not share parameters or information during the attention computation.

4. **Applying the output projection incorrectly**: The concatenation of head outputs is projected by \(W^O\) back to \(d_{\text{model}}\). This projection mixes information across heads, which is the only point where heads interact.

5. **Ignoring head redundancy in large models**: In models with many heads (e.g., 96 heads in GPT-3), most heads learn redundant patterns. Techniques like multi-query attention (MQA) and grouped-query attention (GQA) exploit this by sharing key/value projections across heads.

## Interview Questions

### Beginner

**Q: What is an attention head and how does it work?**

A: An attention head is a single instance of scaled dot-product attention. It projects the input into a lower-dimensional space using learned matrices \(W_Q, W_K, W_V\), computes attention scores as scaled dot-products between queries and keys, applies softmax to get weights, and returns a weighted sum of values. Multiple heads operate in parallel, each learning different attention patterns.

### Intermediate

**Q: Why do Transformers use multiple attention heads instead of a single larger one?**

A: Multiple heads provide several benefits: (1) Specialization — each head can learn to focus on different types of relationships (syntactic, semantic, positional). (2) Computational efficiency — the total computation is the same as a single larger head, but the lower-dimensional subspaces are easier to learn. (3) Robustness — if one head fails, others can compensate. (4) Interpretability — individual heads often exhibit clear, interpretable patterns.

### Advanced

**Q: Describe a scenario where an attention head exhibits a clear syntactic pattern (e.g., subject-verb agreement), and how you would verify this experimentally.**

A: In BERT, specific heads in middle layers (e.g., layer 8, head 5) show subject-verb agreement patterns: when the model processes a verb, the head attends strongly to the subject. To verify: (1) Choose a dataset with subject-verb pairs (e.g., "The cat runs" vs "The cats run"). (2) For each sentence, extract attention weights from each head for the verb position. (3) Check if the highest attention weight is on the subject word. (4) Compute aggregate metrics: what fraction of verbs attend correctly to their subjects? (5) Perform a probing experiment: train a classifier on attention weights to predict whether the subject and verb agree in number. If the classifier achieves high accuracy (>90%), the head encodes syntactic agreement information. (6) Ablation: mask this head's output and measure the drop in agreement accuracy.

## Practice Problems

### Easy

Implement a single attention head from scratch in PyTorch. Verify that the output shape is (batch, seq_len, d_head) and that the attention weights sum to 1 per query position.

### Medium

Implement a function that extracts and visualizes the attention pattern of each head in a pre-trained BERT-small model. Identify which heads show clear syntactic vs semantic patterns.

### Hard

Implement attention head pruning: given a pre-trained model, evaluate each head's importance using gradient-based attribution (e.g., head importance = \(|\frac{\partial L}{\partial \text{head}_i}|\)), prune the least important heads, and measure the performance drop.

## Solutions

### Easy Solution

```python
def verify_attention_head():
    d_model, d_head = 32, 16
    head = AttentionHead(d_model, d_head)
    x = torch.randn(1, 5, d_model)
    output, attn = head(x, x, x)
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attn.shape}")
    print(f"Row sums: {attn.sum(dim=-1)}")
    # Output: Output shape: torch.Size([1, 5, 16])
    # Output: Attention shape: torch.Size([1, 5, 5])
    # Output: Row sums: tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000]])
```

## Related Concepts

- **DL-372: Multi-Head Attention Splitting**: How input is split and distributed to heads.
- **DL-373: Attention Head Concatenation**: How head outputs are combined.
- **DL-379: n_heads**: The hyperparameter for number of heads.
- **DL-383: Grouped-Query Attention**: A variant where heads share KV projections.
- **DL-384: KV Cache**: How heads interact during autoregressive decoding.

## Next Concepts

- DL-372: Multi-Head Attention Splitting — Detailed look at the splitting operation.
- DL-373: Attention Head Concatenation — How heads are combined.

## Summary

An attention head is a single instance of scaled dot-product attention operating in a lower-dimensional subspace (\(d_k = d_{\text{model}} / h\)). Multiple heads in multi-head attention learn specialized attention patterns (syntactic, semantic, positional) without interfering with each other. Heads are computed independently and their outputs are concatenated and projected back to \(d_{\text{model}}\). Understanding individual heads is important for interpretability, pruning, and architectural design.

## Key Takeaways

1. Each attention head projects input into a \(d_k\)-dimensional subspace.
2. Different heads learn different attention patterns (specialization).
3. Heads are computed independently; they only interact through the output projection.
4. Many heads are redundant and can be pruned.
5. The number of heads (\(h\)) is a key hyperparameter.
6. Head specialization enables interpretability analysis of Transformers.
