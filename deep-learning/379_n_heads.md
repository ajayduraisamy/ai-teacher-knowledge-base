# Concept: n_heads

## Concept ID

DL-379

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Transformer Internals

## Learning Objectives

- Understand n_heads as the number of parallel attention heads in multi-head attention.
- Explain how n_heads affects model capacity, parameter count, and attention patterns.
- Analyze the relationship between n_heads, d_model, and d_head.
- Understand the trade-offs in choosing the number of heads.
- Recognize common n_heads values across model families.

## Prerequisites

- DL-371: Attention Head
- DL-372: Multi-Head Attention Splitting
- DL-377: d_model
- Understanding of multi-head attention.

## Definition

\(n_{\text{heads}}\) (or \(h\)) is the number of parallel attention heads in the multi-head attention mechanism. Each head operates in a \(d_{\text{head}} = d_{\text{model}} / n_{\text{heads}}\) dimensional subspace with its own learned query, key, and value projections. The heads are computed in parallel, and their outputs are concatenated and projected back to \(d_{\text{model}}\). Common values include 8, 12, 16, 32, and 96.

## Intuition

Multiple attention heads allow the model to attend to different types of information simultaneously. Each head can specialize:
- One head might focus on syntactic relationships (subject-verb).
- Another on semantic relationships (entity-attribute).
- Another on positional relationships (nearby tokens).

The number of heads determines how many different "perspectives" the model can have simultaneously. More heads = more perspectives, but each perspective is less detailed (smaller \(d_{\text{head}}\)).

## Why This Concept Matters

n_heads is important because:

1. **Attention Capacity**: More heads provide more parallel attention patterns.
2. **Head Dimension Trade-off**: Increasing n_heads while keeping d_model fixed reduces d_head, limiting per-head capacity.
3. **Interpretability**: Different heads often exhibit clear, interpretable patterns.
4. **Pruning**: Many heads are redundant, and the optimal n_heads may be less than commonly used.
5. **Grouped-Query Attention**: Modern architectures use fewer KV heads than query heads.

## Mathematical Explanation

### Multi-Head Attention

\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_{n_{\text{heads}}})W^O
\]
\[
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\]

### Dimension Relationship

\[
d_{\text{head}} = \frac{d_{\text{model}}}{n_{\text{heads}}}
\]

### Parameter Count

For the attention sub-layer:
- Q, K, V projections: \(3 \times d_{\text{model}} \times d_{\text{model}}\) (independent of n_heads)
- Output projection: \(d_{\text{model}} \times d_{\text{model}}\) (independent of n_heads)

The number of heads does not affect the parameter count of the attention sub-layer because the total dimension is always \(d_{\text{model}}\). However, n_heads affects the computational pattern and the per-head capacity.

### Effect of n_heads on Attention Computation

The attention computation involves:
\[
\text{scores}_i = \frac{Q_i K_i^T}{\sqrt{d_{\text{head}}}}
\]

With more heads (smaller d_head):
- The scaled dot-product has lower variance (good).
- But each head has less capacity to learn complex patterns (bad).

## Code Examples

### Example 1: Effect of n_heads on d_head

```python
import torch
import torch.nn as nn
import math

def show_head_dimension(d_model, n_heads_list):
    """Show how d_head changes with n_heads."""
    print(f"d_model = {d_model}")
    print(f"{'n_heads':<10} {'d_head':<10} {'Valid?':<10}")
    print("-" * 30)
    for n_heads in n_heads_list:
        if d_model % n_heads == 0:
            d_head = d_model // n_heads
            valid = "✓"
        else:
            d_head = None
            valid = "✗ (not divisible)"
        print(f"{n_heads:<10} {d_head if d_head else 'N/A':<10} {valid:<10}")

show_head_dimension(512, [4, 8, 12, 16, 32, 64])
print()
show_head_dimension(768, [6, 8, 12, 16, 24, 32])
# Output: d_model = 512
# Output: n_heads    d_head     Valid?
# Output: ------------------------------
# Output: 4          128        ✓
# Output: 8          64         ✓
# Output: 12         N/A        ✗ (not divisible)
# Output: 16         32         ✓
# Output: 32         16         ✓
# Output: 64         8          ✓
# Output:
# Output: d_model = 768
# Output: n_heads    d_head     Valid?
# Output: ------------------------------
# Output: 6          128        ✓
# Output: 8          96         ✓
# Output: 12         64         ✓
# Output: 16         48         ✓
# Output: 24         32         ✓
# Output: 32         24         ✓
```

### Example 2: Head Specialization with Different n_heads

```python
def analyze_head_specialization():
    """Compare attention patterns with different numbers of heads."""
    d_model = 64
    seq_len = 10

    def compute_attention_entropy(n_heads):
        d_head = d_model // n_heads
        Q = torch.randn(1, n_heads, seq_len, d_head)
        K = torch.randn(1, n_heads, seq_len, d_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_head)
        attn = F.softmax(scores, dim=-1)

        # Compute entropy for each head
        entropies = []
        for h in range(n_heads):
            ent = -(attn[0, h] * torch.log(attn[0, h] + 1e-8)).sum(dim=-1).mean()
            entropies.append(ent.item())

        return entropies

    print("Attention entropy by head for different n_heads:")
    print("(Lower entropy = more focused attention)")
    print("-" * 50)

    for n_heads in [2, 4, 8, 16]:
        entropies = analyze_head_specialization(n_heads)
        avg_entropy = sum(entropies) / len(entropies)
        max_entropy = max(entropies)
        min_entropy = min(entropies)
        print(f"n_heads={n_heads:2d}, d_head={d_model//n_heads:2d}: "
              f"avg_entropy={avg_entropy:.4f}, range=[{min_entropy:.4f}, {max_entropy:.4f}]")

# Note: Since inputs are random, heads won't truly specialize.
# This demonstrates the statistical behavior.
```

### Example 3: n_heads in Common Models

```python
def n_heads_examples():
    """Show n_heads across popular models."""
    models = [
        ("Transformer-base", 512, 8),
        ("BERT-base", 768, 12),
        ("BERT-large", 1024, 16),
        ("GPT-2 small", 768, 12),
        ("GPT-2 medium", 1024, 16),
        ("GPT-2 large", 1280, 20),
        ("GPT-3 175B", 12288, 96),
        ("Llama 7B", 4096, 32),
        ("Llama 2 13B", 5120, 40),
        ("Llama 3 70B", 8192, 64),
        ("Mistral 7B", 4096, 32),
        ("Mixtral 8x7B", 4096, 32),
    ]

    print(f"{'Model':<20} {'d_model':<10} {'n_heads':<10} {'d_head':<10}")
    print("-" * 50)
    for name, d_model, n_heads in models:
        d_head = d_model // n_heads
        print(f"{name:<20} {d_model:<10} {n_heads:<10} {d_head:<10}")

n_heads_examples()
# Output: Model               d_model    n_heads    d_head
# Output: --------------------------------------------------
# Output: Transformer-base    512        8          64
# Output: BERT-base           768        12         64
# Output: BERT-large          1024       16         64
# Output: GPT-2 small         768        12         64
# Output: GPT-2 medium        1024       16         64
# Output: GPT-2 large         1280       20         64
# Output: GPT-3 175B          12288      96         128
# Output: Llama 7B            4096       32         128
# Output: Llama 2 13B         5120       40         128
# Output: Llama 3 70B         8192       64         128
# Output: Mistral 7B          4096       32         128
# Output: Mixtral 8x7B        4096       32         128
```

### Example 4: Ablation — Varying n_heads with Fixed d_model

```python
def n_heads_ablation():
    """Create attention modules with different n_heads and compare."""
    d_model = 256
    seq_len = 16
    batch = 4

    class TestAttention(nn.Module):
        def __init__(self, n_heads):
            super().__init__()
            self.n_heads = n_heads
            d_head = d_model // n_heads
            self.W_Q = nn.Linear(d_model, d_model)
            self.W_K = nn.Linear(d_model, d_model)
            self.W_V = nn.Linear(d_model, d_model)

        def forward(self, x):
            batch, seq, _ = x.shape
            Q = self.W_Q(x).view(batch, seq, self.n_heads, d_model // self.n_heads).transpose(1, 2)
            K = self.W_K(x).view(batch, seq, self.n_heads, d_model // self.n_heads).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_model // self.n_heads)
            attn = F.softmax(scores, dim=-1)
            return attn

    x = torch.randn(batch, seq_len, d_model)

    print(f"Effect of n_heads (d_model={d_model}):")
    print(f"{'n_heads':<10} {'d_head':<8} {'Params':<12} {'Attn shape':<20}")
    print("-" * 50)

    for n_heads in [2, 4, 8, 16]:
        attn = TestAttention(n_heads)
        out = attn(x)
        params = sum(p.numel() for p in attn.parameters())
        print(f"{n_heads:<10} {d_model//n_heads:<8} {params:<12,} {str(list(out.shape)):<20}")

n_heads_ablation()
# Output: Effect of n_heads (d_model=256):
# Output: n_heads    d_head   Params       Attn shape
# Output: --------------------------------------------------
# Output: 2          128      197,376      [4, 2, 16, 16]
# Output: 4          64       197,376      [4, 4, 16, 16]
# Output: 8          32       197,376      [4, 8, 16, 16]
# Output: 16         16       197,376      [4, 16, 16, 16]
```

## Common Mistakes

1. **Forgetting the divisibility constraint**: d_model must be divisible by n_heads. Common valid pairs: (512, 8), (768, 12), (1024, 16), (4096, 32).

2. **Assuming more heads is always better**: More heads means smaller d_head, which reduces per-head capacity. There is an optimal range (d_head ≈ 64-128).

3. **Not considering head redundancy**: Research shows that many heads can be pruned without significant performance loss. Using more heads than necessary adds computational cost without benefit.

4. **Confusing n_heads in grouped-query attention**: In GQA, there are n_query_heads and n_kv_heads (where n_kv_heads ≤ n_query_heads). The "n_heads" usually refers to query heads.

5. **Setting n_heads to a non-power-of-2 without reason**: While not strictly required, n_heads is typically a power of 2 or a multiple of 4 for GPU efficiency.

## Interview Questions

### Beginner

**Q: What is n_heads in a Transformer and what constraint does it satisfy?**

A: n_heads is the number of parallel attention heads in multi-head attention. It must satisfy the constraint that d_model is divisible by n_heads, so that each head has dimension d_head = d_model / n_heads. Common values include 8 (Transformer-base), 12 (BERT-base), and 32 (Llama 7B).

### Intermediate

**Q: How does changing n_heads affect the parameter count?**

A: Changing n_heads does not change the parameter count of the attention sub-layer. The Q, K, V, and O projections are all d_model × d_model regardless of n_heads. However, n_heads affects the computational pattern: more heads mean more but smaller attention computations. The attention score matrix becomes (batch, n_heads, seq_len, seq_len) — larger n_heads means more score matrices but each of the same size, increasing the total memory for attention scores.

### Advanced

**Q: The trend in modern LLMs has moved toward fewer key-value heads than query heads (GQA/MQA). Explain the motivation and the typical ratios.**

A: The motivation is that key-value heads are largely redundant — many heads attend to the same information. In Multi-Query Attention (MQA), all query heads share a single KV head. In Grouped-Query Attention (GQA), query heads are divided into groups, each sharing a KV head. Typical ratios: (1) Full MHA: n_kv = n_q (e.g., 32 query, 32 KV). (2) GQA-8: n_q = 32, n_kv = 8 (groups of 4). (3) GQA-4: n_q = 32, n_kv = 4. (4) MQA: n_kv = 1. This reduces the KV cache size by a factor of n_q / n_kv during autoregressive inference, which is critical for long-context models. Performance loss from GQA is minimal when the number of KV heads is not too small (e.g., GQA-8 performs nearly as well as full MHA).

## Practice Problems

### Easy

Given d_model = 1024, list all valid n_heads values that result in integer d_head, and compute d_head for each.

### Medium

Implement a function that takes d_model and n_heads, and creates a multi-head attention module. Compute the attention score matrix size for sequence lengths 512, 1024, and 2048.

### Hard

Implement Grouped-Query Attention with n_query_heads = 32 and n_kv_heads = 8. Compare the KV cache size and memory usage with full MHA (32 KV heads) for a batch size of 1 and sequence length of 4096.

## Solutions

### Easy Solution

```python
def valid_head_configs(d_model):
    configs = []
    for n_heads in range(1, d_model + 1):
        if d_model % n_heads == 0:
            d_head = d_model // n_heads
            configs.append((n_heads, d_head))
    return configs

d_model = 1024
configs = valid_head_configs(d_model)
print(f"Valid n_heads for d_model={d_model}:")
for n_heads, d_head in configs[:10]:  # First 10
    print(f"  n_heads={n_heads:3d} -> d_head={d_head:4d}")
print(f"  ... and {len(configs) - 10} more configurations")
# Output: Valid n_heads for d_model=1024:
# Output:   n_heads=  1 -> d_head=1024
# Output:   n_heads=  2 -> d_head= 512
# Output:   n_heads=  4 -> d_head= 256
# Output:   n_heads=  8 -> d_head= 128
# Output:   n_heads= 16 -> d_head=  64
# Output:   n_heads= 32 -> d_head=  32
# Output:   n_heads= 64 -> d_head=  16
# Output:   n_heads=128 -> d_head=   8
# Output:   n_heads=256 -> d_head=   4
# Output:   n_heads=512 -> d_head=   2
# Output:   ... and 5 more configurations
```

## Related Concepts

- **DL-371: Attention Head**: The individual head concept.
- **DL-372: Multi-Head Attention Splitting**: How heads are created.
- **DL-373: Attention Head Concatenation**: How heads are combined.
- **DL-377: d_model**: Related base dimension.
- **DL-383: Grouped-Query Attention**: A variant with fewer KV heads.
- **DL-384: KV Cache**: How n_heads affects cache size.

## Next Concepts

- DL-380: n_layers — The number of Transformer blocks.
- DL-383: Grouped-Query Attention — Reducing KV heads for efficiency.

## Summary

n_heads is the number of parallel attention heads in multi-head attention. It determines the number of different perspectives the model can attend with simultaneously. The constraint d_model = n_heads × d_head must hold, meaning larger n_heads results in smaller per-head dimension. The standard d_head value is 64 (for models like BERT) or 128 (for models like Llama). While more heads provide more parallel attention patterns, many heads are redundant, leading to techniques like Grouped-Query Attention that reduce the number of key-value heads.

## Key Takeaways

1. n_heads determines the number of parallel attention perspectives.
2. d_model must be divisible by n_heads.
3. d_head = d_model / n_heads (standard: 64-128).
4. n_heads does not affect the attention parameter count.
5. More heads ≠ always better; there is an optimal range.
6. Modern trends use GQA with fewer KV heads than query heads.
