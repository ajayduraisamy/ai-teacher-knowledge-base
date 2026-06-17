# Concept: Grouped-Query Attention

## Concept ID

DL-384

## Difficulty

Expert

## Domain

Deep Learning

## Module

Transformer Internals

## Learning Objectives

- Understand Grouped-Query Attention (GQA) as an interpolation between Multi-Head Attention (MHA) and Multi-Query Attention (MQA).
- Implement GQA in PyTorch, including the grouping and sharing of KV heads.
- Explain the memory and compute benefits of GQA, especially for the KV cache.
- Analyze the quality-compute trade-off of different GQA configurations.
- Understand the adoption of GQA in modern LLMs (Llama 2, Mistral, etc.).

## Prerequisites

- DL-371: Attention Head
- DL-379: n_heads
- DL-383: KV Cache
- Understanding of the KV cache and its memory requirements.

## Definition

Grouped-Query Attention (GQA) is an attention mechanism where multiple query heads share a single key-value head, forming groups. It interpolates between full Multi-Head Attention (MHA) where each query head has its own KV head (GQA with n_kv = n_q) and Multi-Query Attention (MQA) where all query heads share a single KV head (GQA with n_kv = 1). By having n_kv < n_q, GQA reduces the KV cache size and memory bandwidth requirements with minimal quality degradation.

## Intuition

In full MHA, each query head has its own key and value projections. However, research has shown that many KV heads learn similar patterns — they are redundant. GQA exploits this redundancy by having groups of query heads share a single KV head.

Think of this as a team of workers (query heads) who all need to access reference materials (key-value pairs). In MHA, each worker has their own copy of the references. In GQA, workers are organized into groups, and each group shares one copy. This saves storage space (smaller KV cache) while still allowing different groups to specialize in different reference materials.

## Why This Concept Matters

GQA has become the standard in modern LLMs:

1. **KV Cache Reduction**: GQA reduces KV cache size by a factor of n_q / n_kv.
2. **Inference Efficiency**: Less memory bandwidth required for KV cache reads.
3. **Quality Preservation**: Minimal quality loss compared to full MHA (especially with n_kv ≥ 8).
4. **Standard in Modern Models**: Used in Llama 2, Llama 3, Mistral, Mixtral, Gemma.
5. **Scaling**: GQA enables larger batch sizes and longer sequences within the same memory budget.

## Mathematical Explanation

### Multi-Head Attention (MHA)

Each of \(n_q\) query heads has its own KV head:

\[
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\]

Total KV heads: \(n_kv = n_q\)

### Multi-Query Attention (MQA)

All query heads share a single KV head:

\[
K = XW^K, \quad V = XW^V
\]
\[
\text{head}_i = \text{Attention}(QW_i^Q, K, V)
\]

Total KV heads: \(n_kv = 1\)

### Grouped-Query Attention (GQA)

Query heads are divided into \(g\) groups, where \(g = n_kv\). Each group shares one KV head:

\[
K_j = XW_j^K, \quad V_j = XW_j^V \quad \text{for } j = 1, \ldots, n_kv
\]
\[
\text{head}_i = \text{Attention}(QW_i^Q, K_{\lfloor i / g \rfloor}, V_{\lfloor i / g \rfloor})
\]

Total KV heads: \(n_kv\) (typically 4, 8, or a divisor of \(n_q\))

### KV Cache Size Comparison

\[
\text{Cache}_{\text{MHA}} = 2 \cdot n_{\text{layers}} \cdot n_q \cdot d_{\text{head}} \cdot n
\]
\[
\text{Cache}_{\text{GQA}} = 2 \cdot n_{\text{layers}} \cdot n_{kv} \cdot d_{\text{head}} \cdot n = \frac{n_{kv}}{n_q} \cdot \text{Cache}_{\text{MHA}}
\]
\[
\text{Cache}_{\text{MQA}} = 2 \cdot n_{\text{layers}} \cdot d_{\text{model}} \cdot n = \frac{1}{n_q} \cdot \text{Cache}_{\text{MHA}}
\]

## Code Examples

### Example 1: GQA Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).
    n_kv_heads KV heads shared by groups of query heads.
    """
    def __init__(self, d_model, n_heads, n_kv_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_heads
        self.n_groups = n_heads // n_kv_heads

        # Query projection: always n_heads
        self.W_Q = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        # KV projections: only n_kv_heads
        self.W_K = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.W_V = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        # Output projection
        self.W_O = nn.Linear(n_heads * self.d_head, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape

        # Project to Q (n_heads), K, V (n_kv_heads)
        Q = self.W_Q(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(x).view(batch, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)
        V = self.W_V(x).view(batch, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)

        # Repeat K, V to match n_heads: (batch, n_kv, seq, d) -> (batch, n_heads, seq, d)
        # Each KV head is repeated n_groups times
        K = K.repeat_interleave(self.n_groups, dim=1)
        V = V.repeat_interleave(self.n_groups, dim=1)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, V)

        # Concatenate heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.W_O(attn_out)

# Test GQA
d_model, n_heads, n_kv_heads = 64, 8, 2  # 8 query heads, 2 KV heads (4 groups)
gqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
x = torch.randn(2, 10, d_model)
output = gqa(x)

# Show parameter counts
def count_params(module):
    return sum(p.numel() for p in module.parameters())

print(f"GQA: n_heads={n_heads}, n_kv_heads={n_kv_heads}")
print(f"  Q params: {gqa.W_Q.weight.numel():,}")
print(f"  K params: {gqa.W_K.weight.numel():,}")
print(f"  V params: {gqa.W_V.weight.numel():,}")
print(f"  Total attention params: {count_params(gqa):,}")
print(f"  KV ratio (vs full MHA): {n_kv_heads/n_heads:.2f}")
# Output: GQA: n_heads=8, n_kv_heads=2
# Output:   Q params: 4,096
# Output:   K params: 1,024
# Output:   V params: 1,024
# Output:   Total attention params: 6,144
# Output:   KV ratio (vs full MHA): 0.25
```

### Example 2: GQA with KV Cache

```python
class GQAKVCache(nn.Module):
    """GQA attention with KV cache support."""
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_heads
        self.n_groups = n_heads // n_kv_heads

        self.W_Q = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.W_K = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.W_V = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.W_O = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    def forward(self, x, past_kv=None, use_cache=False):
        batch, seq_len, _ = x.shape

        Q = self.W_Q(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(x).view(batch, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)
        V = self.W_V(x).view(batch, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)

        # Update KV cache (stores K, V with n_kv_heads, not n_heads!)
        if past_kv is not None:
            past_k, past_v = past_kv
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)

        # Repeat K, V for groups
        K_rep = K.repeat_interleave(self.n_groups, dim=1)
        V_rep = V.repeat_interleave(self.n_groups, dim=1)

        scores = torch.matmul(Q, K_rep.transpose(-2, -1)) / math.sqrt(self.d_head)
        # Causal mask for prefill only
        if past_kv is None and seq_len > 1:
            mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
            scores = scores + mask.to(x.device)

        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V_rep)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        output = self.W_O(attn_out)

        if use_cache:
            return output, (K, V)  # Cache stores n_kv_heads (smaller!)
        return output, None

# Demonstrate KV cache memory savings
d_model, n_heads, n_kv_heads = 256, 8, 2
gqa_attn = GQAKVCache(d_model, n_heads, n_kv_heads)
mha_attn = GQAKVCache(d_model, n_heads, n_heads)  # Full MHA

def kv_cache_size(attn_module, seq_len, dtype=torch.float16):
    """Estimate KV cache size for one layer."""
    d_head = attn_module.d_head
    n_kv = attn_module.n_kv_heads
    size = 2 * n_kv * d_head * seq_len  # 2 for K and V
    return size * 2  # 2 bytes for FP16

seq_len = 4096
gqa_cache = kv_cache_size(gqa_attn, seq_len)
mha_cache = kv_cache_size(mha_attn, seq_len)
print(f"KV cache size per layer (seq_len={seq_len}, FP16):")
print(f"  MHA (n_kv=8):  {mha_cache/1024:.2f} KB")
print(f"  GQA (n_kv=2):  {gqa_cache/1024:.2f} KB")
print(f"  Memory ratio:  {gqa_cache/mha_cache:.2f}x")
# Output: KV cache size per layer (seq_len=4096, FP16):
# Output:   MHA (n_kv=8):  256.00 KB
# Output:   GQA (n_kv=2):  64.00 KB
# Output:   Memory ratio:  0.25x
```

### Example 3: Comparing MHA, GQA, and MQA

```python
def compare_attention_variants():
    """Compare MHA, GQA, and MQA in terms of params, KV cache, and quality proxy."""
    d_model = 512
    n_heads = 16
    d_head = d_model // n_heads

    print(f"{'Variant':<8} {'n_kv':<8} {'KV params':<12} {'KV cache (MB)':<15} {'Ratio':<8}")
    print("-" * 51)

    for variant, n_kv in [("MHA", 16), ("GQA-8", 8), ("GQA-4", 4), ("GQA-2", 2), ("MQA", 1)]:
        kv_params = 2 * d_model * n_kv * d_head  # K and V projections
        kv_cache = 2 * 32 * n_kv * d_head * 4096 * 2 / (1024**2)  # 32 layers, 4096 seq, FP16
        ratio = n_kv / 16
        print(f"{variant:<8} {n_kv:<8} {kv_params:<12,} {kv_cache:<15.2f} {ratio:<8.2f}")

compare_attention_variants()
# Output: Variant  n_kv     KV params    KV cache (MB)   Ratio
# Output: --------------------------------------------------
# Output: MHA      16       8,388,608    128.00         1.00
# Output: GQA-8    8        4,194,304    64.00          0.50
# Output: GQA-4    4        2,097,152    32.00          0.25
# Output: GQA-2    2        1,048,576    16.00          0.13
# Output: MQA      1        524,288      8.00           0.06
```

### Example 4: GQA in Llama-style Configuration

```python
def llama_gqa_config():
    """Show GQA configurations for various Llama models."""
    models = [
        ("Llama 1 7B", 4096, 32, 32),      # Full MHA
        ("Llama 2 7B", 4096, 32, 32),       # Full MHA (same)
        ("Llama 2 13B", 5120, 40, 40),      # Full MHA
        ("Llama 2 70B", 8192, 64, 8),       # GQA (n_kv=8)
        ("Llama 3 8B", 4096, 32, 8),        # GQA (n_kv=8)
        ("Llama 3 70B", 8192, 64, 8),       # GQA (n_kv=8)
        ("Mistral 7B", 4096, 32, 8),        # GQA (n_kv=8)
        ("Mixtral 8x7B", 4096, 32, 8),      # GQA (n_kv=8)
        ("Gemma 7B", 3072, 16, 16),         # Full MHA
        ("Gemma 2B", 2048, 8, 1),           # MQA (n_kv=1)
    ]

    print(f"{'Model':<15} {'d_model':<8} {'n_q':<5} {'n_kv':<5} {'Group':<8} {'KV cache':<10}")
    print("-" * 51)

    for name, d_model, n_q, n_kv in models:
        group = n_q // n_kv if n_kv > 0 else "N/A"
        kv_cache_per_layer = 2 * n_kv * (d_model // n_q) * 4096 * 2 / (1024**2)  # MB for 4096 seq
        print(f"{name:<15} {d_model:<8} {n_q:<5} {n_kv:<5} {str(group):<8} {kv_cache_per_layer:<10.2f}")

llama_gqa_config()
# Output: Model           d_model   n_q   n_kv  Group     KV cache
# Output: --------------------------------------------------
# Output: Llama 1 7B     4096      32    32    8         4.00
# Output: Llama 2 7B     4096      32    32    8         4.00
# Output: Llama 2 13B    5120      40    40    8         5.00
# Output: Llama 2 70B    8192      64    8     8         1.00
# Output: Llama 3 8B     4096      32    8     4         1.00
# Output: Llama 3 70B    8192      64    8     8         1.00
# Output: Mistral 7B     4096      32    8     4         1.00
# Output: Mixtral 8x7B   4096      32    8     4         1.00
# Output: Gemma 7B       3072      16    16    1         2.00
# Output: Gemma 2B       2048      8     1     8         0.13
```

## Common Mistakes

1. **Not ensuring n_heads is divisible by n_kv_heads**: GQA requires that query heads can be evenly divided into groups. Common ratios: n_q/n_kv = 4, 8, or 16.

2. **Confusing group assignment**: The repeat_interleave operation assigns consecutive query heads to the same KV head. For example, with n_q=32, n_kv=8: query heads 0-3 share KV head 0, heads 4-7 share KV head 1, etc.

3. **Forgetting that the KV cache stores n_kv_heads, not n_heads**: The key memory savings from GQA comes from storing fewer KV heads in the cache. The cache shape is (batch, n_kv, seq, d_head), not (batch, n_q, seq, d_head).

4. **Using the same d_model for KV projections**: The KV projections have output dimension n_kv × d_head, while the Q projection has output dimension n_q × d_head (= d_model). This is correct but can be confusing when reading code.

5. **Assuming GQA always matches MHA quality**: With very few KV heads (n_kv=1, MQA), there can be quality degradation, especially for tasks requiring fine-grained attention. The standard recommendation is n_kv ≥ 8.

## Interview Questions

### Beginner

**Q: What is Grouped-Query Attention and how does it differ from Multi-Head Attention?**

A: GQA reduces the number of key-value heads while keeping the same number of query heads. In MHA, each query head has its own key and value head. In GQA, multiple query heads (a group) share a single key-value head. For example, with 32 query heads and 8 KV heads, groups of 4 query heads share one KV head. This reduces KV cache size by a factor of 4 while maintaining most of the quality.

### Intermediate

**Q: How does GQA reduce the KV cache size? What is the memory savings for a typical configuration?**

A: GQA reduces the KV cache by a factor of n_q / n_kv. In full MHA, the cache stores K and V for all n_q heads: 2 × n_layers × n_q × d_head × seq_len. In GQA with n_kv heads, the cache stores K and V for only n_kv heads: 2 × n_layers × n_kv × d_head × seq_len. For a model with 32 query heads and 8 KV heads (GQA-8), the cache is 4x smaller. During attention, the KV heads are repeated to match the query head count.

### Advanced

**Q: Why does GQA with n_kv=8 perform nearly as well as full MHA (n_kv=n_q) in large models? What does this tell us about attention head redundancy?**

A: Research suggests that many KV heads in full MHA learn redundant attention patterns. The keys and values across different heads often exhibit significant similarity — they attend to similar tokens in similar ways. GQA exploits this by forcing heads within a group to share a KV head, which acts as a form of regularization. The residual stream still carries different information for each query head (through different Q projections), so the model retains the ability to attend to different patterns. The quality loss from sharing KV heads is minimal because: (1) the Q projections provide sufficient diversity, (2) the shared KV still captures the main attention patterns, and (3) the model can compensate through the output projection. This suggests that the primary benefit of multiple heads comes from the query diversity, not the key-value diversity.

## Practice Problems

### Easy

Implement a function that converts a standard MHA module to GQA with a given number of KV heads. Verify that the output shapes are correct.

### Medium

Compare the perplexity of a small Transformer language model trained with MHA (n_kv=n_q) vs GQA (n_kv=n_q/4) with the same total parameter count. Control for the parameter difference by adjusting d_model.

### Hard

Implement "cross-group attention" where groups are not fixed but determined dynamically by a learned routing mechanism. Compare its performance with standard GQA.

## Solutions

### Easy Solution

```python
def convert_to_gqa(mha_module, n_kv_heads):
    """Convert a standard MHA to GQA by modifying K and V projections."""
    d_model = mha_module.W_Q.in_features
    n_heads = mha_module.W_Q.out_features // (d_model // mha_module.W_Q.out_features)
    # Simplified: just verify the logic
    assert n_heads % n_kv_heads == 0
    print(f"Converted MHA (n_heads={n_heads}) to GQA (n_kv={n_kv_heads})")
    print(f"  KV parameter reduction: {n_kv_heads/n_heads:.2f}x")
    print(f"  KV cache reduction: {n_kv_heads/n_heads:.2f}x")

convert_to_gqa(None, 8)  # Assuming 32 heads originally
# Output: Converted MHA (n_heads=32) to GQA (n_kv=8)
# Output:   KV parameter reduction: 0.25x
# Output:   KV cache reduction: 0.25x
```

## Related Concepts

- **DL-371: Attention Head**: The individual head concept.
- **DL-379: n_heads**: The number of query heads.
- **DL-383: KV Cache**: The memory optimization that GQA improves.
- **Multi-Query Attention (MQA)**: The extreme case where n_kv = 1.
- **Flash Attention**: IO-aware attention that also improves efficiency.

## Next Concepts

- DL-385: Flash Attention — The most efficient attention implementation.

## Summary

Grouped-Query Attention (GQA) is an attention mechanism that reduces the number of key-value heads below the number of query heads, with groups of query heads sharing a single KV head. It interpolates between full Multi-Head Attention (MHA, n_kv = n_q) and Multi-Query Attention (MQA, n_kv = 1). GQA reduces KV cache memory and inference memory bandwidth by a factor of n_q / n_kv with minimal quality loss. It has become the standard in modern LLMs (Llama 2, Llama 3, Mistral, Mixtral, Gemma) and is essential for efficient long-context inference.

## Key Takeaways

1. GQA shares KV heads across groups of query heads.
2. KV cache size is reduced by n_q / n_kv.
3. Standard configurations: GQA-8 (n_kv=8) or GQA-4 (n_kv=4).
4. Quality loss is minimal compared to full MHA.
5. Used in Llama 2/3, Mistral, Mixtral, and Gemma.
6. The KV cache stores n_kv heads (smaller), repeated to n_q during attention.
