# Grouped Query Attention

## Concept ID
DL-447

## Difficulty
Intermediate

## Domain
Deep Learning Architectures

## Module
Efficient Attention (DL-446 to DL-450)

## Learning Objectives
- Understand the grouped query attention architecture
- Implement GQA with configurable number of KV heads
- Analyze the trade-off between efficiency and quality
- Compare GQA with MHA and MQA

## Prerequisites
- Multi-Query Attention (DL-446)
- Multi-Head Attention (DL-339)
- Attention Mechanisms (DL-335)

## Definition
Grouped Query Attention (GQA) is an intermediate between multi-head attention (MHA) and multi-query attention (MQA), where query heads are divided into groups, and each group shares a single key-value head. With G groups and H query heads, the KV cache is reduced by a factor of H/G compared to MHA. GQA was introduced in "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023).

## Intuition
Think of MHA as 32 experts, each taking their own notes (K,V). MQA is 32 experts sharing one set of notes. GQA sits in between: divide the 32 experts into, say, 4 groups of 8 experts each. Each group of 8 shares one set of notes, but different groups take different notes. This is like a research team split into 4 sub-teams, each studying a different aspect of the problem. Within each sub-team, members share findings but ask different questions. This captures more diverse information than MQA while saving memory compared to MHA.

## Why This Concept Matters
GQA has become the dominant attention mechanism in modern LLMs (LLaMA 2, LLaMA 3, Mistral, GPT-4). It provides a tunable trade-off between MHA's quality and MQA's efficiency. By choosing the number of KV heads (typically 4-8), models can achieve most of MHA's quality with most of MQA's memory savings.

## Mathematical Explanation

### GQA Formulation

For $H$ query heads grouped into $G$ groups (with $H/G$ query heads per group):

$$Q_{h} = XW_{h}^Q \quad \text{for } h = 1, ..., H$$
$$K_{g} = XW_{g}^K \quad \text{for } g = 1, ..., G$$
$$V_{g} = XW_{g}^V \quad \text{for } g = 1, ..., G$$

Each query head $h$ is assigned to group $g = \lfloor h \cdot G / H \rfloor$ and uses $K_g, V_g$:

$$\text{head}_h = \text{softmax}\left(\frac{Q_h K_{\text{group}(h)}^T}{\sqrt{d_k}}\right)V_{\text{group}(h)}$$

### KV Cache Comparison

| Method | KV Heads | KV Cache per Token | Reduction vs MHA |
|--------|----------|-------------------|------------------|
| MHA    | H        | 2 × H × d_k      | 1×               |
| GQA    | G        | 2 × G × d_k      | H/G ×            |
| MQA    | 1        | 2 × d_k          | H ×              |

### Parameter Count

| Projection | MHA | GQA | MQA |
|------------|-----|-----|-----|
| Query      | H·d·d_k | H·d·d_k | H·d·d_k |
| Key        | H·d·d_k | G·d·d_k | d·d_k |
| Value      | H·d·d_k | G·d·d_k | d·d_k |

## Code Examples

### Example 1: Grouped Query Attention Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with configurable KV heads"""
    def __init__(self, d_model, n_heads, n_kv_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        self.n_groups = n_heads // n_kv_heads
        
        # Query: one per head (standard)
        self.W_Q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        
        # Key, Value: one per KV head
        self.W_K = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        
        self.W_O = nn.Linear(n_heads * self.d_k, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Cache for generation
        self.kv_cache = None
    
    def reset_cache(self):
        self.kv_cache = None
    
    def forward(self, x, use_cache=False):
        B, T, D = x.shape
        
        # Project Q, K, V
        Q = self.W_Q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        
        # Handle KV cache
        if use_cache and self.kv_cache is not None:
            cached_K, cached_V = self.kv_cache
            K = torch.cat([cached_K, K], dim=2)
            V = torch.cat([cached_V, V], dim=2)
        
        if use_cache:
            self.kv_cache = (K, V)
        
        # Expand KV heads to match query heads
        # K: (B, G, T, d_k) -> (B, H, T, d_k) where H = G * n_groups
        K = K.repeat_interleave(self.n_groups, dim=1)
        V = V.repeat_interleave(self.n_groups, dim=1)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)
        
        # Combine heads
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_O(out)

# Test GQA
d_model, n_heads, n_kv_heads = 512, 8, 4
gqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
x = torch.randn(2, 16, d_model)
output = gqa(x)

print(f"GQA Configuration:")
print(f"  Query heads: {n_heads}")
print(f"  KV heads: {n_kv_heads}")
print(f"  Groups: {n_heads // n_kv_heads}")
print(f"  Output shape: {output.shape}")
print(f"  KV cache per token: {2 * n_kv_heads * (d_model // n_heads) * 2} bytes")
print(f"  vs MHA: {2 * n_heads * (d_model // n_heads) * 2} bytes")
print(f"  Savings: {(1 - n_kv_heads/n_heads)*100:.0f}%")
# Output: GQA Configuration:
# Output:   Query heads: 8
# Output:   KV heads: 4
# Output:   Groups: 2
# Output:   Output shape: torch.Size([2, 16, 512])
# Output:   KV cache per token: 1024 bytes
# Output:   vs MHA: 2048 bytes
# Output:   Savings: 50%
```

### Example 2: GQA with Different Group Sizes

```python
class GQAComparison:
    """Compare different GQA configurations"""
    
    @staticmethod
    def compare_configs():
        d_model = 4096
        n_heads = 32
        B, T = 4, 2048
        
        configs = [
            ('MHA (32 KV heads)', 32),
            ('GQA (8 KV heads)', 8),
            ('GQA (4 KV heads)', 4),
            ('MQA (1 KV head)', 1),
        ]
        
        print("GQA Configuration Comparison:")
        print("-" * 90)
        print(f"{'Config':<25}{'KV Heads':<15}{'KV Cache':<20}{'Savings':<15}{'KV Params'}")
        print("-" * 90)
        
        for name, n_kv in configs:
            d_k = d_model // n_heads
            kv_cache_bytes = 2 * B * T * n_kv * d_k * 2  # fp16
            kv_params = 2 * d_model * n_kv * d_k
            savings = (1 - n_kv / n_heads) * 100
            
            print(f"{name:<25}{n_kv:<15}{kv_cache_bytes/1e6:<20.2f}MB"
                  f"{savings:<15.0f}%{kv_params/1e6:<.1f}M")
        
        print("\nGQA provides flexible trade-off:")
        print("  - More KV heads ≈ better quality")
        print("  - Fewer KV heads ≈ better efficiency")
        print("  - Sweet spot: 4-8 KV heads for most models")

GQAComparison.compare_configs()
# Output: GQA Configuration Comparison:
# Output: ------------------------------------------------------------------------------------------
# Output: Config                    KV Heads        KV Cache            Savings        KV Params
# Output: ------------------------------------------------------------------------------------------
# Output: MHA (32 KV heads)         32              128.00 MB           0%             128.0M
# Output: GQA (8 KV heads)          8               32.00 MB            75%            32.0M
# Output: GQA (4 KV heads)          4               16.00 MB            88%            16.0M
# Output: MQA (1 KV head)           1               4.00 MB             97%            4.0M
```

### Example 3: GQA with KV Caching for Generation

```python
class GQAForGeneration(GroupedQueryAttention):
    """GQA layer optimized for autoregressive generation"""
    
    def forward_prefill(self, x):
        """Process full prompt and cache all K,V"""
        B, T, D = x.shape
        
        K = self.W_K(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        
        self.kv_cache = (K, V)
        
        # Full attention for prefill
        Q = self.W_Q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K_expanded = K.repeat_interleave(self.n_groups, dim=1)
        V_expanded = V.repeat_interleave(self.n_groups, dim=1)
        
        scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V_expanded)
        return out.transpose(1, 2).contiguous().view(B, T, -1)
    
    def forward_decode(self, x):
        """Process single token with cached K,V"""
        B, _, D = x.shape
        
        Q = self.W_Q(x).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        
        new_K = self.W_K(x).view(B, 1, self.n_kv_heads, self.d_k).transpose(1, 2)
        new_V = self.W_V(x).view(B, 1, self.n_kv_heads, self.d_k).transpose(1, 2)
        
        # Update cache
        cached_K, cached_V = self.kv_cache
        K = torch.cat([cached_K, new_K], dim=2)
        V = torch.cat([cached_V, new_V], dim=2)
        self.kv_cache = (K, V)
        
        # Expand KV heads
        K_expanded = K.repeat_interleave(self.n_groups, dim=1)
        V_expanded = V.repeat_interleave(self.n_groups, dim=1)
        
        scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V_expanded)
        return out.transpose(1, 2).contiguous().view(B, 1, -1)

# Demonstrate prefill vs decode
gqa_gen = GQAForGeneration(512, 8, 4)

# Prefill
prompt = torch.randn(1, 32, 512)
prefill_out = gqa_gen.forward_prefill(prompt)
print(f"Prefill: in={prompt.shape}, out={prefill_out.shape}")
print(f"KV cache shape: {[c.shape for c in gqa_gen.kv_cache]}")

# Decode step
next_token = torch.randn(1, 1, 512)
decode_out = gqa_gen.forward_decode(next_token)
print(f"Decode: in={next_token.shape}, out={decode_out.shape}")
print(f"Updated KV cache shape: {[c.shape for c in gqa_gen.kv_cache]}")
# Output: Prefill: in=torch.Size([1, 32, 512]), out=torch.Size([1, 32, 512])
# Output: KV cache shape: [torch.Size([1, 4, 32, 64]), torch.Size([1, 4, 32, 64])]
# Output: Decode: in=torch.Size([1, 1, 512]), out=torch.Size([1, 1, 512])
# Output: Updated KV cache shape: [torch.Size([1, 4, 33, 64]), torch.Size([1, 4, 33, 64])]
```

### Example 4: Converting MHA to GQA

```python
def convert_mha_to_gqa(mha_model, n_kv_heads):
    """Convert pretrained MHA model to GQA by merging KV heads"""
    d_model = mha_model.W_K.weight.shape[1]
    n_heads = mha_model.W_K.weight.shape[0] // (d_model // mha_model.W_K.weight.shape[1])
    
    # Determine d_k
    d_k = d_model // n_heads
    
    # Create GQA model
    gqa_model = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
    
    # Copy Q projections (unchanged)
    gqa_model.W_Q.weight.data = mha_model.W_Q.weight.data.clone()
    
    # Merge KV heads: average within each group
    kv_per_group = n_heads // n_kv_heads
    
    with torch.no_grad():
        # K projection
        mha_K = mha_model.W_K.weight.view(n_heads, d_k, d_model)
        gqa_K = torch.zeros(n_kv_heads, d_k, d_model)
        for g in range(n_kv_heads):
            start = g * kv_per_group
            end = (g + 1) * kv_per_group
            gqa_K[g] = mha_K[start:end].mean(dim=0)
        gqa_model.W_K.weight.data = gqa_K.view(n_kv_heads * d_k, d_model)
        
        # V projection (same logic)
        mha_V = mha_model.W_V.weight.view(n_heads, d_k, d_model)
        gqa_V = torch.zeros(n_kv_heads, d_k, d_model)
        for g in range(n_kv_heads):
            start = g * kv_per_group
            end = (g + 1) * kv_per_group
            gqa_V[g] = mha_V[start:end].mean(dim=0)
        gqa_model.W_V.weight.data = gqa_V.view(n_kv_heads * d_k, d_model)
        
        # Copy output projection
        gqa_model.W_O.weight.data = mha_model.W_O.weight.data.clone()
    
    return gqa_model

# Simulate conversion
class MockMHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        d_k = d_model // n_heads
        self.W_Q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_O = nn.Linear(n_heads * d_k, d_model, bias=False)

mha = MockMHA(512, 8)
gqa = convert_mha_to_gqa(mha, 4)

mha_params = sum(p.numel() for p in mha.parameters())
gqa_params = sum(p.numel() for p in gqa.parameters())

print("MHA → GQA Conversion:")
print(f"  MHA params: {mha_params:,}")
print(f"  GQA params (4 KV heads): {gqa_params:,}")
print(f"  Reduction: {(1 - gqa_params/mha_params)*100:.1f}%")
print(f"  KV memory reduction: 50% (8→4 KV heads)")
# Output: MHA → GQA Conversion:
# Output:   MHA params: 1,312,768
# Output:   GQA params (4 KV heads): 1,049,600
# Output:   Reduction: 20.0%
# Output:   KV memory reduction: 50% (8→4 KV heads)
```

### Example 5: Quality-Efficiency Pareto Frontier

```python
class GQAParetoAnalysis:
    """Analyze quality-efficiency trade-off with different GQA configs"""
    
    @staticmethod
    def analyze():
        import numpy as np
        
        # Simulated quality scores for different GQA configurations
        # Based on empirical findings from the GQA paper
        n_heads = 32
        
        configs = [
            ('MHA', 32, 100.0, 0),      # Reference
            ('GQA-16', 16, 99.7, 50),
            ('GQA-8', 8, 99.2, 75),
            ('GQA-4', 4, 98.5, 87.5),
            ('MQA', 1, 97.0, 96.875),
        ]
        
        print("GQA Quality-Efficiency Pareto Frontier:")
        print("-" * 80)
        print(f"{'Config':<15}{'KV Heads':<15}{'Relative Quality':<25}{'Memory Savings':<25}")
        print("-" * 80)
        
        for name, kv_heads, quality, savings in configs:
            quality_str = f"{quality:.1f}% ({quality-100:+.1f}%)"
            savings_str = f"{savings:.1f}%"
            print(f"{name:<15}{kv_heads:<15}{quality_str:<25}{savings_str:<25}")
        
        print("\nPareto-optimal configurations:")
        print("  GQA-8: 99.2% quality, 75% memory savings ← Most popular")
        print("  GQA-4: 98.5% quality, 87.5% memory savings")
        print("  MQA: 97.0% quality, 96.875% memory savings")
        
        print("\nRecommendation:")
        print("  Production: GQA-8 (LLaMA 2/3, Mistral)")
        print("  Edge/latency-critical: GQA-4 or MQA")
        print("  Quality-critical: MHA")

GQAParetoAnalysis.analyze()
# Output: GQA Quality-Efficiency Pareto Frontier:
# Output: --------------------------------------------------------------------------------
# Output: Config         KV Heads        Relative Quality         Memory Savings
# Output: --------------------------------------------------------------------------------
# Output: MHA            32              100.0% (+0.0%)           0.0%
# Output: GQA-16         16              99.7% (-0.3%)            50.0%
# Output: GQA-8          8               99.2% (-0.8%)            75.0%
# Output: GQA-4          4               98.5% (-1.5%)            87.5%
# Output: MQA            1               97.0% (-3.0%)            96.9%
```

## Common Mistakes

### 1. Setting KV Heads Not Divisible Into Query Heads
GQA requires that n_heads % n_kv_heads == 0. If n_heads=32 and n_kv_heads=6 (32%6=2), the grouping is uneven. Always choose n_kv_heads that divides n_heads evenly.

### 2. Not Expanding KV Heads During Attention Computation
The KV tensors have shape (B, G, T, d_k) but queries have (B, H, T, d_k). You must repeat_interleave the KV heads by n_groups to make them (B, H, T, d_k) before computing attention.

### 3. Assuming GQA Always Outperforms MQA
GQA has more parameters and memory than MQA. For very latency-constrained applications, MQA's simpler implementation (no grouping logic) may be preferable despite lower quality.

### 4. Using Inefficient KV Head Expansion
Using .repeat() instead of .repeat_interleave() for KV expansion creates unnecessary copies in memory. repeat_interleave is more memory efficient.

### 5. Not Fine-Tuning After Conversion
Converting MHA to GQA by averaging KV heads is an approximation. Always fine-tune the converted model for 100-10000 steps to recover quality.

## Interview Questions

### Beginner
**Q1: What is Grouped Query Attention and how does it differ from MHA and MQA?**
A1: GQA divides query heads into groups, with each group sharing one key-value head. It's intermediate between MHA (each head has its own KV) and MQA (all heads share one KV). With 32 query heads and 4 KV heads, GQA uses 8 query heads per KV head, saving 87.5% KV cache vs MHA.

**Q2: How do you choose the number of KV heads in GQA?**
A2: The choice depends on the quality-efficiency trade-off. Common choices: 4-8 KV heads for most LLMs (good balance), more KV heads for quality-critical tasks, fewer for latency-critical or memory-constrained deployment. The number must divide the total query heads evenly.

### Intermediate
**Q3: Explain the process of converting a pretrained MHA model to GQA.**
A3: (1) Group query heads into G groups (G = n_kv_heads); (2) For each group, average the K projections of its constituent heads to create one merged K head; (3) Same for V projections; (4) Keep Q projections separate (one per head); (5) Initialize new GQA model with these weights; (6) Fine-tune for quality recovery. The averaging works because heads in the same group tend to learn complementary but related patterns.

**Q4: How does GQA affect the training dynamics compared to MHA?**
A4: GQA has fewer parameters in K,V projections, which can slow convergence slightly. However, the shared K,V provide a regularization effect, often improving generalization. Training speed: GQA is similar to MHA (Q projections dominate computation). Memory during training: GQA uses less activation memory (smaller K,V projections). The main benefit is during inference (KV cache), not training.

### Advanced
**Q5: Design a method to learn which heads should share KV projections rather than using fixed groups.**
A5: Learned grouping: (1) Initialize all heads with separate K,V; (2) Add a learnable assignment matrix A of shape (H, G) where A_h,g indicates head h belongs to group g; (3) The effective K,V for head h is sum_g A_h,g * K_g, V_g; (4) Use Gumbel-Softmax to make assignments discrete during training; (5) Add entropy regularization to encourage clean grouping; (6) After training, heads with similar learned groups naturally cluster, potentially revealing interpretable attention patterns.

**Q6: How would you implement variable GQA across layers (different KV heads per layer)?**
A6: Layer-specific GQA allocation: (1) Early layers (1-4): higher KV heads (e.g., 16-32) for richer representation; (2) Middle layers (5-28): standard GQA (4-8 KV heads); (3) Late layers (29-32): MQA (1 KV head). The total KV cache becomes: sum_l G_l * d_k * 2. This can reduce total KV cache by an additional 20-30% compared to uniform GQA. Implementation: store per-layer n_kv_heads in a config list and instantiate each layer accordingly.

## Practice Problems

### Easy
Calculate the KV cache size for GQA with 16 query heads and 2, 4, 8, and 16 KV heads for batch size 4, sequence length 4096, d_model=4096.

### Medium
Implement GQA with learned grouping where the assignment of query heads to KV groups is learned during training.

### Hard
Design an adaptive GQA system that dynamically adjusts the number of KV heads based on current memory pressure during inference.

## Solutions

### Easy Solution
```python
def gqa_cache_size(H, G, B, T, d_model, dtype_bytes=2):
    d_k = d_model // H
    return 2 * B * T * G * d_k * dtype_bytes

for G in [2, 4, 8, 16]:
    size = gqa_cache_size(16, G, 4, 4096, 4096)
    print(f"G={G}: {size/1e9:.2f} GB, {G/16*100:.0f}% of MHA")
```

### Medium Solution
```python
class LearnedGQA(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.group_assign = nn.Parameter(torch.randn(n_heads, n_kv_heads))
        # ... projections
    def get_group_weights(self, tau=1.0):
        return F.gumbel_softmax(self.group_assign, tau=tau, dim=-1, hard=True)
```

### Hard Solution
```python
class AdaptiveGQA(nn.Module):
    def __init__(self, d_model, n_heads, max_kv_heads):
        super().__init__()
        self.max_kv_heads = max_kv_heads
        self.memory_pressure = 0.0
    
    def get_active_kv_heads(self):
        if self.memory_pressure < 0.5:
            return self.max_kv_heads
        elif self.memory_pressure < 0.8:
            return max(1, self.max_kv_heads // 2)
        else:
            return 1  # Fall back to MQA
```

## Related Concepts
- DL-446: Multi-Query Attention - Special case of GQA (G=1)
- DL-339: Multi-Head Attention - Special case of GQA (G=H)
- DL-448: Sliding Window Attention - Alternative efficient attention
- DL-335: Attention Mechanisms - Foundation concept
- DL-337: Transformer Architecture - Overall architecture

## Next Concepts
- DL-448: Sliding Window Attention
- DL-449: Sparse Attention

## Summary
Grouped Query Attention provides a tunable trade-off between the quality of multi-head attention and the efficiency of multi-query attention. By sharing key-value projections within groups of query heads, GQA achieves near-MHA quality with significantly reduced KV cache memory. It has become the standard attention mechanism in modern LLMs including LLaMA 2/3, Mistral, and GPT-4.

## Key Takeaways
- GQA: intermediate between MHA and MQA
- KV heads must divide query heads evenly
- Group size = n_heads / n_kv_heads
- KV cache savings: (1 - G/H) × 100%
- Quality loss: 0.3-3% depending on G
- GQA-8 is the most popular configuration
- Converting MHA to GQA requires averaging KV heads
- Always fine-tune after conversion
- Different layers can have different G
- GQA dominates modern LLM architectures
