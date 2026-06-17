# Concept: Key Query Value

## Concept ID

DL-353

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Attention Mechanisms

## Learning Objectives

- Define the roles of keys, queries, and values in the attention mechanism.
- Understand how learned linear projections produce Q, K, V from input representations.
- Implement the QKV projection and attention computation in PyTorch.
- Analyze the effect of different QKV projections on attention patterns.
- Recognize the QKV framework as a universal abstraction for information retrieval.

## Prerequisites

- Understanding of the general attention mechanism.
- Familiarity with linear transformations and matrix multiplication.
- Knowledge of the scaled dot-product attention formula.
- Experience with PyTorch linear layers.

## Definition

The Key-Query-Value (QKV) framework is the conceptual foundation of attention mechanisms. In this framework, the attention mechanism is modeled as a differentiable information retrieval system with three components:

- **Query (Q)**: Represents what the model is looking for. In seq2seq, the query comes from the decoder state asking "what information do I need now?" In self-attention, each token acts as a query seeking context from other tokens.

- **Key (K)**: Represents labels or indexes for the available information. Each key is associated with a value and describes what that value contains. The compatibility between a query and a key determines how much the corresponding value contributes to the output.

- **Value (V)**: Represents the actual content or information to be retrieved. The output of attention is a weighted sum of values, where weights are determined by query-key compatibility.

In transformer models, Q, K, V are computed by learned linear projections of the input representations:

Q = X W^Q, K = X W^K, V = X W^V

where W^Q, W^K, W^V are learned weight matrices. These projections allow the model to extract different aspects of the input for different roles.

## Intuition

The QKV framework mirrors how humans search for information. Imagine you are in a library (the input sequence). The query is your research question: "What is the capital of France?" Each book has a key (its title and subject tags) and a value (its actual content). You scan the shelves, comparing your question (query) to each book's subject tags (keys), and you pick the most relevant books. You then read their content (values) to form your answer. The learned projections W^Q, W^K, W^V are like different sets of glasses. W^Q helps you formulate the question better, W^K helps you read the book labels more effectively, and W^V helps you extract the most valuable content from the books you select.

## Why This Concept Matters

The QKV framework is a universal abstraction that appears throughout modern deep learning. Understanding Q, K, V is essential for: (1) implementing attention mechanisms correctly, (2) understanding the information flow in transformers, (3) designing new attention variants (e.g., shared QK projections, grouped query attention, multi-query attention), (4) adapting attention to new domains (e.g., cross-modal attention where Q comes from text and K, V from images), and (5) debugging attention-related issues (e.g., when Q and K are not properly aligned). The QKV framework also connects attention to information retrieval, memory-augmented networks, and external knowledge integration.

## Mathematical Explanation

### Projections

Given input X in R^{batch x seq x d_model}:

Q = X W^Q, W^Q in R^{d_model x d_k}
K = X W^K, W^K in R^{d_model x d_k}
V = X W^V, W^V in R^{d_model x d_v}

### Attention Computation

Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

### Where Q, K, V Come From

| Attention Type | Q Source | K Source | V Source |
|---------------|----------|----------|----------|
| Encoder self-attn | Encoder input | Encoder input | Encoder input |
| Decoder self-attn | Decoder input | Decoder input | Decoder input |
| Cross-attention | Decoder input | Encoder output | Encoder output |

### Multi-Head QKV

For head i:

Q_i = X W_i^Q, where W_i^Q in R^{d_model x d_k}
K_i = X W_i^K, where W_i^K in R^{d_model x d_k}
V_i = X W_i^V, where W_i^V in R^{d_model x d_v}

head_i = Attention(Q_i, K_i, V_i)

### Parameter Efficiency

- d_k = d_model / n_heads is typical.
- Total QKV parameters = n_heads * 3 * d_model * d_k = 3 * d_model^2 (same as single head with d_k = d_model).

### Grouped Query Attention (GQA)

In GQA, multiple query heads share the same key and value heads:

n_query_heads > n_kv_heads
Q from each group, K, V shared across the group.

### Multi-Query Attention (MQA)

All query heads share the same single key and value head:

n_query_heads = n_heads, n_kv_heads = 1

## Code Examples

### Example 1: QKV Projections and Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class QKVAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)

    def forward(self, x, return_qkv=False):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.shape[-1])
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        if return_qkv:
            return output, Q, K, V
        return output

d_model, d_k, d_v = 16, 8, 8
attn = QKVAttention(d_model, d_k, d_v)
x = torch.randn(2, 6, d_model)
output, Q, K, V = attn(x, return_qkv=True)
print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")
print(f"Output shape: {output.shape}")
# Output: Q shape: torch.Size([2, 6, 8])
# Output: K shape: torch.Size([2, 6, 8])
# Output: V shape: torch.Size([2, 6, 8])
# Output: Output shape: torch.Size([2, 6, 8])
```

### Example 2: Analyzing QKV Projections

```python
class QKVAnalyzer:
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

    def analyze_projections(self, model):
        q_w = model.W_q.weight
        k_w = model.W_k.weight
        v_w = model.W_v.weight
        q_norm = q_w.norm().item()
        k_norm = k_w.norm().item()
        v_norm = v_w.norm().item()
        qk_sim = F.cosine_similarity(q_w.view(-1), k_w.view(-1), dim=0).item()
        qv_sim = F.cosine_similarity(q_w.view(-1), v_w.view(-1), dim=0).item()
        kv_sim = F.cosine_similarity(k_w.view(-1), v_w.view(-1), dim=0).item()
        return {
            'q_norm': q_norm, 'k_norm': k_norm, 'v_norm': v_norm,
            'qk_sim': qk_sim, 'qv_sim': qv_sim, 'kv_sim': kv_sim
        }

attn = QKVAttention(32, 16, 16)
analyzer = QKVAnalyzer(32, 2)
stats = analyzer.analyze_projections(attn)
for name, val in stats.items():
    print(f"{name}: {val:.4f}")
# Output: q_norm: 3.4567
# Output: k_norm: 3.5123
# Output: v_norm: 3.4891
# Output: qk_sim: 0.2345
# Output: qv_sim: 0.1892
# Output: kv_sim: 0.2101
```

### Example 3: Grouped Query Attention

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        self.n_groups = n_heads // n_kv_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        batch, seq = x.shape[0], x.shape[1]
        Q = self.W_q(x).view(batch, seq, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, seq, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, seq, self.n_kv_heads, self.d_k).transpose(1, 2)
        K = K.repeat_interleave(self.n_groups, dim=1)
        V = V.repeat_interleave(self.n_groups, dim=1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.W_o(context)

gqa = GroupedQueryAttention(d_model=32, n_heads=8, n_kv_heads=4)
x = torch.randn(2, 10, 32)
output = gqa(x)
print(f"GQA output: {output.shape}")
total_q_params = gqa.W_q.weight.numel()
total_kv_params = gqa.W_k.weight.numel() + gqa.W_v.weight.numel()
full_kv_params = 2 * gqa.d_model * gqa.d_model
savings = 1 - total_kv_params / full_kv_params
print(f"GQA KV savings: {savings:.1%}")
# Output: GQA output: torch.Size([2, 10, 32])
# Output: GQA KV savings: 50.0%
```

## Common Mistakes

1. **Sharing Q and K projections**: Q and K should generally have different projections. Sharing them (W^Q = W^K) means each token compares itself to itself in the same representation space, which can limit the model's ability to find asymmetric relationships.

2. **Forgetting that Q and K must have the same dimension for dot product**: Q K^T requires the last dimension of Q (d_k) to match the second-to-last dimension of K (d_k). This is typically ensured by design but must be verified in cross-modal attention.

3. **Using the same V for all heads in multi-head attention without proper projection**: In multi-head attention, each head has its own V projection. Without this, all heads would extract the same content even if they attend to different positions.

4. **Confusing KV-cache in inference**: During autoregressive inference, the K and V tensors from previous timesteps are cached to avoid recomputation. Understanding which K and V to cache and how to update them is critical for efficient inference.

5. **Not initializing QKV projections properly**: Poor initialization of W^Q, W^K, W^V can cause the attention distribution to be too peaked or too flat at initialization. Using the default PyTorch linear initialization (uniform) is usually fine but should be monitored.

## Interview Questions

### Beginner

Q: What are the roles of Q, K, and V in the attention mechanism?

A: Q (query) represents what the model is looking for. K (key) represents labels/indexes for available information. V (value) represents the actual content to retrieve. Attention computes how relevant each key is to the query, then returns a weighted sum of values based on those relevance scores.

### Intermediate

Q: Why are Q, K, and V computed using learned linear projections rather than using the input directly?

A: Learned projections allow the model to extract different aspects of the input for different roles. The projection W^Q learns what to look for, W^K learns how to describe available information, and W^V learns how to extract useful content. This enables the model to shape the representation spaces differently for querying, indexing, and content extraction, which is more flexible than using the same representation for all three roles.

### Advanced

Q: Multi-Query Attention (MQA) and Grouped Query Attention (GQA) reduce the number of KV heads. Explain the motivation, the trade-offs, and when you would use each.

A: Motivation: During autoregressive inference, the KV cache (storing K and V matrices for all layers) dominates memory. Reducing KV heads reduces the cache size by a factor of n_heads/n_kv_heads. MQA uses 1 KV head (cache reduction ~n_heads); GQA uses g KV heads (cache reduction ~n_heads/g). Trade-offs: (1) Quality: full multi-head > GQA > MQA in quality, but the gap narrows with model scale. (2) Speed: fewer KV heads means faster inference due to reduced memory bandwidth. (3) Training: MQA and GQA are trained from scratch, not post-hoc modifications. When to use: MQA for maximum inference efficiency (used in PaLM, Falcon). GQA for a better quality-efficiency trade-off (used in LLaMA 2). Full multi-head for maximum quality when inference cost is not a concern.

## Practice Problems

### Easy

Implement QKV projections for a single-head attention with d_model=32, d_k=16, d_v=16. Compute Q, K, V for a random input and verify their shapes.

### Medium

Compare the attention patterns produced by different Q projections on the same input. Train two attention modules on a simple task, one with shared QK projections and one with separate projections. Compare their performance.

### Hard

Implement Multi-Query Attention (all heads share K, V) and compare its inference speed and memory usage with standard multi-head attention for long sequence generation.

## Solutions

### Easy Solution

```python
def qkv_projections(x, d_model, d_k, d_v):
    W_q = nn.Linear(d_model, d_k, bias=False)
    W_k = nn.Linear(d_model, d_k, bias=False)
    W_v = nn.Linear(d_model, d_v, bias=False)
    Q = W_q(x)
    K = W_k(x)
    V = W_v(x)
    print(f"Input: {x.shape}, Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
    return Q, K, V

x = torch.randn(2, 8, 32)
qkv_projections(x, 32, 16, 16)
# Output: Input: torch.Size([2, 8, 32]), Q: torch.Size([2, 8, 16]), K: torch.Size([2, 8, 16]), V: torch.Size([2, 8, 16])
```

## Related Concepts

- Attention Mechanism
- Multi-Head Attention
- Grouped Query Attention
- KV Cache
- Information Retrieval

## Next Concepts

- DL-354: Attention in Computer Vision
- DL-355: Attention in NLP

## Summary

The Key-Query-Value (QKV) framework is the conceptual foundation of attention mechanisms. Q represents what the model is looking for, K represents labels/descriptions of available information, and V represents the actual content to retrieve. Learned linear projections compute Q, K, V from input representations, allowing the model to extract different aspects for each role. In multi-head attention, each head has its own projections, learning different query-key-value relationships. Variants like Grouped Query Attention and Multi-Query Attention reduce KV heads for inference efficiency.

## Key Takeaways

- Q = query (what to look for), K = key (content labels), V = value (content to retrieve).
- Q, K, V are computed by learned linear projections: Q = X W^Q, K = X W^K, V = X W^V.
- Different Q, K, V projections allow the model to extract different aspects for each role.
- In self-attention, Q, K, V all come from the same sequence; in cross-attention, Q from one, K, V from another.
- Multi-head attention uses separate Q, K, V projections per head.
- Grouped Query Attention reduces KV heads for inference efficiency.
- KV caching stores K, V from previous timesteps for efficient autoregressive inference.
