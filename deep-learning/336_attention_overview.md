# Concept: Attention Overview

## Concept ID

DL-336

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Attention Mechanisms

## Learning Objectives

- Define the attention mechanism as a general neural network component for selectively focusing on relevant information.
- Understand the three core components: query, key, and value in attention computation.
- Differentiate between the major types of attention: self-attention, cross-attention, and encoder-decoder attention.
- Recognize attention as a universal building block used across NLP, computer vision, and multimodal learning.
- Implement a basic attention mechanism in PyTorch.

## Prerequisites

- Understanding of seq2seq models and their limitations.
- Basic familiarity with neural network architectures and linear transformations.
- Knowledge of softmax normalization and weighted sums.
- Some exposure to transformer architectures is helpful but not required.

## Definition

Attention is a neural network mechanism that allows a model to selectively focus on specific parts of the input when producing each element of the output. Formally, attention computes a weighted sum of values, where the weights are determined by the compatibility between a query and a set of keys. Given a query q, a set of keys K = (k_1, ..., k_T), and a set of values V = (v_1, ..., v_T), attention computes:

Attention(q, K, V) = sum_{i=1}^{T} alpha_i * v_i

where alpha_i = softmax(score(q, k_i)) are attention weights normalized to sum to 1. The score function can be additive (Bahdanau), multiplicative (Luong), or scaled dot-product (Vaswani). Attention is a differentiable module that can be inserted into any neural architecture, enabling the model to learn which input elements are relevant for each output element. It has become a fundamental building block in modern deep learning, forming the core of transformer architectures and powering applications from machine translation to image captioning to protein folding.

## Intuition

Think of attention as a spotlight that the model can shine on different parts of its input. When you read a sentence, your eyes don't process every word equally — you focus on the important words based on what you need to understand. When you see the word "ate," your attention shifts to the subject (who ate?) and the object (ate what?). Attention mechanisms work the same way. The query is like a question the model is asking: "What information do I need right now?" The keys are like labels on the available information, and the values are the actual information content. The model compares the query to each key to decide which information to focus on (attention weights), then reads the corresponding values. This process is repeated for each output the model produces, allowing it to dynamically access the most relevant information at each step.

## Why This Concept Matters

Attention is arguably the most important architectural innovation in deep learning of the past decade. It solves the fundamental limitation of fixed-context representations in earlier models by providing dynamic, query-specific access to input information. Attention enabled the transformer architecture, which replaced recurrent and convolutional networks as the dominant paradigm in NLP and is now spreading to computer vision, speech, and scientific applications. Attention also provides interpretability — by visualizing attention weights, we can understand what information the model uses to make decisions. The attention mechanism's flexibility and effectiveness have made it a universal building block, and understanding attention is now essential for virtually any deep learning practitioner. This concept overview provides the foundation for all subsequent concepts on specific attention types and applications.

## Mathematical Explanation

### General Attention Formulation

Let:
- Q in R^{d_q} be the query vector
- K in R^{T x d_k} be the key matrix (T keys, each of dimension d_k)
- V in R^{T x d_v} be the value matrix (T values, each of dimension d_v)

### Score Computation

e_i = score(q, k_i)

Common score functions:
- Additive: score(q, k) = v^T tanh(W_q q + W_k k)
- Multiplicative: score(q, k) = q^T W k
- Scaled dot-product: score(q, k) = q^T k / sqrt(d_k)

### Attention Weights

alpha_i = softmax(e_i) = exp(e_i) / sum_{j=1}^{T} exp(e_j)

### Context Vector

c = sum_{i=1}^{T} alpha_i * v_i

### Multi-Head Attention

Multiple attention heads run in parallel:

head_h = Attention(Q W_h^Q, K W_h^K, V W_h^V)
MultiHead(Q, K, V) = Concat(head_1, ..., head_H) W^O

### Self-Attention vs. Cross-Attention

- Self-attention: Q, K, V all come from the same sequence (each token attends to all tokens including itself).
- Cross-attention: Q comes from one sequence (e.g., decoder), K and V come from another sequence (e.g., encoder).

## Code Examples

### Example 1: Simple Scaled Dot-Product Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
        self.scale = d_k ** 0.5

    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

d_k = 8
attn = ScaledDotProductAttention(d_k)
q = torch.randn(2, 4, d_k)
k = torch.randn(2, 6, d_k)
v = torch.randn(2, 6, d_k)
output, weights = attn(q, k, v)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"Sample weights (first batch, first head):\n{weights[0, 0]}")
# Output: Output shape: torch.Size([2, 4, 8])
# Output: Attention weights shape: torch.Size([2, 4, 6])
# Output: Sample weights (first batch, first head):
# Output: tensor([0.15, 0.12, 0.25, 0.18, 0.20, 0.10])
```

### Example 2: Additive (Bahdanau) Attention

```python
class AdditiveAttention(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim):
        super().__init__()
        self.W_q = nn.Linear(query_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(key_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query, keys, mask=None):
        query = query.unsqueeze(1)
        energy = torch.tanh(self.W_q(query) + self.W_k(keys))
        scores = self.v(energy).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)
        return context, attn_weights

attn_add = AdditiveAttention(query_dim=8, key_dim=8, hidden_dim=16)
query = torch.randn(2, 8)
keys = torch.randn(2, 6, 8)
context, weights = attn_add(query, keys)
print(f"Additive attention context shape: {context.shape}")
# Output: Additive attention context shape: torch.Size([2, 8])
```

### Example 3: Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
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

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        Q = self.W_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        return output, attn

mha = MultiHeadAttention(d_model=64, n_heads=8)
x = torch.randn(2, 10, 64)
y = torch.randn(2, 12, 64)
output, attn = mha(x, y, y)
print(f"Multi-head output shape: {output.shape}")
print(f"Attention shape: {attn.shape}")
# Output: Multi-head output shape: torch.Size([2, 10, 64])
# Output: Attention shape: torch.Size([2, 8, 10, 12])
```

## Common Mistakes

1. **Confusing self-attention with encoder-decoder attention**: Self-attention processes a single sequence (each token attends to all tokens in the same sequence). Encoder-decoder attention (cross-attention) allows one sequence to attend to another sequence. These have different purposes and are used in different parts of the architecture.

2. **Forgetting to scale dot-product attention**: Without scaling by sqrt(d_k), the dot products grow large in magnitude, pushing the softmax into regions of extremely small gradients, which hurts training.

3. **Ignoring the quadratic complexity of self-attention**: Self-attention has O(T^2) time and memory complexity in sequence length. For long sequences, this becomes prohibitive and requires sparse or linear attention alternatives.

4. **Not masking future tokens in decoder self-attention**: In autoregressive decoding, each token should only attend to itself and previous tokens (causal masking). Without this mask, the decoder sees future tokens and learns to cheat.

5. **Applying attention without considering positional information**: Attention is permutation-invariant — it treats the input as a set, not a sequence. Without positional encodings, the model loses all information about token order, which is critical for most sequence tasks.

## Interview Questions

### Beginner

Q: What are the three components of the attention mechanism, and what role does each play?

A: The three components are queries (Q), keys (K), and values (V). The query represents what the model is looking for. The keys are labels on the available information that are compared to the query to compute attention weights. The values are the actual information content that is weighted by the attention weights and summed to produce the output.

### Intermediate

Q: Why is scaled dot-product attention preferred over additive attention in transformer models?

A: Scaled dot-product attention is faster and more space-efficient because it can be implemented using highly optimized matrix multiplication code. It also has better theoretical properties: with proper scaling, the dot products have unit variance, preventing the softmax from entering the saturation regime. However, for very large d_k, additive attention may be more stable because it avoids the large dot products that can overwhelm the softmax even with scaling.

### Advanced

Q: Explain the relationship between attention and the concept of "soft dictionary lookup." How does this perspective help in understanding attention's role in transformer architectures?

A: Attention can be viewed as a differentiable, soft dictionary lookup where keys are matched against a query, and the values of the matching keys are retrieved. Unlike a hard dictionary (which returns exactly one value), attention returns a weighted combination of all values. This perspective helps understand transformers as follows: (1) self-attention layers allow each token to retrieve information from all other tokens in the sequence, acting as a learnable, contextualized lookup. (2) Multi-head attention performs multiple lookups in parallel, each focusing on different types of relationships (syntactic, semantic, positional). (3) Cross-attention layers allow the decoder to retrieve information from the encoder, acting as a bridge between the two sequences. The softness of the lookup ensures differentiability, while the weighted combination allows the model to blend information from multiple sources.

## Practice Problems

### Easy

Implement a simple attention mechanism that takes a single query and a set of key-value pairs, computes dot-product attention scores, and returns the weighted sum of values.

### Medium

Implement a function that visualizes the attention weights of a pre-trained transformer model for a given input sentence. Show which tokens attend to which other tokens for a specific layer and head.

### Hard

Implement a comparative analysis of additive attention vs. scaled dot-product attention on a synthetic sequence alignment task. Measure both the quality of the learned alignments and the computational efficiency.

## Solutions

### Easy Solution

```python
def simple_attention(query, keys, values):
    scores = torch.matmul(query, keys.transpose(-2, -1))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, values)
    return output, attn_weights

q = torch.randn(1, 4)
k = torch.randn(5, 4)
v = torch.randn(5, 8)
out, w = simple_attention(q, k, v)
print(f"Output: {out.shape}, Weights: {w.shape}")
# Output: Output: torch.Size([1, 8]), Weights: torch.Size([1, 5])
```

## Related Concepts

- Self-Attention and Cross-Attention
- Multi-Head Attention
- Scaled Dot-Product Attention
- Transformer Architecture
- Positional Encoding

## Next Concepts

- DL-337: Bahdanau Attention
- DL-338: Luong Attention
- DL-343: Self-Attention

## Summary

Attention is a fundamental neural mechanism that allows models to selectively focus on relevant parts of the input when producing output. It computes a weighted sum of values, where weights are determined by the compatibility between a query and a set of keys. Attention comes in several varieties — additive, multiplicative, scaled dot-product — and can be applied as self-attention (within a sequence) or cross-attention (between sequences). Multi-head attention runs multiple attention computations in parallel, capturing different types of relationships. Attention is the core building block of transformer architectures and has revolutionized deep learning across NLP, computer vision, and beyond.

## Key Takeaways

- Attention computes a weighted sum of values based on query-key compatibility.
- The three components are query (what to look for), key (what is available), and value (content to retrieve).
- Attention mechanisms are differentiable and can be inserted into any neural architecture.
- Self-attention operates within a single sequence; cross-attention operates between sequences.
- Multi-head attention runs parallel attention computations for richer representations.
- Attention with proper scaling and positional encoding is the foundation of transformer models.
