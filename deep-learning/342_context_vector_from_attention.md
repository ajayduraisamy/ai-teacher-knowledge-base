# Concept: Context Vector from Attention

## Concept ID

DL-342

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Attention Mechanisms

## Learning Objectives

- Understand how the attention mechanism produces a dynamic context vector from a weighted sum of values.
- Differentiate between the fixed context vector in basic seq2seq and the dynamic context vector in attention.
- Implement context vector computation using attention weights in PyTorch.
- Analyze how the context vector aggregates information from multiple input positions.
- Recognize the context vector as the bridge between attention weights and the final output.

## Prerequisites

- Understanding of attention scores and attention weights.
- Familiarity with the seq2seq context vector bottleneck.
- Knowledge of weighted sums and softmax normalization.
- Experience with PyTorch tensor operations (bmm, matmul).

## Definition

The context vector from attention is a dynamic, query-specific vector computed as a weighted sum of value vectors, where the weights are the attention weights. Formally, given value vectors V = (v_1, ..., v_T) and attention weights alpha = (alpha_1, ..., alpha_T), the context vector c is:

c = sum_{i=1}^{T} alpha_i * v_i

The context vector aggregates information from all input positions, weighted by their relevance to the current query. Unlike the fixed context vector in basic seq2seq models (which is just the final encoder state), the dynamic context vector in attention changes for each query (e.g., each decoder timestep). This allows the model to retrieve different information from the input depending on what is needed at each generation step. The context vector is then combined with the query or decoder state to produce the final output.

## Intuition

Imagine you are a detective investigating a crime scene. At each step of your investigation, you look at different pieces of evidence. When identifying the suspect, you focus on witness statements and video footage. When determining the motive, you focus on financial records and social media. The attention context vector is like your current understanding of the case based on the evidence you're focusing on right now. It's a dynamic summary that changes as your investigation progresses. In the same way, a seq2seq decoder at timestep 1 might compute a context vector focused on the subject of the input sentence, while at timestep 2 it focuses on the verb. Each context vector is a query-specific condensation of the relevant information, assembled from the full set of available evidence (the value vectors).

## Why This Concept Matters

The context vector is the output of the attention mechanism and the key artifact that bridges attention computation with downstream model components. Understanding how context vectors are computed and used is essential for: (1) integrating attention into existing architectures (e.g., how to combine context with decoder states). (2) Debugging attention — if context vectors are dominated by a few positions or contain no useful information, the attention mechanism may be failing. (3) Designing variations — some models use the context vector in multiple ways (e.g., feeding it through the RNN, concatenating it with decoder output, using it for both prediction and state update). (4) Understanding cross-attention in transformers — the decoder's cross-attention layers produce context vectors from encoder outputs, enabling the decoder to retrieve encoder information at each layer.

## Mathematical Explanation

### Basic Context Vector

Given attention weights alpha in R^T and values V in R^{T x d_v}:

c = sum_{i=1}^{T} alpha_i * v_i = V^T alpha

In matrix form for batched attention with multiple queries:

C = softmax(Q K^T / sqrt(d_k)) V

where C in R^{batch x seq_q x d_v}, Q in R^{batch x seq_q x d_k}, K in R^{batch x seq_k x d_k}, V in R^{batch x seq_k x d_v}.

### Context Vector in Seq2Seq with Attention

In Bahdanau-style attention:

c_{t'} = sum_{t=1}^{T} alpha_{t', t} * h_t

where h_t are encoder hidden states (acting as both keys and values).

This context vector is then used in the decoder update:

s_{t'} = f(s_{t'-1}, y_{t'-1}, c_{t'})

Or in Luong-style attention:

c_{t'} = sum_{t=1}^{T} alpha_{t', t} * h_t

tilde_s_{t'} = tanh(W_c [s_{t'}; c_{t'}])

### Context Vector in Transformers

In transformer cross-attention:

C = softmax(Q_decoder K_encoder^T / sqrt(d_k)) V_encoder

Each decoder layer computes a new context vector from the encoder outputs, allowing the decoder to iteratively refine its understanding of the source.

### Context Vector Properties

- The context vector lies in the convex hull of the value vectors.
- It is a point estimate that averages information, potentially losing fine-grained details.
- The effective degrees of freedom are bounded by the entropy of the attention distribution.

## Code Examples

### Example 1: Computing Context Vectors

```python
import torch
import torch.nn.functional as F
import math

def compute_context(attn_weights, values):
    return torch.bmm(attn_weights, values)

batch, n_queries, n_keys, d_v = 2, 4, 6, 8
attn_weights = F.softmax(torch.randn(batch, n_queries, n_keys), dim=-1)
values = torch.randn(batch, n_keys, d_v)
context = compute_context(attn_weights, values)
print(f"Context shape: {context.shape}")
print(f"Context min/max: {context.min().item():.3f}/{context.max().item():.3f}")
# Output: Context shape: torch.Size([2, 4, 8])
# Output: Context min/max: -0.823/0.912
```

### Example 2: Context Vector in Seq2Seq with Attention

```python
import torch.nn as nn

class AttentionWithContext(nn.Module):
    def __init__(self, enc_dim, dec_dim, d_v):
        super().__init__()
        self.attn = nn.Linear(enc_dim + dec_dim, 1)
        self.W_c = nn.Linear(enc_dim + dec_dim, d_v)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        src_len = encoder_outputs.shape[1]
        dec_hidden = decoder_hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
        energy = self.attn(torch.cat((encoder_outputs, dec_hidden), dim=2)).squeeze(2)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(energy, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        combined = torch.cat((context, decoder_hidden[-1]), dim=1)
        output = self.W_c(combined)
        return output, attn_weights

enc_dim, dec_dim, d_v = 32, 32, 64
attn = AttentionWithContext(enc_dim, dec_dim, d_v)
dec_hid = torch.randn(1, 2, dec_dim)
enc_out = torch.randn(2, 5, enc_dim)
context, weights = attn(dec_hid, enc_out)
print(f"Output context shape: {context.shape}")
# Output: Output context shape: torch.Size([2, 64])
```

### Example 3: Analyzing Context Vector Content

```python
def analyze_context_content(attn_weights, values):
    context = torch.bmm(attn_weights, values)
    context_norm = context.norm(dim=-1)
    dominant_contributor = attn_weights.argmax(dim=-1)
    entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1)
    return context, context_norm, dominant_contributor, entropy

attn_weights = F.softmax(torch.randn(2, 4, 10), dim=-1)
values = torch.randn(2, 10, 16)
context, norms, dominant, entropy = analyze_context_content(attn_weights, values)
print(f"Context norms: {norms}")
print(f"Dominant positions: {dominant}")
print(f"Entropy: {entropy}")
# Output: Context norms: tensor([[2.34, 1.89, 2.56, 2.12], [2.01, 2.45, 1.78, 2.33]])
# Output: Dominant positions: tensor([[5, 2, 8, 3], [1, 7, 4, 6]])
# Output: Entropy: tensor([[1.82, 1.94, 1.67, 1.88], [1.91, 1.73, 1.98, 1.79]])
```

## Common Mistakes

1. **Using attention weights directly as the output instead of computing the context vector**: Attention weights indicate relevance, but they are not the contextualized representation. The context vector is the weighted sum of values, which is a different quantity.

2. **Forgetting to unsqueeze attention weights for batch matrix multiplication**: torch.bmm requires (batch, seq_q, seq_k) and (batch, seq_k, d_v). The attention weights often need to be unsqueezed to add the query dimension.

3. **Computing the context vector but not using it in the output**: The context vector must be combined with the decoder state or query. Computing attention and the context vector but then ignoring it in the final prediction is a common architectural mistake.

4. **Using the same context vector for all queries**: Even with attention, if the score computation is flawed (e.g., all scores are equal), the context vector becomes the average of all values, losing query-specific information.

5. **Not projecting the context vector to the appropriate dimension**: The context vector has dimension d_v (value dimension), which may differ from the decoder hidden dimension. A linear projection is needed to combine them.

## Interview Questions

### Beginner

Q: How is the context vector computed from attention weights and values?

A: The context vector is the weighted sum of value vectors, where the weights are the attention weights. For each query, c = sum_i alpha_i * v_i. This produces a vector that aggregates information from all input positions, weighted by their relevance.

### Intermediate

Q: How does the dynamic context vector in attention differ from the fixed context vector in basic seq2seq?

A: The basic seq2seq context vector is a single fixed vector (the final encoder hidden state) that contains a compressed representation of the entire input sequence. The attention context vector is dynamic: a new context is computed for each decoder timestep as a weighted sum of all encoder states, focusing on different parts of the input for different output tokens.

### Advanced

Q: In transformer cross-attention, the context vector is computed as softmax(QK^T)V. How would you modify this to allow the model to attend to both local and global information simultaneously, and what would the resulting context vector represent?

A: I would use multi-scale context vectors by combining multiple attention computations: (1) A local context attending to a sliding window of nearby positions (fine-grained, local patterns). (2) A global context attending to a subsampled set of positions (coarse-grained, long-range patterns). (3) A learned gating mechanism to blend the two contexts: c_total = g * c_local + (1-g) * c_global, where g is a learned gate vector. The resulting context vector would simultaneously contain fine-grained local information (useful for syntax and nearby context) and coarse-grained global information (useful for topic and long-range dependencies). This multi-scale approach is used in architectures like Longformer and BigBird.

## Practice Problems

### Easy

Write a function that takes attention weights (batch, seq_q, seq_k) and values (batch, seq_k, d_v) and returns the context vector. Verify the output shape.

### Medium

Implement a comparison between context vectors from uniform attention (all weights equal) and peaked attention (one weight close to 1). Show that the uniform context vector approximates the mean of the values.

### Hard

Implement a gated context vector fusion mechanism that combines context vectors from two different attention layers (e.g., from encoder-decoder attention and self-attention) using a learned gate. Show that this improves performance on a translation task.

## Solutions

### Easy Solution

```python
def attention_context(attn_weights, values):
    return torch.bmm(attn_weights, values)

batch, n_q, n_k, d = 2, 3, 5, 8
weights = F.softmax(torch.randn(batch, n_q, n_k), dim=-1)
values = torch.randn(batch, n_k, d)
ctx = attention_context(weights, values)
print(f"Context shape: {ctx.shape}")
# Output: Context shape: torch.Size([2, 3, 8])
```

## Related Concepts

- Attention Weights
- Value Vectors in Attention
- Seq2Seq with Attention
- Cross-Attention
- Transformer Decoder

## Next Concepts

- DL-343: Self-Attention
- DL-344: Cross-Attention
- DL-346: Multi-Head Attention

## Summary

The context vector from attention is the output of the attention mechanism: a weighted sum of value vectors that aggregates information from all input positions weighted by their relevance. Unlike the fixed context vector in basic seq2seq models, the attention context vector is dynamic and query-specific. It serves as the bridge between attention computation and downstream model components, providing a query-specific representation of the input that the model uses for prediction. In transformers, context vectors are computed at each layer in both self-attention and cross-attention, enabling deep, hierarchical information retrieval.

## Key Takeaways

- The context vector is a weighted sum of values: c = sum_i alpha_i * v_i.
- It is dynamic and query-specific, unlike the fixed context vector in basic seq2seq.
- The context vector lies in the convex hull of value vectors.
- It aggregates information from all positions, weighted by relevance.
- The context vector must be combined with the query or decoder state for the final output.
- In transformers, context vectors are computed at each layer for both self and cross attention.
