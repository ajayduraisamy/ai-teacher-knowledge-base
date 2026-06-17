# Concept: Attention Weights

## Concept ID

DL-341

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Attention Mechanisms

## Learning Objectives

- Define attention weights as normalized importance scores over input elements.
- Understand the softmax normalization process that converts raw scores to probability distributions.
- Implement attention weight computation and interpret their meaning.
- Analyze the properties of attention weight distributions: sparsity, entropy, and peakiness.
- Apply masking techniques to control where attention weights can be non-zero.

## Prerequisites

- Understanding of attention scores and score functions.
- Familiarity with softmax normalization.
- Knowledge of padding and causal masking in sequence models.
- Experience with PyTorch tensor operations and masking.

## Definition

Attention weights are a probability distribution over input elements that determine how much each element contributes to the attention output. Formally, given a set of raw attention scores e = (e_1, ..., e_T) where e_i is the compatibility between a query and key i, the attention weights alpha = (alpha_1, ..., alpha_T) are computed via softmax normalization:

alpha_i = softmax(e)_i = exp(e_i) / sum_{j=1}^{T} exp(e_j)

The attention weights satisfy alpha_i >= 0 and sum_i alpha_i = 1. The output of the attention mechanism is then the weighted sum of values: c = sum_i alpha_i * v_i. The attention weights are the core output of the attention mechanism — they reveal what the model focuses on and provide interpretability. The distribution of attention weights can be peaked (focusing on few elements), flat (spreading across many elements), or structured (following patterns like diagonal alignment in translation).

## Intuition

Attention weights are like a budget you allocate across different information sources. If you have $1.00 to spend on information gathering, you spend $0.70 on the most relevant source, $0.20 on the second most, and $0.10 split across the rest. The softmax function ensures you always spend exactly $1.00. When the model is very confident about which information is relevant, the weights become highly peaked — almost all the budget goes to one source. When the model is uncertain or multiple sources are equally relevant, the weights are spread more evenly. In machine translation, attention weights often form a diagonal pattern: the first target word attends to the first source word, the second to the second, and so on. This reveals that the model has learned a monotonic alignment between the languages. In self-attention, the weights show which tokens in the same sentence are most relevant for understanding each token, revealing syntactic and semantic relationships.

## Why This Concept Matters

Attention weights are the primary mechanism through which attention models communicate relevance information. They determine the model's behavior: what it focuses on, how it blends information, and what it ignores. Understanding attention weights is essential for: (1) interpreting model predictions — visualizing attention weights reveals what the model "looks at" when making decisions. (2) Diagnosing model failures — if attention weights are distributed over padding tokens, the model has learned spurious patterns. (3) Applying masks correctly — causal masks, padding masks, and sparse attention patterns all work by manipulating where attention weights can be non-zero. (4) Designing efficient attention — sparse attention mechanisms reduce computation by enforcing that only certain attention weights are computed. The properties of attention weights — their entropy, sparsity, and structure — are active areas of research with implications for model efficiency and interpretability.

## Mathematical Explanation

### Softmax Normalization

Given raw scores e in R^T:

alpha_i = exp(e_i) / sum_{j=1}^{T} exp(e_j)

Properties:
- alpha_i in (0, 1) for all i
- sum_i alpha_i = 1
- The distribution is invariant to adding a constant to all scores: softmax(e + c) = softmax(e)

### Temperature

With temperature parameter tau:

alpha_i = exp(e_i / tau) / sum_{j=1}^{T} exp(e_j / tau)

- tau -> 0: distribution becomes a one-hot (only max score gets weight 1)
- tau -> infinity: distribution becomes uniform
- tau = 1: standard softmax

### Entropy

The entropy of the attention distribution measures its peakiness:

H(alpha) = -sum_i alpha_i log(alpha_i)

- H = 0: all weight on one element (completely peaked)
- H = log(T): uniform distribution (completely flat)

### Masking

Padding mask: set e_i = -infinity for padding positions, so alpha_i = 0

Causal mask: for position j, set e_i = -infinity for all i > j, preventing attending to future tokens

### Sparse Attention

In sparse attention, most attention weights are forced to zero by setting corresponding scores to -infinity. Common patterns:
- Sliding window: attend to nearby tokens within a window
- Global: certain tokens (like [CLS]) attend to all tokens
- Dilated: skip tokens with a fixed stride

## Code Examples

### Example 1: Computing and Analyzing Attention Weights

```python
import torch
import torch.nn.functional as F
import math

def compute_attention_weights(scores, mask=None, temperature=1.0):
    scores = scores / temperature
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    return weights

scores = torch.randn(2, 4, 6)
weights = compute_attention_weights(scores)
print(f"Weights shape: {weights.shape}")
print(f"Sum of weights (per query): {weights.sum(dim=-1)}")
print(f"Sample weights:\n{weights[0, 0]}")
# Output: Weights shape: torch.Size([2, 4, 6])
# Output: Sum of weights (per query): tensor([[1., 1., 1., 1.], [1., 1., 1., 1.]])
# Output: Sample weights:
# Output: tensor([0.15, 0.08, 0.22, 0.19, 0.25, 0.11])
```

### Example 2: Effect of Temperature on Attention Weights

```python
def analyze_temperature(scores, temperatures):
    results = {}
    for tau in temperatures:
        weights = compute_attention_weights(scores, temperature=tau)
        entropy = -(weights * torch.log(weights + 1e-8)).sum(-1).mean().item()
        max_w = weights.max(-1).values.mean().item()
        results[tau] = (entropy, max_w)
    return results

scores = torch.randn(1, 1, 10)
temps = [0.1, 0.5, 1.0, 2.0, 5.0]
results = analyze_temperature(scores, temps)
for tau, (entropy, max_w) in results.items():
    print(f"T={tau:.1f}: entropy={entropy:.3f}, max_weight={max_w:.3f}")
# Output: T=0.1: entropy=0.000, max_weight=1.000
# Output: T=0.5: entropy=0.891, max_weight=0.612
# Output: T=1.0: entropy=1.892, max_weight=0.312
# Output: T=2.0: entropy=2.101, max_weight=0.175
# Output: T=5.0: entropy=2.291, max_weight=0.128
```

### Example 3: Padding and Causal Masking

```python
def apply_padding_mask(scores, lengths):
    batch, seq_q, seq_k = scores.shape
    mask = torch.arange(seq_k).unsqueeze(0).unsqueeze(0) < lengths.unsqueeze(1).unsqueeze(2)
    mask = mask.expand(-1, seq_q, -1)
    return mask.float()

def apply_causal_mask(scores):
    seq_len = scores.shape[-1]
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
    return mask

scores = torch.randn(2, 4, 6)
lengths = torch.tensor([4, 5])
padding_mask = apply_padding_mask(scores, lengths)
weights_padded = compute_attention_weights(scores, mask=padding_mask)
print(f"Weights with padding mask (first batch, first query):")
print(f"  Non-zero: {(weights_padded[0, 0] > 0).sum().item()}, Sum: {weights_padded[0, 0].sum().item():.2f}")
# Output: Weights with padding mask (first batch, first query):
# Output:   Non-zero: 4, Sum: 1.00

causal_scores = torch.randn(1, 5, 5)
causal_mask = apply_causal_mask(causal_scores)
weights_causal = compute_attention_weights(causal_scores, mask=causal_mask)
print(f"Causal weights:\n{weights_causal[0]}")
# Output: Causal weights:
# Output: tensor([[1.00, 0.00, 0.00, 0.00, 0.00],
# Output:         [0.62, 0.38, 0.00, 0.00, 0.00],
# Output:         [0.40, 0.35, 0.25, 0.00, 0.00],
# Output:         [0.30, 0.28, 0.25, 0.17, 0.00],
# Output:         [0.22, 0.21, 0.20, 0.19, 0.18]])
```

## Common Mistakes

1. **Applying masking after softmax instead of before**: Masks must be applied to raw scores (by setting masked positions to -infinity) before softmax. Applying masks after softmax (e.g., zeroing out weights) breaks the sum-to-1 property.

2. **Using -inf instead of a very negative number**: In practice, use -1e9 as a substitute for -infinity to avoid numerical issues. True -inf can cause NaN in gradient computation.

3. **Forgetting that attention weights should sum to 1**: If weights don't sum to 1 (due to numerical issues, incorrect masking, or softmax on the wrong dimension), the attention output is incorrectly scaled.

4. **Ignoring the effect of padding on gradient computation**: When padded positions contribute to attention weights, gradients flow through them. Using proper masking ensures gradients only flow through valid positions.

5. **Using causal mask for non-autoregressive tasks**: Causal masks should only be used in autoregressive decoding to prevent attending to future tokens. Using them in encoders or bidirectional tasks unnecessarily restricts the model.

## Interview Questions

### Beginner

Q: How are attention weights computed from raw scores?

A: Attention weights are computed by applying softmax to the raw scores: alpha_i = exp(e_i) / sum_j exp(e_j). This produces a probability distribution where each weight is between 0 and 1 and all weights sum to 1.

### Intermediate

Q: What is the effect of temperature on attention weights, and when would you use a temperature different from 1?

A: Temperature controls the peakiness of the attention distribution. Lower temperature (tau < 1) makes the distribution more peaked, focusing on fewer elements. Higher temperature (tau > 1) makes it more uniform. Temperature != 1 is used to control attention sharpness in knowledge distillation, to enforce sparsity, or to adjust model confidence in specific tasks.

### Advanced

Q: How would you design attention weights for a long-document transformer that needs to handle sequences of 100K+ tokens? What masking strategies would you use?

A: For very long sequences, full O(T^2) attention is infeasible. I would use a combination of sparse attention patterns: (1) Sliding window attention: each token attends to a local window of size w around it (e.g., w=256), capturing local context. (2) Global tokens: a small set of tokens (e.g., [CLS] tokens every 512 tokens) attend to the full sequence and vice versa, capturing long-range dependencies. (3) Dilated sliding window: stack multiple layers with different dilation rates to increase the receptive field exponentially. (4) Combining these patterns via a mixture of heads: some heads use sliding window, some use global attention, some use random attention. The masking for each pattern is implemented by setting out-of-window scores to -infinity before softmax.

## Practice Problems

### Easy

Write a function that takes raw attention scores and returns attention weights. Verify that the weights sum to 1 and are non-negative.

### Medium

Implement a function that visualizes the attention weight matrix as a heatmap. Given source tokens ["I", "love", "deep", "learning"] and target tokens ["J", "aime", "l_apprentissage", "profond"], show the attention alignment pattern.

### Hard

Implement a mixture-of-attention-heads where each head uses a different temperature. Show that this allows the model to simultaneously capture fine-grained (low temperature) and coarse-grained (high temperature) relationships.

## Solutions

### Easy Solution

```python
def scores_to_weights(scores, mask=None):
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(scores.shape[:-1])), "Weights must sum to 1"
    assert (weights >= 0).all(), "Weights must be non-negative"
    return weights

s = torch.randn(2, 3, 5)
w = scores_to_weights(s)
print(f"Weights verified: sum={w.sum(dim=-1)[0, 0].item():.2f}")
# Output: Weights verified: sum=1.00
```

## Related Concepts

- Attention Score
- Softmax Normalization
- Masking (Padding, Causal)
- Sparse Attention
- Temperature in Attention

## Next Concepts

- DL-342: Context Vector from Attention
- DL-348: Attention Temperature
- DL-345: Causal Masked Attention

## Summary

Attention weights are the normalized importance distribution over input elements in an attention mechanism. Computed via softmax from raw attention scores, they determine how much each value contributes to the attention output. The distribution of attention weights — their peakiness, entropy, and structure — reveals what the model focuses on and provides interpretability. Masking techniques manipulate attention weights by setting masked position scores to -infinity before softmax. Understanding attention weights is essential for implementing correct attention mechanisms, interpreting model behavior, and designing efficient sparse attention patterns.

## Key Takeaways

- Attention weights are computed by softmax normalization of raw scores.
- Weights form a probability distribution: non-negative and sum to 1.
- Temperature controls distribution peakiness (low temp = peaked, high temp = flat).
- Padding masks prevent attending to padding tokens.
- Causal masks prevent attending to future tokens in autoregressive decoding.
- Attention weight visualization is a primary tool for model interpretability.
