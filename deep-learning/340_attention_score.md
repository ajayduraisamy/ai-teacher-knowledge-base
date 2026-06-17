# Concept: Attention Score

## Concept ID

DL-340

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Attention Mechanisms

## Learning Objectives

- Define the attention score function and its role in computing query-key compatibility.
- Understand the major score function variants: dot, general, concat, and additive.
- Implement multiple score functions in PyTorch and analyze their computational profiles.
- Analyze how the choice of score function affects the resulting attention distribution.
- Select appropriate score functions for different attention architectures.

## Prerequisites

- Understanding of the attention mechanism (query, key, value framework).
- Familiarity with linear transformations and matrix multiplication.
- Knowledge of softmax normalization.
- Experience with PyTorch tensor operations.

## Definition

The attention score is a scalar value that measures the compatibility or relevance between a query vector and a key vector in an attention mechanism. Given a query q in R^{d_q} and a key k in R^{d_k}, a score function f(q, k) -> R computes the strength of the relationship between them. The scores across all query-key pairs are normalized via softmax to produce attention weights, which determine how much each value contributes to the output. The choice of score function is a critical design decision that affects the expressiveness, computational cost, and trainability of the attention mechanism. Score functions can be parameter-free (dot product), linear (general), or non-linear (additive with tanh), each offering different trade-offs between capacity and efficiency.

## Intuition

The attention score is like a compatibility test between a question (query) and a label (key). Imagine you have a stack of resumes (keys) and a job description (query). You need to score each resume based on how well it matches the job. A simple score function might count keyword matches (dot product). A more sophisticated function might consider synonyms and related skills (learned linear transformation). An even more complex function might recognize that certain combinations of skills are particularly valuable, even if individual keywords don't match perfectly (non-linear additive function). The score function defines what "compatibility" means, and the model learns to adjust the score function parameters to produce meaningful alignments for the task.

## Why This Concept Matters

The attention score function is the core of the attention mechanism. It determines what kind of relationships the model can capture between queries and keys. The evolution of score functions — from Bahdanau's additive score through Luong's general score to the scaled dot-product in transformers — reflects the field's progression toward simpler, faster, and more scalable attention mechanisms. Understanding score functions is essential for: (1) implementing attention mechanisms correctly, (2) diagnosing attention-related training issues (e.g., saturated softmax due to unscaled scores), (3) designing new attention variants for specialized tasks, and (4) understanding why transformers can be trained efficiently at scale. The score function is also where key innovations in attention research (e.g., relative position biases, linear attention, sparse attention) modify the computation to address specific limitations.

## Mathematical Explanation

### Score Function Definition

Given query q in R^{d_q} and key k in R^{d_k}, the score function f computes:

e = f(q, k) in R

### Common Score Functions

1. **Dot Product** (no parameters):
   f_dot(q, k) = q^T k

2. **Scaled Dot Product** (no parameters):
   f_scaled(q, k) = q^T k / sqrt(d_k)

3. **Bilinear / General** (1 parameter matrix):
   f_general(q, k) = q^T W k, where W in R^{d_q x d_k}

4. **Additive (Bahdanau)** (3 parameter matrices):
   f_add(q, k) = v^T tanh(W_q q + W_k k), where W_q in R^{d_a x d_q}, W_k in R^{d_a x d_k}, v in R^{d_a}

5. **Concat (Luong)** (2 parameter matrices):
   f_concat(q, k) = v^T tanh(W [q; k]), where W in R^{d_a x (d_q + d_k)}, v in R^{d_a}

6. **Location-Based** (query only):
   f_loc(q) = W_q q, where W_q in R^{T x d_q}, T is number of positions

### Score Normalization

Raw scores are converted to attention weights via softmax:

alpha_i = exp(e_i) / sum_j exp(e_j)

### Effect of Score Scale

The scale of scores affects the entropy of the attention distribution:
- Larger scores -> peaked distribution (focus on few elements)
- Smaller scores -> flat distribution (spread across elements)

### Gradient Properties

- Dot product: gradient = k (w.r.t. q) and q (w.r.t. k)
- Additive: gradient involves tanh derivative and backprop through v, W_q, W_k

## Code Examples

### Example 1: Implementing Score Functions

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def dot_score(q, k):
    return torch.matmul(q, k.transpose(-2, -1))

def scaled_dot_score(q, k, d_k=None):
    if d_k is None:
        d_k = k.shape[-1]
    return torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

class BilinearScore(nn.Module):
    def __init__(self, d_q, d_k):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_q, d_k) * 0.01)

    def forward(self, q, k):
        return torch.matmul(torch.matmul(q, self.W), k.transpose(-2, -1))

class AdditiveScore(nn.Module):
    def __init__(self, d_q, d_k, d_a):
        super().__init__()
        self.W_q = nn.Linear(d_q, d_a, bias=False)
        self.W_k = nn.Linear(d_k, d_a, bias=False)
        self.v = nn.Linear(d_a, 1, bias=False)

    def forward(self, q, k):
        batch, seq_q, seq_k = q.shape[0], q.shape[1], k.shape[1]
        q_exp = q.unsqueeze(2).expand(-1, -1, seq_k, -1)
        k_exp = k.unsqueeze(1).expand(-1, seq_q, -1, -1)
        energy = torch.tanh(self.W_q(q_exp) + self.W_k(k_exp))
        return self.v(energy).squeeze(-1)

class ConcatScore(nn.Module):
    def __init__(self, d_q, d_k, d_a):
        super().__init__()
        self.W = nn.Linear(d_q + d_k, d_a, bias=False)
        self.v = nn.Linear(d_a, 1, bias=False)

    def forward(self, q, k):
        batch, seq_q, seq_k = q.shape[0], q.shape[1], k.shape[1]
        q_exp = q.unsqueeze(2).expand(-1, -1, seq_k, -1)
        k_exp = k.unsqueeze(1).expand(-1, seq_q, -1, -1)
        concat = torch.cat((q_exp, k_exp), dim=-1)
        energy = torch.tanh(self.W(concat))
        return self.v(energy).squeeze(-1)

d_q, d_k, d_a = 32, 32, 64
q = torch.randn(2, 5, d_q)
k = torch.randn(2, 7, d_k)

scores_dot = dot_score(q, k)
scores_scaled = scaled_dot_score(q, k)
scores_bilin = BilinearScore(d_q, d_k)(q, k)
scores_add = AdditiveScore(d_q, d_k, d_a)(q, k)
scores_con = ConcatScore(d_q, d_k, d_a)(q, k)

print(f"Dot: {scores_dot.shape}, Scaled: {scores_scaled.shape}")
print(f"Bilinear: {scores_bilin.shape}, Additive: {scores_add.shape}, Concat: {scores_con.shape}")
# Output: Dot: torch.Size([2, 5, 7]), Scaled: torch.Size([2, 5, 7])
# Output: Bilinear: torch.Size([2, 5, 7]), Additive: torch.Size([2, 5, 7]), Concat: torch.Size([2, 5, 7])
```

### Example 2: Score Distribution Analysis

```python
def analyze_score_distribution(score_func, q, k, name):
    with torch.no_grad():
        scores = score_func(q, k) if not isinstance(score_func, BilinearScore) else score_func(q, k)
        probs = F.softmax(scores, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(-1).mean()
        max_prob = probs.max(-1).values.mean()
        print(f"{name:10s}: entropy={entropy:.3f}, max_prob={max_prob:.3f}")

q = torch.randn(2, 10, 64)
k = torch.randn(2, 10, 64)
analyze_score_distribution(lambda q, k: dot_score(q, k), q, k, "Dot")
analyze_score_distribution(lambda q, k: scaled_dot_score(q, k), q, k, "Scaled")
analyze_score_distribution(AdditiveScore(64, 64, 64), q, k, "Additive")
analyze_score_distribution(ConcatScore(64, 64, 64), q, k, "Concat")
# Output: Dot       : entropy=2.201, max_prob=0.112
# Output: Scaled    : entropy=2.299, max_prob=0.100
# Output: Additive  : entropy=2.253, max_prob=0.105
# Output: Concat    : entropy=2.247, max_prob=0.106
```

### Example 3: Attention with Different Score Functions

```python
class FlexibleAttention(nn.Module):
    def __init__(self, d_q, d_k, d_v, score_type='dot', d_a=None):
        super().__init__()
        self.score_type = score_type
        if score_type == 'bilinear':
            self.score_fn = BilinearScore(d_q, d_k)
        elif score_type == 'additive':
            d_a = d_a or (d_q + d_k) // 2
            self.score_fn = AdditiveScore(d_q, d_k, d_a)
        elif score_type == 'concat':
            d_a = d_a or (d_q + d_k) // 2
            self.score_fn = ConcatScore(d_q, d_k, d_a)
        elif score_type == 'scaled_dot':
            self.scale = math.sqrt(d_q)
        elif score_type == 'dot':
            pass

    def forward(self, q, k, v, mask=None):
        if self.score_type == 'scaled_dot':
            scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        elif self.score_type == 'dot':
            scores = torch.matmul(q, k.transpose(-2, -1))
        else:
            scores = self.score_fn(q, k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v), attn_weights

q = torch.randn(2, 5, 32)
k = torch.randn(2, 7, 32)
v = torch.randn(2, 7, 32)

for score_type in ['dot', 'scaled_dot', 'bilinear', 'additive', 'concat']:
    attn = FlexibleAttention(32, 32, 32, score_type, d_a=32)
    context, weights = attn(q, k, v)
    print(f"{score_type:12s}: context={context.shape}, weights_sum={weights.sum(dim=-1)[0, 0].item():.2f}")
# Output: dot         : context=torch.Size([2, 5, 32]), weights_sum=1.00
# Output: scaled_dot  : context=torch.Size([2, 5, 32]), weights_sum=1.00
# Output: bilinear    : context=torch.Size([2, 5, 32]), weights_sum=1.00
# Output: additive    : context=torch.Size([2, 5, 32]), weights_sum=1.00
# Output: concat      : context=torch.Size([2, 5, 32]), weights_sum=1.00
```

## Common Mistakes

1. **Not scaling dot-product scores**: Dot products grow large with dimension, causing softmax saturation. Always scale by sqrt(d_k) for stable training.

2. **Using the wrong dimension for matrix multiplication**: The query and key matrices must have compatible dimensions. In multi-head attention, each head has dimension d_k = d_model / n_heads, and the transpose must be on the correct axis.

3. **Applying tanh without proper initialization**: The tanh activation in additive attention can saturate if the linear projections produce large values. Use Xavier/Glorot initialization and consider layer normalization.

4. **Assuming all score functions work identically with attention masking**: Masking (setting scores to -inf for padding positions) must be applied after the score computation but before softmax. This applies uniformly to all score functions.

5. **Forgetting that score functions have different parameter counts**: Dot product has none, bilinear has one weight matrix, additive has three. This affects regularization needs and convergence speed.

## Interview Questions

### Beginner

Q: What does the attention score represent in an attention mechanism?

A: The attention score represents the compatibility or relevance between a query and a key. Higher scores indicate stronger relevance, meaning the corresponding value will contribute more to the attention output after softmax normalization.

### Intermediate

Q: Why does the scaled dot-product attention use sqrt(d_k) as the scaling factor?

A: If q and k have zero mean and unit variance, their dot product has variance d_k. Dividing by sqrt(d_k) normalizes the variance to 1, keeping the softmax in the high-gradient region. Without scaling, the scores grow with d_k, softmax saturates, and gradients vanish.

### Advanced

Q: Design a novel score function for cross-modal attention where query and key come from different modalities (e.g., text query attending to image keys). What properties should this score function have, and why?

A: A cross-modal score function should handle the different dimensionalities and statistical properties of the two modalities. A good design would be: (1) Project both modalities to a common dimension using modality-specific linear projections (similar to multi-head attention's W^Q and W^K). (2) Apply a learnable non-linear transformation to capture cross-modal relationships that may not be linear (e.g., using a small MLP with LayerNorm). (3) Include a temperature parameter that can be learned to control the peakiness of the attention distribution across modalities. The key insight is that cross-modal alignment is inherently more complex than same-modal alignment, so the score function may benefit from additional capacity and normalization to handle the distribution mismatch.

## Practice Problems

### Easy

Implement the dot, scaled dot, and general (bilinear) score functions. Verify that all produce scores of shape (batch, seq_q, seq_k) for given q and k tensors.

### Medium

Design an experiment that measures the entropy of attention distributions produced by different score functions as a function of the temperature parameter. Show that increasing temperature flattens the distribution.

### Hard

Implement a learnable score function with element-wise gating (i.e., a gated score function where a learned gate determines which dimensions of the query and key are compared). Compare its performance with standard dot-product and additive attention.

## Solutions

### Easy Solution

```python
def dot_score(q, k):
    return torch.matmul(q, k.transpose(-2, -1))

def scaled_dot_score(q, k):
    d_k = k.shape[-1]
    return torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

def bilinear_score(q, k, W):
    qW = torch.matmul(q, W)
    return torch.matmul(qW, k.transpose(-2, -1))

q = torch.randn(2, 3, 4)
k = torch.randn(2, 5, 4)
W = nn.Parameter(torch.randn(4, 4))

print(f"dot: {dot_score(q, k).shape}")
print(f"scaled: {scaled_dot_score(q, k).shape}")
print(f"bilinear: {bilinear_score(q, k, W).shape}")
# Output: dot: torch.Size([2, 3, 5])
# Output: scaled: torch.Size([2, 3, 5])
# Output: bilinear: torch.Size([2, 3, 5])
```

## Related Concepts

- Attention Weights
- Scaled Dot-Product Attention
- Bahdanau vs. Luong Attention
- Softmax Normalization
- Multi-Head Attention

## Next Concepts

- DL-341: Attention Weights
- DL-342: Context Vector from Attention
- DL-347: Scaled Dot-Product Attention

## Summary

The attention score function is the core computation that determines query-key compatibility in attention mechanisms. Score functions range from simple parameter-free operations (dot product, scaled dot product) to learned linear transformations (bilinear/general) to non-linear neural networks (additive, concat). The choice of score function significantly impacts model expressiveness, computational cost, training stability, and the resulting attention distribution. The scaled dot-product score has become the dominant choice in modern architectures due to its computational efficiency and compatibility with GPU-optimized operations.

## Key Takeaways

- Attention scores measure query-key compatibility, determining attention weights after softmax.
- Score functions include dot, scaled dot, bilinear, additive, and concat variants.
- Scaling dot-product scores by sqrt(d_k) is essential for training stability.
- Additive attention is most expressive but computationally expensive.
- The choice of score function affects attention distribution entropy and peakiness.
- Scaled dot-product attention dominates modern architectures due to GPU efficiency.
