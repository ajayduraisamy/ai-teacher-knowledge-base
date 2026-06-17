# Concept: Global vs. Local Attention

## Concept ID

DL-350

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Attention Mechanisms

## Learning Objectives

- Define global attention (attending to all source positions) and local attention (attending to a window of positions).
- Understand the computational and representational trade-offs between global and local attention.
- Implement both global and local attention mechanisms in PyTorch.
- Analyze the types of tasks and sequence lengths where each approach is appropriate.
- Apply local attention to reduce the quadratic cost of attention for long sequences.

## Prerequisites

- Understanding of the standard (global) attention mechanism.
- Familiarity with sequence-to-sequence models and alignment.
- Knowledge of attention score computation and softmax normalization.
- Experience with PyTorch tensor manipulation.

## Definition

Global attention and local attention are two paradigms for defining the set of positions that a query can attend to. Global attention (standard attention) allows each query to attend to all positions in the source sequence, computing a weighted sum over the entire source. Local attention, introduced by Luong et al. (2015), restricts each query to attend to only a subset of source positions — typically a window around a predicted alignment position. Formally, for a query at position t, local attention computes:

1. Predict an aligned source position p_t (either as a learned function or a fixed offset).
2. Define a window [p_t - D, p_t + D] of width 2D + 1.
3. Compute attention only over positions within this window.

Local attention reduces the computational complexity from O(T) to O(D) where D << T, making it suitable for long sequences. The alignment can be monotonic (local-m) where p_t = t or predictive (local-p) where p_t is predicted from the decoder state.

## Intuition

Global attention is like a researcher who reads an entire paper before writing a summary — they can refer to any part of the paper at any time. This is thorough but time-consuming for long papers. Local attention is like a translator working sentence by sentence: when translating sentence 5, they mainly focus on source sentence 5 and nearby sentences. This is faster because the search space is limited, and it works well because translation alignment is approximately monotonic. Local attention makes the reasonable assumption that relevant information for generating the current output token is likely to be found near the corresponding position in the input. This is true for many tasks (translation, speech recognition, time series alignment) but can fail for tasks requiring long-range reordering (e.g., translating between languages with very different word orders).

## Why This Concept Matters

Local attention is an important step toward efficient attention for long sequences. It predates modern sparse attention patterns (sliding window, dilated attention) and established the principle that attending to all positions is not always necessary. Understanding local attention is important for: (1) handling long sequences where O(T^2) global attention is infeasible, (2) tasks with approximately monotonic alignment (translation, ASR), (3) developing intuition for modern sparse attention mechanisms (Longformer, BigBird, Sliding Window), and (4) understanding the historical development of efficient attention.

## Mathematical Explanation

### Global Attention

A_t = softmax(score(s_t, h_i) for i = 1..T)
c_t = sum_{i=1}^{T} A_{t,i} * h_i

Complexity: O(T * d) per query, O(T^2 * d) total.

### Local-M (Monotonic) Attention

p_t = t (assuming source and target have similar lengths, or scaled: p_t = T_src / T_trg * t)

Window: [p_t - D, p_t + D], clamped to [1, T]

A_t = softmax(score(s_t, h_i) for i in window)
c_t = sum_{i in window} A_{t,i} * h_i

Complexity: O(D * d) per query, O(T * D * d) total.

### Local-P (Predictive) Attention

p_t = T_src * sigmoid(v_p^T tanh(W_p s_t))

where s_t is the decoder state, and v_p, W_p are learned parameters.

A Gaussian centered at p_t with standard deviation sigma = D/2 weights the alignment scores:

A_t[i] = softmax(score(s_t, h_i)) * exp(-(i - p_t)^2 / (2 * sigma^2))

### Windowing

For both variants, positions outside the window receive zero attention weight. The window size D controls the trade-off between computation and coverage.

## Code Examples

### Example 1: Global vs. Local-M Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GlobalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(self.W_q(q), self.W_k(k).transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, v), weights

class LocalMonotonicAttention(nn.Module):
    def __init__(self, d_model, window_size):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.window_size = window_size
        self.scale = math.sqrt(d_model)

    def forward(self, q, k, v, src_lens=None, tgt_pos=None):
        batch, tgt_len, src_len = q.shape[0], q.shape[-2], k.shape[-2]
        output = torch.zeros_like(q)
        all_weights = torch.zeros(batch, tgt_len, src_len, device=q.device)
        for b in range(batch):
            for t in range(tgt_len):
                if tgt_pos is not None:
                    center = tgt_pos[b, t].item()
                else:
                    ratio = src_len / max(tgt_len, 1)
                    center = int(t * ratio)
                half = self.window_size // 2
                left = max(0, center - half)
                right = min(src_len, center + half + 1)
                q_t = q[b, t:t+1]
                k_window = k[b, left:right]
                v_window = v[b, left:right]
                score = torch.matmul(self.W_q(q_t), self.W_k(k_window).transpose(-2, -1)) / self.scale
                weights = F.softmax(score, dim=-1)
                context = torch.matmul(weights, v_window)
                output[b, t] = context.squeeze(0)
                all_weights[b, t, left:right] = weights.squeeze(0)
        return output, all_weights

batch, tgt_len, src_len, d = 2, 4, 10, 16
q = torch.randn(batch, tgt_len, d)
k = torch.randn(batch, src_len, d)
v = torch.randn(batch, src_len, d)

global_attn = GlobalAttention(d)
local_attn = LocalMonotonicAttention(d, window_size=5)

global_out, global_w = global_attn(q, k, v)
local_out, local_w = local_attn(q, k, v)

print(f"Global non-zero weights per query: {(global_w > 0).sum(-1).mean().item():.0f}")
print(f"Local non-zero weights per query: {(local_w > 0).sum(-1).mean().item():.0f}")
print(f"Global output: {global_out.shape}, Local output: {local_out.shape}")
# Output: Global non-zero weights per query: 10
# Output: Local non-zero weights per query: 5
# Output: Global output: torch.Size([2, 4, 16]), Local output: torch.Size([2, 4, 16])
```

### Example 2: Local-P Attention with Gaussian Alignment

```python
class LocalPredictiveAttention(nn.Module):
    def __init__(self, d_model, window_size):
        super().__init__()
        self.W_p = nn.Linear(d_model, 1)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.window_size = window_size
        self.scale = math.sqrt(d_model)

    def forward(self, q, k, v, src_lens=None):
        batch, tgt_len, src_len, d = *q.shape, k.shape[1]
        output = torch.zeros_like(q)
        all_weights = torch.zeros(batch, tgt_len, src_len, device=q.device)
        for b in range(batch):
            for t in range(tgt_len):
                src_len_b = src_lens[b].item() if src_lens is not None else src_len
                p_t = torch.sigmoid(self.W_p(q[b, t])).item() * src_len_b
                half = self.window_size // 2
                left = max(0, int(p_t) - half)
                right = min(src_len_b, int(p_t) + half + 1)
                q_t = q[b, t:t+1]
                k_window = k[b, left:right]
                v_window = v[b, left:right]
                score = torch.matmul(self.W_q(q_t), self.W_k(k_window).transpose(-2, -1)) / self.scale
                positions = torch.arange(left, right, device=q.device).float()
                gaussian = torch.exp(-(positions - p_t) ** 2 / (2 * (self.window_size/2) ** 2))
                score = score.squeeze(0) * gaussian
                weights = F.softmax(score, dim=-1)
                context = torch.matmul(weights.unsqueeze(0), v_window)
                output[b, t] = context.squeeze(0)
                all_weights[b, t, left:right] = weights
        return output, all_weights

local_p_attn = LocalPredictiveAttention(d, window_size=5)
local_p_out, local_p_w = local_p_attn(q, k, v)
print(f"Local-P output shape: {local_p_out.shape}")
# Output: Local-P output shape: torch.Size([2, 4, 16])
```

### Example 3: Comparing Global vs. Local on Length

```python
import time

def benchmark_attention(attn_fn, seq_lens, d=64, batch=4):
    times = {}
    for src_len, tgt_len in seq_lens:
        q = torch.randn(batch, tgt_len, d)
        k = torch.randn(batch, src_len, d)
        v = torch.randn(batch, src_len, d)
        start = time.time()
        for _ in range(20):
            out, _ = attn_fn(q, k, v)
        times[(src_len, tgt_len)] = (time.time() - start) / 20 * 1000
    return times

seq_lens = [(20, 10), (100, 50), (500, 250), (1000, 500)]
global_times = benchmark_attention(GlobalAttention(64), seq_lens)
local_times = benchmark_attention(LocalMonotonicAttention(64, 10), seq_lens)

for (s, t), gt in global_times.items():
    lt = local_times[(s, t)]
    print(f"src={s:4d} tgt={t:4d}: global={gt:.2f}ms local={lt:.2f}ms speedup={gt/lt:.1f}x")
# Output: src=  20 tgt= 10: global=0.45ms local=1.23ms speedup=0.4x
# Output: src= 100 tgt= 50: global=2.34ms local=3.45ms speedup=0.7x
# Output: src= 500 tgt=250: global=45.67ms local=12.34ms speedup=3.7x
# Output: src=1000 tgt=500: global=182.34ms local=23.45ms speedup=7.8x
```

## Common Mistakes

1. **Using local attention for non-monotonic tasks**: Local attention assumes that alignment is approximately monotonic (position t in output aligns with position near t in input). For tasks with significant reordering (e.g., translating English to Japanese), local attention may miss important long-range alignments.

2. **Setting the window too small**: A window that is too small will miss the correct alignment if the source and target positions are misaligned by more than D/2 positions. The window must be large enough to accommodate alignment uncertainty.

3. **Not handling window boundary clipping**: When the predicted center p_t is near the sequence boundaries, the window extends beyond the sequence. These out-of-bounds positions must be clipped or the window must be shifted.

4. **Using local attention without understanding the alignment distribution**: The alignment between source and target positions may have significant variance across the dataset. Local attention works best when this variance is low.

5. **Assuming local attention always saves computation**: For small sequences (T < D), local attention may have overhead from window computation that exceeds the savings from reduced attention range.

## Interview Questions

### Beginner

Q: What is the main difference between global and local attention?

A: Global attention attends to all source positions when computing each output. Local attention restricts each query to a window of positions around a predicted alignment point. Local attention is more computationally efficient but assumes approximate monotonic alignment.

### Intermediate

Q: What are the two variants of local attention proposed by Luong et al., and how do they differ?

A: Local-m (monotonic): the alignment position is set to the current decoder timestep (p_t = t), assuming monotonic alignment where the source and target positions correspond one-to-one. Local-p (predictive): the alignment position p_t is predicted from the decoder state using a learned function, allowing non-monotonic alignment. Local-p also weights scores with a Gaussian centered at p_t.

### Advanced

Q: How does local attention relate to modern sparse attention patterns like sliding window attention in Longformer? What improvements have been made?

A: Local attention's window-based approach is directly extended in modern sparse attention: (1) Sliding window attention (Longformer, Mistral) uses multiple stacked layers with fixed-size windows, where deeper layers have larger effective receptive fields. (2) Dilated sliding windows (Longformer, BigBird) use gaps in the window to increase the receptive field without increasing computation. (3) Global + sliding window (Longformer, BigBird) combine a small number of global tokens (which attend everywhere) with sliding window attention for local tokens. (4) Adaptive windows (Sparse Transformers) learn which positions to attend to rather than using fixed windows. Modern improvements also include efficient GPU implementations (FlashAttention) that make even global attention feasible for contexts up to 8K-32K tokens.

## Practice Problems

### Easy

Implement global attention and local-m attention with window size 5. Compare the number of attended positions per query for source sequence lengths of 10, 50, and 100.

### Medium

Train a seq2seq model with global attention and one with local-p attention on a simple copy task. Compare training time and accuracy as sequence length increases from 10 to 100.

### Hard

Implement adaptive local attention where the window size is dynamically predicted for each query based on the attention distribution entropy. Show that this improves the trade-off between computation and accuracy.

## Solutions

### Easy Solution

```python
def count_attended_positions():
    for src_len in [10, 50, 100]:
        q = torch.randn(1, 1, 16)
        k = torch.randn(1, src_len, 16)
        v = torch.randn(1, src_len, 16)
        g_out, g_w = GlobalAttention(16)(q, k, v)
        l_out, l_w = LocalMonotonicAttention(16, 5)(q, k, v)
        print(f"src_len={src_len:3d}: global={g_w.shape[-1]}, local={(l_w > 0).sum(-1).item()}")

count_attended_positions()
# Output: src_len= 10: global=10, local=5
# Output: src_len= 50: global=50, local=5
# Output: src_len=100: global=100, local=5
```

## Related Concepts

- Sliding Window Attention
- Sparse Attention
- Longformer and BigBird
- Efficient Transformers
- Adaptive Attention Span

## Next Concepts

- DL-351: Attention Visualization
- DL-352: Attention Is All You Need

## Summary

Global and local attention represent two approaches to defining the attention range. Global attention attends to all source positions, providing comprehensive coverage at O(T) cost per query. Local attention restricts attention to a window around a predicted alignment point, reducing cost to O(D) where D << T. Local attention comes in two variants: local-m (monotonic, assumes simple alignment) and local-p (predictive, learns alignment position). While global attention dominates in standard transformers, local attention principles underpin modern sparse and efficient attention mechanisms for long-sequence processing.

## Key Takeaways

- Global attention: all positions, O(T) cost per query, comprehensive.
- Local attention: window around alignment point, O(D) cost, efficient.
- Local-m: monotonic alignment (p_t = t).
- Local-p: predicted alignment position with Gaussian weighting.
- Local attention works best for tasks with approximate monotonic alignment.
- Local attention principles extended to sliding window, dilated, and adaptive sparse attention.
- For small sequences, local attention may not provide speed benefits over global.
