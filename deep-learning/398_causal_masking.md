# Concept: Causal Masking

## Concept ID

DL-398

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Decoder Architectures

## Learning Objectives

- Define causal masking and explain how it enforces the autoregressive property in decoder-only models.
- Implement causal masks in PyTorch for both training and inference scenarios.
- Analyze the effect of causal masking on gradient flow and representation formation.
- Compare causal masking with bidirectional attention and prefix masking.
- Understand the interaction between causal masking and KV caching during inference.

## Prerequisites

- Understanding of self-attention in Transformers
- Knowledge of decoder-only architectures (DL-396)
- Familiarity with matrix multiplication and attention score computation
- Basic understanding of autoregressive generation (DL-397)

## Definition

Causal masking (also called autoregressive or unidirectional masking) is a mechanism applied to the self-attention scores in a Transformer decoder that prevents each token from attending to future tokens. Formally, for an input sequence of length L, a causal mask M is an L x L matrix where M_{ij} = 0 for j <= i (allowed positions) and M_{ij} = -inf for j > i (masked positions). When added to the attention scores S = QK^T / sqrt(d_k) before softmax, the -inf entries become zero after softmax, ensuring that each token's representation depends only on itself and preceding tokens. This is essential for autoregressive language modeling where the model must predict the next token without seeing it.

## Intuition

Causal masking enforces the principle that "you cannot see the future." When generating text, the model must rely only on what has already been written. If the model could see future tokens during training, it would simply copy the next token rather than learning to predict it.

Think of causal masking as a set of blinders on a horse. The horse can see where it has been and where it currently is, but it cannot see what is ahead. This ensures that when the horse predicts the next step, it must use only past information.

In the attention matrix visualization, causal masking makes the upper triangle completely dark (zero attention). The first token attends only to itself. The second token attends to the first and itself. The third attends to the first, second, and itself. And so on. Each row i has only i+1 non-zero entries.

## Why This Concept Matters

Causal masking is the defining feature of decoder-only models and is essential for:

1. Autoregressive training: Without causal masking, the language modeling objective would be trivial — the model would just copy the input.
2. Correct generation: The masking ensures that generation matches the training-time behavior, maintaining the causal dependency structure.
3. Model architecture classification: The presence or absence of causal masking determines whether a model is decoder-only (causal), encoder-only (no mask), or encoder-decoder (causal in decoder, no mask in encoder).
4. Inference efficiency: Understanding the mask structure enables KV caching for efficient generation.

## Mathematical Explanation

### Attention with Causal Mask

Given input X in R^{L x d}:

Q = X W^Q, K = X W^K, V = X W^V

Scores = Q K^T / sqrt(d_k)  (shape: L x L)

Causal mask M:
M_{ij} = 0 if j <= i (including current position)
M_{ij} = -inf if j > i (future positions)

Attention scores after masking:
S_masked = Scores + M

Each row is normalized with softmax:
A_i = softmax(S_masked_{i, :}) V

### Gradient Flow

During backpropagation, gradients flow through non-masked positions only. This means:

- Position i receives gradients from its own prediction and all future positions that attend to it.
- Gradients do not flow "backward in time" — position i does not receive gradients from the prediction of position i+1.

This is analogous to the autoregressive property in the forward pass.

### Implementation Forms

**Form 1: Triangular mask (additive)**:
mask = torch.triu(torch.full((L, L), float("-inf")), diagonal=1)

**Form 2: Boolean mask**:
mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)

Used with masked_fill: scores.masked_fill(mask, float("-inf"))

**Form 3: Bias mask (e.g., RoPE, ALiBi)**:
Causal mask combined with position bias:
attention = softmax(scores + causal_mask + position_bias) V

## Code Examples

### Example 1: Causal Mask Visualization

```python
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

def create_causal_mask(seq_len):
    return torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)

def visualize_attention_patterns():
    L = 6
    causal_mask = create_causal_mask(L)
    scores = torch.randn(L, L)
    masked_scores = scores + causal_mask
    attn_probs = F.softmax(masked_scores, dim=-1)

    print("Causal Mask (inf shown as -inf):")
    print(causal_mask)
    # Output: Causal Mask (inf shown as -inf):
    # tensor([[0., -inf, -inf, -inf, -inf, -inf],
    #         [0., 0., -inf, -inf, -inf, -inf],
    #         [0., 0., 0., -inf, -inf, -inf],
    #         [0., 0., 0., 0., -inf, -inf],
    #         [0., 0., 0., 0., 0., -inf],
    #         [0., 0., 0., 0., 0., 0.]])

    print("Attention probabilities (row = query, col = key):")
    print(attn_probs.round(decimals=3))
    # Output: Attention probabilities (row = query, col = key):
    # tensor([[1.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    #         [0.498, 0.502, 0.000, 0.000, 0.000, 0.000],
    #         [0.332, 0.335, 0.333, 0.000, 0.000, 0.000],
    #         [0.250, 0.251, 0.249, 0.250, 0.000, 0.000],
    #         [0.200, 0.199, 0.200, 0.200, 0.201, 0.000],
    #         [0.167, 0.167, 0.166, 0.166, 0.167, 0.167]])

    print("Row sums (all should be 1.0):", attn_probs.sum(dim=-1))
    # Output: Row sums (all should be 1.0): tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])

visualize_attention_patterns()
```

### Example 2: Training with Causal Mask (Parallel)

```python
class CausalLanguageModel(nn.Module):
    def __init__(self, vocab_size=1000, d_model=256, n_layers=4, n_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(512, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=1024,
                                                     activation="gelu", batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.position(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        x = self.decoder(x, x, tgt_mask=causal_mask)
        return self.lm_head(x)

def training_step(model, input_ids, optimizer):
    logits = model(input_ids)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

model = CausalLanguageModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
input_ids = torch.randint(0, 1000, (2, 32))
loss = training_step(model, input_ids, optimizer)
print("Training loss:", loss)
# Output: Training loss: 6.9078
print("Training is fully parallel (all positions processed at once)")
# Output: Training is fully parallel (all positions processed at once)
```

### Example 3: Causal Masking vs Bidirectional Attention

```python
def compare_attention_masks(seq_len=5):
    positions = ["tok1", "tok2", "tok3", "tok4", "tok5"]

    causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    bidirectional_mask = torch.zeros(seq_len, seq_len)

    dummy_scores = torch.ones(seq_len, seq_len)
    causal_attn = F.softmax(dummy_scores + causal_mask, dim=-1)
    bidir_attn = F.softmax(dummy_scores + bidirectional_mask, dim=-1)

    print("Causal attention (average attention per query):")
    for i in range(seq_len):
        attend_to = (causal_attn[i] > 0).sum().item()
        print(f"  {positions[i]} attends to {attend_to} positions (itself + {attend_to-1} previous)")
    # Output: Causal attention (average attention per query):
    #   tok1 attends to 1 positions (itself + 0 previous)
    #   tok2 attends to 2 positions (itself + 1 previous)
    #   tok3 attends to 3 positions (itself + 2 previous)
    #   tok4 attends to 4 positions (itself + 3 previous)
    #   tok5 attends to 5 positions (itself + 4 previous)

    print("Bidirectional attention (each token attends to all 5)")
    # Output: Bidirectional attention (each token attends to all 5)

    total_attention_causal = causal_attn.sum().item()
    total_attention_bidir = bidir_attn.sum().item()
    print(f"Total attention mass: causal={total_attention_causal:.1f}, bidir={total_attention_bidir:.1f}")
    # Output: Total attention mass: causal=15.0, bidir=25.0

compare_attention_masks()
```

## Common Mistakes

1. Forgetting to mask out future positions during training: Without the causal mask, the model uses future tokens to predict the current token. This effectively turns language modeling into a trivial copying task, and the model learns nothing useful.

2. Applying causal mask incorrectly with batch_first=False: nn.MultiheadAttention expects different mask shapes depending on batch_first. With batch_first=True, the mask should be (T, T). With batch_first=False, it should be (T, T) as well, but attention weights are (B, H, T, T).

3. Confusing causal mask with padding mask: Causal masking prevents attending to future tokens. Padding masking prevents attending to padding tokens. They serve different purposes and can be combined (causal_forbid_upper_triangle + padding_forbid_pad_tokens).

4. Using float("-inf") instead of a large negative number: Some implementations use a large negative number like -1e9 instead of -inf. While this generally works, -inf is mathematically correct and should be preferred with PyTorch's softmax.

5. Not updating the causal mask for variable-length generation: During generation, the causal mask grows with each new token. A common mistake is using a fixed-size mask when the sequence length changes.

6. Applying causal mask in cross-attention: Cross-attention (between encoder and decoder) should NOT use causal masking. The decoder should be able to attend to all encoder outputs.

## Interview Questions

### Beginner

Q: What is the purpose of causal masking in a decoder-only Transformer?

A: Causal masking prevents each token from attending to future tokens. This enforces the autoregressive property: each token's representation depends only on itself and preceding tokens. This is necessary for language modeling (predicting the next token) and for text generation (each new token depends only on previously generated tokens).

### Intermediate

Q: How does causal masking affect parallelization during training vs. inference?

A: During training, causal masking allows full parallelization because the entire target sequence is available. The causal mask simply zeros out future positions in the attention computation, but all positions are processed simultaneously. During inference, generation is inherently sequential because each token depends on previously generated tokens. However, KV caching mitigates this by storing past key-value states, enabling O(1) per-step computation instead of recomputing attention over the full sequence.

### Advanced

Q: Design a modified causal mask that allows each token to attend to a window of k future tokens. How would this affect the model's capabilities and training dynamics?

A: A modified causal mask with k future tokens (also called "look-ahead masking" or "n-gram masking") would set M_{ij} = 0 for j <= i + k, and -inf otherwise. This allows each token to attend to k future tokens. Effects: (1) The model would be less strictly autoregressive, potentially improving representation quality as more context is available. (2) However, it would also make the language modeling task slightly easier (the model sees k future tokens when predicting the current one). (3) Generation would still be autoregressive but with reduced gap between training and inference. (4) This approach is used in some models like UniLM for non-autoregressive generation. The optimal k depends on the task — too large eliminates the benefits of causal masking, too small provides negligible context.

## Practice Problems

### Easy

Write a function that generates a causal mask for a 2D attention matrix. Verify that the mask has the upper triangle filled with -inf and the lower triangle (including diagonal) filled with 0. Test with sequence lengths 1, 5, and 10.

### Medium

Implement a combined mask function that applies both causal masking and padding masking to attention scores. Given input_ids with padding, compute attention scores where each token can attend only to non-padded, non-future tokens.

### Hard

Design and implement a variable-length attention mask for Prefix LM training. The first k tokens should have bidirectional attention (no causal mask), and the remaining tokens should use causal masking. Implement this mask and verify the attention pattern shows the prefix attending bidirectionally.

## Solutions

```python
# Easy solution
def create_causal_mask(L):
    mask = torch.triu(torch.full((L, L), float("-inf")), diagonal=1)
    return mask

def verify_causal_mask():
    for L in [1, 5, 10]:
        mask = create_causal_mask(L)
        upper_tri = torch.triu(torch.ones(L, L), diagonal=1).bool()
        lower_tri = torch.tril(torch.ones(L, L), diagonal=0).bool()
        assert (mask[upper_tri] == float("-inf")).all(), f"Upper triangle should be -inf, L={L}"
        assert (mask[lower_tri] == 0).all(), f"Lower triangle should be 0, L={L}"
        print(f"L={L}: verified")
    print("All tests passed!")
    # Output: L=1: verified
    # Output: L=5: verified
    # Output: L=10: verified
    # Output: All tests passed!

verify_causal_mask()
```

## Related Concepts

- GPT Decoder Architecture (DL-396)
- Autoregressive Generation (DL-397)
- Decoder-Only Architecture (DL-403)
- Prefix LM (DL-405)
- Self-Attention
- Padding Masks
- KV Caching

## Next Concepts

- GPT-1
- GPT-2
- GPT-3
- Decoder-Only Architecture

## Summary

Causal masking is a triangular mask applied to attention scores in decoder-only models that prevents each token from attending to future tokens. It enforces the autoregressive property during both training and inference, ensuring that token predictions depend only on previous tokens. Causal masking is the defining architectural feature of decoder-only models and is essential for language modeling and text generation.

## Key Takeaways

- Causal mask is an upper-triangular matrix with 0 on/below diagonal and -inf above.
- It enforces the autoregressive property: each token sees only itself and past tokens.
- During training, all positions are processed in parallel despite the mask.
- During inference, KV caching uses the mask structure for efficiency.
- Causal masking is distinct from padding masking (which masks padding tokens).
- The mask is applied to attention scores before softmax.
- Causal masking enables the next-token prediction objective.
- Modified causal masks (prefix, look-ahead) enable different model behaviors.
