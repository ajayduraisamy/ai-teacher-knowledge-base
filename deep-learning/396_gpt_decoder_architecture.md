# Concept: GPT Decoder Architecture

## Concept ID

DL-396

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Decoder Architectures

## Learning Objectives

- Understand the architectural components of the GPT decoder: causal self-attention, feed-forward networks, and layer normalization placement.
- Explain how the decoder differs from the Transformer encoder, particularly regarding attention masking.
- Implement the GPT decoder block from scratch in PyTorch.
- Analyze the computational flow during training (parallel) vs inference (sequential).
- Compare the GPT decoder architecture with BERT's encoder and the original Transformer decoder.

## Prerequisites

- Thorough understanding of the Transformer architecture
- Knowledge of self-attention, multi-head attention, and scaled dot-product attention
- Understanding of causal/autoregressive generation concepts
- Familiarity with layer normalization and residual connections
- Experience implementing Transformer components in PyTorch

## Definition

The GPT decoder is a Transformer-based autoregressive architecture that forms the backbone of the GPT family of models (GPT-1, GPT-2, GPT-3, GPT-4). Unlike the original Transformer decoder which includes cross-attention to encoder outputs, the GPT decoder is a decoder-only architecture that relies solely on causal self-attention. It consists of stacked decoder blocks, each containing a masked multi-head self-attention sublayer followed by a position-wise feed-forward network sublayer. Layer normalization is applied before each sublayer (pre-norm) in GPT-2 and later versions, unlike the post-norm used in the original Transformer. The causal masking ensures that each token can only attend to itself and preceding tokens, preserving the autoregressive property necessary for language generation.

## Intuition

Imagine reading a book one word at a time, covering the rest of the page with your hand. At each new word, you can only use the words you have already read to guess what comes next. This is exactly how the GPT decoder operates: when processing the 10th token, it has only seen tokens 1 through 9. It cannot peek ahead.

This might seem restrictive compared to BERT's bidirectional attention, but it is essential for generation. When a model generates text, it must produce one token at a time, and each new token can only depend on previously generated tokens. The causal mask enforces this constraint during both training and inference.

The GPT decoder has been the foundation for the most impactful language models ever built. GPT-3 demonstrated that scaling this architecture to 175 billion parameters produces remarkable few-shot learning abilities. The architecture's simplicity — just stacked decoder blocks with causal attention — has proven remarkably effective.

## Why This Concept Matters

The GPT decoder architecture is arguably the most influential architecture in modern AI, powering models from GPT-3 to ChatGPT to GPT-4 to LLaMA, Mistral, and beyond. Understanding it is essential because:

1. It is the architecture behind the most capable language models ever built.
2. Its simplicity (decoder-only, no cross-attention) has become the dominant paradigm.
3. Architectural improvements like pre-norm, rotary position encodings, and parallel attention originated from decoder-only models.
4. The causal masking mechanism is central to autoregressive generation.
5. The scale at which this architecture has been deployed (100B+ parameters) has driven innovations in distributed training and inference.

## Mathematical Explanation

### GPT Decoder Block

Each decoder block l receives input X_l and produces output X_{l+1}:

**Causal Self-Attention**:
Q = X_l W^Q, K = X_l W^K, V = X_l W^V

Scores = Q K^T / sqrt(d_k)

Causal mask M where M_{ij} = 0 if i >= j else -inf:

A = softmax(Scores + M) V

Output = Concat(head_1, ..., head_h) W^O

**Feed-Forward Network**:
FFN(x) = GELU(x W_1 + b_1) W_2 + b_2

**Pre-Norm** (GPT-2 and later):
x_attn = x + Attention(LayerNorm(x))
x_out = x_attn + FFN(LayerNorm(x_attn))

Note: GPT-1 used post-norm (LayerNorm after addition and sublayer), but GPT-2 and later use pre-norm which provides more stable training at scale.

### Training vs Inference

**Training**: All positions processed in parallel. The causal mask ensures position i only uses positions <= i. O(n^2) attention computation.

**Inference**: Tokens generated one at a time. Each new token attends to all previous tokens (which can be cached). O(n * d) per step with KV caching.

### KV Cache

During inference, previously computed K and V matrices are cached to avoid recomputation:

cache_l = [(K_1, V_1), (K_2, V_2), ..., (K_{t-1}, V_{t-1})]

At step t:
Q_t = X_t W^Q (query for new token only)
K_full = Concat(cache_k, K_t)
V_full = Concat(cache_v, V_t)

Score_t = Q_t K_full^T / sqrt(d_k)

## Code Examples

### Example 1: GPT Decoder Block Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GPTDecoderBlock(nn.Module):
    def __init__(self, d_model=768, n_heads=12, d_ff=3072, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, causal_mask=None):
        attn_out, _ = self.attention(
            self.norm1(x), self.norm1(x), self.norm1(x),
            attn_mask=causal_mask
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

def create_causal_mask(seq_len):
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask

block = GPTDecoderBlock()
x = torch.randn(2, 8, 768)
mask = create_causal_mask(8)
out = block(x, mask)
print("Output shape:", out.shape)
# Output: Output shape: torch.Size([2, 8, 768])
print("Causal mask ensures no peeking ahead")
# Output: Causal mask ensures no peeking ahead
```

### Example 2: Full GPT Decoder with KV Cache

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=12):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x, layer_past=None, use_cache=True):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        present = (k, v) if use_cache else None

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.full((T, k.size(-2)), float("-inf"), device=x.device), diagonal=1 + k.size(-2) - T)
        att = att + mask.unsqueeze(0).unsqueeze(0)
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y, present

attn = CausalSelfAttention()
x = torch.randn(1, 5, 768)
out, present = attn(x, use_cache=True)
print("KV cache shapes:", present[0].shape, present[1].shape)
# Output: KV cache shapes: torch.Size([1, 12, 5, 64]) torch.Size([1, 12, 5, 64])
```

### Example 3: Autoregressive Generation Loop

```python
class GPT(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, n_layers=12, n_heads=12, max_seq_len=1024):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            GPTDecoderBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids, past_kv=None, use_cache=True):
        B, T = input_ids.shape
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(pos_ids)
        presents = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            past = past_kv[i] if past_kv is not None else None
            causal_mask = create_causal_mask(T)
            x = block(x, causal_mask)
            if use_cache:
                pass

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

@torch.no_grad()
def generate(model, input_ids, max_new_tokens=20, temperature=0.8):
    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_logits = logits[:, -1, :] / temperature
        next_token = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    return input_ids

model = GPT()
start_ids = torch.randint(0, 100, (1, 5))
output = generate(model, start_ids, max_new_tokens=5)
print("Input length:", start_ids.shape[1])
# Output: Input length: 5
print("Output length:", output.shape[1])
# Output: Output length: 10
print("Autoregressive generation: one token at a time")
# Output: Autoregressive generation: one token at a time
```

## Common Mistakes

1. Confusing the GPT decoder with the original Transformer decoder: The original Transformer decoder includes cross-attention to encoder outputs. The GPT decoder removes cross-attention entirely, making it a decoder-only architecture that relies solely on self-attention.

2. Forgetting the causal mask during training: Without the causal mask, the model can see future tokens, making the language modeling task trivial (just copy the next token from the input). The mask is essential for the autoregressive property.

3. Not using KV caching during inference: Without KV cache, each generation step recomputes attention for all previous tokens, making inference O(n^3) instead of O(n^2). KV caching reduces this to O(n^2) total across all steps.

4. Using post-norm for large models: GPT-1 used post-norm, but GPT-2 and later shifted to pre-norm. Post-norm can cause training instability in deep models, especially at scale.

5. Mishandling position encodings during generation: Position IDs must continue incrementing during generation, not reset to 0 for each new token. The position embedding is based on the absolute position in the sequence.

6. Not tying embedding weights: GPT models typically tie the weights of the token embedding and the LM head (output projection). Forgetting this tie increases parameters without improving performance.

## Interview Questions

### Beginner

Q: What is the key architectural difference between the GPT decoder and the original Transformer decoder?

A: The GPT decoder removes cross-attention. The original Transformer decoder has two attention sublayers: causal self-attention and cross-attention (to encoder outputs). The GPT decoder has only causal self-attention, making it a decoder-only architecture without any encoder component.

### Intermediate

Q: Explain how KV caching works in GPT inference. Why is it important?

A: During autoregressive generation, each new token needs to attend to all previous tokens. Without caching, computing attention for step t requires recomputing K and V for all t-1 previous tokens, which is O(t * n) compute where n is the total context length. With KV caching, K and V from previous steps are stored and reused. At step t, only K_t and V_t (for the new token) need to be computed, and the full attention uses the concatenation [K_{1:t-1}, K_t]. This reduces per-step compute from O(t * d) to O(d).

### Advanced

Q: The GPT decoder uses pre-norm (LayerNorm before each sublayer) while the original Transformer uses post-norm. Explain the effect of this choice on training stability and representation quality.

A: Pre-norm places LayerNorm before the sublayer (attention or FFN), while post-norm places it after. Pre-norm has several advantages for deep models: (1) The residual path has no LayerNorm, allowing unnormalized gradients to flow through the skip connection, which improves gradient signal in early layers. (2) Pre-norm prevents activation growth — the LayerNorm constrains inputs to each sublayer, keeping activations bounded. (3) Empirically, pre-norm enables training deeper models (e.g., 100+ layers) without gradient explosion. However, post-norm can produce better representations because the final output of each block is normalized, which may benefit downstream task performance. Modern large models universally prefer pre-norm.

## Practice Problems

### Easy

Implement a causal attention mask function that takes a sequence length and returns a boolean mask where True indicates positions that should be masked (cannot be attended to). Verify that position i can attend to positions 0 through i but not i+1 through L-1.

### Medium

Train a small GPT model (6 layers, 384 hidden) on the Wikitext-2 dataset for language modeling. Implement KV caching during inference and measure the speedup compared to without caching. Plot the time per generated token as the sequence length grows.

### Hard

Implement a GPT decoder that uses rotary position encodings (RoPE) instead of learned position embeddings. Train on a language modeling task and compare perplexity and training speed with the standard learned position embeddings. Analyze the extrapolation capability to sequence lengths beyond training.

## Solutions

```python
# Easy solution
def create_causal_mask_bool(seq_len):
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    mask = torch.triu(mask, diagonal=1)
    return mask

mask = create_causal_mask_bool(5)
print("Causal mask (True = cannot attend):")
print(mask)
# Output: Causal mask (True = cannot attend):
# tensor([[False,  True,  True,  True,  True],
#         [False, False,  True,  True,  True],
#         [False, False, False,  True,  True],
#         [False, False, False, False,  True],
#         [False, False, False, False, False]])
print("Position 3 can attend to [0,1,2,3]:", mask[3].tolist())
# Output: Position 3 can attend to [0,1,2,3]: [False, False, False, False, True]
```

## Related Concepts

- Causal Masking (DL-398)
- Autoregressive Generation (DL-397)
- GPT-1 (DL-399)
- GPT-2 (DL-400)
- GPT-3 (DL-401)
- Decoder-Only Architecture (DL-403)
- Inference with Decoder (DL-404)

## Next Concepts

- Autoregressive Generation
- Causal Masking
- GPT-1
- GPT-2
- GPT-3

## Summary

The GPT decoder is a decoder-only Transformer architecture that uses causal self-attention to generate text autoregressively. It removes cross-attention from the original Transformer decoder, uses pre-norm layer normalization, and relies on stacked decoder blocks with causal masking. This architecture powers the GPT family and has become the dominant paradigm for large language models.

## Key Takeaways

- GPT decoder is decoder-only (no cross-attention from an encoder).
- Causal masking prevents attending to future tokens, preserving the autoregressive property.
- Pre-norm (LayerNorm before sublayers) improves training stability at scale.
- KV caching reduces inference computation by storing previous key-value pairs.
- The architecture supports both parallel training and sequential generation.
- Weight tying between embedding and LM head reduces parameters.
- Position encodings are learned and applied to the input embeddings.
- The simplicity and scalability of this architecture have made it the foundation of modern LLMs.
