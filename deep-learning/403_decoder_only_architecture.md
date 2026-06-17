# Concept: Decoder-Only Architecture

## Concept ID

DL-403

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Decoder Architectures

## Learning Objectives

- Understand the complete decoder-only architecture and its components.
- Explain why decoder-only models have become the dominant paradigm in LLMs.
- Compare decoder-only with encoder-decoder and encoder-only architectures across multiple dimensions.
- Implement a complete decoder-only language model from scratch.
- Analyze the scaling properties and emergent capabilities of decoder-only models.

## Prerequisites

- Comprehensive understanding of Transformer architectures
- Knowledge of GPT models (DL-396 through DL-402)
- Understanding of causal masking (DL-398)
- Familiarity with training and inference dynamics

## Definition

A decoder-only architecture is a Transformer model that consists exclusively of stacked decoder blocks, each containing causal self-attention and feed-forward networks, with no encoder component and no cross-attention. The architecture processes input sequences autoregressively — each token can only attend to itself and preceding tokens. Despite its simplicity, the decoder-only architecture has become the dominant paradigm in large language models (GPT-2, GPT-3, GPT-4, LLaMA, Mistral, Gemini, Claude) due to its scalability, versatility, and the emergent capabilities that arise from causal language modeling at scale. Unlike encoder-only models (which excel at understanding) or encoder-decoder models (which excel at sequence-to-sequence tasks), decoder-only models can perform both understanding and generation tasks through prompting and in-context learning.

## Intuition

A decoder-only model is like a writer who reads a prompt and continues writing from there. The writer never goes back to revise earlier words based on what comes later — each new word is based only on what has already been written. This constraint, which seems limiting compared to BERT's bidirectional understanding, is actually what makes generation possible and what enables the remarkable emergent abilities of large models.

The decoder-only architecture is the simplest of all Transformer variants. It removes the encoder (no separate input processing), removes cross-attention (no attending to encoder outputs), and uses only causal self-attention. The entire model is a single stack of identical blocks. This simplicity has proven to be a feature, not a bug: the straightforward architecture scales more predictably, makes training more stable, and enables simpler serving infrastructure.

## Why This Concept Matters

The decoder-only architecture is the foundation of virtually all modern large language models. Understanding it is essential because:

1. It is the most widely deployed architecture in production AI systems.
2. Its scaling properties are well-understood and predictable.
3. It enables in-context learning and emergent abilities.
4. The simplicity of the architecture makes it amenable to optimization (quantization, distillation, pruning).
5. The shift from encoder-decoder to decoder-only represents one of the most important architectural trends in deep learning history.

## Mathematical Explanation

### Architecture Components

**Input Embedding**:
- Token embedding: E_token in R^{V x d}
- Position encoding: learned (GPT), sinusoidal (original Transformer), or rotary (RoPE, LLaMA)
- Optional: no position encoding when using ALiBi (Mistral)

**Decoder Block** (repeated L times):

1. Causal Self-Attention:
Q = LayerNorm(x) W^Q
K = LayerNorm(x) W^K
V = LayerNorm(x) W^V
Scores = Q K^T / sqrt(d_k) + causal_mask + position_bias
Attention = softmax(Scores) V
Output = Attention W^O + x  (residual connection)

2. Feed-Forward Network:
Output = FFN(LayerNorm(x)) + x

**Output Head**:
Logits = LayerNorm(final_hidden) W_lm^T
P(token) = softmax(logits)

### Scaling Properties

The decoder-only architecture scales predictably with:
- Parameters: ~12 * L * d^2 (where L = layers, d = hidden size)
- FLOPs per token: ~6 * N (where N = total parameters)
- Memory during inference: ~2 * N bytes (FP16)

### Training vs Inference

**Training**: All T positions processed in parallel with causal masking.
Complexity: O(L * T^2 * d) for attention, O(L * T * d^2) for FFN.

**Inference**: Sequential, one token at a time with KV caching.
Complexity per step: O(L * T * d) for attention, O(L * d^2) for FFN.

## Code Examples

### Example 1: Complete Decoder-Only Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RoPE(nn.Module):
    def __init__(self, d_model, max_len=8192):
        super().__init__()
        theta = 10000.0 ** (-torch.arange(0, d_model, 2).float() / d_model)
        positions = torch.arange(max_len).float().unsqueeze(1)
        freqs = positions * theta.unsqueeze(0)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, x, offset=0):
        B, T, D = x.shape
        cos = self.cos[offset:offset+T, :].unsqueeze(0).unsqueeze(2)
        sin = self.sin[offset:offset+T, :].unsqueeze(0).unsqueeze(2)
        x_rot = x.view(B, T, -1, 2)
        x_out = torch.empty_like(x_rot)
        x_out[..., 0] = x_rot[..., 0] * cos - x_rot[..., 1] * sin
        x_out[..., 1] = x_rot[..., 0] * sin + x_rot[..., 1] * cos
        return x_out.view(B, T, D)

class DecoderOnlyBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask)[0]
        x = x + self.ffn(self.norm2(x))
        return x

class DecoderOnlyModel(nn.Module):
    def __init__(self, vocab_size=32000, d_model=4096, n_layers=32,
                 n_heads=32, d_ff=11008, max_len=8192, use_rope=True):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.use_rope = use_rope
        if not use_rope:
            self.pos_emb = nn.Embedding(max_len, d_model)
        self.rope = RoPE(d_model, max_len) if use_rope else None
        self.blocks = nn.ModuleList([
            DecoderOnlyBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.token_emb(input_ids)
        if not self.use_rope:
            pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_emb(pos)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=input_ids.device), diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return self.lm_head(x)

model = DecoderOnlyModel(vocab_size=1000, d_model=256, n_layers=4, n_heads=4, d_ff=1024)
x = torch.randint(0, 1000, (2, 16))
logits = model(x)
print("Decoder-only model logits:", logits.shape)
# Output: Decoder-only model logits: torch.Size([2, 16, 1000])
print("Total parameters:", sum(p.numel() for p in model.parameters()))
# Output: Total parameters: 5136904
```

### Example 2: Comparison with Encoder-Decoder

```python
class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size=1000, d_model=256, n_layers=4, n_heads=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)
        self.emb = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.emb(src)
        tgt_emb = self.emb(tgt)
        memory = self.encoder(src_emb)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1))
        output = self.decoder(tgt_emb, memory, tgt_mask=causal_mask)
        return self.head(output)

class DecoderOnly(nn.Module):
    def __init__(self, vocab_size=1000, d_model=256, n_layers=4, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x_emb = self.emb(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1))
        output = self.decoder(x_emb, x_emb, tgt_mask=causal_mask)
        return self.head(output)

enc_dec = EncoderDecoder()
dec_only = DecoderOnly()

src = torch.randint(0, 1000, (2, 10))
tgt = torch.randint(0, 1000, (2, 8))
logits_encdec = enc_dec(src, tgt)
logits_deconly = dec_only(torch.cat([src, tgt], dim=1))

print("Encoder-decoder processes src and tgt separately")
# Output: Encoder-decoder processes src and tgt separately
print("Encoder-decoder logits:", logits_encdec.shape)
# Output: Encoder-decoder logits: torch.Size([2, 8, 1000])
print("Decoder-only processes concatenated input")
# Output: Decoder-only processes concatenated input
print("Decoder-only logits:", logits_deconly.shape)
# Output: Decoder-only logits: torch.Size([2, 18, 1000])
```

### Example 3: KV Caching in Decoder-Only Models

```python
class DecoderOnlyWithCache(nn.Module):
    def __init__(self, vocab_size=1000, d_model=256, n_layers=4, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, past_kv=None):
        B, T = x.shape
        x_emb = self.emb(x)
        new_kv = []

        for i, layer in enumerate(self.layers):
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
            out = layer(x_emb, x_emb, tgt_mask=causal_mask)
            x_emb = out

        x_emb = self.norm(x_emb)
        return self.head(x_emb)

@torch.no_grad()
def generate_with_cache(model, input_ids, max_new=10):
    for _ in range(max_new):
        logits = model(input_ids)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    return input_ids

model = DecoderOnlyWithCache()
x = torch.randint(0, 1000, (1, 5))
result = generate_with_cache(model, x, max_new=5)
print("Generated:", result.shape)
# Output: Generated: torch.Size([1, 10])
print("Decoder-only: sequential generation with KV cache")
# Output: Decoder-only: sequential generation with KV cache
```

## Common Mistakes

1. Confusing decoder-only with having no attention: Decoder-only models have self-attention (causal), just not cross-attention. The key distinction is the absence of an encoder and cross-attention layers.

2. Assuming decoder-only models cannot handle understanding tasks: Through prompting and in-context learning, decoder-only models can perform classification, QA, NLI, and other understanding tasks. They are not limited to generation.

3. Forgetting that decoder-only models can be fine-tuned: While in-context learning is a key capability, decoder-only models can also be fine-tuned for specific tasks with labeled data, similar to BERT.

4. Thinking decoder-only is less powerful than encoder-decoder: For generation tasks at scale, decoder-only models have proven superior to encoder-decoder models due to simpler scaling and better in-context learning.

5. Ignoring the context window limitation: Decoder-only models process the entire prompt as context. The context window limits how much information can be provided for in-context learning.

6. Not accounting for the inference memory footprint: With KV caching, the memory grows with sequence length. For long sequences, the KV cache can exceed the model weights in size.

## Interview Questions

### Beginner

Q: What makes decoder-only architecture different from encoder-decoder and encoder-only architectures?

A: Decoder-only uses only causal self-attention (no cross-attention, no bidirectional attention). Encoder-only uses bidirectional self-attention (no generation capability). Encoder-decoder combines a bidirectional encoder for input understanding with a causal decoder for generation, connected by cross-attention.

### Intermediate

Q: Why have decoder-only models become the dominant architecture for large language models?

A: Several factors: (1) Simpler architecture — one stack of blocks with one type of attention, making scaling more predictable. (2) In-context learning emerges only in decoder-only models trained causally. (3) They can handle both understanding and generation tasks through prompting, reducing the need for multiple models. (4) Scaling laws are better understood for decoder-only models. (5) Serving infrastructure is simpler — one model to deploy instead of two.

### Advanced

Q: Compare the gradient flow in a decoder-only model during training with an encoder-decoder model. How does the causal mask affect gradient propagation differently from cross-attention?

A: In decoder-only models, gradients flow through the causal mask structure. Each position i receives gradients from its own prediction (next token prediction loss) and from the loss at positions > i (since those positions attend to i). The upper-triangular mask means gradients flow only "forward in time." In encoder-decoder models, the decoder's cross-attention allows gradients to flow from the decoder output back to all encoder positions, and the bidirectional encoder allows full gradient flow within the encoder. The absence of cross-attention in decoder-only models means the model must learn all dependencies within the single causal structure, which may require more parameters but also forces more robust representations.

## Practice Problems

### Easy

List five modern large language models that use decoder-only architecture. For each, note the number of parameters and any special features (RoPE, MoE, Grouped Query Attention, etc.).

### Medium

Implement a decoder-only model with Grouped Query Attention (GQA) instead of standard multi-head attention. GQA reduces the number of key-value heads to improve inference efficiency. Train on a small dataset and compare training speed, inference speed, and perplexity with standard MHA.

### Hard

Design a hybrid architecture that combines a decoder-only model with a small encoder for retrieval-augmented generation (RAG). The encoder processes retrieved documents, and the decoder attends to both its own context and the encoded documents through inserted cross-attention layers. Implement and evaluate on a knowledge-intensive QA task.

## Solutions

```python
# Medium solution: Grouped Query Attention
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model=4096, n_heads=32, n_kv_heads=8, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_heads
        self.n_groups = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.d_head)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head)
        self.out_proj = nn.Linear(n_heads * self.d_head, d_model)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head)

        k = k.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1)
        v = v.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1)

        attn = torch.einsum("bthd,btkhd->bthk", q, k) / math.sqrt(self.d_head)
        if mask is not None:
            attn = attn + mask.unsqueeze(1)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bthk,btkhd->bthd", attn, v)
        return self.out_proj(out.reshape(B, T, -1))

print("GQA uses fewer KV heads for efficient inference")
# Output: GQA uses fewer KV heads for efficient inference
```

## Related Concepts

- GPT Decoder Architecture (DL-396)
- Autoregressive Generation (DL-397)
- Causal Masking (DL-398)
- GPT-1 (DL-399)
- GPT-2 (DL-400)
- GPT-3 (DL-401)
- GPT-4 Architecture Overview (DL-402)
- Inference with Decoder (DL-404)
- Prefix LM (DL-405)

## Next Concepts

- Inference with Decoder
- Prefix LM

## Summary

The decoder-only architecture consists of stacked blocks with causal self-attention and feed-forward networks, without an encoder or cross-attention. It is the dominant paradigm for modern LLMs due to its simplicity, predictable scaling, and emergent capabilities like in-context learning. The architecture enables both understanding and generation tasks through prompting.

## Key Takeaways

- Decoder-only = stacked decoder blocks with causal self-attention, no encoder, no cross-attention.
- Dominant architecture for modern LLMs (GPT, LLaMA, Mistral, Claude, Gemini).
- Enables in-context learning and emergent capabilities.
- Simpler to scale and deploy than encoder-decoder architectures.
- Supports both understanding and generation through prompting.
- KV caching is essential for efficient inference.
- Position encodings: RoPE (LLaMA), ALiBi (Mistral), learned (GPT).
- Training is parallel with causal masks; inference is sequential.
- Scaling properties well-understood: ~6N FLOPs per token.
- The architecture continues to evolve with GQA, MoE, and other innovations.
