# Concept: Cross-Attention

## Concept ID

DL-344

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Attention Mechanisms

## Learning Objectives

- Define cross-attention as attention between two different sequences.
- Understand the role of cross-attention in encoder-decoder architectures.
- Implement cross-attention in PyTorch within a transformer decoder.
- Differentiate between self-attention and cross-attention in terms of inputs and outputs.
- Analyze how cross-attention enables the decoder to retrieve information from the encoder.

## Prerequisites

- Solid understanding of self-attention and the query-key-value framework.
- Knowledge of the encoder-decoder architecture in transformers.
- Familiarity with transformer decoder layers.
- Understanding of how attention mechanisms relate sequences.

## Definition

Cross-attention is an attention mechanism where the queries come from one sequence (e.g., the decoder) and the keys and values come from another sequence (e.g., the encoder). Formally, given queries Q in R^{batch x T_q x d_k} from sequence X_Q, and keys K in R^{batch x T_k x d_k} and values V in R^{batch x T_k x d_v} from sequence X_KV:

CrossAttention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

where Q = X_Q W^Q, K = X_KV W^K, V = X_KV W^V. The output has shape (batch, T_q, d_v) — the same sequence length as the query sequence. Cross-attention is the mechanism that allows the decoder to retrieve relevant information from the encoder in encoder-decoder architectures (like transformers for translation, T5, BART). It is also used in multimodal models (e.g., text queries attending to image features) and retrieval-augmented generation (e.g., queries attending to retrieved documents).

## Intuition

Cross-attention is like a translator who has a source document (the encoder output) and is writing a target document (the decoder output). For each word the translator writes, they look back at the source document to find the relevant information. The query is the translator's current focus ("What word am I writing now?"). The keys are the source document's table of contents ("What information is available?"). The values are the actual content of the source document. Cross-attention computes which parts of the source document are most relevant for the current word being written, retrieves that content, and incorporates it into the writing process. In a transformer, cross-attention is repeated at every decoder layer, allowing the decoder to iteratively refine its understanding of the source.

## Why This Concept Matters

Cross-attention is the critical bridge between encoder and decoder in encoder-decoder models. It is the mechanism that enables conditional generation: the decoder doesn't just generate text from scratch, it generates text conditioned on the encoder's understanding of the input. Cross-attention is what makes translation, summarization, and image captioning possible in transformer architectures. Understanding cross-attention is essential for: (1) building encoder-decoder models for conditional generation, (2) understanding how information flows from encoder to decoder, (3) implementing multimodal models where different modalities are bridged through cross-attention, and (4) designing retrieval-augmented generation systems where retrieved documents are attended to via cross-attention.

## Mathematical Explanation

### Standard Cross-Attention

Given decoder hidden states H_dec in R^{batch x T_dec x d_model} and encoder outputs H_enc in R^{batch x T_enc x d_model}:

Q = H_dec W^Q, W^Q in R^{d_model x d_k}
K = H_enc W^K, W^K in R^{d_model x d_k}
V = H_enc W^V, W^V in R^{d_model x d_v}

Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Output in R^{batch x T_dec x d_v}

### Cross-Attention in Transformer Decoder

In each transformer decoder layer, three sublayers are applied in order:

1. Masked self-attention: H_dec <- SelfAttention(H_dec)
2. Cross-attention: H_dec <- CrossAttention(H_dec, H_enc, H_enc)
3. Feedforward: H_dec <- FFN(H_dec)

The cross-attention uses the decoder's self-attention output as queries and the encoder's output as keys and values.

### Multi-Head Cross-Attention

head_i = CrossAttention(H_dec W_i^Q, H_enc W_i^K, H_enc W_i^V)
MultiHeadCross(H_dec, H_enc) = Concat(head_1, ..., head_h) W^O

### Cross-Attention vs. Self-Attention

| Aspect | Self-Attention | Cross-Attention |
|--------|---------------|-----------------|
| Query source | Same as K, V | Different from K, V |
| Sequence length | T from single sequence | T_q (queries) and T_k (keys) |
| Purpose | Contextualize tokens within sequence | Retrieve info from another sequence |
| Masking | Causal for decoder | None (full access to encoder) |
| Output length | Same as input T | Same as query length T_q |

## Code Examples

### Example 1: Basic Cross-Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)
        self.scale = math.sqrt(d_k)

    def forward(self, decoder_states, encoder_outputs, mask=None):
        Q = self.W_q(decoder_states)
        K = self.W_k(encoder_outputs)
        V = self.W_v(encoder_outputs)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        return context, attn_weights

d_model, d_k, d_v = 16, 8, 8
cross_attn = CrossAttention(d_model, d_k, d_v)
dec_states = torch.randn(2, 5, d_model)
enc_outputs = torch.randn(2, 7, d_model)
context, weights = cross_attn(dec_states, enc_outputs)
print(f"Context shape: {context.shape} (decoder length x d_v)")
print(f"Weights shape: {weights.shape} (decoder_len x encoder_len)")
# Output: Context shape: torch.Size([2, 5, 8]) (decoder length x d_v)
# Output: Weights shape: torch.Size([2, 5, 7]) (decoder_len x encoder_len)
```

### Example 2: Transformer Decoder Layer with Cross-Attention

```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.self_attn_q = nn.Linear(d_model, d_model)
        self.self_attn_k = nn.Linear(d_model, d_model)
        self.self_attn_v = nn.Linear(d_model, d_model)
        self.cross_attn_q = nn.Linear(d_model, d_model)
        self.cross_attn_k = nn.Linear(d_model, d_model)
        self.cross_attn_v = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_outputs, self_mask=None, cross_mask=None):
        batch, seq_len = x.shape[0], x.shape[1]
        Q = self.self_attn_q(x).view(batch, seq_len, -1, self.d_k).transpose(1, 2)
        K = self.self_attn_k(x).view(batch, seq_len, -1, self.d_k).transpose(1, 2)
        V = self.self_attn_v(x).view(batch, seq_len, -1, self.d_k).transpose(1, 2)
        scores_self = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if self_mask is not None:
            scores_self = scores_self.masked_fill(self_mask == 0, -1e9)
        attn_self = F.softmax(scores_self, dim=-1)
        x = torch.matmul(attn_self, V).transpose(1, 2).contiguous().view(batch, seq_len, -1)
        x = self.norm1(x + self.dropout(x))

        enc_seq_len = encoder_outputs.shape[1]
        Q = self.cross_attn_q(x).view(batch, seq_len, -1, self.d_k).transpose(1, 2)
        K = self.cross_attn_k(encoder_outputs).view(batch, enc_seq_len, -1, self.d_k).transpose(1, 2)
        V = self.cross_attn_v(encoder_outputs).view(batch, enc_seq_len, -1, self.d_k).transpose(1, 2)
        scores_cross = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if cross_mask is not None:
            scores_cross = scores_cross.masked_fill(cross_mask == 0, -1e9)
        attn_cross = F.softmax(scores_cross, dim=-1)
        context = torch.matmul(attn_cross, V).transpose(1, 2).contiguous().view(batch, seq_len, -1)
        x = self.norm2(x + self.dropout(context))

        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x, attn_cross

dec_layer = TransformerDecoderLayer(d_model=32, n_heads=4, d_ff=128)
x = torch.randn(2, 5, 32)
enc_out = torch.randn(2, 7, 32)
output, cross_attn = dec_layer(x, enc_out)
print(f"Decoder layer output: {output.shape}")
print(f"Cross-attention shape: {cross_attn.shape}")
# Output: Decoder layer output: torch.Size([2, 5, 32])
# Output: Cross-attention shape: torch.Size([2, 4, 5, 7])
```

### Example 3: Cross-Attention for Retrieval-Augmented Generation

```python
class RetrievalCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.cross_attn = CrossAttention(d_model, d_model, d_model)
        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, query, retrieved_docs, mask=None):
        context, weights = self.cross_attn(query, retrieved_docs, mask)
        gate = torch.sigmoid(self.gate(torch.cat((query, context), dim=-1)))
        output = gate * context + (1 - gate) * query
        return output, weights

d_model = 32
rag_attn = RetrievalCrossAttention(d_model, 4)
query = torch.randn(2, 3, d_model)
docs = torch.randn(2, 10, d_model)
output, weights = rag_attn(query, docs)
print(f"RAG cross-attention output: {output.shape}")
# Output: RAG cross-attention output: torch.Size([2, 3, 32])
```

## Common Mistakes

1. **Confusing self-attention and cross-attention in the decoder**: The decoder has both self-attention (attending to previous decoder tokens) and cross-attention (attending to encoder outputs). Failing to apply both or mixing up their purposes leads to incorrect behavior.

2. **Applying causal masking to cross-attention**: Cross-attention should NOT be causally masked. The decoder should see all encoder positions. Only self-attention in the decoder needs causal masking.

3. **Forgetting that cross-attention changes the query sequence length**: Cross-attention produces output with the same length as the query sequence (decoder length), not the key/value sequence (encoder length). This is important when connecting layers.

4. **Using cross-attention queries from the wrong layer**: In transformer decoders, cross-attention queries come from the output of the self-attention sublayer, not from the raw decoder input. This allows the decoder to first contextualize its own tokens before retrieving encoder information.

5. **Not projecting dimensions properly**: The decoder hidden dimension and encoder hidden dimension must be compatible for cross-attention. If they differ (e.g., in multimodal models), projection layers are needed.

## Interview Questions

### Beginner

Q: What is the difference between self-attention and cross-attention?

A: In self-attention, queries, keys, and values all come from the same sequence. In cross-attention, queries come from one sequence (e.g., decoder) and keys/values come from another sequence (e.g., encoder). Self-attention contextualizes tokens within a sequence; cross-attention retrieves information from one sequence for use in another.

### Intermediate

Q: In a transformer encoder-decoder model, why does cross-attention use the encoder output as keys and values and the decoder state as queries? What would happen if this were reversed?

A: This is correct because the decoder needs to retrieve information from the encoder to generate each token. The decoder's current state (query) determines what information is needed, and the encoder outputs (keys/values) provide the available information. If reversed, the encoder would try to retrieve information from the decoder (which doesn't exist yet), and the model would fail at conditional generation.

### Advanced

Q: How would you design cross-attention for a multimodal model where the encoder is a vision transformer (ViT) and the decoder is a text transformer? What modifications to standard cross-attention are needed?

A: Several modifications are needed: (1) Dimensionality alignment: the ViT output dimension typically differs from the text decoder dimension. A learned projection or adapter layer is needed to map between them. (2) Object-centric prompts: ViT outputs patch representations, but text usually refers to objects. Adding object queries learned from the visual features can bridge this gap. (3) Spatial awareness: cross-attention should incorporate spatial positional information from the image (absolute positions, relative offsets, or learned spatial biases). (4) Modality-specific normalization: LayerNorm statistics differ between modalities, so separate normalization may be needed. (5) Multi-resolution cross-attention: images have hierarchical structure, so attending at multiple resolution levels (patch-level, region-level, image-level) can capture both fine details and global context.

## Practice Problems

### Easy

Implement a cross-attention module where queries come from sequence A (length 4) and keys/values from sequence B (length 6). Verify the output shape matches sequence A's length.

### Medium

Build a complete transformer decoder layer with both self-attention and cross-attention. Test it with random decoder states and encoder outputs, verifying that the output matches the decoder sequence length.

### Hard

Implement a two-level cross-attention for document summarization: the first level computes cross-attention between the decoder and sentence-level representations, and the second level computes cross-attention between the decoder and token-level representations within selected sentences.

## Solutions

### Easy Solution

```python
def cross_attention_simple(q_seq, kv_seq):
    d_k = q_seq.shape[-1]
    scores = torch.matmul(q_seq, kv_seq.transpose(-2, -1)) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    context = torch.matmul(weights, kv_seq)
    return context, weights

q = torch.randn(2, 4, 16)
kv = torch.randn(2, 6, 16)
c, w = cross_attention_simple(q, kv)
print(f"Output length: {c.shape[1]} (matches query length 4)")
# Output: Output length: 4 (matches query length 4)
```

## Related Concepts

- Self-Attention
- Encoder-Decoder Architecture
- Transformer Decoder
- Multimodal Attention
- Retrieval-Augmented Generation

## Next Concepts

- DL-345: Causal Masked Attention
- DL-346: Multi-Head Attention
- DL-352: Attention Is All You Need

## Summary

Cross-attention is an attention mechanism that allows one sequence (the query) to retrieve information from another sequence (the key/value). It is the critical bridge between encoder and decoder in encoder-decoder architectures, enabling conditional generation in translation, summarization, and multimodal tasks. In transformer decoders, cross-attention is applied after self-attention, using decoder states as queries and encoder outputs as keys and values. Cross-attention can also be used for retrieval-augmented generation and multimodal fusion.

## Key Takeaways

- Cross-attention retrieves information from one sequence for use in another.
- Queries come from the target sequence; keys and values from the source sequence.
- Cross-attention output has the same length as the query sequence.
- In transformers, cross-attention follows self-attention in each decoder layer.
- Cross-attention should NOT be causally masked (full access to source).
- Cross-attention is essential for conditional generation in encoder-decoder models.
- It is also used in multimodal models and retrieval-augmented generation.
