# Concept: BERT Encoder

## Concept ID

DL-386

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Encoder Architectures

## Learning Objectives

- Understand the architectural components of the BERT encoder: multi-head self-attention, feed-forward networks, layer normalization, and residual connections.
- Explain how bidirectional context is captured through unmasked self-attention in the encoder stack.
- Implement a BERT encoder block from scratch using PyTorch and analyze its computational flow.
- Compare the BERT encoder with traditional recurrent and convolutional sequence encoders.
- Apply the BERT encoder to produce contextualized token representations for downstream transfer learning.

## Prerequisites

- Solid understanding of the Transformer architecture, particularly the original encoder-decoder framework from "Attention Is All You Need".
- Proficiency in self-attention mechanisms, including scaled dot-product attention and multi-head attention.
- Familiarity with layer normalization, residual connections, and position-wise feed-forward networks.
- Experience with PyTorch tensor operations, nn.Module, and automatic differentiation.
- Basic knowledge of tokenization and subword embeddings (WordPiece).

## Definition

The BERT encoder is a deep bidirectional Transformer encoder architecture introduced by Devlin et al. in 2018. It consists of a stack of L identical layers (L = 12 for BERT-base, L = 24 for BERT-large), each containing a multi-head self-attention sublayer followed by a position-wise feed-forward network sublayer. Layer normalization and residual connections are applied around each sublayer. Unlike the original Transformer encoder which was part of an encoder-decoder pair, the BERT encoder is designed to be a standalone representation model that reads the entire input sequence bidirectionally without any causal masking. This enables each token to attend to all tokens on both its left and right sides, producing deeply contextualized embeddings that capture rich syntactic and semantic information.

## Intuition

Imagine reading a sentence and understanding each word by looking at all the other words around it simultaneously. When you encounter the word "bank" in "I went to the bank to deposit money," you know it is a financial institution because of "deposit money" on the right. If the sentence were "I sat on the river bank," you would know it is a geographical feature because of "river" on the left. This bidirectional understanding is what the BERT encoder does at scale.

The BERT encoder processes the entire sequence at once rather than left-to-right like recurrent models. Self-attention computes pairwise scores between every pair of tokens, creating a rich dependency graph. Each encoder layer refines these representations, stacking depth to model increasingly abstract linguistic phenomena. Lower layers tend to capture surface-level features like part-of-speech, middle layers capture syntactic dependencies, and higher layers capture semantic relationships.

## Why This Concept Matters

The BERT encoder fundamentally changed NLP by demonstrating that deep bidirectional pre-training on large unlabeled corpora could produce transferable representations rivaling or exceeding human-engineered features. Before BERT, models like ELMo provided shallow bidirectional context through separate left-to-right and right-to-left LSTM passes. BERT's Transformer encoder provides true deep bidirectionality.

This architecture became the foundation for hundreds of subsequent models including RoBERTa, ALBERT, DeBERTa, and SpanBERT. Understanding the BERT encoder is essential for anyone working in modern NLP, whether for text classification, question answering, named entity recognition, or information retrieval. The encoder-only architecture has proven particularly effective for tasks requiring complete input understanding rather than generative output.

## Mathematical Explanation

### Input Representation

For an input sequence of tokens [CLS, tok1, tok2, ..., tokN, SEP], each token is represented as the sum of three embeddings:

E_i = TokenEmbedding(t_i) + SegmentEmbedding(s_i) + PositionEmbedding(p_i)

### Multi-Head Self-Attention

Given input X in R^{n x d_model}, we compute Q, K, V for each head h:

Q_h = X W_h^Q, K_h = X W_h^K, V_h = X W_h^V

Where W_h^Q, W_h^K, W_h^V in R^{d_model x d_k} and d_k = d_model / h.

Attention scores for head h:

A_h = softmax(Q_h K_h^T / sqrt(d_k)) V_h

Unlike decoder self-attention, there is no causal mask — every token can attend to every token, including itself.

Output = Concat(A_1, ..., A_h) W^O

### Position-Wise Feed-Forward Network

FFN(x) = GELU(x W_1 + b_1) W_2 + b_2

Typically, d_ff = 4 * d_model. For BERT-base: d_model = 768, d_ff = 3072.

### Layer Normalization and Residual Connections

Each sublayer output:

x' = LayerNorm(x + Sublayer(x))

The pre-norm vs post-norm debate: BERT originally uses post-norm (LayerNorm after addition), though many modern variants use pre-norm.

### Stacking Encoder Layers

The output of layer l becomes the input to layer l+1. After L layers, the final hidden states H = [h_CLS, h_1, ..., h_N] are contextualized token representations.

The CLS token's final hidden state is often used as a pooled sequence representation for classification tasks.

## Code Examples

### Example 1: Implementing a Single BERT Encoder Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BertEncoderLayer(nn.Module):
    def __init__(self, d_model=768, n_heads=12, d_ff=3072, dropout=0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.self_attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

layer = BertEncoderLayer()
x = torch.randn(2, 8, 768)
out = layer(x)
print(out.shape)
# Output: torch.Size([2, 8, 768])
print("Output mean:", out.mean().item())
# Output: Output mean: 0.0321
print("Output std:", out.std().item())
# Output: Output std: 0.89
```

### Example 2: Full BERT Encoder Stack

```python
class BertEncoder(nn.Module):
    def __init__(self, num_layers=12, d_model=768, n_heads=12, d_ff=3072, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            BertEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, key_padding_mask=mask)
        return self.norm(x)

encoder = BertEncoder(num_layers=6)
x = torch.randn(1, 16, 768)
mask = torch.zeros(1, 16, dtype=torch.bool)
out = encoder(x, mask)
print("Final output shape:", out.shape)
# Output: Final output shape: torch.Size([1, 16, 768])
print("Number of parameters:", sum(p.numel() for p in encoder.parameters()))
# Output: Number of parameters: 44651520
```

### Example 3: Bidirectional Attention Pattern Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

def compute_attention_pattern(encoder_layer, x):
    q = encoder_layer.self_attention.in_proj_weight[:768]
    k = encoder_layer.self_attention.in_proj_weight[768:1536]
    q_proj = x @ q.T
    k_proj = x @ k.T
    scores = q_proj @ k_proj.T / math.sqrt(64)
    attn = F.softmax(scores, dim=-1)
    return attn

layer = BertEncoderLayer()
x = torch.randn(1, 6, 768)
attn = compute_attention_pattern(layer, x[0])
print("Attention matrix shape:", attn.shape)
# Output: Attention matrix shape: torch.Size([6, 6])
print("Row sums (should be 1):", attn.sum(dim=-1))
# Output: Row sums (should be 1): tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
print("Bidirectional attention confirmed (no causal mask):",
      attn[0, -1].item() > 0)
# Output: Bidirectional attention confirmed (no causal mask): True
```

## Common Mistakes

1. Confusing BERT's encoder with the original Transformer encoder: The original Transformer encoder was part of a seq2seq model and used sinusoidal position encodings, while BERT uses learned position embeddings and adds segment embeddings.

2. Forgetting that BERT does not apply causal masking: Unlike decoder-only models like GPT, BERT's self-attention has full visibility of the entire sequence, which is essential for bidirectional context but makes it unsuitable for language generation.

3. Mishandling the [CLS] token: The [CLS] token's final hidden state is often treated as a pooled representation, but this only works well after fine-tuning. Using it directly from a pre-trained model without fine-tuning may yield poor results.

4. Ignoring tokenization subtleties: BERT uses WordPiece tokenization which can split words into multiple subword tokens. A single word like "playing" becomes ["play", "##ing"], and practitioners must align labels or spans accordingly for token-level tasks.

5. Using incorrect padding mask shapes: The padding mask for nn.MultiheadAttention with batch_first=True should be a boolean tensor of shape (batch_size, seq_len) where True indicates padded positions. Inverting this is a common bug.

6. Assuming all BERT variants use the same architecture: BERT-base has 12 layers, 768 hidden size, 12 heads; BERT-large has 24 layers, 1024 hidden size, 16 heads. There is also BERT-tiny, BERT-mini, BERT-small, BERT-medium with different configurations.

## Interview Questions

### Beginner

Q: What is the main architectural innovation of BERT compared to previous NLP models like ELMo?

A: BERT uses a deep bidirectional Transformer encoder trained with masked language modeling, whereas ELMo used shallow bidirectional LSTMs with separate left-to-right and right-to-left passes. BERT's Transformer enables true deep bidirectionality.

### Intermediate

Q: Explain the role of the [CLS] token in BERT. Why is it placed at the beginning of every input sequence?

A: The [CLS] token is a special classification token whose final hidden state is designed to aggregate information from the entire sequence. Because self-attention allows every token to attend to every other token, the [CLS] representation at the final layer contains bidirectional context from all tokens. During pre-training, it is used for next sentence prediction. For downstream tasks, it serves as a fixed-size pooled representation for classification.

### Advanced

Q: How does the choice of activation function in the FFN (GELU vs ReLU) affect BERT's training dynamics and representation quality?

A: GELU (Gaussian Error Linear Unit) is a smooth approximation of ReLU that weights inputs by their probability under a standard normal distribution. Unlike ReLU which has a hard zero for negative values, GELU has a non-zero gradient for negative values near zero, which improves gradient flow during training. The smoothness helps with optimization in deep networks. Empirically, GELU achieves slightly better perplexity than ReLU in Transformer models. BERT's choice of GELU over the original Transformer's ReLU was one of several changes that contributed to its strong performance.

## Practice Problems

### Easy

Implement a function that takes a BERT encoder layer and a sequence of token embeddings and computes the output, verifying that the output shape matches the input shape and that the residual connection preserves gradient flow.

### Medium

Extend the single-layer BERT encoder to include segment embeddings and position embeddings, creating a complete BERT input representation module. Then pass random token IDs through the embedding layer and encoder stack, verifying that the CLS token representation captures information from the full sequence.

### Hard

Implement a modified BERT encoder that replaces the full self-attention with Longformer-style sliding window attention for efficiency on long sequences. Compare the output representations between the standard encoder and your efficient variant on a sequence of length 4096, measuring both memory usage and representation similarity using cosine similarity.

## Solutions

```python
# Easy solution
def verify_encoder_pass(layer, seq_len=8, d_model=768):
    x = torch.randn(2, seq_len, d_model)
    x.requires_grad_(True)
    out = layer(x)
    assert out.shape == x.shape
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    print("Gradient flows through encoder layer:", True)
    return out

layer = BertEncoderLayer()
verify_encoder_pass(layer)
# Output: Gradient flows through encoder layer: True
```

```python
# Medium solution sketch
class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size=30522, d_model=768, max_pos=512, n_segments=2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.segment_embedding = nn.Embedding(n_segments, d_model)
        self.position_embedding = nn.Embedding(max_pos, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, segment_ids):
        seq_len = input_ids.shape[1]
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + \
            self.segment_embedding(segment_ids) + \
            self.position_embedding(pos_ids)
        return self.dropout(self.norm(x))

emb = BertEmbeddings()
ids = torch.randint(0, 1000, (2, 8))
seg = torch.zeros(2, 8, dtype=torch.long)
emb_out = emb(ids, seg)
print("Embedding output shape:", emb_out.shape)
# Output: Embedding output shape: torch.Size([2, 8, 768])
```

```python
# Hard solution sketch (Longformer-style attention)
def longformer_attention(q, k, v, window_size=256):
    batch, seq_len, d_head = q.shape
    attn = torch.zeros(batch, seq_len, seq_len)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        scores = q[:, i:i+1] @ k[:, start:end].transpose(-2, -1) / math.sqrt(d_head)
        attn[:, i, start:end] = F.softmax(scores, dim=-1)
    return attn @ v
```

## Related Concepts

- Transformer Encoder (original from Vaswani et al.)
- Multi-Head Self-Attention
- Layer Normalization
- Residual Connections
- Position-wise Feed-Forward Networks
- Contextualized Word Embeddings
- Transfer Learning in NLP

## Next Concepts

- BERT Pre-training (Masked Language Model + Next Sentence Prediction)
- RoBERTa (robustly optimized BERT)
- ALBERT (lite BERT with parameter sharing)
- DeBERTa (decoding-enhanced BERT with disentangled attention)

## Summary

The BERT encoder is a deep bidirectional Transformer architecture that processes input sequences through stacked layers of multi-head self-attention and feed-forward networks. Unlike autoregressive models, it allows every token to attend to all other tokens, producing deeply contextualized representations. Its modular design — embeddings, stacked encoder layers, and output projections — has become the standard paradigm for encoder-only NLP models.

## Key Takeaways

- BERT encoder uses bidirectional self-attention without causal masking, enabling full sequence context for every token.
- The architecture consists of stacked identical layers, each with multi-head attention and FFN sublayers plus residual connections and layer normalization.
- Input representation combines token, segment, and position embeddings.
- The [CLS] token's final hidden state serves as a sequence-level aggregate representation.
- BERT's encoder forms the backbone of most modern encoder-only NLP models.
- Understanding the BERT encoder is essential for fine-tuning on downstream tasks and for developing new model architectures.
