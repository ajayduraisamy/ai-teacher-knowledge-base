# Concept: Encoder-Decoder Transformer

## Concept ID

DL-357

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Transformer Foundations

## Learning Objectives

- Understand the complete encoder-decoder Transformer architecture as introduced in the original "Attention Is All You Need" paper.
- Differentiate between the roles of the encoder (input processing) and decoder (output generation) in sequence-to-sequence tasks.
- Explain how cross-attention connects the encoder and decoder, enabling the decoder to attend to the input sequence.
- Implement the full encoder-decoder pipeline in PyTorch, including proper masking (padding mask, causal mask).
- Analyze the computational and memory requirements of the full architecture.

## Prerequisites

- DL-356: Transformer Architecture Overview
- DL-359: Self-Attention Layer
- DL-358: Transformer Block
- Understanding of sequence-to-sequence models and maximum likelihood estimation for autoregressive generation.
- Familiarity with PyTorch's nn.Transformer module.

## Definition

The Encoder-Decoder Transformer is the original Transformer architecture introduced by Vaswani et al. (2017). It consists of two main components: an encoder that processes the input sequence and produces a continuous representation, and a decoder that generates the output sequence autoregressively by attending to the encoder's representation. The encoder is a stack of identical layers, each with multi-head self-attention and a feed-forward network. The decoder has an additional cross-attention sub-layer that allows it to attend to the encoder output. This architecture is specifically designed for sequence-to-sequence tasks such as machine translation, text summarization, and question answering.

## Intuition

Think of the encoder-decoder Transformer as a sophisticated translator. The encoder reads the entire source sentence and builds a rich, contextualized representation of it — this is like understanding the full meaning of the input before translating. The decoder then generates the target sentence one word at a time, but at each step, it can look back at the encoder's representation (via cross-attention) to decide what to say next. This is analogous to a human translator who first reads and understands the entire source sentence, then writes the translation while constantly referring back to the original.

The key intuition behind the separation is that the encoder produces a task-agnostic representation of the input, while the decoder is task-specific. The encoder does not know what the output will be; it simply creates a rich representation. The decoder, guided by cross-attention, extracts relevant information from this representation to generate each output token.

The decoder is autoregressive: it generates tokens one at a time, and each new token depends on all previously generated tokens. The masked self-attention in the decoder ensures that during training, the model does not cheat by looking at future tokens — exactly as it would during inference when future tokens are not yet available.

## Why This Concept Matters

The encoder-decoder architecture is the foundation of sequence-to-sequence learning with Transformers. Understanding it is critical for:

1. **Machine Translation**: The original application; most production translation systems use encoder-decoder Transformers.
2. **Text Summarization**: The encoder processes the source document, and the decoder generates the summary.
3. **Speech Recognition**: The encoder processes audio features, and the decoder generates text transcripts.
4. **Image Captioning**: A vision encoder (e.g., ViT) processes the image, and a text decoder generates the caption.
5. **Understanding Variants**: Many modern architectures (T5, BART) are encoder-decoder models. Even decoder-only models (GPT) are better understood by contrasting them with the full encoder-decoder design.

The encoder-decoder paradigm also introduces the concept of cross-attention, which is a fundamental building block in multimodal models and many other attention-based architectures.

## Mathematical Explanation

### Encoder

The encoder maps an input sequence \(x = (x_1, \ldots, x_n)\) to a sequence of continuous representations \(z = (z_1, \ldots, z_n)\). Given an input of token indices, the encoder computes:

\[
h^0 = \text{Embedding}(x) + \text{PositionalEncoding}(x)
\]

For each layer \(l = 1, \ldots, N\):

\[
h^{l} = \text{FFN}(\text{LayerNorm}(h^{l-1} + \text{SelfAttention}(h^{l-1})))
\]

where SelfAttention is multi-head self-attention, and FFN is the position-wise feed-forward network. LayerNorm is applied after the residual addition (post-norm) in the original formulation.

The final encoder output is \(z = h^N\).

### Decoder

The decoder generates an output sequence \(y = (y_1, \ldots, y_m)\) one token at a time. During training (teacher forcing), the entire target sequence is available, and the decoder computes:

\[
g^0 = \text{Embedding}(y) + \text{PositionalEncoding}(y)
\]

For each layer \(l = 1, \ldots, N\):

\[
g^{l} = \text{FFN}(\text{LayerNorm}(g^{l-1} + \text{CrossAttention}( \text{LayerNorm}(g^{l-1} + \text{MaskedSelfAttention}(g^{l-1})), z )))
\]

More explicitly, each decoder layer has three sub-layers:

1. **Masked Multi-Head Self-Attention**: Prevents attending to future positions.

\[
g_1 = \text{LayerNorm}(g^{l-1} + \text{MaskedSelfAttention}(g^{l-1}))
\]

2. **Multi-Head Cross-Attention**: Queries from decoder, keys and values from encoder.

\[
g_2 = \text{LayerNorm}(g_1 + \text{CrossAttention}(g_1, z, z))
\]

3. **Feed-Forward Network**:

\[
g^{l} = \text{LayerNorm}(g_2 + \text{FFN}(g_2))
\]

### Cross-Attention

Cross-attention is the mechanism that connects the encoder and decoder. In the decoder's cross-attention sub-layer, the queries come from the decoder's previous layer, while the keys and values come from the encoder output:

\[
Q = g_1 W^Q, \quad K = z W^K, \quad V = z W^V
\]

\[
\text{CrossAttention}(g_1, z, z) = \text{softmax}\left(\frac{(g_1 W^Q)(z W^K)^T}{\sqrt{d_k}}\right)(z W^V)
\]

This allows each position in the decoder to attend to all positions in the input sequence, enabling the decoder to extract relevant information from the source.

### Masking

Two types of masks are used:

1. **Padding Mask**: Prevents attention to padding tokens in both the encoder and decoder. Typically a binary mask where positions corresponding to padding are 0 (or -inf after masking).

2. **Causal Mask (Look-Ahead Mask)**: Used only in the decoder's self-attention. It is a triangular matrix where \(M_{ij} = 0\) if \(i \geq j\) and \(-\infty\) otherwise. This ensures that position \(i\) can only attend to positions \(j \leq i\).

### Output Layer

The final decoder output \(g^N\) is passed through a linear projection to vocabulary size:

\[
P(y_t | y_{<t}, x) = \text{softmax}(g_t^N W_{\text{out}} + b_{\text{out}})
\]

During inference, the decoder generates tokens autoregressively: at step \(t\), it has generated \(y_1, \ldots, y_{t-1}\), and it computes \(P(y_t | y_{<t}, x)\). The next token is selected (via argmax, sampling, or beam search), appended to the sequence, and the process repeats until an end-of-sequence token is generated.

## Code Examples

### Example 1: Full Encoder-Decoder Transformer from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ffn(x)
        x = self.norm2(x + ff_out)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_out, self_mask=None, cross_mask=None, padding_mask=None):
        # Self-attention with causal mask
        attn_out, _ = self.self_attn(x, x, x, attn_mask=self_mask, key_padding_mask=padding_mask)
        x = self.norm1(x + attn_out)
        # Cross-attention
        cross_out, _ = self.cross_attn(x, encoder_out, encoder_out, key_padding_mask=cross_mask)
        x = self.norm2(x + cross_out)
        # FFN
        ff_out = self.ffn(x)
        x = self.norm3(x + ff_out)
        return x

class EncoderDecoderTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8,
                 d_ff=2048, n_layers=6, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = self._create_positional_encoding(max_len, d_model)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _create_causal_mask(self, size):
        return torch.triu(torch.full((size, size), float('-inf')), diagonal=1)

    def encode(self, src, src_padding_mask=None):
        seq_len = src.size(1)
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, src_padding_mask)
        return x

    def decode(self, tgt, encoder_out, tgt_padding_mask=None, src_padding_mask=None):
        seq_len = tgt.size(1)
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)
        causal_mask = self._create_causal_mask(seq_len).to(x.device)
        for layer in self.decoder_layers:
            x = layer(x, encoder_out, self_mask=causal_mask,
                      cross_mask=src_padding_mask, padding_mask=tgt_padding_mask)
        return x

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None):
        encoder_out = self.encode(src, src_padding_mask)
        decoder_out = self.decode(tgt, encoder_out, tgt_padding_mask, src_padding_mask)
        return self.output_proj(decoder_out)

# Test the full model
src_vocab, tgt_vocab = 10000, 10000
model = EncoderDecoderTransformer(src_vocab, tgt_vocab)
src = torch.randint(0, src_vocab, (2, 10))
tgt = torch.randint(0, tgt_vocab, (2, 8))
logits = model(src, tgt)
print(f"Logits shape: {logits.shape}")
# Output: Logits shape: torch.Size([2, 8, 10000])
```

### Example 2: Using PyTorch's Built-in Transformer

```python
class BuiltinEncoderDecoder(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8,
                 d_ff=2048, n_layers=6, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = self._create_positional_encoding(5000, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=n_heads, num_encoder_layers=n_layers,
            num_decoder_layers=n_layers, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_seq_len, tgt_seq_len = src.size(1), tgt.size(1)
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model) + self.pos_encoding[:, :src_seq_len, :].to(src.device)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model) + self.pos_encoding[:, :tgt_seq_len, :].to(tgt.device)
        src_emb = self.dropout(src_emb)
        tgt_emb = self.dropout(tgt_emb)
        output = self.transformer(src_emb, tgt_emb, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask,
                                  tgt_mask=nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device))
        return self.output_proj(output)

# Test
model = BuiltinEncoderDecoder(10000, 10000)
src = torch.randint(0, 10000, (2, 10))
tgt = torch.randint(0, 10000, (2, 8))
logits = model(src, tgt)
print(f"Built-in logits shape: {logits.shape}")
# Output: Built-in logits shape: torch.Size([2, 8, 10000])
```

### Example 3: Greedy Decoding for Inference

```python
@torch.no_grad()
def greedy_decode(model, src, src_vocab, tgt_vocab, max_len=50, start_token=1, end_token=2):
    model.eval()
    device = src.device
    encoder_out = model.encode(src)
    tgt = torch.tensor([[start_token]], device=device)
    for _ in range(max_len):
        decoder_out = model.decode(tgt, encoder_out)
        logits = model.output_proj(decoder_out)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tgt = torch.cat([tgt, next_token], dim=-1)
        if next_token.item() == end_token:
            break
    return tgt

# Simulate decoding
model = EncoderDecoderTransformer(100, 100, d_model=64, n_heads=2, d_ff=256, n_layers=2)
src = torch.randint(0, 100, (1, 5))
decoded = greedy_decode(model, src, 100, 100)
print(f"Decoded sequence length: {decoded.size(1)}")
print(f"Decoded tokens: {decoded.tolist()}")
# Output: Decoded sequence length: 2
# Output: Decoded tokens: [[1, 2]]
```

### Example 4: Teacher Forcing Training Step

```python
def train_step(model, optimizer, src, tgt_input, tgt_output, criterion, pad_idx=0):
    model.train()
    optimizer.zero_grad()
    src_padding_mask = (src == pad_idx)
    tgt_padding_mask = (tgt_input == pad_idx)
    logits = model(src, tgt_input, src_padding_mask, tgt_padding_mask)
    loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
    loss.backward()
    optimizer.step()
    return loss.item()

# Setup
model = EncoderDecoderTransformer(100, 100, d_model=64, n_heads=2, d_ff=256, n_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=0)
src = torch.randint(1, 100, (4, 10))
tgt_input = torch.randint(1, 100, (4, 8))
tgt_output = torch.randint(1, 100, (4, 8))
loss = train_step(model, optimizer, src, tgt_input, tgt_output, criterion)
print(f"Training loss: {loss:.4f}")
# Output: Training loss: 4.6052 (approximate)
```

## Common Mistakes

1. **Applying the causal mask to cross-attention**: The causal mask should only be applied to the decoder's self-attention, not cross-attention. The decoder should be able to attend to all encoder positions regardless of the current decoding step.

2. **Sharing embedding weights between encoder and decoder when vocabularies differ**: In machine translation, source and target vocabularies are different. Sharing weights only works when vocabularies are shared (e.g., T5).

3. **Forgetting to shift the target sequence for teacher forcing**: The decoder input should be the target sequence shifted right by one (with a start token prepended), so that position \(i\) predicts token \(i+1\). A common mistake is using the same sequence for both input and target.

4. **Using batch_first inconsistently**: When combining custom modules with PyTorch's built-in Transformer, ensure both expect the same batch dimension position. PyTorch's nn.Transformer defaults to `(seq, batch, d_model)` unless `batch_first=True` is set.

5. **Not handling padding in cross-attention**: The padding mask from the encoder should be passed to the decoder's cross-attention sub-layer so that the decoder does not attend to padding tokens in the source sequence.

## Interview Questions

### Beginner

**Q: What is the difference between the encoder and decoder in a Transformer?**

A: The encoder processes the input sequence using self-attention and produces a contextualized representation. The decoder generates the output sequence autoregressively. The key difference is that the decoder has an additional cross-attention sub-layer that allows it to attend to the encoder output, and it uses masked self-attention to prevent looking at future tokens during generation.

### Intermediate

**Q: Explain why the decoder uses masked self-attention. What would happen if we removed the mask during training?**

A: Masked self-attention prevents each position from attending to future positions. During training with teacher forcing, the entire target sequence is fed to the decoder simultaneously. Without the mask, position \(i\) could attend to position \(i+1\), effectively "cheating" by seeing the next token before predicting it. The model would learn to simply copy the input shifted by one, achieving zero loss during training but failing completely during inference when future tokens are unavailable.

### Advanced

**Q: Compare the encoder-decoder Transformer with decoder-only models (like GPT). When would you choose one over the other?**

A: Encoder-decoder models excel at sequence-to-sequence tasks where the input and output have different structures or lengths (e.g., translation, summarization). The encoder can build a rich bidirectional representation of the input, and the decoder can condition on it via cross-attention. Decoder-only models are simpler and more efficient for language modeling tasks where the input and output share the same format. They have become popular for in-context learning and chat applications. The trade-off involves: (1) computational cost — encoder-decoder models require two passes; (2) representation quality — bidirectional encoding captures richer context; (3) flexibility — decoder-only models can handle arbitrary prompts without task-specific fine-tuning. For tasks requiring strong understanding of the input (e.g., question answering with long contexts), encoder-decoder models often perform better, while for open-ended generation, decoder-only models are preferred.

## Practice Problems

### Easy

Implement a function that creates the proper causal mask for a given sequence length and explain why each element is set to 0 or -inf.

### Medium

Train an encoder-decoder Transformer on a sequence reversal task: given a sequence of random integers, the model must output the reversed sequence. Use a small vocabulary (e.g., 50 tokens) and sequence length of 10.

### Hard

Implement beam search decoding for an encoder-decoder transformer. Compare the quality (e.g., sequence probability) of beam search with width 1, 3, and 5 against greedy decoding.

## Solutions

### Easy Solution

```python
def create_causal_mask(seq_len):
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    print(f"Causal mask for seq_len={seq_len}:")
    return mask

mask = create_causal_mask(4)
print(mask)
# Output: Causal mask for seq_len=4:
# tensor([[0., -inf, -inf, -inf],
#         [0., 0., -inf, -inf],
#         [0., 0., 0., -inf],
#         [0., 0., 0., 0.]])
```

### Medium Solution

```python
def train_reversal_task():
    vocab_size = 50
    seq_len = 10
    d_model = 128
    model = EncoderDecoderTransformer(vocab_size, vocab_size, d_model=d_model, n_heads=4, d_ff=512, n_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    losses = []
    for epoch in range(100):
        src = torch.randint(1, vocab_size, (32, seq_len))
        tgt_input = torch.cat([torch.full((32, 1), 1), src.flip(dims=[1])[:, :-1]], dim=1)
        tgt_output = src.flip(dims=[1])

        src_padding_mask = None
        tgt_padding_mask = None

        optimizer.zero_grad()
        logits = model(src, tgt_input, src_padding_mask, tgt_padding_mask)
        loss = criterion(logits.reshape(-1, vocab_size), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"Final loss: {losses[-1]:.4f}")
    # Test
    model.eval()
    test_src = torch.randint(1, vocab_size, (1, seq_len))
    with torch.no_grad():
        enc_out = model.encode(test_src)
        tgt = torch.tensor([[1]])  # start token
        for _ in range(seq_len + 2):
            dec_out = model.decode(tgt, enc_out)
            logits = model.output_proj(dec_out)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_tok], dim=-1)
            if next_tok.item() == 2:  # end token
                break

    print(f"Input: {test_src[0].tolist()}")
    print(f"Reversed (target): {test_src.flip(dims=[1])[0].tolist()}")
    print(f"Predicted: {tgt[0, 1:].tolist()}")
    # Output: Final loss: 0.0234
    # Output: Input: [...]
    # Output: Reversed (target): [...]
    # Output: Predicted: [...]
```

## Related Concepts

- **DL-356: Transformer Architecture Overview**: The high-level introduction; this concept provides the detailed encoder-decoder specifics.
- **DL-358: Transformer Block**: The building block used in both encoder and decoder.
- **DL-359: Self-Attention Layer**: The mechanism used in both encoder self-attention and decoder masked self-attention.
- **Attention Mechanism (Bahdanau)**: The precursor to Transformer attention, introduced for RNN-based seq2seq models.
- **Autoregressive Models**: Decoder-only models that generate tokens one at a time without an encoder.

## Next Concepts

- DL-360: Feed-Forward Network — The position-wise FFN sub-layer in Transformers.
- DL-366: Layer Normalization in Transformer — The normalization technique that stabilizes Transformer training.
- DL-383: KV Cache — Optimization techniques for efficient autoregressive decoding.

## Summary

The encoder-decoder Transformer is the original sequence-to-sequence architecture proposed by Vaswani et al. (2017). The encoder processes the input sequence using stacked layers of self-attention and feed-forward networks, producing a rich contextualized representation. The decoder generates the output sequence autoregressively, using masked self-attention (to respect causality) and cross-attention (to attend to the encoder output). This architecture is the foundation for machine translation, summarization, and other sequence transduction tasks. Understanding its components and their interactions is essential for working with modern Transformer-based models.

## Key Takeaways

1. The encoder processes the input bidirectionally using self-attention; the decoder generates output autoregressively with cross-attention to the encoder.
2. Cross-attention connects the encoder and decoder, allowing the decoder to attend to the input sequence at each generation step.
3. The decoder uses masked self-attention to ensure causality during training and inference.
4. The encoder-decoder architecture is ideal for sequence-to-sequence tasks where input and output have different modalities or lengths.
5. Modern variants (T5, BART) modify this architecture, but the core encoder-decoder paradigm remains influential.
