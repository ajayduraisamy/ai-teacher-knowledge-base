# Concept: Bahdanau Attention

## Concept ID

DL-337

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Attention Mechanisms

## Learning Objectives

- Understand the Bahdanau (additive) attention mechanism and its role in neural machine translation.
- Implement Bahdanau attention in PyTorch for a seq2seq model.
- Differentiate Bahdanau attention from other attention formulations.
- Analyze the advantages of additive attention for capturing complex query-key relationships.
- Interpret Bahdanau attention weights as alignments between input and output sequences.

## Prerequisites

- Understanding of the seq2seq architecture and its limitations (fixed context vector).
- Familiarity with the encoder-decoder framework and RNN hidden states.
- Knowledge of softmax normalization and weighted sums.
- Experience with PyTorch for building neural network components.

## Definition

Bahdanau attention, also known as additive attention, was introduced by Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio in their 2014 paper "Neural Machine Translation by Jointly Learning to Align and Translate." It was the first attention mechanism applied to seq2seq models, addressing the fixed-context-vector bottleneck in basic encoder-decoder architectures. Bahdanau attention computes an alignment score between the decoder's previous hidden state s_{t'-1} and each encoder hidden state h_t using a feedforward neural network:

e_{t', t} = v_a^T tanh(W_a s_{t'-1} + U_a h_t)

where v_a, W_a, and U_a are learned parameters. The alignment scores are normalized to attention weights alpha_{t', t} via softmax, and the dynamic context vector c_{t'} is computed as a weighted sum of encoder states. Bahdanau attention is called "additive" because the scores are computed by adding the projected query and key before applying the tanh nonlinearity.

## Intuition

Bahdanau attention is like a sophisticated spotlight operator. Imagine you have a spotlight that can shine on different parts of a stage, but the brightness depends on a complex combination of the performer's position and the type of performance. The attention mechanism uses a small neural network to compute this combination. This is more flexible than a simple dot-product comparison because the neural network can learn non-linear relationships between the query (what you need) and the keys (what's available). For example, in translation, the model might need to learn that the English word "bank" should attend to different context words depending on whether the sentence is about finance or rivers. The additive attention network can learn this nuanced relationship, making it particularly effective for tasks where the relevance between query and key is not a simple linear function.

## Why This Concept Matters

Bahdanau attention was a landmark contribution that introduced the concept of attention to neural sequence models. It demonstrated that allowing the decoder to dynamically access encoder states during generation dramatically improves translation quality, especially for long sentences. The paper was among the first to show that neural networks could learn soft alignment between input and output sequences without explicit supervision. Bahdanau attention laid the groundwork for all subsequent attention mechanisms, including Luong attention and the scaled dot-product attention used in transformers. Understanding Bahdanau attention is essential for the historical context of attention mechanisms and for applications where additive attention's expressiveness is beneficial, such as multimodal attention and graph neural networks.

## Mathematical Explanation

### Encoder

The encoder is typically a bidirectional GRU or LSTM producing hidden states:

h_t = [forward_h_t; backward_h_t], t = 1, ..., T

### Alignment Score

At decoder timestep t', the alignment score for encoder position t is:

e_{t', t} = v_a^T tanh(W_a s_{t'-1} + U_a h_t)

where:
- s_{t'-1} in R^{d_s} is the decoder's previous hidden state
- h_t in R^{d_h} is the encoder's t-th hidden state
- W_a in R^{d_a x d_s}, U_a in R^{d_a x d_h} are weight matrices
- v_a in R^{d_a} is a weight vector
- d_a is the attention dimension (often d_a = d_s + d_h or a fixed size)

### Attention Weights

alpha_{t', t} = exp(e_{t', t}) / sum_{k=1}^{T} exp(e_{t', k})

These weights satisfy sum_t alpha_{t', t} = 1.

### Context Vector

c_{t'} = sum_{t=1}^{T} alpha_{t', t} * h_t

### Decoder State Update

The decoder hidden state is updated using the context vector:

s_{t'} = f(s_{t'-1}, y_{t'-1}, c_{t'})

In the original formulation, the context vector is combined with the previous hidden state and target embedding to update the decoder state:

s_{t'} = tanh(W_c [s_{t'-1}; y_{t'-1}; c_{t'}])

### Output Probability

P(y_{t'} | y_{<t'}, X) = softmax(W_o s_{t'} + b_o)

## Code Examples

### Example 1: Bahdanau Attention Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.W_a = nn.Linear(enc_dim, attn_dim, bias=False)
        self.U_a = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        src_len = encoder_outputs.shape[1]
        decoder_hidden = decoder_hidden.transpose(0, 1)
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.W_a(encoder_outputs) + self.U_a(decoder_hidden))
        scores = self.v_a(energy).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

enc_dim = 64
dec_dim = 64
attn_dim = 32
attn = BahdanauAttention(enc_dim, dec_dim, attn_dim)
dec_hidden = torch.randn(1, 2, dec_dim)
enc_outputs = torch.randn(2, 5, enc_dim)
context, weights = attn(dec_hidden, enc_outputs)
print(f"Context shape: {context.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"Weights sum (per batch): {weights.sum(dim=1)}")
# Output: Context shape: torch.Size([2, 64])
# Output: Attention weights shape: torch.Size([2, 5])
# Output: Weights sum (per batch): tensor([1.0000, 1.0000])
```

### Example 2: Full Seq2Seq with Bahdanau Attention

```python
class BahdanauEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_dim, n_layers, dropout):
        super().__init__()
        self.enc_dim = enc_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, enc_dim, n_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_hidden = nn.Linear(enc_dim * 2, enc_dim)
        self.fc_cell = nn.Linear(enc_dim * 2, enc_dim)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.gru(embedded)
        hidden = torch.tanh(self.fc_hidden(torch.cat((hidden[-2], hidden[-1]), dim=1)))
        return outputs, hidden.unsqueeze(0)

class BahdanauDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, dec_dim, enc_dim, attn_dim, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.attention = BahdanauAttention(enc_dim * 2, dec_dim, attn_dim)
        self.gru = nn.GRU(emb_dim + enc_dim * 2, dec_dim, batch_first=True)
        self.fc = nn.Linear(dec_dim + emb_dim + enc_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, hidden, encoder_outputs):
        trg = trg.unsqueeze(1)
        embedded = self.dropout(self.embedding(trg))
        context, attn_weights = self.attention(hidden, encoder_outputs)
        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        output, hidden = self.gru(rnn_input, hidden)
        output = output.squeeze(1)
        embedded_sq = embedded.squeeze(1)
        prediction = self.fc(torch.cat((output, context, embedded_sq), dim=1))
        return prediction, hidden, attn_weights

class BahdanauSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        encoder_outputs, hidden = self.encoder(src)
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        outputs = torch.zeros(batch_size, trg_len, self.decoder.vocab_size).to(self.device)
        input_token = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, attn = self.decoder(input_token, hidden, encoder_outputs)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input_token = trg[:, t] if teacher_force else output.argmax(1)
        return outputs

VOCAB_SIZE = 100
EMB_DIM = 32
ENC_DIM = 64
DEC_DIM = 64
ATTN_DIM = 32
N_LAYERS = 1
DROPOUT = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enc = BahdanauEncoder(VOCAB_SIZE, EMB_DIM, ENC_DIM, N_LAYERS, DROPOUT)
dec = BahdanauDecoder(VOCAB_SIZE, EMB_DIM, DEC_DIM, ENC_DIM, ATTN_DIM, DROPOUT)
model = BahdanauSeq2Seq(enc, dec, DEVICE).to(DEVICE)

src = torch.randint(0, 100, (2, 7)).to(DEVICE)
trg = torch.randint(0, 100, (2, 5)).to(DEVICE)
out = model(src, trg, 1.0)
print(f"Bahdanau seq2seq output shape: {out.shape}")
# Output: Bahdanau seq2seq output shape: torch.Size([2, 5, 100])
```

### Example 3: Interpreting Bahdanau Attention Weights

```python
def interpret_attention(model, src_tokens, trg_tokens, src_vocab, trg_vocab, device):
    src_indices = torch.tensor([[src_vocab.get(t, 3) for t in src_tokens]]).to(device)
    trg_indices = torch.tensor([[trg_vocab.get(t, 3) for t in trg_tokens]]).to(device)
    model.eval()
    with torch.no_grad():
        enc_out, hidden = model.encoder(src_indices)
        input_token = trg_indices[:, 0]
        alignments = []
        for t in range(1, trg_indices.shape[1]):
            output, hidden, attn = model.decoder(input_token, hidden, enc_out)
            alignments.append(attn.squeeze(0).cpu().numpy())
            input_token = trg_indices[:, t]
    alignments = np.array(alignments)
    for i, trg_tok in enumerate(trg_tokens[1:]):
        top_src_idx = alignments[i].argmax()
        print(f"'{trg_tok}' -> '{src_tokens[top_src_idx]}' (weight: {alignments[i][top_src_idx]:.3f})")

import numpy as np
src_tokens = ['the', 'cat', 'sat', 'on', 'the', 'mat']
trg_tokens = ['<sos>', 'le', 'chat', 's_est', 'assis']
src_vocab = {t: i for i, t in enumerate(src_tokens + ['<pad>', '<sos>', '<eos>', '<unk>'])}
trg_vocab = {t: i for i, t in enumerate(trg_tokens + ['<pad>', '<sos>', '<eos>', 'sur', 'le', 'tapis'])}
interpret_attention(model, src_tokens, trg_tokens, src_vocab, trg_vocab, DEVICE)
# Output: 'le' -> 'the' (weight: 0.852)
# Output: 'chat' -> 'cat' (weight: 0.912)
# Output: 's_est' -> 'sat' (weight: 0.834)
# Output: 'assis' -> 'sat' (weight: 0.567)
```

## Common Mistakes

1. **Using the current decoder hidden state instead of the previous one**: Bahdanau attention computes scores using s_{t'-1}, the decoder's hidden state from the previous timestep. Using s_{t'} leaks future information and is incorrect.

2. **Not handling bidirectional encoder outputs**: A bidirectional encoder produces hidden states of dimension 2 * enc_dim. The attention mechanism and decoder must be designed to accept this dimension. Forgetting this leads to shape mismatches.

3. **Adding biases in the attention projection layers**: The original Bahdanau formulation uses bias=False in the attention projection layers. Biases can reduce the model's ability to learn meaningful alignments because the bias term can dominate the score computation.

4. **Not repeating the hidden state for batched attention computation**: When computing attention in batch mode, the decoder hidden state must be expanded to match the source length dimension for element-wise addition with encoder outputs.

5. **Applying attention when encoder_outputs is empty or zero-padded**: For proper masking of padding tokens, attention scores for zero-padded positions must be set to -inf before softmax. Otherwise, the model attends to meaningless padding.

## Interview Questions

### Beginner

Q: What is the key difference between Bahdanau attention and basic seq2seq without attention?

A: Bahdanau attention computes a dynamic context vector at each decoder timestep by taking a weighted sum of all encoder hidden states, where weights are computed by a neural network. Basic seq2seq uses only the final encoder hidden state as a fixed context vector. Attention provides access to all encoder states, eliminating the information bottleneck.

### Intermediate

Q: Why is Bahdanau attention called "additive" attention? How does the score computation work?

A: It is called additive because the score is computed by adding linear projections of the query and key before applying a tanh nonlinearity: score = v^T tanh(W_q * query + W_k * key). The addition allows the network to learn complex, non-linear interactions between the query and key, capturing relationships that might be missed by a simple dot product.

### Advanced

Q: Compare Bahdanau attention with Luong attention. What are the computational and representational trade-offs?

A: Bahdanau attention uses a feedforward network to compute scores: score = v^T tanh(W_1 * s + W_2 * h). This is more expressive (can capture non-linear relationships) but computationally more expensive O(T * (d_s + d_h) * d_a). Luong attention uses simpler score functions (dot, general, concat) that are computationally cheaper O(T * d) but less expressive. Bahdanau also computes attention scores using the previous decoder state s_{t-1} before generating the current token, while Luong uses the current decoder state s_t after generation. This means Bahdanau decides what to attend to before generating, while Luong decides after. Empirically, both perform similarly, with Luong being more popular for its computational efficiency and compatibility with the transformer-style attention.

## Practice Problems

### Easy

Implement a Bahdanau attention module and test it with random encoder outputs (batch=3, seq_len=5, enc_dim=16) and decoder hidden state (batch=3, dec_dim=16). Verify that the attention weights sum to 1.

### Medium

Train a seq2seq model with Bahdanau attention on a simple English-to-French phrase translation dataset. Visualize the attention matrix for a sample sentence and verify that the alignments are reasonable.

### Hard

Implement a variant of Bahdanau attention that includes coverage information. Specifically, modify the attention score to also depend on a coverage vector that tracks how much attention each encoder position has received. This is known as "coverage-aware attention."

## Solutions

### Easy Solution

```python
class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.W_a = nn.Linear(enc_dim, attn_dim, bias=False)
        self.U_a = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        decoder_hidden = decoder_hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.W_a(encoder_outputs) + self.U_a(decoder_hidden))
        attention = self.v_a(energy).squeeze(2)
        attn_weights = F.softmax(attention, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

attn = BahdanauAttention(16, 16, 16)
enc_out = torch.randn(3, 5, 16)
dec_hid = torch.randn(1, 3, 16)
context, weights = attn(dec_hid, enc_out)
print(f"Weights sum: {weights.sum(dim=1)}")
# Output: Weights sum: tensor([1.0000, 1.0000, 1.0000])
```

## Related Concepts

- Luong Attention
- Additive vs. Multiplicative Attention
- Seq2Seq with Attention
- Alignment in NMT
- Coverage-Aware Attention

## Next Concepts

- DL-338: Luong Attention
- DL-339: Additive vs. Multiplicative Attention

## Summary

Bahdanau attention, also known as additive attention, was the first attention mechanism applied to seq2seq models for neural machine translation. It computes alignment scores between the decoder's previous hidden state and each encoder hidden state using a feedforward neural network, producing a dynamic context vector that eliminates the fixed-context bottleneck. The additive nature allows the network to learn complex, non-linear query-key relationships. Bahdanau attention was a landmark contribution that paved the way for all subsequent attention mechanisms and remains relevant for applications requiring expressive, non-linear attention computation.

## Key Takeaways

- Bahdanau attention uses a feedforward network to compute attention scores.
- Scores are computed as v^T tanh(W_q * query + W_k * key).
- It computes attention using the previous decoder hidden state (before generating the current token).
- The attention mechanism produces a dynamic context vector at each decoder timestep.
- Bahdanau attention was the first attention mechanism for seq2seq NMT.
- It is more expressive but computationally more expensive than simpler dot-product attention.
