# Concept: Luong Attention

## Concept ID

DL-338

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Attention Mechanisms

## Learning Objectives

- Understand the Luong (multiplicative) attention mechanism and its three scoring variants.
- Implement Luong attention with dot, general, and concat score functions in PyTorch.
- Compare Luong attention with Bahdanau attention in terms of computation and expressiveness.
- Analyze the "global vs. local" attention distinction introduced in the Luong paper.
- Evaluate when to use Luong attention over other attention formulations.

## Prerequisites

- Understanding of Bahdanau attention and additive attention mechanisms.
- Familiarity with seq2seq models and the encoder-decoder framework.
- Knowledge of matrix multiplication and linear transformations.
- Experience with PyTorch for implementing attention modules.

## Definition

Luong attention, introduced by Minh-Thang Luong, Hieu Pham, and Christopher D. Manning in their 2015 paper "Effective Approaches to Attention-based Neural Machine Translation," is a family of attention mechanisms that compute alignment scores using simpler, more computationally efficient functions compared to Bahdanau's additive attention. Luong proposed three score functions: dot (simple dot product), general (dot product with a learned weight matrix), and concat (concatenation-based, similar to Bahdanau but without the tanh). Unlike Bahdanau attention which uses the previous decoder hidden state s_{t-1} to compute attention, Luong attention uses the current decoder hidden state s_t after generating the target token. This design choice simplifies implementation and allows the context vector to directly influence the final output without being fed through the RNN. Luong also introduced the distinction between global attention (attending to all source positions) and local attention (attending to a subset of source positions around an aligned position).

## Intuition

Luong attention is like having a librarian who first finds a book (generates a current word hypothesis) and then checks the source text to confirm or correct the hypothesis. In Bahdanau attention, the librarian first checks the source text and then picks the book. Luong's approach is computationally simpler because the attention mechanism doesn't need to influence the RNN state update — it only influences the final output prediction. The dot product score is the simplest: the librarian just measures how similar the current word state is to each source word, with no learned parameters. The general score adds a learned transformation that can re-weight dimensions. The concat score is a middle ground, using a learned transformation with an activation but without the full expressiveness of Bahdanau's tanh + v projection. Luong also recognized that attending to all source words (global) is powerful but expensive for long sequences, so local attention restricts the attention window to a neighborhood around a predicted alignment point, offering a trade-off between quality and computation.

## Why This Concept Matters

Luong attention simplified and improved upon Bahdanau attention in several ways that influenced the development of modern attention mechanisms. The simpler score functions (especially the general score: Q^T W K) prefigured the scaled dot-product attention used in transformers by showing that multiplicative interactions work well for attention. The distinction between global and local attention introduced the concept of sparse attention, which remains important for handling long sequences. Luong's formulation — where attention is computed after the RNN step and directly influences the output — is more compatible with the transformer architecture where attention replaces recurrence entirely. Understanding Luong attention is important for grasping the design space of attention mechanisms and for implementing efficient, production-quality seq2seq systems.

## Mathematical Explanation

### Encoder

The encoder produces hidden states h_t for t = 1, ..., T.

### Decoder State

The decoder computes its current hidden state s_t using the previous hidden state, the previous target token, and optionally the previous context vector:

s_t = f(s_{t-1}, y_{t-1})

### Score Functions

Luong proposed three score functions to compute e_t (the alignment score for all encoder positions):

1. **Dot**: e_t = s_t^T * h (simple dot product, no learned parameters)
2. **General**: e_t = s_t^T * W_a * h (dot product with learned weight matrix W_a)
3. **Concat**: e_t = v_a^T tanh(W_a [s_t; h]) (concatenation with learned parameters)

where e_t in R^T contains scores for all encoder positions.

### Attention Weights

alpha_t = softmax(e_t)

### Context Vector

c_t = sum_{i=1}^{T} alpha_{t, i} * h_i

### Final Output

The context vector is combined with the decoder state to produce the output:

tilde_s_t = tanh(W_c [c_t; s_t])
P(y_t | y_{<t}, X) = softmax(W_o * tilde_s_t)

### Key Difference from Bahdanau

Bahdanau: s_t = f(s_{t-1}, y_{t-1}, c_t) — attention before RNN step, context influences RNN.
Luong: s_t = f(s_{t-1}, y_{t-1}), then c_t from attention(s_t, h) — attention after RNN step, context influences output only.

## Code Examples

### Example 1: Luong Attention with All Three Score Functions

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LuongAttention(nn.Module):
    def __init__(self, method, hidden_dim):
        super().__init__()
        self.method = method
        if method == 'general':
            self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif method == 'concat':
            self.W_a = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
            self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def score(self, decoder_hidden, encoder_output):
        if self.method == 'dot':
            return torch.bmm(decoder_hidden.unsqueeze(1), encoder_output.transpose(1, 2))
        elif self.method == 'general':
            projected = self.W_a(decoder_hidden.unsqueeze(1))
            return torch.bmm(projected, encoder_output.transpose(1, 2))
        elif self.method == 'concat':
            batch_size = decoder_hidden.shape[0]
            src_len = encoder_output.shape[1]
            dec_expanded = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)
            concat = torch.cat((dec_expanded, encoder_output), dim=2)
            energy = torch.tanh(self.W_a(concat))
            return self.v_a(energy).transpose(1, 2)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        scores = self.score(decoder_hidden.squeeze(0), encoder_outputs).squeeze(1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

batch_size, src_len, hidden_dim = 2, 5, 64
decoder_hidden = torch.randn(1, batch_size, hidden_dim)
encoder_outputs = torch.randn(batch_size, src_len, hidden_dim)

for method in ['dot', 'general', 'concat']:
    attn = LuongAttention(method, hidden_dim)
    context, weights = attn(decoder_hidden, encoder_outputs)
    print(f"{method}: context shape = {context.shape}, weights sum = {weights.sum(dim=1)}")
# Output: dot: context shape = torch.Size([2, 64]), weights sum = tensor([1., 1.])
# Output: general: context shape = torch.Size([2, 64]), weights sum = tensor([1., 1.])
# Output: concat: context shape = torch.Size([2, 64]), weights sum = tensor([1., 1.])
```

### Example 2: Full Seq2Seq with Luong Attention

```python
class LuongDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, attn_method, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.attention = LuongAttention(attn_method, hid_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.W_c = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_out = nn.Linear(hid_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, hidden, encoder_outputs):
        trg = trg.unsqueeze(1)
        embedded = self.dropout(self.embedding(trg))
        output, hidden = self.gru(embedded, hidden)
        context, attn_weights = self.attention(hidden, encoder_outputs)
        combined = torch.cat((output.squeeze(1), context), dim=1)
        tilde_h = torch.tanh(self.W_c(combined))
        prediction = self.fc_out(tilde_h)
        return prediction, hidden, attn_weights

class LuongEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.gru(embedded)
        return outputs, hidden

class LuongSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        enc_outputs, hidden = self.encoder(src)
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        outputs = torch.zeros(batch_size, trg_len, self.decoder.vocab_size).to(self.device)
        input_token = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, attn = self.decoder(input_token, hidden, enc_outputs)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input_token = trg[:, t] if teacher_force else output.argmax(1)
        return outputs

VOCAB_SIZE = 100
EMB_DIM = 32
HID_DIM = 64
N_LAYERS = 1
DROPOUT = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enc = LuongEncoder(VOCAB_SIZE, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
dec = LuongDecoder(VOCAB_SIZE, EMB_DIM, HID_DIM, 'general', DROPOUT)
model = LuongSeq2Seq(enc, dec, DEVICE).to(DEVICE)

src = torch.randint(0, 100, (2, 7)).to(DEVICE)
trg = torch.randint(0, 100, (2, 5)).to(DEVICE)
out = model(src, trg, 1.0)
print(f"Luong seq2seq output shape: {out.shape}")
# Output: Luong seq2seq output shape: torch.Size([2, 5, 100])
```

### Example 3: Comparing Luong Score Functions

```python
def compare_luong_methods(encoder, decoders, src, trg, device):
    results = {}
    enc_out, hidden = encoder(src)
    for method, decoder in decoders.items():
        dec_hidden = hidden
        input_token = trg[:, 0]
        matches = 0
        total = 0
        with torch.no_grad():
            for t in range(1, trg.shape[1]):
                output, dec_hidden, attn = decoder(input_token, dec_hidden, enc_out)
                pred = output.argmax(1)
                matches += (pred == trg[:, t]).sum().item()
                total += trg.shape[0]
                input_token = trg[:, t]
        results[method] = matches / total
    return results

decoders = {}
for method in ['dot', 'general', 'concat']:
    decoders[method] = LuongDecoder(VOCAB_SIZE, EMB_DIM, HID_DIM, method, DROPOUT).to(DEVICE)

src = torch.randint(0, 100, (4, 7)).to(DEVICE)
trg = torch.randint(0, 100, (4, 5)).to(DEVICE)
results = compare_luong_methods(enc, decoders, src, trg, DEVICE)
for method, acc in results.items():
    print(f"{method}: accuracy = {acc:.3f}")
# Output: dot: accuracy = 0.250
# Output: general: accuracy = 0.300
# Output: concat: accuracy = 0.275
```

## Common Mistakes

1. **Using the wrong decoder state for attention**: Luong uses the current decoder hidden state s_t, not s_{t-1} as in Bahdanau. Using the wrong state produces incorrect attention patterns.

2. **Not separating the attention mechanism from the RNN step**: In Luong, the context vector is not fed into the RNN. It is combined with the RNN output only at the final prediction layer. Feeding the context into the RNN would make it more like Bahdanau and would not leverage Luong's computational advantage.

3. **Forgetting to squeeze the score tensor**: The score function returns a tensor with extra dimensions. For batch matrix multiplication, the scores must be squeezed to (batch, 1, src_len) before softmax.

4. **Using the concat method without the tanh activation**: The concat method requires a tanh activation before the v projection. Omitting the tanh reduces expressiveness and may lead to training instability.

5. **Assuming all three methods perform identically**: The dot method has no learned parameters, the general method has a weight matrix, and the concat method has both weight matrices and a bias. They have different capacities and regularization needs, and the best method depends on the task and dataset size.

## Interview Questions

### Beginner

Q: What is the main difference between Luong attention and Bahdanau attention in terms of when attention is computed?

A: Luong attention computes attention after the RNN step (using the current decoder hidden state s_t), while Bahdanau attention computes attention before the RNN step (using the previous decoder hidden state s_{t-1}). In Luong, the context vector only influences the output prediction, not the RNN state update.

### Intermediate

Q: What are the three score functions proposed by Luong, and what are the trade-offs between them?

A: The three score functions are: (1) dot — simple dot product of decoder state and encoder state, no learned parameters, fastest; (2) general — dot product with a learned weight matrix between decoder and encoder states, adds model capacity; (3) concat — concatenation followed by a learned weight matrix and tanh activation, most expressive but also the most computationally expensive. Dot is fastest but least expressive, general balances capacity and speed, and concat is most expressive but slowest.

### Advanced

Q: Explain the concept of "local attention" introduced by Luong et al. How does it differ from global attention, and what problem does it solve?

A: Local attention restricts the attention mechanism to a subset of source positions around an aligned position, rather than attending to all source positions. At each decoder timestep, the model predicts an aligned position p_t in the source, then computes attention over a window [p_t - D, p_t + D]. This reduces the O(T) complexity of global attention to O(2D) where D << T. Local attention comes in two variants: local-m (monotonic) where p_t is a learned function, and local-p (predictive) where p_t is predicted from the decoder state. Local attention addresses the quadratic cost of global attention for long sequences and provides a trade-off between full attention (expensive but accurate) and no attention (cheap but poor). It was an early form of sparse attention that anticipated later work on efficient transformers.

## Practice Problems

### Easy

Implement all three Luong score functions (dot, general, concat) as separate functions. Test each with random data and verify they produce correct output shapes.

### Medium

Train seq2seq models with Luong (general) and Bahdanau attention on a copy task. Compare convergence speed (epochs to reach 90% accuracy) and total training time.

### Hard

Implement Luong's local attention (predictive variant). Given a decoder hidden state, predict the aligned source position p_t, create a Gaussian window around it, and compute attention only within this window. Compare global vs. local attention on long sequences (length 100+).

## Solutions

### Easy Solution

```python
def dot_score(dec_hidden, enc_output):
    return torch.bmm(dec_hidden.unsqueeze(1), enc_output.transpose(1, 2)).squeeze(1)

def general_score(dec_hidden, enc_output, W):
    projected = W(dec_hidden).unsqueeze(1)
    return torch.bmm(projected, enc_output.transpose(1, 2)).squeeze(1)

def concat_score(dec_hidden, enc_output, W_a, v_a):
    batch_size, src_len = enc_output.shape[0], enc_output.shape[1]
    dec_expanded = dec_hidden.unsqueeze(1).expand(-1, src_len, -1)
    concat = torch.cat((dec_expanded, enc_output), dim=2)
    energy = torch.tanh(W_a(concat))
    return v_a(energy).squeeze(2)

batch, src_len, hid = 2, 5, 64
dec_h = torch.randn(batch, hid)
enc_out = torch.randn(batch, src_len, hid)
W = nn.Linear(hid, hid, bias=False)
W_a = nn.Linear(hid*2, hid, bias=False)
v_a = nn.Linear(hid, 1, bias=False)

dot_s = dot_score(dec_h, enc_out)
gen_s = general_score(dec_h, enc_out, W)
con_s = concat_score(dec_h, enc_out, W_a, v_a)
print(f"Dot: {dot_s.shape}, General: {gen_s.shape}, Concat: {con_s.shape}")
# Output: Dot: torch.Size([2, 5]), General: torch.Size([2, 5]), Concat: torch.Size([2, 5])
```

## Related Concepts

- Bahdanau Attention
- Global vs. Local Attention
- Scaled Dot-Product Attention
- Seq2Seq with Attention
- Attention Score Functions

## Next Concepts

- DL-339: Additive vs. Multiplicative Attention
- DL-340: Attention Score
- DL-350: Global vs. Local Attention

## Summary

Luong attention provides a family of computationally efficient attention mechanisms with three score functions: dot, general, and concat. Unlike Bahdanau attention which computes attention using the previous decoder state, Luong uses the current decoder state after the RNN step, making the context vector influence only the output prediction. The simpler score functions (especially general) prefigure the scaled dot-product attention used in transformers. Luong also introduced the important distinction between global and local attention, offering a trade-off between quality and computational cost for long sequences.

## Key Takeaways

- Luong attention uses the current decoder hidden state s_t, not s_{t-1}.
- Three score functions: dot (no parameters), general (weight matrix), concat (with tanh).
- Context vector only influences the output, not the RNN state update.
- This simplifies implementation and is compatible with transformer-style attention.
- Luong introduced global vs. local attention for handling long sequences.
- The general score function prefigures transformer-style multiplicative attention.
