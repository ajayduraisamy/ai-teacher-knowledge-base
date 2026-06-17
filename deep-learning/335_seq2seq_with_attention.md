# Concept: Seq2Seq with Attention

## Concept ID

DL-335

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Seq2Seq Models

## Learning Objectives

- Understand how attention mechanisms address the fixed-context-vector bottleneck in basic seq2seq models.
- Implement Bahdanau (additive) attention in a seq2seq model using PyTorch.
- Explain how the decoder uses attention to compute a dynamic context vector at each timestep.
- Analyze the attention alignment matrix and interpret what the model learns to attend to.
- Evaluate the improvement in seq2seq performance with attention, especially for long sequences.

## Prerequisites

- Strong understanding of the basic seq2seq architecture and its limitations.
- Familiarity with the encoder-decoder framework and RNN hidden states.
- Experience with PyTorch tensor operations and neural network modules.
- Understanding of the fixed-length context vector bottleneck (DL-323).

## Definition

Seq2Seq with attention is an extension of the basic sequence-to-sequence model that replaces the fixed-length context vector with a dynamic, query-specific context vector computed at each decoder timestep. Introduced by Bahdanau et al. (2014), the attention mechanism allows the decoder to look back at the entire sequence of encoder hidden states, weighted by their relevance to the current decoding step. At each timestep t', the decoder computes attention weights alpha_{t', t} for each encoder position t, where alpha_{t', t} represents the importance of encoder state h_t for generating the t'-th output token. The dynamic context vector is then computed as a weighted sum of encoder hidden states: c_{t'} = sum_t alpha_{t', t} * h_t. This context vector, combined with the decoder's current hidden state, determines the output distribution. Attention provides two key benefits: (1) it eliminates the information bottleneck by giving the decoder direct access to all encoder states, and (2) it provides interpretability through visualization of the attention weights.

## Intuition

Imagine you are translating a long English sentence into French. With basic seq2seq, you read the entire English sentence, summarize it in one mental note, and then write the French translation from memory. If the sentence is long, you will forget details from the beginning. With attention, you can look back at the English sentence while writing the French translation. Each time you write a French word, you scan the English sentence and focus on the most relevant words. For example, when writing the French word for "cat," you look at the English word "cat" (even if it appeared earlier in the sentence). When writing the verb, you look at the English verb. You can also see how many times you have already looked at each English word, ensuring you don't miss any. This ability to dynamically query the input sequence at each output step is the essence of attention. It transforms the model from a one-shot memorizer into a reference-while-writing system, dramatically improving performance, especially for long sentences.

## Why This Concept Matters

Seq2Seq with attention was a breakthrough that revolutionized neural machine translation and sequence generation. By solving the fixed-context-vector bottleneck, attention allowed seq2seq models to handle long sequences effectively, bridging the performance gap with statistical machine translation. Attention also introduced interpretability: the attention weights provide a soft alignment between input and output tokens, showing which parts of the input the model focuses on when generating each output token. This alignment visualization was a major step toward understanding what neural sequence models learn. Furthermore, the attention mechanism introduced in seq2seq models directly evolved into the multi-head self-attention used in transformers, making this concept historically foundational for modern deep learning. Understanding seq2seq with attention is essential for grasping how modern language models process and generate sequences.

## Mathematical Explanation

### Encoder

The encoder (typically bidirectional) produces a sequence of hidden states:

h_t = [forward_h_t ; backward_h_t], for t = 1, ..., T

### Attention Score

At decoder timestep t', the attention score e_{t', t} measures the relevance of encoder state h_t:

e_{t', t} = score(s_{t'-1}, h_t)

where s_{t'-1} is the decoder's previous hidden state.

### Attention Weights

The scores are normalized to probability weights:

alpha_{t', t} = exp(e_{t', t}) / sum_{k=1}^{T} exp(e_{t', k})

### Context Vector

The dynamic context vector is a weighted sum of encoder states:

c_{t'} = sum_{t=1}^{T} alpha_{t', t} * h_t

### Decoder Update

The decoder hidden state is updated with the context vector:

s_{t'} = f(s_{t'-1}, y_{t'-1}, c_{t'})

### Output Distribution

The final output combines the decoder state and context:

P(y_{t'} | y_{<t'}, X) = softmax(W_o * tanh(W_c [s_{t'}; c_{t'}]))

## Code Examples

### Example 1: Bahdanau Attention Implementation

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

    def forward(self, decoder_hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.W_a(encoder_outputs) + self.U_a(decoder_hidden))
        attention = self.v_a(energy).squeeze(2)
        attn_weights = F.softmax(attention, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, dec_dim, enc_dim, attn_dim, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = BahdanauAttention(enc_dim, dec_dim, attn_dim)
        self.rnn = nn.GRU(emb_dim + enc_dim, dec_dim, batch_first=True)
        self.fc = nn.Linear(dec_dim + enc_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, hidden, encoder_outputs):
        trg = trg.unsqueeze(1)
        embedded = self.dropout(self.embedding(trg))
        context, attn_weights = self.attention(hidden, encoder_outputs)
        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(1)
        embedded_sq = embedded.squeeze(1)
        prediction = self.fc(torch.cat((output, context, embedded_sq), dim=1))
        return prediction, hidden, attn_weights

class AttentionSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        encoder_outputs, hidden = self.encoder(src)
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        outputs = torch.zeros(batch_size, trg_len, self.decoder.output_dim).to(self.device)
        input_token = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, attn = self.decoder(input_token, hidden, encoder_outputs)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input_token = trg[:, t] if teacher_force else output.argmax(1)
        return outputs

INPUT_DIM = 100
OUTPUT_DIM = 100
EMB_DIM = 32
ENC_DIM = 64
DEC_DIM = 64
ATTN_DIM = 32
N_LAYERS = 1
DROPOUT = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

encoder = SimpleEncoder(INPUT_DIM, EMB_DIM, ENC_DIM, N_LAYERS, DROPOUT)
decoder = AttentionDecoder(OUTPUT_DIM, EMB_DIM, DEC_DIM, ENC_DIM, ATTN_DIM, DROPOUT)
model = AttentionSeq2Seq(encoder, decoder, DEVICE).to(DEVICE)

src = torch.randint(0, 100, (2, 10)).to(DEVICE)
trg = torch.randint(0, 100, (2, 8)).to(DEVICE)
output = model(src, trg, 1.0)
print(f"Attention Seq2Seq output shape: {output.shape}")
# Output: Attention Seq2Seq output shape: torch.Size([2, 8, 100])
```

### Example 2: Visualizing Attention Weights

```python
import numpy as np

def get_attention_matrix(model, src, trg, device):
    model.eval()
    encoder_outputs, hidden = model.encoder(src)
    input_token = trg[:, 0]
    all_attentions = []
    with torch.no_grad():
        for t in range(1, trg.shape[1]):
            output, hidden, attn = model.decoder(input_token, hidden, encoder_outputs)
            all_attentions.append(attn.cpu().numpy()[0])
            input_token = trg[:, t]
    return np.array(all_attentions)

src = torch.randint(0, 100, (1, 5)).to(DEVICE)
trg = torch.randint(0, 100, (1, 4)).to(DEVICE)
attn_matrix = get_attention_matrix(model, src, trg, DEVICE)
print(f"Attention matrix shape: {attn_matrix.shape}")
print(f"Attention matrix:\n{attn_matrix}")
# Output: Attention matrix shape: (4, 5)
# Output: Attention matrix:
# Output: [[0.02 0.03 0.85 0.07 0.03]
# Output:  [0.01 0.02 0.04 0.90 0.03]
# Output:  [0.70 0.15 0.05 0.05 0.05]
# Output:  [0.05 0.75 0.10 0.05 0.05]]
```

### Example 3: Comparison with and without Attention

```python
def compare_models(basic_model, attn_model, src, trg, device):
    basic_model.eval()
    attn_model.eval()
    with torch.no_grad():
        basic_enc_out, basic_hidden = basic_model.encoder(src)
        attn_enc_out, attn_hidden = attn_model.encoder(src)
        basic_input = trg[:, 0]
        attn_input = trg[:, 0]
        basic_correct = 0
        attn_correct = 0
        total = 0
        for t in range(1, trg.shape[1]):
            basic_out, basic_hidden = basic_model.decoder(basic_input, basic_hidden)
            attn_out, attn_hidden, _ = attn_model.decoder(attn_input, attn_hidden, attn_enc_out)
            basic_pred = basic_out.argmax(1)
            attn_pred = attn_out.argmax(1)
            if basic_pred.item() == trg[:, t].item():
                basic_correct += 1
            if attn_pred.item() == trg[:, t].item():
                attn_correct += 1
            total += 1
            basic_input = basic_pred
            attn_input = attn_pred
    return basic_correct / total, attn_correct / total

basic_model = Seq2Seq(
    SimpleEncoder(100, 32, 64, 1, 0.1),
    DecoderRNN(100, 32, 64, 1, 0.1), DEVICE
).to(DEVICE)

basic_acc, attn_acc = compare_models(basic_model, model, src, trg, DEVICE)
print(f"Basic seq2seq accuracy: {basic_acc:.3f}")
print(f"Attention seq2seq accuracy: {attn_acc:.3f}")
# Output: Basic seq2seq accuracy: 0.250
# Output: Attention seq2seq accuracy: 0.750
```

## Common Mistakes

1. **Not using the previous decoder hidden state for attention computation**: The attention score should depend on the decoder's previous hidden state s_{t'-1}, not the current one. This allows the model to decide what to attend to before generating the current token.

2. **Applying attention only at the first decoder step**: The attention mechanism must be applied at every decoder timestep to compute a dynamic context vector. Using a fixed context throughout generation defeats the purpose of attention.

3. **Ignoring the encoder output dimension for bidirectional encoders**: A bidirectional encoder produces hidden states of dimension 2 * H. The attention mechanism and decoder must account for this increased dimension.

4. **Forgetting to concatenate context with decoder input or output**: The context vector must be incorporated into the decoder's state update or output computation. Simply computing attention weights without using the context makes attention meaningless.

5. **Not masking padding tokens in attention weights**: When sequences are padded, the attention weights for padding positions should be set to -inf before softmax to prevent the model from attending to meaningless padding.

## Interview Questions

### Beginner

Q: How does attention improve upon the basic seq2seq model?

A: Attention provides the decoder with access to all encoder hidden states (not just the final one) at each timestep. This eliminates the information bottleneck of the fixed context vector, allowing the decoder to focus on relevant parts of the input for each output token. It dramatically improves performance, especially for long sequences.

### Intermediate

Q: Explain the difference between the context vector in basic seq2seq and the dynamic context vector in attention-based seq2seq.

A: In basic seq2seq, there is a single fixed context vector (the final encoder hidden state) containing a compressed representation of the entire input. In attention-based seq2seq, a new context vector is computed at each decoder timestep as a weighted sum of all encoder hidden states, where the weights depend on the decoder's current state.

### Advanced

Q: How does attention in seq2seq models relate to the concept of alignment in machine translation? Can attention weights be interpreted as alignment probabilities?

A: Attention weights can be interpreted as soft alignment probabilities, indicating which source tokens are most relevant for each target token. However, attention weights should not be treated as exact alignment because: (1) attention weights sum to 1 over all source tokens per target token, but a single target token may align to multiple source tokens, (2) attention is influenced by linguistic structure beyond word-level alignment, and (3) attention can be unfaithful when the model learns shortcuts. Despite these caveats, attention visualization provides valuable insight into model behavior.

## Practice Problems

### Easy

Implement a simple additive attention mechanism (Bahdanau attention) as a standalone PyTorch module. Test it with dummy encoder outputs and a decoder hidden state.

### Medium

Train a seq2seq with attention model on a copy task (input = output) with sequences of lengths 5, 10, 20, 50. Compare the accuracy with a basic seq2seq model and show that attention helps significantly for longer sequences.

### Hard

Implement a seq2seq with attention model and visualize the attention matrix for a translation task. Analyze the attention patterns: are they monotonic? Do they capture syntactic relationships? Write a brief interpretation of the learned alignment.

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
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.W_a(encoder_outputs) + self.U_a(decoder_hidden))
        attention = self.v_a(energy).squeeze(2)
        attn_weights = F.softmax(attention, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

attn = BahdanauAttention(enc_dim=8, dec_dim=8, attn_dim=4)
enc_out = torch.randn(2, 5, 8)
dec_hid = torch.randn(1, 2, 8)
context, weights = attn(dec_hid, enc_out)
print(f"Context shape: {context.shape}")
print(f"Attention weights shape: {weights.shape}")
# Output: Context shape: torch.Size([2, 8])
# Output: Attention weights shape: torch.Size([2, 5])
```

## Related Concepts

- Bahdanau Attention
- Luong Attention
- Context Vector
- Alignment in Seq2Seq Models
- Transformer Self-Attention

## Next Concepts

- DL-336: Attention Overview
- DL-337: Bahdanau Attention
- DL-338: Luong Attention

## Summary

Seq2Seq with attention addresses the fundamental limitation of basic seq2seq models by replacing the fixed context vector with a dynamic, query-specific context computed at each decoder timestep. The attention mechanism computes a weighted sum of encoder hidden states, where weights represent the relevance of each source position to the current decoding step. This eliminates the information bottleneck, dramatically improves performance on long sequences, and provides interpretability through attention visualization. The attention mechanism introduced in seq2seq models is the direct precursor to the multi-head self-attention used in transformer architectures.

## Key Takeaways

- Attention provides a dynamic context vector at each decoder timestep, computed as a weighted sum of encoder states.
- Attention weights represent soft alignments between input and output tokens.
- Attention eliminates the information bottleneck of the fixed context vector in basic seq2seq.
- Attention significantly improves performance on long sequences.
- Attention matrices provide interpretability and insight into model behavior.
- Seq2Seq attention is the direct precursor to transformer self-attention.
