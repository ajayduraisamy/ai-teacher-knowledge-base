# Concept: Attention in NLP

## Concept ID

DL-355

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Attention Mechanisms

## Learning Objectives

- Understand the central role of attention mechanisms in modern natural language processing.
- Differentiate between encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5) attention patterns.
- Analyze how attention enables key NLP capabilities: contextualization, coreference resolution, and long-range dependency modeling.
- Implement key attention-based NLP architectures in PyTorch.
- Evaluate the impact of attention on NLP task performance across different model families.

## Prerequisites

- Solid understanding of self-attention, cross-attention, and multi-head attention.
- Familiarity with the transformer architecture.
- Knowledge of masked language modeling and autoregressive language modeling.
- Experience with PyTorch for sequence modeling.

## Definition

Attention in NLP refers to the application of attention mechanisms to natural language processing tasks. Since the introduction of the Transformer, attention has become the dominant paradigm in NLP, replacing recurrent and convolutional architectures. Attention in NLP can be categorized by architecture type: (1) Encoder-only models (BERT, RoBERTa, ELECTRA) use bidirectional self-attention for representation learning. (2) Decoder-only models (GPT, LLaMA, Claude) use causal self-attention for autoregressive generation. (3) Encoder-decoder models (T5, BART) use both bidirectional encoder attention and causal decoder attention with cross-attention. Attention enables NLP models to capture contextual word meanings, resolve coreferences, handle long-range dependencies, and generate coherent text. The specific masking patterns (bidirectional, causal, prefix) define how information flows through the model and determine what tasks the model can perform.

## Intuition

In NLP, attention allows words to understand their context. When you read the word "bank" in "river bank" vs. "savings bank," you automatically use surrounding words to disambiguate the meaning. Attention does the same: each word representation is a weighted combination of all other word representations, with weights depending on relevance. In BERT-style models, bidirectional attention means every word sees all other words — like reading the whole sentence before understanding any word. In GPT-style models, causal attention means each word only sees words to its left — like reading left to right, predicting each next word. The differences seem subtle but have huge implications: bidirectional models are better at understanding (classification, QA), while causal models are better at generation (text completion, dialogue).

## Why This Concept Matters

Attention is the foundation of virtually all modern NLP. Every major NLP system released in the past 5+ years is based on transformer attention. Understanding how attention is applied in NLP is essential for: (1) Choosing the right model architecture for a task (encoder-only for understanding, decoder-only for generation, encoder-decoder for translation/summarization). (2) Understanding model capabilities and limitations (bidirectional vs. causal context affects what the model can do). (3) Implementing and fine-tuning NLP models effectively. (4) Designing new NLP architectures and attention variants. Since attention-based NLP models are now used in production by virtually every tech company, this knowledge is practically essential.

## Mathematical Explanation

### Encoder-Only (BERT-style)

Input tokens x = (x_1, ..., x_T). Add [CLS] at position 0 and [SEP] at appropriate positions.

For each encoder layer:

z_l = EncoderLayer(z_{l-1})

where EncoderLayer uses bidirectional self-attention:

Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + M_bidirectional) V

M_bidirectional has no restrictions (all positions attend to all positions). Padding tokens are masked.

Output: z_L[0] for classification, all z_L for token-level tasks.

### Decoder-Only (GPT-style)

Input tokens x = (x_1, ..., x_T).

For each decoder layer:

z_l = DecoderLayer(z_{l-1})

where DecoderLayer uses causal self-attention:

Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + M_causal) V

M_causal is upper-triangular (-inf for future positions).

Generation: P(x_{t+1} | x_{<=t}) = softmax(W * z_L[t])

### Encoder-Decoder (T5-style)

Encoder: bidirectional self-attention on input.
Decoder: causal self-attention on target + cross-attention to encoder.

CrossAttention(Q_dec, K_enc, V_enc) = softmax(Q_dec K_enc^T / sqrt(d_k)) V_enc

### Key Attention Patterns in NLP

| Model | Encoder | Decoder | Best For |
|-------|---------|---------|----------|
| BERT | Bidirectional self-attn | N/A | Classification, QA, NER |
| GPT | N/A | Causal self-attn | Generation, dialogue |
| T5 | Bidirectional self-attn | Causal + cross-attn | Translation, summarization |

## Code Examples

### Example 1: BERT-style Encoder for Classification

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BertEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class BertClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=32, n_heads=4, d_ff=128, n_layers=4, num_classes=2, max_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        self.layers = nn.ModuleList([BertEncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask=None):
        seq_len = input_ids.shape[1]
        x = self.embedding(input_ids) + self.pos_embedding[:, :seq_len]
        padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        for layer in self.layers:
            x = layer(x, padding_mask)
        x = self.norm(x)
        cls_output = x[:, 0]
        return self.classifier(cls_output)

model = BertClassifier(vocab_size=100, d_model=32, n_heads=4, d_ff=128, n_layers=2, num_classes=2)
input_ids = torch.randint(0, 100, (2, 10))
attention_mask = torch.ones(2, 10)
output = model(input_ids, attention_mask)
print(f"BERT classifier output: {output.shape}")
# Output: BERT classifier output: torch.Size([2, 2])
```

### Example 2: GPT-style Decoder for Text Generation

```python
class GPTDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, seq = x.shape[0], x.shape[1]
        Q = self.W_q(x).view(batch, seq, -1, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, seq, -1, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, seq, -1, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        causal = torch.triu(torch.ones(seq, seq, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(batch, seq, -1)
        x = self.norm1(x + self.dropout(self.W_o(context)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model=32, n_heads=4, d_ff=128, n_layers=4, max_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        self.layers = nn.ModuleList([GPTDecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        x = self.embedding(input_ids) + self.pos_embedding[:, :seq_len]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

    def generate(self, input_ids, max_new_tokens=10, temperature=0.8):
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)[:, -1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

gpt = GPTModel(vocab_size=100)
input_ids = torch.randint(0, 100, (1, 5))
logits = gpt(input_ids)
print(f"GPT logits shape: {logits.shape}")
generated = gpt.generate(input_ids, max_new_tokens=5)
print(f"Generated sequence length: {generated.shape[1]}")
# Output: GPT logits shape: torch.Size([1, 5, 100])
# Output: Generated sequence length: 10
```

### Example 3: T5-style Encoder-Decoder for Translation

```python
class T5Model(nn.Module):
    def __init__(self, vocab_size, d_model=32, n_heads=4, d_ff=128, n_layers=3, max_len=128):
        super().__init__()
        self.encoder_embed = nn.Embedding(vocab_size, d_model)
        self.decoder_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model))
        self.encoder_layers = nn.ModuleList([BertEncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.decoder_layers = nn.ModuleList([GPTDecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.enc_norm = nn.LayerNorm(d_model)
        self.dec_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def encode(self, input_ids):
        seq_len = input_ids.shape[1]
        x = self.encoder_embed(input_ids) + self.pos_embed[:, :seq_len]
        for layer in self.encoder_layers:
            x = layer(x)
        return self.enc_norm(x)

    def decode(self, decoder_ids, encoder_output):
        seq_len = decoder_ids.shape[1]
        x = self.decoder_embed(decoder_ids) + self.pos_embed[:, :seq_len]
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.dec_norm(x)
        return self.lm_head(x)

    def forward(self, src, trg):
        enc_out = self.encode(src)
        return self.decode(trg, enc_out)

t5 = T5Model(vocab_size=100)
src = torch.randint(0, 100, (2, 8))
trg = torch.randint(0, 100, (2, 6))
output = t5(src, trg)
print(f"T5 output shape: {output.shape}")
# Output: T5 output shape: torch.Size([2, 6, 100])
```

## Common Mistakes

1. **Using bidirectional attention for language modeling**: Bidirectional attention allows each token to see future tokens. For language modeling (predicting next token), this is cheating — the model would simply copy the future token. Causal masking is required for autoregressive language models.

2. **Using causal attention for text classification**: Causal attention only allows each token to see previous tokens. For classification tasks like sentiment analysis, bidirectional attention is better because the model needs full context.

3. **Not adding task-specific heads to pre-trained models**: Pre-trained transformers produce contextualized token representations. For sequence classification, adding a [CLS] head or pooling layer is necessary. For token classification, using all token outputs is needed.

4. **Ignoring the difference in attention patterns between encoder and decoder in seq2seq models**: The encoder uses bidirectional attention (full context); the decoder uses causal self-attention + cross-attention. Architectures must be designed accordingly.

5. **Overlooking prompt engineering for decoder-only models**: Decoder-only models (GPT-style) process everything through causal attention. The way prompts are structured significantly affects attention patterns and generation quality.

## Interview Questions

### Beginner

Q: What is the key difference between BERT-style and GPT-style attention?

A: BERT uses bidirectional self-attention where every token can attend to all other tokens. This is good for understanding tasks (classification, QA). GPT uses causal (masked) self-attention where each token only attends to previous tokens. This is good for generation tasks (text completion, dialogue).

### Intermediate

Q: How does the attention pattern in encoder-decoder models (T5, BART) differ from encoder-only and decoder-only models?

A: Encoder-decoder models have two distinct attention stages: (1) The encoder uses bidirectional self-attention (like BERT) on the input sequence. (2) The decoder uses causal self-attention (like GPT) on the target sequence, plus cross-attention that allows the decoder to attend to the encoder outputs. This combines the understanding capability of bidirectional models with the generation capability of causal models.

### Advanced

Q: The T5 model uses prefix-LM (also called causal LM with prefix) attention during pre-training. Explain this attention pattern and its advantages over strict encoder-decoder or decoder-only architectures.

A: Prefix-LM attention divides the sequence into a prefix (first k tokens) and a suffix (remaining tokens). The prefix uses bidirectional attention among itself, while the suffix uses causal attention (each token attends to all prefix tokens and previous suffix tokens). Advantages: (1) It unifies the encoder and decoder into a single model, reducing parameters. (2) It handles both understanding tasks (use the prefix bidirectionally) and generation tasks (generate the suffix causally). (3) It enables efficient span corruption pre-training where corrupted spans are predicted from their bidirectional context. This is more parameter-efficient than separate encoder-decoder models while maintaining comparable performance.

## Practice Problems

### Easy

Train a BERT-style classifier on a simple sentiment analysis dataset. Compare the performance with a GPT-style classifier that uses only the last token.

### Medium

Implement a prefix-LM attention mask and compare it with strict causal and full bidirectional masks on a text completion task.

### Hard

Implement a mixture-of-experts (MoE) attention layer where different attention heads are routed to different expert feedforward networks. Compare its efficiency and quality with a standard transformer layer.

## Solutions

### Easy Solution

```python
def compare_bert_gpt():
    vocab, d_model, n_heads = 50, 32, 4
    bert = BertClassifier(vocab, d_model, n_heads, 128, 2, num_classes=2)
    gpt = GPTModel(vocab, d_model, n_heads, 128, 2)
    gpt_classifier = nn.Sequential(gpt, nn.Linear(vocab, 2))
    input_ids = torch.randint(0, 50, (4, 8))
    bert_out = bert(input_ids, torch.ones(4, 8))
    gpt_out = gpt_classifier(input_ids)[:, -1]
    print(f"BERT output: {bert_out.shape}, GPT output: {gpt_out.shape}")
    print("BERT uses [CLS] token; GPT uses last token for classification")
# Output: BERT output: torch.Size([4, 2]), GPT output: torch.Size([4, 2])
# Output: BERT uses [CLS] token; GPT uses last token for classification
```

## Related Concepts

- BERT and Masked Language Modeling
- GPT and Autoregressive Language Modeling
- T5 and Text-to-Text Framework
- Prefix-LM
- Multimodal Attention (CLIP, LLaVA)

## Next Concepts

- This is the last concept in the sequence (DL-355).

## Summary

Attention in NLP encompasses the application of transformer attention mechanisms to natural language processing tasks. Three major architecture families dominate: encoder-only (BERT, bidirectional self-attention for understanding), decoder-only (GPT, causal self-attention for generation), and encoder-decoder (T5, bidirectional + causal + cross-attention for seq2seq tasks). Each architecture's attention pattern determines what tasks it excels at. Attention enables contextual word representations, long-range dependency modeling, and coherent text generation. All modern NLP systems are built on attention mechanisms, making this knowledge fundamental to the field.

## Key Takeaways

- NLP attention has three architecture families: encoder-only, decoder-only, and encoder-decoder.
- BERT uses bidirectional attention — each token sees all tokens (best for understanding).
- GPT uses causal attention — each token sees only previous tokens (best for generation).
- T5 uses bidirectional encoder + causal decoder with cross-attention (best for seq2seq).
- Attention patterns determine what tasks a model can perform.
- All modern NLP systems use transformer attention as their core mechanism.
- Pre-training + fine-tuning is the standard paradigm for attention-based NLP models.
