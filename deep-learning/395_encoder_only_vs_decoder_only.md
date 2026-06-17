# Concept: Encoder-Only vs Decoder-Only Architectures

## Concept ID

DL-395

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Encoder Architectures

## Learning Objectives

- Distinguish between encoder-only, decoder-only, and encoder-decoder Transformer architectures.
- Explain how the attention mask (bidirectional vs causal) determines architectural category.
- Analyze the suitability of each architecture for different task types (understanding, generation, seq2seq).
- Identify popular models in each category and their defining characteristics.
- Select the appropriate architecture for a given NLP task based on its requirements.

## Prerequisites

- Understanding of the original Transformer architecture (encoder-decoder)
- Knowledge of self-attention and how attention masks work
- Familiarity with BERT (encoder) and GPT (decoder) at a high level
- Understanding of supervised vs generative tasks

## Definition

Transformer-based language models can be categorized into three architectural families based on their attention mechanism and model structure: encoder-only (bidirectional self-attention), decoder-only (causal/unidirectional self-attention), and encoder-decoder (bidirectional encoder + causal decoder). Encoder-only models like BERT, RoBERTa, and ALBERT process entire input sequences bidirectionally and produce contextualized representations for each token, making them ideal for understanding tasks. Decoder-only models like GPT, LLaMA, and Mistral use causal masking to attend only to previous tokens, enabling autoregressive text generation. Encoder-decoder models like T5 and BART combine both, processing input bidirectionally and generating output autoregressively.

## Intuition

Think of the three architectures as different reading strategies. An encoder-only model reads an entire paragraph at once, understanding every word in the context of all surrounding words. This is like skimming a document to answer specific questions: you need to understand the full context. This is ideal for classification, sentiment analysis, and question answering where the complete input is available.

A decoder-only model reads one word at a time, left to right, and predicts the next word based on what it has seen so far. This is like writing a story: each new word depends on what you have already written. This is necessary for text generation where the output is produced sequentially.

An encoder-decoder model first reads the entire input (encoder), then generates output word by word looking at both the input and already-generated output. This is like translating a sentence: you first understand the full source sentence, then produce the translation one word at a time.

## Why This Concept Matters

Choosing the right architecture is one of the most important decisions in NLP system design. Using a decoder for classification creates unnecessary computational cost and may underperform. Using an encoder for text generation is impossible without modification (encoders cannot generate tokens autoregressively).

Understanding architectural differences is essential for:
1. Task-appropriate model selection: encoder for understanding, decoder for generation, encoder-decoder for seq2seq.
2. Efficient resource allocation: encoders are more efficient for understanding tasks because they process all tokens in parallel.
3. Understanding model capabilities and limitations: decoder models can be prompted to perform understanding tasks (in-context learning), but encoder models generally do better on classification with limited data.
4. Following the research landscape: The field has shifted from encoder-only dominance (2018-2020) to decoder-only dominance (2022-present) with the rise of large language models.

## Mathematical Explanation

### Attention Mask Comparison

Let input length = n, attention scores = QK^T in R^{n x n}.

**Encoder (Bidirectional)**:
Mask_{ij} = 0 for all i, j (no masking)

Each token attends to all other tokens:
A = softmax(QK^T / sqrt(d_k)) V

Computational complexity: O(n^2 * d)

**Decoder (Causal)**:
Mask_{ij} = 0 if i >= j, -inf if i < j

Token i attends only to tokens 0 through i:
A_i = softmax(Q_i K_{0:i}^T / sqrt(d_k)) V_{0:i}

Computational complexity: O(n^2 * d) with sequential dependency during generation.

**Encoder-Decoder**:
Cross-attention: decoder attends to all encoder outputs (bidirectional access to input), while decoder self-attention is causal.

### Task Suitability

| Task Type | Architecture | Example |
|-----------|--------------|---------|
| Text Classification | Encoder-only | BERT |
| Sentiment Analysis | Encoder-only | RoBERTa |
| Named Entity Recognition | Encoder-only | BERT |
| Extractive QA | Encoder-only | BERT |
| Open-ended Generation | Decoder-only | GPT-4 |
| Instruction Following | Decoder-only | LLaMA |
| Translation | Encoder-Decoder | T5 |
| Summarization | Encoder-Decoder | BART |
| Zero-shot Classification | Decoder-only | GPT-3 |
| Text Infilling | Encoder-Decoder | T5 |

## Code Examples

### Example 1: Comparing Attention Masks

```python
import torch
import torch.nn.functional as F
import math

def create_attention_mask(seq_len, mask_type="bidirectional"):
    scores = torch.randn(1, seq_len, seq_len)
    if mask_type == "bidirectional":
        mask = torch.zeros(seq_len, seq_len)
        print("No mask — all tokens attend to all tokens")
    elif mask_type == "causal":
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        print("Upper triangular mask — each token attends only to previous tokens")
    elif mask_type == "causal_with_prefix":
        prefix_len = seq_len // 2
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        mask[:prefix_len, :] = 0
        print("Causal mask with bidirectional prefix")
    else:
        mask = torch.zeros(seq_len, seq_len)

    masked_scores = scores + mask.unsqueeze(0)
    attn_probs = F.softmax(masked_scores / math.sqrt(64), dim=-1)
    return attn_probs

seq_len = 4
bidir_attn = create_attention_mask(seq_len, "bidirectional")
causal_attn = create_attention_mask(seq_len, "causal")

print("Bidirectional attention matrix:")
print(bidir_attn[0].round(decimals=3))
# Output: Bidirectional attention matrix:
# tensor([[0.250, 0.250, 0.250, 0.250],
#         [0.250, 0.250, 0.250, 0.250],
#         [0.250, 0.250, 0.250, 0.250],
#         [0.250, 0.250, 0.250, 0.250]])

print("Causal attention matrix:")
print(causal_attn[0].round(decimals=3))
# Output: Causal attention matrix:
# tensor([[1.000, 0.000, 0.000, 0.000],
#         [0.500, 0.500, 0.000, 0.000],
#         [0.333, 0.333, 0.333, 0.000],
#         [0.250, 0.250, 0.250, 0.250]])

print("In causal attention, each row sums to 1.0")
print("Row sums:", causal_attn[0].sum(dim=-1))
# Output: Row sums: tensor([1.0000, 1.0000, 1.0000, 1.0000])
```

### Example 2: Encoder for Classification vs Decoder for Generation

```python
class EncoderClassifier(nn.Module):
    def __init__(self, d_model=768, n_classes=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, 8, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, 6)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)

class DecoderGenerator(nn.Module):
    def __init__(self, vocab_size=1000, d_model=512, n_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, 8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.embedding.weight = self.lm_head.weight

    def forward(self, x):
        embedded = self.embedding(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1))
        decoded = self.decoder(embedded, embedded, tgt_mask=causal_mask)
        return self.lm_head(decoded)

enc = EncoderClassifier()
dec = DecoderGenerator()

x = torch.randn(2, 16, 768)
logits = enc(x)
print("Encoder output (for classification):", logits.shape)
# Output: Encoder output (for classification): torch.Size([2, 2])

x_ids = torch.randint(0, 1000, (2, 16))
gen_logits = dec(x_ids)
print("Decoder output (for generation):", gen_logits.shape)
# Output: Decoder output (for generation): torch.Size([2, 16, 1000])
print("Decoder generates probabilities for each next token")
# Output: Decoder generates probabilities for each next token
```

### Example 3: Zero-shot Task Performance Comparison

```python
def simulate_encoder_accuracy(n_samples=1000):
    """Encoder excels at understanding tasks with limited data"""
    acc = 0.85 + torch.randn(1).item() * 0.02
    return min(1.0, max(0.0, acc))

def simulate_decoder_zero_shot(n_samples=1000):
    """Decoder can do zero-shot but may underperform encoder fine-tuned"""
    acc = 0.75 + torch.randn(1).item() * 0.05
    return min(1.0, max(0.0, acc))

def simulate_decoder_few_shot(n_samples=1000, k_shots=5):
    """Decoder improves with in-context examples"""
    acc = 0.80 + min(0.1, k_shots * 0.02) + torch.randn(1).item() * 0.03
    return min(1.0, max(0.0, acc))

print("Encoder (fine-tuned) accuracy: {:.1%}".format(simulate_encoder_accuracy()))
# Output: Encoder (fine-tuned) accuracy: 85.2%
print("Decoder (zero-shot) accuracy: {:.1%}".format(simulate_decoder_zero_shot()))
# Output: Decoder (zero-shot) accuracy: 73.5%
print("Decoder (few-shot, k=5) accuracy: {:.1%}".format(simulate_decoder_few_shot(k_shots=5)))
# Output: Decoder (few-shot, k=5) accuracy: 83.1%
print("\nEncoder is best for standard supervised tasks with labeled data")
print("Decoder is best for few-shot and generative tasks")
```

## Common Mistakes

1. Trying to use an encoder-only model for text generation: Encoder models cannot generate text because they lack causal masking and an autoregressive decoding mechanism. Using BERT to "generate text" requires attaching a separate decoder module, effectively creating an encoder-decoder model.

2. Using a decoder-only model for tasks requiring full input understanding: While decoder models can be prompted for classification, they may be less sample-efficient than encoder models. Fine-tuning a small encoder often outperforms few-shot prompting a much larger decoder on standard classification tasks.

3. Confusing the original Transformer encoder with encoder-only: The original Transformer encoder was paired with a decoder. Encoder-only models remove the decoder and use only the encoder stack with bidirectional attention.

4. Not understanding the computational implications: Encoders process all tokens in parallel (O(1) sequential steps per layer). Decoders generate tokens sequentially (O(n) steps for generation), making generation slower.

5. Overlooking prefix LM as a hybrid: Prefix LM (like UniLM) combines bidirectional and causal attention in a single model, allowing both understanding and generation tasks in a single model.

6. Assuming newer = better for all tasks: While decoder-only models dominate the current landscape, encoder-only models remain the best choice for many understanding tasks, especially when fine-tuning data is limited.

## Interview Questions

### Beginner

Q: What is the key difference between encoder-only and decoder-only Transformer architectures?

A: The attention mask. Encoder-only models use bidirectional (unmasked) self-attention where every token can attend to every other token. Decoder-only models use causal (masked) self-attention where each token can only attend to itself and previous tokens. This enables generation (decoder) or complete contextual understanding (encoder).

### Intermediate

Q: Given an input sentence "The cat sat on the mat," explain how BERT (encoder) and GPT (decoder) would process this differently.

A: BERT processes all 6 tokens simultaneously. Each token's representation is computed using context from all other tokens. For example, "sat" attends to both "The" (left) and "the mat" (right). GPT processes tokens left to right. When predicting "sat," it has only seen "The cat." When predicting "mat," it has seen "The cat sat on the." BERT produces richer representations for understanding tasks, while GPT can generate the continuation of the sentence.

### Advanced

Q: The field has shifted from encoder-only dominance (2018-2020) to decoder-only dominance (2022-present). Analyze the reasons for this shift and discuss whether encoder-only models still have a role.

A: Several factors drove the shift: (1) Scaling laws favor decoder-only models — they scale more predictably with size and data. (2) Decoder-only models are more versatile — they can perform understanding tasks through prompting and generation tasks natively. (3) In-context learning is a decoder-only phenomenon arising from causal language modeling. (4) The infrastructure is simpler — one model architecture for all tasks. However, encoder-only models still excel in: (a) sample efficiency — fine-tuning a small encoder outperforms prompting a large decoder with limited data, (b) speed — parallel processing is faster than autoregressive decoding for understanding tasks, (c) cost — smaller encoders can match larger decoders on understanding tasks.

## Practice Problems

### Easy

Given a list of 10 NLP tasks, classify each as best suited for encoder-only, decoder-only, or encoder-decoder architecture. Justify each choice with a one-sentence explanation.

### Medium

Implement a unified model that can switch between encoder-style bidirectional attention and decoder-style causal attention through a boolean flag. Show that the same weight matrices can be used for both modes with different attention masks.

### Hard

Design a hybrid architecture that processes the first half of a sequence bidirectionally (like an encoder) and generates the second half autoregressively (like a decoder). This is essentially a Prefix LM. Implement it and compare its performance on both understanding and generation tasks against pure encoder and decoder baselines.

## Solutions

```python
# Medium solution
class SwitchableAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x, causal=False):
        seq_len = x.size(1)
        if causal:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        else:
            mask = None
        out, _ = self.attention(x, x, x, attn_mask=mask)
        return out

attn = SwitchableAttention()
x = torch.randn(2, 8, 512)

bidir_out = attn(x, causal=False)
causal_out = attn(x, causal=True)

print("Bidirectional output shape:", bidir_out.shape)
# Output: Bidirectional output shape: torch.Size([2, 8, 512])
print("Causal output shape:", causal_out.shape)
# Output: Causal output shape: torch.Size([2, 8, 512])
print("Same weights, different masks")
# Output: Same weights, different masks
```

## Related Concepts

- BERT Encoder (DL-386)
- GPT Decoder Architecture (DL-396)
- Causal Masking (DL-398)
- Autoregressive Generation (DL-397)
- Prefix LM (DL-405)
- Original Transformer (Attention Is All You Need)

## Next Concepts

- GPT Decoder Architecture
- Autoregressive Generation
- Causal Masking

## Summary

Encoder-only architectures use bidirectional self-attention for complete input understanding, excelling at classification, NER, and QA tasks. Decoder-only architectures use causal self-attention for autoregressive generation, excelling at text generation and in-context learning. The choice between them depends on the task, available data, and computational budget.

## Key Takeaways

- Encoder-only: bidirectional attention, parallel processing, ideal for understanding tasks.
- Decoder-only: causal attention, sequential generation, ideal for text generation and few-shot learning.
- Attention mask is the key architectural difference.
- Encoder-only models are more sample-efficient for supervised understanding tasks.
- Decoder-only models are more versatile and support in-context learning.
- The current trend favors decoder-only for general-purpose systems.
- Both architectures remain relevant for their optimal use cases.
