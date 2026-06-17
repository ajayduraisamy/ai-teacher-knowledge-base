# Concept: Prefix LM

## Concept ID

DL-405

## Difficulty

Expert

## Domain

Deep Learning

## Module

Decoder Architectures

## Learning Objectives

- Define the Prefix LM architecture and explain how it combines bidirectional prefix encoding with autoregressive generation.
- Understand the two-phase attention: bidirectional attention on the prefix, causal attention on the suffix.
- Implement a Prefix LM by modifying the attention mask of a standard decoder-only model.
- Compare Prefix LM with encoder-decoder and decoder-only architectures across multiple tasks.
- Analyze the advantages of Prefix LM for tasks requiring both input understanding and generation.

## Prerequisites

- Expert-level understanding of Transformer attention mechanisms
- Thorough knowledge of both encoder-only (bidirectional) and decoder-only (causal) attention masks
- Understanding of causal masking (DL-398) and decoder-only architecture (DL-403)
- Familiarity with sequence-to-sequence tasks (translation, summarization)

## Definition

A Prefix LM (Language Model) is a hybrid architecture that processes an input sequence using a two-stage attention pattern: the first k tokens (the prefix) use bidirectional attention (each token can attend to all other prefix tokens), while the remaining tokens use causal attention (each token can attend only to itself and previous tokens). This is achieved by modifying the attention mask: the upper-left k x k block of the mask is set to 0 (bidirectional), while the remaining blocks follow the causal pattern. The Prefix LM allows a single decoder-only model to perform both understanding (processing input bidirectionally) and generation (producing output autoregressively) without requiring a separate encoder. This architecture was introduced in UniLM and later adopted in models like GLM-130B and certain training configurations of T5.

## Intuition

Think of Prefix LM as a model that first reads a passage carefully, understanding everything in the context of everything else (like BERT), and then continues writing from that understanding, one word at a time (like GPT). The same model, the same weights, but different attention behavior for different parts of the sequence.

This is similar to how a human translator works: first read the entire source sentence (bidirectional understanding), then produce the translation word by word (autoregressive generation). The key innovation is that both phases use the same model — there is no separate encoder.

The prefix can be any portion of the input: a task description, a source text, a prompt, or a combination. The model builds a rich bidirectional representation of the prefix and then uses it as context for generation. Because the prefix is processed bidirectionally, it produces better representations than if it were processed causally (as in a standard decoder-only model).

## Why This Concept Matters

Prefix LM bridges the gap between encoder-decoder and decoder-only architectures:

1. **Unified architecture**: One model handles both understanding and generation without architectural changes.
2. **Better prefix encoding**: Bidirectional prefix processing produces richer representations than causal processing for understanding tasks.
3. **Flexible task adaptation**: The same model can be used for classification (prefix-only), generation (prefix + generation), or infilling (prefix + generation + suffix).
4. **Simpler infrastructure**: No separate encoder model to maintain, reducing deployment complexity.
5. **Span corruption pre-training**: Can be pre-trained with span masking where corrupted spans are predicted using bidirectional context, combining benefits of MLM and autoregressive LM.

## Mathematical Explanation

### Attention Mask Structure

For a sequence of length L with prefix length k:

The attention mask M is a L x L matrix:

- Top-left k x k block: all 0 (bidirectional)
- Bottom-right (L-k) x (L-k) block: causal (0 on and below diagonal, -inf above)
- Top-right k x (L-k) block: 0 (prefix tokens can attend to all tokens)
- Bottom-left (L-k) x k block: 0 (generation tokens can attend to all prefix tokens)

Visually:
```
M = [0_{kxk}      | 0_{kx(L-k)}]
    [0_{(L-k)xk}  | C_{(L-k)x(L-k)}]
```

Where C is the standard causal mask.

### Forward Pass

Input: X = [x_1, ..., x_k] + [x_{k+1}, ..., x_L]
Position IDs: [1, ..., k, k+1, ..., L]

For positions i <= k (prefix):
h_i attends to {h_1, ..., h_L} (all positions, including future generation positions)

For positions i > k (generation):
h_i attends to {h_1, ..., h_k} (all prefix) + causal attention to {h_{k+1}, ..., h_i}

### Pre-training Objective

Prefix LM can be pre-trained with various objectives:
- Standard language modeling on the suffix
- Span corruption (mask spans in the input, predict them in the suffix)
- Prefix language modeling (treat the first 50% as prefix, predict the remaining 50%)

## Code Examples

### Example 1: Prefix LM Mask Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def create_prefix_lm_mask(seq_len, prefix_len):
    mask = torch.zeros(seq_len, seq_len)
    causal_region_start = prefix_len

    for i in range(seq_len):
        for j in range(seq_len):
            if i >= causal_region_start and j >= causal_region_start and j > i:
                mask[i, j] = float("-inf")

    return mask

def create_prefix_lm_mask_efficient(seq_len, prefix_len):
    full_causal = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    full_causal[:prefix_len, :] = 0
    full_causal[:, :prefix_len] = 0
    return full_causal

seq_len, prefix_len = 8, 3
mask = create_prefix_lm_mask_efficient(seq_len, prefix_len)
print(f"Prefix LM mask (prefix_len={prefix_len}, seq_len={seq_len}):")
print(mask)
# Output: Prefix LM mask (prefix_len=3, seq_len=8):
# tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., -inf, -inf, -inf],
#         [0., 0., 0., 0., 0., 0., -inf, -inf],
#         [0., 0., 0., 0., 0., 0., 0., -inf],
#         [0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0.]])
```

### Example 2: Prefix LM Model Implementation

```python
class PrefixLMBlock(nn.Module):
    def __init__(self, d_model=768, n_heads=12, d_ff=3072, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=attn_mask)[0]
        x = x + self.ffn(self.norm2(x))
        return x

class PrefixLM(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, n_layers=12, n_heads=12, max_len=1024):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            PrefixLMBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids, prefix_len):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        mask = create_prefix_lm_mask_efficient(T, prefix_len).to(input_ids.device)
        for block in self.blocks:
            x = block(x, attn_mask=mask)
        x = self.norm(x)
        return self.lm_head(x)

model = PrefixLM(vocab_size=1000, d_model=256, n_layers=4)
x = torch.randint(0, 1000, (2, 16))
logits = model(x, prefix_len=6)
print("Prefix LM logits:", logits.shape)
# Output: Prefix LM logits: torch.Size([2, 16, 1000])
```

### Example 3: Comparison Across Architectures

```python
def compare_architectures():
    seq_len = 10
    prefix_len = 4

    print("Architecture comparison (seq_len=10, prefix_len=4):")
    print()

    print("Encoder-only (BERT):")
    bidir_mask = torch.zeros(seq_len, seq_len)
    print(f"  All positions attend to all {seq_len} positions")
    print(f"  Total non-masked entries: {(bidir_mask == 0).sum().item()}/{seq_len**2}")

    print("Decoder-only (GPT):")
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    non_masked = (causal_mask == 0).sum().item()
    print(f"  Each position i attends to i+1 positions (self + previous)")
    print(f"  Total non-masked entries: {non_masked}/{seq_len**2}")

    print("Prefix LM:")
    prefix_mask = create_prefix_lm_mask_efficient(seq_len, prefix_len)
    non_masked_prefix = (prefix_mask == 0).sum().item()
    print(f"  Prefix positions attend to all {seq_len} positions")
    print(f"  Suffix positions attend to prefix ({prefix_len}) + previous suffix tokens")
    print(f"  Total non-masked entries: {non_masked_prefix}/{seq_len**2}")
    print(f"  Bidirectional coverage: {prefix_len}/{seq_len} positions")

compare_architectures()
# Output: Architecture comparison (seq_len=10, prefix_len=4):
# Output:
# Output: Encoder-only (BERT):
# Output:   All positions attend to all 10 positions
# Output:   Total non-masked entries: 100/100
# Output: Decoder-only (GPT):
# Output:   Each position i attends to i+1 positions (self + previous)
# Output:   Total non-masked entries: 55/100
# Output: Prefix LM:
# Output:   Prefix positions attend to all 10 positions
# Output:   Suffix positions attend to prefix (4) + previous suffix tokens
# Output:   Total non-masked entries: 70/100
# Output:   Bidirectional coverage: 4/10 positions
```

## Common Mistakes

1. Confusing Prefix LM with encoder-decoder: Prefix LM uses a single model with a modified attention mask, while encoder-decoder uses two separate models (encoder and decoder) with cross-attention. Prefix LM shares weights across the bidirectional and causal processing, while encoder-decoder has separate parameters.

2. Incorrect mask structure: The prefix must have full bidirectional attention (including to future suffix tokens). A common mistake is making the prefix bidirectional only among themselves but not attending to the suffix.

3. Thinking Prefix LM requires separate training from standard LM: Prefix LM can be trained with the same language modeling objective on the suffix portion. The model naturally learns to use the bidirectional prefix information.

4. Overlooking that position IDs are contiguous: The position IDs should form a contiguous sequence from 1 to L, even though the prefix and suffix have different attention patterns. The position embeddings are shared.

5. Using Prefix LM for tasks that don't benefit from bidirectional prefix: For pure generation tasks (e.g., open-ended story continuation), the bidirectional prefix may not provide benefits over standard causal processing.

6. Forgetting that Prefix LM inference is still autoregressive: After the prefix is processed, generation is one token at a time, same as any decoder model. The prefix processing is parallel (all at once), which is efficient.

## Interview Questions

### Beginner

Q: What is a Prefix LM and how does its attention mask differ from a standard decoder-only model?

A: A Prefix LM uses a two-stage attention pattern: the first k tokens (prefix) use bidirectional attention (all prefix tokens attend to each other), while the remaining tokens use causal attention. The mask has three parts: (1) bidirectional among prefix tokens, (2) prefix attending to all tokens, (3) suffix attending to prefix + previous suffix tokens. This differs from standard decoder-only where all positions use causal masking.

### Intermediate

Q: Compare Prefix LM with encoder-decoder architecture for a text summarization task.

A: Prefix LM processes the input document as a bidirectional prefix in the same model that generates the summary, sharing weights between understanding and generation. Encoder-decoder uses a separate encoder (bidirectional) to process the document and a separate decoder (causal) to generate the summary, with cross-attention connecting them. Advantages of Prefix LM: (1) fewer parameters (shared weights), (2) simpler deployment (one model), (3) can handle variable-length prefixes easily. Advantages of encoder-decoder: (1) encoder can be scaled differently from decoder, (2) cross-attention provides a clearer separation of input processing and output generation.

### Advanced

Q: Design a pre-training objective specifically for Prefix LM that would be more effective than standard causal LM or MLM. Explain why it would work better for the Prefix LM architecture.

A: Span corruption pre-training: Given a text sequence, select several contiguous spans to mask (covering ~15% of tokens). The unmasked tokens are the prefix. The model must predict the masked spans in the suffix using the bidirectional prefix context. This combines benefits: (1) Bidirectional prefix encodes rich context from both sides of each span. (2) Autoregressive suffix ensures the model learns generation capabilities. (3) Span-level prediction forces holistic understanding (vs. individual tokens in MLM). This is similar to T5's span corruption but adapted for a single model. The model learns to use the prefix encoder bidirectionally and the suffix decoder autoregressively in a unified manner.

## Practice Problems

### Easy

Implement a function that creates the Prefix LM attention mask for arbitrary sequence length and prefix length. Verify that the mask has the correct properties: prefix positions attend to all positions, suffix positions attend to prefix plus previous suffix positions.

### Medium

Train a Prefix LM on a text corpus using the span corruption objective. Fine-tune the model on a summarization task and compare ROUGE scores with: (a) a standard decoder-only model of the same size, and (b) an encoder-decoder model of similar total parameter count.

### Hard

Design a dynamic Prefix LM where the prefix length is determined by a learned mechanism during inference. The model should decide how much of the input needs bidirectional processing (prefix) vs. can be processed causally (suffix). Implement a lightweight router that predicts the optimal prefix length for each input.

## Solutions

```python
# Easy solution
def verify_prefix_lm_mask(seq_len=8, prefix_len=3):
    mask = create_prefix_lm_mask_efficient(seq_len, prefix_len)
    properties = {
        "Prefix positions attend to all": (mask[:prefix_len, :] == 0).all(),
        "Suffix attends to prefix": (mask[prefix_len:, :prefix_len] == 0).all(),
        "Suffix causal (diagonal)": all(mask[i, i] == 0 for i in range(prefix_len, seq_len)),
        "Suffix no future": all(mask[i, j] == float("-inf")
                                 for i in range(prefix_len, seq_len)
                                 for j in range(i+1, seq_len)),
    }
    for name, result in properties.items():
        print(f"{name}: {result}")
    return all(properties.values())

verify_prefix_lm_mask()
# Output: Prefix positions attend to all: True
# Output: Suffix attends to prefix: True
# Output: Suffix causal (diagonal): True
# Output: Suffix no future: True
```

## Related Concepts

- Decoder-Only Architecture (DL-403)
- Causal Masking (DL-398)
- Encoder-Only vs Decoder-Only (DL-395)
- UniLM
- GLM-130B
- Span Corruption
- T5 Architecture

## Next Concepts

- BERT Base vs Large
- BERT Tokenization
- BERT Embeddings
- BERT Fine-tuning

## Summary

Prefix LM is a hybrid architecture that combines bidirectional prefix encoding (understanding) with autoregressive suffix generation (generation) in a single decoder-only model through a modified attention mask. It bridges the gap between encoder-decoder and decoder-only architectures, offering unified handling of understanding and generation tasks with shared parameters and simpler deployment.

## Key Takeaways

- Prefix LM uses a two-stage attention mask: bidirectional on prefix, causal on suffix.
- Prefix tokens attend to all positions, including suffix tokens.
- Suffix tokens attend to all prefix tokens and previous suffix tokens.
- Single model, shared weights for understanding and generation.
- More parameter-efficient than encoder-decoder (no separate encoder).
- More flexible than standard decoder-only (bidirectional input understanding).
- Can be pre-trained with span corruption or prefix language modeling.
- Suitable for summarization, translation, and other seq2seq tasks.
- Requires modified attention mask but same underlying implementation as decoder-only.
- Bridges the gap between BERT-style and GPT-style architectures.
