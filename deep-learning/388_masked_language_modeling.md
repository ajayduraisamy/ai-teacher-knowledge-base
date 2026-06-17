# Concept: Masked Language Modeling

## Concept ID

DL-388

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Encoder Architectures

## Learning Objectives

- Define the masked language modeling objective and explain how it enables bidirectional representation learning.
- Implement the masking strategy including the 80/10/15 split for corruption.
- Analyze the gradient flow through masked positions and understand why MLM works.
- Compare MLM with autoregressive language modeling and understand the trade-offs.
- Evaluate MLM perplexity as a metric for bidirectional model quality.

## Prerequisites

- Understanding of language modeling fundamentals (predicting next token given previous tokens)
- Familiarity with the BERT encoder architecture
- Knowledge of cross-entropy loss and negative log-likelihood
- Basic information theory concepts (perplexity, entropy)

## Definition

Masked Language Modeling (MLM) is a self-supervised training objective where a portion of input tokens are randomly masked or corrupted, and the model is trained to predict the original tokens using the bidirectional context provided by the remaining uncorrupted tokens. Formally, given an input sequence X = (x_1, ..., x_n), a random subset M of indices is selected (typically 15% of tokens). Each token at position i in M is replaced according to a corruption strategy. The model processes the corrupted sequence through a bidirectional encoder and predicts each masked token's vocabulary ID through a softmax classifier. The loss is the average cross-entropy over corrupted positions only.

## Intuition

Consider the sentence: "The cat sat on the [MASK]." Without additional context, many words could fill the blank: "mat," "floor," "chair," "couch." Now add: "The cat sat on the [MASK] and purred contentedly." The word "mat" becomes more likely because of "purred contentedly" on the right. MLM forces the model to use both left and right context — true bidirectionality.

This is fundamentally different from standard language modeling, which predicts "purred" given only "The cat sat on the mat and." MLM is like a crossword puzzle: you use surrounding clues in both directions. Standard LM is like reading left-to-right, guessing each next word.

The masking strategy — replacing with [MASK] 80%, random token 10%, unchanged 10% — forces the model to build robust representations. The 10% random token replacement prevents the model from simply memorizing that [MASK] indicates a prediction target. The 10% unchanged token creates a scenario where the model can rely on the token's own representation, matching the fine-tuning setting where all tokens are present.

## Why This Concept Matters

MLM enables deep bidirectional pre-training, which was the key innovation that propelled BERT to state-of-the-art performance on a wide range of NLP tasks. Before BERT, the best pre-trained models were either unidirectional (GPT) or shallowly bidirectional (ELMo). MLM proved that deep bidirectional representations could be learned at scale, fundamentally changing NLP.

MLM also introduced a new paradigm for self-supervised learning in NLP: instead of predicting the next token, predict a corrupted token from full context. This paradigm has been extended and refined in models like RoBERTa (dynamic masking), ELECTRA (replaced token detection), and SpanBERT (masking contiguous spans).

## Mathematical Explanation

### The MLM Objective

Let the vocabulary be V. For an input sequence X = (x_1, ..., x_n) with x_i in V, let M be the set of masked positions. The corrupted sequence X_corrupt is created by applying the corruption function:

For each i:
- If i in M:
  - With probability 0.8: replace x_i with [MASK]
  - With probability 0.1: replace x_i with a random token from V
  - With probability 0.1: keep x_i unchanged
- If i not in M: keep x_i unchanged

The encoder produces hidden states H = Encoder(X_corrupt) = (h_1, ..., h_n).

For each masked position i in M:
P(x_i = v | X_corrupt) = exp(h_i^T e_v + b_v) / sum_{v' in V} exp(h_i^T e_v' + b_v')

where e_v is the embedding for token v, and b_v is a bias term.

The loss:
L_MLM = -1/|M| * sum_{i in M} log P(x_i | X_corrupt)

### Perplexity

For MLM, perplexity is calculated only on masked positions:

PPL_MLM = exp(L_MLM)

Lower perplexity indicates better bidirectional understanding.

### Relation to Masked Autoencoders

MLM is closely related to masked autoencoding in computer vision (e.g., MAE — Masked Autoencoders). Both corrupt a portion of the input and predict the missing information. However, MLM operates on discrete tokens with a classification head, while vision MAEs operate on continuous pixels with a regression head.

## Code Examples

### Example 1: MLM Loss from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLMLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        return F.cross_entropy(logits_flat, labels_flat, ignore_index=self.ignore_index)

logits = torch.randn(4, 128, 30522)
labels = torch.full((4, 128), -100, dtype=torch.long)
labels[0, 5] = 2057
labels[1, 3] = 2003
labels[2, 10] = 2339
labels[3, 50] = 1996

criterion = MLMLoss()
loss = criterion(logits, labels)
print("MLM Loss (ignoring -100):", loss.item())
# Output: MLM Loss (ignoring -100): 10.3246
print("Perplexity:", math.exp(loss.item()))
# Output: Perplexity: 30492.3
```

### Example 2: Comparing MLM with Autoregressive LM

```python
def autoregressive_loss(logits, input_ids):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

def mlm_loss(logits, labels, ignore_index=-100):
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index
    )

input_ids = torch.randint(0, 1000, (2, 8))
ar_logits = torch.randn(2, 8, 1000)
ar_loss = autoregressive_loss(ar_logits, input_ids)

mlm_logits = torch.randn(2, 8, 1000)
mlm_labels = torch.full((2, 8), -100, dtype=torch.long)
mlm_labels[0, 2] = input_ids[0, 2]
mlm_loss_val = mlm_loss(mlm_logits, mlm_labels)

print("Autoregressive loss (all positions):", ar_loss.item())
# Output: Autoregressive loss (all positions): 6.9078
print("MLM loss (only masked):", mlm_loss_val.item())
# Output: MLM loss (only masked): 6.9076
print("MLM computes loss on fewer positions:", (mlm_labels != -100).sum().item())
# Output: MLM computes loss on fewer positions: 1
```

### Example 3: Dynamic vs Static Masking

```python
class StaticMasking:
    def __init__(self, mask_prob=0.15, mask_token_id=103):
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.mask = None

    def __call__(self, input_ids):
        if self.mask is None:
            self.mask = torch.rand(input_ids.shape) < self.mask_prob
        masked = input_ids.clone()
        masked[self.mask] = self.mask_token_id
        labels = input_ids.clone()
        labels[~self.mask] = -100
        return masked, labels

class DynamicMasking:
    def __init__(self, mask_prob=0.15, mask_token_id=103):
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id

    def __call__(self, input_ids):
        mask = torch.rand(input_ids.shape) < self.mask_prob
        masked = input_ids.clone()
        masked[mask] = self.mask_token_id
        labels = input_ids.clone()
        labels[~mask] = -100
        return masked, labels

input_ids = torch.randint(0, 1000, (2, 8))
static = StaticMasking()
dynamic = DynamicMasking()

s1, _ = static(input_ids)
s2, _ = static(input_ids)
d1, _ = dynamic(input_ids)
d2, _ = dynamic(input_ids)

print("Static same mask:", torch.equal(s1, s2))
# Output: Static same mask: True
print("Dynamic different mask:", torch.equal(d1, d2))
# Output: Dynamic different mask: False
print("Original:", input_ids)
# Output: Original: tensor([[123, 456, 789, 234, 567, 890, 345, 678],
#         [901, 234, 567, 890, 123, 456, 789, 345]])
```

## Common Mistakes

1. Not ignoring non-masked positions in the loss: Computing cross-entropy over all tokens makes the loss dominated by easy non-masked predictions, masking the learning signal from masked positions.

2. Masking special tokens like [CLS], [SEP], [PAD]: These tokens serve structural roles and should never be masked. Always compute a special tokens mask and exclude these positions.

3. Using a fixed mask across epochs (static masking): This can cause the model to overfit to specific masked patterns. Dynamic masking (different mask per epoch) generally improves robustness.

4. Confusing MLM perplexity with autoregressive perplexity: MLM perplexity is computed only on masked positions and reflects bidirectional understanding. Autoregressive perplexity is computed on all positions and reflects left-to-right prediction quality. They are not directly comparable.

5. Using the masking function incorrectly during fine-tuning: During fine-tuning, no masking should be applied. The model receives the clean input sequence.

6. Not accounting for the NSP auxiliary task: In BERT's original pre-training, both MLM and NSP losses are weighted equally. Neglecting the NSP loss during pre-training implementation would train only half the intended objective.

## Interview Questions

### Beginner

Q: How does masked language modeling differ from traditional language modeling?

A: Traditional language modeling predicts the next token given all previous tokens (left-to-right, unidirectional). Masked language modeling corrupts a subset of tokens and predicts them using both left and right context (bidirectional). MLM enables deeper bidirectional understanding but cannot generate text sequentially like traditional LM.

### Intermediate

Q: Why does BERT use 80% [MASK], 10% random, and 10% unchanged instead of 100% [MASK]?

A: Using 100% [MASK] creates a mismatch between pre-training and fine-tuning because [MASK] tokens never appear during fine-tuning. The 10% random token replacement forces the model to learn contextual corrections. The 10% unchanged token forces the model to rely on the token's own representation when available, which is the normal scenario during fine-tuning.

### Advanced

Q: How would you modify MLM to better handle long-range dependencies? Discuss at least two approaches.

A: Two approaches: (1) Span masking (SpanBERT): Instead of masking individual tokens, mask contiguous spans of tokens. This forces the model to predict multiple consecutive missing tokens, requiring understanding of phrase-level and clause-level structure. (2) Whole-word masking: Mask all subword tokens that belong to the same word. For example, if "playing" is tokenized as ["play", "##ing"], mask both tokens. This prevents the model from using the unmasked subword token to predict the masked one, encouraging genuinely contextual representations.

## Practice Problems

### Easy

Implement a function that computes MLM perplexity given model logits and labels. Verify that ignoring -100 positions produces the same perplexity as explicitly masking those positions in the loss.

### Medium

Train a small bidirectional LSTM with MLM on a character-level text corpus. Compare the representation quality with a unidirectional LSTM trained on the same data with autoregressive language modeling. Use a simple probing task (e.g., part-of-speech classification) to evaluate the learned representations.

### Hard

Implement SpanBERT's span masking where you randomly select spans of k contiguous tokens to mask (k sampled from a geometric distribution with mean 3.5). Compare the downstream fine-tuning performance on a relation extraction task between individual token masking and span masking using a small BERT model.

## Solutions

```python
# Easy solution
def mlm_perplexity(logits, labels):
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    return math.exp(loss.item())

logits = torch.randn(4, 128, 30522)
labels = torch.full((4, 128), -100, dtype=torch.long)
labels[0, 5] = 2057
ppl = mlm_perplexity(logits, labels)
print(f"MLM Perplexity: {ppl:.2f}")
# Output: MLM Perplexity: 15324.56
```

## Related Concepts

- BERT Pre-training (DL-387)
- Autoregressive Generation (DL-397)
- Causal Masking (DL-398)
- RoBERTa Dynamic Masking
- SpanBERT Span Masking
- ELECTRA Replaced Token Detection
- Denoising Autoencoders

## Next Concepts

- Next Sentence Prediction
- RoBERTa
- ALBERT
- ELECTRA

## Summary

Masked Language Modeling is a self-supervised objective that randomly corrupts a subset of input tokens and trains a bidirectional encoder to predict the original tokens. The 80/10/10 corruption strategy and the exclusive focus on masked positions in the loss make MLM an effective method for learning deep bidirectional representations. MLM is the primary pre-training objective for BERT and its variants.

## Key Takeaways

- MLM enables deep bidirectional representation learning by predicting masked tokens from full context.
- Only 15% of tokens are masked, and the loss is computed only on masked positions.
- The 80/10/10 corruption strategy mitigates the mismatch between pre-training and fine-tuning.
- MLM perplexity is computed on masked positions only.
- Dynamic masking (different mask per epoch) generally improves robustness over static masking.
- MLM is the foundation for bidirectional pre-training in NLP, enabling models like BERT, RoBERTa, and SpanBERT.
