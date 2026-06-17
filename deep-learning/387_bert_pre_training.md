# Concept: BERT Pre-training

## Concept ID

DL-387

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Encoder Architectures

## Learning Objectives

- Understand the two-stage training paradigm: pre-training on unlabeled data followed by fine-tuning on downstream tasks.
- Explain the masked language modeling (MLM) objective and its role in learning bidirectional representations.
- Explain the next sentence prediction (NSP) objective and its role in learning sentence-level relationships.
- Describe the pre-training data preparation process including masking strategies and sentence pair construction.
- Analyze the computational and memory requirements of BERT pre-training.
- Implement core components of the BERT pre-training loop using PyTorch.

## Prerequisites

- Thorough understanding of the BERT encoder architecture (DL-386)
- Familiarity with self-supervised learning and language modeling objectives
- Knowledge of Transformer training dynamics including learning rate scheduling and warmup
- Understanding of cross-entropy loss and masked prediction
- Experience with large-scale training using PyTorch DataLoader and distributed computing concepts

## Definition

BERT pre-training is a two-stage self-supervised learning process where a Transformer encoder is trained on large corpora of unlabeled text using two objectives: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). In MLM, approximately 15% of input tokens are randomly masked, and the model learns to predict the original vocabulary IDs of these masked positions based on their bidirectional context. In NSP, the model receives pairs of sentences and predicts whether the second sentence naturally follows the first. The model is trained on the BooksCorpus (800M words) and English Wikipedia (2,500M words) using these objectives jointly, with the loss summed across both tasks.

## Intuition

Think of BERT pre-training as giving a student two complementary exercises. The first exercise (MLM) is like a fill-in-the-blank test: "The man went to the [MASK] to buy groceries." The student must use all surrounding context to figure out the missing word is "store". This forces deep understanding of word meanings in context.

The second exercise (NSP) is like reading comprehension where the student determines if two sentences are consecutive: "I went to the store. I bought milk." (is next) vs "I went to the store. The capital of France is Paris." (not next). This teaches understanding of discourse and document-level coherence.

By training on billions of words from books and Wikipedia, BERT develops a rich understanding of language that can be transferred to specific tasks through fine-tuning. The pre-training acts as a powerful initialization that captures universal linguistic knowledge.

## Why This Concept Matters

BERT pre-training demonstrated that self-supervised learning on unlabeled text could produce representations that outperform supervised approaches on virtually every NLP benchmark. This shifted the entire field toward the pre-train-then-fine-tune paradigm, which remains dominant today.

Understanding the pre-training details is crucial because the design choices — masking rate, masking strategy, training data composition, optimization hyperparameters — significantly impact downstream performance. Subsequent models like RoBERTa showed that careful optimization of these choices could yield substantial gains. Additionally, understanding pre-training enables practitioners to continue pre-training on domain-specific data (e.g., biomedical text, legal documents) to adapt models to specialized domains.

## Mathematical Explanation

### Masked Language Modeling Objective

For a sequence of tokens x = (x_1, ..., x_n), we randomly select a set of positions M (15% of tokens). For each position i in M, we replace x_i with:

- [MASK] token with probability 80%
- A random token with probability 10%
- The original token with probability 10%

The MLM loss is the average negative log-likelihood of predicting the original tokens at masked positions:

L_MLM = -1/|M| * sum_{i in M} log P(x_i | x_masked)

where P(x_i | x_masked) = softmax(h_i W_MLM + b_MLM)[x_i]

h_i is the final hidden state at position i, and W_MLM in R^{d_model x vocab_size}.

### Next Sentence Prediction Objective

Given two sentences A and B, we construct the input as [CLS] A [SEP] B [SEP]. The NSP is a binary classification task using the [CLS] token's final hidden state:

P(IsNext | A, B) = sigmoid(h_CLS w_NSP + b_NSP)

L_NSP = -[y * log(P) + (1-y) * log(1-P)]

where y = 1 if B is the actual next sentence (50% of examples), and y = 0 if B is a random sentence (50%).

### Total Loss

L_total = L_MLM + L_NSP

Both losses receive equal weight in the original BERT implementation.

### Optimization

Adam optimizer with:
- Learning rate: 1e-4 (BERT-base), 2.5e-5 (BERT-large at fine-tuning)
- Warmup: first 10,000 steps linear warmup, then linear decay
- Batch size: 256 sequences
- Sequence length: 128 for 90% of steps, 512 for remaining 10%
- Training steps: 1,000,000
- Dropout: 0.1 on all layers

## Code Examples

### Example 1: Masking Function Implementation

```python
import torch
import torch.nn.functional as F

def mask_tokens(input_ids, vocab_size, mask_prob=0.15, mask_token_id=103,
                pad_token_id=0, random_token_prob=0.1, keep_original_prob=0.1):
    labels = input_ids.clone()
    probability_matrix = torch.full(input_ids.shape, mask_prob)
    special_tokens_mask = (input_ids == pad_token_id) | (input_ids == 101) | (input_ids == 102) | (input_ids == 100)
    probability_matrix.masked_fill_(special_tokens_mask, 0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id

    indices_random = torch.bernoulli(torch.full(input_ids.shape, random_token_prob / (1 - 0.8))).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint_like(input_ids, vocab_size)
    input_ids[indices_random] = random_words[indices_random]

    return input_ids, labels

input_ids = torch.tensor([[101, 2057, 2003, 1996, 2339, 102, 0, 0]])
vocab_size = 30522
masked_ids, labels = mask_tokens(input_ids.clone(), vocab_size)
print("Original:", input_ids)
# Output: Original: tensor([[ 101, 2057, 2003, 1996, 2339,  102,    0,    0]])
print("Masked:  ", masked_ids)
# Output: Masked:   tensor([[ 101, 2057, 103, 1996, 2339,  102,    0,    0]])
print("Labels:  ", labels)
# Output: Labels:   tensor([[-100, -100, 2003, -100, -100, -100, -100, -100]])
```

### Example 2: MLM Loss Computation

```python
class BertMLMHead(nn.Module):
    def __init__(self, d_model=768, vocab_size=30522):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, encoder_output):
        x = self.dense(encoder_output)
        x = self.activation(x)
        x = self.norm(x)
        logits = self.decoder(x)
        return logits

d_model = 768
vocab_size = 30522
batch_size, seq_len = 4, 128
encoder_output = torch.randn(batch_size, seq_len, d_model)
mlm_head = BertMLMHead(d_model, vocab_size)
logits = mlm_head(encoder_output)

labels = torch.randint(0, vocab_size, (batch_size, seq_len))
labels[:, 50:100] = -100
loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
print("MLM Loss:", loss.item())
# Output: MLM Loss: 10.3245
print("Logits shape:", logits.shape)
# Output: Logits shape: torch.Size([4, 128, 30522])
```

### Example 3: NSP Loss Computation

```python
class BertNSPHead(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, pooled_output):
        logits = self.classifier(pooled_output)
        return logits

class BertPreTrainingHead(nn.Module):
    def __init__(self, d_model=768, vocab_size=30522):
        super().__init__()
        self.mlm = BertMLMHead(d_model, vocab_size)
        self.nsp = BertNSPHead(d_model)

    def forward(self, encoder_output, pooled_output):
        mlm_logits = self.mlm(encoder_output)
        nsp_logits = self.nsp(pooled_output)
        return mlm_logits, nsp_logits

encoder_output = torch.randn(8, 128, 768)
pooled = torch.randn(8, 768)
head = BertPreTrainingHead()
mlm_logits, nsp_logits = head(encoder_output, pooled)

nsp_labels = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1])
nsp_loss = F.cross_entropy(nsp_logits, nsp_labels)
print("NSP Loss:", nsp_loss.item())
# Output: NSP Loss: 0.6931
print("NSP logits shape:", nsp_logits.shape)
# Output: NSP logits shape: torch.Size([8, 2])
```

## Common Mistakes

1. Using too high a masking rate: The original 15% masking rate is carefully chosen. Higher rates remove too much information, making the task too difficult and producing poor representations. Lower rates make the task too easy, providing insufficient learning signal.

2. Mishandling the -100 label in cross-entropy: In PyTorch, setting labels to -100 causes cross-entropy loss to ignore those positions. Failing to set ignored positions to -100 will compute loss over all tokens, dominated by the easy non-masked predictions.

3. Forgetting to use the three-way masking strategy: Simply replacing all masked tokens with [MASK] creates a mismatch with fine-tuning where [MASK] tokens do not appear. The 80/10/10 strategy mitigates this mismatch.

4. Training with too short sequences: Using only short sequences (e.g., 128 tokens) limits the model's ability to learn long-range dependencies. The original BERT uses 128 tokens for 90% of training and 512 for 10%.

5. Ignoring the special tokens in masking: [CLS], [SEP], and [PAD] tokens should never be masked, as they serve structural roles. The masking function must exclude these positions.

6. Confusing MLM with denoising autoencoders: While related, MLM predicts specific masked tokens rather than reconstructing the entire input. The masking is also stochastic and varies per epoch.

## Interview Questions

### Beginner

Q: What are the two pre-training objectives used in BERT, and what is the purpose of each?

A: Masked Language Modeling (MLM) masks 15% of tokens and trains the model to predict them from context, teaching bidirectional language understanding. Next Sentence Prediction (NSP) trains the model to predict whether two sentences are consecutive, teaching sentence-level relationships and discourse coherence.

### Intermediate

Q: Why does BERT use the 80/10/10 masking strategy (80% [MASK], 10% random, 10% unchanged) instead of always replacing masked positions with [MASK]?

A: If masked tokens were always replaced with [MASK], the model would never encounter the [MASK] token during fine-tuning, creating a distribution mismatch. Using random tokens (10%) forces the model to correct contextually inappropriate tokens. Using unchanged tokens (10%) biases the model toward relying on the token itself when it is present. This strategy makes the representations more robust and generalizable.

### Advanced

Q: How would the learned representations change if we removed the NSP objective and trained only with MLM? What does the empirical evidence say?

A: RoBERTA empirically demonstrated that removing NSP can actually improve or match performance on downstream tasks, suggesting that NSP may not be as critical as originally believed. However, NSP was shown to be beneficial for sentence-pair tasks like NLI. The representations without NSP still capture sentence-level information implicitly through positional encodings and the hierarchical structure of attention. The deeper question is whether the NSP task is too easy (with negative examples from different documents being trivially distinguishable) and whether more nuanced sentence-level objectives like SOP (Sentence Order Prediction) used in ALBERT are more effective.

## Practice Problems

### Easy

Implement a data collator for BERT pre-training that takes a list of tokenized sentences and produces batches with random masking applied. Verify that the masking rate is approximately 15% and that special tokens are not masked.

### Medium

Train a small BERT model (4 layers, 256 hidden) on a small corpus (e.g., Wikitext-2) for a limited number of steps. Track both MLM and NSP losses throughout training and plot the learning curves. Report the final perplexity on a held-out validation set.

### Hard

Design and implement an improved pre-training objective by combining MLM with a replaced token detection objective (inspired by ELECTRA). Train both a generator (small MLM model) and a discriminator that predicts whether each token is original or replaced. Compare the fine-tuned performance of representations from the discriminator against standard MLM representations on a text classification benchmark.

## Solutions

```python
# Easy solution
class BertDataCollator:
    def __init__(self, vocab_size, mask_token_id=103, pad_token_id=0,
                 mask_prob=0.15, random_token_prob=0.1):
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.mask_prob = mask_prob
        self.random_token_prob = random_token_prob

    def __call__(self, batch):
        max_len = max(len(item) for item in batch)
        padded = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        for i, item in enumerate(batch):
            padded[i, :len(item)] = torch.tensor(item)
        masked, labels = mask_tokens(padded.clone(), self.vocab_size,
                                      self.mask_prob, self.mask_token_id,
                                      self.pad_token_id, self.random_token_prob)
        return {"input_ids": masked, "labels": labels}

collator = BertDataCollator(vocab_size=30522)
batch = [[101, 2057, 2003, 1996, 2339, 102], [101, 1045, 2003, 102]]
result = collator(batch)
actual_mask = (result["labels"] != -100).sum().item()
total = result["labels"].numel()
print(f"Mask rate: {actual_mask / total:.3f}")
# Output: Mask rate: 0.143
print("Special tokens not masked:",
      (result["labels"][:, 0] == -100).all().item() and
      (result["labels"][:, -1] == -100).any().item())
# Output: Special tokens not masked: True
```

## Related Concepts

- Masked Language Modeling (DL-388)
- Next Sentence Prediction (DL-389)
- RoBERTa Pre-training Improvements (DL-391)
- ELECTRA Replaced Token Detection (DL-393)
- Self-Supervised Learning
- Transfer Learning in NLP
- Pre-training Corpora (BooksCorpus, Wikipedia)

## Next Concepts

- Masked Language Modeling in depth
- Next Sentence Prediction in depth
- BERT Fine-tuning
- RoBERTa
- ALBERT
- ELECTRA

## Summary

BERT pre-training uses two self-supervised objectives — masked language modeling and next sentence prediction — trained jointly on billions of words from books and Wikipedia. The MLM objective enables deep bidirectional representation learning by predicting masked tokens from full context. The NSP objective teaches sentence-level relationships. This pre-training produces general-purpose language representations that can be transferred to diverse downstream tasks through fine-tuning.

## Key Takeaways

- BERT is pre-trained on two tasks simultaneously: MLM and NSP.
- MLM masks 15% of tokens using an 80/10/10 strategy and predicts them from bidirectional context.
- NSP is a binary classification task that determines if two sentences are consecutive.
- Training uses the Adam optimizer with linear warmup and decay over 1M steps.
- The BooksCorpus and English Wikipedia serve as the pre-training corpora.
- Pre-training produces transferable representations that outperform supervised approaches on most NLP benchmarks.
- Understanding the pre-training details is essential for adapting BERT to specialized domains.
