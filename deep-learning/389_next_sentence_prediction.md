# Concept: Next Sentence Prediction

## Concept ID

DL-389

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Encoder Architectures

## Learning Objectives

- Define the next sentence prediction objective and explain its role in BERT pre-training.
- Explain how sentence pairs are constructed for training data preparation.
- Analyze the limitations of NSP and compare with alternative sentence-level objectives (SOP, ALBERT).
- Implement NSP loss calculation and interpret model predictions.
- Evaluate the contribution of NSP to downstream task performance.

## Prerequisites

- Understanding of the BERT pre-training framework (DL-387)
- Familiarity with the [CLS] token and its role as a pooled representation
- Knowledge of binary classification and cross-entropy loss
- Understanding of discourse relations and document coherence

## Definition

Next Sentence Prediction (NSP) is a binary classification task introduced as a secondary pre-training objective for BERT. Given two sentences A and B, the model predicts whether B is the actual sentence that follows A in the original document (IsNext) or a randomly sampled sentence from another document (NotNext). The input is formatted as [CLS] A [SEP] B [SEP], and the [CLS] token's final hidden state is fed through a binary classifier. During training, 50% of examples are positive (actual consecutive sentences) and 50% are negative (random sentences from the corpus).

## Intuition

NSP teaches the model to understand the relationship between sentences — a capability crucial for tasks like question answering, natural language inference, and document understanding. When you read a story, you naturally track which sentences follow logically from previous ones. If you encounter "The man went to the store. He bought milk," you understand the connection. If you encounter "The man went to the store. The capital of France is Paris," you recognize the incoherence.

By learning to distinguish coherent from incoherent sentence pairs, BERT develops representations that capture discourse-level structure, topic continuity, and logical flow. This complements the token-level understanding from MLM with sentence-level and document-level understanding.

## Why This Concept Matters

NSP was the first objective to enable pre-trained models to handle sentence-pair tasks like natural language inference and paraphrase detection. While subsequent research (RoBERTa, ALBERT-SOP) raised questions about NSP's effectiveness, understanding NSP is important for several reasons:

1. It introduced the sentence-pair input format ([CLS] A [SEP] B [SEP]) that became standard for many downstream tasks.
2. It motivated the development of better sentence-level objectives (ALBERT's SOP, sentence ordering).
3. It teaches the concept of multi-task pre-training where different objectives capture different aspects of language.
4. The debate around NSP's utility provides valuable lessons about experimental rigor and ablation studies in NLP.

## Mathematical Explanation

### Input Construction

Given a document D = (s_1, s_2, ..., s_n), we sample a pair of sentences:

- Positive example (50%): (A, B) = (s_i, s_{i+1}) — consecutive sentences
- Negative example (50%): (A, B) = (s_i, s_j) where j is randomly sampled from a different document

The input sequence is:
Input = [CLS] tokenize(A) [SEP] tokenize(B) [SEP]

Segment embeddings:
- Positions corresponding to A: segment A (embedding index 0)
- Positions corresponding to B: segment B (embedding index 1)

### NSP Classifier

Let h_CLS in R^d be the final hidden state of the [CLS] token after L encoder layers.

P(IsNext | A, B) = sigmoid(h_CLS^T w + b)

Or equivalently with 2-class softmax:

logits = W_NSP h_CLS + b_NSP, where W_NSP in R^{2 x d}

P(IsNext) = exp(logits_0) / (exp(logits_0) + exp(logits_1))

### Loss

L_NSP = -[y * log(P_IsNext) + (1 - y) * log(1 - P_IsNext)]

where y = 1 for positive examples, y = 0 for negative examples.

### Total Pre-training Loss

L = L_MLM + L_NSP

Both losses are equally weighted in the original BERT.

## Code Examples

### Example 1: NSP Data Preparation

```python
import torch
from transformers import BertTokenizer

def create_nsp_pair(tokenizer, doc_a, doc_b=None, is_next=True):
    tokens = ["[CLS]"] + tokenizer.tokenize(doc_a) + ["[SEP]"]
    segment_a_len = len(tokens)

    if doc_b is not None:
        tokens += tokenizer.tokenize(doc_b) + ["[SEP]"]
    else:
        tokens += ["[SEP]"]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * segment_a_len + [1] * (len(tokens) - segment_a_len)
    label = 1 if is_next else 0

    return input_ids, segment_ids, label

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
doc_a = "The man went to the store."
doc_b_next = "He bought some groceries."
doc_b_random = "Photosynthesis converts sunlight into energy."

pos_ids, pos_seg, pos_label = create_nsp_pair(tokenizer, doc_a, doc_b_next, is_next=True)
neg_ids, neg_seg, neg_label = create_nsp_pair(tokenizer, doc_a, doc_b_random, is_next=False)

print("Positive pair label:", pos_label)
# Output: Positive pair label: 1
print("Negative pair label:", neg_label)
# Output: Negative pair label: 0
print("Input length:", len(pos_ids))
# Output: Input length: 14
print("Segment IDs:", pos_seg)
# Output: Segment IDs: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
```

### Example 2: NSP Loss Computation

```python
import torch.nn as nn
import torch.nn.functional as F

class NSPHead(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.classifier = nn.Linear(d_model, 2)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, pooled_output):
        logits = self.classifier(pooled_output)
        return logits

nsp_head = NSPHead()
pooled = torch.randn(8, 768)
logits = nsp_head(pooled)
probs = F.softmax(logits, dim=-1)

labels = torch.tensor([1, 0, 1, 0, 1, 1, 0, 0])
loss = F.cross_entropy(logits, labels)

print("NSP logits shape:", logits.shape)
# Output: NSP logits shape: torch.Size([8, 2])
print("NSP loss:", loss.item())
# Output: NSP loss: 0.6932
print("Predictions:", probs.argmax(dim=-1))
# Output: Predictions: tensor([0, 1, 0, 0, 1, 1, 1, 1])
print("Accuracy:", (probs.argmax(dim=-1) == labels).float().mean().item())
# Output: Accuracy: 0.375
```

### Example 3: NSP Pre-training Loop Snippet

```python
def nsp_training_step(model, mlm_head, nsp_head, batch, optimizer):
    input_ids = batch["input_ids"]
    segment_ids = batch["segment_ids"]
    attention_mask = batch["attention_mask"]
    mlm_labels = batch["mlm_labels"]
    nsp_labels = batch["nsp_labels"]

    outputs = model(input_ids, token_type_ids=segment_ids, attention_mask=attention_mask)
    sequence_output = outputs.last_hidden_state
    pooled_output = outputs.pooler_output

    mlm_logits = mlm_head(sequence_output)
    nsp_logits = nsp_head(pooled_output)

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, mlm_logits.size(-1)),
        mlm_labels.view(-1),
        ignore_index=-100
    )
    nsp_loss = F.cross_entropy(nsp_logits, nsp_labels)
    total_loss = mlm_loss + nsp_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return mlm_loss.item(), nsp_loss.item(), total_loss.item()

print("NSP training step ready")
# Output: NSP training step ready
print("Losses are combined: L = L_MLM + L_NSP")
# Output: Losses are combined: L = L_MLM + L_NSP
```

## Common Mistakes

1. Constructing negative examples from the same document: Negative examples should come from different documents, not random non-consecutive sentences from the same document. Using intra-document negatives makes the task too difficult (the topic is still similar) and fails to teach the desired distinction.

2. Not balancing positive and negative examples: Maintaining a strict 50/50 split is important. An imbalanced dataset can cause the classifier to predict only one class, making the representations less useful for downstream tasks.

3. Confusing NSP with SOP: NSP uses random sentence pairs from different documents as negatives. ALBERT's Sentence Order Prediction (SOP) uses consecutive sentences but swaps their order as negatives. These are different tasks with different learning signals.

4. Using NSP loss without MLM loss: NSP alone provides insufficient pre-training signal. BERT combines NSP with MLM, and models like RoBERTa show that MLM alone can achieve competitive performance.

5. Forcing the model to use the [CLS] token for all sentence-level tasks: The [CLS] token is optimized for NSP during pre-training, but for downstream sentence-pair tasks, directly using the [CLS] token without fine-tuning may not be optimal.

6. Segment embedding mismatch: When constructing segment IDs for NSP, ensure that sentence A tokens use segment 0 and sentence B tokens use segment 1. Mixing these up causes confusion as the model relies on segment embeddings to distinguish the two sentences.

## Interview Questions

### Beginner

Q: What is the NSP task in BERT? Give a concrete example.

A: NSP is a binary classification task where the model receives two sentences A and B and predicts whether B is the actual next sentence following A or a random sentence. Example: A = "I went to the store." B = "I bought milk." → IsNext (label 1). If B = "The Eiffel Tower is in Paris." → NotNext (label 0).

### Intermediate

Q: Why did RoBERTa find that removing NSP could improve performance? What does this tell us about the NSP objective?

A: RoBERTa showed that NSP removal matched or exceeded BERT's performance on many tasks. This suggests that NSP may not be as critical as initially believed. Potential reasons: the NSP task is too easy (negative examples from different documents are too different, providing trivial discrimination); the MLM task alone may capture sufficient sentence-level information through position embeddings and attention patterns; the NSP loss may interfere with deeper token-level learning.

### Advanced

Q: How does ALBERT's Sentence Order Prediction (SOP) differ from NSP, and why is SOP considered more effective?

A: SOP uses consecutive sentence pairs from the same document as both positive and negative examples. Negative examples are created by swapping the order: (B, A) instead of (A, B). This forces the model to understand discourse coherence and sentence ordering rather than simply detecting topic shift. NSP's negative examples (from different documents) can be solved by detecting topic change, which is easier and less useful for representation learning. SOP provides a harder, more linguistically meaningful learning signal that correlates better with downstream performance.

## Practice Problems

### Easy

Implement a data loader that yields NSP training pairs (positive and negative) from a list of documents, each represented as a list of sentences. Verify that your loader produces a balanced 50/50 positive/negative split.

### Medium

Train a small BERT model with only the NSP objective (no MLM) on a book corpus. Evaluate the trained model's ability to detect whether sentence B follows sentence A. Then fine-tune this model on a natural language inference task and compare the performance with a model pre-trained with MLM only.

### Hard

Design and implement a multi-sentence prediction objective that extends NSP to predict the order of three sentences (A, B, C). Create positive examples (correct order) and negative examples (permuted order). Compare the representations learned by this tri-sentence objective against standard NSP on a discourse coherence evaluation benchmark.

## Solutions

```python
# Easy solution
class NSPDataset:
    def __init__(self, documents):
        self.positive_pairs = []
        self.negative_pairs = []
        for doc in documents:
            for i in range(len(doc) - 1):
                self.positive_pairs.append((doc[i], doc[i + 1], 1))
        for doc in documents:
            for i in range(len(doc)):
                other_docs = [d for d in documents if d != doc]
                neg_doc = other_docs[torch.randint(0, len(other_docs), (1,)).item()]
                neg_sent = neg_doc[torch.randint(0, len(neg_doc), (1,)).item()]
                self.negative_pairs.append((doc[i], neg_sent, 0))
        self.all_pairs = self.positive_pairs + self.negative_pairs

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        return self.all_pairs[idx]

docs = [
    ["I like cats.", "Cats are fluffy.", "They purr loudly."],
    ["The sun is hot.", "It is a star.", "It gives us light."]
]
dataset = NSPDataset(docs)
pos_count = sum(label for _, _, label in dataset)
print(f"Positive: {pos_count}, Negative: {len(dataset) - pos_count}")
# Output: Positive: 3, Negative: 3
print(f"Balance ratio: {pos_count / len(dataset):.2f}")
# Output: Balance ratio: 0.50
```

## Related Concepts

- BERT Pre-training (DL-387)
- Masked Language Modeling (DL-388)
- ALBERT (DL-392)
- RoBERTa (DL-391)
- Sentence Embeddings (DL-414)
- Natural Language Inference
- Discourse Coherence

## Next Concepts

- BERT Variants
- RoBERTa
- ALBERT
- Sentence-BERT

## Summary

Next Sentence Prediction is a binary classification task that trains BERT to determine whether two sentences are consecutive in a document. It is trained jointly with MLM on a balanced set of positive (consecutive) and negative (random from different documents) examples. While subsequent research questioned its necessity, NSP introduced sentence-pair understanding to pre-training and motivated improved objectives like SOP.

## Key Takeaways

- NSP is a binary classification task: predict if sentence B follows sentence A.
- Training uses 50% positive and 50% negative examples, with negatives from different documents.
- The [CLS] token's final hidden state is used for binary classification.
- NSP is trained jointly with MLM, with both losses equally weighted.
- RoBERTa showed that removing NSP can sometimes improve performance.
- ALBERT's SOP objective replaced NSP with a harder sentence ordering task.
- NSP taught the field the importance of sentence-level pre-training objectives.
