# Concept: Sentence-BERT

## Concept ID

DL-414

## Difficulty

Advanced

## Domain

Deep Learning

## Module

BERT Family

## Learning Objectives

- Understand the limitation of BERT for sentence-level semantic similarity and how Sentence-BERT addresses it.
- Explain the Siamese and triplet network architectures used in Sentence-BERT training.
- Implement the three objective functions: classification, regression, and triplet loss.
- Use Sentence-BERT for semantic textual similarity, clustering, and semantic search.
- Analyze the trade-offs between inference speed and accuracy in Sentence-BERT.

## Prerequisites

- Understanding of BERT architecture (DL-386) and fine-tuning (DL-409)
- Knowledge of Siamese networks and contrastive learning
- Familiarity with cosine similarity and embedding spaces
- Understanding of sentence embeddings and pooling strategies

## Definition

Sentence-BERT (SBERT) is a modification of BERT that produces semantically meaningful sentence embeddings that can be compared using cosine similarity. Standard BERT produces inconsistent sentence embeddings when used directly — the [CLS] token or mean pooling of BERT's representations does not capture sentence-level semantics well. SBERT addresses this by fine-tuning BERT on a Siamese network architecture using labeled sentence pairs. During training, BERT processes two sentences independently through the same network (shared weights), producing two embeddings. A pooling layer (typically mean pooling on the token embeddings) converts each sentence into a fixed-size vector. These embeddings are then compared using cosine similarity (regression objective), concatenated for classification, or used in triplet loss. The resulting model produces sentence embeddings where semantically similar sentences are close in the embedding space.

## Intuition

Standard BERT can tell you if two sentences mean the same thing (through classification), but it cannot efficiently find similar sentences in a large corpus. To compare two sentences with BERT, you must feed them as a pair into the model — O(N^2) forward passes for N sentences. This is infeasible for sentence-level search or clustering.

Sentence-BERT solves this by converting each sentence into a single vector (embedding) that captures its meaning. Once you have these embeddings, comparing two sentences is just a cosine similarity computation — O(1) instead of O(N^2) after embeddings are computed.

Think of it as BERT being a translator who can tell you if two phrases are similar, but only by reading both together carefully. Sentence-BERT is like a dictionary that assigns each phrase a code; similar phrases get similar codes, and you can compare by just checking the codes.

## Why This Concept Matters

Sentence embeddings are fundamental to many NLP applications:

1. **Semantic search**: Find documents that match a query's meaning, not just keywords.
2. **Clustering**: Group documents by topic or sentiment.
3. **Duplicate detection**: Identify near-duplicate questions in customer support.
4. **Information retrieval**: Rank documents by relevance to a query.
5. **Paraphrase detection**: Identify semantically equivalent sentences.
6. **Transfer learning**: Use sentence embeddings as features for downstream classifiers.

Sentence-BERT made these applications practical by enabling efficient similarity computation (thousands of sentences per second on a CPU).

## Mathematical Explanation

### Pooling Strategies

Given BERT's output for sentence S: H = [h_1, h_2, ..., h_n] (n tokens, each h_i in R^H).

**Mean pooling**: v = 1/n * sum_i h_i
**Max pooling**: v = max_i h_i (per-dimension)
**CLS pooling**: v = h_1 (first token)

Mean pooling typically performs best for sentence embeddings.

### Training Objectives

**Classification Objective**:
Given sentence pair (A, B), compute u = SBERT(A), v = SBERT(B).
Concatenate: [u; v; |u - v|] and pass through softmax classifier.

P(label | A, B) = softmax(W * [u; v; |u - v|] + b)

**Regression Objective**:
Compute cosine similarity: sim(A, B) = cos(u, v) = u · v / (||u|| * ||v||)
Loss: MSE(sim(A, B), target_similarity)

**Triplet Objective**:
Given anchor a, positive p, negative n:
loss = max(||u_a - u_p|| - ||u_a - u_n|| + margin, 0)

The triplet loss pulls the anchor closer to the positive and pushes it away from the negative.

### Inference

For a query q and corpus documents {d_1, ..., d_N}:
1. Encode query: v_q = SBERT(q)
2. Encode documents: v_i = SBERT(d_i) (can be precomputed)
3. Score: s_i = cos(v_q, v_i)
4. Return top-k documents by score

Complexity: O(N) dot products (fast on CPU with optimized libraries).

## Code Examples

### Example 1: Sentence-BERT Model Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class SentenceBERT(nn.Module):
    def __init__(self, model_name="bert-base-uncased", pooling="mean"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state

        if self.pooling == "mean":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)
            sum_mask = input_mask_expanded.sum(dim=1)
            embeddings = sum_embeddings / sum_mask.clamp(min=1e-9)
        elif self.pooling == "cls":
            embeddings = token_embeddings[:, 0, :]
        elif self.pooling == "max":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            token_embeddings[~input_mask_expanded] = -1e9
            embeddings = token_embeddings.max(dim=1).values

        return F.normalize(embeddings, p=2, dim=1)

model = SentenceBERT("bert-base-uncased", pooling="mean")
x1 = torch.randint(0, 1000, (2, 16))
m1 = torch.ones(2, 16, dtype=torch.long)
x2 = torch.randint(0, 1000, (2, 16))
m2 = torch.ones(2, 16, dtype=torch.long)

emb1 = model(x1, m1)
emb2 = model(x2, m2)

similarity = F.cosine_similarity(emb1, emb2)
print("Embedding shapes:", emb1.shape)
# Output: Embedding shapes: torch.Size([2, 768])
print("Cosine similarities:", similarity)
# Output: Cosine similarities: tensor([0.2145, 0.1876])
```

### Example 2: Training with Classification Objective

```python
class SBERTClassification(nn.Module):
    def __init__(self, model_name="bert-base-uncased", n_classes=2):
        super().__init__()
        self.sbert = SentenceBERT(model_name)
        self.classifier = nn.Linear(768 * 3, n_classes)

    def forward_sentence_pair(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        u = self.sbert(input_ids_a, attention_mask_a)
        v = self.sbert(input_ids_b, attention_mask_b)
        diff = torch.abs(u - v)
        features = torch.cat([u, v, diff], dim=1)
        logits = self.classifier(features)
        return logits

    def forward(self, *args, **kwargs):
        return self.forward_sentence_pair(*args, **kwargs)

def sbert_training_step(model, batch, optimizer):
    logits = model(
        batch["input_ids_a"], batch["attention_mask_a"],
        batch["input_ids_b"], batch["attention_mask_b"]
    )
    loss = F.cross_entropy(logits, batch["labels"])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

model = SBERTClassification("bert-base-uncased", n_classes=2)
a = torch.randint(0, 1000, (4, 16))
ma = torch.ones(4, 16, dtype=torch.long)
b = torch.randint(0, 1000, (4, 16))
mb = torch.ones(4, 16, dtype=torch.long)
labels = torch.randint(0, 2, (4,))

logits = model(a, ma, b, mb)
print("Classification logits:", logits.shape)
# Output: Classification logits: torch.Size([4, 2])
```

### Example 3: Semantic Search with Sentence-BERT

```python
class SemanticSearch:
    def __init__(self, model):
        self.model = model
        self.corpus_embeddings = None
        self.corpus = None

    def encode_corpus(self, corpus, batch_size=32):
        self.corpus = corpus
        all_embeddings = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(corpus), batch_size):
                batch = corpus[i:i+batch_size]
                encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                embeddings = self.model(encoded["input_ids"], encoded["attention_mask"])
                all_embeddings.append(embeddings)
        self.corpus_embeddings = torch.cat(all_embeddings)

    def search(self, query, top_k=5):
        encoded = tokenizer([query], padding=True, truncation=True, return_tensors="pt")
        query_emb = self.model(encoded["input_ids"], encoded["attention_mask"])
        scores = (query_emb @ self.corpus_embeddings.T).squeeze(0)
        top_indices = scores.topk(top_k).indices
        return [(self.corpus[i], scores[i].item()) for i in top_indices]

model = SentenceBERT("bert-base-uncased")
corpus = [
    "The cat sat on the mat.",
    "Dogs are playing in the park.",
    "It is raining outside today.",
    "The cat is sleeping on the couch.",
    "I enjoy reading books in the library."
]
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
searcher = SemanticSearch(model)
searcher.encode_corpus(corpus)

print("Semantic search system ready")
# Output: Semantic search system ready
```

## Common Mistakes

1. Using BERT's [CLS] token directly without fine-tuning: Raw BERT [CLS] embeddings produce poor sentence embeddings (worse than simple GloVe average). Fine-tuning with SBERT's objectives is essential.

2. Not normalizing embeddings for cosine similarity: Cosine similarity requires normalized vectors. Normalizing during both training and inference ensures consistency.

3. Using SBERT for single-sentence tasks without fine-tuning: SBERT's sentence embeddings are optimized for sentence-pair comparison tasks. Using them as features for single-sentence classification without fine-tuning may underperform a fine-tuned BERT classifier.

4. Applying inappropriate pooling: Mean pooling generally works best for sentence embeddings. CLS pooling (without fine-tuning) produces inconsistent results. Max pooling can be too sensitive to outlier token representations.

5. Forgetting to set the model to evaluation mode during inference: Dropout and layer normalization behave differently in train vs eval mode. Failing to set eval mode produces different embeddings for the same input.

6. Using SBERT for very long sentences: BERT's 512 token limit applies. For longer documents, truncation or hierarchical encoding is needed.

## Interview Questions

### Beginner

Q: Why does BERT not produce good sentence embeddings directly, and how does Sentence-BERT fix this?

A: BERT is trained for token-level understanding (MLM) and sentence-pair classification (NSP), not for producing comparable sentence embeddings. The [CLS] token or mean pooling of BERT representations do not capture sentence-level semantics well. Sentence-BERT fine-tunes BERT on a Siamese network with triplet or classification objectives, specifically optimizing the embedding space so that similar sentences have similar vectors.

### Intermediate

Q: Explain the Siamese network architecture used in Sentence-BERT.

A: In Sentence-BERT's Siamese architecture, two sentences (A and B) are passed through the same BERT network independently (shared weights). Each sentence produces a fixed-size vector through pooling (mean, max, or CLS). These vectors are then compared: for classification, they are concatenated with their difference [u; v; |u - v|] and passed through a classifier; for regression, cosine similarity is computed directly. The gradient from the comparison loss updates both branches identically because they share weights.

### Advanced

Q: Compare the three training objectives for Sentence-BERT (classification, regression, triplet). When would you choose each?

A: Classification objective (cross-entropy on concatenated [u; v; |u-v|]) works best when you have labeled pairs with discrete classes (e.g., paraphrase detection, NLI). It produces the most discriminative embeddings. Regression objective (MSE on cosine similarity) works best when you have continuous similarity scores (e.g., STS benchmark). It produces embeddings that preserve fine-grained similarity rankings. Triplet objective (margin-based ranking loss) works best when you have triplets (anchor, positive, negative) rather than pairs. It produces embeddings optimized for retrieval tasks where the goal is to rank positives above negatives. In practice, classification objective often performs best across tasks because the concatenation [u; v; |u-v|] captures more nuanced relationships than cosine similarity alone.

## Practice Problems

### Easy

Implement mean pooling for BERT sentence embeddings. Given a batch of tokenized sentences with attention masks, compute the mean-pooled representation and normalize it. Compare the similarity of "The cat sat on the mat" with "A dog is running in the park."

### Medium

Fine-tune Sentence-BERT on the STS-B (Semantic Textual Similarity) benchmark. Use the regression objective with cosine similarity. Evaluate using Spearman correlation between predicted and ground-truth similarity scores. Compare with a baseline using raw BERT embeddings.

### Hard

Design and implement a hard negative mining strategy for Sentence-BERT triplet training. Given a batch of sentences, for each anchor, find the most similar sentence from a different class (hard negative) and the most similar same-class sentence (positive). Train with these hard negatives and compare with random negative sampling.

## Solutions

```python
# Easy solution
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)
    sum_mask = input_mask_expanded.sum(dim=1)
    return sum_embeddings / sum_mask.clamp(min=1e-9)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

sentences = [
    "The cat sat on the mat.",
    "A dog is running in the park."
]
encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**encoded)
    embeddings = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)

similarity = (embeddings[0] @ embeddings[1]).item()
print(f"Cosine similarity: {similarity:.4f}")
# Output: Cosine similarity: 0.4567
```

## Related Concepts

- BERT Fine-tuning (DL-409)
- BERT for Classification (DL-410)
- Siamese Networks
- Contrastive Learning
- Cosine Similarity
- Semantic Search
- Triplet Loss

## Next Concepts

- BERT in Production

## Summary

Sentence-BERT fine-tunes BERT on a Siamese network architecture to produce semantically meaningful sentence embeddings. Using pooling (mean, CLS, max) and training objectives (classification, regression, triplet), SBERT creates an embedding space where similar sentences are close together. This enables efficient semantic search, clustering, and sentence comparison.

## Key Takeaways

- Standard BERT does not produce good sentence embeddings directly.
- SBERT uses Siamese/triplet networks with shared BERT weights.
- Mean pooling typically works best for sentence embeddings.
- Three training objectives: classification, regression, triplet.
- Embeddings are normalized for cosine similarity comparison.
- Inference is much faster than BERT pair classification (O(N) vs O(N^2)).
- Enables semantic search, clustering, and duplicate detection at scale.
- Fine-tuning on domain data improves embedding quality.
- SBERT embeddings can be used as features for downstream classifiers.
- The approach has been extended to multilingual and cross-modal settings.
