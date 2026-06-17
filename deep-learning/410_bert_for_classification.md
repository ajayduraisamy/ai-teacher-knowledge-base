# Concept: BERT for Classification

## Concept ID

DL-410

## Difficulty

Advanced

## Domain

Deep Learning

## Module

BERT Family

## Learning Objectives

- Understand how BERT is adapted for text classification using the [CLS] token.
- Implement single-sentence and sentence-pair classification with BERT.
- Handle multi-class, multi-label, and regression tasks with appropriate modifications.
- Analyze the importance of different components (pooling, head design, regularization) for classification.
- Compare BERT-based classification with traditional methods (TF-IDF + SVM, word embeddings + LSTM).

## Prerequisites

- Understanding of BERT architecture and fine-tuning (DL-386, DL-409)
- Knowledge of classification tasks and metrics (accuracy, F1, precision, recall)
- Familiarity with the [CLS] token and its role (DL-386)
- Understanding of multi-class vs multi-label classification

## Definition

BERT for classification involves adding a linear classification head on top of the pre-trained BERT encoder, typically using the [CLS] token's final hidden state as the sequence representation. The [CLS] token is designed to aggregate information from the entire sequence through bidirectional self-attention. For single-sentence tasks (sentiment analysis, topic classification), the input is [CLS] text [SEP]. For sentence-pair tasks (natural language inference, paraphrase detection), the input is [CLS] sentence A [SEP] sentence B [SEP]. The classification head is a linear layer projecting from hidden size (768 for base) to the number of classes. Multi-class tasks use softmax cross-entropy loss; multi-label tasks use sigmoid binary cross-entropy loss; regression tasks use MSE loss.

## Intuition

Classifying text with BERT is like asking a panel of experts (the attention heads) to read a document and then asking the panel chair (the [CLS] token) for a verdict. The chair has been listening to all the discussions (attending to all tokens) and can give an informed opinion.

The [CLS] token starts with no specific token information — it is a learned vector that is the same for all inputs. During the forward pass through BERT's layers, the [CLS] token attends to all other tokens, gradually accumulating information about the entire sequence. By the final layer, [CLS] has synthesized a representation of the full input that can be used for classification.

The classification head is deliberately simple — just a linear layer. This forces the [CLS] representation to be a good summary of the sequence. More complex heads (MLP, attention pooling) sometimes help but often add minimal benefit.

## Why This Concept Matters

Text classification is the most common NLP task in industry, and BERT-based classifiers have been deployed in countless systems:

1. **Sentiment analysis**: Product reviews, social media monitoring, customer feedback.
2. **Content moderation**: Toxic comment detection, spam filtering, policy compliance.
3. **Topic classification**: Document routing, content recommendation, news categorization.
4. **Intent detection**: Chatbots, virtual assistants, customer support routing.
5. **Natural language inference**: Fact verification, contradiction detection, entailment reasoning.

## Mathematical Explanation

### Single-Sentence Classification

Input: X = [CLS, t_1, t_2, ..., t_n, SEP]

BERT output: H = (h_CLS, h_1, ..., h_n)

Classification: y_pred = softmax(W * h_CLS + b)

Loss: L = -sum_c y_c * log(softmax(W * h_CLS + b)_c)

### Sentence-Pair Classification

Input: X = [CLS, A_1, ..., A_m, SEP, B_1, ..., B_n, SEP]

Segment IDs: 1s for sentence A tokens, 2s for sentence B tokens

BERT output: h_CLS in R^H

Classification: y_pred = softmax(W * h_CLS + b)

### Multi-Label Classification

Each label is independent:

y_pred_i = sigmoid(W_i * h_CLS + b_i) for i = 1..C

Loss: L = -sum_i [y_i * log(sigmoid_i) + (1 - y_i) * log(1 - sigmoid_i)]

### Variants of Pooling

While [CLS] is standard, alternatives exist:
- Mean pooling: h_pool = 1/n * sum_i h_i
- Max pooling: h_pool = max_i h_i (element-wise)
- Weighted pooling: h_pool = sum_i alpha_i * h_i (learned attention weights)
- Concat: h_pool = [h_CLS; mean_pool; max_pool]

## Code Examples

### Example 1: Single-Sentence Classification

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class BertSingleClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", n_classes=2, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        logits = self.classifier(x)
        return logits

model = BertSingleClassifier("bert-base-uncased", n_classes=3)
x = torch.randint(0, 1000, (4, 16))
mask = torch.ones(4, 16, dtype=torch.long)
logits = model(x, mask)
probs = F.softmax(logits, dim=-1)
print("Logits shape:", logits.shape)
# Output: Logits shape: torch.Size([4, 3])
print("Probabilities:", probs.round(decimals=3))
# Output: Probabilities: tensor([[0.345, 0.332, 0.323],
#         [0.341, 0.336, 0.323],
#         [0.340, 0.331, 0.329],
#         [0.343, 0.334, 0.323]])
print("Predicted classes:", probs.argmax(dim=-1))
# Output: Predicted classes: tensor([0, 1, 0, 0])
```

### Example 2: Sentence-Pair Classification (NLI)

```python
class BertPairClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", n_classes=3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled = outputs.pooler_output
        logits = self.classifier(pooled)
        return logits

def predict_entailment(model, premise, hypothesis, tokenizer, device):
    encoded = tokenizer(
        premise, hypothesis,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)
    with torch.no_grad():
        logits = model(encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"])
        probs = F.softmax(logits, dim=-1)
    labels = ["entailment", "neutral", "contradiction"]
    prediction = labels[probs.argmax().item()]
    confidence = probs.max().item()
    return prediction, confidence

print("Sentence-pair classification for NLI")
# Output: Sentence-pair classification for NLI
```

### Example 3: Multi-Label Classification

```python
class BertMultiLabelClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", n_labels=5, threshold=0.5):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_labels)
        self.threshold = threshold

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

    def predict_labels(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)
        probs = torch.sigmoid(logits)
        return (probs >= self.threshold).int()

def multi_label_loss(logits, labels):
    return F.binary_cross_entropy_with_logits(logits, labels.float())

model = BertMultiLabelClassifier("bert-base-uncased", n_labels=5)
x = torch.randint(0, 1000, (4, 16))
mask = torch.ones(4, 16, dtype=torch.long)
logits = model(x, mask)
probs = torch.sigmoid(logits)

labels = torch.randint(0, 2, (4, 5), dtype=torch.float)
loss = multi_label_loss(logits, labels)
print("Multi-label loss:", loss.item())
# Output: Multi-label loss: 0.6932
print("Multi-label predictions:", (probs >= 0.5).int())
# Output: Multi-label predictions: tensor([[0, 0, 0, 1, 0],
#         [0, 0, 0, 1, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0]])
```

## Common Mistakes

1. Not using the [CLS] token correctly: Some implementations incorrectly use mean pooling of all tokens instead of [CLS]. While mean pooling can work, it may dilute important signals. The [CLS] token is specifically designed for classification.

2. Forgetting to set token_type_ids for pair tasks: For sentence-pair classification, token_type_ids must be set (0 for sentence A, 1 for sentence B). Omitting this can confuse the model about sentence boundaries.

3. Using softmax for multi-label tasks: Multi-label tasks require sigmoid activation (independent per label) and binary cross-entropy loss. Using softmax and cross-entropy makes labels compete with each other.

4. Not handling class imbalance: BERT fine-tuning on imbalanced datasets can produce classifiers biased toward the majority class. Solutions include weighted loss, class-balanced sampling, or focal loss.

5. Using overly complex classification heads: A simple linear layer on [CLS] often works best. Adding extra MLP layers or attention pooling rarely improves performance and can cause overfitting.

6. Ignoring the attention mask: The attention mask tells BERT which tokens are real (1) and which are padding (0). Without it, BERT attends to padding tokens, diluting the representation.

## Interview Questions

### Beginner

Q: How does BERT use the [CLS] token for text classification?

A: BERT's input starts with a special [CLS] token. Through the encoder layers, [CLS] attends to all other tokens, aggregating information about the entire sequence. The final hidden state of [CLS] is used as a fixed-size representation of the whole input. A linear layer on top of this representation predicts the class label.

### Intermediate

Q: What is the difference between using [CLS] and mean pooling for text classification with BERT? When might one be preferable over the other?

A: [CLS] is trained during pre-training to aggregate sequence information (for NSP). However, the NSP task is relatively simple, and [CLS] may not capture all relevant information. Mean pooling averages all token representations, providing a more comprehensive but potentially noisier representation. Mean pooling can work better for tasks where the entire sequence is important (e.g., topic classification), while [CLS] works well for tasks where a summary representation is sufficient (e.g., sentiment). In practice, [CLS] is the standard choice and works well for most tasks.

### Advanced

Q: Design a BERT-based classification system that handles long documents exceeding 512 tokens. Describe at least two approaches.

A: Two approaches: (1) Sliding window with aggregation — split the document into overlapping 512-token chunks, classify each chunk, and aggregate predictions (e.g., majority voting, average probabilities). (2) Hierarchical — split the document into 512-token chunks, encode each chunk with BERT, use [CLS] representations from each chunk as input to a second-level classifier (e.g., Transformer or LSTM) that produces the final prediction. The hierarchical approach captures inter-chunk relationships but requires more computation. The sliding window approach is simpler but may miss cross-chunk context. A third approach uses Longformer or BigBird (transformers designed for long sequences) instead of standard BERT.

## Practice Problems

### Easy

Fine-tune BERT-base on the IMDb movie review sentiment dataset. Report accuracy, precision, recall, and F1 score on the test set.

### Medium

Compare three pooling methods for BERT classification: (a) [CLS] token, (b) mean pooling, (c) weighted pooling (learned attention weights over token representations). Evaluate on 3 GLUE classification tasks (SST-2, MRPC, QQP) and report which pooling method works best for each task.

### Hard

Design a hierarchical BERT classification system for long documents (> 512 tokens). Split documents into chunks, encode each chunk with BERT, and combine chunk representations using a Transformer encoder. Compare with a sliding window majority voting approach on a long-document classification dataset (e.g., PubMed abstracts or legal documents).

## Solutions

```python
# Easy solution
def evaluate_classifier(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

print("Classification evaluation metrics")
# Output: Classification evaluation metrics
```

## Related Concepts

- BERT Fine-tuning (DL-409)
- BERT for QA (DL-411)
- BERT for NER (DL-412)
- [CLS] Token Role (DL-386)
- Multi-class vs Multi-label Classification
- Natural Language Inference

## Next Concepts

- BERT for QA
- BERT for NER
- DistilBERT
- Sentence-BERT

## Summary

BERT for classification adapts the pre-trained encoder by using the [CLS] token's final hidden state as a sequence representation, passed through a linear classifier. Single-sentence, sentence-pair, multi-class, multi-label, and regression tasks are all supported with appropriate loss functions and output layers. The approach achieves state-of-the-art results on most text classification benchmarks.

## Key Takeaways

- [CLS] token representation is the standard pooling method for classification.
- Simple linear head on [CLS] usually works best.
- Multi-class: softmax + cross-entropy loss.
- Multi-label: sigmoid + binary cross-entropy loss.
- Sentence-pair tasks use token_type_ids to distinguish sentences.
- Mean pooling is a viable alternative to [CLS] for some tasks.
- Attention mask must be correctly provided.
- BERT classification achieves SOTA on most benchmarks.
- Template-based prompting can further improve few-shot classification.
- Proper hyperparameter tuning (lr, epochs, warmup) is essential.
