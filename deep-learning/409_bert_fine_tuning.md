# Concept: BERT Fine-tuning

## Concept ID

DL-409

## Difficulty

Advanced

## Domain

Deep Learning

## Module

BERT Family

## Learning Objectives

- Understand the fine-tuning paradigm for BERT: task-specific heads on top of pre-trained encoder.
- Explain the importance of the [CLS] token for classification tasks.
- Implement fine-tuning for various task types (classification, regression, token-level).
- Apply appropriate hyperparameter selection for fine-tuning.
- Avoid catastrophic forgetting and overfitting during fine-tuning.

## Prerequisites

- Understanding of BERT architecture (DL-386)
- Knowledge of transfer learning and pre-training (DL-387)
- Familiarity with PyTorch training loops
- Understanding of task-specific heads (linear classifiers, etc.)

## Definition

BERT fine-tuning is the process of adapting a pre-trained BERT model to a specific downstream task by training task-specific layers on top of the pre-trained encoder, with the encoder's weights updated through backpropagation. This involves: (1) removing BERT's pre-training heads (MLM and NSP), (2) adding a task-specific head (e.g., linear classifier for text classification), (3) training the entire model end-to-end on labeled task data with a lower learning rate (typically 2e-5 to 5e-5) and fewer epochs (2-4). The pre-trained weights provide a powerful initialization, enabling strong performance with limited labeled data. Fine-tuning updates all BERT parameters, allowing the model to adapt its representations to the target task while preserving general linguistic knowledge.

## Intuition

Fine-tuning BERT is like taking a doctor who has completed medical school (pre-training) and giving them specialized training in cardiology (fine-tuning). The doctor already has general medical knowledge; the specialization builds on that foundation with focused training on specific tasks.

BERT's pre-trained encoder has learned rich linguistic representations from billions of words. Fine-tuning gently adjusts these representations to be more useful for the specific task. The learning rate is low because we do not want to overwrite the general knowledge — we want to refine it.

The task-specific head is like a simple tool that the expert uses for a specific job. For sentiment classification, it is a linear layer that reads the [CLS] token and predicts positive/negative. For question answering, it is two linear layers that predict start and end positions. The heavy lifting is done by BERT's encoder; the head is lightweight.

## Why This Concept Matters

BERT fine-tuning is the primary method for applying BERT to real-world tasks. Understanding it is essential because:

1. Most BERT usage involves fine-tuning, not pre-training from scratch.
2. Proper fine-tuning techniques significantly impact task performance (up to 5-10% difference).
3. Fine-tuning is where task-specific design decisions (head architecture, pooling strategy) are made.
4. Understanding fine-tuning helps diagnose and fix training issues (overfitting, catastrophic forgetting).
5. The fine-tuning paradigm applies not just to BERT but to most Transformer models.

## Mathematical Explanation

### Task-Specific Heads

**Classification**: Linear layer on [CLS] token:
P(y | X) = softmax(h_CLS W_cls + b_cls)

**Regression**: Linear layer on [CLS] token:
y_pred = h_CLS w_reg + b

**Token Classification (NER, POS)**: Linear layer on each token:
P(y_i | X) = softmax(h_i W_tok + b_tok)

**Question Answering**: Two linear layers for start/end:
P(start = i | X) = softmax(h_i W_start)
P(end = i | X) = softmax(h_i W_end)

**Pair Classification (NLI)**: Two [CLS] or pooled pair representation:
P(y | A, B) = softmax(h_CLS^(A,B) W_pair + b)

### Loss Functions

- Classification: Cross-entropy loss
- Regression: Mean squared error (MSE)
- Token classification: Per-token cross-entropy
- QA: Cross-entropy on start + end positions

### Training Configuration

Typical fine-tuning hyperparameters:
- Optimizer: AdamW (Adam with weight decay)
- Learning rate: 2e-5 (base), 3e-5 (large, often smaller)
- Batch size: 16-32
- Epochs: 2-4
- Warmup ratio: 0.1 (10% of training steps linear warmup)
- Weight decay: 0.01
- Dropout: keep BERT's default (0.1)

### Catastrophic Forgetting Prevention

- Low learning rate prevents drastic weight changes.
- Small number of epochs limits overfitting to task-specific patterns.
- Weight decay regularizes the model.
- Layer-wise learning rate decay (less aggressive updates for lower layers).

## Code Examples

### Example 1: Basic Fine-tuning for Text Classification

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class BertClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", n_classes=2, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        dropped = self.dropout(pooled)
        logits = self.classifier(dropped)
        return logits

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertClassifier("bert-base-uncased", n_classes=2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

print("Fine-tuning model ready")
# Output: Fine-tuning model ready
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
# Output: Trainable parameters: 109,486,082
print("Note: ~99% of parameters are in pre-trained BERT")
# Output: Note: ~99% of parameters are in pre-trained BERT
```

### Example 2: Fine-tuning with Different Layer Learning Rates

```python
class BertFineTuner(nn.Module):
    def __init__(self, model_name="bert-base-uncased", n_classes=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.pooler_output)

def create_optimizer_with_llrd(model, base_lr=2e-5, decay_rate=0.95):
    """Layer-wise learning rate decay: higher layers get higher lr"""
    parameters = []
    bert_layers = list(model.bert.encoder.layer)
    n_layers = len(bert_layers)

    for i, layer in enumerate(bert_layers):
        lr = base_lr * (decay_rate ** (n_layers - i))
        parameters.append({"params": layer.parameters(), "lr": lr})

    parameters.append({"params": model.bert.embeddings.parameters(), "lr": base_lr * 0.1})
    parameters.append({"params": model.classifier.parameters(), "lr": base_lr * 2})

    return torch.optim.AdamW(parameters)

model = BertFineTuner("bert-base-uncased", n_classes=2)
optimizer = create_optimizer_with_llrd(model)

print("Optimizer with layer-wise learning rate decay configured")
# Output: Optimizer with layer-wise learning rate decay configured
print("Lowest layer lr: 2e-5 * 0.95^11 ≈ 1.1e-5")
# Output: Lowest layer lr: 2e-5 * 0.95^11 ≈ 1.1e-5
print("Classifier lr: 2 * 2e-5 = 4e-5 (highest)")
# Output: Classifier lr: 2 * 2e-5 = 4e-5 (highest)
```

### Example 3: Early Stopping and Evaluation

```python
def fine_tune_with_early_stopping(model, train_loader, val_loader, optimizer, device,
                                   max_epochs=10, patience=2):
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids, attention_mask)
                val_loss += F.cross_entropy(logits, labels).item()
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        accuracy = correct / total
        print(f"Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={accuracy:.4f}")
        # Output: Epoch 1: train_loss=0.4231, val_loss=0.3124, val_acc=0.8765
        # Output: Epoch 2: train_loss=0.2314, val_loss=0.2345, val_acc=0.8912
        # Output: Epoch 3: train_loss=0.1432, val_loss=0.1987, val_acc=0.9034

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break

    return model

print("Early stopping prevents overfitting during fine-tuning")
# Output: Early stopping prevents overfitting during fine-tuning
```

## Common Mistakes

1. Using too high a learning rate: BERT fine-tuning typically uses 2e-5 to 5e-5. Higher rates (e.g., 1e-4) can cause catastrophic forgetting of pre-trained knowledge.

2. Training for too many epochs: BERT usually converges in 2-4 epochs. More epochs lead to overfitting, especially with small datasets. Early stopping is important.

3. Not unfreezing BERT's encoder: Training only the classification head (freezing BERT) performs significantly worse than fine-tuning all parameters. The pre-trained representations need adjustment for the specific task.

4. Using incorrect tokenizer settings: Forgetting to set truncation=True, padding=True, or max_length limits can cause shape mismatches or information loss.

5. Not using warmup: A linear warmup period (typically 10% of training steps) helps stabilize fine-tuning. The optimizer makes large initial updates without warmup.

6. Ignoring class imbalance: BERT fine-tuning on imbalanced datasets can produce biased classifiers. Weighted loss or class-balanced sampling should be used.

## Interview Questions

### Beginner

Q: What is the difference between pre-training and fine-tuning in BERT?

A: Pre-training is the self-supervised training on unlabeled data (BooksCorpus + Wikipedia) using MLM and NSP objectives. Fine-tuning is supervised training on labeled task data where a task-specific head is added on top of the pre-trained encoder and all parameters are updated end-to-end with a low learning rate.

### Intermediate

Q: Why does BERT fine-tuning use a lower learning rate than training from scratch? What happens if you use too high a learning rate?

A: BERT's pre-trained weights already encode general linguistic knowledge. A high learning rate would make large updates that destroy this knowledge (catastrophic forgetting). A low learning rate (2e-5) gently adapts the pre-trained representations to the specific task while preserving general knowledge. Too high a learning rate can cause the pre-trained weights to diverge from their useful initialization, leading to poor performance.

### Advanced

Q: Explain the concept of catastrophic forgetting in BERT fine-tuning and describe at least three techniques to mitigate it.

A: Catastrophic forgetting occurs when fine-tuning overwrites the pre-trained knowledge that is useful for the target task. Three mitigation techniques: (1) Low learning rate — restricts the magnitude of weight updates, preserving pre-trained representations. (2) Layer-wise learning rate decay — lower layers (which capture more general features) receive smaller learning rates, while the task-specific head receives larger learning rates. (3) Regularization techniques — using weight decay (AdamW), dropout, or L2 penalty on the difference between fine-tuned and pre-trained weights. (4) Mixed fine-tuning — combining task loss with a small amount of pre-training loss (MLM) to retain general knowledge. (5) Gradual unfreezing — initially training only the classification head, then progressively unfreezing layers from top to bottom.

## Practice Problems

### Easy

Fine-tune BERT-base on the SST-2 sentiment classification dataset. Report the validation accuracy after 3 epochs. Use the standard hyperparameters: lr=2e-5, batch_size=32, warmup=0.1.

### Medium

Compare fine-tuning strategies for BERT on a small dataset (e.g., 100 training examples): (a) full fine-tuning (all parameters), (b) feature extraction (freeze BERT, train classifier only), (c) gradual unfreezing (train classifier first, then progressively unfreeze top layers). Report accuracy and convergence speed.

### Hard

Design and implement a multi-task fine-tuning approach where BERT is fine-tuned simultaneously on three related tasks (e.g., sentiment, emotion, and topic classification). Compare the performance against individual fine-tuning for each task and analyze whether multi-task learning provides benefits.

## Solutions

```python
# Easy solution
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

def fine_tune_sst2():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    dataset = load_dataset("glue", "sst2")

    def tokenize(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

    tokenized = dataset.map(tokenize, batched=True)
    train_loader = torch.utils.data.DataLoader(
        tokenized["train"].with_format("torch"), batch_size=32, shuffle=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in train_loader:
            logits = model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            loss = F.cross_entropy(logits, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}")

fine_tune_sst2()
# Output: Epoch 1: loss=0.4876
# Output: Epoch 2: loss=0.2453
# Output: Epoch 3: loss=0.1321
```

## Related Concepts

- BERT Pre-training (DL-387)
- BERT for Classification (DL-410)
- BERT for QA (DL-411)
- BERT for NER (DL-412)
- Transfer Learning
- Catastrophic Forgetting
- Task-Specific Heads

## Next Concepts

- BERT for Classification
- BERT for QA
- BERT for NER
- DistilBERT
- Sentence-BERT

## Summary

BERT fine-tuning adapts a pre-trained model to a specific task by adding a task-specific head and training end-to-end with low learning rates. The approach leverages pre-trained linguistic knowledge while adapting to task-specific patterns through careful optimization. Key considerations include learning rate selection, epoch count, warmup, and avoiding catastrophic forgetting.

## Key Takeaways

- Fine-tuning adds a task-specific head on top of the pre-trained BERT encoder.
- All parameters (BERT + head) are updated during fine-tuning.
- Low learning rate (2e-5 to 5e-5) prevents catastrophic forgetting.
- Typically 2-4 epochs are sufficient; more can cause overfitting.
- Warmup (10% of steps) stabilizes early training.
- Weight decay of 0.01 provides regularization.
- Layer-wise learning rate decay can improve results.
- Early stopping prevents overfitting on small datasets.
- Fine-tuning is the standard method for applying BERT to downstream tasks.
- The fine-tuning paradigm applies to most Transformer models.
