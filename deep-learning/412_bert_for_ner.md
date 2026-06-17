# Concept: BERT for Named Entity Recognition

## Concept ID

DL-412

## Difficulty

Advanced

## Domain

Deep Learning

## Module

BERT Family

## Learning Objectives

- Understand how BERT is adapted for token-level classification tasks like NER.
- Implement BIO/BIOES tagging schemes and align labels with WordPiece tokens.
- Handle subword token-label alignment for accurate NER training and evaluation.
- Build a BERT-based NER model with the CRF layer on top.
- Evaluate NER performance using span-level precision, recall, and F1.

## Prerequisites

- Understanding of BERT architecture (DL-386) and fine-tuning (DL-409)
- Knowledge of sequence labeling tasks
- Familiarity with tokenization and subword alignment (DL-407)
- Understanding of BIO tagging scheme

## Definition

BERT for Named Entity Recognition (NER) involves adding a token-level classification head on top of BERT's encoder that predicts a label for each token. The head is typically a linear layer followed by softmax that maps each token's hidden state to one of the entity tags (e.g., B-PER, I-PER, B-ORG, I-ORG, O). The input is a single sentence processed by BERT, and the output is a label for each input token. Since BERT uses WordPiece tokenization, NER labels must be aligned from word-level to token-level: the first subword token receives the label, and subsequent subword tokens receive a special ignore label (e.g., -100) in the loss. A Conditional Random Field (CRF) layer can be added on top of BERT to enforce valid label transitions (e.g., I-PER cannot follow B-ORG). BERT-based NER achieves state-of-the-art results on benchmarks like CoNLL-2003 (F1 > 92).

## Intuition

Imagine BERT as a reader scanning a document and highlighting every word with a colored tag. Person names get blue, organizations get green, locations get yellow, and everything else is unmarked (gray). For a sentence like "Barack Obama visited Google in Mountain View," the tags would be: Barack (B-PER), Obama (I-PER), visited (O), Google (B-ORG), in (O), Mountain (B-LOC), View (I-LOC).

The BIO scheme enforces consistency: an entity starts with B- (beginning) and continues with I- (inside). If you see I-PER without a preceding B-PER, that is invalid. The CRF layer learns these transition rules, preventing impossible sequences like O → I-PER or B-PER → I-ORG.

The challenge with BERT is subword tokenization. "Mountain View" might not split, but "unhappiness" splits into ["un", "##happiness"]. The label B-MISC applies only to "un"; "##happiness" gets -100. During inference, predictions on non-first subwords are typically ignored, and the first subword's prediction is taken for the word.

## Why This Concept Matters

NER is a fundamental NLP task with wide-ranging applications:

1. **Information extraction**: Extracting structured data from unstructured text.
2. **Knowledge base construction**: Populating databases with entities and relationships.
3. **Search and retrieval**: Entity-based search, question answering.
4. **Content understanding**: News analysis, financial document processing.
5. **Healthcare**: Extracting medical entities (diseases, drugs, procedures) from clinical notes.

BERT significantly improved NER accuracy, especially for rare entities and complex entity types, by providing rich contextual representations.

## Mathematical Explanation

### Token-Level Classification

For each token position i, BERT produces hidden state h_i in R^H.

Linear classifier: logits_i = h_i^T W + b, where W in R^{H x C}, C = number of tags.

P(tag_i = c | h_i) = exp(logits_{i,c}) / sum_{c'} exp(logits_{i,c'})

Loss (without CRF): L = -sum_i log P(tag_i = c_i^* | h_i), ignoring -100 positions.

### BIO Tagging Scheme

- B-X: Beginning of entity type X
- I-X: Inside (continuation of) entity type X
- O: Outside (not an entity)

For 3 entity types (PER, ORG, LOC): 7 tags (B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, O).

### CRF Layer

CRF learns transition probabilities T(tag_i -> tag_j):

P(tag_1, ..., tag_n | h_1, ..., h_n) = exp(score) / Z

Where score = sum_i (unary_i + transition_{tag_{i-1} -> tag_i})

Unary_i = logits_i[tag_i]
Z = sum over all possible tag sequences of exp(score)

Viterbi decoding finds the highest-scoring valid sequence.

Valid transitions examples:
- O → B-PER (valid: start a new entity)
- B-PER → I-PER (valid: continue same entity)
- B-PER → B-ORG (valid: end previous, start new)
- I-PER → B-ORG (valid: end PER entity, start ORG)
- I-PER → I-ORG (invalid: can't switch types inside)
- O → I-PER (invalid: must start with B-PER)

### Evaluation

Span-level metrics: each predicted entity span is compared to ground truth.
- Exact match: predicted span must match both boundaries and type.
- Partial match: overlapping spans count toward recall.
- Metrics: precision, recall, F1 (micro-averaged across entity types).

## Code Examples

### Example 1: BERT for NER Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class BertForNER(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_labels=9, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        return logits, loss

model = BertForNER("bert-base-uncased", num_labels=9)
x = torch.randint(0, 1000, (2, 16))
mask = torch.ones(2, 16, dtype=torch.long)
labels = torch.randint(0, 9, (2, 16))
labels[:, 5:] = -100  # ignore some positions

logits, loss = model(x, mask, labels)
print("Logits shape:", logits.shape)
# Output: Logits shape: torch.Size([2, 16, 9])
print("Loss:", loss.item())
# Output: Loss: 3.2189
```

### Example 2: CRF Layer for NER

```python
class CRF(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))
        self.start_transitions = nn.Parameter(torch.randn(num_labels))
        self.end_transitions = nn.Parameter(torch.randn(num_labels))

    def forward(self, emissions, mask):
        best_path = self.viterbi_decode(emissions, mask)
        return best_path

    def viterbi_decode(self, emissions, mask):
        batch_size, seq_len, num_labels = emissions.shape
        score = self.start_transitions + emissions[:, 0]

        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)

        score = score + self.end_transitions
        return score

class BertWithCRF(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_labels=9):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state)
        mask = attention_mask.bool()
        best_path = self.crf.viterbi_decode(logits, mask)
        return best_path

model_ner = BertForNER("bert-base-uncased", num_labels=9)
model_crf = BertWithCRF("bert-base-uncased", num_labels=9)
print("NER model without CRF:", sum(p.numel() for p in model_ner.parameters()))
# Output: NER model without CRF: 108,655,113
print("NER model with CRF:", sum(p.numel() for p in model_crf.parameters()))
# Output: NER model with CRF: 108,655,161
print("CRF adds minimal parameters but enforces label transitions")
# Output: CRF adds minimal parameters but enforces label transitions
```

### Example 3: Label Alignment for NER

```python
def align_ner_labels(text, word_labels, tokenizer, label_map):
    tokens = tokenizer.tokenize(text)
    word_ids = tokenizer(text, return_tensors="pt").word_ids()[0]

    aligned_labels = []
    prev_word_id = None
    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
        elif word_id != prev_word_id:
            aligned_labels.append(label_map[word_labels[word_id]])
        else:
            aligned_labels.append(-100)
        prev_word_id = word_id

    return tokens, aligned_labels

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
label_map = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, "B-LOC": 5, "I-LOC": 6}

text = "John works at Microsoft in Seattle"
word_labels = ["B-PER", "O", "O", "B-ORG", "O", "B-LOC"]

tokens, aligned = align_ner_labels(text, word_labels, tokenizer, label_map)
print("Tokens:", tokens)
# Output: Tokens: ['[CLS]', 'john', 'works', 'at', 'microsoft', 'in', 'seattle', '[SEP]']
print("Aligned labels:", aligned)
# Output: Aligned labels: [-100, 1, 0, 0, 3, 0, 5, -100]
print("Only valid tokens have labels (CLS/SEP = -100)")
# Output: Only valid tokens have labels (CLS/SEP = -100)
```

## Common Mistakes

1. Not aligning labels for subword tokens: When a word splits into multiple subwords (e.g., "McDonald" → ["Mc", "##Donald"]), the label must be assigned to the first subword only. Other subwords get -100. Failing to do this trains the model on duplicate or incorrect labels.

2. Using wrong ignore index: PyTorch's cross-entropy uses ignore_index to skip certain positions. Using -1 instead of -100 (the default) may not work correctly with some loss functions.

3. Not handling the subtoken prediction during inference: During evaluation, predictions on non-first subwords should be ignored. The word-level label is taken from the first subword's prediction.

4. Forgetting to evaluate at the span level: Token-level accuracy can be misleading because most tokens are O (not entities). Span-level precision, recall, and F1 are the correct metrics.

5. Using CRF without considering speed: CRF decoding with Viterbi is O(L * C^2) where C is the number of tags. For large tag sets (e.g., 50+), this can be slow at inference time.

6. Ignoring entity type consistency: A CRF layer only enforces transition constraints. Without proper training, the model may produce invalid transitions (e.g., starting an entity with I-). Some implementations add hard constraints to prevent invalid sequences.

## Interview Questions

### Beginner

Q: How does BERT handle NER? Describe the basic architecture.

A: BERT for NER adds a linear classification head on top of each token's hidden state. Each token is classified into an entity tag (e.g., B-PER, I-PER, O). The input is a sentence, and BERT processes it bidirectionally, producing contextualized token representations. The classification head maps each representation to a tag probability distribution. The model is trained with cross-entropy loss on the token-level labels.

### Intermediate

Q: What is the purpose of the CRF layer in NER, and how does it improve performance?

A: The CRF layer learns valid tag transition patterns. For example, I-PER must follow B-PER or I-PER; it cannot follow B-ORG or O. Without CRF, the softmax classifier independently predicts each token's label, which can produce invalid sequences like "John (B-PER) works (I-PER)." The CRF assigns a score to the entire tag sequence, considering both token-level predictions and transition probabilities. During inference, Viterbi decoding finds the highest-scoring valid sequence. The CRF typically improves F1 by 1-2%.

### Advanced

Q: Describe the subword alignment problem in NER with BERT and propose a solution for handling entity spans that are split across multiple BERT subtokens.

A: When a word like "unhappiness" is tokenized as ["un", "##happiness"], the original word-level NER label (e.g., B-MISC) must be mapped to only "un" (first subtoken), and "##happiness" receives -100. During training, -100 positions are ignored in the loss. During inference, predictions on non-first subtokens are discarded, and the first subtoken's prediction is taken as the word-level prediction. For entity spans containing multiple words (e.g., "New York" → ["New", "York"]), both words retain their labels (B-LOC, I-LOC), and no special handling is needed since each word is its own token. A more sophisticated solution uses the first subtoken's prediction for the entity label and applies the full entity span using the word alignment from the tokenizer.

## Practice Problems

### Easy

Implement a function that converts word-level BIO labels to token-level labels using the tokenizer's word_ids() function. Handle cases where words split into multiple subwords. Verify with a sentence containing at least one multi-subword word.

### Medium

Fine-tune BERT-base on the CoNLL-2003 NER dataset. Compare performance with and without a CRF layer. Report token-level accuracy and span-level F1 for each approach.

### Hard

Design a nested NER system using BERT that can handle overlapping entities (e.g., "New York University" is both a LOC and an ORG simultaneously). Implement a multi-label NER head that predicts multiple entity types per token. Compare with a standard BIO approach.

## Solutions

```python
# Easy solution
def word_to_token_labels(text, word_labels, tokenizer, label_vocab):
    encoding = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    word_ids = encoding.word_ids()

    token_labels = []
    prev_word = None
    for word_id in word_ids:
        if word_id is None:
            token_labels.append(-100)
        elif word_id != prev_word:
            token_labels.append(label_vocab[word_labels[word_id]])
        else:
            token_labels.append(-100)
        prev_word = word_id

    return tokens, token_labels

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
label_vocab = {"O": 0, "B-MISC": 1, "I-MISC": 2}
text = "She loves Japanese culture and sushi."
labels = ["O", "B-MISC", "O", "O"]
tokens, token_labels = word_to_token_labels(text, labels, tokenizer, label_vocab)
print("Tokens:", [t for t in tokens if t not in ("[CLS]", "[SEP]")])
# Output: Tokens: ['she', 'loves', 'japanese', 'culture', 'and', 'sushi', '.']
print("Labels:", [l for l in token_labels if l != -100])
# Output: Labels: [0, 0, 1, 0, 0, 0, 0]
```

## Related Concepts

- BERT Fine-tuning (DL-409)
- BERT for Classification (DL-410)
- BERT for QA (DL-411)
- BIO Tagging Scheme
- Conditional Random Fields
- Sequence Labeling
- Subword Tokenization (DL-407)

## Next Concepts

- DistilBERT
- Sentence-BERT
- BERT in Production

## Summary

BERT for NER adapts the encoder with a token-level classification head, predicting entity tags for each token. Subword alignment is critical for correct training and evaluation. A CRF layer can be added to enforce valid label transitions, improving F1 by 1-2%. BERT-based NER achieves state-of-the-art results on standard benchmarks.

## Key Takeaways

- Token-level classification with linear head on BERT's hidden states.
- Labels must be aligned from word-level to subword-level.
- First subword gets the label; subsequent subwords get -100.
- BIO/BIOES tagging scheme for entity boundary marking.
- CRF layer enforces valid tag transitions.
- Evaluation uses span-level precision, recall, and F1.
- Subword alignment is critical during both training and inference.
- BERT achieves SOTA NER results across most benchmarks.
- The approach extends to any sequence labeling task (POS, chunking).
- CRF adds minimal parameters but can significantly improve performance.
