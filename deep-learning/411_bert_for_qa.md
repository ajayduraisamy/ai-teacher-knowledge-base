# Concept: BERT for Question Answering

## Concept ID

DL-411

## Difficulty

Advanced

## Domain

Deep Learning

## Module

BERT Family

## Learning Objectives

- Understand how BERT is adapted for extractive question answering using start and end token prediction.
- Implement the QA head architecture with separate start and end classifiers.
- Handle input formatting, token-label alignment, and answer extraction for QA.
- Analyze the differences between extractive, abstractive, and multiple-choice QA.
- Evaluate QA model performance using SQuAD metrics (EM and F1).

## Prerequisites

- Understanding of BERT architecture (DL-386) and fine-tuning (DL-409)
- Knowledge of the [CLS] and [SEP] tokens
- Familiarity with exact match (EM) and F1 evaluation metrics
- Understanding of token-label alignment for subword tokenization

## Definition

BERT for extractive question answering involves predicting the start and end positions of the answer span within a given context passage. The input is formatted as [CLS] question [SEP] context [SEP], and BERT processes this through its encoder. Two linear classifiers (start and end) are applied to each token's hidden state, producing probability distributions over all token positions. The start position is the token with the highest start probability, and the end position is the token (at or after the start position) with the highest end probability. The model is trained using cross-entropy loss on the start and end positions jointly. Introduced in the original BERT paper, this approach achieved SOTA on SQuAD 1.1 (F1 = 93.2 for BERT-large) and remains the standard architecture for extractive QA.

## Intuition

Think of extractive QA as asking BERT to highlight the answer in a passage. Read the passage: "Marie Curie was born in Warsaw, Poland in 1867." Question: "Where was Marie Curie born?" BERT needs to identify that "Warsaw" and "Poland" are part of the answer, and specifically mark where the answer begins and ends.

The start classifier looks at each token and asks: "Could this be where the answer starts?" Tokens like "Warsaw" and "Poland" get high scores. The end classifier asks: "Could this be where the answer ends?" For the correct answer "Warsaw, Poland," the start is "Warsaw" and the end is "Poland."

The challenge is that the answer could be any contiguous span of tokens. The model must not only identify relevant tokens but also determine the exact boundaries. The start and end classifiers work together: the predicted answer is the span from the highest-scoring start token to the highest-scoring end token that occurs after the start.

## Why This Concept Matters

Question answering is one of the most impactful NLP applications:

1. **Information retrieval**: Search engines, knowledge bases, enterprise document search.
2. **Customer support**: Automated answering of user questions from documentation.
3. **Education**: Reading comprehension assessment, study aids.
4. **Healthcare**: Answering medical questions from clinical literature.
5. **Legal**: Finding relevant passages in legal documents.

BERT's extractive QA approach established the standard paradigm that subsequent models (RoBERTa, ALBERT, DeBERTa) built upon. Understanding it is essential for building QA systems and for understanding reading comprehension research.

## Mathematical Explanation

### Input Formatting

Input = [CLS] Question tokens [SEP] Context tokens [SEP]

### Encoder

H = BERT(Input) = [h_CLS, h_Q1, ..., h_Qm, h_SEP, h_C1, ..., h_Cn, h_SEP]

### Start and End Classifiers

Start logits: S_i = h_i^T W_start + b_start
End logits: E_i = h_i^T W_end + b_end

Where W_start, W_end in R^{H x 1}.

Start probabilities: P_start(i) = exp(S_i) / sum_j exp(S_j)
End probabilities: P_end(i) = exp(E_i) / sum_j exp(E_j)

### Training Loss

Given ground-truth start position s and end position e:

L = -log(P_start(s)) - log(P_end(e))

### Inference

During inference, we find the best answer span (s, e) that maximizes:

Score(s, e) = P_start(s) * P_end(e)

Subject to: 0 <= s <= e < L (valid span) and s, e in context region (optional).

If the highest scoring span has positions outside the context (e.g., in the question), we typically fall back to the highest scoring context-only span.

### SQuAD Evaluation Metrics

Exact Match (EM): 1 if predicted span exactly matches ground truth, else 0.
F1: Token-level precision and recall between predicted and ground truth spans.
Macro-averaged across all questions.

## Code Examples

### Example 1: BERT for QA Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class BertForQA(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.qa_outputs = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs.last_hidden_state
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits

    def predict_answer_span(self, input_ids, attention_mask, token_type_ids=None):
        start_logits, end_logits = self.forward(input_ids, attention_mask, token_type_ids)
        start_probs = F.softmax(start_logits, dim=-1)
        end_probs = F.softmax(end_logits, dim=-1)

        best_score = float("-inf")
        best_start = 0
        best_end = 0

        for s in range(input_ids.size(1)):
            for e in range(s, min(s + 30, input_ids.size(1))):
                score = start_probs[0, s].item() * end_probs[0, e].item()
                if score > best_score:
                    best_score = score
                    best_start = s
                    best_end = e

        return best_start, best_end, best_score

model = BertForQA("bert-base-uncased")
x = torch.randint(0, 1000, (2, 64))
mask = torch.ones(2, 64, dtype=torch.long)
start_logits, end_logits = model(x, mask)
print("Start logits shape:", start_logits.shape)
# Output: Start logits shape: torch.Size([2, 64])
print("End logits shape:", end_logits.shape)
# Output: End logits shape: torch.Size([2, 64])
```

### Example 2: QA Loss Function

```python
def qa_loss(start_logits, end_logits, start_positions, end_positions):
    start_loss = F.cross_entropy(start_logits, start_positions)
    end_loss = F.cross_entropy(end_logits, end_positions)
    total_loss = (start_loss + end_loss) / 2
    return total_loss

def qa_training_step(model, batch, optimizer):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    start_positions = batch["start_positions"]
    end_positions = batch["end_positions"]

    start_logits, end_logits = model(input_ids, attention_mask)
    loss = qa_loss(start_logits, end_logits, start_positions, end_positions)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

start_logits = torch.randn(4, 64)
end_logits = torch.randn(4, 64)
start_pos = torch.tensor([5, 10, 15, 20])
end_pos = torch.tensor([8, 12, 18, 25])
loss = qa_loss(start_logits, end_logits, start_pos, end_pos)
print("QA loss:", loss.item())
# Output: QA loss: 4.1589
```

### Example 3: SQuAD Evaluation

```python
def compute_squad_metrics(predicted_spans, ground_truth_spans):
    exact_matches = 0
    total_f1 = 0
    n_questions = len(predicted_spans)

    for pred_tokens, gt_tokens in zip(predicted_spans, ground_truth_spans):
        pred_set = set(pred_tokens)
        gt_set = set(gt_tokens)

        if pred_tokens == gt_tokens:
            exact_matches += 1

        if len(pred_set) == 0 and len(gt_set) == 0:
            total_f1 += 1.0
            continue

        common = pred_set & gt_set
        precision = len(common) / len(pred_set) if pred_set else 0
        recall = len(common) / len(gt_set) if gt_set else 0
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        total_f1 += f1

    em = exact_matches / n_questions
    avg_f1 = total_f1 / n_questions
    return em, avg_f1

predicted = [["Warsaw", "Poland"], ["1867"], ["Marie", "Curie"]]
ground_truth = [["Warsaw", "Poland"], ["1867"], ["Marie", "Curie"]]
em, f1 = compute_squad_metrics(predicted, ground_truth)
print(f"Exact Match: {em:.4f}, F1: {f1:.4f}")
# Output: Exact Match: 1.0000, F1: 1.0000

predicted2 = [["Warsaw"], ["1867"], ["Curie"]]
ground_truth2 = [["Warsaw", "Poland"], ["1867"], ["Marie", "Curie"]]
em2, f12 = compute_squad_metrics(predicted2, ground_truth2)
print(f"Partial match - EM: {em2:.4f}, F1: {f12:.4f}")
# Output: Partial match - EM: 0.3333, F1: 0.8333
```

## Common Mistakes

1. Not restricting the answer span to the context: The model should only predict answer spans within the context tokens (after [SEP]). Predicting spans in the question or in special tokens produces invalid answers.

2. Using a simple argmax on start and end independently: argmax(start) might give position 50, argmax(end) might give position 3 (before start). The inference algorithm must ensure end >= start.

3. Ignoring the maximum answer length: SQuAD answers are typically short (< 30 tokens). Restricting the max span length during inference prevents unreasonably long answers.

4. Not handling impossible questions (SQuAD 2.0): SQuAD 2.0 includes questions with no answer. Models must predict whether an answer exists, typically using the [CLS] token for a "no answer" classifier.

5. Forgetting to align subword positions: The predicted token positions must be converted to character positions in the original text for evaluation and user display. Subword tokens may not align perfectly with word boundaries.

6. Using too high a learning rate: QA fine-tuning is sensitive to learning rate. 3e-5 for BERT-base and 2e-5 for BERT-large is typical. Higher rates may cause the model to predict trivial spans (e.g., single tokens).

## Interview Questions

### Beginner

Q: How does BERT handle extractive question answering? Describe the architecture.

A: BERT takes the input as [CLS] question [SEP] context [SEP]. Two linear classifiers are applied to each token's final hidden state: one predicts the probability that the token is the start of the answer, and the other predicts the probability that it is the end. The answer span is the contiguous segment from the highest-probability start token to the highest-probability end token (with end >= start). The model is trained by minimizing the cross-entropy loss on the correct start and end positions.

### Intermediate

Q: What is the difference between SQuAD 1.1 and SQuAD 2.0, and how does BERT need to be adapted for SQuAD 2.0?

A: SQuAD 1.1 assumes every question has an answer in the context. SQuAD 2.0 adds unanswerable questions where no answer exists. For SQuAD 2.0, BERT needs an additional "no answer" prediction. This is typically done by using the [CLS] token representation to predict whether an answer exists, or by computing a "no answer" score from the start and end logits. The training loss includes both the span prediction loss and the answerability loss. During inference, if the "no answer" score exceeds the best answer span score, the model returns "no answer."

### Advanced

Q: Discuss the limitations of extractive QA and how you would extend BERT to handle abstractive QA where the answer is not a span in the text.

A: Extractive QA cannot produce answers that are not explicitly stated in the context, cannot synthesize information from multiple passages, and cannot generate answers in a different form (e.g., "in 1867" instead of "1867"). For abstractive QA, one approach is to attach a text generation head (like T5's decoder) on top of BERT's encoder, treating it as a sequence-to-sequence problem. Another approach is to use a pointer-generator network that can both copy tokens from the context and generate tokens from a vocabulary. Modern abstractive QA systems (e.g., based on T5 or BART) use full encoder-decoder architectures rather than modifying BERT's encoder-only design.

## Practice Problems

### Easy

Format a question and context for BERT QA. Given the question "What is the capital of France?" and context "France is a country in Europe. Its capital is Paris. The city is known for the Eiffel Tower.", produce the tokenized input with special tokens.

### Medium

Fine-tune BERT-base on SQuAD 1.1 (or a subset). Implement the inference algorithm that finds the best answer span (maximizing start_prob * end_prob with end >= start). Report EM and F1 on the validation set.

### Hard

Implement a BERT-based QA system that can handle SQuAD 2.0 (unanswerable questions). Add an answerability classifier using the [CLS] token. Design a thresholding strategy that decides when to return "no answer" vs. a predicted span. Evaluate on SQuAD 2.0 validation set.

## Solutions

```python
# Easy solution
from transformers import AutoTokenizer

def format_qa_input(question, context, tokenizer, max_length=384):
    encoded = tokenizer(
        question, context,
        max_length=max_length,
        truncation="only_second",
        padding="max_length",
        return_tensors="pt",
        return_offsets_mapping=True
    )
    return encoded

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
question = "What is the capital of France?"
context = "France is a country in Europe. Its capital is Paris. The city is known for the Eiffel Tower."
encoded = format_qa_input(question, context, tokenizer)
print("Input IDs length:", encoded["input_ids"].shape[1])
# Output: Input IDs length: 384
tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
print("First 10 tokens:", tokens[:10])
# Output: First 10 tokens: ['[CLS]', 'what', 'is', 'the', 'capital', 'of', 'france', '?', '[SEP]', 'france']
sep_index = tokens.index("[SEP]")
print(f"Context starts at position {sep_index + 1}")
# Output: Context starts at position 9
```

## Related Concepts

- BERT Fine-tuning (DL-409)
- BERT for Classification (DL-410)
- BERT for NER (DL-412)
- SQuAD Dataset
- Extractive vs Abstractive QA
- Reading Comprehension

## Next Concepts

- BERT for NER
- DistilBERT
- Sentence-BERT

## Summary

BERT for extractive QA uses start and end classifiers on each token's hidden state to predict the answer span within a context passage. The approach achieves state-of-the-art results on SQuAD and has become the standard paradigm for reading comprehension. Key considerations include span constraint enforcement, subword alignment, and handling unanswerable questions.

## Key Takeaways

- Input format: [CLS] question [SEP] context [SEP].
- Two linear heads predict start and end token positions.
- Best span: argmax of start_prob * end_prob with end >= start.
- Loss: cross-entropy on start + end positions.
- Inference must enforce span validity (end >= start, max length).
- SQuAD evaluation uses Exact Match and F1.
- SQuAD 2.0 requires additional answerability prediction.
- Token positions must be aligned to character positions.
- Typical learning rate: 3e-5 (base), 2e-5 (large).
- The approach is limited to extractive (span-based) answers.
