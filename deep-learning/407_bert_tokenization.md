# Concept: BERT Tokenization

## Concept ID

DL-407

## Difficulty

Advanced

## Domain

Deep Learning

## Module

BERT Family

## Learning Objectives

- Understand WordPiece tokenization and how it differs from BPE and other subword methods.
- Explain the role of special tokens ([CLS], [SEP], [PAD], [MASK], [UNK]) in BERT.
- Implement a WordPiece tokenizer and analyze its behavior on various inputs.
- Handle token-label alignment for token-level tasks (NER, QA, POS tagging).
- Apply proper tokenization preprocessing for fine-tuning BERT on downstream tasks.

## Prerequisites

- Understanding of tokenization concepts (word-level, character-level, subword)
- Familiarity with BERT's input format (DL-386)
- Knowledge of vocabulary and embedding concepts
- Understanding of sequence length constraints and padding

## Definition

BERT tokenization is the process of converting raw text into a sequence of subword tokens that BERT can process. BERT uses WordPiece tokenization, a data-driven subword algorithm that splits words into smaller units based on frequency, with a vocabulary of 30,522 tokens (BERT-base uncased). Special tokens encode structure: [CLS] (classification start), [SEP] (separator/end), [PAD] (padding), [MASK] (masked language modeling), and [UNK] (unknown). The tokenizer also handles lowercasing (for uncased models), Unicode normalization, and maximum sequence length truncation (typically 512 tokens). Proper tokenization is essential for correct BERT usage and significantly impacts downstream task performance.

## Intuition

WordPiece tokenization walks a middle path between word-level and character-level tokenization. Word-level ("playing" as one token) creates vocabulary sizes too large and cannot handle unknown words. Character-level ("p-l-a-y-i-n-g") loses word-level meaning and creates very long sequences. WordPiece splits words into meaningful subword units ("play" + "##ing"), balancing vocabulary size and sequence length.

The "##" prefix indicates a continuation of a previous word. "Playing" → ["play", "##ing"]. The "##" signals that "ing" is not a standalone word but a suffix attached to "play". This allows the model to understand morphological structure.

Special tokens provide structural signals to BERT: [CLS] tells the model "aggregate information here for classification," [SEP] indicates boundaries between sentences, [PAD] fills space for batched processing, and [MASK] signals prediction targets during pre-training.

## Why This Concept Matters

Tokenization is the gateway between raw text and BERT. Correct tokenization is essential for:

1. **Input validity**: Improper tokenization can produce invalid inputs (e.g., tokens outside the vocabulary, incorrect special tokens).
2. **Task performance**: Tokenizer choice (cased vs uncased, vocabulary version) affects accuracy by 1-3%.
3. **Label alignment**: For token-level tasks (NER, QA), labels must align with subword tokens, not original word boundaries.
4. **Sequence length**: Tokenization affects sequence length. Longer tokenization (more subword splits) may exceed BERT's 512-token limit.
5. **Reproducibility**: Different tokenizer versions produce different results. Standardizing tokenization is essential for reproducible research.

## Mathematical Explanation

### WordPiece Algorithm

WordPiece builds a vocabulary greedily by maximizing the likelihood of the training data given the current vocabulary:

1. Start with a vocabulary of characters.
2. Iteratively merge the pair of tokens (A, B) that maximizes the score:
   Score(A, B) = P(AB) / (P(A) * P(B))
3. Continue until the vocabulary reaches the desired size (30,522 for BERT).

The score measures the "coherence" of the pair: if P(AB) is much higher than expected from independence, the pair is a meaningful unit.

### Tokenization Process

1. **Text normalization**: Lowercasing (uncased), Unicode normalization (NFD), accent removal (uncased).
2. **Pre-tokenization**: Split text by whitespace and punctuation into "words."
3. **WordPiece tokenization**: For each word, greedily find the longest prefix match in the vocabulary. Continue with the remaining suffix. If a character cannot be matched, the whole word becomes [UNK].
4. **Special token addition**: Add [CLS] at start, [SEP] between sentences and at end.
5. **Truncation**: Truncate to maximum sequence length (typically 512 tokens).
6. **Padding**: Pad to the longest sequence in the batch with [PAD] tokens.
7. **Attention mask**: 1 for real tokens, 0 for padding tokens.

### Token-Label Alignment

For token-level tasks, labels (e.g., BIO tags for NER) must be aligned with subword tokens:
- Original: ["John", "lives", "in", "New", "York"]
- Tokens: ["John", "lives", "in", "New", "York"]
- (No splitting for these words)
- But: ["unhappiness"] → ["un", "##happiness"]
- Label for "unhappiness" (e.g., "O") must be applied to the first subword token "un", and "X" (ignore) to "##happiness".

## Code Examples

### Example 1: Basic BERT Tokenization

```python
from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

text = "I can't believe BERT tokenizes differently!"
tokens = tokenizer.tokenize(text)
print("Original:", text)
# Output: Original: I can't believe BERT tokenizes differently!
print("Tokens:", tokens)
# Output: Tokens: ['i', 'can', "'", 't', 'believe', 'bert', 'token', '##izes', 'differently', '!']

input_ids = tokenizer.encode(text)
print("Input IDs:", input_ids[:15])
# Output: Input IDs: [101, 1045, 2064, 1005, 1056, 2876, 6146, 19204, 10437, 5470, 10619, 999, 102]
print("Decoded:", tokenizer.decode(input_ids))
# Output: Decoded: [CLS] i can't believe bert tokenizes differently! [SEP]
```

### Example 2: Sentence Pair Tokenization

```python
def tokenize_pair(sentence_a, sentence_b, tokenizer, max_length=128):
    encoded = tokenizer(
        sentence_a, sentence_b,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_token_type_ids=True,
        return_attention_mask=True
    )
    return encoded

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encoded = tokenize_pair(
    "The cat sat on the mat.",
    "It was a sunny day.",
    tokenizer
)

print("Input IDs shape:", encoded["input_ids"].shape)
# Output: Input IDs shape: torch.Size([1, 128])
print("Token type IDs shape:", encoded["token_type_ids"].shape)
# Output: Token type IDs shape: torch.Size([1, 128])
print("Attention mask shape:", encoded["attention_mask"].shape)
# Output: Attention mask shape: torch.Size([1, 128])

print("First 20 input IDs:", encoded["input_ids"][0, :20].tolist())
# Output: First 20 input IDs: [101, 1996, 4937, 5806, 2006, 1996, 4938, 1012, 102, 2009, 2001, 1037, 7262, 2159, 1012, 102, 0, 0, 0, 0]
print("Token types (0=A, 1=B):", encoded["token_type_ids"][0, :20].tolist())
# Output: Token types (0=A, 1=B): [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
```

### Example 3: Token-Label Alignment for NER

```python
def align_labels_with_tokens(labels, word_ids, ignore_label=-100):
    aligned_labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(ignore_label)
        elif word_idx != previous_word_idx:
            aligned_labels.append(labels[word_idx])
        else:
            aligned_labels.append(ignore_label)
        previous_word_idx = word_idx
    return aligned_labels

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
sentence = "John lives in New York"
ner_labels = ["B-PER", "O", "O", "B-LOC", "I-LOC"]

encoding = tokenizer(sentence, return_tensors="pt")
word_ids = encoding.word_ids()

aligned_labels = align_labels_with_tokens(ner_labels, word_ids)
tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

print("Word IDs:", word_ids)
# Output: Word IDs: (None, 0, 1, 2, 3, 4, 4, None)
print("Tokens:", tokens)
# Output: Tokens: ['[CLS]', 'john', 'lives', 'in', 'new', 'york', '[SEP]']
print("Original labels:", ner_labels)
# Output: Original labels: ['B-PER', 'O', 'O', 'B-LOC', 'I-LOC']
print("Aligned labels:", aligned_labels)
# Output: Aligned labels: [-100, 'B-PER', 'O', 'O', 'B-LOC', -100, -100]
```

## Common Mistakes

1. Not handling special tokens in loss computation: Labels for [CLS], [SEP], [PAD] should be set to -100 (ignore) in cross-entropy loss to avoid penalizing the model for its predictions on these tokens.

2. Forgetting to set return_token_type_ids=False for single-sentence tasks: Using token type IDs for single-sentence inputs is redundant but not harmful. However, forgetting to handle them for models that don't support them (RoBERTa) will cause errors.

3. Using the wrong max_length: BERT's maximum sequence length is 512 tokens. If tokenized text exceeds this, it must be truncated. Not truncating will cause errors, and choosing too small a max_length loses information.

4. Not handling cased vs uncased consistently: BERT uncased lowercase all text and removes accents. Using uncased tokenizer on cased input is fine (it lowercases), but using cased tokenizer on uncased input may produce [UNK] tokens.

5. Confusing word IDs: encoding.word_ids() returns the index of the original word for each token (None for special tokens). This is used for label alignment but is easy to misuse.

6. Ignoring tokenizer version differences: Different versions of the same tokenizer may have different vocabularies. Using a tokenizer from one checkpoint with a model from another can cause mismatches.

## Interview Questions

### Beginner

Q: What special tokens does BERT use, and what is the purpose of each?

A: [CLS] — classification token placed at the start; its final hidden state is used for classification. [SEP] — separator between sentences and end of sequence marker. [PAD] — padding token to make all sequences in a batch the same length. [MASK] — replaces masked tokens during pre-training. [UNK] — represents unknown tokens not in the vocabulary.

### Intermediate

Q: How does BERT handle out-of-vocabulary words that are not in its WordPiece vocabulary?

A: WordPiece tokenization splits unknown words into subword units that are in the vocabulary. For example, if "unaffectionate" is not in the vocabulary, it may be split into ["un", "##affection", "##ate"]. If any character cannot be represented by any subword unit in the vocabulary, the entire word is replaced with the [UNK] token. In practice, with a 30K vocabulary, [UNK] rarely appears for English text.

### Advanced

Q: Describe the label alignment problem for NER with BERT tokenization. How would you handle a word like "unhappiness" that splits into three subword tokens with BIO labels?

A: When a word like "unhappiness" (B-MISC label) is tokenized as ["un", "##happiness", "##ness"], the label must be assigned to only the first subword token "un" as "B-MISC". The remaining subword tokens ["##happiness", "##ness"] should be assigned a special ignore label (e.g., -100 in PyTorch, -1 in TensorFlow) so the loss function ignores them. This prevents the model from being penalized for incorrect predictions on subword tokens that are not the start of a word. During inference, predictions on non-first subword tokens are typically ignored, and the first subword token's prediction is taken as the word-level prediction.

## Practice Problems

### Easy

Tokenize the sentence "BERT is a bidirectional encoder!" using BERT's WordPiece tokenizer. Identify which words are split into multiple subwords and explain why.

### Medium

Implement a function that takes a sentence and its word-level part-of-speech tags and returns token-level aligned labels. Handle words that split into multiple subwords by assigning the label to the first subword and -100 to the rest. Verify with a sentence containing at least one multi-subword token.

### Hard

Implement a custom WordPiece tokenizer training from scratch on a small corpus. Train a vocabulary of 5000 tokens. Show that it can tokenize new text and compare the tokenization quality (compression ratio, OOV rate) with BERT's standard tokenizer.

## Solutions

```python
# Easy solution
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text = "BERT is a bidirectional encoder!"
tokens = tokenizer.tokenize(text)
print("Sentence:", text)
print("Tokens:", tokens)
# Output: Sentence: BERT is a bidirectional encoder!
# Output: Tokens: ['bert', 'is', 'a', 'bidirectional', 'encoder', '!']
words = text.lower().split()
print("Words and their subword splits:")
for word in words:
    subwords = tokenizer.tokenize(word)
    if len(subwords) > 1:
        print(f"  '{word}' -> {subwords} (split into {len(subwords)} subwords)")
    else:
        print(f"  '{word}' -> '{subwords[0]}' (no split)")
# Output: Words and their subword splits:
#   'bert!' -> 'bert!' (no split)
#   'is' -> 'is' (no split)
#   'a' -> 'a' (no split)
#   'bidirectional' -> 'bidirectional' (no split)
#   'encoder!' -> ['encoder', '!'] (split into 2 subwords)
```

## Related Concepts

- BERT Architecture (DL-386)
- BERT Embeddings (DL-408)
- BERT Fine-tuning (DL-409)
- BERT for NER (DL-412)
- WordPiece Tokenization
- Byte-Pair Encoding
- SentencePiece Tokenization

## Next Concepts

- BERT Embeddings
- BERT Fine-tuning
- BERT for Classification
- BERT for QA
- BERT for NER

## Summary

BERT tokenization uses WordPiece to convert text into subword tokens from a 30,522 vocabulary. Special tokens ([CLS], [SEP], [PAD], [MASK]) structure the input. Proper tokenization handling — including label alignment for token-level tasks, truncation to 512 tokens, and attention mask creation — is essential for correct BERT usage and downstream performance.

## Key Takeaways

- WordPiece splits words into subword units based on frequency.
- The "##" prefix indicates subword continuation.
- Special tokens: [CLS], [SEP], [PAD], [MASK], [UNK].
- Uncased models lowercase text and remove accents.
- Maximum sequence length is 512 tokens (including special tokens).
- Label alignment: assign labels to first subword, -100 to rest.
- Token type IDs distinguish sentence A (0) from sentence B (1).
- Attention mask distinguishes real tokens (1) from padding (0).
- Proper tokenization is critical for task performance.
- Tokenizer version must match model checkpoint.
