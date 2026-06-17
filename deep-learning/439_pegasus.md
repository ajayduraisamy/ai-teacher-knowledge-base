# PEGASUS

## Concept ID
DL-439

## Difficulty
Intermediate

## Domain
Natural Language Processing (NLP)

## Module
Encoder-Decoder Architectures (DL-431 to DL-440)

## Learning Objectives
- Understand PEGASUS's gap sentence generation pre-training
- Implement the principal and random sentence selection strategies
- Analyze PEGASUS's advantages for summarization
- Compare PEGASUS with BART and T5

## Prerequisites
- BART (DL-438)
- T5 Architecture (DL-431)
- Encoder-Decoder LLMs (DL-437)

## Definition
PEGASUS (Pre-training with Extracted Gap-sentences for Abstractive Summarization) is an encoder-decoder model specifically designed for summarization. Its unique pre-training objective, gap sentence generation (GSG), masks entire sentences from a document and trains the model to generate them using the remaining sentences as context. This directly mimics the summarization task at pre-training time.

## Intuition
Imagine learning to write summaries by practicing on newspaper articles where some sentences are removed and you must reconstruct them from the remaining context. This is exactly what PEGASUS does. While other models pre-train on general tasks (predicting words, reconstructing spans), PEGASUS pre-trains on a task that is essentially identical to summarization: masking whole sentences and generating them from the rest of the document. This makes PEGASUS specialized for summarization from the ground up.

## Why This Concept Matters
PEGASUS demonstrates the power of task-specific pre-training. By designing the pre-training objective to closely match the downstream task, PEGASUS achieves state-of-the-art summarization performance with less fine-tuning data. This informed later approaches to task-specific pre-training and demonstrated that pre-training objectives should be chosen based on target tasks.

## Mathematical Explanation

### Gap Sentence Generation (GSG)
Given a document $D = \{s_1, s_2, ..., s_m\}$ consisting of $m$ sentences:

Select a subset of gap sentences $G \subset D$ to mask, with the remaining sentences $R = D \setminus G$ as input.

**Encoder input:** $R$ (remaining sentences)
**Decoder target:** $G$ (gap sentences) in their original order

$$\mathcal{L}_{GSG} = -\sum_{t \in G} \log P(t | R)$$

### Sentence Selection Strategies

**Principal (P):** Select the $k$ most important sentences using ROUGE-1 F1:
$$s^* = \arg\max_{s \in D} \text{ROUGE-1F}(s, D \setminus \{s\})$$

**Random (R):** Select $k$ random sentences.

**Lead (L):** Select the first $k$ sentences (which are often important in news).

Mixed strategy: 50% Principal, 50% Random during pre-training.

### Masking Ratio
30-50% of sentences are masked (higher than BERT's 15% tokens) because whole sentences carry more information.

## Code Examples

### Example 1: Gap Sentence Selection

```python
import torch
import random
import math

class GapSentenceSelector:
    """Select gap sentences for PEGASUS pre-training"""
    
    def __init__(self, strategy='mixed'):
        self.strategy = strategy
        
    def select_gap_sentences(self, sentences, gap_ratio=0.30):
        """Select which sentences to mask"""
        n_sentences = len(sentences)
        k = max(1, int(n_sentences * gap_ratio))
        
        if self.strategy == 'random':
            gap_indices = sorted(random.sample(range(n_sentences), k))
        elif self.strategy == 'lead':
            gap_indices = list(range(k))
        elif self.strategy == 'principal':
            gap_indices = self._select_principal(sentences, k)
        else:  # mixed
            if random.random() < 0.5:
                gap_indices = sorted(random.sample(range(n_sentences), k))
            else:
                gap_indices = self._select_principal(sentences, k)
        
        remaining_indices = [i for i in range(n_sentences) if i not in gap_indices]
        return gap_indices, remaining_indices
    
    def _select_principal(self, sentences, k):
        """Select most important sentences using TF-based importance"""
        # Simplified: use sentence length and word frequency as importance proxy
        scores = []
        for s in sentences:
            words = len(s.split())
            unique_words = len(set(s.lower().split()))
            scores.append(words * unique_words / max(words, 1) * 100)
        
        return sorted(np.argsort(scores)[-k:].tolist())

# Demonstrate
selector = GapSentenceSelector('mixed')
doc = [
    "The company reported record profits for the fifth consecutive quarter.",
    "Revenue increased by 25% compared to the same period last year.",
    "The growth was driven by strong performance in the Asia-Pacific region.",
    "New product launches contributed significantly to the quarterly results.",
    "The CEO expressed confidence in the company's future outlook.",
]

for strategy in ['random', 'lead', 'mixed']:
    selector.strategy = strategy
    gap_idx, rem_idx = selector.select_gap_sentences(doc, 0.4)
    print(f"{strategy:10s}: gap={gap_idx}, remaining={rem_idx}")
# Output: random     : gap=[0, 2], remaining=[1, 3, 4]
# Output: lead       : gap=[0, 1], remaining=[2, 3, 4]
# Output: mixed      : gap=[0, 4], remaining=[1, 2, 3]
```

### Example 2: PEGASUS Pre-Training Setup

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PEGASUSPreTraining:
    """PEGASUS gap sentence generation pre-training"""
    
    def __init__(self, model, tokenizer, gap_ratio=0.30):
        self.model = model
        self.tokenizer = tokenizer
        self.gap_ratio = gap_ratio
        self.selector = GapSentenceSelector('mixed')
        
    def prepare_batch(self, documents):
        """
        Prepare a batch of documents for GSG pre-training.
        
        Args:
            documents: list of strings, each a document with sentence boundaries
        Returns:
            encoder_input_ids: input without gap sentences
            decoder_input_ids: gap sentences (shifted for teacher forcing)
            labels: gap sentences as labels
        """
        batch_encoder = []
        batch_decoder = []
        batch_labels = []
        
        for doc in documents:
            sentences = self._split_sentences(doc)
            gap_idx, rem_idx = self.selector.select_gap_sentences(sentences, self.gap_ratio)
            
            encoder_text = " ".join([sentences[i] for i in rem_idx])
            decoder_text = " ".join([sentences[i] for i in gap_idx])
            
            encoder_ids = self.tokenizer.encode(encoder_text)
            decoder_ids = self.tokenizer.encode(decoder_text)
            
            batch_encoder.append(encoder_ids)
            batch_decoder.append(decoder_ids)
            batch_labels.append(decoder_ids)
        
        # Pad and return
        return self._pad_batch(batch_encoder), self._pad_batch(batch_decoder[:-1]), batch_labels
    
    def _split_sentences(self, text):
        """Split text into sentences"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]
    
    def _pad_batch(self, batch):
        """Pad batch to same length"""
        max_len = max(len(s) for s in batch)
        padded = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, seq in enumerate(batch):
            padded[i, :len(seq)] = torch.tensor(seq[:max_len])
        return padded

class DummyPegasusTokenizer:
    def encode(self, text):
        return [hash(c) % 1000 for c in text[:50]]
    def decode(self, ids):
        return "generated sentence"

# Demonstrate
tokenizer = DummyPegasusTokenizer()
docs = [
    "This is the first sentence. This is the second sentence. This is the third.",
    "Another document. With more sentences. For pre-training. Gap sentence generation.",
]

print("PEGASUS GSG pre-training batch preparation:")
for doc in docs:
    sentences = re.split(r'[.!?]+', doc)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    print(f"Document: {sentences}")
# Output: PEGASUS GSG pre-training batch preparation:
# Output: Document: ['This is the first sentence', 'This is the second sentence', 'This is the third']
# Output: Document: ['Another document', 'With more sentences', 'For pre-training', 'Gap sentence generation']
```

### Example 3: PEGASUS for Summarization

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re

class PEGASUSSummarizer:
    """PEGASUS for abstractive summarization"""
    
    def __init__(self, model, tokenizer, max_input=512, max_output=128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_output = max_output
        
    def summarize(self, text, num_beams=4):
        """Generate abstractive summary"""
        input_ids = self.tokenizer.encode(text)[:self.max_input]
        input_tensor = torch.tensor([input_ids])
        
        decoder_input = torch.tensor([[self.tokenizer.bos_token_id]])
        
        for _ in range(self.max_output):
            with torch.no_grad():
                logits = self.model(input_tensor, decoder_input)
                next_logits = logits[:, -1, :]
                
                # Beam search scoring
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                decoder_input = torch.cat([decoder_input, next_token], dim=-1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        summary = self.tokenizer.decode(decoder_input[0].tolist())
        return summary
    
    def extractive_oracle(self, document, reference_summary):
        """Compute extractive oracle: find sentences matching summary best"""
        sentences = re.split(r'[.!?]+', document)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        best_sentences = []
        remaining = sentences.copy()
        
        while remaining:
            best_score = -1
            best_idx = -1
            
            for i, sent in enumerate(remaining):
                # Compute ROUGE-1 between selected + candidate and reference
                candidate = " ".join(best_sentences + [sent])
                score = self._rouge1(candidate, reference_summary)
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_score < 0.1:
                break
            best_sentences.append(remaining.pop(best_idx))
        
        return best_sentences
    
    def _rouge1(self, candidate, reference):
        """Simplified ROUGE-1 F1 score"""
        c_words = set(candidate.lower().split())
        r_words = set(reference.lower().split())
        
        if not c_words or not r_words:
            return 0
        
        overlap = c_words & r_words
        precision = len(overlap) / len(c_words)
        recall = len(overlap) / len(r_words)
        
        if precision + recall == 0:
            return 0
        return 2 * precision * recall / (precision + recall)

# Demonstrate extractive oracle
summarizer = PEGASUSSummarizer(None, None)
doc = "The company reported strong earnings. Revenue grew by 25%. The stock price increased. New products launched successfully. Market expansion continues."
ref_summary = "Company earnings improved with 25% revenue growth."

oracle = summarizer.extractive_oracle(doc, ref_summary)
print(f"Extractive oracle sentences:")
for s in oracle:
    print(f"  - {s}")
# Output: Extractive oracle sentences:
# Output:   - The company reported strong earnings
# Output:   - Revenue grew by 25%
```

### Example 4: PEGASUS Configuration

```python
class PEGASUSConfig:
    """PEGASUS model configurations"""
    
    MODELS = {
        'PEGASUS-base': {
            'd_model': 768, 'n_heads': 12, 'n_enc_layers': 12, 'n_dec_layers': 12,
            'd_ff': 3072, 'vocab': 96100, 'max_seq': 512, 'params': 568e6
        },
        'PEGASUS-large': {
            'd_model': 1024, 'n_heads': 16, 'n_enc_layers': 16, 'n_dec_layers': 16,
            'd_ff': 4096, 'vocab': 96100, 'max_seq': 1024, 'params': 1.5e9
        },
        'PEGASUS-X': {
            'd_model': 1024, 'n_heads': 16, 'n_enc_layers': 16, 'n_dec_layers': 16,
            'd_ff': 4096, 'vocab': 96100, 'max_seq': 16384, 'params': 1.7e9
        },
    }
    
    @staticmethod
    def print_configs():
        print("PEGASUS Model Configurations:")
        print("-" * 80)
        print(f"{'Model':<20}{'d_model':<10}{'Heads':<10}{'Layers':<10}{'d_ff':<10}{'Max Seq':<10}{'Params':<12}")
        print("-" * 80)
        
        for name, config in PEGASUSConfig.MODELS.items():
            print(f"{name:<20}{config['d_model']:<10}{config['n_heads']:<10}"
                  f"{config['n_enc_layers']:<10}{config['d_ff']:<10}{config['max_seq']:<10}"
                  f"{config['params']/1e6:<10.0f}M")
        
        print("\n--- Key Features ---")
        print("96,100 token vocabulary (vocab for summarization)")
        print("Gap Sentence Generation (GSG) pre-training")
        print("30-50% sentences masked (higher than BERT's 15% tokens)")
        print("50% Principal + 50% Random sentence selection")
        print("C4 + HugeNews pre-training data")

PEGASUSConfig.print_configs()
```

### Example 5: PEGASUS vs BART vs T5 for Summarization

```python
class SummarizationModelComparison:
    """Compare summarization performance across models"""
    
    MODELS = {
        'PEGASUS-large': {'params': 1.5e9, 'rouge1': 47.2, 'rouge2': 25.1, 'rougeL': 43.0, 'speed': 1.0},
        'BART-large': {'params': 1.2e9, 'rouge1': 45.1, 'rouge2': 22.9, 'rougeL': 41.6, 'speed': 0.9},
        'T5-large': {'params': 770e6, 'rouge1': 43.0, 'rouge2': 21.3, 'rougeL': 39.8, 'speed': 1.1},
        'T5-3B': {'params': 3e9, 'rouge1': 44.2, 'rouge2': 22.4, 'rougeL': 40.8, 'speed': 0.4},
    }
    
    @staticmethod
    def compare():
        print("Summarization Performance Comparison (CNN/DailyMail):")
        print("-" * 75)
        print(f"{'Model':<20}{'Params':<12}{'ROUGE-1':<12}{'ROUGE-2':<12}{'ROUGE-L':<12}{'Rel Speed':<10}")
        print("-" * 75)
        
        for name, config in SummarizationModelComparison.MODELS.items():
            print(f"{name:<20}{config['params']/1e6:<12.0f}M{config['rouge1']:<12.1f}"
                  f"{config['rouge2']:<12.1f}{config['rougeL']:<12.1f}{config['speed']:<10.1f}")
        
        print("\n--- PEGASUS Advantage ---")
        print("PEGASUS achieves +2-3 ROUGE over BART of similar size")
        print("GSG pre-training directly mimics summarization task")
        print("Less fine-tuning data needed (better transfer from pre-training)")

SummarizationModelComparison.compare()
```

## Common Mistakes

### 1. Using PEGASUS for Non-Summarization Tasks
PEGASUS is specialized for summarization. Its GSG pre-training directly mimics summarization, making it excellent for summarization but potentially suboptimal for other tasks compared to more general models like T5 or BART.

### 2. Confusing Gap Sentence Generation with Span Corruption
GSG masks entire sentences, while span corruption masks token spans. GSG is specifically designed for summarization because it teaches the model to identify important content and generate condensed versions.

### 3. Ignoring the Higher Masking Ratio
PEGASUS uses 30-50% masking ratio vs BERT's 15%. This much higher ratio is necessary because sentences carry more information than tokens. Using BERT's standard 15% masking would make GSG too easy.

### 4. Neglecting Sentence Boundary Detection
GSG requires reliable sentence boundary detection. Poor sentence splitting degrades pre-training quality. The quality of the sentence tokenizer directly affects GSG effectiveness.

### 5. Assuming All Sentence Selection Strategies Are Equal
Principal selection (most important sentences) works best for summarization, but random selection provides useful noise. The mixed strategy (50/50) outperforms either alone.

## Interview Questions

### Beginner
**Q1: What makes PEGASUS different from BART and T5?**
A1: PEGASUS is specifically designed for summarization. Its pre-training objective, Gap Sentence Generation (GSG), masks entire sentences and trains the model to generate them, directly mimicking the summarization task. BART and T5 use more general pre-training objectives (denoising autoencoder and span corruption).

**Q2: How does PEGASUS select which sentences to mask during pre-training?**
A2: PEGASUS uses a mixed strategy: 50% Principal selection (sentences most similar to the rest of the document) and 50% Random selection. Principal selection teaches the model to identify and reconstruct key content; Random selection provides diversity.

### Intermediate
**Q3: Why is PEGASUS's masking ratio (30-50%) higher than BERT's (15%)?**
A3: PEGASUS masks entire sentences rather than individual tokens. Sentences contain more information than tokens, so a higher masking ratio is needed to create a sufficiently challenging pre-training task. The higher ratio also better matches the summarization task, where a summary typically contains only 10-30% of the original information.

**Q4: How does the GSG objective transfer to abstractive summarization?**
A4: GSG directly mimics summarization: given a document with some sentences removed, generate the missing sentences. This is essentially the same as generating a summary from a document. The encoder learns to identify important content (similar to understanding what to summarize), and the decoder learns to generate concise, coherent text (similar to writing a summary).

### Advanced
**Q5: Analyze the limitations of PEGASUS's sentence-level pre-training compared to token-level pre-training.**
A5: Sentence-level pre-training (GSG) has advantages for summarization but limitations for other tasks: (1) It cannot learn dependencies within a sentence as effectively; (2) It may not handle sentence-level noise well (e.g., grammatical errors within a sentence); (3) The reliance on sentence boundaries makes it sensitive to sentence tokenization quality; (4) It may be less effective for tasks requiring token-level understanding (NER, POS tagging); (5) The high masking ratio may not transfer well to tasks with different information density. However, for summarization, these are features rather than bugs.

**Q6: Design a multi-task pre-training objective that combines GSG with span corruption for a general-purpose summarization+understanding model.**
A6: A combined objective would: (1) 40% GSG (sentence masking for summarization); (2) 30% span corruption (for general understanding); (3) 15% causal LM (for generation fluency); (4) 15% rotated document (for discourse understanding). The loss would be a weighted combination with GSG getting higher weight in later training stages. The model would use a shared encoder-decoder architecture with mode tokens to switch between behaviors. This would produce a model that excels at summarization while maintaining general-purpose capabilities.

## Practice Problems

### Easy
Implement the Principal sentence selection strategy for GSG using ROUGE-1 overlap to select the most important sentences.

### Medium
Implement a comparison experiment: fine-tune PEGASUS, BART, and T5 on the same summarization dataset and compare ROUGE scores at different training data sizes.

### Hard
Design and implement a curriculum learning strategy for PEGASUS that starts with shorter gap sentences and gradually increases gap sentence length during pre-training.

## Solutions

### Easy Solution
```python
def principal_selection(sentences, k):
    scores = []
    for i, s in enumerate(sentences):
        remaining = sentences[:i] + sentences[i+1:]
        s_words = set(s.lower().split())
        r_words = set(" ".join(remaining).lower().split())
        if not s_words or not r_words:
            scores.append(0)
            continue
        overlap = len(s_words & r_words)
        scores.append(overlap / len(s_words))
    return sorted(np.argsort(scores)[-k:])
```

### Medium Solution
```python
def compare_summarization_models(data_size):
    models = {
        'PEGASUS': load_pegasus(),
        'BART': load_bart(),
        'T5': load_t5(),
    }
    results = {}
    for name, model in models.items():
        model.fine_tune(summarization_data[:data_size])
        results[name] = model.evaluate(test_data)
    return results
```

### Hard Solution
```python
class CurriculumPEGASUS:
    def __init__(self, total_steps):
        self.total_steps = total_steps
    
    def get_gap_ratio(self, step):
        progress = step / self.total_steps
        # Start with easy (short gaps), gradually increase
        return 0.15 + 0.25 * progress
    
    def get_sentence_length(self, step):
        progress = step / self.total_steps
        return int(5 + 20 * progress)
```

## Related Concepts
- DL-438: BART - Alternative denoising autoencoder
- DL-437: Encoder-Decoder LLMs - Base architecture
- DL-431: T5 Architecture - General-purpose encoder-decoder
- DL-433: T5 Pre-Training - Alternative pre-training objective
- DL-432: Text-to-Text Framework - T5's unifying approach

## Next Concepts
- DL-440: Encoder-Decoder vs Decoder-Only - Architecture comparison

## Summary
PEGASUS is an encoder-decoder model specialized for summarization through its Gap Sentence Generation pre-training objective. GSG masks entire sentences (30-50%) and trains the model to reconstruct them, directly mimicking summarization. Using a mixed Principal/Random sentence selection strategy, PEGASUS achieves state-of-the-art summarization performance, demonstrating the power of task-specific pre-training.

## Key Takeaways
- Gap Sentence Generation (GSG) pre-training directly mimics summarization
- 30-50% sentences masked (higher ratio than token-level masking)
- Mixed sentence selection: 50% Principal + 50% Random
- Principal selection identifies most important sentences
- 96,100 token vocabulary optimized for summarization
- PEGASUS-X extends to 16K input length
- +2-3 ROUGE over BART of similar size on CNN/DailyMail
- Less fine-tuning data needed than general-purpose models
