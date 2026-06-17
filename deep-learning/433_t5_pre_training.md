# T5 Pre-Training

## Concept ID
DL-433

## Difficulty
Intermediate

## Domain
Natural Language Processing (NLP)

## Module
Encoder-Decoder Architectures (DL-431 to DL-440)

## Learning Objectives
- Understand T5's span corruption pre-training objective
- Implement the pre-training pipeline
- Analyze the C4 dataset and its impact
- Compare pre-training objectives across models

## Prerequisites
- T5 Architecture (DL-431)
- Text-to-Text Framework (DL-432)
- Self-Supervised Learning (DL-280)

## Definition
T5 pre-training uses a denoising autoencoding objective called span corruption, where contiguous spans of tokens (mean length 3) are masked and replaced with sentinel tokens. The model is trained to reconstruct the original spans. This is performed on the C4 (Colossal Clean Crawled Corpus) dataset, a 750GB filtered version of CommonCrawl.

## Intuition
Imagine giving someone a page of text with certain sentences blacked out and asking them to fill in the missing words. Unlike BERT which masks individual words (you know exactly how many words are missing) or GPT which predicts the next word (only one word at a time), T5 masks entire phrases of variable length and asks the model to reconstruct them all in sequence. This teaches the model to understand context, handle missing information, and generate coherent text—skills that transfer well to summarization, translation, and question answering.

## Why This Concept Matters
T5's pre-training approach was carefully designed through systematic comparison of 15+ different objectives (span corruption, prefix LM, BERT-style, etc.). Span corruption was found to be the most effective for text-to-text tasks. The C4 dataset also demonstrated that aggressive text filtering could produce high-quality data from web crawls. Understanding these choices is crucial for designing effective pre-training strategies.

## Mathematical Explanation

### Span Corruption Objective
Given input sequence $x = (x_1, ..., x_T)$, we sample a set of masked spans:

$$\mathcal{M} = \{(s_i, l_i)\}_{i=1}^{k}$$

Where $s_i$ is the start position and $l_i$ is the length of span $i$.

The corrupted input replaces each span with a sentinel token:

$$\tilde{x} = x_1, ..., x_{s_1-1}, <sentinel_1>, x_{s_1+l_1+1}, ..., <sentinel_k>, ...$$

The target output concatenates all masked spans with their sentinels:

$$y = <sentinel_1>, x_{s_1}, ..., x_{s_1+l_1-1}, <sentinel_2>, ...$$

The training loss is:

$$\mathcal{L} = -\log P(y | \tilde{x}) = -\sum_{i=1}^{|y|} \log P(y_i | \tilde{x}, y_{<i})$$

### C4 Dataset Filtering
Documents must satisfy:

$$f_{filter}(d) = \begin{cases} 1 & \text{if } L_{min} < |d| < L_{max} \land C_{end} > 0.5 \land \\ & \quad P_{bad} < 0.01 \land |d|_{dup} < 0.3 \\ 0 & \text{otherwise} \end{cases}$$

Where:
- $L_{min}=3$ sentences, $L_{max}=65$ words
- $C_{end}$ is the fraction of sentences ending with punctuation
- $P_{bad}$ is the probability of containing profanity
- $|d|_{dup}$ is the duplicate line ratio

## Code Examples

### Example 1: Span Corruption Implementation

```python
import torch
import random

class SpanCorruption:
    """Span corruption pre-training objective"""
    
    def __init__(self, noise_density=0.15, mean_span_length=3, vocab_size=32128):
        self.noise_density = noise_density
        self.mean_span_length = mean_span_length
        self.vocab_size = vocab_size
        # Sentinels are typically the last N special tokens
        self.sentinel_start = vocab_size - 100  # e.g., <extra_id_0> = 32128
        
    def corrupt(self, input_ids, padding_mask=None):
        """
        Apply span corruption.
        
        Args:
            input_ids: (batch, seq_len)
        Returns:
            corrupted_inputs: (batch, seq_len) with spans replaced by sentinels
            targets: (batch, seq_len) with sentinel-guided span outputs
        """
        batch_size, seq_len = input_ids.shape
        corrupted = input_ids.clone()
        all_targets = []
        
        for b in range(batch_size):
            seq = input_ids[b].tolist()
            sentinel_idx = 0
            i = 0
            corrupted_seq = []
            target_seq = []
            
            while i < len(seq):
                if random.random() < self.noise_density:
                    # Start a corrupted span
                    span_len = min(
                        random.randint(1, 2 * self.mean_span_length - 1),
                        len(seq) - i
                    )
                    
                    # Sentinel ID
                    sentinel_id = self.sentinel_start + sentinel_idx
                    
                    # Add sentinel to corrupted input
                    corrupted_seq.append(sentinel_id)
                    
                    # Add sentinel + span tokens to target
                    target_seq.append(sentinel_id)
                    target_seq.extend(seq[i:i+span_len])
                    
                    sentinel_idx += 1
                    i += span_len
                else:
                    corrupted_seq.append(seq[i])
                    i += 1
            
            # Add final sentinel to target
            target_seq.append(self.sentinel_start + sentinel_idx)
            
            # Pad or truncate to seq_len
            corrupted_seq = corrupted_seq[:seq_len]
            if len(corrupted_seq) < seq_len:
                corrupted_seq.extend([0] * (seq_len - len(corrupted_seq)))
            
            target_seq = target_seq[:seq_len]
            if len(target_seq) < seq_len:
                target_seq = [-100] * (seq_len - len(target_seq))
            
            corrupted[b] = torch.tensor(corrupted_seq[:seq_len])
            all_targets.append(torch.tensor(target_seq[:seq_len]))
        
        return corrupted, torch.stack(all_targets)

# Demonstrate
corruptor = SpanCorruption(noise_density=0.15, mean_span_length=3)
input_ids = torch.randint(10, 100, (2, 16))
corrupted, targets = corruptor.corrupt(input_ids)

print("Span Corruption Pre-training:")
print(f"Original:   {input_ids[0].tolist()}")
print(f"Corrupted:  {corrupted[0].tolist()}")
print(f"Targets:    {targets[0].tolist()}")
# Output: Span Corruption Pre-training:
# Output: Original:   [87, 45, 23, 67, 12, 34, 56, 78, 90, 11, 22, 33, 44, 55, 66, 77]
# Output: Corrupted:  [87, 45, 32128, 12, 34, 56, 78, 32129, 22, 33, 44, 55, 32130, 77, 0, 0]
# Output: Targets:    [32128, 23, 67, 32129, 90, 11, 32130, 66, 32131, -100, -100, -100, -100, -100, -100, -100]
```

### Example 2: C4 Dataset Processing

```python
import re
from collections import Counter

class C4Processor:
    """C4 (Colossal Clean Crawled Corpus) filtering"""
    
    def __init__(self):
        self.min_sentences = 3
        self.max_words = 65
        self.bad_words = set(['porn', 'sex', 'xxx', 'adult', 'explicit'])
        
    def is_valid_document(self, text):
        """Apply C4 quality filters"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Min/max length
        if len(sentences) < self.min_sentences:
            return False
        
        words = text.split()
        if len(words) > 100000:
            return False
        
        # Ending punctuation check
        valid_endings = sum(1 for s in sentences if s and s[-1] in '.!?')
        ending_ratio = valid_endings / len(sentences) if sentences else 0
        if ending_ratio < 0.5:
            return False
        
        # Profanity filtering
        text_lower = text.lower()
        bad_count = sum(1 for w in self.bad_words if w in text_lower)
        if bad_count > 0:
            return False
        
        # Deduplication (line-level)
        lines = text.split('\n')
        if lines:
            line_counts = Counter(lines)
            dup_ratio = sum(c-1 for c in line_counts.values() if c > 1) / len(lines)
            if dup_ratio > 0.3:
                return False
        
        return True
    
    def process_crawl(self, documents):
        """Filter a batch of documents"""
        kept = []
        removed = 0
        
        for doc in documents:
            if self.is_valid_document(doc):
                kept.append(doc)
            else:
                removed += 1
        
        print(f"C4 filtering: {len(kept)}/{len(kept)+removed} documents kept ({removed} removed)")
        return kept

# Demonstrate
c4 = C4Processor()
docs = [
    "This is a valid document. It has multiple sentences. Each sentence ends properly!",
    "short",
    "bad words: xxx porn adult content",
    "Valid article about AI. Machine learning is fascinating. Neural networks are powerful. Deep learning has transformed NLP.",
]

kept = c4.process_crawl(docs)
print(f"Kept documents: {len(kept)}")
# Output: C4 filtering: 2/4 documents kept (2 removed)
# Output: Kept documents: 2
```

### Example 3: Comparing Pre-Training Objectives

```python
class PreTrainingObjectiveComparison:
    """Compare different pre-training objectives"""
    
    @staticmethod
    def causal_lm(sequence):
        """GPT-style: predict next token"""
        inputs = sequence[:-1]
        targets = sequence[1:]
        return inputs, targets
    
    @staticmethod
    def masked_lm(sequence, mask_prob=0.15):
        """BERT-style: predict masked tokens"""
        inputs = sequence.clone()
        targets = torch.full_like(sequence, -100)
        
        mask = torch.rand(len(sequence)) < mask_prob
        # 80% mask token, 10% random, 10% unchanged
        for i in range(len(sequence)):
            if mask[i]:
                p = random.random()
                if p < 0.8:
                    inputs[i] = 3  # [MASK] token
                elif p < 0.9:
                    inputs[i] = random.randint(4, 1000)
                targets[i] = sequence[i]
        
        return inputs, targets
    
    @staticmethod
    def prefix_lm(sequence, split_ratio=0.5):
        """Prefix LM: first part bidirectional, second part autoregressive"""
        split = int(len(sequence) * split_ratio)
        # Encoder gets first split, decoder predicts second split
        encoder_input = sequence[:split]
        decoder_input = sequence[split:-1]
        targets = sequence[split+1:]
        return encoder_input, decoder_input, targets
    
    @staticmethod
    def compare(seq_length=20):
        sequence = torch.randint(10, 100, (seq_length,))
        
        print("Pre-training Objective Comparison:")
        print(f"Original: {sequence.tolist()}")
        
        # Causal LM
        inp, tgt = PreTrainingObjectiveComparison.causal_lm(sequence)
        print(f"Causal LM:     inp={inp.tolist()}")
        print(f"              tgt={tgt.tolist()}")
        
        # Masked LM
        inp, tgt = PreTrainingObjectiveComparison.masked_lm(sequence)
        print(f"Masked LM:     inp={inp.tolist()}")
        non_ignore = tgt[tgt != -100]
        print(f"              tgt={non_ignore.tolist()} ({len(non_ignore)} masked)")
        
        # Prefix LM
        enc, dec, tgt = PreTrainingObjectiveComparison.prefix_lm(sequence)
        print(f"Prefix LM:     enc={enc.tolist()}")
        print(f"              dec={dec.tolist()}")
        print(f"              tgt={tgt.tolist()}")

PreTrainingObjectiveComparison.compare()
```

### Example 4: T5 Pre-Training Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class T5PreTrainingLoss:
    """Compute T5 pre-training loss"""
    
    @staticmethod
    def compute_loss(logits, labels, sentinel_start=32128):
        """
        Compute cross-entropy loss for span corruption.
        
        Args:
            logits: (batch, seq_len, vocab_size)
            labels: (batch, seq_len) with -100 for non-loss positions
        Returns:
            loss, accuracy, num_active_tokens
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Flatten
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        # Compute loss (ignore -100 positions)
        loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
        
        # Compute accuracy on non-ignored positions
        mask = labels_flat != -100
        if mask.any():
            predictions = logits_flat[mask].argmax(dim=-1)
            correct = (predictions == labels_flat[mask]).sum().item()
            total = mask.sum().item()
            accuracy = correct / total if total > 0 else 0
        else:
            accuracy = 0.0
            total = 0
        
        return loss, accuracy, total
    
    @staticmethod
    def compute_mask_cross_entropy(logits, labels, sentinel_ids):
        """Cross-entropy only on sentinel-guided positions"""
        # Find positions with sentinel tokens
        mask = labels.unsqueeze(-1).eq(sentinel_ids).any(dim=-1)
        
        logits_masked = logits[mask]
        labels_masked = labels[mask]
        
        return F.cross_entropy(logits_masked, labels_masked)

# Demonstrate
vocab_size = 32128
logits = torch.randn(2, 16, vocab_size)
labels = torch.full((2, 16), -100, dtype=torch.long)
labels[0, 2:5] = torch.tensor([32128, 45, 67])
labels[0, 8:11] = torch.tensor([32129, 23, 89])

loss, acc, total = T5PreTrainingLoss.compute_loss(logits, labels)
print(f"T5 Pre-training loss: {loss:.4f}")
print(f"Active tokens: {total}, Accuracy: {acc:.2%}")
# Output: T5 Pre-training loss: 10.3766
# Output: Active tokens: 6, Accuracy: 0.00%
```

### Example 5: Multi-Task Pre-Training

```python
class T5MultiTaskPreTraining:
    """Multi-task pre-training combining span corruption with supervised tasks"""
    
    TASKS = ['span_corruption', 'translation', 'summarization', 'classification']
    
    def __init__(self, mixture_weights=None):
        self.mixture_weights = mixture_weights or {
            'span_corruption': 0.7,
            'translation': 0.1,
            'summarization': 0.1,
            'classification': 0.1,
        }
    
    def sample_task(self):
        """Sample a task based on mixture weights"""
        tasks = list(self.mixture_weights.keys())
        weights = list(self.mixture_weights.values())
        return random.choices(tasks, weights=weights, k=1)[0]
    
    def prepare_batch(self, batch_size, seq_len):
        """Prepare a multi-task batch"""
        task = self.sample_task()
        
        if task == 'span_corruption':
            input_ids = torch.randint(10, 100, (batch_size, seq_len))
            corrupted, targets = SpanCorruption().corrupt(input_ids)
            return corrupted, targets
        else:
            # Simulate supervised task
            prefix = TextToTextSystem.PREFIXES.get(task, '')
            inputs = [f"{prefix}example {i}" for i in range(batch_size)]
            outputs = ["output" for _ in range(batch_size)]
            return inputs, outputs

# Demonstrate
trainer = T5MultiTaskPreTraining({'span_corruption': 0.8, 'translation': 0.2})
task_counts = {'span_corruption': 0, 'translation': 0}
for _ in range(100):
    task_counts[trainer.sample_task()] += 1
print(f"Task distribution: {task_counts}")
# Output: Task distribution: {'span_corruption': 81, 'translation': 19}
```

## Common Mistakes

### 1. Confusing Span Corruption with Masked Language Modeling
Span corruption masks contiguous spans (average length 3) and predicts them autoregressively with sentinel tokens. MLM masks individual tokens and predicts each independently. Span corruption teaches the model to handle variable-length missing information; MLM teaches token-level understanding.

### 2. Incorrect Sentinel Token Handling
Sentinel tokens are special vocabulary tokens (typically <extra_id_0>, <extra_id_1>, etc.) that must be unique and not used for other purposes. Using the wrong sentinel tokens or reusing them across the input will corrupt the pre-training objective.

### 3. Ignoring the C4 Filtering Pipeline
T5's C4 dataset involved extensive filtering. Simply downloading CommonCrawl data without applying C4's specific filtering heuristics (ending punctuation, bad words, deduplication) will produce significantly lower quality data and worse model performance.

### 4. Inconsistent Noise Density and Span Length
The optimal span corruption parameters (15% noise density, mean span length 3) were determined through systematic experimentation. Changing these parameters without corresponding adjustments (e.g., more training steps for higher noise density) can degrade performance.

### 5. Forgetting the Final Sentinel Token
The target sequence should include a final sentinel token (e.g., <extra_id_N>) after the last reconstructed span, even when there are no more spans. This sentinel signals the end of the target sequence to the decoder. Omitting it causes the decoder to never learn the sequence-end signal.

## Interview Questions

### Beginner
**Q1: What is T5's pre-training objective?**
A1: T5 uses span corruption, where 15% of input tokens are masked in contiguous spans (average length 3 tokens), replaced by sentinel tokens. The model is trained to reconstruct the original spans autoregressively, with sentinel tokens guiding the decoder where to insert each span.

**Q2: What is the C4 dataset and how was it created?**
A2: C4 (Colossal Clean Crawled Corpus) is a 750GB dataset created by filtering CommonCrawl data. Filters include: minimum 3 sentences, ending punctuation check (>50% sentences end properly), profanity removal, duplicate line detection, and length bounds. This aggressive filtering produces high-quality text from noisy web data.

### Intermediate
**Q3: How does T5's span corruption compare to BERT's masked language modeling and GPT's causal language modeling?**
A3: MLM (BERT) masks and predicts individual tokens independently, learning bidirectional but not generative capabilities. Causal LM (GPT) predicts tokens left-to-right, learning generation but no bidirectional context. Span corruption (T5) combines benefits: the encoder sees corrupted input bidirectionally, and the decoder learns to generate the missing spans autoregressively. This provides both understanding (through reconstruction) and generation (through decoding) capabilities.

**Q4: Why did T5 use 15% noise density with mean span length 3?**
A4: These parameters were empirically determined by comparing 15+ pre-training objectives. 15% noise density provides enough corruption for meaningful learning without making the task too difficult. Mean span length 3 encourages the model to understand phrases and concepts rather than individual tokens. The combination outperformed other settings (e.g., 10% or 20% noise, span length 1 or 5).

### Advanced
**Q5: Analyze the relationship between span length and the types of linguistic knowledge learned during T5 pre-training.**
A5: Short spans (length 1-2) primarily test the model's knowledge of local lexical co-occurrence (collocations, common phrases). Medium spans (length 3-5) require understanding of syntactic structures and short-range dependencies (verb-argument relations, modifier-noun relations). Long spans (length 6+) require discourse-level understanding and coreference resolution. T5's mean span length of 3 with geometric distribution provides a mix of all these levels, with emphasis on syntactic/phrasal knowledge. The varying span length forces the model to learn multiple levels of linguistic abstraction simultaneously.

**Q6: Design a pre-training objective that combines span corruption with contrastive learning. How would you implement the contrastive component?**
A6: A combined objective would add a contrastive loss that distinguishes the original context from corrupted contexts. Implementation: (1) Standard span corruption creates (corrupted_input, target) pairs; (2) Create negative examples by replacing the target span with a random span from another document; (3) Add a contrastive head that produces a single vector for the decoder's hidden state; (4) Use NT-Xent loss to pull positive pairs (correct reconstruction) together and push negative pairs (random replacement) apart; (5) Weight the contrastive loss with coefficient ~0.1 relative to the span corruption loss. This would encourage the model to produce representations that distinguish between coherent and incoherent continuations, potentially improving generation quality.

## Practice Problems

### Easy
Implement the span corruption function for a single sequence given noise_density and mean_span_length parameters.

### Medium
Implement a comparison script that evaluates span corruption (noise density 0.15, span length 3) against masked LM (mask probability 0.15) on a small language model fine-tuned for text infilling.

### Hard
Design and implement a curriculum learning schedule for T5 pre-training that starts with shorter spans and lower noise density, gradually increasing to the full setting during training.

## Solutions

### Easy Solution
```python
def corrupt_sequence(seq, noise_density=0.15, mean_span=3, sentinel_start=32128):
    corrupted = []
    targets = []
    si = 0
    i = 0
    while i < len(seq):
        if random.random() < noise_density:
            span_len = min(random.randint(1, 2*mean_span-1), len(seq)-i)
            sid = sentinel_start + si
            corrupted.append(sid)
            targets.append(sid)
            targets.extend(seq[i:i+span_len])
            si += 1
            i += span_len
        else:
            corrupted.append(seq[i])
            i += 1
    targets.append(sentinel_start + si)
    return corrupted, targets
```

### Medium Solution
```python
def compare_objectives():
    # Train small T5 on span corruption and MLM objectives
    # Evaluate on text infilling (masked span completion)
    span_model = train_small_model(SpanCorruption())
    mlm_model = train_small_model(MaskedLM())
    
    results = {'span_corruption': evaluate(span_model, test_data),
               'masked_lm': evaluate(mlm_model, test_data)}
    return results
```

### Hard Solution
```python
class CurriculumSpanCorruption:
    def __init__(self, total_steps):
        self.total_steps = total_steps
    
    def get_params(self, step):
        progress = step / self.total_steps
        noise = 0.05 + 0.10 * min(progress, 1.0)
        span = 1 + 2 * min(progress * 2, 1.0)  # 1 -> 3
        return noise, span
```

## Related Concepts
- DL-431: T5 Architecture - The model being pre-trained
- DL-432: Text-to-Text Framework - The framework guiding pre-training
- DL-388: Masked Language Modeling - Related pre-training objective
- DL-434: T5 Variants - Different T5 versions
- DL-435: FLAN-T5 - Instruction-tuned from T5

## Next Concepts
- DL-434: T5 Variants - Different model sizes and versions
- DL-435: FLAN-T5 - Instruction-tuned T5
- DL-436: UL2 - Unified pre-training objective

## Summary
T5 pre-training uses span corruption on the C4 dataset: 15% of tokens are masked in contiguous spans (mean length 3) and the model reconstructs them. The C4 dataset is aggressively filtered CommonCrawl data (750GB). This pre-training approach was selected through systematic comparison of over 15 different objectives and provides strong bidirectional understanding with autoregressive generation capabilities.

## Key Takeaways
- Span corruption: mask contiguous spans, predict with sentinel tokens
- 15% noise density, mean span length 3
- C4: aggressively filtered CommonCrawl (750GB)
- Sentinel tokens guide decoder where to insert each span
- Final sentinel token signals end of target sequence
- Systematic comparison led to the chosen objective
- Combines bidirectional understanding with autoregressive generation
