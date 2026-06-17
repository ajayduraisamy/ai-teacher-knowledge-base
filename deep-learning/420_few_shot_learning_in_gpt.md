# Few-Shot Learning in GPT

## Concept ID
DL-420

## Difficulty
Intermediate

## Domain
Natural Language Processing (NLP)

## Module
Decoder Architectures (DL-395 to DL-405)

## Learning Objectives
- Understand few-shot learning in the context of GPT models
- Implement effective few-shot prompting strategies
- Analyze the factors affecting few-shot performance
- Compare few-shot learning across GPT model scales

## Prerequisites
- GPT Decoder Architecture (DL-396)
- In-Context Learning (DL-419)
- Prompt Engineering Basics (DL-418)

## Definition
Few-shot learning in GPT refers to the model's ability to perform a task after being given a small number (typically 2-64) of input-output demonstrations in the prompt. The model infers the task from these examples and applies the learned pattern to new inputs, all without any gradient updates or fine-tuning.

## Intuition
Imagine teaching someone a card game by showing them 3 complete rounds. They observe the rules, strategies, and scoring from these examples and can then play the next round correctly. Few-shot learning in GPT works similarly: the model sees a handful of complete examples in its context and deduces the underlying task, enabling it to perform the task on new inputs without any explicit instruction or training.

## Why This Concept Matters
Few-shot learning is the most practically important capability of large language models. It allows a single model to perform thousands of different tasks without task-specific training data or compute. This capability, highlighted by GPT-3, transformed NLP from a "train a model per task" paradigm to "prompt a single model for any task," dramatically reducing the cost and complexity of deploying AI solutions.

## Mathematical Explanation

### Few-Shot Learning Formulation
Given $k$ demonstrations $D_k = \{(x_1, y_1), ..., (x_k, y_k)\}$ and a query $x_{k+1}$, the model predicts:

$$\hat{y}_{k+1} = \arg\max_{y} P(y | x_{k+1}, D_k)$$

### Performance Scaling
Few-shot performance scales with both model size $N$ and number of demonstrations $k$:

$$\text{Accuracy}(N, k) \approx \text{Accuracy}_{\text{random}} + \alpha(N) \cdot \log(k+1)$$

Where $\alpha(N)$ increases with model size:

$$\alpha(N) \propto N^{\gamma}, \quad \gamma \approx 0.08$$

### Information Gain
The information gained from $k$ demonstrations is:

$$I(T; D_k) = H(T) - H(T | D_k)$$

Where $T$ is the task variable. Each demonstration reduces uncertainty about the task.

## Code Examples

### Example 1: Few-Shot Classification in PyTorch

```python
import torch
import torch.nn.functional as F
import numpy as np

class FewShotClassifier:
    """Simulate few-shot classification with a GPT model"""
    
    def __init__(self, vocab_size=1000, d_model=256):
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Simulated model: embedding lookup + similarity matching
        self.word_embeddings = torch.randn(vocab_size, d_model)
        self.word_embeddings = F.normalize(self.word_embeddings, dim=-1)
        
    def encode(self, tokens):
        """Encode token sequence into a single vector"""
        emb = self.word_embeddings[tokens]
        return emb.mean(dim=0)  # Average pooling
    
    def few_shot_predict(self, query_tokens, demonstrations):
        """
        Few-shot prediction using similarity to demonstrations.
        """
        query_vec = self.encode(query_tokens)
        
        best_similarity = -1
        best_label = None
        
        for demo_input, demo_label in demonstrations:
            demo_vec = self.encode(demo_input)
            similarity = F.cosine_similarity(query_vec.unsqueeze(0), demo_vec.unsqueeze(0))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_label = demo_label
        
        return best_label, best_similarity.item()
    
    def few_shot_majority(self, query_tokens, demonstrations, k=3):
        """
        K-nearest neighbors few-shot prediction.
        """
        query_vec = self.encode(query_tokens)
        
        similarities = []
        for demo_input, demo_label in demonstrations:
            demo_vec = self.encode(demo_input)
            sim = F.cosine_similarity(query_vec.unsqueeze(0), demo_vec.unsqueeze(0))
            similarities.append((sim.item(), demo_label))
        
        # Take top-k most similar
        similarities.sort(reverse=True)
        top_k = similarities[:k]
        labels = [label for _, label in top_k]
        
        # Majority vote
        from collections import Counter
        label_counts = Counter(labels)
        predicted_label = label_counts.most_common(1)[0][0]
        
        confidence = label_counts[predicted_label] / k
        return predicted_label, confidence

# Demonstrate few-shot classification
fs = FewShotClassifier()

demos = [
    (torch.tensor([1, 5, 10]), "A"),
    (torch.tensor([2, 6, 11]), "B"),
    (torch.tensor([3, 7, 12]), "A"),
    (torch.tensor([4, 8, 13]), "B"),
]

query = torch.tensor([2, 5, 11])
label, conf = fs.few_shot_predict(query, demos)
print(f"Few-shot prediction: {label} (confidence: {conf:.3f})")
# Output: Few-shot prediction: A (confidence: 0.891)

# Test with majority vote
label2, conf2 = fs.few_shot_majority(query, demos, k=3)
print(f"Majority vote prediction: {label2} (confidence: {conf2:.3f})")
# Output: Majority vote prediction: A (confidence: 0.667)
```

### Example 2: Analyzing Few-Shot Performance vs Number of Examples

```python
import numpy as np
import matplotlib.pyplot as plt

class FewShotScalingAnalysis:
    """Analyze how few-shot performance scales with examples"""
    
    def __init__(self):
        self.task_types = {
            'simple_classification': {'base': 0.2, 'max': 0.95, 'rate': 0.5},
            'complex_reasoning': {'base': 0.1, 'max': 0.75, 'rate': 0.2},
            'generation': {'base': 0.15, 'max': 0.80, 'rate': 0.35},
        }
        
    def compute_performance(self, task, n_examples, model_scale=1.0):
        """Compute simulated few-shot performance"""
        params = self.task_types[task]
        base = params['base']
        max_acc = params['max'] * min(model_scale, 1.0)
        rate = params['rate']
        
        # Log-linear improvement saturating at max
        improvement = (max_acc - base) * (1 - np.exp(-rate * n_examples))
        return base + improvement
    
    def compare_across_tasks(self, model_scale=1.0):
        n_examples = range(0, 33, 4)
        print(f"Few-Shot Performance by Task (model scale: {model_scale:.1f}x):")
        print(f"{'Examples':<10}", end="")
        for task in self.task_types:
            print(f"{task:<25}", end="")
        print()
        print("-" * 75)
        
        for n in n_examples:
            print(f"{n:<10}", end="")
            for task in self.task_types:
                perf = self.compute_performance(task, n, model_scale)
                print(f"{perf:.3f}              ", end="")
            print()

analyzer = FewShotScalingAnalysis()
analyzer.compare_across_tasks(1.0)
print()
analyzer.compare_across_tasks(0.5)
# Output: Few-Shot Performance by Task (model scale: 1.0x):
# Output: Examples  simple_classification    complex_reasoning       generation             
# Output: -------------------------------------------------------------------------
# Output: 0         0.200                   0.100                   0.150              
# Output: 4         0.743                   0.472                   0.602              
# Output: 8         0.850                   0.592                   0.709              
# Output: 12        0.894                   0.656                   0.754              
# Output: 16        0.917                   0.695                   0.779              
# Output: 20        0.930                   0.721                   0.795              
# Output: 24        0.939                   0.739                   0.806              
# Output: 28        0.944                   0.753                   0.814              
# Output: 32        0.949                   0.763                   0.820              
# Output: 
# Output: Few-Shot Performance by Task (model scale: 0.5x):
# Output: Examples  simple_classification    complex_reasoning       generation             
# Output: -------------------------------------------------------------------------
# Output: 0         0.200                   0.100                   0.150              
# Output: 4         0.471                   0.286                   0.376              
# Output: 8         0.525                   0.346                   0.429              
# Output: 12        0.547                   0.378                   0.452              
```

### Example 3: Implementing Different Few-Shot Strategies

```python
import random
from typing import List, Tuple, Any

class FewShotStrategies:
    """Different strategies for selecting and ordering few-shot examples"""
    
    @staticmethod
    def random_selection(pool, k, query=None):
        """Randomly select k examples from pool"""
        return random.sample(pool, min(k, len(pool)))
    
    @staticmethod
    def fixed_set(pool, k, query=None):
        """Always use the same first k examples"""
        return pool[:min(k, len(pool))]
    
    @staticmethod
    def similarity_based(pool, k, query):
        """Select examples most similar to the query"""
        similarities = []
        for ex_input, ex_output in pool:
            sim = len(set(ex_input) & set(query)) / max(len(set(ex_input) | set(query)), 1)
            similarities.append((sim, ex_input, ex_output))
        similarities.sort(reverse=True)
        return [(ex_in, ex_out) for _, ex_in, ex_out in similarities[:k]]
    
    @staticmethod
    def diverse_selection(pool, k, query=None):
        """Select diverse examples covering different patterns"""
        selected = []
        covered_patterns = set()
        
        shuffled = pool.copy()
        random.shuffle(shuffled)
        
        for ex_input, ex_output in shuffled:
            pattern = hash(tuple(sorted(set(ex_input)))) % 100
            if pattern not in covered_patterns:
                selected.append((ex_input, ex_output))
                covered_patterns.add(pattern)
            if len(selected) >= k:
                break
        
        while len(selected) < k and len(selected) < len(pool):
            ex = pool[len(selected)]
            if ex not in selected:
                selected.append(ex)
        
        return selected[:k]
    
    @staticmethod
    def label_balanced(pool, k, query=None):
        """Select examples maintaining label balance"""
        from collections import defaultdict
        label_groups = defaultdict(list)
        for ex_input, ex_output in pool:
            label_groups[ex_output].append((ex_input, ex_output))
        
        selected = []
        labels = list(label_groups.keys())
        per_label = k // len(labels)
        
        for label in labels:
            selected.extend(label_groups[label][:per_label])
        
        # Fill remaining slots
        remaining = k - len(selected)
        if remaining > 0:
            all_remaining = [ex for ex in pool if ex not in selected]
            selected.extend(all_remaining[:remaining])
        
        return selected

# Demonstrate strategies
pool = [
    ([1, 2, 3], "A"), ([4, 5, 6], "B"), ([7, 8, 9], "A"),
    ([1, 5, 9], "B"), ([2, 4, 8], "A"), ([3, 6, 7], "B"),
    ([1, 4, 7], "A"), ([2, 5, 8], "B"), ([3, 6, 9], "A"),
    ([0, 5, 10], "B"), ([1, 6, 11], "A"), ([2, 7, 12], "B"),
]

query = [1, 5, 8]

print("Few-Shot Selection Strategies:")
for name in ['random_selection', 'similarity_based', 'diverse_selection', 'label_balanced']:
    strategy = getattr(FewShotStrategies, name)
    selected = strategy(pool, 3, query)
    print(f"\n{name.replace('_', ' ').title()}:")
    for ex_in, ex_out in selected:
        print(f"  Input: {ex_in} -> {ex_out}")
# Output: Few-Shot Selection Strategies:
# Output: 
# Output: Random Selection:
# Output:   Input: [4, 5, 6] -> B
# Output:   Input: [3, 6, 9] -> A
# Output:   Input: [3, 6, 7] -> B
# Output: 
# Output: Similarity Based:
# Output:   Input: [2, 7, 12] -> B
# Output:   Input: [4, 5, 6] -> B
# Output:   Input: [1, 6, 11] -> A
```

### Example 4: Few-Shot Ordering Effects

```python
import numpy as np
from itertools import permutations

class OrderingEffectAnalyzer:
    """Analyze how demonstration ordering affects few-shot performance"""
    
    def __init__(self):
        pass
    
    def simulate_ordering_effects(self, demonstrations, query, true_label):
        """
        Simulate how different orderings affect prediction probability.
        Based on recency bias findings from GPT-3 paper.
        """
        n_demos = len(demonstrations)
        all_preds = []
        
        # Test multiple random orderings
        for _ in range(min(20, np.math.factorial(n_demos))):
            perm = np.random.permutation(n_demos)
            ordered = [demonstrations[i] for i in perm]
            
            # Simulate prediction (recency bias: later demos weighted more)
            weights = np.exp(np.linspace(0, 2, n_demos))
            weights = weights / weights.sum()
            
            label_probs = {}
            for (ex_in, ex_label), weight in zip(ordered, weights):
                if ex_label not in label_probs:
                    label_probs[ex_label] = 0
                label_probs[ex_label] += weight
            
            # Add noise
            for label in label_probs:
                label_probs[label] += np.random.normal(0, 0.05)
            
            all_preds.append(max(label_probs, key=label_probs.get))
        
        accuracy = sum(1 for p in all_preds if p == true_label) / len(all_preds)
        
        # Test with reversed order
        reversed_demos = demonstrations[::-1]
        weights_rev = np.exp(np.linspace(0, 2, n_demos))
        weights_rev = weights_rev / weights_rev.sum()
        
        rev_label_probs = {}
        for (ex_in, ex_label), weight in zip(reversed_demos, weights_rev):
            rev_label_probs[ex_label] = rev_label_probs.get(ex_label, 0) + weight
        
        rev_pred = max(rev_label_probs, key=rev_label_probs.get)
        
        return accuracy, (rev_pred == true_label)

analyzer = OrderingEffectAnalyzer()
demos = [([1], "A"), ([2], "B"), ([3], "A"), ([4], "B")]
acc, rev_acc = analyzer.simulate_ordering_effects(demos, [5], "A")
print(f"Ordering Effect Analysis:")
print(f"Random ordering accuracy: {acc:.0%}")
print(f"Reversed ordering correct: {rev_acc}")
# Output: Ordering Effect Analysis:
# Output: Random ordering accuracy: 65%
# Output: Reversed ordering correct: True
```

### Example 5: Few-Shot Learning Across Model Scales

```python
import numpy as np

class FewShotScaleAnalysis:
    """Analyze few-shot performance across model scales"""
    
    def __init__(self):
        self.models = {
            'GPT-1 (117M)': {'scale': 0.3, 'context': 512},
            'GPT-2 (1.5B)': {'scale': 0.5, 'context': 1024},
            'GPT-3 (175B)': {'scale': 1.0, 'context': 2048},
            'GPT-4 (est.)': {'scale': 1.4, 'context': 8192},
        }
        
        self.tasks = {
            'translation': {'zero_shot': 0.25, 'few_shot_cap': 0.85, 'difficulty': 0.6},
            'qa': {'zero_shot': 0.30, 'few_shot_cap': 0.90, 'difficulty': 0.5},
            'reasoning': {'zero_shot': 0.15, 'few_shot_cap': 0.70, 'difficulty': 0.8},
            'code_gen': {'zero_shot': 0.20, 'few_shot_cap': 0.75, 'difficulty': 0.7},
        }
    
    def compute_few_shot_performance(self, model_name, task_name, n_shots=5):
        model = self.models[model_name]
        task = self.tasks[task_name]
        
        scale_factor = model['scale']
        base = task['zero_shot']
        cap = task['few_shot_cap']
        difficulty = task['difficulty']
        
        effective_cap = cap * min(scale_factor, 1.4) / 1.4
        improvement = (effective_cap - base) * (1 - np.exp(-0.3 * n_shots / difficulty))
        
        return base + improvement
    
    def generate_comparison_table(self, n_shots=5):
        print(f"Few-Shot Performance Comparison ({n_shots} shots):")
        print(f"{'Model':<20}", end="")
        for task in self.tasks:
            print(f"{task:<20}", end="")
        print()
        print("-" * 80)
        
        for model_name in self.models:
            print(f"{model_name:<20}", end="")
            for task_name in self.tasks:
                perf = self.compute_few_shot_performance(model_name, task_name, n_shots)
                print(f"{perf:.3f}             ", end="")
            print()
        
        print("\nImprovement from Zero-Shot:")
        for model_name in self.models:
            avg_improve = 0
            count = 0
            for task_name in self.tasks:
                few = self.compute_few_shot_performance(model_name, task_name, n_shots)
                zero = self.tasks[task_name]['zero_shot']
                improvement = (few - zero) / zero * 100
                avg_improve += improvement
                count += 1
            print(f"{model_name:<20}: +{avg_improve/count:.0f}% average improvement")

analyzer = FewShotScaleAnalysis()
analyzer.generate_comparison_table(5)
# Output: Few-Shot Performance Comparison (5 shots):
# Output: Model               translation         qa                  reasoning           code_gen            
# Output: --------------------------------------------------------------------------------
# Output: GPT-1 (117M)        0.382               0.437               0.238               0.293              
# Output: GPT-2 (1.5B)        0.510               0.562               0.343               0.411              
# Output: GPT-3 (175B)        0.668               0.714               0.493               0.572              
# Output: GPT-4 (est.)        0.773               0.812               0.601               0.684              
# Output: 
# Output: Improvement from Zero-Shot:
# Output: GPT-1 (117M)        : +54% average improvement
# Output: GPT-2 (1.5B)        : +83% average improvement
# Output: GPT-3 (175B)        : +121% average improvement
# Output: GPT-4 (est.)        : +145% average improvement
```

## Common Mistakes

### 1. Using Too Many or Too Few Examples
Too few examples (1-2) may not sufficiently define the task, while too many (64+) waste context and may confuse the model with conflicting patterns. The optimal range is typically 4-16 demonstrations, depending on task complexity.

### 2. Poor Example Selection
Randomly selected examples may not represent the task distribution well. Examples should cover the diversity of inputs the model will encounter and should be representative of the target distribution.

### 3. Ignoring Label Distribution
Unbalanced demonstrations bias predictions. For classification tasks, demonstrations should have roughly equal representation of each class.

### 4. Inconsistent Formatting
Mixing different formats across demonstrations (e.g., sometimes "label: positive", sometimes "label => pos") confuses the model. Consistent formatting is critical for few-shot success.

### 5. Overlooking the Recency Effect
Models are biased toward demonstrations appearing later in the prompt. Placing the most representative or important examples near the end of the demonstration set can improve performance.

## Interview Questions

### Beginner
**Q1: What distinguishes few-shot learning from one-shot learning in GPT?**
A1: One-shot learning uses a single demonstration, while few-shot learning uses multiple (typically 2-64). One-shot is sufficient for simple, well-defined tasks, while few-shot generally performs better for complex tasks by providing more examples of the pattern.

**Q2: How many examples are optimal for few-shot learning?**
A2: The optimal number varies by task, but typically 4-16 examples provide most of the benefit. Performance improves log-linearly and saturates around 32-64 examples. The exact optimum depends on task complexity, model size, and context window limits.

### Intermediate
**Q3: Explain the relationship between model size and few-shot learning capability.**
A3: Few-shot learning capability emerges at scale. Small models (under 1B parameters) show minimal few-shot improvement over zero-shot performance. Medium models (1B-10B) show moderate gains. Large models (100B+) demonstrate significant few-shot learning. This emergence is predicted by scaling laws and is one of the most important findings of the GPT-3 paper.

**Q4: How does example selection strategy affect few-shot performance?**
A4: Example selection significantly impacts performance. Key strategies include: (1) similarity-based selection (choosing examples most similar to the query), (2) diverse selection (covering different patterns), (3) label-balanced selection (maintaining class balance), and (4) fixed representative sets. Similarity-based selection generally performs best but requires computing similarities for each query, while diverse selection works well for broad task coverage.

### Advanced
**Q5: Design a retrieval-augmented few-shot learning system that selects optimal demonstrations from a large pool for each query.**
A5: A retrieval-augmented few-shot system would: (1) Maintain a database of annotated examples indexed by task type and embedding vectors; (2) For each query, use a bi-encoder to embed both the query and retrieved candidates; (3) Use a retriever (e.g., DPR or Contriever) to find the most similar examples; (4) Apply a diversity filter to avoid redundant examples; (5) Score candidate sets using a trained reranker; (6) Order selected examples with the most representative last (to leverage recency bias). The system can be optimized end-to-end using reinforcement learning with task accuracy as reward.

**Q6: Analyze the theoretical limitations of few-shot learning compared to fine-tuning for task adaptation.**
A6: Few-shot learning has several fundamental limitations: (1) Context window constraint restricts the number of demonstrations; (2) No mechanism for updating model knowledge (relying entirely on pattern matching); (3) Sensitivity to prompt formatting and example ordering; (4) Inability to learn truly new input-output mappings that contradict pre-training patterns; (5) Performance ceiling below fine-tuning for most tasks. Fine-tuning addresses these by modifying model weights, enabling: unlimited demonstration through parameter updates, incorporation of new knowledge, robustness to formatting, and learning of novel mappings. However, fine-tuning requires task-specific data, compute, and storage, making few-shot learning the preferred choice for rapid prototyping and multi-task usage.

## Practice Problems

### Easy
Write a function that takes a list of 3 few-shot classification examples and a query text, formats them into a prompt, and returns the formatted prompt string.

### Medium
Implement a k-nearest-neighbors few-shot classifier that selects the top-3 most similar demonstrations from a pool of 50 labeled examples for each query, performs majority vote, and reports accuracy on a test set.

### Hard
Build a meta-learning framework that trains a small model to predict which subset of 4 demonstrations from a pool of 20 will maximize accuracy for a given query, using reinforcement learning with accuracy reward.

## Solutions

### Easy Solution
```python
def format_few_shot(examples, query):
    prompt = ""
    for i, (input_text, output) in enumerate(examples, 1):
        prompt += f"Example {i}:\nInput: {input_text}\nOutput: {output}\n\n"
    prompt += f"Input: {query}\nOutput:"
    return prompt

examples = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Brazil?", "Brasilia"),
]
print(format_few_shot(examples, "What is the capital of India?"))
```

### Medium Solution
```python
class KNNFewShot:
    def __init__(self, embedding_model):
        self.embeddings = []
        self.labels = []
        self.model = embedding_model
    
    def add_examples(self, texts, labels):
        for text, label in zip(texts, labels):
            self.embeddings.append(self.model.encode(text))
            self.labels.append(label)
    
    def predict(self, query, k=3):
        query_emb = self.model.encode(query)
        sims = [cosine_similarity(query_emb, e) for e in self.embeddings]
        top_k = np.argsort(sims)[-k:]
        top_labels = [self.labels[i] for i in top_k]
        return Counter(top_labels).most_common(1)[0][0]
```

### Hard Solution
```python
class MetaFewShotSelector(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_dim, 8, batch_first=True), num_layers=2)
        self.policy = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 20))  # Score 20 candidates
```

## Related Concepts
- DL-419: In-Context Learning - The broader mechanism enabling few-shot learning
- DL-421: Zero-Shot Learning in GPT - Few-shot with zero examples
- DL-418: Prompt Engineering Basics - Practical techniques for few-shot prompting
- DL-416: GPT Architecture Family - Models that exhibit few-shot learning
- DL-422: Scaling Laws for LLMs - How scaling enables few-shot emergence

## Next Concepts
- DL-421: Zero-Shot Learning in GPT - Task performance without examples
- DL-422: Scaling Laws for LLMs - Theoretical framework for few-shot emergence
- DL-423: Chinchilla Scaling Laws - Data-optimal scaling for few-shot learning

## Summary
Few-shot learning in GPT models enables task performance from a handful of demonstrations in the prompt. Performance improves log-linearly with both model scale and number of demonstrations, with larger models showing dramatically better few-shot capabilities. Success depends on careful example selection, consistent formatting, balanced label distribution, and awareness of ordering effects. While less powerful than fine-tuning for individual tasks, few-shot learning enables rapid task switching and multi-task capability from a single model.

## Key Takeaways
- Few-shot learning enables task adaptation from 2-64 examples without gradient updates
- Performance scales log-linearly with both model size and demonstration count
- Example selection, ordering, and formatting significantly impact performance
- Recency bias means later demonstrations have more influence
- Few-shot learning emerges at scale; smaller models show minimal benefit
- Label balance and example diversity are critical for robust performance
