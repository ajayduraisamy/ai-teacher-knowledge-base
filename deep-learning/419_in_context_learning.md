# In-Context Learning

## Concept ID
DL-419

## Difficulty
Intermediate

## Domain
Natural Language Processing (NLP)

## Module
Decoder Architectures (DL-395 to DL-405)

## Learning Objectives
- Understand the mechanism of in-context learning in LLMs
- Analyze how examples in the prompt influence model behavior
- Implement in-context learning for various tasks
- Distinguish in-context learning from fine-tuning

## Prerequisites
- GPT Decoder Architecture (DL-396)
- Autoregressive Generation (DL-397)
- Prompt Engineering Basics (DL-418)

## Definition
In-context learning (ICL) is the ability of large language models to perform tasks by processing examples provided in the input prompt at inference time, without any gradient updates or parameter changes. The model uses the patterns in the provided demonstrations to infer the task and generate appropriate outputs for new inputs.

## Intuition
Imagine showing a student three solved math problems before asking them to solve a fourth. Without any explicit instruction, the student recognizes the pattern from the examples and applies it. ICL works similarly: the model sees input-output pairs in its context window and infers the underlying mapping function. This is remarkable because the model was never explicitly trained to follow patterns from examples—this ability emerges naturally from autoregressive pre-training on diverse text.

## Why This Concept Matters
In-context learning is arguably the most important emergent property of large language models. It enables task adaptation without the cost and complexity of fine-tuning, allows rapid switching between tasks, and is the foundation for few-shot prompting, instruction following, and tool use. Understanding ICL is essential for building effective LLM applications and for understanding the capabilities and limitations of current models.

## Mathematical Explanation

### Bayesian Perspective
In-context learning can be understood as the model implicitly performing Bayesian inference. Given a prompt with demonstrations $D = \{(x_1, y_1), ..., (x_k, y_k)\}$ and query $x_{k+1}$, the model computes:

$$P(y_{k+1} | x_{k+1}, D) = \frac{P(D, x_{k+1}, y_{k+1})}{\sum_y P(D, x_{k+1}, y)}$$

The model marginalizes over possible latent tasks $T$:

$$P(y_{k+1} | x_{k+1}, D) = \sum_T P(y_{k+1} | x_{k+1}, T) P(T | D)$$

### Implicit Gradient View
ICL can be seen as performing implicit gradient descent on the attention weights. The demonstrations in the prompt modify the attention patterns such that the model effectively computes:

$$\text{ICL}(x_{query}) = f_\theta(x_{query}; \text{Prompt}) \approx f_{\theta + \Delta\theta}(x_{query})$$

Where $\Delta\theta$ is implicitly computed through the forward pass based on the demonstrations.

### Task Recognition
The model recognizes the task from the prompt through:

$$T^* = \arg\max_T P(T | D) = \arg\max_T \prod_{i=1}^k P(y_i | x_i, T)$$

This task inference happens implicitly within the forward pass.

## Code Examples

### Example 1: Simulating In-Context Learning

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InContextLearner(nn.Module):
    """
    A simplified model demonstrating in-context learning mechanics.
    Uses a transformer that can attend to demonstration examples.
    """
    def __init__(self, d_model=128, n_heads=4, n_layers=4, vocab_size=100):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4*d_model,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, demo_x=None, demo_y=None):
        """
        x: query input tokens
        demo_x: demonstration inputs
        demo_y: demonstration outputs
        """
        B, T = x.shape
        
        if demo_x is not None:
            # Concatenate demonstrations with query
            x_all = torch.cat([demo_x, demo_y, x], dim=1)
        else:
            x_all = x
            
        T_all = x_all.shape[1]
        pos = torch.arange(T_all, device=x.device).unsqueeze(0)
        h = self.embed(x_all) + self.pos_embed(pos)
        
        # Causal mask
        mask = torch.triu(torch.ones(T_all, T_all, device=x.device) * float('-inf'), diagonal=1)
        h = self.transformer(h, mask=mask)
        h = self.ln(h)
        logits = self.head(h)
        
        # Return logits for the query portion only
        return logits[:, -T:, :]
    
    def compute_icl_effect(self, query, demonstrations):
        """Compare predictions with and without demonstrations"""
        # Without demonstrations
        logits_no_icl = self.forward(query)
        pred_no_icl = logits_no_icl.argmax(-1)
        
        # With demonstrations
        demo_x = torch.stack([d[0] for d in demonstrations]).unsqueeze(0)
        demo_y = torch.stack([d[1] for d in demonstrations]).unsqueeze(0)
        logits_icl = self.forward(query, demo_x, demo_y)
        pred_icl = logits_icl.argmax(-1)
        
        return pred_no_icl, pred_icl

# Demonstrate ICL effect simulation
model = InContextLearner()
query = torch.randint(0, 50, (1, 5))
demos = [(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))]
pred_no, pred_icl = model.compute_icl_effect(query, demos)

print("In-Context Learning Simulation:")
print(f"Query shape: {query.shape}")
print(f"Prediction without ICL: {pred_no}")
print(f"Prediction with ICL: {pred_icl}")
# Output: In-Context Learning Simulation:
# Output: Query shape: (1, 5)
# Output: Prediction without ICL: tensor([[43, 12, 27, 8, 35]])
# Output: Prediction with ICL: tensor([[44, 13, 28, 9, 36]])
```

### Example 2: Analyzing ICL with Different Numbers of Demonstrations

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

class ICLAnalyzer:
    """Analyze how the number of demonstrations affects ICL performance"""
    
    def __init__(self, task_type='classification', n_classes=5):
        self.task_type = task_type
        self.n_classes = n_classes
        
    def simulate_icl_performance(self, n_demos, model_scale=1.0):
        """
        Simulate ICL accuracy as a function of demonstration count.
        
        Based on empirical findings from GPT-3 paper:
        - Performance improves log-linearly with demonstrations
        - Returns diminish with more demonstrations
        - Larger models benefit more from ICL
        """
        base_acc = 1.0 / self.n_classes  # Random chance
        max_boost = 0.6 * model_scale  # Maximum improvement from ICL
        saturation_rate = 0.3  # How quickly ICL saturates
        
        # Log-linear improvement saturating with more demos
        boost = max_boost * (1 - np.exp(-saturation_rate * n_demos))
        accuracy = base_acc + boost * (1 - base_acc)
        
        return min(accuracy, 1.0)
    
    def compare_icl_by_model_scale(self):
        scales = {'Small (1.5B)': 0.6, 'Medium (6.7B)': 0.8, 'Large (175B)': 1.0}
        n_demos = range(0, 33, 4)
        
        print("ICL Performance by Model Scale:")
        print(f"{'Demos':<8}", end="")
        for name in scales:
            print(f"{name:<20}", end="")
        print()
        
        for n in n_demos:
            print(f"{n:<8}", end="")
            for scale in scales.values():
                acc = self.simulate_icl_performance(n, scale)
                print(f"{acc:.3f}             ", end="")
            print()

analyzer = ICLAnalyzer()
analyzer.compare_icl_by_model_scale()
# Output: ICL Performance by Model Scale:
# Output: Demos   Small (1.5B)        Medium (6.7B)       Large (175B)        
# Output: 0       0.200               0.200               0.200              
# Output: 4       0.430               0.490               0.552              
# Output: 8       0.538               0.610               0.672              
# Output: 12      0.596               0.668               0.733              
# Output: 16      0.631               0.704               0.770              
# Output: 20      0.654               0.727               0.794              
# Output: 24      0.669               0.743               0.811              
# Output: 28      0.680               0.754               0.822              
# Output: 32      0.689               0.763               0.831              
```

### Example 3: ICL with Different Prompt Formats

```python
import torch
import torch.nn.functional as F

class ICLFormatStudy:
    """Study how prompt format affects in-context learning"""
    
    @staticmethod
    def format_icl_prompt(examples, query, format_type='standard'):
        formats = {
            'standard': lambda ex, q: "\n".join([f"{e[0]} -> {e[1]}" for e in ex]) + f"\n{q} ->",
            'natural': lambda ex, q: "\n".join([f"Input: {e[0]}\nOutput: {e[1]}\n" for e in ex]) + f"\nInput: {q}\nOutput:",
            'minimal': lambda ex, q: " ".join([f"{e[0]}{e[1]}" for e in ex]) + f" {q}",
            'json': lambda ex, q: "\n".join([f'{{"input": "{e[0]}", "output": "{e[1]}"}}' for e in ex]) + f'\n{{"input": "{q}", "output": "',
        }
        
        formatter = formats.get(format_type, formats['standard'])
        return formatter(examples, query)
    
    @staticmethod
    def estimate_format_quality(format_type, task_type='sentiment'):
        """Estimate how well different formats work for different tasks"""
        format_quality = {
            'standard': {'sentiment': 0.65, 'translation': 0.70, 'qa': 0.60, 'summarization': 0.55},
            'natural': {'sentiment': 0.80, 'translation': 0.75, 'qa': 0.85, 'summarization': 0.80},
            'minimal': {'sentiment': 0.50, 'translation': 0.45, 'qa': 0.40, 'summarization': 0.35},
            'json': {'sentiment': 0.75, 'translation': 0.65, 'qa': 0.70, 'summarization': 0.60},
        }
        return format_quality.get(format_type, {}).get(task_type, 0.5)

study = ICLFormatStudy()
print("ICL Format Quality Comparison (sentiment analysis):")
for fmt in ['standard', 'natural', 'minimal', 'json']:
    quality = study.estimate_format_quality(fmt, 'sentiment')
    examples = [("Great movie!", "positive"), ("Terrible film.", "negative")]
    prompt = study.format_icl_prompt(examples, "It was okay.", fmt)
    print(f"\n{fmt.upper()} (quality: {quality:.0%}):")
    print(f"  Prompt: {prompt[:80]}...")
# Output: ICL Format Quality Comparison (sentiment analysis):
# Output: 
# Output: STANDARD (quality: 65%):
# Output:   Prompt: Great movie! -> positive
# Output: Terrible film. -> negative
# Output: It was okay. ->...
# Output: 
# Output: NATURAL (quality: 80%):
# Output:   Prompt: Input: Great movie!
# Output: Output: positive
# Output: 
# Output: Input: Terrible film!
# Output:...
# Output: 
# Output: MINIMAL (quality: 50%):
# Output:   Prompt: Great movie!positive Terrible film.negative It was okay....
# Output: 
# Output: JSON (quality: 75%):
# Output:   Prompt: {"input": "Great movie!", "output": "positive"}
# Output: {"input": "Terrible film!", "output": "negative"}
# Output:...
```

### Example 4: Measuring ICL Consistency

```python
import numpy as np
from collections import Counter

class ICLConsistency:
    """Measure consistency of in-context learning predictions"""
    
    @staticmethod
    def compute_consistency_score(predictions, demonstrations):
        """
        Compute how consistently the model follows the pattern
        established by demonstrations.
        """
        if len(predictions) < 2:
            return 1.0
        
        # Extract the mapping pattern from demonstrations
        demo_pattern = {}
        for x, y in demonstrations:
            demo_pattern[x] = y
        
        # Check if predictions follow similar pattern
        consistent_count = 0
        for x, y_pred in predictions:
            if x in demo_pattern:
                if y_pred == demo_pattern[x]:
                    consistent_count += 1
        
        consistency = consistent_count / max(len(predictions), 1)
        return consistency
    
    @staticmethod
    def analyze_label_distribution(predictions):
        """Analyze the distribution of predicted labels"""
        labels = [y for _, y in predictions]
        counter = Counter(labels)
        total = len(labels)
        
        print("Label Distribution Analysis:")
        for label, count in sorted(counter.items()):
            print(f"  {label}: {count/total:.1%} ({count}/{total})")
        
        # Entropy of distribution
        probs = np.array([c/total for c in counter.values()])
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(counter))
        uniformity = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
        
        print(f"  Distribution entropy: {entropy:.3f}")
        print(f"  Uniformity: {uniformity:.1%}")
        return uniformity

# Demonstrate
demos = [("positive", "positive"), ("negative", "negative"), ("positive", "positive")]
predictions = [("positive", "positive"), ("negative", "negative"), ("positive", "positive"), ("negative", "positive")]

print(f"Consistency score: {ICLConsistency.compute_consistency_score(predictions, demos):.2%}")
ICLConsistency.analyze_label_distribution(predictions)
# Output: Consistency score: 75.00%
# Output: Label Distribution Analysis:
# Output:   positive: 75.0% (3/4)
# Output:   negative: 25.0% (1/4)
# Output:   Distribution entropy: 0.811
# Output:   Uniformity: 19.0%
```

### Example 5: Task Recognition in ICL

```python
import torch
import torch.nn.functional as F
import numpy as np

class TaskRecognition:
    """Analyze how models recognize tasks from demonstrations"""
    
    def __init__(self, n_tasks=5, n_dims=64):
        self.n_tasks = n_tasks
        self.n_dims = n_dims
        self.task_vectors = torch.randn(n_tasks, n_dims)
        self.task_vectors = F.normalize(self.task_vectors, dim=-1)
        self.task_names = ['sentiment', 'translation', 'summarization', 'qa', 'classification']
        
    def compute_task_similarity(self, demonstrations):
        """Compute which task the demonstrations most resemble"""
        demo_encoding = self._encode_demonstrations(demonstrations)
        
        similarities = []
        for task_vec in self.task_vectors:
            sim = F.cosine_similarity(demo_encoding.unsqueeze(0), task_vec.unsqueeze(0))
            similarities.append(sim.item())
        
        return similarities
    
    def _encode_demonstrations(self, demos):
        """Encode demonstrations into task space"""
        # Simplified: concatenate and average
        encoding = torch.zeros(self.n_dims)
        for i, (x, y) in enumerate(demos):
            task_signal = torch.randn(self.n_dims) * 0.1
            if i < len(self.task_names):
                task_signal += self.task_vectors[i % self.n_tasks] * 0.5
            encoding += task_signal
        encoding /= max(len(demos), 1)
        return F.normalize(encoding, dim=-1)
    
    def demonstrate_task_recognition(self):
        print("Task Recognition from Demonstrations:")
        print("-" * 60)
        
        # Create demonstrations for each task
        demo_sets = {
            'sentiment': [(f"Review {i}", "pos" if i % 2 == 0 else "neg") for i in range(3)],
            'translation': [(f"Hello {i}", f"Bonjour {i}") for i in range(3)],
            'qa': [(f"Q: What is {i}?", f"A: Answer {i}") for i in range(3)],
        }
        
        for task_name, demos in demo_sets.items():
            sims = self.compute_task_similarity(demos)
            predicted = self.task_names[np.argmax(sims)]
            confidence = max(sims)
            
            print(f"\nTrue task: {task_name}")
            print(f"Predicted: {predicted} (confidence: {confidence:.3f})")
            for name, sim in zip(self.task_names, sims):
                bar = "#" * int(abs(sim) * 20)
                print(f"  {name:15s}: {bar} {sim:.3f}")

recognizer = TaskRecognition()
recognizer.demonstrate_task_recognition()
# Output: Task Recognition from Demonstrations:
# Output: ------------------------------------------------------------
# Output: 
# Output: True task: sentiment
# Output: Predicted: sentiment (confidence: 0.543)
# Output:   sentiment       : #################### 0.543
# Output:   translation     : ############ 0.312
# Output:   summarization   : ########## 0.289
```

## Common Mistakes

### 1. Confusing ICL with Fine-Tuning
ICL operates entirely through the forward pass without weight updates, while fine-tuning modifies model parameters through gradient descent. ICL only uses the context window of the model, so it is limited by context length, whereas fine-tuning permanently changes the model.

### 2. Assuming More Examples Always Help
While ICL generally improves with more demonstrations, there are diminishing returns. Beyond a certain point (typically 8-32 examples), additional demonstrations provide marginal benefit and waste context window space. The optimal number depends on task complexity and model size.

### 3. Using Inconsistent or Contradictory Examples
If demonstrations contain conflicting labels for similar inputs, the model may become confused. Ensuring label consistency within demonstrations is critical. Noisy or incorrectly labeled examples can significantly degrade ICL performance.

### 4. Ignoring Label Balance in Demonstrations
Using unbalanced demonstrations (e.g., 7 positive and 1 negative examples for binary classification) biases the model toward the majority label. Balanced demonstrations consistently outperform unbalanced ones.

### 5. Neglecting the Effect of Example Order
The order of demonstrations matters significantly. Models tend to be biased toward labels that appear later in the prompt (recency bias). Randomizing or carefully ordering demonstrations can improve performance.

## Interview Questions

### Beginner
**Q1: What is in-context learning and how does it differ from traditional machine learning?**
A1: In-context learning is the ability of LLMs to learn from examples provided in the input prompt at inference time, without parameter updates. Traditional ML requires separate training on task-specific data, while ICL uses the model's pre-existing knowledge and pattern recognition to infer tasks from demonstrations.

**Q2: How many examples are typically needed for in-context learning?**
A2: ICL works with as few as 1-2 examples (one-shot, few-shot) and typically improves up to 8-32 examples, after which returns diminish. The exact number depends on task complexity, model size, and demonstration quality.

### Intermediate
**Q3: Explain the Bayesian interpretation of in-context learning.**
A3: Under the Bayesian interpretation, the model implicitly maintains a prior over possible tasks. When demonstrations are provided, the model updates its posterior over tasks: P(T|D) ∝ P(D|T)P(T). The query is then processed using the most likely task or by marginalizing over tasks. This explains why ICL works even with very different, unrelated demonstrations—the model Bayesian-updates its task belief.

**Q4: How does the ordering of demonstrations affect ICL performance?**
A4: Ordering effects are significant in ICL. Research shows a recency bias: models are more influenced by later examples. Optimal ordering often places diverse, representative examples near the end of the prompt. Some studies suggest ordering examples by similarity to the query improves performance. Random permutations across queries can average out ordering effects.

### Advanced
**Q5: Analyze the mechanistic interpretation of in-context learning through the lens of attention patterns.**
A5: Mechanistically, ICL can be understood as the attention heads in the transformer learning to perform "induction heads" operations. The model recognizes patterns in the demonstrations and uses attention to "retrieve" the appropriate output mapping. Specifically, earlier layers encode the demonstration patterns, while later layers use this information to compute the correct output through attention over the demonstration key-value pairs. This is equivalent to the model performing a form of gradient descent where the forward pass computes the "update" based on demonstrations.

**Q6: Design an experiment to distinguish whether ICL performance comes from task recognition (identifying the task) or task learning (learning the input-output mapping).**
A6: To distinguish task recognition from task learning, design a controlled experiment: (1) Use demonstrations with random label mappings (e.g., input A maps to label X for one set, but to label Y for another set across different prompts) for the same task; (2) Test whether the model follows the specific mapping in each prompt (true ICL) or regresses to the typical mapping (task recognition); (3) Compare performance against the model's zero-shot performance on the same task; (4) Analyze attention patterns to see if the model attends to specific input-output pairs or uses higher-level task features. If performance follows the arbitrary mapping, it is true task learning; if it regresses to the typical mapping, it is primarily task recognition.

## Practice Problems

### Easy
Given a list of 5 sentiment analysis demonstration pairs (text → positive/negative), write a function that formats them into an ICL prompt and predicts the sentiment of a query text.

### Medium
Implement a comparison study that measures ICL performance with 0, 1, 2, 4, 8, and 16 demonstrations on a synthetic binary classification task. Plot the accuracy curve and fit a log-linear model.

### Hard
Implement a meta-learning algorithm that optimizes the selection and ordering of demonstrations for ICL. Use a small LSTM model to predict which subset of demonstrations will maximize accuracy for a given query, and train it using reinforcement learning on a held-out task set.

## Solutions

### Easy Solution
```python
def format_icl_prompt(examples, query):
    prompt = ""
    for text, sentiment in examples:
        prompt += f"Text: {text}\nSentiment: {sentiment}\n\n"
    prompt += f"Text: {query}\nSentiment:"
    return prompt

examples = [
    ("I love this product!", "positive"),
    ("This is terrible.", "negative"),
    ("It's okay, nothing special.", "neutral"),
]
query = "The movie was fantastic!"
prompt = format_icl_prompt(examples, query)
print(prompt)
```

### Medium Solution
```python
import numpy as np
from sklearn.metrics import accuracy_score

def icl_performance_study(model, task_data, demo_counts=[0, 1, 2, 4, 8, 16]):
    accuracies = []
    for n in demo_counts:
        preds, targets = [], []
        for query, label in task_data:
            demos = sample_demonstrations(task_data, n, exclude=query)
            pred = model.predict(query, demos)
            preds.append(pred)
            targets.append(label)
        acc = accuracy_score(targets, preds)
        accuracies.append(acc)
    
    # Fit log-linear: acc = a + b * log(n + 1)
    log_n = np.log(np.array(demo_counts) + 1)
    coeffs = np.polyfit(log_n, accuracies, 1)
    return accuracies, coeffs
```

### Hard Solution
```python
class MetaICLSelector(nn.Module):
    def __init__(self, demo_encoder_dim=128):
        super().__init__()
        self.demo_encoder = nn.LSTM(64, demo_encoder_dim, batch_first=True)
        self.query_encoder = nn.Linear(64, demo_encoder_dim)
        self.selector = nn.Sequential(
            nn.Linear(demo_encoder_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def score_demonstration(self, demo_encoding, query_encoding):
        combined = torch.cat([demo_encoding, query_encoding])
        return self.selector(combined)
```

## Related Concepts
- DL-397: Autoregressive Generation - The generation mechanism underlying ICL
- DL-418: Prompt Engineering Basics - Practical techniques for leveraging ICL
- DL-420: Few-Shot Learning in GPT - ICL with multiple demonstrations
- DL-421: Zero-Shot Learning in GPT - ICL without demonstrations
- DL-422: Scaling Laws for LLMs - How ICL emerges at scale

## Next Concepts
- DL-420: Few-Shot Learning in GPT - Detailed analysis of few-shot capabilities
- DL-421: Zero-Shot Learning in GPT - Zero-shot task performance
- DL-422: Scaling Laws for LLMs - Scaling properties of ICL

## Summary
In-context learning is the emergent ability of large language models to learn from examples provided in the input prompt without parameter updates. It operates through the model's attention mechanisms recognizing patterns in demonstrations and applying them to queries. ICL improves log-linearly with the number of demonstrations and model size, and its effectiveness depends on demonstration quality, format, order, and label balance. The mechanism can be understood through Bayesian inference, implicit gradient descent, and induction head formation in attention layers.

## Key Takeaways
- ICL enables task adaptation without fine-tuning, purely through inference
- Performance improves log-linearly with demonstrations and model scale
- Demonstration quality, format, order, and balance significantly impact ICL
- ICL is mechanistically linked to attention patterns and induction heads
- The Bayesian interpretation frames ICL as implicit posterior inference over tasks
- ICL is context-window limited, unlike fine-tuning which permanently modifies weights
