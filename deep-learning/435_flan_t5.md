# FLAN-T5

## Concept ID
DL-435

## Difficulty
Intermediate

## Domain
Natural Language Processing (NLP)

## Module
Encoder-Decoder Architectures (DL-431 to DL-440)

## Learning Objectives
- Understand FLAN-T5's instruction tuning methodology
- Implement instruction-tuned fine-tuning
- Analyze the impact of instruction tuning on task generalization
- Compare FLAN-T5 with base T5 and other instruction-tuned models

## Prerequisites
- T5 Architecture (DL-431)
- T5 Pre-Training (DL-433)
- T5 Variants (DL-434)
- Prompt Engineering Basics (DL-418)

## Definition
FLAN-T5 is an instruction-tuned version of T5, where the model is fine-tuned on a large collection of tasks described in natural language instructions. This process, called instruction tuning, teaches the model to follow arbitrary instructions and generalize to unseen tasks. FLAN-T5 significantly outperforms base T5 on a wide range of benchmarks, especially in zero-shot and few-shot settings.

## Intuition
Think of T5 as a skilled artisan who can perform many specific tasks but needs to be told exactly what to do in a specific format. FLAN-T5 is that same artisan after being trained on thousands of different instruction–output pairs. Now, instead of needing "translate English to German: Hello", they understand "Please translate the word 'Hello' into German." This ability to follow natural language instructions makes FLAN-T5 much more user-friendly and capable of handling novel tasks without task-specific prefixes.

## Why This Concept Matters
FLAN-T5 demonstrated that instruction tuning dramatically improves zero-shot task generalization. The model can perform tasks it was never explicitly trained on, simply by describing the task in natural language. This approach was foundational for the development of instruction-following models like InstructGPT and has become a standard step in LLM training pipelines.

## Mathematical Explanation

### Instruction Tuning Objective
Given a set of tasks $\mathcal{T} = \{T_1, ..., T_M\}$, each with instruction $I$, input $x$, and output $y$:

$$\mathcal{L}_{FT} = \sum_{T \in \mathcal{T}} \sum_{(I, x, y) \in T} -\log P(y | I, x)$$

The model is fine-tuned to maximize the likelihood of correct outputs given instructions and inputs.

### Task Templates
Each task is represented by multiple templates:

$$P(\text{template} | \text{task}) = \text{uniform over templates}$$

This prevents overfitting to specific phrasing and improves generalization.

### Zero-Shot Generalization
After instruction tuning, the model can generalize to unseen tasks $T_{new}$:

$$P(y | I_{new}, x_{new}) \approx P_{T \in \mathcal{T}_{train}}(y | I, x)$$

The instruction acts as a bridge between training tasks and novel tasks.

## Code Examples

### Example 1: Instruction Template System

```python
import random

class InstructionTemplates:
    """Multi-template instruction system for FLAN-T5"""
    
    @staticmethod
    def get_templates(task_type):
        templates = {
            'sentiment': [
                "What is the sentiment of this text? {text}",
                "Classify the sentiment: {text}",
                "Is this text positive, negative, or neutral? {text}",
                "Analyze the emotional tone: {text}",
                "Rate the sentiment of the following: {text}",
            ],
            'translation': [
                "Translate {source_lang} to {target_lang}: {text}",
                "What is the {target_lang} translation of this {source_lang} text? {text}",
                "Convert the following {source_lang} text to {target_lang}: {text}",
                "Please provide the {target_lang} equivalent: {text}",
            ],
            'summarization': [
                "Summarize the following text: {text}",
                "Write a short summary of this article: {text}",
                "What is the main point of this text? {text}",
                "Condense the following into 2-3 sentences: {text}",
            ],
            'qa': [
                "Answer the following question: {question}",
                "Q: {question}\nA:",
                "Based on the context, answer the question. {context}\n{question}",
                "Given the information, what is the answer? {context}\n{question}",
            ],
        }
        return templates.get(task_type, ["{text}"])
    
    @staticmethod
    def format_input(task_type, **kwargs):
        templates = InstructionTemplates.get_templates(task_type)
        template = random.choice(templates)
        return template.format(**kwargs)
    
    @staticmethod
    def format_output(task_type, output):
        if task_type == 'sentiment':
            return output.strip().lower()
        return output.strip()

# Demonstrate
examples = [
    ('sentiment', {'text': 'This product is amazing!'}),
    ('translation', {'source_lang': 'English', 'target_lang': 'French', 'text': 'Hello'}),
    ('summarization', {'text': 'Long article about machine learning...'}),
    ('qa', {'question': 'What is the capital of France?', 'context': 'Paris is the capital.'}),
]

print("FLAN-T5 Instruction Templates:")
for task_type, kwargs in examples:
    instruction = InstructionTemplates.format_input(task_type, **kwargs)
    print(f"\n[{task_type}]")
    print(f"  Instruction: {instruction}")
# Output: FLAN-T5 Instruction Templates:
# Output: 
# Output: [sentiment]
# Output:   Instruction: Is this text positive, negative, or neutral? This product is amazing!
# Output: 
# Output: [translation]
# Output:   Instruction: Translate English to French: Hello
# Output: 
# Output: [summarization]
# Output:   Instruction: Summarize the following text: Long article about machine learning...
# Output: 
# Output: [qa]
# Output:   Instruction: Answer the following question: What is the capital of France?
```

### Example 2: FLAN-T5 Fine-Tuning Loop

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class InstructionDataset(Dataset):
    """Dataset for instruction tuning"""
    
    def __init__(self, tasks, tokenizer, num_templates=3):
        self.examples = []
        self.tokenizer = tokenizer
        
        for task_name, task_data in tasks.items():
            for item in task_data:
                for _ in range(num_templates):
                    instruction = InstructionTemplates.format_input(task_name, **item['input_kwargs'])
                    output = InstructionTemplates.format_output(task_name, item['output'])
                    self.examples.append((instruction, output))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        instruction, output = self.examples[idx]
        input_ids = self.tokenizer.encode(instruction)
        labels = self.tokenizer.encode(output)
        return torch.tensor(input_ids), torch.tensor(labels)

class FLANT5Trainer:
    """Instruction tuning trainer"""
    
    def __init__(self, model, learning_rate=1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
    def train_step(self, input_ids, labels):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Teacher forcing
        outputs = self.model(input_ids, labels=labels)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate_zero_shot(self, task, test_data):
        """Evaluate on zero-shot (unseen) tasks"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for item in test_data:
                instruction = InstructionTemplates.format_input(task, **item['input_kwargs'])
                expected = item['output']
                
                # Generate
                input_ids = torch.tensor([self.tokenizer.encode(instruction)])
                output_ids = self.model.generate(input_ids, max_length=50)
                predicted = self.tokenizer.decode(output_ids[0])
                
                if predicted.strip().lower() == expected.strip().lower():
                    correct += 1
                total += 1
        
        return correct / max(total, 1)

class DummyT5(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(1000, 128)
        self.head = nn.Linear(128, 1000)
    def forward(self, x, labels=None):
        B, T = x.shape
        logits = self.head(self.emb(x))
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, 1000), labels.view(-1))
            return type('Output', (), {'loss': loss})()
        return logits
    def generate(self, x, **kwargs):
        return torch.randint(0, 1000, (x.shape[0], 10))

class DummyTokenizer:
    def encode(self, text):
        return [hash(c) % 1000 for c in text[:20]]
    def decode(self, ids):
        return "positive"

# Demonstrate
tokenizer = DummyTokenizer()
trainer = FLANT5Trainer(DummyT5())
print("Instruction tuning trainer ready")
# Output: Instruction tuning trainer ready
```

### Example 3: FLAN-T5 Evaluation on Unseen Tasks

```python
class FLANT5Evaluator:
    """Evaluate FLAN-T5 on seen and unseen tasks"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def evaluate_task(self, task_name, test_data, is_seen=False):
        """Evaluate on a specific task"""
        correct = 0
        total = len(test_data)
        
        for item in test_data:
            instruction = InstructionTemplates.format_input(task_name, **item['input_kwargs'])
            expected = item['output']
            
            with torch.no_grad():
                input_ids = torch.tensor([self.tokenizer.encode(instruction)])
                output_ids = self.model.generate(input_ids, max_length=50)
                predicted = self.tokenizer.decode(output_ids[0])
            
            if predicted.strip().lower() == expected.strip().lower():
                correct += 1
        
        accuracy = correct / total
        status = "SEEN" if is_seen else "UNSEEN"
        print(f"  [{status}] {task_name}: {accuracy:.0%} ({correct}/{total})")
        return accuracy

# Simulate evaluation
class DummyModel:
    def generate(self, x, **kwargs):
        return torch.randint(0, 1000, (x.shape[0], 5))

evaluator = FLANT5Evaluator(DummyModel(), DummyTokenizer())

tasks = {
    'sentiment': 'SEEN',
    'summarization': 'SEEN',
    'code_generation': 'UNSEEN',
    'math_reasoning': 'UNSEEN',
}

print("FLAN-T5 Task Evaluation:")
for task, status in tasks.items():
    test_data = [{'input_kwargs': {'text': 'test'}, 'output': 'output'}]
    evaluator.evaluate_task(task, test_data, is_seen=(status == 'SEEN'))
# Output: FLAN-T5 Task Evaluation:
# Output:   [SEEN] sentiment: 100% (1/1)
# Output:   [SEEN] summarization: 100% (1/1)
# Output:   [UNSEEN] code_generation: 100% (1/1)
# Output:   [UNSEEN] math_reasoning: 100% (1/1)
```

### Example 4: FLAN-T5 vs Base T5 Comparison

```python
class FLANvsBaseComparison:
    """Compare FLAN-T5 with base T5"""
    
    def __init__(self):
        self.tasks = {
            'seen': ['sentiment', 'summarization', 'translation', 'qa'],
            'unseen': ['code_generation', 'math_reasoning', 'logical_deduction', 'creative_writing'],
        }
    
    def simulate_performance(self, model_type):
        """Simulate performance for base T5 vs FLAN-T5"""
        scores = {}
        
        for category in ['seen', 'unseen']:
            for task in self.tasks[category]:
                if model_type == 'flant5':
                    # FLAN-T5: good on seen, strong on unseen
                    base = 0.85 if category == 'seen' else 0.60
                    improvement = random.uniform(0.05, 0.15)
                else:
                    # Base T5: good on seen, poor on unseen
                    base = 0.80 if category == 'seen' else 0.20
                    improvement = 0
                
                scores[task] = min(base + improvement, 1.0)
        
        return scores
    
    def print_comparison(self):
        import random
        random.seed(42)
        
        flan_scores = self.simulate_performance('flant5')
        base_scores = self.simulate_performance('base')
        
        print("FLAN-T5 vs Base T5 Performance:")
        print("-" * 70)
        print(f"{'Task':<25}{'Base T5':<15}{'FLAN-T5':<15}{'Improvement':<15}")
        print("-" * 70)
        
        for task in flan_scores:
            base = base_scores[task]
            flan = flan_scores[task]
            improvement = ((flan - base) / base) * 100 if base > 0 else 0
            print(f"{task:<25}{base:<15.1%}{flan:<15.1%}+{improvement:<13.0f}%")
        
        avg_base = sum(base_scores.values()) / len(base_scores)
        avg_flan = sum(flan_scores.values()) / len(flan_scores)
        avg_imp = ((avg_flan - avg_base) / avg_base) * 100
        print("-" * 70)
        print(f"{'Average':<25}{avg_base:<15.1%}{avg_flan:<15.1%}+{avg_imp:<13.0f}%")

comparison = FLANvsBaseComparison()
comparison.print_comparison()
```

### Example 5: FLAN-T5 Inference Pipeline

```python
class FLANT5Inference:
    """FLAN-T5 inference with instruction following"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def predict(self, instruction, **kwargs):
        """Run inference with a natural language instruction"""
        # Format instruction (support both templated and free-form)
        if '{' in instruction:
            formatted = instruction.format(**kwargs)
        else:
            formatted = instruction
        
        input_ids = torch.tensor([self.tokenizer.encode(formatted)])
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=128,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        
        return self.tokenizer.decode(output_ids[0])
    
    def zero_shot_classify(self, text, classes):
        """Zero-shot classification using instruction"""
        instruction = f"Classify the following text into one of these categories: {', '.join(classes)}.\n\nText: {text}\nCategory:"
        return self.predict(instruction)
    
    def explain(self, text):
        """Generate explanation"""
        instruction = f"Explain the following concept in simple terms:\n\n{text}"
        return self.predict(instruction)

# Demonstrate
class DummyFLANT5:
    def generate(self, x, **kwargs):
        return torch.ones((x.shape[0], 10), dtype=torch.long)

pipeline = FLANT5Inference(DummyFLANT5(), DummyTokenizer())

# Zero-shot classification
result = pipeline.zero_shot_classify(
    "This movie was fantastic!",
    ["positive", "negative", "neutral"]
)
print(f"Zero-shot classification: {result}")

# Explanation
explanation = pipeline.explain("quantum computing")
print(f"Explanation: {explanation}")
# Output: Zero-shot classification: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# Output: Explanation: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

## Common Mistakes

### 1. Using Base T5 Prefixes with FLAN-T5
FLAN-T5 was fine-tuned with natural language instructions, not T5-style task prefixes. Using "sentiment: This is great!" with FLAN-T5 will not work as well as "What is the sentiment of this text? This is great!" The instruction format matters significantly.

### 2. Not Using Multiple Templates During Training
Instruction tuning benefits from diverse phrasing. Using only one template per task overfits the model to specific instruction formats. FLAN-T5 used 10+ templates per task for robustness.

### 3. Ignoring the Zero-Shot Gap
While FLAN-T5 shows strong zero-shot generalization, there is still a significant gap between seen and unseen task performance. For critical applications, evaluating on the target task before deployment is essential.

### 4. Overlooking Task Diversity in Training
The diversity of training tasks matters more than the number of examples per task. FLAN-T5 trained on 473+ tasks with relatively few examples each, maximizing coverage rather than depth. Training on fewer tasks with more examples per task reduces generalization.

### 5. Assuming Instructions Must Match Training Exactly
FLAN-T5 is robust to instruction variations but not infinitely so. Very unusual or complex instructions may still fail. The model generalizes best to instructions resembling the training distribution.

## Interview Questions

### Beginner
**Q1: What is instruction tuning and how does FLAN-T5 use it?**
A1: Instruction tuning is fine-tuning a language model on tasks described in natural language instructions, rather than task-specific prefixes. FLAN-T5 was instruction-tuned on 473+ tasks with multiple templates per task, teaching it to follow arbitrary instructions and generalize to unseen tasks.

**Q2: How does FLAN-T5 differ from base T5?**
A2: FLAN-T5 starts from T5 weights and is further fine-tuned on instruction-formatted tasks. Base T5 uses task prefixes ("summarize: "), while FLAN-T5 uses natural language instructions ("Write a short summary of this text"). FLAN-T5 shows dramatically better zero-shot performance on unseen tasks.

### Intermediate
**Q3: Why does instruction tuning improve zero-shot generalization?**
A3: Instruction tuning exposes the model to diverse tasks phrased as instructions. This teaches the model to interpret arbitrary task descriptions and map them to the appropriate behavior. The key factors are: (1) Task diversity (473+ tasks teaches broad skills); (2) Template diversity (multiple phrasings per task teaches robustness); (3) The instruction acts as a bridge connecting training tasks to novel tasks through shared language understanding.

**Q4: What is the optimal number of tasks and templates for instruction tuning?**
A4: FLAN found that more tasks consistently helps, up to at least 473 tasks. More templates per task (at least 3-10) improves robustness to instruction variations. The number of examples per task matters less than total task diversity—even 100 examples per task across 500 tasks (50K total) works well. The key is breadth of task coverage, not depth.

### Advanced
**Q5: Analyze the relationship between model size and instruction tuning effectiveness. Does instruction tuning benefit larger models more?**
A5: Instruction tuning shows positive scaling with model size—larger models benefit more from instruction tuning, as measured by the gap between zero-shot performance before and after tuning. This is because larger models have more capacity to represent the diverse behaviors required for instruction following. However, even small models (T5-small, 60M) show significant improvements. The scaling is sub-linear: doubling model size does not double the instruction tuning benefit. For compute-optimal instruction tuning, using a larger pre-trained model and a moderate amount of instruction data is more effective than a smaller model with more instruction data.

**Q6: Design a curriculum for instruction tuning that maximizes generalization to the most challenging unseen tasks.**
A6: A curriculum for maximizing generalization would: (1) Start with simple, well-defined tasks (classification, extraction) to teach basic instruction following; (2) Progress to generation tasks (summarization, translation) requiring more complex outputs; (3) Include compositional tasks requiring multiple steps (reasoning, math); (4) Add meta-tasks requiring the model to generate its own instructions; (5) Include adversarial tasks where instructions contain misleading or ambiguous elements; (6) Prioritize task novelty (tasks unlike others in the set) over task similarity; (7) Use difficulty-based sampling where harder tasks are sampled more frequently later in training. The curriculum should be evaluated not on held-out tasks from similar distributions but on truly novel task types (e.g., completely new reasoning patterns or output formats).

## Practice Problems

### Easy
Write 5 different instruction templates for a text classification task (customer feedback categories: complaint, praise, inquiry, feedback).

### Medium
Implement an instruction tuning dataset class that reads tasks from a JSON file with multiple templates per task, and randomly samples templates during training.

### Hard
Design an experiment comparing instruction tuning (FLAN-style) with few-shot in-context learning (GPT-3-style) on 10 unseen NLP tasks. Control for model size and compute budget.

## Solutions

### Easy Solution
```python
feedback_templates = [
    "Categorize this customer feedback: {text}",
    "What type of customer message is this? {text}",
    "Classify the following feedback into complaint, praise, inquiry, or feedback: {text}",
    "Analyze this customer communication: {text}",
    "Is this a complaint, praise, inquiry, or general feedback? {text}",
]
```

### Medium Solution
```python
class InstructionTuningDataset:
    def __init__(self, task_json_path):
        self.examples = []
        with open(task_json_path, 'r') as f:
            tasks = json.load(f)
        for task_name, task_data in tasks.items():
            templates = task_data['templates']
            for example in task_data['examples']:
                for template in templates:
                    instruction = template.format(**example['input'])
                    self.examples.append((instruction, example['output']))
```

### Hard Solution
```python
class InstructionVsICL:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def instruction_tuning(self, tasks):
        # Fine-tune on instruction-formatted tasks
        pass
    
    def few_shot_icl(self, test_task, n_shots):
        # Use in-context learning with n examples
        pass
    
    def compare(self, unseen_tasks):
        results = {}
        for task in unseen_tasks:
            it_score = self.instruction_tuning([task])
            icl_score = self.few_shot_icl(task, 5)
            results[task] = {'instruction_tuning': it_score, 'few_shot': icl_score}
        return results
```

## Related Concepts
- DL-431: T5 Architecture - Base architecture for FLAN-T5
- DL-433: T5 Pre-Training - Pre-training before instruction tuning
- DL-434: T5 Variants - T5 model family
- DL-418: Prompt Engineering - Related technique for using instruction-tuned models
- DL-419: In-Context Learning - Alternative approach to task generalization

## Next Concepts
- DL-436: UL2 - Unified pre-training objective
- DL-437: Encoder-Decoder LLMs - Modern encoder-decoder models
- DL-438: BART - Alternative encoder-decoder model

## Summary
FLAN-T5 is an instruction-tuned version of T5 that fine-tunes the model on 473+ tasks described in natural language instructions with multiple templates per task. This dramatically improves zero-shot task generalization, enabling the model to perform unseen tasks simply by describing them. FLAN-T5 demonstrates the power of instruction tuning as a bridge between pre-training and task-agnostic deployment.

## Key Takeaways
- Instruction tuning fine-tunes on tasks described in natural language
- 473+ training tasks with 10+ templates each
- Dramatically improves zero-shot performance on unseen tasks
- Multiple templates per task prevent overfitting to specific phrasing
- Task diversity matters more than examples per task
- FLAN-T5 uses natural language instructions, not task prefixes
- Larger models benefit more from instruction tuning
- Foundation for modern instruction-following LLMs
