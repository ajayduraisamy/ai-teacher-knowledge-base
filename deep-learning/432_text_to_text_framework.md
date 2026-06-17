# Text-to-Text Framework

## Concept ID
DL-432

## Difficulty
Beginner

## Domain
Natural Language Processing (NLP)

## Module
Encoder-Decoder Architectures (DL-431 to DL-440)

## Learning Objectives
- Understand the text-to-text framework unifying NLP tasks
- Implement task prefix formatting for different tasks
- Analyze the advantages and limitations of the framework

## Prerequisites
- T5 Architecture (DL-431)
- Encoder-Decoder Models (DL-380)

## Definition
The text-to-text framework, introduced by T5, unifies all NLP tasks into a single input-output format where both the input and output are text strings. Task-specific behavior is controlled through a task prefix added to the input text, enabling a single model to perform classification, translation, summarization, question answering, and regression tasks without task-specific output heads.

## Intuition
Instead of having one model for translation (input: English, output: French), another for classification (input: text, output: label index), and another for summarization (input: article, output: summary), the text-to-text framework treats them all identically: take text, produce text. A task prefix tells the model what to do: "translate English to German: Hello" or "summarize: long article...". This unified approach simplifies model architecture, training, and deployment while enabling zero-shot transfer between tasks.

## Why This Concept Matters
The text-to-text framework demonstrated that a single model could match or exceed task-specific models across diverse NLP benchmarks. This insight was foundational for the development of instruction-tuned models and contributed to the paradigm shift toward unified, multi-task models. It also simplified model serving (one API for all tasks) and enabled easier knowledge sharing between tasks.

## Code Examples

### Example 1: Task Prefix System

```python
class TextToTextSystem:
    """Complete text-to-text task system"""
    
    PREFIXES = {
        'classification': 'classify: ',
        'binary_classification': 'cola: ',
        'sentiment': 'sentiment: ',
        'nli': 'mnli premise: ',
        'qa': 'question: ',
        'summarization': 'summarize: ',
        'translation_en_de': 'translate English to German: ',
        'translation_en_fr': 'translate English to French: ',
        'translation_en_es': 'translate English to Spanish: ',
        'generation': 'generate: ',
        'grammar': 'fix grammar: ',
        'paraphrase': 'paraphrase: ',
    }
    
    LABELS = {
        'cola': {0: 'acceptable', 1: 'unacceptable'},
        'sst2': {0: 'negative', 1: 'positive'},
        'mnli': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    }
    
    @classmethod
    def format_input(cls, task, text, **kwargs):
        prefix = cls.PREFIXES.get(task, '')
        
        if task == 'nli':
            hypothesis = kwargs.get('hypothesis', '')
            return f"{prefix}{text} hypothesis: {hypothesis}"
        elif task == 'qa':
            context = kwargs.get('context', '')
            return f"{prefix}{text} context: {context}"
        elif task.startswith('translation'):
            return f"{prefix}{text}"
        else:
            return f"{prefix}{text}"
    
    @classmethod
    def parse_output(cls, task, output_text, label_set=None):
        """Parse model output back to structured format"""
        if label_set:
            # Try to map output to label
            output_lower = output_text.strip().lower()
            for label_id, label_name in label_set.items():
                if label_name == output_lower:
                    return label_id
            return -1
        
        # For generation tasks, return raw text
        return output_text.strip()

# Demonstrate
system = TextToTextSystem()
test_cases = [
    ('sentiment', 'This movie was fantastic!', {}),
    ('translation_en_de', 'Hello, how are you?', {}),
    ('qa', 'What is the capital of France?', {'context': 'Paris is the capital of France.'}),
    ('nli', 'A man is running.', {'hypothesis': 'A person is exercising.'}),
]

print("Text-to-Text Task Formatting:")
for task, text, kwargs in test_cases:
    formatted = system.format_input(task, text, **kwargs)
    print(f"  {task:25s} -> '{formatted}'")
# Output: Text-to-Text Task Formatting:
# Output:   sentiment                 -> 'sentiment: This movie was fantastic!'
# Output:   translation_en_de         -> 'translate English to German: Hello, how are you?'
# Output:   qa                        -> 'question: What is the capital of France? context: Paris is the capital of France.'
# Output:   nli                       -> 'mnli premise: A man is running. hypothesis: A person is exercising.'
```

### Example 2: Multi-Task Training Data

```python
class MultiTaskDataset:
    """Dataset combining multiple tasks in text-to-text format"""
    
    def __init__(self, task_data):
        self.examples = []
        for task, data in task_data.items():
            for example in data:
                self.examples.append((task, example))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        task, example = self.examples[idx]
        input_text = TextToTextSystem.format_input(task, example['input'], **example.get('kwargs', {}))
        output_text = example['output']
        return input_text, output_text

# Demonstrate multi-task dataset
tasks = {
    'sentiment': [
        {'input': 'Great product!', 'output': 'positive'},
        {'input': 'Terrible service.', 'output': 'negative'},
        {'input': 'It was okay.', 'output': 'neutral'},
    ],
    'summarization': [
        {'input': 'Long article about AI advancements...', 'output': 'AI is advancing rapidly.'},
    ],
    'translation_en_de': [
        {'input': 'Hello', 'output': 'Hallo'},
        {'input': 'Goodbye', 'output': 'Auf Wiedersehen'},
    ],
}

dataset = MultiTaskDataset(tasks)
print(f"Multi-task dataset: {len(dataset)} examples")
for i in range(len(dataset)):
    inp, out = dataset[i]
    print(f"  [{i}] Input: {inp[:60]}... -> Output: {out}")
# Output: Multi-task dataset: 6 examples
# Output:   [0] Input: sentiment: Great product!... -> Output: positive
# Output:   [1] Input: sentiment: Terrible service.... -> Output: negative
# Output:   [2] Input: sentiment: It was okay..... -> Output: neutral
# Output:   [3] Input: summarize: Long article about AI advancements..... -> Output: AI is advancing rapidly.
# Output:   [4] Input: translate English to German: Hello... -> Output: Hallo
# Output:   [5] Input: translate English to German: Goodbye... -> Output: Auf Wiedersehen
```

### Example 3: Text-to-Text Generation with Decoding

```python
import torch
import torch.nn.functional as F

class TextToTextGenerator:
    """Generate text from a text-to-text model"""
    
    def __init__(self, model, tokenizer, max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def generate(self, task, text, **kwargs):
        input_text = TextToTextSystem.format_input(task, text, **kwargs)
        input_ids = self.tokenizer.encode(input_text)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=torch.tensor([input_ids]),
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True,
            )
        
        output_text = self.tokenizer.decode(output_ids[0])
        return TextToTextSystem.parse_output(task, output_text)

class DummyT5Model:
    """Dummy model for demonstration"""
    def generate(self, **kwargs):
        return torch.tensor([[101, 205, 307, 409]])
    
class DummyTokenizer:
    def encode(self, text):
        return [101] + [hash(c) % 100 for c in text[:20]] + [102]
    def decode(self, ids):
        return "Hallo"

generator = TextToTextGenerator(DummyT5Model(), DummyTokenizer())
result = generator.generate('translation_en_de', 'Hello')
print(f"Generated: {result}")
# Output: Generated: Hallo
```

### Example 4: Framework Evaluation Across Tasks

```python
class TextToTextEvaluator:
    """Evaluate text-to-text model across multiple tasks"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate(self, model, task, test_data):
        correct = 0
        total = len(test_data)
        
        for example in test_data:
            input_text = example['input']
            expected = example['output']
            
            if task in TextToTextSystem.LABELS:
                predicted = model(task, input_text)
                if str(predicted).lower() == str(expected).lower():
                    correct += 1
            else:
                predicted = model(task, input_text)
                # For generation, use approximate matching
                if predicted == expected:
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0
        self.results[task] = accuracy
        return accuracy
    
    def overall_performance(self):
        if not self.results:
            return 0.0
        return sum(self.results.values()) / len(self.results)

# Demonstrate evaluation
evaluator = TextToTextEvaluator()

def dummy_model(task, text):
    return "positive" if task == 'sentiment' else "output"

evaluator.evaluate(dummy_model, 'sentiment', [
    {'input': 'Great!', 'output': 'positive'},
    {'input': 'Bad.', 'output': 'negative'},
])
print(f"Sentiment accuracy: {evaluator.results.get('sentiment', 0):.0%}")
# Output: Sentiment accuracy: 50%
```

### Example 5: Custom Task Registration

```python
class ExtensibleTextToText(TextToTextSystem):
    """Extensible text-to-text framework with custom tasks"""
    
    _custom_tasks = {}
    
    @classmethod
    def register_task(cls, name, prefix, output_parser=None):
        cls._custom_tasks[name] = {
            'prefix': prefix,
            'parser': output_parser or (lambda x: x.strip()),
        }
        cls.PREFIXES[name] = prefix
    
    @classmethod
    def parse_output(cls, task, output_text):
        if task in cls._custom_tasks:
            return cls._custom_tasks[task]['parser'](output_text)
        return super().parse_output(task, output_text, 
                                    label_set=cls.LABELS.get(task))

# Register a custom task
ExtensibleTextToText.register_task(
    'code_generation',
    'generate python code: ',
    lambda x: x.strip().strip('```')
)

formatted = ExtensibleTextToText.format_input('code_generation', 'sort a list')
print(f"Custom task: '{formatted}'")
# Output: Custom task: 'generate python code: sort a list'
```

## Common Mistakes

### 1. Overlooking Task Prefix Sensitivity
The exact wording of task prefixes matters significantly. Small changes (e.g., "summarize:" vs "summarization:") can affect performance because the prefix must be consistent with pre-training or fine-tuning data distribution.

### 2. Using Ambiguous Output Formats
For classification tasks, the output must exactly match the labels seen during training (e.g., "positive"/"negative", not "pos"/"neg"). Using different output text for the same label confuses the model.

### 3. Mixing Training and Evaluation Formats
The text-to-text framework requires consistent formatting between training and evaluation. Using different prefix formats during evaluation than those used during training will likely produce incorrect results.

### 4. Ignoring Whitespace and Punctuation
The model treats "classify: " and "classify:" as different prefixes. Consistent formatting, including trailing spaces and punctuation, is essential.

### 5. Assuming Text-to-Text Works for All Tasks
Some tasks naturally produce continuous outputs (regression, sts-b) that are difficult to represent as discrete text. T5 handled this by quantizing continuous values into discrete buckets (e.g., 0-5 score mapped to "0", "0.5", "1", etc.), which introduces quantization error.

## Interview Questions

### Beginner
**Q1: What is the text-to-text framework and what problem does it solve?**
A1: The text-to-text framework unifies all NLP tasks into a single format where both input and output are text strings, differentiated by task prefixes. It solves the problem of having different model architectures and training procedures for each task, enabling a single model to handle classification, generation, translation, and more.

**Q2: How does T5 handle classification tasks in the text-to-text framework?**
A2: T5 converts classification to text generation: instead of predicting a class index, it generates the class name as text. For example, for sentiment analysis, the output is "positive" or "negative" as strings, not integer labels. The model's vocabulary includes all possible class label strings.

### Intermediate
**Q3: What are the limitations of representing all tasks as text-to-text?**
A3: Key limitations include: (1) Regression tasks require quantization (losing precision); (2) The framework is less efficient for classification (generating 10+ characters instead of a single logit); (3) Task prefixes consume tokens and increase input length; (4) Output format must be strictly controlled (case, punctuation, spacing); (5) Tasks with complex output structures (tables, structured data) are difficult to represent as linear text.

**Q4: How does the text-to-text framework enable multi-task learning and transfer?**
A4: By using a common format, the model can be trained on multiple tasks simultaneously by randomly sampling from different task datasets during training. The task prefix tells the model which task to perform. This enables positive transfer between tasks (e.g., translation helping summarization) and allows the model to learn shared representations across tasks.

### Advanced
**Q5: Analyze the relationship between the text-to-text framework and modern instruction tuning. How did T5's approach influence models like FLAN-T5 and GPT-3?**
A5: The text-to-text framework directly inspired instruction tuning. T5 showed that task prefixes could control model behavior. FLAN extended this by using natural language instructions instead of short prefixes. GPT-3's in-context learning takes this further by providing demonstrations in natural language. The progression is: T5 (fixed task prefixes) → FLAN-T5 (natural language instructions) → InstructGPT/GPT-3 (arbitrary instructions + few-shot examples). The fundamental insight that a single model can perform many tasks by varying its input text is the core contribution.

**Q6: Design an extension to the text-to-text framework that handles structured outputs (JSON, tables) while maintaining the simplicity of text generation.**
A6: Several approaches: (1) JSON-formatted outputs with special tokens and constrained decoding to ensure valid JSON; (2) Schema-guided generation where the output schema is included in the input prefix; (3) Hierarchical text-to-text where the model generates a sequence of key-value pairs; (4) Programmatic decoding with grammar constraints that only allow valid tokens for each position based on the target schema. The key challenge is balancing generation flexibility with output structure guarantees.

## Practice Problems

### Easy
Create task prefixes for 5 new tasks (question answering, named entity recognition, text generation, paraphrasing, grammar correction) following T5's convention.

### Medium
Implement a text-to-text dataset class that can load data from multiple task files, format each with the appropriate prefix, and interleave them for multi-task training.

### Hard
Design an output parser for the text-to-text framework that can handle structured outputs (JSON format with specific keys) using constrained decoding.

## Solutions

### Easy Solution
```python
new_prefixes = {
    'ner': 'extract entities: ',
    'text_generation': 'generate: ',
    'paraphrase': 'paraphrase: ',
    'grammar': 'correct grammar: ',
}
```

### Medium Solution
```python
class MultiTaskTextToText(Dataset):
    def __init__(self, task_configs):
        self.examples = []
        for task, path, prefix in task_configs:
            with open(path, 'r') as f:
                data = json.load(f)
            for item in data:
                self.examples.append({
                    'input': prefix + item['input'],
                    'output': item['output'],
                    'task': task
                })
```

### Hard Solution
```python
class ConstrainedJSONDecoder:
    def __init__(self, schema):
        self.schema = schema
        self.state = 'key'  # key, colon, value
    
    def allowed_tokens(self, vocab):
        if self.state == 'key':
            return [vocab[k] for k in self.schema.keys()]
        return list(range(len(vocab)))
```

## Related Concepts
- DL-431: T5 Architecture - The model implementing the framework
- DL-433: T5 Pre-Training - Pre-training for text-to-text
- DL-434: T5 Variants - Different implementations
- DL-435: FLAN-T5 - Instruction-tuned version
- DL-418: Prompt Engineering - Related technique for LLMs

## Next Concepts
- DL-433: T5 Pre-Training - Pre-training methodology
- DL-434: T5 Variants - T5 family
- DL-435: FLAN-T5 - Instruction tuning

## Summary
The text-to-text framework unifies NLP tasks by representing all inputs and outputs as text, differentiated by task prefixes. Introduced by T5, this approach enables a single model to perform classification, generation, translation, and more without task-specific architectures. The framework influenced modern instruction tuning and multi-task learning approaches.

## Key Takeaways
- All tasks represented as text input → text output
- Task prefixes control model behavior
- Classification outputs class names as strings
- Enables multi-task learning and positive transfer
- Requires careful formatting consistency
- Regression tasks require output quantization
- Influenced instruction tuning and in-context learning
- Simplifies model serving and deployment
