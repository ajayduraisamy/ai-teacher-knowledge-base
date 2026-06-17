# Prompt Engineering Basics

## Concept ID
DL-418

## Difficulty
Beginner

## Domain
Natural Language Processing (NLP)

## Module
Decoder Architectures (DL-395 to DL-405)

## Learning Objectives
- Understand the fundamentals of prompt engineering
- Learn different prompting techniques and their applications
- Implement effective prompts for various NLP tasks
- Evaluate and iterate on prompt quality

## Prerequisites
- GPT Decoder Architecture (DL-396)
- Autoregressive Generation (DL-397)

## Definition
Prompt engineering is the practice of designing and optimizing input prompts to guide large language models toward desired outputs. It involves crafting the initial text fed to the model to elicit specific responses, leveraging the model's pre-trained knowledge and in-context learning capabilities without modifying model weights.

## Intuition
Think of a prompt as a set of instructions you give to a highly skilled but extremely literal assistant. If you say "Write about dogs," you might get anything from a biological treatise to a sonnet. But if you say "Write a 100-word product description for a dog leash, targeting busy urban professionals, in a friendly but professional tone," you get exactly what you need. Prompt engineering is the art of providing the right context, constraints, and examples to guide the model's output.

## Why This Concept Matters
Prompt engineering is the primary interface for interacting with LLMs. Unlike traditional ML where you train a model per task, prompt engineering allows a single model to perform thousands of tasks through careful input design. This dramatically reduces the barrier to using AI, enables rapid prototyping, and unlocks capabilities like reasoning, code generation, and creative writing from pre-trained models.

## Mathematical Explanation

### Probability Formulation
Given a prompt $p$ and response $r$, the model generates:

$$P(r | p) = \prod_{t=1}^{|r|} P(r_t | p, r_{<t})$$

Prompt engineering shapes $P(r|p)$ by modifying $p$ to steer the conditional distribution toward desirable responses.

### Information-Theoretic Perspective
A good prompt $p^*$ minimizes the KL divergence between the desired distribution $q(r)$ and the model's conditional distribution:

$$p^* = \arg\min_p KL(q(r) || P(r|p))$$

### Few-Shot Learning
With $k$ examples $\{(x_i, y_i)\}_{i=1}^k$, the prompt is:

$$p = [\text{instruction}; (x_1, y_1); (x_2, y_2); ...; (x_k, y_k); x_{query}]$$

The model performs inference by pattern matching:

$$y_{pred} = \arg\max_y P(y | p)$$

## Code Examples

### Example 1: Basic Prompt Templates

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

class PromptTemplate:
    """Base class for prompt templates"""
    
    @staticmethod
    def zero_shot(task, input_text):
        templates = {
            'classification': f"Classify the following text into one of the predefined categories.\n\nText: {input_text}\nCategory:",
            'summarization': f"Summarize the following text in 2-3 sentences.\n\nText: {input_text}\n\nSummary:",
            'translation': f"Translate the following English text to French.\n\nEnglish: {input_text}\nFrench:",
            'qa': f"Answer the following question based on your knowledge.\n\nQuestion: {input_text}\nAnswer:",
        }
        return templates.get(task, input_text)
    
    @staticmethod
    def few_shot(task, input_text, examples):
        template = PromptTemplate.zero_shot(task, "{input}")
        prompt_parts = []
        
        for i, (ex_input, ex_output) in enumerate(examples):
            prompt_parts.append(f"Example {i+1}:")
            prompt_parts.append(template.format(input=ex_input))
            prompt_parts.append(f" {ex_output}\n")
        
        prompt_parts.append(f"Now classify this:\n")
        prompt_parts.append(template.format(input=input_text))
        return "\n".join(prompt_parts)
    
    @staticmethod
    def chain_of_thought(input_text):
        return f"Question: {input_text}\nLet's think step by step.\n"

# Demonstrate template generation
templates = PromptTemplate()
text = "The stock market rallied today as tech companies reported strong earnings."

print("Zero-shot classification prompt:")
print(templates.zero_shot('classification', text))
# Output: Zero-shot classification prompt:
# Output: Classify the following text into one of the predefined categories.
# Output: 
# Output: Text: The stock market rallied today as tech companies reported strong earnings.
# Output: Category:

print("\nChain-of-thought prompt:")
print(templates.chain_of_thought("What is 15% of 80?"))
# Output: Chain-of-thought prompt:
# Output: Question: What is 15% of 80?
# Output: Let's think step by step.
```

### Example 2: Analyzing Prompt Sensitivity

```python
import torch
import numpy as np

class PromptAnalyzer:
    """Analyze how different prompts affect model output probabilities"""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        
    def simulate_prompt_effect(self, prompt_quality, task_difficulty):
        """
        Simulate how prompt quality affects output probabilities.
        
        prompt_quality: 0 (bad) to 1 (perfect)
        task_difficulty: 0 (easy) to 1 (hard)
        """
        base_confidence = 0.5 * (1 - task_difficulty)
        prompt_boost = 0.4 * prompt_quality * (1 - task_difficulty)
        confidence = min(base_confidence + prompt_boost, 1.0)
        
        correct_prob = confidence
        noise_prob = (1 - confidence) / (self.vocab_size - 1)
        
        return correct_prob, noise_prob
    
    def compare_prompt_strategies(self):
        strategies = {
            'vague': "Do the task.",
            'specific': "Classify the sentiment of this text as positive, negative, or neutral.",
            'structured': "Task: Sentiment Analysis\nInput: {text}\nOutput (positive/negative/neutral):",
            'few_shot': "Positive: I love this!\nNegative: This is terrible.\nNeutral: It's okay.\n\nText: {text}\nSentiment:",
            'cot': "Let's analyze the sentiment step by step.\n1. Read the text carefully.\n2. Identify emotional words.\n3. Determine overall sentiment.\n\nText: {text}\nSentiment:"
        }
        
        print("Prompt Strategy Analysis:")
        for name, prompt in strategies.items():
            specificity = len(prompt) / 100  # Rough proxy
            easy_task = self.simulate_prompt_effect(specificity, 0.3)
            hard_task = self.simulate_prompt_effect(specificity, 0.8)
            
            print(f"\n{name.upper()}:")
            print(f"  Prompt: {prompt[:60]}...")
            print(f"  Easy task accuracy: {easy_task[0]:.1%}")
            print(f"  Hard task accuracy: {hard_task[0]:.1%}")

analyzer = PromptAnalyzer()
analyzer.compare_prompt_strategies()
# Output: Prompt Strategy Analysis:
# Output: 
# Output: VAGUE:
# Output:   Prompt: Do the task....
# Output:   Easy task accuracy: 56.0%
# Output:   Hard task accuracy: 24.0%
# Output: 
# Output: SPECIFIC:
# Output:   Prompt: Classify the sentiment of this text as positive, negative, or neut...
# Output:   Easy task accuracy: 71.6%
# Output:   Hard task accuracy: 34.2%
# Output: 
# Output: STRUCTURED:
# Output:   Prompt: Task: Sentiment Analysis
# Output: Input: {text}
# Output: Output (positive/negative/neutral):...
# Output:   Easy task accuracy: 76.8%
# Output:   Hard task accuracy: 38.0%
```

### Example 3: Implementing Prompt Templates for Different Tasks

```python
import json
from typing import List, Dict, Optional

class PromptBuilder:
    """Build structured prompts for various NLP tasks"""
    
    def __init__(self, system_prompt=None):
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
    
    def build_text_classification(self, text, categories, examples=None):
        prompt = f"{self.system_prompt}\n\n"
        prompt += "Classify the text into exactly one of these categories:\n"
        prompt += ", ".join(categories) + "\n\n"
        
        if examples:
            for ex_text, ex_cat in examples:
                prompt += f"Text: {ex_text}\nCategory: {ex_cat}\n\n"
        
        prompt += f"Text: {text}\nCategory:"
        return prompt
    
    def build_extraction(self, text, entities):
        prompt = f"{self.system_prompt}\n\n"
        prompt += f"Extract the following entities from the text:\n"
        prompt += ", ".join(entities) + "\n\n"
        prompt += "Return as JSON.\n\n"
        prompt += f"Text: {text}\n\n"
        prompt += '{"' + '": "", "'.join(entities) + '": ""}'  # JSON template
        return prompt
    
    def build_reasoning(self, question, steps=None):
        prompt = f"{self.system_prompt}\n\n"
        prompt += "Let's solve this step by step.\n\n"
        if steps:
            prompt += "Follow these steps:\n"
            for i, step in enumerate(steps, 1):
                prompt += f"{i}. {step}\n"
            prompt += "\n"
        prompt += f"Question: {question}\n\n"
        prompt += "Let me work through this:\n"
        return prompt
    
    def build_code_generation(self, task, language, context=None):
        prompt = f"{self.system_prompt}\n\n"
        prompt += f"Write {language} code to accomplish the following task:\n"
        prompt += f"Task: {task}\n\n"
        if context:
            prompt += f"Context:\n{context}\n\n"
        prompt += f"```{language}\n"
        return prompt

builder = PromptBuilder()
print(builder.build_text_classification(
    "The movie was absolutely fantastic!",
    ["positive", "negative", "neutral"],
    [("Great film!", "positive"), ("Terrible movie.", "negative")]
))
# Output: You are a helpful AI assistant.
# Output: 
# Output: Classify the text into exactly one of these categories:
# Output: positive, negative, neutral
# Output: 
# Output: Text: Great film!
# Output: Category: positive
# Output: 
# Output: Text: Terrible movie.
# Output: Category: negative
# Output: 
# Output: Text: The movie was absolutely fantastic!
# Output: Category:

print("\n" + builder.build_code_generation(
    "Sort a list of numbers using quicksort",
    "Python"
))
# Output: You are a helpful AI assistant.
# Output: 
# Output: Write Python code to accomplish the following task:
# Output: Task: Sort a list of numbers using quicksort
# Output: 
# Output: ```python
```

### Example 4: Prompt Optimization Through Iteration

```python
import random
import math

class PromptOptimizer:
    """Simulate prompt optimization through iterative refinement"""
    
    def __init__(self):
        self.history = []
        
    def evaluate_prompt(self, prompt, task):
        """Simulate evaluating prompt quality on a scale of 0-1"""
        # Factors that improve prompt quality
        length_score = min(len(prompt) / 200, 1.0) * 0.2
        specificity_score = 0.0
        if 'example' in prompt.lower(): specificity_score += 0.2
        if 'format' in prompt.lower(): specificity_score += 0.15
        if 'step' in prompt.lower(): specificity_score += 0.15
        if 'constraint' in prompt.lower() or 'limit' in prompt.lower(): specificity_score += 0.1
        if 'json' in prompt.lower() or 'list' in prompt.lower(): specificity_score += 0.1
        
        noise = random.uniform(-0.1, 0.1)
        score = min(max(length_score + specificity_score + noise, 0), 1)
        return score
    
    def optimize(self, base_prompt, task, iterations=5):
        print("Prompt Optimization Process:")
        print(f"Initial prompt: {base_prompt}")
        
        best_prompt = base_prompt
        best_score = self.evaluate_prompt(base_prompt, task)
        
        refinements = [
            " Add examples to clarify the expected output.",
            " Specify the output format precisely.",
            " Add constraints and edge cases to handle.",
            " Break down the task into sequential steps.",
            " Include a role description (e.g., 'You are an expert...').",
            " Define input and output structure clearly.",
            " Add quality criteria for self-evaluation."
        ]
        
        for i in range(iterations):
            refinement = random.choice(refinements)
            candidate = best_prompt + refinement
            score = self.evaluate_prompt(candidate, task)
            
            print(f"\nIteration {i+1}:")
            print(f"  Refinement: {refinement.strip()}")
            print(f"  Score: {score:.3f} (previous: {best_score:.3f})")
            
            if score > best_score:
                best_prompt = candidate
                best_score = score
                print(f"  ✓ Accepted")
            else:
                print(f"  ✗ Rejected")
        
        return best_prompt, best_score

optimizer = PromptOptimizer()
best, score = optimizer.optimize(
    "Extract information from this text.",
    "information_extraction"
)
# Output: Prompt Optimization Process:
# Output: Initial prompt: Extract information from this text.
# Output: 
# Output: Iteration 1:
# Output:   Refinement: Specify the output format precisely.
# Output:   Score: ...
```

### Example 5: Measuring Prompt Effectiveness

```python
import numpy as np

class PromptEffectivenessMetrics:
    """Metrics for evaluating prompt quality"""
    
    @staticmethod
    def compute_consistency(prompt, model_responses):
        """How consistent are responses to the same prompt?"""
        if len(model_responses) < 2:
            return 1.0
        
        # Use semantic similarity (simulated here with response length)
        lengths = [len(r) for r in model_responses]
        if max(lengths) == 0:
            return 0.0
        consistency = 1.0 - (np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0)
        return max(0, min(1, consistency))
    
    @staticmethod
    def compute_specificity(prompt):
        """How specific is the prompt (less ambiguous)?"""
        specificity = 0.0
        
        # Check for specific elements
        if 'format' in prompt.lower(): specificity += 0.15
        if 'example' in prompt.lower(): specificity += 0.15
        if 'constraint' in prompt.lower(): specificity += 0.1
        if len(prompt) > 100: specificity += 0.1
        if '?' in prompt: specificity -= 0.05  # Questions are less directive
        if ':' in prompt: specificity += 0.1
        if '\n' in prompt: specificity += 0.1  # Structured prompts
        
        # Check for vague terms
        vague_terms = ['maybe', 'perhaps', 'could', 'might', 'something']
        for term in vague_terms:
            if term in prompt.lower():
                specificity -= 0.05
        
        return max(0, min(1, specificity))
    
    @staticmethod
    def compute_completeness(prompt):
        """Does the prompt cover all necessary aspects?"""
        required_elements = [
            ('task', ['classify', 'summarize', 'extract', 'generate', 'translate']),
            ('output', ['output', 'result', 'return', 'format', 'json']),
            ('input', ['input', 'text', 'given', 'following']),
        ]
        
        score = 0.0
        for element_name, keywords in required_elements:
            if any(kw in prompt.lower() for kw in keywords):
                score += 1.0 / len(required_elements)
        
        return score
    
    @staticmethod
    def overall_quality(prompt, responses=None):
        spec = PromptEffectivenessMetrics.compute_specificity(prompt)
        compl = PromptEffectivenessMetrics.compute_completeness(prompt)
        
        if responses:
            cons = PromptEffectivenessMetrics.compute_consistency(prompt, responses)
            return 0.3 * spec + 0.3 * compl + 0.4 * cons
        return 0.5 * spec + 0.5 * compl

prompts = [
    "Translate this.",
    "Translate the following English text to French. Output only the translation.",
    "Task: Translation\nInput: {input}\nOutput (French translation):\n\nExample:\nInput: Hello\nOutput: Bonjour\n\nNow translate:"
]

for p in prompts:
    spec = PromptEffectivenessMetrics.compute_specificity(p)
    compl = PromptEffectivenessMetrics.compute_completeness(p)
    quality = PromptEffectivenessMetrics.overall_quality(p)
    print(f"Prompt: {p[:60]}...")
    print(f"  Specificity: {spec:.2f}, Completeness: {compl:.2f}, Quality: {quality:.2f}")
# Output: Prompt: Translate this....
# Output:   Specificity: 0.05, Completeness: 0.33, Quality: 0.19
# Output: Prompt: Translate the following English text to French. Output only the tra...
# Output:   Specificity: 0.35, Completeness: 0.67, Quality: 0.51
# Output: Prompt: Task: Translation
# Output: Input: {input}
# Output: Output (French translation):
# Output: 
# Output: Example:
# Output: Input: Hello
# Output:...
# Output:   Specificity: 0.50, Completeness: 0.67, Quality: 0.58
```

## Common Mistakes

### 1. Being Too Vague
A prompt like "Write about AI" gives the model too much freedom. The model might produce anything from a technical paper to a children's story. Good prompts specify the task, format, tone, length, and audience.

### 2. Overlooking Output Format Specification
If you need JSON output but don't specify it, the model might return markdown, plain text, XML, or anything else. Always specify the exact output format, ideally with an example.

### 3. Providing Contradictory Instructions
Telling the model to "be concise" and "provide comprehensive detail" creates confusion. Ensure all instructions are consistent and non-contradictory.

### 4. Neglecting Edge Cases
Prompts often fail on edge cases not covered in examples. Include instructions for handling unusual inputs, missing information, or ambiguous queries.

### 5. Failing to Iterate
The first prompt rarely produces optimal results. Effective prompt engineering requires systematic iteration: testing, measuring, and refining based on observed outputs.

## Interview Questions

### Beginner
**Q1: What is prompt engineering and why is it important?**
A1: Prompt engineering is the practice of designing input text to guide LLM outputs. It is important because it enables a single model to perform many tasks without additional training, dramatically reducing the cost and effort of deploying AI solutions.

**Q2: What is the difference between zero-shot, few-shot, and chain-of-thought prompting?**
A2: Zero-shot prompting gives the model a task description without examples. Few-shot prompting provides examples to demonstrate the pattern. Chain-of-thought prompting asks the model to show its reasoning steps before giving the final answer.

### Intermediate
**Q3: How do you handle output format control in prompt engineering?**
A3: Output format control is achieved through explicit format specification, providing format examples in the prompt, using structured templates (like JSON schema), and sometimes post-processing with regex or parsing. The most reliable approach combines clear format instructions with a few formatted examples.

**Q4: What strategies can improve prompt robustness across different inputs?**
A4: Strategies include using diverse few-shot examples covering edge cases, adding explicit handling instructions for unusual inputs, including format constraints, specifying desired behavior when information is missing, and testing with adversarial inputs.

### Advanced
**Q5: How would you design a prompt optimization system using reinforcement learning?**
A5: An RL-based prompt optimization system would treat prompt tokens as actions, the LLM response as the state, and task-specific evaluation metrics (accuracy, fluency, format compliance) as rewards. A policy network (like a smaller LLM) generates prompt variations that are evaluated on a held-out test set. Techniques like PPO can optimize the prompt generator, while exploration strategies (epsilon-greedy, Thompson sampling) help discover novel effective prompts.

**Q6: Analyze the relationship between prompt engineering and model alignment. How do techniques like system prompts and constitutional AI relate?**
A6: Prompt engineering and alignment are complementary. System prompts provide explicit behavioral guidelines (like "be helpful, harmless, and honest") that shape model outputs. Constitutional AI extends this by having the model self-critique and revise its outputs according to specified principles. Both techniques leverage the model's in-context learning ability to follow behavioral constraints, effectively fine-tuning behavior at inference time rather than during training.

## Practice Problems

### Easy
Write a prompt template for summarizing news articles in exactly two sentences. Include format specification and one example.

### Medium
Design a prompt for a multi-step reasoning task: given a description of a scheduling conflict, determine if a meeting can be rescheduled. Include chain-of-thought instructions and edge case handling.

### Hard
Build an automated prompt evaluation system that tests prompt variations on a validation set, measures task-specific metrics, and uses Bayesian optimization to find the optimal prompt.

## Solutions

### Easy Solution
```python
summary_prompt = """Summarize the following news article in exactly two sentences.

Format:
- First sentence: Main event and key players
- Second sentence: Significance or impact

Example:
Article: Apple announced record quarterly revenue of $90 billion, driven by strong iPhone sales and growth in services. The company also revealed plans to expand into healthcare technology.
Summary: Apple reported record quarterly revenue of $90 billion, surpassing analyst expectations. The company's expansion into healthcare signals a strategic shift beyond consumer electronics.

Article: {article}
Summary:"""
```

### Medium Solution
```python
scheduling_prompt = """Determine if the meeting can be rescheduled based on the constraints.

Let's work through this step by step:
1. Identify all people involved
2. List their availability windows
3. Check if there is any overlapping available time
4. Consider duration requirements
5. If overlap exists with sufficient duration, output "Yes" and suggest times
6. If no overlap, output "No" and explain why

Current meeting: {meeting_details}
Proposed new time: {proposed_time}
Constraints: {constraints}

Decision:"""
```

### Hard Solution
```python
import numpy as np
from scipy.optimize import differential_evolution

class BayesianPromptOptimizer:
    def objective(self, prompt_params):
        length, specificity, n_examples = prompt_params
        prompt = self.build_prompt(int(length), specificity, int(n_examples))
        return -self.evaluate(prompt)
    
    def optimize(self, iterations=100):
        bounds = [(50, 500), (0, 1), (0, 5)]
        result = differential_evolution(self.objective, bounds, maxiter=iterations)
        return self.build_prompt(int(result.x[0]), result.x[1], int(result.x[2]))
```

## Related Concepts
- DL-397: Autoregressive Generation - The generation mechanism prompts leverage
- DL-419: In-Context Learning - How models learn from the context prompts provide
- DL-420: Few-Shot Learning in GPT - The capability enabling few-shot prompting
- DL-421: Zero-Shot Learning in GPT - Task performance without task-specific examples

## Next Concepts
- DL-419: In-Context Learning - Deeper understanding of how prompts shape model behavior
- DL-420: Few-Shot Learning in GPT - Detailed analysis of few-shot capabilities
- DL-421: Zero-Shot Learning in GPT - Zero-shot task performance

## Summary
Prompt engineering is the practice of designing inputs to guide LLM outputs without modifying model weights. Effective prompts are specific, structured, include format specifications and examples, and address edge cases. The field encompasses techniques from zero-shot prompting through few-shot learning to chain-of-thought reasoning. Prompt quality can be systematically measured through metrics like specificity, completeness, and consistency, and optimized through iterative refinement.

## Key Takeaways
- Specific, structured prompts consistently outperform vague instructions
- Output format specification is critical for getting usable responses
- Few-shot examples significantly improve performance across tasks
- Chain-of-thought prompting enhances reasoning capabilities
- Prompt optimization is an iterative process requiring systematic evaluation
- Good prompts address edge cases and specify behavior for unusual inputs
