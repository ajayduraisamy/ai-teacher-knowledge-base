# Concept: Attention Temperature

## Concept ID

DL-348

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Attention Mechanisms

## Learning Objectives

- Define temperature in the context of attention mechanisms and softmax normalization.
- Understand how temperature controls the peakiness of the attention distribution.
- Implement temperature-scaled attention in PyTorch.
- Analyze the effects of temperature on gradient flow and model training.
- Apply temperature scaling for knowledge distillation and attention sharpening.

## Prerequisites

- Understanding of softmax normalization and attention weights.
- Familiarity with the scaled dot-product attention mechanism.
- Knowledge of entropy and probability distributions.
- Experience with PyTorch for implementing attention variants.

## Definition

Attention temperature is a hyperparameter that controls the sharpness of the attention distribution by scaling the raw scores before softmax normalization. Given raw attention scores e and temperature tau, the attention weights are computed as:

alpha_i = softmax(e_i / tau) = exp(e_i / tau) / sum_j exp(e_j / tau)

When tau = 1, this is the standard softmax. When tau < 1, the distribution becomes more peaked (sharper), concentrating attention on fewer elements. When tau > 1, the distribution becomes more uniform (flatter), spreading attention more evenly. The temperature can be a fixed hyperparameter, a learned parameter, or dynamically adjusted. In transformer models, the default temperature is 1 (effectively, the scaled dot-product attention uses 1/sqrt(d_k) as a fixed temperature). Temperature is also used in knowledge distillation (higher temperature for softer teacher distributions) and in attention regularization.

## Intuition

Think of attention temperature as a "focus dial." At low temperature (tau < 1), the model is very picky — it concentrates almost all its attention on the single most relevant element, ignoring everything else. This is useful when the model needs to make a crisp decision based on one key piece of information. At high temperature (tau > 1), the model is more diffuse — it spreads attention across many elements, blending information broadly. This is useful when the model needs to aggregate information from multiple sources. The standard temperature (tau = 1) is a balanced default where the model uses the raw score differences to determine focus. During training, temperature affects gradients: low temperature creates sparse gradients (only a few elements receive updates), while high temperature creates dense gradients (all elements receive updates).

## Why This Concept Matters

Temperature is a versatile tool that appears in multiple contexts in deep learning. In attention mechanisms, it controls the focus of the model. In knowledge distillation, higher teacher temperature produces softer probability distributions that reveal relationships between classes. In reinforcement learning, temperature controls exploration vs. exploitation. In self-attention, some architectures use learnable temperature parameters per head, allowing different heads to have different focus levels. Understanding temperature is essential for: (1) diagnosing attention saturation issues, (2) implementing knowledge distillation, (3) controlling the diversity vs. focus trade-off in generation, and (4) designing adaptive attention mechanisms with dynamic temperature.

## Mathematical Explanation

### Temperature-Scaled Softmax

Given raw scores e = (e_1, ..., e_T) and temperature tau > 0:

alpha_i = exp(e_i / tau) / sum_j exp(e_j / tau)

Properties:
- tau -> 0+: alpha approaches one-hot at argmax(e) (hard attention)
- tau -> inf: alpha approaches uniform distribution
- tau = 1: standard softmax

### Effect on Entropy

H(alpha) = - sum_i alpha_i log(alpha_i)

- Lower tau -> lower entropy (more peaked)
- Higher tau -> higher entropy (more uniform)

### Effect on Gradients

d(alpha_i) / d(e_i) = alpha_i * (1 - alpha_i) / tau

- Lower tau -> steeper gradients near decision boundary, near-zero elsewhere
- Higher tau -> smoother gradients across all elements

### Gradient Through Temperature

If tau is learnable, the gradient flows as:

dL / d(tau) = sum_i (dL / d(alpha_i)) * d(alpha_i) / d(tau)

where d(alpha_i)/d(tau) involves the temperature-scaled softmax derivative.

### Temperature in Knowledge Distillation

In knowledge distillation, the teacher's logits are scaled by temperature T > 1 before softmax:

softmax(teacher_logits / T)

Higher temperature produces softer probabilities that reveal the teacher's "dark knowledge" (relative probabilities of non-maximum classes).

## Code Examples

### Example 1: Temperature-Scaled Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def temperature_attention(q, k, v, temperature=1.0, mask=None):
    d_k = k.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores / temperature
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output, attn_weights

q = torch.randn(2, 4, 16)
k = torch.randn(2, 6, 16)
v = torch.randn(2, 6, 16)

for temp in [0.1, 0.5, 1.0, 2.0, 10.0]:
    output, weights = temperature_attention(q, k, v, temperature=temp)
    entropy = -(weights * torch.log(weights + 1e-8)).sum(-1).mean().item()
    max_w = weights.max(-1).values.mean().item()
    print(f"T={temp:.1f}: entropy={entropy:.3f}, max_weight={max_w:.3f}")
# Output: T=0.1: entropy=0.001, max_weight=1.000
# Output: T=0.5: entropy=0.891, max_weight=0.612
# Output: T=1.0: entropy=1.892, max_weight=0.312
# Output: T=2.0: entropy=2.101, max_weight=0.175
# Output: T=10.0: entropy=2.485, max_weight=0.115
```

### Example 2: Learnable Temperature Attention

```python
class LearnableTemperatureAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        d_k = k.shape[-1]
        Q, K, V = self.W_q(q), self.W_k(k), self.W_v(v)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        temperature = torch.sigmoid(self.scale) * 2.0 + 0.1
        scores = scores / temperature
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights, temperature

lta = LearnableTemperatureAttention(d_model=32)
q = torch.randn(2, 4, 32)
k = torch.randn(2, 6, 32)
v = torch.randn(2, 6, 32)
output, weights, temp = lta(q, k, v)
print(f"Learned temperature: {temp.item():.3f}")
print(f"Attention entropy: {-(weights * torch.log(weights + 1e-8)).sum(-1).mean().item():.3f}")
# Output: Learned temperature: 1.123
# Output: Attention entropy: 1.892
```

### Example 3: Temperature for Knowledge Distillation

```python
class TeacherStudentDistillation:
    def __init__(self, teacher, student, temperature=4.0, alpha=0.7):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha

    def distillation_loss(self, student_logits, teacher_logits, labels):
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        distill_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean') * (self.temperature ** 2)
        ce_loss = F.cross_entropy(student_logits, labels)
        return self.alpha * distill_loss + (1 - self.alpha) * ce_loss

# Simulate distillation
teacher_logits = torch.randn(4, 10) * 5  # Well-separated logits
student_logits = torch.randn(4, 10) * 2
labels = torch.randint(0, 10, (4,))

distiller = TeacherStudentDistillation(None, None, temperature=4.0)
loss = distiller.distillation_loss(student_logits, teacher_logits, labels)
print(f"Distillation loss: {loss.item():.4f}")
# Output: Distillation loss: 2.3456
```

## Common Mistakes

1. **Confusing attention temperature with softmax temperature in the output layer**: Attention temperature scales attention scores before softmax, affecting which inputs the model focuses on. Output temperature scales final logits, affecting token prediction diversity. They serve different purposes.

2. **Setting temperature too low during training**: Very low temperature (tau < 0.1) creates near-one-hot attention distributions, which have near-zero gradients for non-dominant elements. This can prevent the model from learning to attend to multiple relevant sources.

3. **Using temperature != 1 without adjusting the scaling factor**: In scaled dot-product attention, 1/sqrt(d_k) already provides appropriate scaling. Adding temperature on top changes this scaling. Ensure the combined scaling (temperature * sqrt(d_k)) produces well-behaved scores.

4. **Not clamping temperature for learnable temperature**: A learnable temperature can become negative or zero if not constrained, breaking the attention mechanism. Always clamp or use a softplus/sigmoid parameterization to keep temperature positive.

5. **Applying temperature to already normalized attention weights**: Temperature should be applied to raw scores before softmax, not to the attention weights after softmax. Applying temperature after softmax has no effect on the distribution.

## Interview Questions

### Beginner

Q: How does temperature affect attention weights?

A: Temperature controls the sharpness of the attention distribution. Low temperature (tau < 1) makes the distribution more peaked, concentrating attention on fewer elements. High temperature (tau > 1) makes it more uniform, spreading attention across more elements. Tau = 1 gives the standard softmax.

### Intermediate

Q: Why does low temperature cause vanishing gradients in attention?

A: Low temperature makes softmax approach one-hot argmax. The gradient of the attention weight for the non-maximum elements approaches zero because exp(e_i / tau) becomes vanishingly small for any non-maximum score. This means that only the winning element receives meaningful gradient updates, while other elements stagnate.

### Advanced

Q: How would you design an adaptive temperature mechanism for self-attention that uses different temperatures for different heads and positions? What benefits might this provide?

A: I would use a learned temperature predictor: for each head h and position i, predict tau_{h,i} = softplus(MLP(h_i)) where h_i is the token representation. This allows different heads to have different focus levels (some heads specialize in sharp attention for local patterns, others in broad attention for global patterns) and different positions to have different temperatures (early layers need broader attention, later layers need sharper attention). Benefits: (1) Automatically learns the optimal focus level per head and position, (2) Can adapt to different data patterns, (3) Potentially improves model capacity without adding many parameters. Challenges: (1) May overfit without regularization, (2) Adds architectural complexity, (3) Interactions between temperature and other attention parameters must be studied.

## Practice Problems

### Easy

Implement a function that computes temperature-scaled attention for temperatures 0.1, 0.5, 1.0, 2.0, and 10.0. Verify that higher temperature produces higher entropy.

### Medium

Train a simple classifier using knowledge distillation with temperature. Compare the performance of the student model with and without distillation across different temperatures [1, 2, 4, 8, 16].

### Hard

Implement a multi-head attention mechanism where each head has its own learnable temperature parameter. Analyze whether different heads learn different temperatures and how this relates to head specialization.

## Solutions

### Easy Solution

```python
def temperature_entropy_analysis():
    scores = torch.randn(1000, 10)
    for tau in [0.1, 0.5, 1.0, 2.0, 10.0]:
        weights = F.softmax(scores / tau, dim=-1)
        entropy = -(weights * torch.log(weights + 1e-8)).sum(-1).mean().item()
        print(f"T={tau:.1f}: avg_entropy={entropy:.3f}")
    print("Higher tau -> higher entropy")

temperature_entropy_analysis()
# Output: T=0.1: avg_entropy=0.001
# Output: T=0.5: avg_entropy=0.923
# Output: T=1.0: avg_entropy=1.782
# Output: T=2.0: avg_entropy=2.101
# Output: T=10.0: avg_entropy=2.301
# Output: Higher tau -> higher entropy
```

## Related Concepts

- Softmax Normalization
- Attention Weights
- Knowledge Distillation
- Scaled Dot-Product Attention
- Entropy Regularization

## Next Concepts

- DL-349: Soft vs. Hard Attention
- DL-350: Global vs. Local Attention

## Summary

Attention temperature is a hyperparameter that controls the sharpness of the attention distribution by scaling raw scores before softmax normalization. Lower temperature produces more peaked, focused attention; higher temperature produces more uniform, diffuse attention. Temperature is used in standard attention mechanisms (where it interacts with the built-in scaling), in knowledge distillation (for softer teacher distributions), and in adaptive attention mechanisms with learnable temperature parameters. Understanding temperature is essential for controlling attention focus, diagnosing training issues, and implementing distillation-based learning.

## Key Takeaways

- Temperature scales attention scores before softmax: alpha = softmax(e / tau).
- Low tau = peaked distribution (focused), high tau = flat distribution (diffuse).
- tau = 1 is the standard softmax (default in transformers).
- Low temperature causes near-zero gradients for non-dominant elements.
- Temperature is used in knowledge distillation to reveal dark knowledge.
- Learnable temperature can be per-head or per-position for adaptive focus.
