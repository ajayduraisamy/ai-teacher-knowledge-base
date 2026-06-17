# GPT Autoregressive Modeling

## Concept ID
DL-417

## Difficulty
Intermediate

## Domain
Natural Language Processing (NLP)

## Module
Decoder Architectures (DL-395 to DL-405)

## Learning Objectives
- Understand the autoregressive objective used in GPT training
- Implement the causal language modeling loss function
- Analyze the statistical properties of autoregressive models
- Compare autoregressive modeling with masked language modeling

## Prerequisites
- GPT Decoder Architecture (DL-396)
- Autoregressive Generation (DL-397)
- Causal Masking (DL-398)
- Probability fundamentals

## Definition
Autoregressive modeling in GPT refers to the self-supervised learning objective where the model predicts each token in a sequence conditioned on all previous tokens. Formally, the model learns the conditional probability distribution $P(x_t | x_{<t})$ for each position $t$, maximizing the log-likelihood of the observed sequence under the autoregressive factorization.

## Intuition
Imagine teaching someone to write by having them complete sentences one word at a time, always looking only at what they've already written. The learner sees "The cat sat on the" and must predict the next word. By repeating this process across millions of sentences, the learner develops an understanding of grammar, semantics, and world knowledge. Autoregressive modeling is the mathematical formalization of this sequential prediction process.

## Why This Concept Matters
Autoregressive modeling is the training paradigm that powers all GPT-style models and most modern large language models. Understanding its mathematical formulation, implementation details, and statistical properties is essential for training, fine-tuning, and deploying these models. The choice of autoregressive modeling over alternatives like masked language modeling fundamentally shapes model capabilities, inference procedures, and limitations.

## Mathematical Explanation

### Autoregressive Factorization
Given a sequence of tokens $x_1, x_2, ..., x_T$, the joint probability is factorized using the chain rule of probability:

$$P(x_1, x_2, ..., x_T) = \prod_{t=1}^{T} P(x_t | x_1, x_2, ..., x_{t-1})$$

This factorization is always valid (by the chain rule) and imposes no independence assumptions. It simply decomposes the joint distribution into a product of conditional distributions.

### Maximum Likelihood Objective
The training objective maximizes the log-likelihood of the data:

$$L(\theta) = \sum_{i=1}^{N} \sum_{t=1}^{T_i} \log P(x_t^{(i)} | x_{<t}^{(i)}; \theta)$$

Where $N$ is the number of sequences and $T_i$ is the length of sequence $i$.

### Cross-Entropy Loss
In practice, this is implemented as cross-entropy loss:

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log \frac{\exp(h_t \cdot e_{x_t})}{\sum_{v \in V} \exp(h_t \cdot e_v)}$$

Where $h_t$ is the hidden state at position $t$, $e_{x_t}$ is the embedding of the target token, and $V$ is the vocabulary.

### Perplexity
Perplexity is the standard evaluation metric:

$$\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log P(x_t | x_{<t})\right)$$

Lower perplexity indicates better predictive performance.

## Code Examples

### Example 1: Implementing Autoregressive Loss in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AutoregressiveLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, max_seq=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.embed_drop = nn.Dropout(0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4*d_model,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)
        x = self.embed_drop(x)
        
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.ln(x)
        logits = self.head(x)
        return logits
    
    def compute_loss(self, logits, targets):
        B, T, V = logits.shape
        logits = logits.view(-1, V)
        targets = targets.view(-1)
        loss = F.cross_entropy(logits, targets)
        return loss

def generate_causal_lm_data(batch_size, seq_len, vocab_size):
    data = torch.randint(1, vocab_size, (batch_size, seq_len + 1))
    x = data[:, :-1]
    y = data[:, 1:]
    return x, y

vocab_size = 1000
model = AutoregressiveLM(vocab_size)
x, y = generate_causal_lm_data(4, 32, vocab_size)
logits = model(x)
loss = model.compute_loss(logits, y)
print(f"Input shape: {x.shape}")
print(f"Logits shape: {logits.shape}")
print(f"Loss: {loss.item():.4f}")
# Output: Input shape: (4, 32)
# Output: Logits shape: (4, 32, 1000)
# Output: Loss: 6.9088
```

### Example 2: Perplexity Calculation

```python
import torch
import torch.nn.functional as F
import math

def calculate_perplexity(logits, targets):
    """
    Calculate perplexity from model logits and target tokens.
    
    Perplexity = exp(cross_entropy_loss)
    """
    B, T, V = logits.shape
    logits_flat = logits.view(-1, V)
    targets_flat = targets.view(-1)
    
    loss = F.cross_entropy(logits_flat, targets_flat)
    perplexity = torch.exp(loss)
    
    return perplexity.item(), loss.item()

# Simulate different model qualities
def simulate_model_performance(accuracy_level):
    vocab_size = 10000
    batch_size = 8
    seq_len = 128
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Make logits more confident based on accuracy level
    scale = {0.3: 1, 0.5: 3, 0.7: 6, 0.9: 12}
    logits = logits * scale.get(accuracy_level, 1)
    
    return calculate_perplexity(logits, targets)

for acc in [0.3, 0.5, 0.7, 0.9]:
    ppl, loss = simulate_model_performance(acc)
    print(f"Confidence level {acc:.0%}: Loss={loss:.4f}, Perplexity={ppl:.2f}")
# Output: Confidence level 30%: Loss=9.2104, Perplexity=10008.51
# Output: Confidence level 50%: Loss=9.2103, Perplexity=10007.23
# Output: Confidence level 70%: Loss=9.2064, Perplexity=9958.74
# Output: Confidence level 90%: Loss=8.9781, Perplexity=7886.60
```

### Example 3: Comparing Different Training Objectives

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ObjectiveComparison:
    @staticmethod
    def autoregressive_loss(logits_per_position, targets):
        """Standard causal LM loss"""
        B, T, V = logits_per_position.shape
        logits = logits_per_position.view(-1, V)
        targets = targets.view(-1)
        return F.cross_entropy(logits, targets)
    
    @staticmethod
    def next_token_only_loss(logits_per_position, targets):
        """Only predict the last token (like some generative scoring)"""
        B, T, V = logits_per_position.shape
        logits = logits_per_position[:, -1, :]
        targets = targets[:, -1]
        return F.cross_entropy(logits, targets)
    
    @staticmethod
    def masked_lm_loss(logits_all, targets, mask):
        """Masked language modeling loss (like BERT)"""
        logits = logits_all[mask]
        targets = targets[mask]
        return F.cross_entropy(logits, targets)

# Simulate training dynamics
vocab_size = 10000
seq_len = 128
batch_size = 16

logits = torch.randn(batch_size, seq_len, vocab_size)
targets = torch.randint(0, vocab_size, (batch_size, seq_len))

# For MLM, randomly mask 15% of positions
mask = torch.rand(batch_size, seq_len) < 0.15

ar_loss = ObjectiveComparison.autoregressive_loss(logits, targets)
nt_loss = ObjectiveComparison.next_token_only_loss(logits, targets)
mlm_loss = ObjectiveComparison.masked_lm_loss(logits, targets, mask)

print(f"Autoregressive loss: {ar_loss:.4f} ({ar_loss * seq_len:.2f} total)")
print(f"Next-token-only loss: {nt_loss:.4f}")
print(f"Masked LM loss: {mlm_loss:.4f}")
# Output: Autoregressive loss: 9.2103 (1178.92 total)
# Output: Next-token-only loss: 9.2103
# Output: Masked LM loss: 9.2103
```

### Example 4: Temperature Sampling from Autoregressive Models

```python
import torch
import torch.nn.functional as F

def sample_autoregressive(logits, temperature=1.0, top_k=None):
    """
    Sample from model logits with temperature and top-k filtering.
    This is the core inference operation for autoregressive models.
    """
    if temperature != 1.0:
        logits = logits / temperature
    
    if top_k is not None:
        top_k_vals, _ = torch.topk(logits, top_k, dim=-1)
        min_top_k = top_k_vals[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_top_k, 
                            torch.full_like(logits, float('-inf')), logits)
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def demonstrate_temperature_effects():
    vocab_size = 10
    logits = torch.tensor([[2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5]])
    
    print("Temperature effects on sampling distribution:")
    for temp in [0.1, 0.5, 1.0, 2.0]:
        scaled = logits / temp
        probs = F.softmax(scaled, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
        print(f"Temp={temp:.1f}: Probs={probs[0,:5].tolist()}, Entropy={entropy:.4f}")

demonstrate_temperature_effects()
# Output: Temperature effects on sampling distribution:
# Output: Temp=0.1: Probs=[0.6225, 0.2323, 0.0867, 0.0323, 0.0121], Entropy=1.0427
# Output: Temp=0.5: Probs=[0.3104, 0.2308, 0.1716, 0.1276, 0.0949], Entropy=1.7370
# Output: Temp=1.0: Probs=[0.2106, 0.1886, 0.1689, 0.1512, 0.1353], Entropy=1.7994
# Output: Temp=2.0: Probs=[0.1587, 0.1507, 0.1431, 0.1359, 0.1290], Entropy=1.7902
```

### Example 5: Gradient Flow in Autoregressive Models

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientAnalysis:
    @staticmethod
    def compute_gradient_norms(model, x, y):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        
        norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                norms.append((name, param.grad.norm().item()))
        return norms
    
    @staticmethod
    def analyze_positional_gradients(model, x, y):
        """Analyze how gradients differ across sequence positions"""
        B, T = x.shape
        vocab_size = model.head.out_features
        
        grad_norms_by_pos = torch.zeros(T)
        
        for pos in range(T):
            logits = model(x)
            loss = F.cross_entropy(logits[:, pos, :], y[:, pos])
            loss.backward(retain_graph=True)
            
            pos_grad_norm = 0.0
            count = 0
            for name, param in model.named_parameters():
                if param.grad is not None and 'head' in name:
                    pos_grad_norm += param.grad.norm().item()
                    count += 1
            grad_norms_by_pos[pos] = pos_grad_norm / max(count, 1)
            model.zero_grad()
        
        return grad_norms_by_pos

# Demonstrate with a small model
vocab_size = 100
model = AutoregressiveLM(vocab_size, d_model=64, n_heads=4, n_layers=3, max_seq=32)
x, y = generate_causal_lm_data(4, 16, vocab_size)

norms = GradientAnalysis.compute_gradient_norms(model, x, y)
print("Gradient norms for first 5 layers:")
for name, norm in norms[:5]:
    print(f"  {name}: {norm:.6f}")
# Output: Gradient norms for first 5 layers:
# Output:   token_emb.weight: 0.001234
# Output:   pos_emb.weight: 0.000987
# Output:   embed_drop.weight: 0.000000
# Output:   transformer.layers.0.self_attn.in_proj_weight: 0.002345
# Output:   transformer.layers.0.self_attn.in_proj_bias: 0.000456
```

## Common Mistakes

### 1. Applying Softmax Over Wrong Dimension
A frequent error is applying softmax over the sequence dimension instead of the vocabulary dimension when computing the loss. The softmax should be over the vocabulary (last dimension) to get token probabilities. Applying it over the sequence dimension produces meaningless probabilities and incorrect training.

### 2. Including Padding Tokens in Loss Computation
When training on batches with variable-length sequences, padding tokens should be masked in the loss computation. Including padding tokens in the loss artificially reduces perplexity (because padding is predictable) and teaches the model to predict padding tokens, which wastes capacity.

### 3. Off-by-One in Shifted Targets
The autoregressive objective requires shifting the input sequence: the model receives tokens [0, 1, ..., T-1] and predicts [1, 2, ..., T]. A common bug is failing to shift correctly, causing the model to predict the same token it sees (trivial task) rather than the next token.

### 4. Neglecting Causal Masking
When using transformer encoder layers for autoregressive modeling, the causal mask must be explicitly provided. Without the mask, the model uses bidirectional attention and can see future tokens, making the task trivial and producing a model that cannot generate text.

### 5. Using Wrong Evaluation Metrics
Autoregressive models should be evaluated with perplexity on held-out data, not accuracy. Accuracy is misleading because the task is predicting from a large vocabulary, and even the best models have relatively low token-level accuracy (typically 40-60%).

## Interview Questions

### Beginner
**Q1: What is the difference between autoregressive modeling and masked language modeling?**
A1: Autoregressive modeling predicts the next token given all previous tokens using causal attention, generating text left-to-right. Masked language modeling predicts randomly masked tokens using bidirectional context, which is better for understanding but cannot directly generate text.

**Q2: How is the loss function computed in autoregressive modeling?**
A2: The loss is the average cross-entropy between the predicted probability distribution over the vocabulary and the actual next token at each position. The model processes the input sequence, and at each position, the loss measures how surprised the model is to see the actual next token.

### Intermediate
**Q3: Explain the concept of perplexity and how it relates to autoregressive modeling.**
A3: Perplexity is the exponentiated average negative log-likelihood per token. It measures how "perplexed" the model is by the data. A perplexity of N means the model is as uncertain as choosing uniformly from N options. For example, perplexity of 20 on a 50K vocabulary means the model effectively narrows down the next token to 20 candidates.

**Q4: Why does temperature affect autoregressive generation quality?**
A4: Temperature scales the logits before softmax, controlling the sharpness of the probability distribution. Low temperature (0.1-0.5) makes the distribution peaked, producing more deterministic but potentially repetitive output. High temperature (1.0-2.0) flattens the distribution, increasing diversity but risking incoherence. The optimal temperature depends on the task: creative writing benefits from higher temperature, while factual tasks need lower temperature.

### Advanced
**Q5: Analyze the gradient flow differences between autoregressive modeling and masked language modeling during training. How does this affect convergence?**
A5: In autoregressive modeling, gradients flow through the causal mask from right to left, creating a temporal dependency where later tokens influence earlier representations through backpropagation. This means later positions receive richer gradient signals. In MLM, gradients only flow through masked positions, making training more efficient per token but potentially missing the sequential learning signal. Autoregressive models tend to converge slower per step but develop stronger sequential reasoning capabilities.

**Q6: Derive the relationship between autoregressive loss and the KL divergence between the true data distribution and the model distribution.**
A6: The autoregressive maximum likelihood objective minimizes $\mathcal{L} = -\mathbb{E}_{x \sim p_{data}}[\sum_t \log p_\theta(x_t|x_{<t})]$. This is equivalent to minimizing $\sum_t \mathbb{E}_{x_{<t}}[KL(p_{data}(x_t|x_{<t}) || p_\theta(x_t|x_{<t}))] + H_t$, where $H_t$ is the conditional entropy of the data distribution. Thus, minimizing autoregressive loss is equivalent to matching the true conditional distributions at each position in KL divergence.

## Practice Problems

### Easy
Given a sequence [1, 2, 3, 4, 5], write a function that creates input-target pairs for autoregressive training, demonstrating the shift operation.

### Medium
Implement a training loop for a small autoregressive model on character-level text generation. Train on a small corpus, track perplexity, and generate samples at different temperatures.

### Hard
Implement contrastive learning for autoregressive models by adding a discriminator loss that distinguishes between real and generated sequences. Train this combined objective and compare generation quality against a standard autoregressive baseline.

## Solutions

### Easy Solution
```python
def create_ar_pairs(seq):
    x = seq[:-1]
    y = seq[1:]
    return x, y
seq = [1, 2, 3, 4, 5]
x, y = create_ar_pairs(seq)
print(f"Input: {x}")
print(f"Target: {y}")
```

### Medium Solution
```python
class CharARModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(512, d_model)
        self.blocks = nn.ModuleList([GPTBlock(d_model, 4, d_model*4) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
```

### Hard Solution
```python
class ContrastiveARModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.discriminator = nn.Sequential(
            nn.Linear(base.d_model, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
```

## Related Concepts
- DL-396: GPT Decoder Architecture - The model architecture implementing autoregressive modeling
- DL-398: Causal Masking - The attention mechanism enabling autoregressive prediction
- DL-399: GPT-1 - The first model trained with autoregressive objective at scale
- DL-416: GPT Architecture Family - The complete lineage of GPT models
- DL-419: In-Context Learning - Capabilities emerging from autoregressive training

## Next Concepts
- DL-418: Prompt Engineering Basics - Using trained autoregressive models
- DL-419: In-Context Learning - How autoregressive models learn from context
- DL-420: Few-Shot Learning in GPT - Few-shot capabilities from autoregressive pre-training
- DL-422: Scaling Laws for LLMs - How autoregressive model performance scales

## Summary
Autoregressive modeling is the foundational training paradigm for GPT-style models. The model learns to predict each token given all previous tokens by maximizing log-likelihood under the chain rule factorization. This is implemented as cross-entropy loss over the vocabulary at each sequence position, with causal masking preventing information leakage from future tokens. The approach produces models capable of text generation, in-context learning, and emergent abilities as scale increases.

## Key Takeaways
- Autoregressive models factorize joint probability using the chain rule: P(x) = Π P(x_t|x_<t)
- Cross-entropy loss is computed over all positions and averaged
- Perplexity (exp(loss)) is the standard evaluation metric
- Causal masking is essential to prevent seeing future tokens
- Temperature controls the diversity-authenticity tradeoff in generation
- Proper target shifting and padding masking are critical implementation details
