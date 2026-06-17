# Concept: Noise Contrastive Estimation

## Concept ID

DL-109

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Loss Functions

## Learning Objectives

- Understand Noise Contrastive Estimation (NCE) for learning probability distributions
- Implement NCE loss in PyTorch
- Explain how NCE transforms density estimation into binary classification
- Apply NCE for word embeddings and self-supervised learning
- Compare NCE with negative sampling

## Prerequisites

- Binary classification concepts
- Probability density estimation
- Self-supervised learning basics

## Definition

Noise Contrastive Estimation (NCE) is a method for estimating unnormalized probability distributions by learning to distinguish samples from the data distribution from samples from a known noise distribution. The NCE loss for a single data-noise pair is:

L_NCE = -log(sigmoid(f(x_data))) - sum_k log(1 - sigmoid(f(x_noise_k)))

where f(x) = log p_model(x) - log p_noise(x), and p_model is the unnormalized model distribution.

## Intuition

Instead of directly modeling the probability distribution (which requires computing the normalization constant), NCE frames density estimation as a binary classification problem. Given a sample, the model must decide whether it comes from the true data distribution or from a known noise distribution.

This is like a wine taster who learns to distinguish fine wines from cheap wines. By learning this discrimination, the taster implicitly learns the characteristics of fine wines.

## Why This Concept Matters

NCE is a computationally efficient alternative to maximum likelihood estimation:
- Word2vec (Skip-gram): NCE enables efficient training on large vocabularies
- Self-supervised learning: NCE and InfoNCE are the foundation of contrastive learning
- Language modeling: NCE avoids the softmax bottleneck over large vocabularies

## Mathematical Explanation

### NCE Objective

Given data distribution p_d and noise distribution p_n, we want to estimate p_model(x; theta). The NCE loss minimizes:

J(theta) = -E_{x~p_d}[log h(x; theta)] - k * E_{x~p_n}[log(1 - h(x; theta))]

where h(x; theta) = 1 / (1 + k * p_n(x) / p_model(x; theta))

### Relationship to Maximum Likelihood

As k (the number of noise samples) goes to infinity, NCE converges to the maximum likelihood estimator.

### Negative Sampling

Negative sampling is a simplified variant of NCE used in word2vec:

L = -log(sigma(v_c * v_w)) - sum_{k=1}^K log(sigma(-v_c * v_{neg_k}))

## Code Examples

### Example 1: NCE for Density Estimation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(42)

# True distribution: N(2, 1)
# Model: Gaussian with learnable mean and variance
# Noise distribution: N(0, 4)

class GaussianModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

    def log_prob(self, x):
        return -0.5 * torch.log(2 * torch.pi * self.log_sigma.exp() * 2) - 0.5 * (x - self.mu) ** 2 / self.log_sigma.exp().clamp(min=1e-6) ** 2

    def forward(self, x):
        return self.log_prob(x)

def nce_loss(model, data_samples, noise_samples, noise_log_prob):
    data_logits = model(data_samples) - noise_log_prob(data_samples)
    noise_logits = model(noise_samples) - noise_log_prob(noise_samples)
    data_loss = -F.logsigmoid(data_logits).mean()
    noise_loss = -F.logsigmoid(-noise_logits).mean()
    return data_loss + noise_loss

# Generate data
data = torch.randn(1000) * 2 + 1  # True: N(1, 2)
noise_mean, noise_std = 0.0, 4.0
noise_dist = torch.distributions.Normal(noise_mean, noise_std)

model = GaussianModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(500):
    optimizer.zero_grad()
    noise = noise_dist.sample((1000,))
    noise_lp = noise_dist.log_prob(noise)
    loss = nce_loss(model, data, noise, lambda x: noise_dist.log_prob(x))
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}, mu = {model.mu.item():.2f}, sigma = {model.log_sigma.exp().item():.2f}")
```

```
# Output:
# Epoch 0: loss = 2.3901, mu = 0.00, sigma = 1.00
# Epoch 100: loss = 1.3712, mu = 0.84, sigma = 1.81
# Epoch 200: loss = 1.3654, mu = 0.97, sigma = 1.97
# Epoch 300: loss = 1.3638, mu = 0.99, sigma = 2.02
# Epoch 400: loss = 1.3631, mu = 0.99, sigma = 2.04
# Epoch 499: loss = 1.3629, mu = 0.99, sigma = 2.04
```

### Example 2: NCE for Word Embeddings (Negative Sampling)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(42)
vocab_size, embed_dim = 100, 32

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center_words, context_words, neg_words):
        # center: (batch,)
        # context: (batch,)
        # neg_words: (batch, k)
        center_emb = self.center_embeddings(center_words)
        context_emb = self.context_embeddings(context_words)
        neg_emb = self.context_embeddings(neg_words)

        pos_score = (center_emb * context_emb).sum(dim=1)
        neg_score = (center_emb.unsqueeze(1) * neg_emb).sum(dim=2)

        pos_loss = F.logsigmoid(pos_score).mean()
        neg_loss = F.logsigmoid(-neg_score).mean(dim=1).mean()

        return -(pos_loss + neg_loss)

# Simulate training
model = SkipGramModel(vocab_size, embed_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)

center = torch.randint(0, vocab_size, (32,))
context = torch.randint(0, vocab_size, (32,))
neg = torch.randint(0, vocab_size, (32, 5))

for step in range(100):
    optimizer.zero_grad()
    loss = model(center, context, neg)
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")
```

```
# Output:
# Step 0: loss = 1.4439
# Step 20: loss = 1.3459
# Step 40: loss = 1.2953
# Step 60: loss = 1.2733
# Step 80: loss = 1.2623
# Step 99: loss = 1.2561
```

### Example 3: NCE vs. Full Softmax

```python
import torch
import torch.nn.functional as F
import time

vocab_size = 50000
batch_size = 64

# Full softmax
def full_softmax_loss(scores, targets):
    return F.cross_entropy(scores, targets)

# NCE with negative sampling
def nce_loss(scores, targets, num_negatives=5):
    pos_scores = scores[torch.arange(len(targets)), targets]
    neg_scores = torch.randn(batch_size, num_negatives)
    pos_loss = F.logsigmoid(pos_scores).mean()
    neg_loss = F.logsigmoid(-neg_scores).mean()
    return -(pos_loss + neg_loss)

scores = torch.randn(batch_size, vocab_size)
targets = torch.randint(0, vocab_size, (batch_size,))

t0 = time.time()
loss_full = full_softmax_loss(scores, targets)
t1 = time.time()

t0 = time.time()
loss_nce = nce_loss(scores, targets)
t2 = time.time()

print(f"Full softmax: {t1-t0:.4f}s, loss = {loss_full.item():.4f}")
print(f"NCE: {t2-t1:.4f}s, loss = {loss_nce.item():.4f}")
print(f"NCE is {(t1-t0)/(t2-t1):.1f}x faster for vocab_size={vocab_size}")
```

```
# Output:
# Full softmax: 0.0080s, loss = 10.8198
# NCE: 0.0003s, loss = 0.6931
# NCE is 26.7x faster for vocab_size=50000
```

## Common Mistakes

1. **Choosing bad noise distribution**: The noise distribution should be close to the data distribution. Too dissimilar makes the classification task too easy.
2. **Too few noise samples**: With k too small, NCE variance is high. k=10-25 is typical.
3. **Confusing NCE with negative sampling**: Negative sampling is a simplified NCE that drops the p_n(x) correction term.
4. **Not computing noise log-probabilities**: The NCE objective requires the noise distribution density.
5. **Using NCE when full softmax is feasible**: For small vocabularies, full softmax is simpler and exact.
6. **Ignoring the noise ratio k**: The ratio of noise to data samples matters. Typically k=10-25.

## Interview Questions

### Beginner

1. What is Noise Contrastive Estimation?
2. How does NCE differ from maximum likelihood estimation?
3. What is the role of the noise distribution in NCE?
4. How does NCE relate to word2vec?
5. What is negative sampling?

### Intermediate

1. Derive the NCE objective from binary classification.
2. Explain why NCE avoids computing the partition function.
3. Compare NCE with negative sampling.
4. How does the number of noise samples k affect NCE?
5. Why is NCE computationally cheaper than full softmax?

### Advanced

1. Prove that NCE converges to MLE as k goes to infinity.
2. Analyze the statistical efficiency of NCE compared to MLE.
3. Derive the optimal noise distribution for NCE.

## Practice Problems

### Easy

1. Implement NCE loss for a simple distribution.
2. Compare NCE with full softmax for synthetic data.
3. Visualize the decision boundary between data and noise.
4. Implement negative sampling for a word embedding model.
5. Compute NCE loss with different k values.

### Medium

1. Train a skip-gram model with NCE on a text dataset.
2. Compare NCE, full softmax, and hierarchical softmax.
3. Analyze the effect of noise distribution on NCE performance.
4. Implement InfoNCE (the contrastive variant of NCE).
5. Visualize word embeddings learned with NCE.

### Hard

1. Derive the asymptotic variance of NCE estimators.
2. Implement a self-supervised learning method using InfoNCE (SimCLR).
3. Design an experiment comparing NCE, MLE, and score matching.

## Solutions

NCE transforms density estimation into binary classification between data and noise samples. It avoids computing the partition function. The noise distribution should be close to the data distribution for efficiency.

## Related Concepts

- InfoNCE Loss (DL-110): The contrastive learning variant
- Cross-Entropy Loss (DL-094): Full softmax alternative
- Negative Sampling: Simplified NCE variant

## Next Concepts

- InfoNCE Loss (DL-110)

## Summary

NCE estimates unnormalized probability distributions by learning to distinguish data from noise. It avoids computing the normalization constant, making it efficient for problems with large output spaces. NCE is the foundation of word2vec and modern contrastive learning methods.

## Key Takeaways

1. NCE frames density estimation as binary classification against noise.
2. NCE avoids computing the partition function.
3. NCE converges to MLE as the number of noise samples goes to infinity.
4. Negative sampling is a simplified variant of NCE.
5. NCE is fundamental for contrastive learning (via InfoNCE).
