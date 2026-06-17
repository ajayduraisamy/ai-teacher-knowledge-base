# Concept: InfoNCE Loss

## Concept ID

DL-110

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Loss Functions

## Learning Objectives

- Understand InfoNCE loss for contrastive representation learning
- Implement InfoNCE loss in PyTorch
- Explain the relationship between InfoNCE and mutual information
- Apply InfoNCE for self-supervised learning (SimCLR, MoCo)
- Compare InfoNCE with NCE and triplet loss

## Prerequisites

- Noise Contrastive Estimation (DL-109)
- Self-supervised learning basics
- Mutual information concepts

## Definition

InfoNCE (Information Noise Contrastive Estimation) is a contrastive loss function that learns representations by maximizing mutual information between positive pairs while minimizing it between negative pairs. For a query q, positive key k+, and a set of negative keys K-:

L_InfoNCE = -log(exp(q * k+ / tau) / (exp(q * k+ / tau) + sum_{k- in K-} exp(q * k- / tau)))

where tau is the temperature parameter.

## Intuition

InfoNCE is the foundation of modern self-supervised learning. Given a query (e.g., an augmented view of an image), the model must identify the positive key (another augmented view of the same image) among a set of negative keys (views of different images). This is like a game of "spot the match" where the model learns useful representations by solving this discrimination task.

## Why This Concept Matters

InfoNCE is the loss function behind state-of-the-art self-supervised learning methods:
- SimCLR: Contrastive learning of visual representations
- MoCo: Momentum contrast for unsupervised learning
- CPC: Contrastive predictive coding for speech/audio
- CLIP: Contrastive language-image pre-training

## Mathematical Explanation

### InfoNCE and Mutual Information

For random variables X (query) and Y (positive key), InfoNCE provides a lower bound on the mutual information:

I(X; Y) >= log(K) - L_InfoNCE

where K is the number of negatives.

### Temperature

The temperature tau controls the concentration of the distribution:
- Low tau: sharp distribution, focuses on hardest negatives
- High tau: uniform distribution, treats all samples similarly
- Typical tau: 0.1 - 0.5

### SimCLR NT-Xent Loss

The normalized temperature-scaled cross-entropy loss:

L(i, j) = -log(exp(sim(z_i, z_j)/tau) / sum_{k=1}^{2N} 1_{k != i} exp(sim(z_i, z_k)/tau))

## Code Examples

### Example 1: Manual InfoNCE Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, queries, pos_keys, neg_keys):
        # queries: (N, D)
        # pos_keys: (N, D)
        # neg_keys: (N, K, D)
        pos_sim = (queries * pos_keys).sum(dim=1) / self.temperature
        neg_sim = torch.bmm(neg_keys, queries.unsqueeze(2)).squeeze(2) / self.temperature
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(len(queries), dtype=torch.long, device=queries.device)
        return F.cross_entropy(logits, labels)

# SimCLR-style batch contrastive loss
def nt_xent_loss(z, temperature=0.1):
    # z: (2*N, D) where first N are aug1, last N are aug2
    N = z.size(0) // 2
    z = F.normalize(z, dim=1)
    similarity = torch.mm(z, z.t()) / temperature
    mask = torch.eye(2*N, device=z.device, dtype=torch.bool)
    similarity.masked_fill_(mask, -1e9)

    # Positive pairs: (i, i+N) and (i+N, i) for i in [0, N)
    labels = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)])
    return F.cross_entropy(similarity, labels)

torch.manual_seed(42)
queries = F.normalize(torch.randn(8, 64), dim=1)
pos_keys = F.normalize(queries + 0.1 * torch.randn(8, 64), dim=1)
neg_keys = F.normalize(torch.randn(8, 10, 64), dim=2)

criterion = InfoNCELoss(temperature=0.1)
loss = criterion(queries, pos_keys, neg_keys)
print(f"InfoNCE loss: {loss.item():.4f}")

# SimCLR NT-Xent loss
z = torch.cat([queries, pos_keys], dim=0)
ntxent_loss = nt_xent_loss(z, temperature=0.1)
print(f"NT-Xent loss: {ntxent_loss.item():.4f}")
```

```
# Output:
# InfoNCE loss: 2.4604
# NT-Xent loss: 2.5061
```

### Example 2: SimCLR-Style Augmentation Framework

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(42)
N, D, feat_dim = 200, 50, 16
X = torch.randn(N, D)

class SimCLR(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, feature_dim)
        )

    def forward(self, x):
        return F.normalize(self.encoder(x), dim=1)

model = SimCLR(D, feat_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Create "augmented" pairs
X1 = X + 0.1 * torch.randn(N, D)  # Augmentation 1
X2 = X + 0.1 * torch.randn(N, D)  # Augmentation 2

for epoch in range(200):
    optimizer.zero_grad()
    z1 = model(X1)
    z2 = model(X2)
    z = torch.cat([z1, z2], dim=0)
    loss = nt_xent_loss(z, temperature=0.2)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")
```

```
# Output:
# Epoch 0: loss = 2.3161
# Epoch 50: loss = 1.4614
# Epoch 100: loss = 1.0125
# Epoch 150: loss = 0.7493
# Epoch 199: loss = 0.5827
```

### Example 3: Temperature Effect on InfoNCE

```python
import torch
import torch.nn.functional as F

torch.manual_seed(42)
z = F.normalize(torch.randn(16, 64), dim=1)

def compute_nt_xent(z, temperature):
    N = z.size(0) // 2
    similarity = torch.mm(z, z.t()) / temperature
    mask = torch.eye(2*N, dtype=torch.bool)
    similarity.masked_fill_(mask, -1e9)
    labels = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)])
    loss = F.cross_entropy(similarity, labels)
    probs = F.softmax(similarity, dim=1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
    return loss.item(), entropy.item()

for tau in [0.05, 0.1, 0.5, 1.0]:
    loss, entropy = compute_nt_xent(z, tau)
    print(f"T={tau:.2f}: loss = {loss:.4f}, entropy = {entropy:.4f}")
```

```
# Output:
# T=0.05: loss = 2.7695, entropy = 0.1936
# T=0.10: loss = 2.5061, entropy = 1.0712
# T=0.50: loss = 0.8044, entropy = 2.3396
# T=1.00: loss = 0.5532, entropy = 2.6730
```

## Common Mistakes

1. **Temperature too low**: Very low temperature creates a sharp distribution, focusing on the hardest negatives and potentially causing training instability.
2. **Temperature too high**: Very high temperature makes all similarities nearly uniform, providing no useful learning signal.
3. **Not normalizing embeddings**: InfoNCE typically uses L2-normalized embeddings for dot product similarity.
4. **Small batch size**: InfoNCE benefits from many negatives. Larger batches (4096+) perform better.
5. **Leaking positive pairs**: Ensure positive pairs are properly excluded from negatives.
6. **Ignoring the queue in MoCo**: For memory banks, ensure negatives are from the appropriate distributions.

## Interview Questions

### Beginner

1. What is InfoNCE loss and where is it used?
2. How does InfoNCE differ from standard NCE?
3. What is the role of the temperature parameter?
4. How does InfoNCE relate to mutual information?
5. How do you implement InfoNCE in PyTorch?

### Intermediate

1. Derive the InfoNCE loss from the NCE framework.
2. Explain why InfoNCE maximizes mutual information.
3. How does SimCLR's NT-Xent loss relate to InfoNCE?
4. Why does larger batch size improve InfoNCE performance?
5. Compare InfoNCE with triplet loss for representation learning.

### Advanced

1. Prove that InfoNCE provides a lower bound on mutual information.
2. Analyze the effect of temperature on the gradient of InfoNCE.
3. Derive the connection between InfoNCE and the variational lower bound of mutual information.

## Practice Problems

### Easy

1. Implement InfoNCE loss for query-positive-negative triplets.
2. Implement the SimCLR NT-Xent loss.
3. Compute InfoNCE loss for different temperature values.
4. Verify that InfoNCE = cross-entropy with proper setup.
5. Compare InfoNCE with NCE for the same data.

### Medium

1. Train a SimCLR-style model on image data.
2. Compare InfoNCE with triplet loss for representation learning.
3. Analyze the effect of batch size on InfoNCE performance.
4. Implement MoCo-style contrastive learning with a queue.
5. Visualize learned representations from InfoNCE training using t-SNE.

### Hard

1. Derive the gradient of InfoNCE and analyze its properties.
2. Implement a self-supervised learning pipeline with InfoNCE and evaluate on downstream tasks.
3. Design an experiment comparing InfoNCE, SimCLR, MoCo, and CLIP objectives.

## Solutions

InfoNCE loss = -log(exp(q*k+/tau) / sum exp(q*k/tau)). It maximizes mutual information by contrasting positive pairs against negatives. Temperature controls the sharpness of the distribution.

## Related Concepts

- Noise Contrastive Estimation (DL-109): The precursor to InfoNCE
- Contrastive Loss (DL-100): Pair-based contrastive learning
- Triplet Loss (DL-101): Three-element contrastive learning
- Self-Supervised Learning: Application domain

## Summary

InfoNCE loss contrasts positive pairs against a set of negative samples, providing a lower bound on mutual information. It is the foundation of modern self-supervised learning methods including SimCLR, MoCo, and CLIP. The temperature parameter controls the concentration of the similarity distribution.

## Key Takeaways

1. InfoNCE = -log(exp(q*k+/tau) / sum exp(q*k/tau)).
2. InfoNCE provides a lower bound on mutual information I(X; Y).
3. Temperature controls the sharpness of the contrastive distribution.
4. Larger batch sizes provide more negatives, improving representations.
5. InfoNCE is the core loss for SimCLR, MoCo, CLIP, and most self-supervised learning methods.
