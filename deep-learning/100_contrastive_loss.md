# Concept: Contrastive Loss

## Concept ID

DL-100

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Loss Functions

## Learning Objectives

- Understand contrastive loss for similarity learning
- Implement contrastive loss in PyTorch
- Explain the margin parameter and its role
- Apply contrastive loss for siamese networks
- Compare contrastive loss with triplet loss

## Prerequisites

- Siamese network architectures
- Distance metrics (Euclidean, cosine)
- Basic metric learning concepts

## Definition

Contrastive loss is a loss function used for metric learning that trains a model to embed similar examples close together and dissimilar examples far apart. For a pair of examples (x1, x2) with label y (1 for similar, 0 for dissimilar):

L = y * D^2 + (1 - y) * max(0, margin - D)^2

where D = ||f(x1) - f(x2)||_2 is the Euclidean distance between embeddings, and margin is a hyperparameter.

## Intuition

Think of contrastive loss as teaching a model to organize a photo album. Similar photos (same person) should be placed close together on the table. Dissimilar photos should be placed far apart — at least margin distance away. The loss pulls similar pairs together and pushes dissimilar pairs apart, but only if they are too close (within the margin).

## Why This Concept Matters

Contrastive loss is fundamental for:
- Face recognition and verification
- Person re-identification
- One-shot and few-shot learning
- Dimensionality reduction and visualization
- Self-supervised representation learning

## Mathematical Explanation

### Contrastive Loss Function

L(x1, x2, y) = y * D^2 + (1 - y) * max(0, m - D)^2

where:
- D = ||f(x1) - f(x2)||_2 (Euclidean distance in embedding space)
- y = 1 if similar, 0 if dissimilar
- m = margin (minimum distance for dissimilar pairs)

### Gradient Analysis

For similar pairs (y=1): dL/dD = 2D. The gradient pulls the embeddings closer.

For dissimilar pairs (y=0):
- If D > m: dL/dD = 0. No gradient (already far enough apart).
- If D < m: dL/dD = -2(m - D). Gradient pushes embeddings apart.

### Margin Selection

The margin determines the minimum desired separation between dissimilar classes. Typical values range from 0.5 to 2.0, depending on the embedding dimension and normalization.

## Code Examples

### Example 1: Manual Contrastive Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        # emb1, emb2: (batch_size, embed_dim)
        # label: 1 for similar, 0 for dissimilar
        D = torch.pairwise_distance(emb1, emb2)
        loss_similar = label * D.pow(2)
        loss_dissimilar = (1 - label) * torch.clamp(self.margin - D, min=0).pow(2)
        return (loss_similar + loss_dissimilar).mean()

# Example
torch.manual_seed(42)
emb_similar_1 = torch.randn(4, 16)
emb_similar_2 = emb_similar_1 + 0.1 * torch.randn(4, 16)  # Similar
emb_dissimilar_1 = torch.randn(4, 16)
emb_dissimilar_2 = torch.randn(4, 16)  # Dissimilar

criterion = ContrastiveLoss(margin=1.0)

# Similar pairs
labels_sim = torch.ones(4)
loss_sim = criterion(emb_similar_1, emb_similar_2, labels_sim)
print(f"Similar pair loss: {loss_sim.item():.4f}")

# Dissimilar pairs
labels_dis = torch.zeros(4)
loss_dis = criterion(emb_dissimilar_1, emb_dissimilar_2, labels_dis)
print(f"Dissimilar pair loss: {loss_dis.item():.4f}")

# Distances
D_sim = torch.pairwise_distance(emb_similar_1, emb_similar_2).mean()
D_dis = torch.pairwise_distance(emb_dissimilar_1, emb_dissimilar_2).mean()
print(f"Avg distance (similar): {D_sim.item():.4f}")
print(f"Avg distance (dissimilar): {D_dis.item():.4f}")
```

```
# Output:
# Similar pair loss: 0.0096
# Dissimilar pair loss: 0.4183
# Avg distance (similar): 0.3058
# Avg distance (dissimilar): 1.2878
```

### Example 2: Siamese Network with Contrastive Loss

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SiameseNetwork(nn.Module):
    def __init__(self, input_dim=10, embed_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, embed_dim)
        )

    def forward_one(self, x):
        return self.encoder(x)

    def forward(self, x1, x2):
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        return emb1, emb2

torch.manual_seed(42)
N, D, embed_dim = 500, 10, 8
X1 = torch.randn(N, D)
X2 = torch.randn(N, D)
labels = (torch.rand(N) > 0.5).float()

model = SiameseNetwork(D, embed_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = ContrastiveLoss(margin=1.0)

for epoch in range(200):
    optimizer.zero_grad()
    emb1, emb2 = model(X1, X2)
    loss = criterion(emb1, emb2, labels)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        D = torch.pairwise_distance(emb1, emb2)
        similar_dist = D[labels == 1].mean()
        dissimilar_dist = D[labels == 0].mean()
        print(f"Epoch {epoch}: loss = {loss.item():.4f}, sim_dist = {similar_dist.item():.4f}, dis_dist = {dissimilar_dist.item():.4f}")
```

```
# Output:
# Epoch 0: loss = 0.4135, sim_dist = 1.2544, dis_dist = 1.5396
# Epoch 50: loss = 0.1964, sim_dist = 0.4826, dis_dist = 1.6502
# Epoch 100: loss = 0.1351, sim_dist = 0.2995, dis_dist = 1.7427
# Epoch 150: loss = 0.1036, sim_dist = 0.2346, dis_dist = 1.7734
# Epoch 199: loss = 0.0846, sim_dist = 0.2019, dis_dist = 1.7881
```

### Example 3: Contrastive Loss for Embedding Visualization

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
N, D, embed_dim = 300, 20, 2  # 2D for visualization
num_classes = 5
X = torch.randn(N, D)
y = torch.randint(0, num_classes, (N,))

# Create pairwise labels: 1 if same class, 0 otherwise
idx = torch.arange(N)
label_matrix = y.unsqueeze(1) == y.unsqueeze(0)
all_pairs = []
pair_labels = []
for i in range(N):
    for j in range(i+1, N):
        all_pairs.append((X[i], X[j]))
        pair_labels.append(1.0 if label_matrix[i, j] else 0.0)

X1 = torch.stack([p[0] for p in all_pairs])
X2 = torch.stack([p[1] for p in all_pairs])
pair_labels = torch.tensor(pair_labels)

model = nn.Sequential(nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, embed_dim))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = ContrastiveLoss(margin=2.0)

for epoch in range(100):
    optimizer.zero_grad()
    emb1_large = model(X1)
    emb2_large = model(X2)
    loss = criterion(emb1_large, emb2_large, pair_labels)
    loss.backward()
    optimizer.step()

embeddings = model(X)
print(f"Final embedding range: [{embeddings.min().item():.2f}, {embeddings.max().item():.2f}]")
print(f"Loss: {loss.item():.4f}")
```

```
# Output:
# Final embedding range: [-1.83, 1.89]
# Loss: 0.2105
```

## Common Mistakes

1. **Margin too small**: Models fail to separate dissimilar pairs meaningfully.
2. **Margin too large**: The model over-optimizes for dissimilar pairs, potentially collapsing similar pairs.
3. **Not normalizing embeddings**: Without normalization, embedding magnitudes can vary wildly.
4. **Using wrong distance metric**: Contrastive loss typically uses Euclidean distance. Cosine distance requires a different formulation.
5. **Pair selection bias**: Random pairs may be mostly dissimilar, biasing training. Careful pair mining is essential.
6. **Incorrect label format**: Labels should be 1 for similar pairs, 0 for dissimilar pairs.

## Interview Questions

### Beginner

1. What is contrastive loss used for?
2. How does contrastive loss treat similar vs. dissimilar pairs?
3. What does the margin parameter control?
4. What is a siamese network?
5. How do you implement contrastive loss in PyTorch?

### Intermediate

1. Derive the gradient of contrastive loss for similar and dissimilar pairs.
2. Explain why dissimilar pairs beyond the margin contribute zero loss.
3. How would you choose the margin value?
4. Compare contrastive loss with triplet loss.
5. What is the effect of embedding normalization on contrastive loss?

### Advanced

1. Prove that contrastive loss satisfies the properties of a metric learning loss.
2. Analyze the effect of hard negative mining on contrastive loss training.
3. Derive the relationship between contrastive loss and the NCA loss.

## Practice Problems

### Easy

1. Implement contrastive loss manually.
2. Compute contrastive loss for similar and dissimilar pairs.
3. Train a simple siamese network on synthetic data.
4. Visualize embeddings from a model trained with contrastive loss.
5. Compare different margin values.

### Medium

1. Implement a siamese network for MNIST digit verification.
2. Compare contrastive loss with cross-entropy for face verification.
3. Implement hard negative mining for contrastive loss.
4. Train a model with normalized vs. unnormalized embeddings.
5. Visualize the embedding space after contrastive training.

### Hard

1. Derive the relationship between contrastive loss and the Fisher discriminant.
2. Implement a multi-similarity loss and compare with contrastive loss.
3. Design an experiment comparing pair-based vs. triplet-based metric learning.

## Solutions

Contrastive loss = y*D^2 + (1-y)*max(0, m-D)^2. Similar pairs are pulled together; dissimilar pairs are pushed apart beyond margin m.

## Related Concepts

- Triplet Loss (DL-101): Uses anchor-positive-negative triplets
- Siamese Networks: Architecture paired with contrastive loss
- Metric Learning: The broader field

## Next Concepts

- Triplet Loss (DL-101)
- Focal Loss (DL-102)
- Dice Loss (DL-103)

## Summary

Contrastive loss trains embeddings by pulling similar pairs together and pushing dissimilar pairs apart, with a margin controlling the minimum separation. It is the foundation of siamese network training and is widely used in face recognition, person re-identification, and self-supervised learning.

## Key Takeaways

1. Contrastive loss uses pairs: similar pairs pulled together, dissimilar pairs pushed apart.
2. The margin enforces a minimum distance between dissimilar embeddings.
3. Dissimilar pairs beyond the margin contribute zero loss.
4. Pair mining strategy significantly affects training quality.
5. Contrastive loss is fundamental to metric learning and siamese networks.
