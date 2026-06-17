# Concept: Few-shot Image Classification

## Concept ID

DL-228

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Image Classification

## Learning Objectives

- Understand the few-shot learning problem setup
- Implement metric-based few-shot learning (prototypical networks)
- Analyze meta-learning approaches
- Evaluate few-shot classification performance

## Prerequisites

DL-224 Feature Extraction, DL-200 ResNet, DL-105 Loss Functions

## Definition

Few-shot image classification aims to classify images from novel classes using only a small number of labeled examples per class (typically 1-5), often using meta-learning strategies that learn to learn from limited data.

## Intuition

Humans can recognize a new object from just one picture. Few-shot learning aims to give machines this ability. Instead of training on thousands of examples per class, the model is trained on many "episodes" where it learns to compare and contrast across classes. A common approach is metric learning: learn a feature space where same-class examples cluster together and different-class examples are far apart. Given a new query image, you compare it to the few available examples and classify based on similarity.

## Why This Concept Matters

Few-shot learning addresses a fundamental limitation of deep learning: the need for large labeled datasets. In many real-world scenarios (rare species identification, medical imaging, product recognition), collecting many labeled examples is impractical. Few-shot methods make deep learning applicable to these data-scarce domains.

## Mathematical Explanation

**N-way K-shot problem**: Given N novel classes with K labeled examples each, classify query images.

**Prototypical Networks**:
1. Compute prototype $p_c$ for each class $c$:
$$p_c = \frac{1}{K} \sum_{i=1}^{K} f_\theta(x_{c,i})$$

2. Classify query $q$ by nearest prototype:
$$P(y=c|q) = \frac{\exp(-d(f_\theta(q), p_c))}{\sum_{c'} \exp(-d(f_\theta(q), p_{c'}))}$$

Where $d$ is Euclidean distance and $f_\theta$ is the embedding network.

**Meta-learning**: $f_\theta$ is trained across many episodes (tasks), each sampled from base classes.

## Code Examples

### Example 1: Prototypical Network

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(42)

class ProtoNet(nn.Module):
    """Prototypical Networks for few-shot learning."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
    
    def forward(self, x):
        return self.encoder(x)
    
    def compute_prototypes(self, support, support_labels, n_way):
        """Compute class prototypes from support set."""
        prototypes = []
        for c in range(n_way):
            class_samples = support[support_labels == c]
            prototype = class_samples.mean(dim=0)
            prototypes.append(prototype)
        return torch.stack(prototypes)
    
    def prototypical_loss(self, query, query_labels, prototypes):
        """Compute loss using prototype distance."""
        dists = torch.cdist(query, prototypes)  # (n_query, n_way)
        log_probs = F.log_softmax(-dists, dim=1)
        return F.nll_loss(log_probs, query_labels)

# Create random few-shot episode
n_way, k_shot, n_query = 5, 3, 5
support = torch.randn(n_way * k_shot, 3, 28, 28)
support_labels = torch.arange(n_way).repeat_interleave(k_shot)
query = torch.randn(n_way * n_query, 3, 28, 28)
query_labels = torch.arange(n_way).repeat_interleave(n_query)

model = ProtoNet()
embeddings = model(support)
prototypes = model.compute_prototypes(embeddings, support_labels, n_way)
query_embeddings = model(query)
loss = model.prototypical_loss(query_embeddings, query_labels, prototypes)

print(f"Support: {support.shape} -> Prototypes: {prototypes.shape}")
print(f"Query: {query.shape} -> Query embeddings: {query_embeddings.shape}")
print(f"Loss: {loss.item():.4f}")
```

### Example 2: Few-shot Training Loop

```python
import torch
import torch.optim as optim

torch.manual_seed(42)

def create_episode(base_dataset, n_way, k_shot, n_query=5):
    """Create a few-shot episode from base dataset."""
    classes = torch.randperm(len(base_dataset))[:n_way]
    support_x, support_y = [], []
    query_x, query_y = [], []
    
    for i, c in enumerate(classes):
        class_samples = base_dataset[c]
        indices = torch.randperm(len(class_samples))
        s_idx = indices[:k_shot]
        q_idx = indices[k_shot:k_shot+n_query]
        
        for idx in s_idx:
            support_x.append(class_samples[idx])
            support_y.append(i)
        for idx in q_idx:
            query_x.append(class_samples[idx])
            query_y.append(i)
    
    return (torch.stack(support_x), torch.tensor(support_y),
            torch.stack(query_x), torch.tensor(query_y))

# Simulate a base dataset 
class DummyDataset:
    def __init__(self, n_classes=20, n_samples=50):
        self.data = {c: torch.randn(n_samples, 3, 28, 28) for c in range(n_classes)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

dataset = DummyDataset(n_classes=20, n_samples=50)

model = ProtoNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
n_episodes = 100
for episode in range(n_episodes):
    n_way = 5
    k_shot = 3
    
    s_x, s_y, q_x, q_y = create_episode(dataset, n_way, k_shot)
    
    optimizer.zero_grad()
    s_emb = model(s_x)
    q_emb = model(q_x)
    prototypes = model.compute_prototypes(s_emb, s_y, n_way)
    loss = model.prototypical_loss(q_emb, q_y, prototypes)
    loss.backward()
    optimizer.step()
    
    if (episode + 1) % 20 == 0:
        print(f"Episode {episode+1}: Loss={loss.item():.4f}")
```

### Example 3: Evaluation (N-way K-shot Accuracy)

```python
import torch

torch.manual_seed(42)

def evaluate_fewshot(model, dataset, n_way=5, k_shot=1, n_episodes=100):
    """Evaluate few-shot accuracy."""
    model.eval()
    accuracies = []
    
    for _ in range(n_episodes):
        s_x, s_y, q_x, q_y = create_episode(dataset, n_way, k_shot, n_query=10)
        
        with torch.no_grad():
            s_emb = model(s_x)
            q_emb = model(q_x)
            prototypes = model.compute_prototypes(s_emb, s_y, n_way)
            
            dists = torch.cdist(q_emb, prototypes)
            predictions = (-dists).argmax(dim=1)
            acc = (predictions == q_y).float().mean().item()
            accuracies.append(acc)
    
    mean_acc = sum(accuracies) / len(accuracies)
    conf_interval = 1.96 * (sum((a - mean_acc)**2 for a in accuracies) / len(accuracies)) ** 0.5
    
    return mean_acc, conf_interval

# Evaluate
dataset = DummyDataset(n_classes=50, n_samples=100)
model.eval()

# 5-way 1-shot
acc_1shot, ci_1shot = evaluate_fewshot(model, dataset, n_way=5, k_shot=1)
# 5-way 5-shot
acc_5shot, ci_5shot = evaluate_fewshot(model, dataset, n_way=5, k_shot=5)

print(f"5-way 1-shot: {acc_1shot*100:.1f}% +/- {ci_1shot*100:.1f}%")
print(f"5-way 5-shot: {acc_5shot*100:.1f}% +/- {ci_5shot*100:.1f}%")
```

## Common Mistakes

1. **Data leakage between base and novel classes**: Base and novel classes must be disjoint.
2. **Not normalizing embeddings**: Prototypical networks work best with normalized embeddings.
3. **Using the wrong distance metric**: Euclidean distance works better than cosine for prototypes.
4. **Too few episodes**: Few-shot performance varies significantly across episodes; evaluate over many.
5. **Training and evaluation mismatch**: Train with N-way K-shot episodes, evaluate with same setup.

## Interview Questions

### Beginner - 5
1. What is few-shot learning?
2. What is N-way K-shot classification?
3. What is a support set vs query set?
4. What is a prototype?
5. Give an example where few-shot learning is useful.

### Intermediate - 5
1. Explain how prototypical networks work.
2. Compare metric-based vs optimization-based meta-learning.
3. How does MAML work?
4. What is the difference between few-shot and zero-shot learning?
5. How do you prevent overfitting in few-shot learning?

### Advanced - 3
1. Design a few-shot learning method for cross-domain scenarios.
2. Analyze the theoretical guarantees of prototypical networks.
3. Implement a transformer-based few-shot learner.

## Practice Problems

### Easy - 5
1. Implement a 5-way 1-shot evaluation.
2. Compute prototypes from support embeddings.
3. Implement cosine similarity-based classification.
4. Create random few-shot episodes.
5. Count parameters in a prototypical network.

### Medium - 5
1. Implement Matching Networks.
2. Train and evaluate a prototypical network on miniImageNet.
3. Compare Euclidean vs cosine distance in prototypes.
4. Implement MAML (first-order).
5. Add data augmentation for few-shot learning.

### Hard - 3
1. Implement a transformer-based few-shot learner (FEAT, CTX).
2. Design a cross-domain few-shot learning benchmark.
3. Implement self-supervised pre-training for few-shot learning.

## Solutions

### Easy - 1 Solution
```python
def compute_prototypes(embeddings, labels, n_way):
    prototypes = []
    for c in range(n_way):
        prototypes.append(embeddings[labels == c].mean(0))
    return torch.stack(prototypes)
```

## Related Concepts

DL-229 Zero-shot Image Classification, DL-224 Feature Extraction, DL-225 Classification Head

## Next Concepts

DL-229 Zero-shot Image Classification

## Summary

Few-shot image classification enables learning from very few labeled examples using meta-learning strategies. Prototypical networks learn a feature space where classes are represented by prototypes, and classification is based on nearest-prototype distance in that space.

## Key Takeaways

- N-way K-shot: classify among N classes using K examples each
- Support set: labeled examples; Query set: examples to classify
- Prototypical networks: class prototype = mean of support embeddings
- Distance-based classification: query assigned to nearest prototype
- Episodic training: each episode is a few-shot task
- Euclidean distance works best for prototypes
- Embedding normalization improves performance
- Base and novel classes must be disjoint
- Meta-learning: learning to learn from few examples
- Applications: rare species, medical imaging, personalization
