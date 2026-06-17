# Concept: Hierarchical Classification

## Concept ID

DL-227

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Image Classification

## Learning Objectives

- Understand hierarchical classification and label taxonomies
- Implement hierarchical classifiers in PyTorch
- Leverage label hierarchy for better predictions
- Evaluate hierarchical classification performance

## Prerequisites

DL-226 Multilabel Image Classification, DL-225 Classification Head

## Definition

Hierarchical classification leverages a predefined taxonomy of labels (e.g., "dog" -> "mammal" -> "animal") to structure predictions, enabling models to make coarse-to-fine decisions and exploit relationships between parent and child classes.

## Intuition

When identifying an animal, humans naturally reason hierarchically: "it's an animal, it's a mammal, it's a carnivore, it's a dog, it's a golden retriever." Hierarchical classification mimics this: the model first predicts coarse categories (animal, plant, vehicle), then finer subcategories. This has several advantages: it provides robustness (if the fine prediction is wrong, the coarse prediction may still be right), it uses fewer classifiers (each level only distinguishes among sibling classes), and it reflects the natural structure of the world.

## Why This Concept Matters

Hierarchical classification is essential when dealing with large, structured taxonomies (ImageNet has 1000+ fine classes under ~20 coarse categories). It improves accuracy, provides interpretability (you can see at what level the model fails), and aligns with how human knowledge is organized.

## Mathematical Explanation

**Hierarchical loss**:
$$L = \sum_{n=1}^{N} \sum_{l=1}^{L} w_l \cdot \ell(y_{n,l}, \hat{y}_{n,l})$$

Where $L$ is the number of hierarchy levels, $y_{n,l}$ is the target at level $l$, and $w_l$ is the weight for level $l$.

**Conditional probability**:
$$P(\text{dog} | \text{mammal}) = \frac{P(\text{dog})}{P(\text{mammal})}$$

The probability of a fine class conditioned on its parent.

**Evaluation metrics**:
- **Tree loss**: Penalizes predictions based on their distance in the hierarchy
- **Hierarchical precision/recall**: Considers partial credit for ancestor classes

## Code Examples

### Example 1: Hierarchical Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Define a simple hierarchy
# Level 0: animal, plant, vehicle (3 coarse)
# Level 1 (animal): mammal, bird, fish (3 fine-animal)
# Level 1 (plant): flower, tree (2 fine-plant)
# Level 1 (vehicle): car, plane (2 fine-vehicle)

class HierarchicalClassifier(nn.Module):
    def __init__(self, num_features=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, num_features),
        )
        
        # Coarse classifier (level 0)
        self.coarse = nn.Linear(num_features, 3)
        
        # Fine classifiers (level 1) — one per coarse class
        self.fine_0 = nn.Linear(num_features, 3)  # animal -> mammal, bird, fish
        self.fine_1 = nn.Linear(num_features, 2)  # plant -> flower, tree
        self.fine_2 = nn.Linear(num_features, 2)  # vehicle -> car, plane
    
    def forward(self, x):
        features = self.backbone(x)
        
        # Coarse prediction
        coarse_logits = self.coarse(features)
        coarse_probs = F.softmax(coarse_logits, dim=1)
        
        # Fine predictions (all branches)
        fine_logits = [
            self.fine_0(features),
            self.fine_1(features),
            self.fine_2(features),
        ]
        
        return coarse_logits, fine_logits, coarse_probs

model = HierarchicalClassifier()
x = torch.randn(4, 3, 64, 64)
coarse_logits, fine_logits, coarse_probs = model(x)

print(f"Coarse logits: {coarse_logits.shape}")
for i, fl in enumerate(fine_logits):
    print(f"Fine branch {i}: {fl.shape}")
```

### Example 2: Hierarchical Training

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Simulate hierarchical data
N = 1000
X = torch.randn(N, 3, 64, 64)

# Hierarchy: coarse (0/1/2) and fine (per coarse)
coarse_labels = torch.randint(0, 3, (N,))
fine_labels = torch.zeros(N, dtype=torch.long)
for i in range(N):
    if coarse_labels[i] == 0:  # animal
        fine_labels[i] = torch.randint(0, 3, (1,))
    elif coarse_labels[i] == 1:  # plant
        fine_labels[i] = torch.randint(0, 2, (1,))
    else:  # vehicle
        fine_labels[i] = torch.randint(0, 2, (1,))

dataset = TensorDataset(X, coarse_labels, fine_labels)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = HierarchicalClassifier().to(device)
coarse_ce = nn.CrossEntropyLoss()
fine_ce = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(10):
    model.train()
    total_loss = 0.0
    
    for images, c_labels, f_labels in loader:
        images = images.to(device)
        c_labels, f_labels = c_labels.to(device), f_labels.to(device)
        
        optimizer.zero_grad()
        coarse_logits, fine_logits, _ = model(images)
        
        # Coarse loss
        loss = coarse_ce(coarse_logits, c_labels)
        
        # Fine loss (only relevant branch)
        for i in range(len(images)):
            branch = c_labels[i].item()
            loss += fine_ce(fine_logits[branch][i:i+1], f_labels[i:i+1])
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}")
```

### Example 3: Hierarchical Inference

```python
import torch
import torch.nn.functional as F

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = HierarchicalClassifier().to(device)
model.eval()

# Inference: predict coarse -> select branch -> predict fine
x = torch.randn(4, 3, 64, 64).to(device)

with torch.no_grad():
    coarse_logits, fine_logits, coarse_probs = model(x)
    
    # Coarse predictions
    coarse_preds = coarse_logits.argmax(dim=1)
    coarse_names = ['animal', 'plant', 'vehicle']
    
    # Fine predictions (from selected branch)
    fine_names = [
        ['mammal', 'bird', 'fish'],  # animal children
        ['flower', 'tree'],          # plant children
        ['car', 'plane'],            # vehicle children
    ]
    
    for i in range(len(x)):
        c_pred = coarse_preds[i].item()
        f_pred = fine_logits[c_pred][i].argmax().item()
        
        print(f"Image {i}: {coarse_names[c_pred]} -> {fine_names[c_pred][f_pred]}")
        
        # Confidence at each level
        print(f"  Coarse confidence: {coarse_probs[i].max():.3f}")
```

## Common Mistakes

1. **Not propagating coarse errors to fine**: If coarse prediction is wrong, fine branch will be wrong too.
2. **Ignoring hierarchy in evaluation**: Treating all mistakes equally loses information.
3. **Training hierarchy levels independently**: Joint training with shared features works better.
4. **Using flat classifiers on hierarchical data**: Ignores the useful structure of the label space.
5. **Hard label assignment**: Soft labels (distribute probability across hierarchy) often work better.

## Interview Questions

### Beginner - 5
1. What is hierarchical classification?
2. How does it differ from flat classification?
3. Why use hierarchical classification?
4. What is a label taxonomy?
5. Give an example of a hierarchical label structure.

### Intermediate - 5
1. Compare flat vs hierarchical classifiers.
2. Explain the coarse-to-fine prediction strategy.
3. How do you train a hierarchical classifier?
4. What are the evaluation metrics for hierarchical classification?
5. How does hierarchy help with rare classes?

### Advanced - 3
1. Design a model that learns the hierarchy from data.
2. Implement a soft hierarchical loss with conditional probabilities.
3. Analyze the theoretical advantages of hierarchical classification for large label spaces.

## Practice Problems

### Easy - 5
1. Define a 2-level hierarchy with 3 coarse and 8 fine classes.
2. Count the number of classifiers needed.
3. Implement a hierarchical evaluation metric.
4. Convert a flat dataset to hierarchical format.
5. Compute hierarchical precision.

### Medium - 5
1. Implement a hierarchical classifier with shared backbone.
2. Train with hierarchical loss (weighted by level).
3. Compare flat vs hierarchical accuracy.
4. Implement conditional probability prediction.
5. Visualize the confusion matrix at each hierarchy level.

### Hard - 3
1. Design a model that learns the hierarchy dynamically.
2. Implement a tree-structured classifier with learned routing.
3. Analyze the information-theoretic advantages of hierarchical classification.

## Solutions

### Easy - 1 Solution
```python
hierarchy = {
    'coarse': ['animal', 'plant', 'vehicle'],
    'fine': {
        'animal': ['mammal', 'bird', 'fish'],
        'plant': ['flower', 'tree'],
        'vehicle': ['car', 'plane'],
    }
}
total_classes = sum(len(v) for v in hierarchy['fine'].values())
print(f"Total fine classes: {total_classes}")
```

## Related Concepts

DL-226 Multilabel Image Classification, DL-225 Classification Head, DL-228 Few-shot Image Classification

## Next Concepts

DL-228 Few-shot Image Classification

## Summary

Hierarchical classification leverages label taxonomies to make structured predictions, improving accuracy, robustness, and interpretability. It's essential for large-scale classification with many fine-grained categories and provides partial credit through the hierarchy.

## Key Takeaways

- Labels organized in a tree/graph structure
- Coarse-to-fine prediction strategy
- Shared backbone features for all levels
- Hierarchical loss: weighted combination of per-level losses
- Evaluation: tree distance, hierarchical precision/recall
- More robust than flat classification
- Useful for large label spaces (thousands of classes)
- Conditional probability: P(fine | coarse)
- Can be extended to dynamic/learned hierarchies
- All mistakes are not equal — hierarchy-aware evaluation is important
