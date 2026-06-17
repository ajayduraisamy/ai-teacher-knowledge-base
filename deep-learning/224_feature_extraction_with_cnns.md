# Concept: Feature Extraction with CNNs

## Concept ID

DL-224

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Image Classification

## Learning Objectives

- Understand how CNNs serve as general-purpose feature extractors
- Implement feature extraction from pretrained CNNs
- Use CNN features for downstream tasks
- Analyze the hierarchy of learned features

## Prerequisites

DL-223 Finetuning Image Classifiers, DL-181 Feature Map, DL-200 ResNet

## Definition

Feature extraction with CNNs involves using a pretrained convolutional neural network to transform raw images into high-dimensional feature vectors that capture semantically meaningful visual information, which can then be used for various downstream tasks.

## Intuition

A CNN trained on ImageNet has learned a rich hierarchy of visual features: early layers detect edges and colors, middle layers detect textures and patterns, and late layers detect objects and scenes. The activations of the last hidden layer (before the classifier) form a compact, information-rich representation of the image. These features are remarkably general — they work well for tasks the model was never trained on. This is why CNN features became the "SIFT of the deep learning era" — a universal visual representation.

## Why This Concept Matters

CNN features enabled the breakthrough in transfer learning that made deep learning practical for small datasets. Instead of training a huge CNN from scratch, you can compute features once, then train a simple classifier. This approach is still widely used and is the foundation of many vision applications.

## Mathematical Explanation

**Feature extraction**:

For a pretrained CNN $f$ with parameters $\theta$, the features for image $x$ are:
$$\phi(x) = f_{L-1}(x; \theta)$$

Where $f_{L-1}$ is the network up to the last hidden layer (before the classifier).

**Feature dimensionality**:
- ResNet-18: 512-dim vector (after global average pooling)
- ResNet-50: 2048-dim vector
- VGG-16: 4096-dim vector (from FC7)
- EfficientNet-B0: 1280-dim vector

**Using features for classification**:
$$y = W \cdot \phi(x) + b$$

Where $W$ and $b$ are learned from the target dataset.

## Code Examples

### Example 1: Extracting Features from ResNet

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

torch.manual_seed(42)

# Load pretrained ResNet without classifier
class FeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet18'):
        super().__init__()
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=False)
            self.feature_dim = 512
            # Remove the FC layer
            self.features = nn.Sequential(*list(self.model.children())[:-1])
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=False)
            self.feature_dim = 2048
            self.features = nn.Sequential(*list(self.model.children())[:-1])
    
    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

extractor = FeatureExtractor('resnet18')
extractor.eval()

print(f"Feature dimension: {extractor.feature_dim}")

# Extract features from a batch of images
x = torch.randn(16, 3, 224, 224)
with torch.no_grad():
    features = extractor(x)

print(f"Input: {x.shape} -> Features: {features.shape}")
```

### Example 2: Feature Extraction Pipeline

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

torch.manual_seed(42)

class FeatureDataset(Dataset):
    """Dataset that lazily extracts features."""
    def __init__(self, images, labels, feature_extractor, transform):
        self.images = images
        self.labels = labels
        self.extractor = feature_extractor
        self.transform = transform
        
        # Pre-extract all features
        print("Pre-extracting features...")
        self.features = []
        with torch.no_grad():
            for img in self.images:
                x = self.transform(img).unsqueeze(0)
                feat = self.extractor(x)
                self.features.append(feat.squeeze(0))
        self.features = torch.stack(self.features)
        print(f"Extracted {len(self.features)} features")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Simulate feature extraction
extractor = FeatureExtractor('resnet18')
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Create synthetic images
fake_images = [Image.fromarray((np.random.rand(256, 256, 3)*255).astype(np.uint8)) 
               for _ in range(100)]
labels = torch.randint(0, 5, (100,))

feature_dataset = FeatureDataset(fake_images, labels, extractor, transform)
print(f"Feature dataset: {len(feature_dataset)} samples, "
      f"dim={feature_dataset[0][0].shape[0]}")
```

### Example 3: Classifier on Extracted Features

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Simulate extracted features
N = 1000
feature_dim = 512
num_classes = 10

features = torch.randn(N, feature_dim)
labels = torch.randint(0, num_classes, (N,))

# Split
train_feat, val_feat = features[:800], features[800:]
train_lbl, val_lbl = labels[:800], labels[800:]

train_loader = DataLoader(TensorDataset(train_feat, train_lbl), 
                          batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(val_feat, val_lbl),
                        batch_size=32)

# Simple classifier on top of features
classifier = nn.Sequential(
    nn.Linear(feature_dim, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# Train classifier
for epoch in range(20):
    classifier.train()
    for feat, lbl in train_loader:
        feat, lbl = feat.to(device), lbl.to(device)
        optimizer.zero_grad()
        out = classifier(feat)
        loss = criterion(out, lbl)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    classifier.eval()
    correct = total = 0
    with torch.no_grad():
        for feat, lbl in val_loader:
            feat, lbl = feat.to(device), lbl.to(device)
            _, pred = classifier(feat).max(1)
            total += lbl.size(0)
            correct += pred.eq(lbl).sum().item()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}: Val Acc={100.*correct/total:.2f}%")
```

## Common Mistakes

1. **Using the wrong layer for features**: Using features from early layers gives low-level info; using classifier features (before softmax) loses spatial structure.
2. **Not using global average pooling before feature extraction**: Without pooling, feature maps are spatial (e.g., 2048x7x7), not vectorized.
3. **Applying incorrect transforms**: Feature extractors expect specific preprocessing.
4. **Not disabling gradient computation**: torch.no_grad() saves memory during feature extraction.
5. **Re-extracting features each epoch**: Extract once, save, and reuse for efficiency.

## Interview Questions

### Beginner - 5
1. What is feature extraction?
2. Why are CNN features general-purpose?
3. What is the feature vector dimension for ResNet-50?
4. How do you extract features from a pretrained model?
5. Why use global average pooling before feature extraction?

### Intermediate - 5
1. Compare features from different layers (early vs late).
2. How does feature extraction enable one-shot learning?
3. Explain the feature hierarchy in CNNs.
4. What distance metric works well with CNN features?
5. How do you handle different input sizes for feature extraction?

### Advanced - 3
1. Design a feature matching algorithm using CNN features.
2. Analyze the semantic meaning of different feature dimensions.
3. Implement unsupervised feature learning with CNNs.

## Practice Problems

### Easy - 5
1. Extract features from ResNet-18 for 100 images.
2. Train a linear classifier on extracted features.
3. Compare features with and without GAP.
4. Compute feature similarity between two images.
5. Visualize features using PCA.

### Medium - 5
1. Build a feature-based image retrieval system.
2. Compare features from different pretrained models.
3. Implement feature normalization and impact on classification.
4. Use features for few-shot learning.
5. Implement a feature pyramid (multi-scale features).

### Hard - 3
1. Design a self-supervised feature learning method.
2. Implement feature inversion (reconstruct image from features).
3. Analyze the information content of features at different layers.

## Solutions

### Easy - 1 Solution
```python
model = models.resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])
x = torch.randn(10, 3, 224, 224)
with torch.no_grad():
    features = model(x).view(10, -1)
print(f"Features shape: {features.shape}")
```

## Related Concepts

DL-223 Finetuning Image Classifiers, DL-225 Classification Head, DL-181 Feature Map

## Next Concepts

DL-225 Classification Head

## Summary

Feature extraction uses pretrained CNNs as universal feature extractors, transforming images into high-dimensional vectors that capture semantic visual information. These features can be used for various downstream tasks, enabling transfer learning with minimal data.

## Key Takeaways

- Features from pretrained CNNs are general-purpose visual representations
- Last hidden layer activations (after GAP) form the feature vector
- Common feature dimensions: ResNet-18=512, ResNet-50=2048, VGG-16=4096
- Extract features once, train simple classifiers on top
- torch.no_grad() essential for memory efficiency
- Layer selection matters: early=low-level, late=semantic
- GAP produces fixed-size vectors regardless of input size
- Features enable fast training and work with limited data
- Foundation for image retrieval, few-shot learning, and more
- Self-supervised learning produces features without labels
