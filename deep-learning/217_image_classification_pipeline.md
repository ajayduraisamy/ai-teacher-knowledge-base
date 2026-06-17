# Concept: Image Classification Pipeline

## Concept ID

DL-217

## Difficulty

Beginner

## Domain

Deep Learning

## Module

Image Classification

## Learning Objectives

- Understand the complete image classification pipeline
- Implement an end-to-end classification workflow in PyTorch
- Preprocess images for model input
- Evaluate classifier performance

## Prerequisites

DL-216 ImageNet Dataset, DL-105 Loss Functions, DL-101 Neural Networks Basics

## Definition

The image classification pipeline is the end-to-end process of training and deploying a model to assign categorical labels to images, encompassing data loading, preprocessing, model architecture, training, evaluation, and inference.

## Intuition

Image classification is like teaching a child to identify animals. You show many labeled pictures (data loading), point out distinguishing features (training), test with new pictures (evaluation), and eventually the child can identify animals they've never seen before (inference). The pipeline connects all these stages: raw images go through preprocessing (resizing, normalization), the model extracts features and makes predictions, the loss measures mistakes, backpropagation updates the model, and the cycle repeats.

## Why This Concept Matters

Understanding the full pipeline is essential for applying deep learning to any image classification task. It's the template from which all vision applications (detection, segmentation) are built. Every practitioner needs to know each component and how they interconnect.

## Mathematical Explanation

**Pipeline stages**:

1. Data: Raw images $(x_i, y_i)$ with labels
2. Preprocessing: $x_i' = T(x_i)$ (resize, normalize, augment)
3. Forward: $\hat{y}_i = f(x_i'; \theta)$ (model prediction)
4. Loss: $L = \frac{1}{N} \sum_i \ell(\hat{y}_i, y_i)$
5. Backward: $\nabla_\theta L$ (gradient computation)
6. Update: $\theta_{t+1} = \theta_t - \eta \nabla_\theta L$ (optimizer step)

**Cross-entropy loss** for multi-class classification:
$$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

## Code Examples

### Example 1: Complete Classification Pipeline

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torch.nn.functional as F

torch.manual_seed(42)

# 1. Create synthetic image dataset
N = 1000
X = torch.randn(N, 3, 32, 32)
y = torch.randint(0, 10, (N,))

dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 2. Define model
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(32, 10),
)

# 3. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training loop
print("Starting training...")
for epoch in range(5):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    accuracy = 100.0 * correct / total
    print(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, "
          f"Acc={accuracy:.2f}%")
```

### Example 2: Full Pipeline with Real Data (CIFAR-10)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

torch.manual_seed(42)

# 1. Preprocessing
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# 2. Data loading
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# 3. Model
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    nn.Linear(128, 10),
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Train
print("Training on CIFAR-10...")
for epoch in range(3):
    model.train()
    for images, labels in trainloader:
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    
    # 5. Evaluate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    print(f"Epoch {epoch+1}: Test Acc={100.*correct/total:.2f}%")
```

### Example 3: Inference Pipeline

```python
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

torch.manual_seed(42)

# 1. Load a trained model
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    nn.Linear(64, 10),
)
model.eval()

# 2. Inference preprocessing
inference_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# 3. Simulated inference
# In practice: img = Image.open('image.jpg')
# input_tensor = inference_transform(img).unsqueeze(0)

random_input = torch.randn(1, 3, 32, 32)

with torch.no_grad():
    output = model(random_input)
    probabilities = F.softmax(output, dim=1)
    predicted_class = output.argmax(dim=1).item()

print(f"Inference output shape: {output.shape}")
print(f"Predicted class: {predicted_class}")
print(f"Confidence: {probabilities[0, predicted_class].item():.4f}")

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
print(f"Class name: {CIFAR10_CLASSES[predicted_class]}")
```

## Common Mistakes

1. **Training/eval mode mismatch**: Forgetting model.eval() during inference affects BN and Dropout.
2. **Different transforms for train and test**: Augmentation (random crop, flip) should only be used during training.
3. **No gradient detachment during evaluation**: Use torch.no_grad() for inference.
4. **Wrong normalization stats**: Must use training set statistics (mean, std).
5. **Data leakage**: Shuffling, split, or normalization that leaks test information into training.

## Interview Questions

### Beginner - 5
1. What are the stages in an image classification pipeline?
2. Why do we use different transforms for train and test?
3. What is the purpose of model.eval()?
4. What is cross-entropy loss?
5. How do you measure classifier accuracy?

### Intermediate - 5
1. Explain the importance of data normalization.
2. How does batch size affect the training pipeline?
3. What is the role of the validation set in the pipeline?
4. How do you handle class imbalance in classification?
5. Compare training from scratch vs transfer learning.

### Advanced - 3
1. Design a pipeline for production deployment with latency constraints.
2. How would you implement distributed training in the pipeline?
3. Design a pipeline that handles multi-label classification.

## Practice Problems

### Easy - 5
1. Implement a train/eval loop for CIFAR-10.
2. Add model checkpointing to save best model.
3. Implement top-5 accuracy metric.
4. Create confusion matrix from predictions.
5. Add learning rate scheduling.

### Medium - 5
1. Implement a complete pipeline for a custom dataset.
2. Add TensorBoard logging to the pipeline.
3. Implement early stopping.
4. Build a pipeline with k-fold cross validation.
5. Implement mixed precision training.

### Hard - 3
1. Design a production inference pipeline with queuing and batching.
2. Implement a distributed training pipeline.
3. Build an AutoML pipeline that searches over architectures.

## Solutions

### Easy - 1 Solution
```python
model = Model()
for epoch in range(num_epochs):
    model.train()
    for x, y in trainloader:
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            correct += (model(x).argmax(1) == y).sum()
    print(f"Epoch {epoch}: Acc={correct/len(testset):.2%}")
```

## Related Concepts

DL-222 Training Image Classifiers, DL-223 Finetuning Image Classifiers, DL-105 Loss Functions

## Next Concepts

DL-218 Top-1/Top-5 Accuracy, DL-219 Preprocessing for Images

## Summary

The image classification pipeline is the fundamental workflow in computer vision, encompassing data loading, preprocessing, model definition, training, evaluation, and inference. Understanding each component and how they connect is essential for applying deep learning to vision tasks.

## Key Takeaways

- Pipeline: data -> preprocessing -> model -> loss -> backprop -> update
- Training augmentation differs from inference preprocessing
- model.eval() mode is critical for correct inference
- Cross-entropy is the standard loss for multi-class classification
- Data normalization with training set statistics is essential
- Accuracy (top-1, top-5) is the standard evaluation metric
- Batch processing enables GPU-efficient training
- Validation set guides model selection
- Checkpointing saves progress for resumption
- Production pipelines add batching, queuing, and monitoring
