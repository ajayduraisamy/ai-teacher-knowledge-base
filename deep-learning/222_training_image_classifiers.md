# Concept: Training Image Classifiers

## Concept ID

DL-222

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Image Classification

## Learning Objectives

- Understand the complete training workflow for image classifiers
- Implement training loops with proper optimization
- Apply regularization and learning rate scheduling
- Monitor and debug training progress

## Prerequisites

DL-217 Image Classification Pipeline, DL-221 ImageDataGenerator, DL-171 Learning Rate Selection

## Definition

Training image classifiers involves optimizing a neural network's parameters using labeled image data through iterative gradient-based optimization, with careful management of learning rate, regularization, and evaluation to achieve good generalization.

## Intuition

Training a classifier is like teaching a student with flashcards. You show an image (forward pass), see what the student predicts, tell them the correct answer (loss), and the student adjusts their understanding (backpropagation). Repeat thousands of times with millions of examples. The key is to do this efficiently — using the right learning rate (how big a mental adjustment each time), regularization (preventing the student from memorizing rather than understanding), and monitoring (checking if the student is actually learning or just memorizing).

## Why This Concept Matters

Training is where the magic happens. Understanding training techniques — from basic loops to advanced optimization — is essential for achieving good model performance. Poor training can make even the best architecture underperform.

## Mathematical Explanation

**Training objective**:
$$\theta^* = \arg\min_\theta \frac{1}{N} \sum_{i=1}^{N} \ell(f(x_i; \theta), y_i)$$

**Standard training loop components**:
1. Forward pass: $\hat{y} = f(x; \theta)$
2. Loss computation: $L = \ell(\hat{y}, y)$
3. Backward pass: $\nabla_\theta L$
4. Optimizer step: $\theta \leftarrow \theta - \eta \cdot \text{optimizer}(\nabla_\theta L)$
5. Learning rate schedule: $\eta_t = \text{schedule}(t)$

**Regularization techniques**:
- Weight decay (L2): $L_{reg} = L + \lambda \sum \theta_i^2$
- Dropout: $h_{drop} = h \cdot \text{Bernoulli}(p)$
- Label smoothing: $y_{smooth} = (1-\epsilon)y + \epsilon/K$

## Code Examples

### Example 1: Complete Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

torch.manual_seed(42)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create synthetic data
X = torch.randn(5000, 3, 32, 32)
y = torch.randint(0, 10, (5000,))
X_train, X_val = X[:4000], X[4000:]
y_train, y_val = y[:4000], y[4000:]

train_loader = DataLoader(TensorDataset(X_train, y_train), 
                          batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val),
                        batch_size=64, shuffle=False)

# Model
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    nn.Dropout(0.5),
    nn.Linear(64, 10),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# Training loop
num_epochs = 50
best_val_acc = 0.0

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    correct = total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    scheduler.step()
    
    train_acc = 100.0 * correct / total
    val_acc = 100.0 * val_correct / val_total
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:2d}: Train Loss={train_loss/len(train_loader):.4f}, "
              f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')

print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
```

### Example 2: Advanced Training Techniques

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Gradient clipping
def train_with_gradient_clipping(model, loader, epochs=10):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs} completed")

# Mixed precision training
def train_mixed_precision(model, loader, epochs=5):
    from torch.cuda.amp import autocast, GradScaler
    
    scaler = GradScaler()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        print(f"Mixed precision epoch {epoch+1}/{epochs} done")

# Label smoothing
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smoothed = (1 - self.smoothing) * one_hot + self.smoothing / n_classes
        log_probs = torch.log_softmax(pred, dim=1)
        return -(smoothed * log_probs).sum(dim=1).mean()

print("Training techniques demonstrated:")
print("- Gradient clipping: prevents exploding gradients")
print("- Mixed precision: faster training with FP16")
print("- Label smoothing: prevents overconfidence")
```

### Example 3: Training CIFAR-10 with Modern Techniques

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.CoarseDropoutV2(1, 8, fill_value=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model (ResNet-18 for CIFAR-10)
from torchvision.models import resnet18
model = resnet18(num_classes=10).to(device)

# Training setup
criterion = LabelSmoothingLoss(smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, 
                      weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], 
                                          gamma=0.2)

# Train
epochs = 200
for epoch in range(epochs):
    model.train()
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    scheduler.step()
    
    if (epoch + 1) % 20 == 0:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                _, predicted = model(images).max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        print(f"Epoch {epoch+1}: Test Acc={100.*correct/total:.2f}%")
```

## Common Mistakes

1. **Not using validation set**: Training accuracy alone doesn't tell you if the model generalizes.
2. **Overfitting without regularization**: Models with high capacity need dropout, weight decay, or data augmentation.
3. **Learning rate too high or too low**: Can cause divergence or extremely slow convergence.
4. **Not shuffling training data**: Leads to biased gradients and poor convergence.
5. **Using test set for validation**: Never use the test set for model selection; it's for final evaluation only.

## Interview Questions

### Beginner - 5
1. What is the training loop?
2. Why do we need a validation set?
3. What is a training epoch?
4. What is the difference between training and inference mode?
5. How often should you validate?

### Intermediate - 5
1. Explain the role of the optimizer in training.
2. How does learning rate scheduling improve training?
3. What is weight decay and why is it used?
4. Compare SGD with momentum vs Adam.
5. How do you diagnose overfitting during training?

### Advanced - 3
1. Design a training strategy for a dataset with 1 billion images.
2. Implement a custom learning rate schedule.
3. Analyze the impact of batch size on generalization.

## Practice Problems

### Easy - 5
1. Implement a basic training loop.
2. Add model checkpointing.
3. Implement early stopping.
4. Add data augmentation to training.
5. Log loss and accuracy to a file.

### Medium - 5
1. Train ResNet-18 on CIFAR-10 with modern techniques.
2. Implement cosine annealing with warm restarts.
3. Add gradient accumulation for large effective batch sizes.
4. Compare training with and without weight decay.
5. Implement knowledge distillation.

### Hard - 3
1. Implement a distributed training loop with DDP.
2. Design a curriculum learning schedule.
3. Implement self-supervised pre-training + fine-tuning.

## Solutions

### Easy - 1 Solution
```python
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(epochs):
    for x, y in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
```

## Related Concepts

DL-217 Image Classification Pipeline, DL-171 Learning Rate Selection, DL-223 Finetuning Image Classifiers

## Next Concepts

DL-223 Finetuning Image Classifiers

## Summary

Training image classifiers involves iterative optimization over labeled data with careful management of learning rate, regularization, and evaluation. Modern training incorporates techniques like gradient clipping, label smoothing, mixed precision, and sophisticated LR schedules.

## Key Takeaways

- Standard loop: forward, loss, backward, optimize, repeat
- Validation set is essential for detecting overfitting
- Regularization: weight decay, dropout, data augmentation, label smoothing
- LR schedules: step decay, cosine annealing, warmup
- Gradient clipping prevents exploding gradients
- Mixed precision training for faster GPU utilization
- Checkpointing the best model is critical
- Test set is only for final evaluation
- Training recipe matters as much as architecture
- Experiment tracking (loss curves, metrics, hyperparameters) is essential
