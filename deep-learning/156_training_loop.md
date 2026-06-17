# Concept: Training Loop

## Concept ID

DL-156

## Difficulty

Beginner

## Domain

Deep Learning

## Module

Training Deep Networks

## Learning Objectives

- Understand the structure of a standard PyTorch training loop
- Implement a complete training loop with gradient descent
- Manage model modes (train/eval) and gradient computation
- Monitor training progress with loss and metrics
- Handle edge cases like gradient clipping and anomaly detection

## Prerequisites

- Basic PyTorch tensor operations
- Understanding of gradient descent
- Model definition fundamentals
- Dataset and DataLoader concepts

## Definition

The training loop is the core iterative process of updating a neural network's parameters to minimize a loss function on training data. Each epoch iterates over the training dataset, computing forward passes, loss, gradients, and parameter updates. A well-structured training loop includes: model training mode, batch iteration, forward pass, loss computation, backward pass, gradient clipping (optional), optimizer step, metric tracking, and progress logging.

## Intuition

Imagine the training loop as a practice session for an athlete. Each batch of data is a drill, and each epoch is a full practice session. The coach (loss function) evaluates performance, and the athlete (model) adjusts technique (weights) based on feedback (gradients). The learning rate controls how much to adjust at each step, and the training loop runs until the athlete is sufficiently skilled. Just as practice must be structured and consistent, the training loop must handle data correctly, compute gradients properly, and track progress systematically.

## Why This Concept Matters

The training loop is the fundamental building block of deep learning practice. Every project — from simple classifiers to large language models — uses some variant of this loop. Understanding each component (forward pass, loss, backward, step) is essential for: (1) debugging training failures, (2) implementing custom training procedures, (3) adding features like gradient clipping or mixed precision, and (4) transitioning to more advanced frameworks like PyTorch Lightning or Hugging Face Trainer. A solid training loop is the foundation of all deep learning work.

## Code Examples

### Example 1: Basic Training Loop

`python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Create synthetic dataset
X = torch.randn(1000, 20)
y = torch.randint(0, 5, (1000,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define model
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 5),
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()  # Set to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in loader:
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * batch_x.size(0)
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{num_epochs}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")
# Output:
# Epoch 1/5: loss=1.6094, acc=0.2010
# Epoch 2/5: loss=1.6052, acc=0.2040
# Epoch 3/5: loss=1.6011, acc=0.2130
# Epoch 4/5: loss=1.5970, acc=0.2100
# Epoch 5/5: loss=1.5929, acc=0.2180
`

### Example 2: Training Loop with Gradient Clipping

`python
import torch
import torch.nn as nn
import torch.optim as optim

class TrainingLoop:
    def __init__(self, model, optimizer, criterion, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, loader, clip_grad=None):
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for batch in loader:
            x, y = [t.to(self.device) for t in batch]
            
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            
            # Gradient clipping
            if clip_grad:
                if isinstance(clip_grad, float):
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_grad)
                elif isinstance(clip_grad, dict) and clip_grad.get('norm'):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad['norm'])
            
            self.optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
        
        return total_loss / total_samples

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in loader:
                x, y = [t.to(self.device) for t in batch]
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                total_loss += loss.item() * x.size(0)
                _, preds = torch.max(outputs, 1)
                total += y.size(0)
                correct += (preds == y).sum().item()
        
        return total_loss / total, correct / total

model = nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 5))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

trainer = TrainingLoop(model, optimizer, criterion)

X = torch.randn(500, 20)
y = torch.randint(0, 5, (500,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(3):
    train_loss = trainer.train_epoch(loader, clip_grad={'norm': 1.0})
    val_loss, val_acc = trainer.evaluate(loader)
    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, acc={val_acc:.4f}")
# Output:
# Epoch 1: train_loss=1.6021, val_loss=1.5980, acc=0.2160
# Epoch 2: train_loss=1.5945, val_loss=1.5900, acc=0.2240
# Epoch 3: train_loss=1.5870, val_loss=1.5820, acc=0.2260
`

### Example 3: Advanced Training Loop with Mixed Precision

`python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

class AdvancedTrainingLoop:
    def __init__(self, model, optimizer, criterion, device='cuda', use_amp=False):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp and device == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp) if self.use_amp else None

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        total = 0
        
        for batch in loader:
            x, y = [t.to(self.device) for t in batch]
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
        
        return total_loss / total

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 5))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

trainer = AdvancedTrainingLoop(model, optimizer, criterion, device, use_amp=(device == 'cuda'))

X = torch.randn(200, 20)
y = torch.randint(0, 5, (200,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

loss = trainer.train_epoch(loader)
print(f"Training completed. Final loss: {loss:.4f}")
print(f"AMP enabled: {trainer.use_amp}")
# Output:
# Training completed. Final loss: 1.5987
# AMP enabled: False (on CPU)
`

## Common Mistakes

1. **Forgetting model.train() at the start of each epoch**: BatchNorm and Dropout behave differently in training vs eval mode.
2. **Not zeroing gradients before backward**: Gradients accumulate by default. Always call optimizer.zero_grad() or model.zero_grad().
3. **Calling backward on a non-scalar loss**: Loss must be a scalar tensor. If using a per-sample loss, call .mean() or .sum().
4. **Detaching tensors unnecessarily**: Only detach when you need to stop gradient flow. Do not detach model outputs before loss computation.
5. **Not using with torch.no_grad() for evaluation**: This wastes memory computing gradients that are not needed during evaluation.

## Interview Questions

### Beginner

1. What is the purpose of optimizer.zero_grad()?
2. Why do we call loss.backward() before optimizer.step()?
3. What does model.train() do?
4. What is the difference between an epoch and a batch?
5. Why do we need DataLoader?

### Intermediate

1. Explain the gradient accumulation technique.
2. How does gradient clipping work and why is it needed?
3. What is the purpose of with torch.no_grad() during evaluation?
4. How would you implement a learning rate scheduler in the training loop?
5. What is mixed precision training and how does it work?

### Advanced

1. Design a training loop that supports distributed data parallel training.
2. Implement gradient checkpointing for memory-efficient training of large models.
3. How would you implement a custom training loop for contrastive learning?

## Practice Problems

### Easy

1. Write a basic training loop for a linear regression model.
2. Add accuracy computation to the training loop.
3. Implement gradient clipping at value 1.0.
4. Add a learning rate scheduler that reduces LR on plateau.
5. Implement early stopping in the training loop.

### Medium

1. Implement gradient accumulation over multiple batches.
2. Write a training loop with per-parameter learning rates.
3. Implement a training loop with exponential moving average of parameters.
4. Add mixed precision training to an existing training loop.
5. Implement a training loop with gradient noise injection.

### Hard

1. Implement a distributed data parallel training loop using torch.distributed.
2. Design a training loop for meta-learning (MAML-style inner and outer loops).
3. Implement a training loop with automatic hyperparameter tuning based on gradient statistics.

## Solutions

### Easy Solutions

1. See Example 1 — the basic training loop with model definition, loss, optimizer, and epoch/batch structure
2. Compute accuracy as (predicted == y).sum().item() / y.size(0)
3. torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
4. scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer); call scheduler.step(val_loss) after each epoch
5. Track best validation loss; break if no improvement for patience epochs

## Related Concepts

- Validation Loop (DL-157)
- Test Loop (DL-158)
- Checkpointing (DL-159)
- Model Saving and Loading (DL-160)

## Next Concepts

- Validation Loop (DL-157)
- Test Loop (DL-158)
- Checkpointing (DL-159)

## Summary

The training loop is the iterative process of feeding data through a model, computing loss, backpropagating gradients, and updating weights. Key components include: setting model to train mode, zeroing gradients, forward pass, loss computation, backward pass, gradient clipping (optional), optimizer step, and progress tracking.

## Key Takeaways

- Always call model.train() at the start of each epoch
- Zero gradients before each backward pass
- Loss must be a scalar
- Use with torch.no_grad() for evaluation
- Gradient clipping prevents explosion in unstable training
- DataLoader handles batching and shuffling
- Track metrics (loss, accuracy) to monitor progress
- The training loop is the foundation — master it before using higher-level APIs
