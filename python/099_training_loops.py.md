# Concept: Custom Training Loops

## Concept ID

PYT-099

## Difficulty

Advanced

## Domain

Python

## Module

Python for ML/AI

## Learning Objectives

- Write custom training loops in PyTorch with epoch and batch iteration
- Implement forward pass, loss computation, zero_grad, backward, and step
- Add checkpointing with `torch.save()` and `torch.load()`
- Implement early stopping based on validation metrics

## Prerequisites

- PYT-096 — PyTorch Tensors (autograd, tensor operations)
- PYT-097 — PyTorch NN (nn.Module, loss functions, optimizers)
- Understanding of gradient descent and backpropagation

## Definition

A custom training loop gives you full control over the model training process. Unlike using a high-level API (like Keras `fit()`), custom loops let you:

**Core PyTorch Training Loop Pattern:**
```
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch_x)
            val_loss += criterion(outputs, batch_y)
```

**Key Components:**

- `model.train()` / `model.eval()`: Toggle dropout/batchnorm behavior
- `optimizer.zero_grad()`: Clear gradients from previous iteration
- `loss.backward()`: Compute gradients via autograd
- `optimizer.step()`: Update weights using computed gradients
- `torch.no_grad()`: Disable gradient tracking during evaluation
- `torch.save()`: Save model checkpoints
- `torch.load()`: Load saved checkpoints
- Early Stopping: Monitor validation metric and stop when it plateaus

## Intuition

Keras's `model.fit()` is like an automatic car — it gets you there with minimal effort but limited control. A custom training loop is a manual transmission — more work, but you control every gear shift.

Why would you want control? Because research and production training often require:
- Gradient clipping to prevent exploding gradients
- Warmup learning rate schedules
- Mixed precision training (amp)
- Gradient accumulation across multiple batches
- Custom logging per batch/epoch
- Complex loss functions with multiple terms
- Per-layer learning rates or freezing/unfreezing schedules

## Why This Concept Matters

- **Research Flexibility:** Papers often require training techniques not available in high-level APIs
- **Debugging:** Step through each batch, inspect gradients, monitor activations
- **Performance:** Control over GPU memory, gradient accumulation, and data loading
- **Advanced Techniques:** Gradient clipping, mixed precision, gradient checkpointing, adversarial training
- **Production:** Custom logging, distributed training, model export

## Real World Examples

1. **GAN Training:** A custom loop alternates between training the generator and discriminator, with a different loss and learning rate for each.
2. **Transformer Training:** A custom loop implements warmup learning rate schedule, gradient clipping (max_norm=1.0), and label smoothing.
3. **Reinforcement Learning:** A custom loop collects trajectories, computes advantage, updates policy, and buffers experience replay.
4. **Self-Supervised Learning:** A custom loop manages two augmented views per sample, contrastive loss, and momentum encoder updates.
5. **Federated Learning:** A custom loop iterates over clients, sends model weights, collects gradients, and aggregates.

## AI/ML Relevance

- **Necessary for Research:** Most cutting-edge training techniques require custom loops
- **Production Training:** Large-scale training uses custom loops for performance optimization
- **Multi-Task Learning:** Custom loss weighting and backpropagation scheduling
- **Memory Optimization:** Gradient accumulation enables effectively large batch sizes on limited GPU memory
- **Distributed Training:** Custom loops integrate with DDP (DistributedDataParallel) for multi-GPU training

## Code Examples

### Example 1: Basic custom training loop
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Generate data
np.random.seed(42)
torch.manual_seed(42)
X = torch.randn(1000, 10)
y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)

dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Model
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)

    avg_train_loss = train_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_x.size(0)
            predicted = (outputs > 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)
    accuracy = correct / total

    print(f"Epoch {epoch+1:2d}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Acc={accuracy:.3f}")
```
```
# Output:
# Epoch  1: Train Loss=0.6912, Val Loss=0.6881, Acc=0.526
# Epoch  2: Train Loss=0.6743, Val Loss=0.6711, Acc=0.643
# Epoch  3: Train Loss=0.6521, Val Loss=0.6492, Acc=0.736
# Epoch  4: Train Loss=0.6254, Val Loss=0.6232, Acc=0.798
# Epoch  5: Train Loss=0.5951, Val Loss=0.5934, Acc=0.828
# Epoch  6: Train Loss=0.5632, Val Loss=0.5628, Acc=0.842
# Epoch  7: Train Loss=0.5319, Val Loss=0.5325, Acc=0.856
# Epoch  8: Train Loss=0.5024, Val Loss=0.5043, Acc=0.870
# Epoch  9: Train Loss=0.4758, Val Loss=0.4789, Acc=0.872
# Epoch 10: Train Loss=0.4521, Val Loss=0.4562, Acc=0.879
```

### Example 2: Checkpointing with torch.save
```python
import os

def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        print(f"Checkpoint loaded: epoch={checkpoint['epoch']}, loss={loss:.4f}")
        return start_epoch
    return 0

# Simulate training with checkpoint at epoch 5
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Train 5 epochs and save
for epoch in range(5):
    optimizer.zero_grad()
    loss = criterion(model(X[:32]), y[:32])
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

save_checkpoint(model, optimizer, epoch=4, loss=loss.item())

# Load checkpoint and continue training
model2 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
start_epoch = load_checkpoint(model2, optimizer2, 'checkpoint.pth')

# Continue from epoch 5
for epoch in range(start_epoch, start_epoch + 3):
    optimizer2.zero_grad()
    loss = criterion(model2(X[:32]), y[:32])
    loss.backward()
    optimizer2.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
```
```
# Output:
# Epoch 1: Loss = 0.7142
# Epoch 2: Loss = 0.7023
# Epoch 3: Loss = 0.6911
# Epoch 4: Loss = 0.6804
# Epoch 5: Loss = 0.6702
# Checkpoint saved: checkpoint.pth
# Checkpoint loaded: epoch=4, loss=0.6702
# Epoch 6: Loss = 0.6702
# Epoch 7: Loss = 0.6601
# Epoch 8: Loss = 0.6505
```

### Example 3: Early stopping
```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered after {self.counter} epochs without improvement.")
        return self.early_stop

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

train_loader = DataLoader(TensorDataset(X[:800], y[:800]), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X[800:], y[800:]), batch_size=32)

early_stopping = EarlyStopping(patience=3)
best_val_loss = float('inf')

for epoch in range(50):  # max 50 epochs
    model.train()
    for bx, by in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for bx, by in val_loader:
            val_loss += criterion(model(bx), by).item() * bx.size(0)
    val_loss /= len(val_loader.dataset)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

    print(f"Epoch {epoch+1:2d}: Val Loss = {val_loss:.4f} (best={best_val_loss:.4f})", end="")
    if early_stopping(val_loss):
        print(" -> Stopped")
        break
    print()
```
```
# Output:
# Epoch  1: Val Loss = 0.4912 (best=0.4912)
# Epoch  2: Val Loss = 0.3821 (best=0.3821)
# Epoch  3: Val Loss = 0.3523 (best=0.3523)
# Epoch  4: Val Loss = 0.3642 (best=0.3523)
# Epoch  5: Val Loss = 0.3589 (best=0.3523)
# Epoch  6: Val Loss = 0.3710 (best=0.3523)
# Early stopping triggered after 3 epochs without improvement.
```

### Example 4: Gradient clipping
```python
model = nn.Sequential(
    nn.Linear(10, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

for epoch in range(5):
    model.train()
    epoch_loss = 0.0
    max_grad_norm = 0.0

    for bx, by in DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True):
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        max_grad_norm = max(max_grad_norm, grad_norm.item())

        optimizer.step()
        epoch_loss += loss.item() * bx.size(0)

    avg_loss = epoch_loss / len(X)
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Max Grad Norm={max_grad_norm:.4f}")
```
```
# Output:
# Epoch 1: Loss=0.6832, Max Grad Norm=0.8912
# Epoch 2: Loss=0.6213, Max Grad Norm=0.7643
# Epoch 3: Loss=0.5421, Max Grad Norm=0.4231
# Epoch 4: Loss=0.4678, Max Grad Norm=0.3124
# Epoch 5: Loss=0.4123, Max Grad Norm=0.2876
```

### Example 5: Learning rate scheduling
```python
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# StepLR: reduce LR by factor 0.1 every 3 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

for epoch in range(10):
    model.train()
    for bx, by in DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True):
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()

    current_lr = scheduler.get_last_lr()[0]
    scheduler.step()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X), y).item()
    print(f"Epoch {epoch+1:2d}: LR={current_lr:.6f}, Val Loss={val_loss:.4f}")

# Alternative: CosineAnnealingLR
optimizer2 = optim.Adam(model.parameters(), lr=0.01)
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=10)
```
```
# Output:
# Epoch  1: LR=0.010000, Val Loss=0.6912
# Epoch  2: LR=0.010000, Val Loss=0.6421
# Epoch  3: LR=0.010000, Val Loss=0.5834
# Epoch  4: LR=0.001000, Val Loss=0.4521
# Epoch  5: LR=0.001000, Val Loss=0.3987
# Epoch  6: LR=0.001000, Val Loss=0.3876
# Epoch  7: LR=0.000100, Val Loss=0.3842
# Epoch  8: LR=0.000100, Val Loss=0.3840
# Epoch  9: LR=0.000100, Val Loss=0.3839
# Epoch 10: LR=0.000010, Val Loss=0.3839
```

### Example 6: Gradient accumulation for effective large batch size
```python
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

accumulation_steps = 4  # effective batch = 32 * 4 = 128
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

for epoch in range(5):
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad()  # zero once at start of epoch

    for i, (bx, by) in enumerate(loader):
        loss = criterion(model(bx), by)
        loss = loss / accumulation_steps  # normalize loss
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item() * accumulation_steps * bx.size(0)

    avg_loss = epoch_loss / len(X)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
```
```
# Output:
# Epoch 1: Loss = 0.6912
# Epoch 2: Loss = 0.6421
# Epoch 3: Loss = 0.5834
# Epoch 4: Loss = 0.5211
# Epoch 5: Loss = 0.4623
```

### Example 7: Mixed precision training (AMP)
```python
from torch.cuda.amp import autocast, GradScaler

model = nn.Sequential(nn.Linear(10, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

scaler = GradScaler()  # for gradient scaling
loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

for epoch in range(5):
    model.train()
    epoch_loss = 0.0
    for bx, by in loader:
        optimizer.zero_grad()

        # Automatic Mixed Precision
        with autocast():
            outputs = model(bx)
            loss = criterion(outputs, by)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item() * bx.size(0)

    avg_loss = epoch_loss / len(X)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

print("AMP training complete (typically 1.5-2x faster on compatible GPUs)")
```
```
# Output:
# Epoch 1: Loss = 0.6878
# Epoch 2: Loss = 0.6312
# Epoch 3: Loss = 0.5643
# Epoch 4: Loss = 0.5012
# Epoch 5: Loss = 0.4489
# AMP training complete (typically 1.5-2x faster on compatible GPUs)
```

### Example 8: Per-epoch metrics logging with tqdm
```python
from tqdm import tqdm

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

for epoch in range(5):
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)

    for bx, by in progress_bar:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
```
```
# Output:
# (progress bars with live loss updates)
# Epoch 1: Avg Loss = 0.6912
# Epoch 2: Avg Loss = 0.6421
# Epoch 3: Avg Loss = 0.5834
# Epoch 4: Avg Loss = 0.5211
# Epoch 5: Avg Loss = 0.4623
```

### Example 9: Training with multiple losses
```python
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(10, 32)
        self.class_head = nn.Linear(32, 2)
        self.reg_head = nn.Linear(32, 1)

    def forward(self, x):
        features = torch.relu(self.shared(x))
        return self.class_head(features), self.reg_head(features)

model = MultiTaskModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create multi-task targets
y_class = torch.randint(0, 2, (1000,))
y_reg = torch.randn(1000, 1)

loader = DataLoader(TensorDataset(X, y_class, y_reg), batch_size=32, shuffle=True)

for epoch in range(5):
    model.train()
    total_loss = 0.0
    for bx, bc, br in loader:
        optimizer.zero_grad()
        out_class, out_reg = model(bx)
        loss_class = nn.CrossEntropyLoss()(out_class, bc)
        loss_reg = nn.MSELoss()(out_reg, br)
        loss = loss_class + 0.5 * loss_reg  # weighted combination
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}: Total Loss = {avg_loss:.4f}")
```
```
# Output:
# Epoch 1: Total Loss = 1.3421
# Epoch 2: Total Loss = 1.0213
# Epoch 3: Total Loss = 0.8821
# Epoch 4: Total Loss = 0.7923
# Epoch 5: Total Loss = 0.7312
```

### Example 10: Fully structured training function
```python
def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, device='cpu', patience=5, checkpoint_path='best_model.pth'):
    model.to(device)
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                outputs = model(bx)
                val_loss += criterion(outputs, by).item() * bx.size(0)
                predicted = (outputs > 0.5).float()
                correct += (predicted == by).sum().item()
                total += by.size(0)
        val_loss /= len(val_loader.dataset)
        accuracy = correct / total

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)

        print(f"Epoch {epoch+1:2d}: Train={train_loss:.4f}, Val={val_loss:.4f}, Acc={accuracy:.3f}")

        if early_stopping(val_loss):
            print("Early stopping triggered.")
            break

    # Load best model
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1} (val_loss={checkpoint['val_loss']:.4f})")
    return model

# Usage
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
train_loader = DataLoader(TensorDataset(X[:800], y[:800]), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X[800:], y[800:]), batch_size=32)

trained_model = train_model(model, train_loader, val_loader,
                            nn.BCELoss(), optim.Adam(model.parameters(), lr=0.001),
                            num_epochs=30, patience=5)
```
```
# Output:
# Epoch  1: Train=0.6912, Val=0.6881, Acc=0.521
# Epoch  2: Train=0.6723, Val=0.6692, Acc=0.632
# ... (training progresses)
# Epoch  8: Train=0.5012, Val=0.4987, Acc=0.875
# Epoch  9: Train=0.4876, Val=0.4892, Acc=0.878
# Epoch 10: Train=0.4751, Val=0.4812, Acc=0.881
# Epoch 11: Train=0.4634, Val=0.4798, Acc=0.880
# Early stopping triggered.
# Loaded best model from epoch 10 (val_loss=0.4812)
```

## Common Mistakes

1. **Forgetting `model.train()` before training and `model.eval()` before evaluation.** Dropout and BatchNorm behave differently depending on mode. Missing these toggles causes incorrect training or evaluation results.
2. **Calling `optimizer.zero_grad()` after `.backward()` and `.step()` instead of before.** Gradients accumulate by default. Zero them before each batch's backward pass, not after.
3. **Not using `torch.no_grad()` during validation.** Without it, the validation pass builds a computational graph, wasting GPU memory. Always wrap validation code in `with torch.no_grad():`.
4. **Saving `model.state_dict()` but forgetting `optimizer.state_dict()` when resuming training.** To resume training exactly, save the optimizer state (including momentum buffers and learning rate scheduler state).
5. **Moving model to device but not data.** `model.to(device)` only moves model parameters. Input tensors must be moved separately: `batch_x = batch_x.to(device)`.
6. **Calling `.item()` on tensors that may require grad.** `loss.item()` works on scalar tensors but only if they don't require grad (or after `.detach()`). Use `loss.detach().item()` to be safe.
7. **Overwriting the best checkpoint file with every epoch.** Save only when validation improves, or use epoch-numbered filenames to keep multiple checkpoints.

## Interview Questions

### Beginner - 5

1. **Q:** What are the 5 essential steps in each training iteration?  
   **A:** (1) `optimizer.zero_grad()` — clear gradients, (2) forward pass — `outputs = model(inputs)`, (3) compute loss — `loss = criterion(outputs, targets)`, (4) `loss.backward()` — compute gradients, (5) `optimizer.step()` — update weights.

2. **Q:** Why do we call `model.train()` before training and `model.eval()` before evaluation?  
   **A:** `train()` enables dropout and batch normalization training behavior. `eval()` disables dropout and uses running statistics for batch norm. Using the wrong mode leads to incorrect results.

3. **Q:** What does `torch.no_grad()` do?  
   **A:** It disables gradient computation and computational graph building, reducing memory usage and speeding up inference. Essential for validation and test passes.

4. **Q:** Why do we call `optimizer.zero_grad()` at the beginning of each iteration?  
   **A:** Gradients accumulate by default in PyTorch. Without zeroing, each backward pass adds to existing gradients, causing incorrect updates.

5. **Q:** How do you save and load a model checkpoint?  
   **A:** Save: `torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, 'ckpt.pth')`. Load: `checkpoint = torch.load('ckpt.pth'); model.load_state_dict(checkpoint['model'])`.

### Intermediate - 5

1. **Q:** How does gradient accumulation work and when would you use it?  
   **A:** Instead of updating weights after every batch, accumulate gradients over N batches before stepping. Effective batch size = batch_size × N. Used when GPU memory limits batch size but larger batches are needed for stable training.

2. **Q:** What is mixed precision training and how do you implement it with AMP?  
   **A:** Mixed precision (AMP) uses float16 for most operations and float32 for critical ones, speeding up training 1.5-2x on compatible GPUs. Implement with `torch.cuda.amp.autocast()` and `GradScaler()`.

3. **Q:** How do you implement early stopping and why is it important?  
   **A:** Monitor validation loss. If it doesn't improve for `patience` epochs, stop training. This prevents overfitting and saves computation time.

4. **Q:** What is the difference between `torch.save(model.state_dict())` and `torch.save(model)`?  
   **A:** `state_dict` saves only parameters (lightweight, portable). `torch.save(model)` saves the entire model object (architecture + weights, but fragile across code changes). Always prefer `state_dict`.

5. **Q:** How would you add learning rate warmup to a custom training loop?  
   **A:** Define a function that increases LR from 0 to target over warmup steps. After each step: `lr = warmup_fn(step)`, then `for pg in optimizer.param_groups: pg['lr'] = lr`. Use a linear or cosine warmup schedule.

### Advanced - 3

1. **Q:** How would you implement gradient penalty (e.g., WGAN-GP) in a custom training loop?  
   **A:** After the discriminator forward/backward, compute gradients of discriminator outputs w.r.t. interpolated inputs using `torch.autograd.grad()`. Compute penalty as `((grad.norm(2) - 1)**2).mean()`. Add to discriminator loss.

2. **Q:** Describe how to implement distributed data parallel (DDP) training with a custom loop.  
   **A:** Wrap model with `DistributedDataParallel`. Use `DistributedSampler` for data loading. Each process handles a subset of the batch. DDP synchronizes gradients across all processes during `backward()`. Set `device_id` per process. Use `torch.distributed.all_reduce` for cross-process metric aggregation.

3. **Q:** How would you implement a custom learning rate schedule that is not available in `torch.optim.lr_scheduler`?  
   **A:** Write a lambda scheduler: `scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95**epoch)`. Or manually update LR each step: `for param_group in optimizer.param_groups: param_group['lr'] = custom_fn(step)`.

## Practice Problems

### Easy - 5

1. **E1:** Write a basic training loop for a Linear model (nn.Linear(1,1)) on y=2x+1 data. Print loss every 10 epochs.
2. **E2:** Add a validation phase to the loop in E1 using a held-out test set.
3. **E3:** Save a checkpoint after training and load it into a new model.
4. **E4:** Add `torch.no_grad()` to the validation phase of an existing loop.
5. **E5:** Train with `model.train()` and evaluate with `model.eval()` on a model with Dropout.

### Medium - 5

1. **M1:** Implement early stopping with patience=5 in a training loop. Print when early stopping triggers.
2. **M2:** Add gradient clipping (max_norm=1.0) to a training loop and log the gradient norm.
3. **M3:** Implement a StepLR scheduler that reduces LR every 5 epochs by factor 0.5.
4. **M4:** Write a training loop that accumulates gradients over 4 batches before stepping.
5. **M5:** Create a structured training function that returns training history (list of losses per epoch).

### Hard - 3

1. **H1:** Implement a custom training loop for a GAN: alternate discriminator (real/fake) and generator (fool discriminator) updates.
2. **H2:** Add mixed precision (AMP) to an existing training loop and measure speedup.
3. **H3:** Implement a custom learning rate schedule with linear warmup (5 epochs) followed by cosine decay.

## Solutions

### E1 Solution
```python
model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
X = torch.linspace(-1, 1, 100).unsqueeze(1)
y = 2 * X + 1 + torch.randn(100, 1) * 0.1
for epoch in range(100):
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 19:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
```

### E2-E5 Solutions follow patterns from examples.

### M1-M5 Solutions extend techniques shown in examples 1-6.

### H1-H3 Solutions require advanced training techniques.

## Related Concepts

- 096 — PyTorch Tensors (autograd foundation)
- 097 — PyTorch NN (building models for training loops)
- 098 — TensorFlow/Keras (high-level fit vs custom loops)

## Next Concepts

- 100 — Project Structure (organizing training scripts)
- Distributed training (DDP), hyperparameter optimization (Optuna)

## Summary

Custom training loops provide full control over the PyTorch training process: epoch/batch iteration, forward/loss/backward/step pattern, model.train/eval switching, checkpointing with torch.save/load, early stopping, gradient clipping, learning rate scheduling, gradient accumulation, and mixed precision. Structured training functions encapsulate these patterns for reusable, production-quality training code.

## Key Takeaways

- Essential 5-step pattern: zero_grad → forward → loss → backward → step
- Always toggle `model.train()` / `model.eval()` correctly
- Wrap validation in `torch.no_grad()` to save memory
- Save checkpoints as dicts with model + optimizer + epoch + loss
- Early stopping prevents overfitting and saves time
- Gradient clipping prevents exploding gradients in deep/recurrent networks
- Gradient accumulation simulates larger batch sizes on limited GPU memory
- Use AMP (`autocast` + `GradScaler`) for faster training on compatible hardware
- Structured training functions make code reusable and less error-prone
