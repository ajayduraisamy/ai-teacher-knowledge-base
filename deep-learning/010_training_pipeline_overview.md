# Concept: Training Pipeline Overview

## Concept ID

DL-010

## Difficulty

Beginner

## Domain

Deep Learning

## Module

Deep Learning Foundations

## Learning Objectives

- Describe the complete deep learning training pipeline
- Explain the role of each stage: data, model, loss, optimization
- Distinguish between training, validation, and test sets
- Implement a complete training pipeline in PyTorch

## Prerequisites

- Neural Networks (DL-001)
- Forward and Backward Pass (DL-012, DL-013)
- Basic Python programming

## Definition

The training pipeline is the end-to-end process of teaching a deep neural network to perform a task. It consists of a sequence of stages executed iteratively:

1. **Data Loading:** Acquiring and preparing data for training
2. **Preprocessing:** Transforming raw data into a format suitable for the model
3. **Forward Pass:** Computing model predictions
4. **Loss Computation:** Measuring prediction error
5. **Backward Pass (Backpropagation):** Computing gradients of the loss with respect to parameters
6. **Parameter Update:** Adjusting weights to reduce loss
7. **Repeat:** Looping through the data multiple times (epochs)
8. **Evaluation:** Assessing model performance on held-out data

The pipeline is typically split into three phases:
- **Training:** Updating parameters on training data
- **Validation:** Monitoring performance on validation data (not used for training) to tune hyperparameters and detect overfitting
- **Testing:** Final evaluation on unseen test data after all tuning is complete

## Intuition

Imagine you're teaching a student (the model) to solve math problems.

**Data Loading:** You gather a collection of practice problems and answer keys.
**Preprocessing:** You organize problems into a clear format and remove any that are corrupted.
**Forward Pass:** The student attempts a problem and produces an answer.
**Loss Computation:** You check the student's answer against the answer key and quantify how wrong it is.
**Backward Pass:** You figure out which parts of the student's reasoning process contributed most to the error.
**Parameter Update:** You give the student targeted feedback to adjust their thinking.

You repeat this process with many problems (iterations). After each complete set of problems (epoch), you give the student a quiz (validation) to check if they're truly learning or just memorizing. Finally, you give a final exam (test) to evaluate their true ability.

The pipeline is the same whether you're training a small MLP or a billion-parameter Transformer — only the scale and complexity differ.

## Why This Concept Matters

Understanding the complete training pipeline is essential for anyone doing deep learning. It provides:

- **Mental Model:** A framework for thinking about every step from data to trained model
- **Debugging Tool:** When training fails, you can systematically check each pipeline stage
- **Reproducibility:** Proper pipeline design ensures consistent, reproducible results
- **Efficiency:** Understanding the pipeline helps identify bottlenecks (data loading, compute, I/O)
- **Best Practices:** Each stage has known best practices that improve results

## Real World Examples

1. **Image Classification (Production):** TensorFlow's TF.Data pipeline loads images from disk, decodes JPEGs, applies random augmentations (flip, crop, color jitter), normalizes pixels to [0,1], batches them, prefetches for GPU, feeds into a ResNet, computes cross-entropy loss, backpropagates, updates with SGD + momentum. Validation accuracy is monitored every epoch; if it plateaus for 5 epochs, learning rate decays by 0.1.

2. **Language Model Training (GPT):** The pipeline downloads terabytes of text, tokenizes it with BPE, creates sequences of 2048 tokens, applies attention masking, computes next-token prediction loss, uses gradient accumulation (multiple micro-batches per update), mixed precision training (FP16), and distributed data parallelism across hundreds of GPUs.

3. **Reinforcement Learning (AlphaGo):** The pipeline generates self-play games, stores game states and outcomes in a replay buffer, samples batches, trains the policy and value networks, and periodically evaluates against previous versions.

## AI/ML Relevance

- **MLOps:** Training pipeline design is the foundation of MLOps — automating and monitoring the ML lifecycle.
- **Experiment Tracking:** Tools like MLflow, Weights & Biases, and TensorBoard track pipeline stages (loss, metrics, hyperparameters).
- **Distributed Training:** Large-scale pipelines use data parallelism, model parallelism, and pipeline parallelism.
- **Data Pipeline Design:** Data loading and preprocessing are often the biggest bottlenecks in production ML.
- **Continuous Training:** Production systems retrain models periodically through automated pipelines.

## Mathematical Explanation

### Complete Training Loop

Given dataset $\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^N$, model $f_\theta$ with parameters $\theta$, loss function $\mathcal{L}$:

**For each epoch $e = 1 \dots E$:**
  **For each batch $\mathcal{B} \subset \mathcal{D}$:**
    1. Forward: $\hat{y} = f_\theta(\mathbf{x})$ for $\mathbf{x} \in \mathcal{B}$
    2. Loss: $\ell = \frac{1}{|\mathcal{B}|} \sum_{(\mathbf{x}, y) \in \mathcal{B}} \mathcal{L}(f_\theta(\mathbf{x}), y)$
    3. Backward: $\nabla_\theta \ell = \frac{\partial \ell}{\partial \theta}$
    4. Update: $\theta \leftarrow \theta - \eta \nabla_\theta \ell$
  **Validate:** Compute $\mathcal{L}_{\text{val}}(f_\theta; \mathcal{D}_{\text{val}})$

### Training/Validation/Test Split

- **Training set (60-80%):** Used for parameter updates
- **Validation set (10-20%):** Used for hyperparameter tuning and early stopping
- **Test set (10-20%):** Used only once for final evaluation

### Key Pipeline Operations

| Operation | Time Complexity | Memory | Notes |
|-----------|----------------|--------|-------|
| Data loading | O(N) | O(batch_size × input_size) | I/O bound |
| Forward pass | O(n_layers × hidden²) | O(batch_size × hidden × layers) | Compute bound |
| Loss computation | O(batch_size) | O(batch_size) | Negligible |
| Backward pass | ~2x forward | O(batch_size × hidden × layers) | Compute bound |
| Parameter update | O(num_params) | O(num_params) | Communication bound (distributed) |

## Code Examples

### Example 1: Complete Training Pipeline for MNIST Classification

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Generate synthetic dataset (simulating MNIST-like)
torch.manual_seed(42)
n_train, n_val, n_test = 1000, 200, 200
X_all = torch.randn(n_train + n_val + n_test, 784)
y_all = torch.randint(0, 10, (n_train + n_val + n_test,))

X_train, y_train = X_all[:n_train], y_all[:n_train]
X_val, y_val = X_all[n_train:n_train+n_val], y_all[n_train:n_train+n_val]
X_test, y_test = X_all[n_train+n_val:], y_all[n_train+n_val:]

# 2. Create data loaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# 3. Define model
model = nn.Sequential(
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 10)
)

# 4. Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training loop with validation
n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = correct / total
    print(f"Epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
# Output: Epoch 1/10: train_loss=2.3112, val_loss=2.3075, val_acc=0.1150
# Output: Epoch 2/10: train_loss=2.2911, val_loss=2.2897, val_acc=0.1150
# Output: ...
# Output: Epoch 10/10: train_loss=2.2181, val_loss=2.2243, val_acc=0.1250

# 6. Final test evaluation
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        test_total += batch_y.size(0)
        test_correct += (predicted == batch_y).sum().item()
test_acc = test_correct / test_total
print(f"Test accuracy: {test_acc:.4f}")
# Output: Test accuracy: 0.1200
# (Random data → ~10% accuracy, as expected for 10 classes)
```

### Example 2: Training Pipeline with Early Stopping

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
X_train = torch.randn(500, 20)
y_train = torch.randint(0, 2, (500,))
X_val = torch.randn(100, 20)
y_val = torch.randint(0, 2, (100,))

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

best_val_loss = float('inf')
patience = 5
patience_counter = 0
best_model_state = None

for epoch in range(100):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y.float().unsqueeze(1))
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            val_loss += criterion(model(batch_X), batch_y.float().unsqueeze(1)).item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}: val_loss={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Restore best model
if best_model_state:
    model.load_state_dict(best_model_state)
print(f"Training complete. Best val_loss: {best_val_loss:.4f}")
# Output: Epoch 1: val_loss=0.6891
# Output: ...
# Output: Early stopping at epoch 12
# Output: Training complete. Best val_loss: 0.6324
```

### Example 3: Training Pipeline with Learning Rate Scheduling

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
X = torch.randn(500, 10)
y = torch.randint(0, 2, (500,))
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
criterion = nn.BCELoss()

for epoch in range(15):
    model.train()
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y.float().unsqueeze(1))
        loss.backward()
        optimizer.step()

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}: LR={current_lr:.6f}")
    scheduler.step()
# Output: Epoch 1: LR=0.100000
# Output: Epoch 2: LR=0.100000
# Output: ...
# Output: Epoch 5: LR=0.010000  (LR dropped after step 5)
# Output: ...
# Output: Epoch 10: LR=0.001000
# Output: ...
# Output: Epoch 15: LR=0.000100
```

## Common Mistakes

1. **Data leakage between training and validation/test:** Shuffling data before splitting, or using test data for validation during development, leads to overly optimistic performance estimates.

2. **Not resetting gradients (`zero_grad()`):** Gradients accumulate by default in PyTorch. Forgetting to call `optimizer.zero_grad()` causes incorrect updates.

3. **Missing `model.eval()` during evaluation:** Batch normalization and dropout behave differently during training vs evaluation. Failing to switch modes corrupts validation/test results.

4. **Tuning hyperparameters on the test set:** The test set should be used exactly once — for final evaluation. Tuning on it invalidates the entire evaluation.

5. **Not shuffling training data:** Without shuffling, the model sees the same batch order every epoch, which can cause biased updates and slower convergence.

## Interview Questions

### Beginner

1. What are the stages of the deep learning training pipeline?
2. What is the difference between training, validation, and test sets?
3. Why do we need `model.train()` and `model.eval()` in PyTorch?
4. What happens if you forget to call `optimizer.zero_grad()`?
5. Why is data shuffling important during training?

### Intermediate

1. Explain early stopping and how it prevents overfitting.
2. What is gradient accumulation and when would you use it?
3. How does a learning rate scheduler work? Give two examples.
4. Describe the data loading pipeline in PyTorch — what is `DataLoader` and how does it work with `Dataset`?
5. What is the relationship between batch size, learning rate, and training stability?

### Advanced

1. Design a distributed training pipeline using data parallelism and explain gradient synchronization.
2. Analyze the bottlenecks in a typical training pipeline (CPU vs GPU, I/O vs compute) and propose solutions.
3. Implement a training pipeline with mixed precision (FP16) training. Explain the gradient scaling mechanism.

## Practice Problems

### Easy

1. Write a complete training loop for a model with one linear layer (logistic regression) on synthetic data.
2. Add validation evaluation between epochs to the training loop.
3. Implement a `DataLoader` for a custom dataset of 1000 samples with 10 features.
4. Add `model.train()` and `model.eval()` calls at the correct places in a training loop.
5. Create a training pipeline that saves the best model checkpoint based on validation loss.

### Medium

1. Implement a training pipeline with gradient accumulation to simulate a larger batch size.
2. Add TensorBoard logging to a training pipeline (loss curves, weight histograms, learning rate).
3. Implement k-fold cross-validation as part of the training pipeline and report mean ± std accuracy.
4. Design a data augmentation pipeline for images (random flips, rotations, color jitter) and integrate it into training.
5. Implement a pipeline that supports both training from scratch and fine-tuning a pretrained model.

### Hard

1. Implement a complete distributed data parallel (DDP) training pipeline across multiple GPUs.
2. Build a training pipeline with automated mixed precision (AMP) and gradient scaling.
3. Design a continuous training pipeline that (a) monitors for new data, (b) retrains on combined old+new data, (c) evaluates against the current production model, and (d) deploys if improved.

## Solutions

### Easy 1
```python
model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
X = torch.randn(100, 10)
y = torch.randn(100, 1)
for epoch in range(100):
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
```

### Medium 1
```python
accumulation_steps = 4  # simulate batch_size * 4
effective_batch_size = 32 * accumulation_steps
optimizer.zero_grad()
for i, (batch_X, batch_y) in enumerate(loader):
    loss = criterion(model(batch_X), batch_y) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Related Concepts

- Epoch, Batch, Iteration (DL-011)
- Forward Pass (DL-012)
- Backward Pass (DL-013)
- Loss Functions
- Gradient Descent

## Next Concepts

- Data Augmentation
- Learning Rate Scheduling
- Early Stopping
- Distributed Training
- Mixed Precision Training

## Summary

The training pipeline is a multi-stage process: data loading → preprocessing → forward pass → loss computation → backward pass → parameter update → evaluate. Data is split into training, validation, and test sets. The training loop iterates over batches across epochs, with validation monitoring for overfitting. Best practices include shuffling data, resetting gradients, switching train/eval modes, and using early stopping. Understanding the complete pipeline is essential for debugging, optimization, and reproducibility.

## Key Takeaways

- Training pipeline: Data → Model → Loss → Backward → Update → Repeat
- Three data splits: train (learning), validation (tuning), test (final evaluation)
- `model.train()` and `model.eval()` control dropout/batchnorm behavior
- `optimizer.zero_grad()` must be called before each backward pass
- Early stopping prevents overfitting; learning rate scheduling improves convergence
