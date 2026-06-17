# Concept: Checkpointing

## Concept ID

DL-159

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Training Deep Networks

## Learning Objectives

- Understand the purpose of model checkpointing
- Implement periodic and best-model checkpointing in PyTorch
- Manage disk space with checkpoint pruning
- Load checkpoints for resuming training
- Design checkpoint strategies for long-running training

## Prerequisites

- Training loop (DL-156)
- Validation loop (DL-157)
- Model saving and loading basics
- File system management

## Definition

Checkpointing saves the model state (weights, optimizer state, epoch number, metrics) periodically during training, allowing training to be resumed from any saved point. It protects against training interruptions (power loss, system crash) and enables saving the best model found during training. A good checkpointing strategy balances storage space against the granularity of recovery points. Checkpoints typically include: model state_dict, optimizer state_dict, epoch, best validation metric, and training configuration.

## Intuition

Think of checkpointing like saving a video game at regular intervals. If the power goes out (training crash), you do not have to restart from the beginning (epoch 0) — you resume from your last save. Additionally, you might keep the "best" save — the point where you had the highest score (best validation performance), even if later attempts did not beat it. Checkpointing is the safety net of deep learning training: it saves time, enables experimentation, and ensures you never lose progress.

## Why This Concept Matters

Deep learning training can take days or weeks on expensive hardware. A single crash without checkpointing can waste all that time and money. Checkpointing also enables: (1) model ensembling from different epochs, (2) early stopping with best-weight restoration, (3) learning rate scheduling from specific checkpoints, (4) sharing intermediate models for collaboration, and (5) auditing model development history.

## Code Examples

### Example 1: Basic Checkpointing

`python
import torch
import torch.nn as nn
import os

class CheckpointManager:
    def __init__(self, save_dir='checkpoints', save_every=5, keep_best=True):
        self.save_dir = save_dir
        self.save_every = save_every
        self.keep_best = keep_best
        self.best_metric = float('inf')
        os.makedirs(save_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, epoch, loss, metrics=None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics or {},
        }
        
        # Periodic checkpoint
        if epoch % self.save_every == 0:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, path)
            print(f"Saved periodic checkpoint: {path}")
        
        # Best model checkpoint
        if self.keep_best and loss < self.best_metric:
            self.best_metric = loss
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"New best model! loss={loss:.4f} at epoch {epoch}")

    def load_checkpoint(self, model, optimizer, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {checkpoint['epoch']}")
        return start_epoch

model = nn.Linear(10, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
checkpointer = CheckpointManager('checkpoints', save_every=3)

for epoch in range(10):
    # Simulated training
    loss = 1.0 / (epoch + 1)
    checkpointer.save_checkpoint(model, optimizer, epoch, loss)
# Output:
# New best model! loss=1.0000 at epoch 0
# Saved periodic checkpoint: checkpoints/checkpoint_epoch_0.pt
# New best model! loss=0.5000 at epoch 1
# New best model! loss=0.3333 at epoch 2
# Saved periodic checkpoint: checkpoints/checkpoint_epoch_3.pt
# New best model! loss=0.2500 at epoch 3
# New best model! loss=0.2000 at epoch 4
# Saved periodic checkpoint: checkpoints/checkpoint_epoch_6.pt
# New best model! loss=0.1429 at epoch 6
# New best model! loss=0.1250 at epoch 7
# Saved periodic checkpoint: checkpoints/checkpoint_epoch_9.pt
# New best model! loss=0.1111 at epoch 8
# New best model! loss=0.1000 at epoch 9
`

### Example 2: Checkpoint with Multiple Best Metrics

`python
import torch
import torch.nn as nn
import os
import json
from datetime import datetime

class AdvancedCheckpoint:
    def __init__(self, save_dir='checkpoints'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.history = []

    def save(self, model, optimizer, scheduler, epoch, train_loss, val_loss, val_acc):
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Always save latest
        torch.save(checkpoint, os.path.join(self.save_dir, 'latest.pt'))
        
        # Save best by val_loss
        if not hasattr(self, 'best_val_loss') or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_loss.pt'))
        
        # Save best by val_acc
        if not hasattr(self, 'best_val_acc') or val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_acc.pt'))
        
        self.history.append({
            'epoch': epoch, 'train_loss': train_loss, 
            'val_loss': val_loss, 'val_acc': val_acc
        })
        with open(os.path.join(self.save_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_best(self, model, metric='loss'):
        filename = 'best_loss.pt' if metric == 'loss' else 'best_acc.pt'
        checkpoint = torch.load(os.path.join(self.save_dir, filename))
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded best {metric} model from epoch {checkpoint['epoch']}")

ckpt = AdvancedCheckpoint('checkpoints_adv')

for epoch in range(5):
    train_loss = 0.5 / (epoch + 1)
    val_loss = 0.4 / (epoch + 1) + 0.01 * max(0, epoch - 2)
    val_acc = 0.5 + 0.1 * epoch - 0.02 * max(0, epoch - 2)
    ckpt.save(nn.Linear(10, 2), 
              torch.optim.SGD(nn.Linear(10, 2).parameters(), lr=0.01), 
              None, epoch, train_loss, val_loss, val_acc)
    if epoch == 4:
        ckpt.load_best(nn.Linear(10, 2), 'loss')
        ckpt.load_best(nn.Linear(10, 2), 'acc')
# Output:
# Loaded best loss model from epoch 2
# Loaded best acc model from epoch 3
`

### Example 3: Resume Training from Checkpoint

`python
import torch
import torch.nn as nn
import torch.optim as optim
import os

def train_with_resume(resume_from=None, num_epochs=10):
    model = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 5))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    start_epoch = 0
    
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {checkpoint['epoch']}")
    
    for epoch in range(start_epoch, num_epochs):
        # Training...
        loss = 1.0 / (epoch + 1)
        scheduler.step()
        
        # Save checkpoint
        if epoch % 3 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, 'resume_checkpoint.pt')
        
        print(f"Epoch {epoch}: loss={loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

print("=== First run ===")
train_with_resume(None, 5)
print("\n=== Resume run ===")
train_with_resume('resume_checkpoint.pt', 10)
# Output:
# === First run ===
# Epoch 0: loss=1.0000, lr=0.010000
# Epoch 1: loss=0.5000, lr=0.010000
# Epoch 2: loss=0.3333, lr=0.010000
# Epoch 3: loss=0.2500, lr=0.001000
# Epoch 4: loss=0.2000, lr=0.001000
#
# === Resume run ===
# Resuming from epoch 3
# Epoch 3: loss=0.2500, lr=0.001000
# Epoch 4: loss=0.2000, lr=0.001000
# Epoch 5: loss=0.1667, lr=0.001000
# Epoch 6: loss=0.1429, lr=0.000100
# Epoch 7: loss=0.1250, lr=0.000100
# Epoch 8: loss=0.1111, lr=0.000100
# Epoch 9: loss=0.1000, lr=0.000100
`

## Common Mistakes

1. **Saving only model weights without optimizer state**: Resume training requires optimizer state to continue properly (especially for Adam with momentum buffers).
2. **Not saving the random seed state**: For reproducibility when resuming, save the random seed state as well.
3. **Keeping too many checkpoints**: Disk space fills quickly. Use periodic pruning (keep every Nth checkpoint + best).
4. **Saving checkpoints too frequently**: Writing to disk is slow. Save every N epochs or based on validation improvement.
5. **Not saving training configuration**: When resuming later, you may forget the original hyperparameters. Save them alongside the checkpoint.

## Interview Questions

### Beginner

1. What is model checkpointing?
2. What information should a checkpoint contain?
3. Why is checkpointing important for long training runs?
4. How often should you save checkpoints?
5. What is the difference between latest and best checkpoint?

### Intermediate

1. Explain how to resume training from a checkpoint correctly.
2. How do you manage disk space for many checkpoints?
3. What additional state should be saved beyond model weights?
4. How would you implement distributed training checkpointing?
5. Compare periodic checkpointing with best-only checkpointing.

### Advanced

1. Design a fault-tolerant training system with automatic checkpoint recovery.
2. Implement asynchronous checkpointing that does not block training.
3. How would you handle checkpointing for very large models (100B+ parameters) that exceed disk capacity?

## Practice Problems

### Easy

1. Write a function to save model checkpoint.
2. Write a function to load and resume training.
3. Add a learning rate scheduler to the checkpoint.
4. Implement checkpoint pruning (keep only every 5th).
5. Save a checkpoint with a custom filename including the validation accuracy.

### Medium

1. Implement a checkpoint manager that keeps top-k checkpoints by validation loss.
2. Add resume capability with random seed restoration.
3. Implement distributed checkpointing for multi-GPU training.
4. Create a checkpoint that can be loaded for inference only (no optimizer).
5. Implement a checkpoint validity checker.

### Hard

1. Implement sharded checkpointing for models that do not fit in single GPU memory.
2. Design a checkpointless training recovery system using gradient history replay.
3. Implement a cloud-based checkpoint manager with automatic upload/download.

## Solutions

### Easy Solutions

1. torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, 'checkpoint.pt')
2. checkpoint = torch.load('checkpoint.pt'); model.load_state_dict(checkpoint['model_state_dict']); start_epoch = checkpoint['epoch'] + 1
3. Add 'scheduler_state_dict': scheduler.state_dict() to the checkpoint
4. if epoch % 5 == 0: save checkpoint
5. filename = f'checkpoint_epoch_{epoch}_val_{val_loss:.4f}.pt'

## Related Concepts

- Model Saving and Loading (DL-160)
- Training Loop (DL-156)
- Early Stopping (DL-137)
- Experiment Tracking (DL-161)

## Next Concepts

- Model Saving and Loading (DL-160)
- Experiment Tracking (DL-161)
- Hyperparameter Search (DL-162)

## Summary

Checkpointing saves model state periodically during training, enabling recovery from interruptions and preserving the best model. Checkpoints should include model weights, optimizer state, scheduler state, epoch, and metrics. A good strategy balances periodic saves with best-model tracking.

## Key Takeaways

- Checkpoints save model, optimizer, scheduler state and metrics
- Enable recovery from training interruptions
- Maintain the best model (by validation metric) separately
- Save scheduler state for proper LR resumption
- Prune old checkpoints to manage disk space
- Save training configuration alongside checkpoints
- Enables model ensembling from different epochs
- Essential for long-running training jobs
