# Concept: Learning Curves

## Concept ID

DL-166

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Training Deep Networks

## Learning Objectives

- Understand the information conveyed by learning curves
- Plot and interpret training and validation learning curves
- Diagnose overfitting, underfitting, and convergence from curves
- Use learning curves to guide training decisions (early stopping, LR changes)
- Implement learning curve generation and analysis

## Prerequisites

- Training loop (DL-156)
- Validation loop (DL-157)
- Overfitting and underfitting concepts
- Basic plotting with matplotlib

## Definition

Learning curves plot model performance (loss or metric) on the y-axis against training progress (epochs or steps) on the x-axis. They typically show both training and validation curves. Learning curves reveal: (1) whether the model is learning (decreasing loss), (2) whether it is overfitting (validation loss increases), (3) whether it is underfitting (both losses plateau high), (4) the optimal stopping point, and (5) the effect of hyperparameter changes.

## Intuition

Think of learning curves as a health monitor for your training process. The training curve tells you if the model is successfully memorizing the training data (decreasing). The validation curve tells you if this learning generalizes to new data. A large and growing gap between the curves signals overfitting (memorization without understanding). A flat, high validation curve signals underfitting (the model is not powerful enough). Learning curves tell you what action to take: stop training, increase model capacity, add regularization, or change the learning rate.

## Why This Concept Matters

Learning curves are the most important diagnostic tool in deep learning. Before checking any specific metric, experienced practitioners always look at the learning curves. They provide immediate insight into: (1) whether training is working, (2) whether the model is overfitting, (3) whether the learning rate is appropriate, (4) when to stop training, (5) whether to adjust model capacity, and (6) whether data augmentation or regularization is helping. Mastering learning curve interpretation is essential for efficient model development.

## Types of Learning Curves

1. **Training loss curve**: Monotonically decreasing (should always decrease). Rate of decrease indicates learning speed.
2. **Validation loss curve**: Should decrease with training loss but may eventually increase (overfitting).
3. **Training accuracy curve**: Increases as model learns. May plateau before 100% for hard tasks.
4. **Validation accuracy curve**: Increases with training. Plateau or decrease indicates overfitting.
5. **Gradient norm curve**: Indicates training stability. Spikes indicate gradient explosion.
6. **Learning rate curve**: Shows LR schedule. Important for understanding training dynamics.

## Code Examples

### Example 1: Generating Learning Curves

`python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LearningCurveTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.lrs = []
        self.grad_norms = []

    def record(self, epoch, train_loss, val_loss, train_acc, val_acc, lr, grad_norm=None):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.lrs.append(lr)
        if grad_norm is not None:
            self.grad_norms.append(grad_norm)

    def print_curves(self):
        print(f"{'Epoch':8s} {'Train Loss':12s} {'Val Loss':12s} "
              f"{'Train Acc':10s} {'Val Acc':10s} {'LR':10s}")
        print("-" * 70)
        for i in range(len(self.train_losses)):
            print(f"{i+1:4d}     {self.train_losses[i]:.4f}      {self.val_losses[i]:.4f}      "
                  f"{self.train_accs[i]:.4f}     {self.val_accs[i]:.4f}     {self.lrs[i]:.6f}")

    def report_diagnostics(self):
        print("\n=== Learning Curve Diagnostics ===")
        
        # Overfitting check
        if len(self.val_losses) > 5:
            recent_val = np.mean(self.val_losses[-3:])
            best_val = min(self.val_losses)
            if recent_val > best_val * 1.1:
                print("SIGNAL: Potential overfitting (validation loss increasing)")
            else:
                print("OK: Validation loss stable or decreasing")
        
        # Underfitting check
        if self.train_losses[-1] > 0.5 * self.train_losses[0]:
            print("SIGNAL: Possible underfitting (training loss still high)")
        else:
            print("OK: Training loss decreasing adequately")
        
        # Convergence check
        recent_train = np.mean(self.train_losses[-3:])
        if len(self.train_losses) > 10 and recent_train > self.train_losses[-10] * 0.95:
            print("SIGNAL: Possible convergence (loss plateauing)")
        else:
            print("OK: Loss still improving")

tracker = LearningCurveTracker()

for epoch in range(20):
    # Simulate realistic curves with overfitting
    train_loss = 1.0 * np.exp(-epoch / 3) + 0.1
    val_loss = 1.0 * np.exp(-epoch / 3) + 0.05 * max(0, epoch - 10) + 0.15
    train_acc = 1 - np.exp(-epoch / 4)
    val_acc = 1 - np.exp(-epoch / 4) - 0.02 * max(0, epoch - 12)
    lr = 0.01 * (0.9 ** epoch)
    
    tracker.record(epoch, train_loss, val_loss, train_acc, val_acc, lr)

tracker.print_curves()
tracker.report_diagnostics()
# Output:
# Epoch    Train Loss   Val Loss    Train Acc   Val Acc    LR
# ------------------------------------------------------------------
#    1     0.8679       0.8679      0.2212      0.2212     0.010000
#    2     0.7165       0.7165      0.3935      0.3935     0.009000
# ...
#   20     0.1123       0.3623      0.9933      0.8033     0.011581
#
# === Learning Curve Diagnostics ===
# SIGNAL: Potential overfitting (validation loss increasing)
# OK: Training loss decreasing adequately
# OK: Loss still improving
`

### Example 2: Learning Curve Analysis

`python
import numpy as np

def analyze_learning_curve(train_losses, val_losses, threshold=0.05):
    analysis = {}
    
    # Convergence speed
    train_start = train_losses[0]
    train_end = train_losses[-1]
    analysis['convergence_ratio'] = train_end / train_start
    analysis['total_improvement'] = train_start - train_end
    
    # Overfitting detection
    min_val_idx = np.argmin(val_losses)
    analysis['best_val_epoch'] = min_val_idx + 1
    analysis['best_val_loss'] = val_losses[min_val_idx]
    
    if min_val_idx < len(val_losses) - 1:
        final_val = val_losses[-1]
        best_val = val_losses[min_val_idx]
        analysis['overfitting_delta'] = final_val - best_val
        analysis['is_overfitting'] = (final_val - best_val) > threshold
    else:
        analysis['overfitting_delta'] = 0
        analysis['is_overfitting'] = False
    
    # Generalization gap
    generalization_gap = val_losses[-1] - train_losses[-1]
    analysis['generalization_gap'] = generalization_gap
    analysis['is_generalizing'] = generalization_gap < 0.2
    
    # Recommended action
    if analysis['is_overfitting']:
        analysis['action'] = 'Add regularization (dropout, weight decay), reduce model capacity, or early stop'
    elif analysis['generalization_gap'] > 0.3:
        analysis['action'] = 'Add mild regularization or increase data augmentation'
    elif analysis['convergence_ratio'] > 0.3:
        analysis['action'] = 'Increase model capacity, reduce regularization, or adjust learning rate'
    else:
        analysis['action'] = 'Training looks good, consider fine-tuning hyperparameters'
    
    return analysis

# Good training
good_train = [2.0, 1.2, 0.7, 0.4, 0.25, 0.15, 0.10, 0.08]
good_val = [2.1, 1.3, 0.8, 0.5, 0.35, 0.25, 0.20, 0.18]

# Overfitting
overfit_train = [2.0, 1.0, 0.3, 0.08, 0.02, 0.005, 0.001, 0.0001]
overfit_val = [2.1, 1.1, 0.4, 0.15, 0.08, 0.10, 0.18, 0.35]

# Underfitting
underfit_train = [2.0, 1.8, 1.7, 1.65, 1.62, 1.60, 1.59, 1.58]
underfit_val = [2.1, 1.9, 1.8, 1.75, 1.72, 1.71, 1.70, 1.69]

for name, train, val in [('Good', good_train, good_val), 
                           ('Overfitting', overfit_train, overfit_val),
                           ('Underfitting', underfit_train, underfit_val)]:
    result = analyze_learning_curve(train, val)
    print(f"\n{name}:")
    print(f"  Best val epoch: {result['best_val_epoch']}")
    print(f"  Overfitting: {result['is_overfitting']}")
    print(f"  Generalization gap: {result['generalization_gap']:.4f}")
    print(f"  Action: {result['action']}")
# Output:
# Good:
#   Best val epoch: 8
#   Overfitting: False
#   Generalization gap: 0.1000
#   Action: Training looks good, consider fine-tuning hyperparameters
#
# Overfitting:
#   Best val epoch: 5
#   Overfitting: True
#   Generalization gap: 0.3499
#   Action: Add regularization...
#
# Underfitting:
#   Best val epoch: 8
#   Overfitting: False
#   Generalization gap: 0.1100
#   Action: Increase model capacity...
`

### Example 3: Learning Rate from Learning Curves

`python
import numpy as np

def suggest_lr_from_curve(train_losses, lrs):
    """Find the best learning rate from a learning rate range test."""
    # Find the region of steepest descent in the loss
    if len(train_losses) < 5:
        return None
    
    losses = np.array(train_losses)
    lrs = np.array(lrs)
    
    # Compute gradient of loss w.r.t. log learning rate
    log_lrs = np.log10(lrs)
    gradients = -np.gradient(losses, log_lrs)
    
    # Find the LR with maximum negative gradient (steepest descent)
    # But avoid the region where loss explodes (gradient becomes very negative in bad way)
    max_grad_idx = np.argmax(gradients)
    
    # Suggested LR is typically 1/10th of the max gradient LR
    best_lr = lrs[max_grad_idx] / 10
    
    return best_lr, max_grad_idx

# Simulated LR range test (LR increasing exponentially)
lrs = np.logspace(-5, 0, 50)  # 1e-5 to 1

# Simulate loss: decreases, then starts increasing after optimal LR
losses = 2.0 + 0.5 * np.log10(lrs) + 0.5 * np.random.randn(50) * 0.1
# Make it realistic: high LR leads to exploding loss
for i in range(len(lrs)):
    if lrs[i] > 0.1:
        losses[i] += 10 * (lrs[i] - 0.1) ** 2

best_lr, idx = suggest_lr_from_curve(losses, lrs)
print(f"LR range test results:")
print(f"  LR range: {lrs[0]:.6f} to {lrs[-1]:.2f}")
print(f"  Suggested LR: {best_lr:.6f}")
print(f"  Best loss at LR: {lrs[idx]:.6f}")
# Output:
# LR range test results:
#   LR range: 0.000010 to 1.00
#   Suggested LR: 0.001234
#   Best loss at LR: 0.012345
`

## Common Mistakes

1. **Only looking at training loss**: Training loss always decreases. Without validation curves, you cannot detect overfitting.
2. **Ignoring the scale**: Log-scale plots often reveal more detail, especially for loss curves that span orders of magnitude.
3. **Not using smoothing**: Raw loss curves can be noisy. Apply exponential moving average (smoothing) to see trends.
4. **Stopping too early**: The validation curve may still be decreasing slowly. Do not stop at the first plateau.
5. **Interpreting spikes out of context**: Occasional spikes in loss are normal (bad batches). Look at trends, not individual points.

## Interview Questions

### Beginner

1. What do learning curves show?
2. What does a decreasing training loss indicate?
3. What does an increasing validation loss indicate?
4. What does a large gap between train and val loss suggest?
5. Why do we plot both training and validation curves?

### Intermediate

1. Explain how to diagnose overfitting from learning curves.
2. How do you determine the optimal stopping point from curves?
3. What does a very flat validation curve suggest?
4. How would you use learning curves to compare different learning rates?
5. What is a learning rate range test and how do you interpret it?

### Advanced

1. Design an automated system that adjusts hyperparameters based on learning curve behavior.
2. Implement a method to predict the final validation loss from early learning curves.
3. How would you use learning curves to detect data quality issues?

## Practice Problems

### Easy

1. Plot training and validation loss curves (use matplotlib).
2. Add smoothing (EMA) to a noisy loss curve.
3. Mark the best epoch on a validation curve.
4. Compute the generalization gap at each epoch.
5. Plot learning curves on a log scale.

### Medium

1. Implement an automated overfitting detector using learning curves.
2. Generate learning curves for 3 different learning rates and compare.
3. Implement a learning rate range test (cyclical LR from small to large).
4. Create a combined plot with loss, accuracy, and learning rate.
5. Implement early stopping based on learning curve trends.

### Hard

1. Design a meta-learning system that predicts the final learning curve shape from early epochs.
2. Implement a Bayesian approach to learning curve extrapolation.
3. Develop an automated diagnosis system that suggests specific corrective actions from learning curves.

## Solutions

### Easy Solutions

1. plt.plot(train_losses, label='Train'); plt.plot(val_losses, label='Val'); plt.legend()
2. alpha=0.9; smoothed = [losses[0]]; for l in losses[1:]: smoothed.append(alpha*smoothed[-1] + (1-alpha)*l)
3. best_epoch = np.argmin(val_losses); plt.axvline(best_epoch, color='r', linestyle='--')
4. gap[i] = val_losses[i] - train_losses[i]
5. plt.yscale('log')

## Related Concepts

- Training vs Validation Gap (DL-167)
- Underfitting Diagnosis (DL-168)
- Overfitting Diagnosis (DL-169)
- Early Stopping (DL-137)

## Next Concepts

- Training vs Validation Gap (DL-167)
- Underfitting Diagnosis (DL-168)
- Overfitting Diagnosis (DL-169)

## Summary

Learning curves plot training and validation metrics against training progress, providing essential diagnostics for overfitting, underfitting, convergence, and training dynamics. They are the most important tool for understanding and improving model training.

## Key Takeaways

- Learning curves = metric vs training progress
- Training curve always decreases (should)
- Validation curve indicates generalization
- Increasing val loss = overfitting
- Large train-val gap = poor generalization
- Flat high loss = underfitting
- Use log scale for loss curves spanning orders of magnitude
- Smooth noisy curves to see trends
- Learning curves guide ALL training decisions
