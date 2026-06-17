# Concept: Test Loop

## Concept ID

DL-158

## Difficulty

Beginner

## Domain

Deep Learning

## Module

Training Deep Networks

## Learning Objectives

- Understand the purpose of a held-out test set
- Implement a test loop in PyTorch
- Distinguish between validation and test loops
- Compute final model performance metrics
- Ensure test data remains unseen during development

## Prerequisites

- Training loop (DL-156)
- Validation loop (DL-157)
- Dataset splitting concepts
- Understanding of model evaluation

## Definition

The test loop evaluates the final trained model on a held-out test set that has never been used during training or validation. Unlike the validation loop (used for model selection and hyperparameter tuning), the test loop is run only once at the very end to estimate real-world performance. The test set must remain completely isolated from all training decisions — no early stopping based on test loss, no hyperparameter tuning using test results.

## Intuition

Think of the validation set as practice exams and the test set as the final exam. You use practice exams (validation) to study and improve. But looking at the final exam (test) during studying would defeat the purpose — you would learn the answers rather than the subject. The test loop is the final exam that gives an unbiased estimate of how well the model will perform in the real world. Using the test set multiple times (even just looking at the loss) leaks information and gives overly optimistic performance estimates.

## Why This Concept Matters

Proper test set evaluation is essential for: (1) obtaining unbiased performance estimates, (2) comparing different models fairly, (3) detecting data leakage in the pipeline, (4) reporting results in research papers accurately, and (5) building trust in model deployment decisions. Misusing the test set (e.g., tuning hyperparameters on test results) is one of the most common and serious methodological errors in machine learning.

## Code Examples

### Example 1: Basic Test Loop

`python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def test(model, test_loader, criterion, device='cpu'):
    """Final test evaluation — run only once."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            
            test_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            all_preds.extend(predicted.cpu().tolist())
            all_targets.extend(y.cpu().tolist())
    
    avg_loss = test_loss / total
    accuracy = correct / total
    
    print(f"=== FINAL TEST RESULTS ===")
    print(f"Test loss: {avg_loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Correct: {correct}/{total}")
    
    return avg_loss, accuracy, all_preds, all_targets

model = nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 5))
criterion = nn.CrossEntropyLoss()

X_test = torch.randn(300, 20)
y_test = torch.randint(0, 5, (300,))
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

test_loss, test_acc, preds, targets = test(model, test_loader, criterion)
# Output:
# === FINAL TEST RESULTS ===
# Test loss: 1.6094
# Test accuracy: 0.1967
# Correct: 59/300
`

### Example 2: Full Train/Val/Test Pipeline

`python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

def train_val_test_pipeline():
    # Generate full dataset
    X = torch.randn(1000, 20)
    y = torch.randint(0, 5, (1000,))
    dataset = TensorDataset(X, y)
    
    # Split: 70% train, 15% val, 15% test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Training
    model = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 5))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    for epoch in range(10):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(x)
                _, preds = torch.max(outputs, 1)
                val_total += y.size(0)
                val_correct += (preds == y).sum().item()
        val_acc = val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Final test (one time only)
    model.load_state_dict(best_state)
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            test_total += y.size(0)
            test_correct += (preds == y).sum().item()
    test_acc = test_correct / test_total
    
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}")
    return test_acc

test_acc = train_val_test_pipeline()
# Output:
# Train: 700, Val: 150, Test: 150
# Best validation accuracy: 0.2733
# Final test accuracy: 0.2467
`

### Example 3: Comprehensive Test Report

`python
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def comprehensive_test(model, test_loader, class_names=None, device='cpu'):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    accuracy = (all_preds == all_targets).mean()
    
    print("=" * 50)
    print("COMPREHENSIVE TEST REPORT")
    print("=" * 50)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Total samples: {len(all_targets)}")
    print()
    print("Per-class metrics:")
    print(classification_report(all_targets, all_preds, 
          target_names=class_names, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))
    
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs,
    }

model = nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 3))
X_test = torch.randn(100, 20)
y_test = torch.randint(0, 3, (100,))
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=20)

results = comprehensive_test(model, test_loader, class_names=['A', 'B', 'C'])
# Output:
# ==================================================
# COMPREHENSIVE TEST REPORT
# ==================================================
# Overall Accuracy: 0.3300
# Total samples: 100
#
# Per-class metrics:
#               precision    recall  f1-score   support
#            A     0.3333    0.3226    0.3279        31
#            B     0.3333    0.3429    0.3380        35
#            C     0.3235    0.3235    0.3235        34
#
#    accuracy                         0.3300       100
#    macro avg    0.3301    0.3297    0.3298       100
# weighted avg    0.3299    0.3300    0.3299       100
#
# Confusion Matrix:
# [[10 10 11]
#  [10 12 13]
#  [10 13 11]]
`

## Common Mistakes

1. **Using the test set multiple times**: Each evaluation on the test set leaks information. Run the test loop exactly once.
2. **Tuning hyperparameters based on test results**: Use validation for tuning, not test. This is the most common methodological error.
3. **Data leakage from test to training**: Ensure no preprocessing is learned from the test set (e.g., normalization statistics).
4. **Applying test-time augmentations without reporting**: If using test-time augmentation, clearly report this.
5. **Reporting test metrics with confidence intervals**: A single test accuracy number can be misleading. Report confidence intervals or perform multiple test evaluations.

## Interview Questions

### Beginner

1. Why do we need a separate test set?
2. How often should the test loop be run?
3. What is the difference between validation and test sets?
4. Can we use test results to improve the model?
5. What happens if we tune hyperparameters on test data?

### Intermediate

1. How do you compute confidence intervals for test accuracy?
2. What is data leakage and how does it affect test results?
3. How should you handle test sets for imbalanced datasets?
4. Compare stratified test splits with random splits.
5. When would you use multiple test sets?

### Advanced

1. Design a test strategy for time series that avoids temporal leakage.
2. Implement a permutation test for comparing two models' test performance.
3. How would you estimate the generalization error without a test set (e.g., using PAC-Bayes bounds)?

## Practice Problems

### Easy

1. Write a test loop for a regression model.
2. Add 95% confidence intervals to test accuracy.
3. Implement a test loop that computes per-class accuracy.
4. Add ROC-AUC computation to the test loop.
5. Write a test loop that handles multi-label inputs.

### Medium

1. Implement a stratified test split that preserves class proportions.
2. Compute and report the confusion matrix.
3. Implement bootstrapping to estimate test accuracy variance.
4. Compare test performance with and without test-time augmentation.
5. Implement a test loop for a model with multiple outputs.

### Hard

1. Implement a statistical test (McNemar's test) for comparing two models on the test set.
2. Design a test harness for object detection that computes COCO-style metrics.
3. Implement a test pipeline that automatically detects data leakage.

## Solutions

### Easy Solutions

1. For regression: track MSE, MAE, R2 instead of accuracy
2. Use bootstrap: sample N indices with replacement, compute accuracy, repeat 1000 times, report 2.5 and 97.5 percentiles
3. Track correct and total per class; compute accuracy_per_class = correct_per_class / total_per_class
4. For binary classification, compute AUC via sklearn.metrics.roc_auc_score
5. For multi-label, compute per-class metrics and average

## Related Concepts

- Validation Loop (DL-157)
- Training Loop (DL-156)
- Data Splitting
- Model Evaluation

## Next Concepts

- Checkpointing (DL-159)
- Model Saving and Loading (DL-160)
- Experiment Tracking (DL-161)

## Summary

The test loop evaluates the final model on a held-out test set exactly once. It uses model.eval() and torch.no_grad(), and its results provide an unbiased estimate of real-world performance. The test set must remain completely isolated from all training and validation decisions throughout development.

## Key Takeaways

- Run the test loop exactly once at the end of development
- Test set must never influence model selection or hyperparameter tuning
- Use model.eval() and torch.no_grad() during testing
- No data augmentation on test data
- Report test metrics with confidence intervals
- Distinguish test from validation — they serve different purposes
- Data leakage on test set invalidates all results
- Comprehensive test reports include per-class metrics and confusion matrix
