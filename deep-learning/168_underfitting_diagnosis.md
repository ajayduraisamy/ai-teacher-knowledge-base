# Concept: Underfitting Diagnosis

## Concept ID

DL-168

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Training Deep Networks

## Learning Objectives

- Identify underfitting from learning curves and metrics
- Distinguish underfitting from overfitting
- Understand the root causes of underfitting
- Apply corrective strategies for underfitting
- Implement diagnostic tools for underfitting detection

## Prerequisites

- Learning curves (DL-166)
- Training vs Validation Gap (DL-167)
- Overfitting Diagnosis (DL-169)
- Model capacity concepts

## Definition

Underfitting occurs when a model fails to capture the underlying patterns in the training data, resulting in high training loss that has not plateaued at an acceptable level. Unlike overfitting where the model memorizes noise, underfitting reflects insufficient model capacity, inadequate training, or inappropriate architecture for the task. Key indicators: (1) training loss remains high and continues to decrease slowly, (2) validation loss is similarly high (small gap), (3) both curves are trending downward but have not converged, (4) accuracy is below expected baseline, (5) the model performs poorly on both training and test data. Underfitting is the opposite of overfitting: the model is too simple rather than too complex.

## Intuition

Imagine trying to fit a straight line through data that forms a parabola. No matter how you adjust the line, it will never capture the curve. That is underfitting - the model lacks the expressive power needed for the task. In deep learning, underfitting feels like the model is stuck: the loss decreases but very slowly, the model keeps making similar mistakes, and increasing training time does not help much. Unlike overfitting where you need to simplify, underfitting means you need to add capacity, train longer, or adjust the learning process. Think of it as giving a student material that is too advanced for a beginner - the student cannot learn effectively because the foundation is missing.

## Why This Concept Matters

Underfitting is more dangerous than overfitting because it is often harder to detect. Overfitting shows a clear gap between train and val curves. Underfitting hides behind both curves looking similar but both being high. Many practitioners misinterpret slow loss decrease as needing more training when the real issue is insufficient capacity. Correctly diagnosing underfitting saves enormous amounts of time: instead of training for 1000 epochs with minor improvement, you should increase model capacity, change architecture, or improve optimization. Understanding underfitting is essential for efficient deep learning workflow.

## Mathematical Explanation

Underfitting can be analyzed through the bias-variance decomposition of expected error: E[error] = bias^2 + variance + irreducible_error. Underfitting corresponds to high bias: the model's hypothesis class is too restricted to represent the true function. For a model with capacity parameter C (e.g., number of parameters), the bias scales as bias ~ 1/C^alpha and variance scales as variance ~ C/N where N is dataset size. The optimal capacity balances these. Underfitting occurs when C is below the optimal capacity. Training dynamics also reveal underfitting: the gradient norm remains high because the model consistently makes large errors. The Hessian spectrum shows many positive eigenvalues indicating the model is far from a good local minimum.

## Code Examples

### Example 1: Detecting Underfitting from Learning Curves

```python
import numpy as np

def diagnose_underfitting(train_losses, val_losses, epochs, threshold=0.3):
    results = {}
    
    final_train = train_losses[-1]
    initial_train = train_losses[0]
    improvement = (initial_train - final_train) / initial_train
    
    final_gap = val_losses[-1] - train_losses[-1]
    
    final_improvement_rate = (train_losses[-3] - train_losses[-1]) / max(1, epochs[-1] - epochs[-3])
    
    results['final_train_loss'] = final_train
    results['improvement_ratio'] = improvement
    results['generalization_gap'] = final_gap
    results['final_improvement_rate'] = final_improvement_rate
    
    is_underfitting = False
    reasons = []
    
    if final_train > threshold:
        is_underfitting = True
        reasons.append(f"Training loss ({final_train:.4f}) exceeds threshold ({threshold})")
    
    if improvement < 0.5:
        is_underfitting = True
        reasons.append(f"Insufficient improvement ({improvement:.1%}) from initial loss")
    
    if final_gap < 0.05 and final_train > threshold * 0.5:
        is_underfitting = True
        reasons.append(f"Small gap ({final_gap:.4f}) but high train loss - classic underfitting")
    
    if final_improvement_rate < 0.001 and final_train > threshold:
        is_underfitting = True
        reasons.append(f"Training stalled with high loss")
    
    results['is_underfitting'] = is_underfitting
    results['reasons'] = reasons
    
    return results

# Underfitting scenario: both losses high, small gap, slow improvement
underfit_train = [2.0, 1.9, 1.8, 1.72, 1.65, 1.60, 1.56, 1.53, 1.50, 1.48]
underfit_val = [2.1, 2.0, 1.9, 1.82, 1.75, 1.70, 1.66, 1.63, 1.60, 1.58]

# Healthy scenario
healthy_train = [2.0, 1.2, 0.7, 0.4, 0.25, 0.15, 0.10, 0.08, 0.06, 0.05]
healthy_val = [2.1, 1.3, 0.8, 0.5, 0.35, 0.25, 0.20, 0.18, 0.16, 0.15]

epochs = list(range(1, 11))

print("=== Underfitting Scenario ===")
result = diagnose_underfitting(underfit_train, underfit_val, epochs)
print(f"Underfitting detected: {result['is_underfitting']}")
for r in result['reasons']:
    print(f"  - {r}")

print("\n=== Healthy Scenario ===")
result = diagnose_underfitting(healthy_train, healthy_val, epochs)
print(f"Underfitting detected: {result['is_underfitting']}")
for r in result['reasons']:
    print(f"  - {r}")
```

### Example 2: Capacity Analysis for Underfitting

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def test_capacity_for_task(task_type='quadratic', hidden_sizes=[4, 16, 64, 256]):
    """Test different model capacities on a synthetic task."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate data
    X = torch.randn(200, 1)
    if task_type == 'quadratic':
        y = 0.5 * X**2 + 0.3 * X + 0.1 * torch.randn(200, 1)
        input_dim, output_dim = 1, 1
    elif task_type == 'sinusoidal':
        y = torch.sin(2 * X) + 0.1 * torch.randn(200, 1)
        input_dim, output_dim = 1, 1
    else:
        X = torch.randn(200, 10)
        y = torch.sum(X**2, dim=1, keepdim=True) + 0.1 * torch.randn(200, 1)
        input_dim, output_dim = 10, 1
    
    train_X, val_X = X[:150], X[150:]
    train_y, val_y = y[:150], y[150:]
    
    results = []
    
    for hidden_size in hidden_sizes:
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )
        
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(500):
            model.train()
            optimizer.zero_grad()
            pred = model(train_X)
            loss = loss_fn(pred, train_y)
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0 or epoch == 499:
                model.eval()
                with torch.no_grad():
                    train_loss = loss_fn(model(train_X), train_y).item()
                    val_loss = loss_fn(model(val_X), val_y).item()
                train_losses.append(train_loss)
                val_losses.append(val_loss)
        
        final_train = train_losses[-1]
        final_val = val_losses[-1]
        gap = final_val - final_train
        
        is_underfit = final_train > 0.15 and gap < 0.05
        results.append({
            'hidden_size': hidden_size,
            'final_train': round(final_train, 4),
            'final_val': round(final_val, 4),
            'gap': round(gap, 4),
            'underfitting': is_underfit
        })
    
    print(f"Capacity Analysis for {task_type} task")
    print("=" * 70)
    print(f"{'Hidden Size':12s} {'Train Loss':12s} {'Val Loss':12s} {'Gap':10s} {'Status':15s}")
    print("-" * 70)
    for r in results:
        status = "UNDERFITTING" if r['underfitting'] else "OK"
        print(f"{r['hidden_size']:<12d} {r['final_train']:<12.4f} {r['final_val']:<12.4f} "
              f"{r['gap']:<+10.4f} {status:15s}")
    print("\nDiagnosis: If small models show underfitting, increase capacity.")
    print("If all models show high loss, the architecture may be fundamentally unsuitable.")

test_capacity_for_task('quadratic')
```

### Example 3: Underfitting vs Overfitting Classification

```python
import numpy as np

def classify_fit(train_losses, val_losses, train_loss_threshold=0.2, gap_threshold=0.15):
    """Classify training as underfitting, overfitting, or good fit."""
    final_train = train_losses[-1]
    final_val = val_losses[-1]
    gap = final_val - final_train
    
    improvement = (train_losses[0] - final_train) / max(train_losses[0], 1e-8)
    
    gap_trend = np.polyfit(range(len(val_losses)), val_losses - np.array(train_losses), 1)[0]
    
    print("Training Fit Classification")
    print("=" * 60)
    print(f"Final training loss: {final_train:.4f}")
    print(f"Final validation loss: {final_val:.4f}")
    print(f"Generalization gap: {gap:.4f}")
    print(f"Improvement from initial: {improvement:.1%}")
    print(f"Gap trend: {gap_trend:+.4f}/epoch")
    print()
    
    if final_train > train_loss_threshold and gap < gap_threshold:
        classification = "UNDERFITTING"
        advice = "Increase model capacity, train longer, or adjust learning rate"
    elif final_train <= train_loss_threshold and gap >= gap_threshold:
        if gap_trend > 0.01:
            classification = "OVERFITTING (active)"
            advice = "Add regularization or stop training early"
        else:
            classification = "OVERFITTING (stable)"
            advice = "Add regularization or reduce model capacity"
    elif final_train > train_loss_threshold and gap >= gap_threshold:
        classification = "BOTH underfitting and overfitting"
        advice = "Increase capacity AND add regularization. Consider different architecture."
    else:
        classification = "GOOD FIT"
        advice = "Training is proceeding well. Monitor for changes."
    
    print(f"Classification: {classification}")
    print(f"Recommended action: {advice}")
    return classification

# Test cases
print("Test 1: Classic Underfitting")
classify_fit(
    [2.0, 1.9, 1.8, 1.72, 1.65, 1.60, 1.56, 1.53],
    [2.1, 2.0, 1.9, 1.82, 1.75, 1.70, 1.66, 1.63]
)

print("\nTest 2: Good Fit")
classify_fit(
    [2.0, 1.2, 0.7, 0.4, 0.25, 0.15, 0.10, 0.08],
    [2.1, 1.3, 0.8, 0.5, 0.35, 0.25, 0.20, 0.18]
)

print("\nTest 3: Overfitting")
classify_fit(
    [2.0, 1.0, 0.3, 0.08, 0.02, 0.005, 0.001, 0.0005],
    [2.1, 1.1, 0.45, 0.20, 0.15, 0.18, 0.28, 0.45]
)
```

## Common Mistakes

1. **Confusing underfitting with insufficient training**: If loss is still decreasing at a reasonable rate, the model needs more epochs, not more capacity. Check the improvement rate.
2. **Adding regularization to underfitting models**: Regularization increases bias, making underfitting worse. Never add dropout or weight decay to an underfitting model.
3. **Increasing capacity without checking data quality**: Sometimes underfitting symptoms are caused by noisy labels or preprocessing issues. Always verify data first.
4. **Ignoring learning rate issues**: A too-low learning rate looks like underfitting (slow decrease). Try LR warmup or higher initial LR before adding capacity.
5. **Assuming deeper is always better**: A model that is too deep can underfit due to optimization difficulties (vanishing gradients). Sometimes wider is better than deeper.

## Interview Questions

### Beginner

1. What is underfitting?
2. How do learning curves look for underfitting?
3. What causes underfitting?
4. How is underfitting different from overfitting?
5. What is the first thing to check when you suspect underfitting?

### Intermediate

1. Explain the relationship between model capacity and underfitting.
2. How do you distinguish underfitting from insufficient training?
3. What architectural changes fix underfitting?
4. How does the learning rate affect underfitting?
5. Can too much regularization cause underfitting?

### Advanced

1. Analyze the bias-variance decomposition in the context of underfitting.
2. Design a meta-learning approach that automatically adjusts capacity to avoid underfitting.
3. How does underfitting manifest in different architectures (CNNs vs Transformers vs RNNs)?

## Practice Problems

### Easy

1. Plot learning curves for an underfitting model.
2. Compute the improvement ratio and determine if underfitting.
3. List three causes of underfitting.
4. Compare train and val loss values to classify fit type.
5. Identify underfitting from gap and absolute loss values.

### Medium

1. Implement an automated underfitting detector.
2. Create a capacity sweep experiment and identify the optimal capacity.
3. Diagnose whether slow improvement is underfitting or insufficient training.
4. Build a training monitor that warns about underfitting.
5. Experiment with different optimizers to mitigate underfitting.

### Hard

1. Implement neural architecture search to automatically find capacity that avoids underfitting.
2. Design a curriculum learning strategy that reduces underfitting for complex tasks.
3. Implement gradient flow analysis to detect underfitting caused by vanishing gradients.

## Solutions

### Easy Solutions

1. `plt.plot(train_losses, label='Train'); plt.plot(val_losses, label='Val'); plt.axhline(0.3, color='r', linestyle='--', label='Threshold')`
2. `ratio = (train_losses[0] - train_losses[-1]) / train_losses[0]; is_underfitting = ratio < 0.5 and train_losses[-1] > 0.3`
3. Insufficient model capacity, too little training, poor optimization
4. If both losses are high and gap is small = underfitting. If train loss is low and val loss is high = overfitting.
5. `gap = val_loss - train_loss; is_underfitting = train_loss > 0.3 and gap < 0.05`

## Related Concepts

- Overfitting Diagnosis (DL-169)
- Training vs Validation Gap (DL-167)
- Learning Curves (DL-166)
- Model Capacity

## Next Concepts

- Overfitting Diagnosis (DL-169)
- Bias-Variance Tradeoff
- Model Selection

## Summary

Underfitting occurs when a model lacks the capacity to learn the training data, characterized by high training loss, small generalization gap, and slow improvement. Correct diagnosis requires distinguishing underfitting from insufficient training and poor optimization. Solutions include increasing model capacity, adjusting learning rate, improving optimization, and verifying data quality.

## Key Takeaways

- Underfitting = high train loss + small gap
- Do not add regularization when underfitting
- Check learning rate before increasing capacity
- Verify data quality before adjusting architecture
- Distinguish underfitting from insufficient training
- Increase capacity (wider or deeper)
- Try better optimizers (Adam, AdamW)
- Consider architecture change
- Monitor improvement rate
- Small gap with high loss = underfitting
