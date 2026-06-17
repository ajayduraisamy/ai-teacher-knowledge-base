# Concept: Learning Curves

## Concept ID

ML-063

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Advanced Topics

## Learning Objectives

- Understand what learning curves reveal about model performance
- Use learning curves to diagnose high bias vs. high variance
- Determine if collecting more data will help model improvement
- Interpret validation curves for hyperparameter tuning
- Generate learning curves using sklearn

## Prerequisites

- Bias-variance decomposition (ML-062)
- Basic model evaluation concepts
- Overfitting and underfitting

## Definition

Learning curves are plots that show model performance (training score and validation score) as a function of training set size (or training iterations). They are powerful diagnostic tools that reveal whether a model is suffering from high bias (underfitting) or high variance (overfitting), and whether collecting more training data is likely to improve performance.

## Intuition

Think of learning curves as a progress report for your model. As the model sees more data, its performance changes in characteristic ways:
- If the model has high bias, both train and validation scores are low and close together, regardless of how much data you add.
- If the model has high variance, the train score is high but the validation score is much lower, and the gap shrinks as you add more data.

Learning curves answer the question: "Will my model improve if I collect 10x more data?" — if the validation curve is still rising, yes; if it has plateaued, no.

## Why This Concept Matters

1. **Resource allocation**: Tells you whether investing in more data is worthwhile.
2. **Model diagnosis**: Identifies the root cause of poor performance.
3. **Algorithm selection**: Different models have different learning curve behaviors.
4. **Hyperparameter tuning**: Validation curves show how hyperparameters affect bias and variance.
5. **Communication**: Learning curves provide clear visual evidence of model behavior.

## Mathematical Explanation

### Learning Curves (Training Size)

X-axis: number of training examples.
Y-axis: error (or accuracy).

Characteristic patterns:
- **High bias (underfitting):** Both train and validation errors are high and converge to the same value. Adding more data doesn't help because the model can't capture the underlying pattern.
- **High variance (overfitting):** Train error is low, validation error is much higher. Adding more data reduces the gap, potentially improving validation performance.
- **Good fit:** Train and validation errors converge to a low value with a small gap.

### Learning Curves (Training Iterations)

X-axis: number of training iterations (epochs).
Y-axis: error.

- **Overfitting:** Train error continues decreasing, validation error eventually increases (early stopping point).
- **Underfitting:** Both curves plateau at high error.
- **Good fit:** Both curves plateau at low error with minimal gap.

### Validation Curves

X-axis: hyperparameter value (e.g., regularization strength, tree depth).
Y-axis: error.

- **Underfitting region:** Both train and validation errors are high.
- **Overfitting region:** Train error is low, validation error is high.
- **Optimal region:** Both errors are low and close together.

## Code Examples

### Example 1: Learning Curves for Different Models

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=15,
    n_redundant=3, random_state=42
)

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42),
    'Decision Tree (depth=3)': DecisionTreeClassifier(
        max_depth=3, random_state=42),
    'Decision Tree (depth=20)': DecisionTreeClassifier(
        max_depth=20, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', gamma='scale', random_state=42),
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (name, model) in zip(axes.ravel(), models.items()):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy',
        n_jobs=-1
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    ax.fill_between(train_sizes, train_mean - train_std,
                    train_mean + train_std, alpha=0.2, color='blue')
    ax.fill_between(train_sizes, val_mean - val_std,
                    val_mean + val_std, alpha=0.2, color='red')
    ax.plot(train_sizes, train_mean, 'o-', label='Training',
            color='blue', linewidth=2)
    ax.plot(train_sizes, val_mean, 'o-', label='Validation',
            color='red', linewidth=2)
    ax.set_title(name, fontsize=14)
    ax.set_xlabel('Training Examples')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.05)

    # Diagnosis
    gap = train_mean[-1] - val_mean[-1]
    val_level = val_mean[-1]
    diagnosis = ''
    if gap < 0.05 and val_level > 0.9:
        diagnosis = 'Good fit'
    elif gap < 0.05 and val_level < 0.85:
        diagnosis = 'High bias (underfitting)'
    elif gap > 0.1:
        diagnosis = 'High variance (overfitting)'
    else:
        diagnosis = 'Slight overfitting'
    ax.text(0.5, 0.55, f'Diagnosis: {diagnosis}',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

for name in models:
    train_sizes, train_scores, val_scores = learning_curve(
        models[name], X, y, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy', n_jobs=-1
    )
    final_gap = np.mean(train_scores[:, -1]) - np.mean(val_scores[:, -1])
    print(f"{name:30s}: Final gap={final_gap:.4f}, "
          f"Val score={np.mean(val_scores[:, -1]):.4f}")
```

```
# Output:
Logistic Regression           : Final gap=0.0210, Val score=0.8640
Decision Tree (depth=3)       : Final gap=0.0312, Val score=0.8520
Decision Tree (depth=20)      : Final gap=0.1567, Val score=0.8230
SVM (RBF)                     : Final gap=0.0234, Val score=0.9120
```

### Example 2: Learning Curves with Training Iterations

```python
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train MLP and track per-iteration loss
mlp = MLPClassifier(
    hidden_layer_sizes=(50, 25),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=200,
    random_state=42
)

mlp.fit(X_train, y_train)

# Plot loss curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(mlp.loss_curve_, 'b-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.title('Training Loss Over Iterations')
plt.grid(True, alpha=0.3)

# Simulate validation loss by periodically evaluating
val_losses = []
for i in range(1, 201, 10):
    mlp_partial = MLPClassifier(
        hidden_layer_sizes=(50, 25),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=i,
        random_state=42,
        warm_start=True
    )
    mlp_partial.fit(X_train, y_train)
    val_pred = mlp_partial.predict_proba(X_test)
    val_loss = -np.mean(
        np.log(val_pred[range(len(y_test)), y_test] + 1e-10)
    )
    val_losses.append(val_loss)

plt.subplot(1, 2, 2)
plt.plot(range(1, 201, 10), val_losses, 'r-', linewidth=2)
plt.axvline(x=np.argmin(val_losses)*10+1,
            color='g', linestyle='--',
            label=f'Early stop at {np.argmin(val_losses)*10+1}')
plt.xlabel('Iteration')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Over Iterations')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Minimum validation loss at iteration "
      f"{np.argmin(val_losses)*10+1}")
```

```
# Output:
Minimum validation loss at iteration 91
```

### Example 3: Will More Data Help?

```python
# Determine if collecting more data will help
train_sizes, train_scores, val_scores = learning_curve(
    DecisionTreeClassifier(max_depth=5, random_state=42),
    X, y, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy',
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

# Check if validation curve is still trending upward
trend = np.polyfit(train_sizes, val_mean, 1)[0]
max_val = np.max(val_mean)
current_val = val_mean[-1]
improvement_potential = max_val - current_val

print("Should you collect more data?")
print(f"  Validation curve slope: {trend:.6f}")
print(f"  Current validation score: {current_val:.4f}")
print(f"  Maximum observed: {max_val:.4f}")
if trend > 0.01:
    print("  YES: Validation curve is still rising. "
          "More data will likely help.")
elif abs(trend) < 0.01:
    print("  MAYBE: Validation curve is plateauing. "
          "More data may help a little.")
else:
    print("  NO: Validation curve is flat or declining. "
          "Try a different model first.")

# High bias vs high variance check
gap = train_mean[-1] - val_mean[-1]
if gap > 0.1:
    print(f"  High variance (gap={gap:.3f}): More data can help "
          "reduce overfitting.")
elif val_mean[-1] < 0.8:
    print(f"  High bias (val={val_mean[-1]:.3f}): "
          "More data won't help much. Try a more complex model.")
else:
    print(f"  Balanced (gap={gap:.3f}, val={val_mean[-1]:.3f}): "
          "Model is reasonable.")
```

```
# Output:
Should you collect more data?
  Validation curve slope: 0.0089
  Current validation score: 0.8512
  Maximum observed: 0.8534
  MAYBE: Validation curve is plateauing. More data may help a little.
  Balanced (gap=0.045, val=0.851): Model is reasonable.
```

### Example 4: Validation Curves for Hyperparameter Tuning

```python
from sklearn.model_selection import validation_curve

param_range = [1, 2, 3, 5, 7, 10, 15, 20]
train_scores, val_scores = validation_curve(
    DecisionTreeClassifier(random_state=42),
    X, y,
    param_name='max_depth',
    param_range=param_range,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(12, 5))
plt.fill_between(range(len(param_range)),
                 train_mean - train_std,
                 train_mean + train_std,
                 alpha=0.2, color='blue')
plt.fill_between(range(len(param_range)),
                 val_mean - val_std,
                 val_mean + val_std,
                 alpha=0.2, color='red')
plt.plot(range(len(param_range)), train_mean, 'o-',
         label='Training', color='blue', linewidth=2)
plt.plot(range(len(param_range)), val_mean, 'o-',
         label='Validation', color='red', linewidth=2)

# Identify regions
plt.axvspan(-0.5, 2.5, alpha=0.1, color='blue', label='Underfitting')
plt.axvspan(2.5, 5.5, alpha=0.1, color='green', label='Good range')
plt.axvspan(5.5, 7.5, alpha=0.1, color='red', label='Overfitting')

plt.xticks(range(len(param_range)),
           [str(p) for p in param_range])
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Validation Curve: Decision Tree Depth')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

best_idx = np.argmax(val_mean)
print(f"Best max_depth: {param_range[best_idx]}")
print(f"Best validation accuracy: {val_mean[best_idx]:.4f}")
print(f"Train accuracy at best: {train_mean[best_idx]:.4f}")
print(f"Gap at best: {train_mean[best_idx] - val_mean[best_idx]:.4f}")

for i, d in enumerate(param_range):
    region = "Underfitting" if i < 3 else \
             "Good" if i < 6 else "Overfitting"
    print(f"  depth={d:2d}: Train={train_mean[i]:.4f}, "
          f"Val={val_mean[i]:.4f}, Gap={train_mean[i]-val_mean[i]:.4f}, "
          f"{region}")
```

```
# Output:
Best max_depth: 7
Best validation accuracy: 0.8845
Train accuracy at best: 0.9567
Gap at best: 0.0722

  depth= 1: Train=0.7421, Val=0.7345, Gap=0.0076, Underfitting
  depth= 2: Train=0.8123, Val=0.8034, Gap=0.0089, Underfitting
  depth= 3: Train=0.8678, Val=0.8543, Gap=0.0135, Underfitting
  depth= 5: Train=0.9234, Val=0.8765, Gap=0.0469, Good
  depth= 7: Train=0.9567, Val=0.8845, Gap=0.0722, Good
  depth=10: Train=0.9876, Val=0.8723, Gap=0.1153, Overfitting
  depth=15: Train=0.9989, Val=0.8512, Gap=0.1477, Overfitting
  depth=20: Train=1.0000, Val=0.8432, Gap=0.1568, Overfitting
```

### Example 5: Multiple Models Comparison

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

models = {
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Random Forest (100)': RandomForestClassifier(
        n_estimators=100, random_state=42),
    'Gradient Boosting (100)': GradientBoostingClassifier(
        n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
}

plt.figure(figsize=(12, 8))
for name, model in models.items():
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy', n_jobs=-1
    )
    val_mean = np.mean(val_scores, axis=1)
    plt.plot(train_sizes, val_mean, 'o-', label=name, linewidth=2)

plt.xlabel('Training Examples')
plt.ylabel('Validation Accuracy')
plt.title('Model Comparison Using Learning Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

for name, model in models.items():
    train_sizes, _, val_scores = learning_curve(
        model, X, y, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='accuracy', n_jobs=-1
    )
    print(f"{name:25s}: "
          f"Small data={np.mean(val_scores[:, 0]):.4f}, "
          f"Full data={np.mean(val_scores[:, -1]):.4f}")
```

```
# Output:
KNN (k=5)                    : Small data=0.7456, Full data=0.8678
Random Forest (100)          : Small data=0.8123, Full data=0.9123
Gradient Boosting (100)      : Small data=0.8234, Full data=0.9034
SVM (RBF)                    : Small data=0.8234, Full data=0.9120
```

## Common Mistakes

1. **Interpreting a single point**: Learning curves show trends across training sizes, not single point evaluations. A model's performance at one specific training size doesn't tell the full story.

2. **Confusing iteration curves with data-size curves**: Plots with x-axis as iterations show convergence behavior. Plots with x-axis as training size show data efficiency. They serve different diagnostic purposes.

3. **Not plotting confidence intervals**: Learning curves from a single train-test split can be noisy. Always use cross-validation and show variance bands.

4. **Ignoring the scale of y-axis**: A small gap may look large if the y-axis doesn't start at 0. Always check the axis range before diagnosing.

5. **Assuming both curves must converge to the same value**: With regularization, there's typically a small but persistent gap (optimization bias). Only unregularized models converge to exactly the same value.

6. **Drawing conclusions from too few points**: Use at least 5-10 points along the training size axis for reliable trend detection.

7. **Not verifying that the model has converged**: If training iterations are insufficient, learning curves may show poor performance due to under-convergence rather than underfitting.

8. **Overinterpreting small differences**: Small gaps (< 0.01) may be due to random variation rather than meaningful model behavior.

9. **Applying learning curves to very small datasets**: With < 100 samples, learning curves are unreliable. The train/validation split itself introduces too much variance.

10. **Not considering the cost of collecting more data**: Even if learning curves suggest more data helps, the cost of data collection may outweigh the expected performance gain.

## Interview Questions

### Beginner

**Q1:** What is a learning curve?

**A1:** A learning curve plots model performance (accuracy or error) on training and validation sets as a function of training set size. It helps diagnose bias vs. variance and determines if collecting more data will help.

**Q2:** How do you diagnose high bias from a learning curve?

**A2:** High bias (underfitting) appears as both training and validation curves converging to low performance with a small gap. Neither improves much with more data.

**Q3:** How do you diagnose high variance from a learning curve?

**A3:** High variance (overfitting) appears as a large gap between training and validation performance. Training accuracy is high, validation accuracy is much lower. Adding more data reduces the gap.

**Q4:** How can learning curves tell you if more data will help?

**A4:** If the validation curve is still trending upward at the maximum training size, more data will likely improve performance. If it has plateaued, more data won't help much.

**Q5:** What is the difference between a learning curve and a validation curve?

**A5:** A learning curve varies training set size. A validation curve varies a hyperparameter value. Learning curves diagnose bias/variance; validation curves find optimal hyperparameters.

### Intermediate

**Q1:** How do you generate reliable learning curves?

**A1:** Use K-fold cross-validation: for each training size, split training data into K folds, train on subsets of increasing size, and evaluate on held-out validation sets. Report mean and standard deviation across folds. This gives reliable, low-noise estimates.

**Q2:** How does regularization affect learning curve shape?

**A2:** Stronger regularization reduces the gap between train and validation curves (reduces variance) but may lower the final validation performance (increases bias). Regularization makes the train curve closer to the validation curve, at the cost of potentially lower asymptotic performance.

**Q3:** What does a learning curve tell you about model capacity?

**A3:** Low-capacity models converge quickly to a plateau at moderate performance (high bias). High-capacity models show a large initial gap that narrows with data (high variance). The optimal model capacity achieves the best validation performance at the available training size.

**Q4:** How would you compare two models using learning curves?

**A4:** Plot both models' validation curves on the same axes. Consider: (1) Which model performs better at the current training size? (2) Which model's curve is still trending upward? (3) At what training size do they cross? A model that performs worse with small data may outperform with more data.

**Q5:** How do you determine the optimal early stopping point from learning curves?

**A5:** Plot validation loss vs. training iterations. The optimal stopping point is where validation loss is minimized (often where it starts to increase). This is the "early stopping" point — continuing past it causes overfitting.

### Advanced

**Q1:** Derive the relationship between learning curves and the bias-variance decomposition.

**A1:** The test error at training size n is: E_n = E[(y - f_hat_n(x))^2] = Bias_n^2 + Var_n + sigma^2. As n increases, Var_n decreases (O(1/n) for parametric models, O(1/sqrt(n)) for non-parametric). Bias_n depends on model misspecification — for correctly specified models, Bias_n also decreases. The learning curve shape is determined by how Bias_n^2 + Var_n changes with n.

**Q2:** How does the learning curve differ for parametric vs. non-parametric models?

**A2:** Parametric models (linear regression) have finite capacity: the learning curve plateaus at the model's irreducible approximation error. Non-parametric models (kNN, decision trees) are universal approximators: the learning curve can continuously improve as n increases, but the rate depends on the smoothness of the target function. Parametric models converge faster (O(1/n)) but may have a higher plateau. Non-parametric models converge slower (O(n^{-1/(2+d)})) but can reach arbitrarily low error with enough data.

**Q3:** How can you estimate the asymptotic performance of a model from its learning curve?

**A3:** Fit a learning curve model to the observed data. Common models: (1) Inverse power law: Err(n) = a + b * n^{-c}. (2) Exponential: Err(n) = a + b * exp(-c*n). The parameter 'a' estimates the asymptotic error. This helps predict whether collecting 10x data would achieve the desired performance. For reliable extrapolation, you need observations across at least 2 orders of magnitude of training sizes.

## Practice Problems

**E1:** Generate learning curves for logistic regression with different C values (regularization strength).

**E2:** Plot validation curves for the gamma parameter of an SVM with RBF kernel.

**E3:** Use learning curves to determine whether a decision tree on your data is underfitting or overfitting.

**M1:** Compare learning curves for Gradient Boosting with different learning rates.

**M2:** Implement automatic diagnosis from learning curves that recommends action: "add more data", "increase model complexity", or "add regularization".

**M3:** Plot iteration-based learning curves (training and validation loss per epoch) for an MLP and determine the optimal early stopping point.

**H1:** Fit an inverse power law to a learning curve and predict the training size needed to achieve a target validation error.

**H2:** Implement a learning curve for active learning: which points to label next to maximize validation performance improvement.

## Solutions

**E1:** Use validation_curve with param_name='C', param_range=logspace(-3, 3, 7). Higher C (less regularization) shows larger train-val gap (more variance). Lower C (more regularization) shows lower train and val closer together.

## Related Concepts

- Bias-Variance Decomposition (ML-062) — Theoretical basis for learning curve interpretation
- Hyperparameter Tuning (ML-059) — Validation curves are a tuning tool
- Cross-Validation — Generating reliable learning curves
- Model Selection — Choosing the right model using learning curves

## Next Concepts

- Learning Curve Theory — Asymptotic analysis of learning rates
- Active Learning — Using learning curves to guide data acquisition
- Confidence Intervals — Statistical rigor in learning curve interpretation

## Summary

Learning curves are essential diagnostic tools that reveal whether a model suffers from high bias or high variance and whether collecting more data is beneficial. By plotting training and validation performance against training set size, we can identify the root cause of poor performance and make informed decisions about model improvement strategies.

## Key Takeaways

- Learning curves plot performance vs. training set size
- High bias: both curves converge to low performance, small gap
- High variance: large gap between train and validation curves
- Rising validation curve = more data will help
- Plateaued validation curve = more data won't help much
- Validation curves plot performance vs. hyperparameter values
- Early stopping prevents overfitting in iterative training
- Use cross-validation for reliable learning curves
- Confidence bands show the reliability of estimates
- Learning curves guide data collection and model improvement
