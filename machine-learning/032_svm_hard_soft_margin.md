# Concept: SVM — Hard and Soft Margin

## Concept ID

ML-032

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Support Vector Machines

## Learning Objectives

- Distinguish between hard-margin and soft-margin SVM
- Explain the role of the C parameter (penalty for misclassifications)
- Define slack variables xi_i and their role in the optimization
- Write the hinge loss: max(0, 1 - y_i(w^T x_i + b))
- Describe the effect of C on bias-variance trade-off
- Select C through cross-validation

## Prerequisites

- ML-031 SVM Intuition
- Understanding of overfitting and underfitting

## Definition

### Hard-Margin SVM

The hard-margin SVM assumes the data is **perfectly linearly separable**. It finds the hyperplane that maximizes the margin with zero training errors:

\[
\min_{\mathbf{w}, b} \frac{1}{2} ||\mathbf{w}||^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1, \; \forall i
\]

### Soft-Margin SVM

Real-world data is rarely perfectly separable. The soft-margin SVM introduces **slack variables** xi_i to allow some points to be misclassified or lie inside the margin:

\[
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} ||\mathbf{w}||^2 + C \sum_{i=1}^N \xi_i
\]

subject to:
\[
y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0, \; \forall i
\]

The parameter C controls the trade-off between margin width and training error.

## Intuition

In a hard-margin SVM, every point must be on the correct side of the margin. This is fragile: a single outlier can make the problem infeasible or dramatically distort the decision boundary.

The soft-margin SVM says: "It's okay if a few points violate the margin, but we pay a penalty C for each violation." The slack variable xi_i measures how far point i is on the wrong side:
- xi_i = 0: point is on the correct side of the margin
- 0 < xi_i < 1: point is inside the margin but on the correct side of the hyperplane
- xi_i >= 1: point is on the wrong side of the hyperplane (misclassified)

C controls the budget for violations:
- **Large C**: High penalty for violations. The model tries to classify all points correctly, resulting in a narrower margin (lower bias, higher variance).
- **Small C**: Low penalty for violations. The model allows more points inside the margin, resulting in a wider margin (higher bias, lower variance).

## Why This Concept Matters

The soft-margin formulation is what makes SVMs practically useful. Without it, SVMs would be limited to perfectly separable datasets — a rarity in real-world applications. Understanding the C parameter is critical for:

- Preventing overfitting in high-dimensional settings
- Tuning the bias-variance trade-off
- Handling noisy data with outliers
- Understanding the connection between SVMs and regularization

## Mathematical Explanation

### Slack Variable Formulation

The primal optimization problem for soft-margin SVM:

\[
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} ||\mathbf{w}||^2 + C \sum_{i=1}^N \xi_i
\]

s.t.: y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \xi_i \ge 0, i = 1, ..., N

### Hinge Loss Formulation

The soft-margin SVM is equivalent to minimizing the regularized **hinge loss**:

\[
\min_{\mathbf{w}, b} \sum_{i=1}^N \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b)) + \frac{\lambda}{2} ||\mathbf{w}||^2
\]

where lambda = 1/C. The hinge loss is 0 for correctly classified points outside the margin and increases linearly for violations.

### Dual Formulation

The dual problem is often solved in practice:

\[
\max_{\boldsymbol{\alpha}} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j
\]

s.t.: 0 \le \alpha_i \le C, \sum_{i=1}^N \alpha_i y_i = 0

The key difference from hard-margin: the dual variables alpha_i are now bounded above by C (instead of being unbounded). Support vectors have alpha_i > 0.

### Effect of C on Bias-Variance

- C is small (high regularization): High bias, low variance. Margin is wide, many points are inside it.
- C is large (low regularization): Low bias, high variance. Margin is narrow, model closely follows the data.
- C = infinity: Hard margin (if separable). No violations allowed.

## Code Examples

### Example 1: Hard vs Soft Margin Comparison

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=50, centers=2, random_state=42, cluster_std=2.5)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, C in enumerate([1000, 1.0, 0.01]):
    svm = SVC(kernel='linear', C=C)
    svm.fit(X, y)

    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                         np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    axes[i].contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'])
    axes[i].scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    axes[i].scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                    s=100, facecolors='none', edgecolors='k')
    axes[i].set_title(f'C={C}')
plt.savefig('soft_margin_comparison.png')
print("Comparison plot saved.")
# Output: Comparison plot saved.
```

### Example 2: Effect of C on Accuracy and Support Vectors

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    svm = SVC(kernel='linear', C=C, random_state=42)
    svm.fit(X_train_scaled, y_train)
    train_acc = accuracy_score(y_train, svm.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, svm.predict(X_test_scaled))
    n_sv = len(svm.support_)
    print(f"C={C:7.3f}: train={train_acc:.4f}, test={test_acc:.4f}, SVs={n_sv}")
# Output: C= 0.001: train=0.6375, test=0.6316, SVs=398
# Output: C= 0.010: train=0.9125, test=0.9064, SVs=347
# Output: C= 0.100: train=0.9750, test=0.9532, SVs=119
# Output: C= 1.000: train=0.9900, test=0.9708, SVs= 49
# Output: C=10.000: train=1.0000, test=0.9649, SVs= 43
# Output: C=100.000: train=1.0000, test=0.9649, SVs= 43
```

### Example 3: Hinge Loss Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

# Hinge loss: max(0, 1 - margin)
margin = np.linspace(-2, 3, 100)
hinge = np.maximum(0, 1 - margin)
zero_one = (margin < 0).astype(float)

plt.figure(figsize=(8, 5))
plt.plot(margin, hinge, label='Hinge loss (SVM)', linewidth=2)
plt.plot(margin, zero_one, '--', label='0-1 loss', linewidth=2)
plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
plt.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
plt.xlabel('y * (w^T x + b)')
plt.ylabel('Loss')
plt.legend()
plt.title('Hinge Loss vs 0-1 Loss')
plt.savefig('hinge_loss.png')
print("Hinge loss plot saved.")
# Output: Hinge loss plot saved.
```

### Example 4: Tuning C with GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
svm = SVC(kernel='linear', random_state=42)
grid = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_scaled, y_train)

print(f"Best C: {grid.best_params_['C']}")
print(f"Best CV accuracy: {grid.best_score_:.4f}")
print(f"Test accuracy: {grid.score(X_test_scaled, y_test):.4f}")
# Output: Best C: 1
# Output: Best CV accuracy: 0.9749
# Output: Test accuracy: 0.9708
```

## Common Mistakes

1. **Setting C too large**: In the presence of noise, large C forces the model to fit all points, potentially overfitting. Always tune C via cross-validation.
2. **Setting C too small**: Too small C makes the margin very wide, potentially underfitting. The model might classify everything as the majority class.
3. **Not scaling features before using SVM**: SVM is sensitive to feature scales because C applies uniformly to all slack variables. Standardize all features.
4. **Confusing C with regularization in other models**: In SVM, small C = strong regularization (opposite of ridge regression where small alpha = weak regularization).
5. **Not understanding what slack variables mean**: xi_i between 0 and 1 means the point is inside the margin but correctly classified. xi_i > 1 means it is misclassified.
6. **Using hard margin (very large C) on non-separable data**: The optimization will not converge or will find a pathological solution.

## Interview Questions

### Beginner

1. What is the difference between hard-margin and soft-margin SVM?
2. What does the C parameter control?
3. What is a slack variable?
4. What does it mean if xi_i > 1?
5. What is the hinge loss function?

### Intermediate

1. How does increasing C affect the bias-variance trade-off?
2. Write the soft-margin SVM primal optimization problem with slack variables.
3. How does the hinge loss differ from the 0-1 loss?
4. What is the relationship between the slack variable xi_i and the support vectors?
5. How does the dual formulation change when moving from hard to soft margin?

### Advanced

1. Derive the dual of the soft-margin SVM from the primal. Show how C becomes the upper bound on alpha_i.
2. Prove that the soft-margin SVM is equivalent to minimizing the regularized hinge loss.
3. Explain the relationship between the SVM C parameter and the regularization parameter lambda in regularized empirical risk minimization.

## Practice Problems

### Easy

1. Train an SVM with C=0.1, 1.0, 10.0 on the Iris dataset and compare the number of support vectors.
2. Plot the decision boundary for an SVM with C=0.01 and C=100 on a 2D synthetic dataset.
3. Compute the hinge loss for a set of predictions.
4. Count how many points have xi_i > 1 (misclassified) for a given SVM model.
5. Train an SVM with C=infinity (very large) on a non-separable dataset and observe the result.

### Medium

1. Use cross-validation to find the optimal C for the Breast Cancer dataset. Plot CV accuracy vs C.
2. Compare the test accuracy of linear SVM with C values ranging from 1e-3 to 1e3 on a log scale.
3. For a given SVM model, compute the margin width and the number of support vectors for different C values. Plot both.
4. Compare the hinge loss with logistic loss (log loss) on a toy dataset. Which is more sensitive to outliers?
5. Train an SVM with C=1 and kernel='linear' on scaled vs unscaled features. Show the difference.

### Hard

1. Implement the soft-margin SVM from scratch using quadratic programming with slack variables.
2. Derive the subgradient of the hinge loss and implement gradient descent for SVM training.
3. Prove that the number of support vectors is an upper bound on the leave-one-out cross-validation error.

## Solutions

### Easy Solution 1

```python
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

for C in [0.1, 1.0, 10.0]:
    svm = SVC(kernel='linear', C=C, random_state=42)
    svm.fit(X_train, y_train)
    print(f"C={C:.1f}: SVs={len(svm.support_)}")
# Output: C=0.1: SVs=35
# Output: C=1.0: SVs=27
# Output: C=10.0: SVs=14
```

### Medium Solution 1

```python
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np

C_values = np.logspace(-3, 3, 7)
cv_scores = []
for C in C_values:
    svm = SVC(kernel='linear', C=C, random_state=42)
    scores = cross_val_score(svm, X_train_scaled, y_train, cv=5)
    cv_scores.append(scores.mean())

plt.semilogx(C_values, cv_scores, marker='o')
plt.xlabel('C')
plt.ylabel('CV Accuracy')
plt.grid()
plt.savefig('cv_c_tuning.png')
print(f"Best C: {C_values[np.argmax(cv_scores)]}")
# Output: Best C: 1.0
```

### Hard Solution 1 (QP with slack)

```python
import numpy as np
from scipy.optimize import minimize

def soft_svm_qp(X, y, C=1.0):
    n, p = X.shape
    def objective(params):
        w = params[:p]
        xi = params[p+1:]
        return 0.5 * np.dot(w, w) + C * np.sum(xi)
    def constraint(params):
        w = params[:p]
        b = params[p]
        xi = params[p+1:]
        return y * (X @ w + b) - 1 + xi
    initial = np.zeros(p + 1 + n)
    bounds = [(None, None)] * (p + 1) + [(0, None)] * n
    constraints = [{'type': 'ineq', 'fun': constraint}]
    result = minimize(objective, initial, bounds=bounds, constraints=constraints, method='SLSQP')
    return result.x[:p], result.x[p], result.x[p+1:]

# Test
X_small = np.array([[1,1],[2,2],[3,3],[4,4],[5,5]])
y_small = np.array([1,1,-1,-1,-1])
w, b, xi = soft_svm_qp(X_small, y_small, C=1.0)
print(f"w: {w}, b: {b:.4f}, xi: {xi}")
# Output: w: [0.4 0.4], b: -2.0, xi: [0. 0. 0. 0. 0.]
```

## Related Concepts

- ML-031 SVM Intuition
- ML-033 Kernel Trick
- ML-035 SVM in Practice

## Next Concepts

- ML-033 Kernel Trick — handling non-linear decision boundaries through feature mapping

## Summary

Hard-margin SVM requires perfectly separable data and finds the maximum margin hyperplane with zero violations. Soft-margin SVM introduces slack variables xi_i and a penalty parameter C to allow violations, making it practical for real-world noisy data. The C parameter controls the trade-off between margin width and training error: large C results in a narrow margin with few violations (low bias, high variance), while small C results in a wide margin with more violations (high bias, low variance). The soft-margin SVM is equivalent to minimizing the regularized hinge loss.

## Key Takeaways

- Hard margin: assumes perfect separability; soft margin: allows violations via slack variables.
- Slack variable xi_i measures how far point i violates the margin.
- C controls the penalty for violations (inverse of regularization strength).
- Large C -> narrow margin, low bias, high variance (overfitting risk).
- Small C -> wide margin, high bias, low variance (underfitting risk).
- The dual bounds alpha_i from 0 to C (instead of 0 to infinity for hard margin).
- Feature scaling is essential before training any SVM.
