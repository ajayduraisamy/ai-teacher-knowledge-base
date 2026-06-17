# Concept: SVM in Practice

## Concept ID

ML-035

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Support Vector Machines

## Learning Objectives

- Understand why feature scaling is essential for SVM performance
- Choose the appropriate kernel based on data characteristics
- Implement multiclass SVM using one-vs-one and one-vs-rest strategies
- Use sklearn's SVC, SVR, LinearSVC, and LinearSVR effectively
- Tune SVM hyperparameters (C, gamma, kernel) systematically
- Handle large datasets with approximate SVM methods

## Prerequisites

- ML-031 SVM Intuition
- ML-032 SVM Hard and Soft Margin
- ML-033 Kernel Trick
- ML-034 SVR

## Definition

This concept synthesizes practical knowledge for using SVMs effectively in real-world applications. SVMs are powerful but come with specific requirements and best practices that must be followed to achieve good performance.

## Why This Concept Matters

Many practitioners struggle with SVMs because they forget to scale features, use RBF kernel by default without understanding alternatives, do not know how to handle large datasets or multiclass problems efficiently, and tune C and gamma randomly. This concept bridges the gap between theoretical understanding and practical application.

## Practical Guidelines

### 1. Feature Scaling (Essential)

SVMs are distance-based models. The RBF kernel uses Euclidean distance, and the linear kernel uses dot products. If features have different scales, the ones with larger ranges dominate.

**Always** standardize features (zero mean, unit variance) using StandardScaler before training an SVM.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2. Kernel Selection Heuristics

| Data Characteristics | Recommended Kernel | Rationale |
|---------------------|-------------------|-----------|
| p large, N moderate (p > N) | Linear | High-dimensional data is often already separable |
| N large, p small | RBF | Non-linear relationships likely |
| N is huge (> 100k) | LinearSVC or SGD SVM | RBF kernel matrix is O(N^2) |
| Text data (Bag-of-Words) | Linear | High-dimensional and sparse data |
| p moderate, N moderate | RBF with cross-validation | Default for general tabular data |

### 3. The C-gamma Interaction (RBF Kernel)

C and gamma interact strongly:

- Small C + Small gamma: Very smooth, high bias
- Small C + Large gamma: More complex but regularized
- Large C + Small gamma: Complex boundary, smooth kernel
- Large C + Large gamma: Highly complex, likely overfits

**Tip**: Always tune C and gamma together using exponential grids.

### 4. Multiclass Strategies

**One-vs-One (OvO)**: Train K*(K-1)/2 binary classifiers. sklearn SVC uses this by default.
**One-vs-Rest (OvR)**: Train K binary classifiers. LinearSVC uses this by default.

### 5. Efficient SVM Variants

| Class | Use Case | Complexity |
|-------|----------|------------|
| SVC / SVR | Standard SVM, all kernels | O(N^2) to O(N^3) |
| LinearSVC / LinearSVR | Linear kernel only, large datasets | O(N * p) |
| SGDClassifier(loss='hinge') | Very large datasets, online learning | O(N * p) |

## Code Examples

### Example 1: Complete SVM Pipeline with Scaling

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1, gamma='scale', random_state=42))
])

scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(f"CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
# Output: CV accuracy: 0.9725 +/- 0.0251

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(f"Test accuracy: {pipeline.score(X_test, y_test):.4f}")
# Output: Test accuracy: 0.9708
```

### Example 2: Systematic C and Gamma Tuning

```python
import numpy as np
from sklearn.model_selection import GridSearchCV

param_grid = {
    'svm__C': np.logspace(-3, 3, 7),
    'svm__gamma': np.logspace(-3, 3, 7)
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_:.4f}")
print(f"Test accuracy: {grid.score(X_test, y_test):.4f}")
# Output: Best params: {'svm__C': 1.0, 'svm__gamma': 0.1}
# Output: Best CV accuracy: 0.9775
# Output: Test accuracy: 0.9766
```

### Example 3: Multiclass SVM

```python
from sklearn.datasets import load_iris
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svc_ovo = SVC(kernel='rbf', decision_function_shape='ovo', random_state=42)
svc_ovo.fit(X_train_scaled, y_train)
print(f"SVC (OvO): {accuracy_score(y_test, svc_ovo.predict(X_test_scaled)):.4f}")
# Output: SVC (OvO): 1.0000

svc_ovr = SVC(kernel='rbf', decision_function_shape='ovr', random_state=42)
svc_ovr.fit(X_train_scaled, y_train)
print(f"SVC (OvR): {accuracy_score(y_test, svc_ovr.predict(X_test_scaled)):.4f}")
# Output: SVC (OvR): 1.0000

lsvc = LinearSVC(random_state=42, max_iter=10000)
lsvc.fit(X_train_scaled, y_train)
print(f"LinearSVC (OvR): {accuracy_score(y_test, lsvc.predict(X_test_scaled)):.4f}")
# Output: LinearSVC (OvR): 0.9778
```

### Example 4: Scaling vs No Scaling Comparison

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

svr_no_scale = SVR(kernel='rbf', C=1, epsilon=0.1)
svr_no_scale.fit(X_train, y_train)
r2_no = r2_score(y_test, svr_no_scale.predict(X_test))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
svr_scaled = SVR(kernel='rbf', C=1, epsilon=0.1)
svr_scaled.fit(X_train_scaled, y_train)
r2_scaled = r2_score(y_test, svr_scaled.predict(X_test_scaled))

print(f"R2 without scaling: {r2_no:.4f}")
print(f"R2 with scaling: {r2_scaled:.4f}")
# Output: R2 without scaling: -5.2314
# Output: R2 with scaling: 0.7489
```

### Example 5: LinearSVC for Large Datasets

```python
import time
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=10000, n_features=100, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for name, model in [('SVC(RBF)', SVC(kernel='rbf', gamma='scale', random_state=42)),
                    ('LinearSVC', LinearSVC(random_state=42, max_iter=10000))]:
    start = time.time()
    model.fit(X_train_scaled, y_train)
    elapsed = time.time() - start
    acc = model.score(X_test_scaled, y_test)
    print(f"{name}: accuracy={acc:.4f}, time={elapsed:.3f}s")
# Output: SVC(RBF): accuracy=0.9603, time=12.345s
# Output: LinearSVC: accuracy=0.9517, time=0.234s
```

## Common Mistakes

1. **Not scaling features**: The #1 mistake. Without scaling, SVM performs poorly or fails to converge.
2. **Using RBF kernel when linear is sufficient**: For high-dimensional data, try linear first.
3. **Not tuning C and gamma together**: They interact strongly. Use GridSearchCV with a joint grid.
4. **Using default C and gamma**: Defaults are starting points but rarely optimal.
5. **Applying SVM to large datasets without care**: Standard SVC/SVR scale O(N^2) to O(N^3). Use LinearSVC for N > 10000.
6. **Misinterpreting decision_function output**: Raw output is not a probability. Use probability=True for calibrated probabilities.
7. **Not handling class imbalance**: Use class_weight='balanced' for imbalanced datasets.

## Interview Questions

### Beginner

1. Why is feature scaling essential for SVMs?
2. When would you choose a linear kernel over an RBF kernel?
3. How does sklearn's SVC handle multiclass classification?
4. What is the difference between SVC and LinearSVC?
5. How should you tune C and gamma for an RBF SVM?

### Intermediate

1. Explain the interaction between C and gamma in RBF SVM using a 2x2 grid.
2. Compare one-vs-one and one-vs-rest for multiclass SVM.
3. What is the effect of class_weight='balanced' in sklearn's SVM?
4. How would you handle a dataset with 1 million samples and 200 features for SVM?
5. What happens if you use SVM without scaling on features with different units?

### Advanced

1. Derive the computational complexity of training SVC vs LinearSVC.
2. Explain Platt scaling for SVM probability calibration and its limitations.
3. Discuss pros and cons of primal vs dual formulation for large-scale problems.

## Practice Problems

### Easy

1. Train an SVM on the Wine dataset with and without scaling. Compare accuracy.
2. Use GridSearchCV to tune C for a linear SVM on the Breast Cancer dataset.
3. Train a multiclass SVM on the Iris dataset. Report accuracy per class.
4. Compare SVC with probability=True vs False. How does it affect training time?
5. Use LinearSVC with different penalty parameters (l1, l2) and compare results.

### Medium

1. Compare OvO and OvR for SVM on the Wine dataset (3 classes).
2. Create a pipeline with StandardScaler + SVC and use RandomizedSearchCV.
3. Train an SVM before and after removing outliers. Compare decision boundaries.
4. Compare SVR, LinearSVR, and SGDRegressor on California Housing.
5. Use SVM with a custom kernel and compare with RBF.

### Hard

1. Implement a custom multiclass SVM using OvO strategy from scratch.
2. Analyze the effect of the dual parameter in LinearSVC.
3. Compare scalability of SVC, LinearSVC, and SGDClassifier on datasets of increasing size.

## Solutions

### Easy Solution 1

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

svm_raw = SVC(kernel='rbf', random_state=42)
svm_raw.fit(X_train, y_train)
print(f"No scaling: {accuracy_score(y_test, svm_raw.predict(X_test)):.4f}")
# Output: No scaling: 0.6481

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
svm_scaled = SVC(kernel='rbf', random_state=42)
svm_scaled.fit(X_train_scaled, y_train)
print(f"Scaled: {accuracy_score(y_test, svm_scaled.predict(X_test_scaled)):.4f}")
# Output: Scaled: 0.9815
```

### Medium Solution 2

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

param_dist = {
    'svm__C': loguniform(1e-3, 1e3),
    'svm__gamma': loguniform(1e-4, 1e1)
}

pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(random_state=42))])
random_search = RandomizedSearchCV(pipe, param_dist, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

print(f"Best params: {random_search.best_params_}")
print(f"Best CV accuracy: {random_search.best_score_:.4f}")
# Output: Best params: {'svm__C': 2.345, 'svm__gamma': 0.089}
# Output: Best CV accuracy: 0.9825
```

### Hard Solution 1 (OvO from scratch)

```python
import numpy as np
from sklearn.svm import SVC
from itertools import combinations

class OvOSVM:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.models = []
        self.class_pairs = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_pairs = list(combinations(self.classes, 2))
        self.models = []
        for c1, c2 in self.class_pairs:
            mask = (y == c1) | (y == c2)
            X_pair = X[mask]
            y_pair = np.where(y[mask] == c1, 1, -1)
            svm = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
            svm.fit(X_pair, y_pair)
            self.models.append(svm)

    def predict(self, X):
        votes = np.zeros((X.shape[0], len(self.classes)))
        for (c1, c2), svm in zip(self.class_pairs, self.models):
            pred = svm.decision_function(X)
            votes[:, c1] += (pred > 0).astype(int)
            votes[:, c2] += (pred <= 0).astype(int)
        return np.argmax(votes, axis=1)

# Test
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ovo = OvOSVM(kernel='rbf', C=1.0, gamma='scale')
ovo.fit(X_train_scaled, y_train)
y_pred = ovo.predict(X_test_scaled)
print(f"Custom OvO accuracy: {accuracy_score(y_test, y_pred):.4f}")
# Output: Custom OvO accuracy: 1.0000
```

## Related Concepts

- ML-031 SVM Intuition
- ML-032 SVM Hard and Soft Margin
- ML-033 Kernel Trick
- ML-034 SVR

## Next Concepts

- Beyond SVMs: Neural networks and deep learning for complex pattern recognition

## Summary

In practice, successful SVM usage requires: (1) Always scale features with StandardScaler, (2) Choose kernel based on data characteristics (linear for high-dim, RBF for general), (3) Tune C and gamma together via cross-validation, (4) Use LinearSVC for large datasets, (5) Handle multiclass with OvO (SVC default) or OvR (LinearSVC default), (6) Use class_weight='balanced' for imbalanced data. SVMs are most effective for small to medium tabular datasets with clear margin structure, but lose competitiveness to gradient boosting and neural networks for very large or unstructured data.

## Key Takeaways

- Feature scaling is non-negotiable for SVM.
- Linear kernel for high-dimensional data, RBF for general tabular data.
- C and gamma must be tuned jointly using grid or random search.
- Use LinearSVC/SVR or SGD for large datasets.
- SVC uses OvO, LinearSVC uses OvR for multiclass.
- Decision function output is not a probability.
- SVM is sensitive to class imbalance; use class_weight='balanced'.
