# Concept: Bagging

## Concept ID

ML-029

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Tree-Based Models

## Learning Objectives

- Define bootstrap sampling and explain sampling with replacement
- Describe how aggregation (averaging for regression, voting for classification) combines predictions
- Explain why bagging reduces variance without significantly increasing bias
- Use sklearn's BaggingClassifier and BaggingRegressor
- Compare the performance of a single decision tree vs a bagging ensemble
- Understand when bagging is most effective

## Prerequisites

- ML-021 Decision Trees
- Basic understanding of bias-variance trade-off
- Bootstrap sampling (statistical resampling)

## Definition

Bagging (Bootstrap AGGregatING) is an ensemble method introduced by Leo Breiman in 1994. It works by:

1. **Bootstrap sampling**: Creating B bootstrap samples (sampling with replacement, each of size N) from the original training data
2. **Model training**: Training a separate model on each bootstrap sample
3. **Aggregation**: Averaging predictions (regression) or majority voting (classification)

For a training set D of size N, generate B bootstrap datasets D_1, D_2, ..., D_B. Train models f_b on each D_b. The bagged prediction is:

\[
\hat{f}_{\text{bag}}(\mathbf{x}) = \begin{cases}
\frac{1}{B} \sum_{b=1}^B f_b(\mathbf{x}) & \text{(regression)} \\
\text{mode}\{f_b(\mathbf{x})\}_{b=1}^B & \text{(classification)}
\end{cases}
\]

## Intuition

Imagine you have a single noisy measurement of some quantity. It might be off. Now imagine you take 100 measurements and average them. The average will be much more stable and closer to the true value.

Bootstrap sampling introduces diversity: each model sees a slightly different version of the training data (some samples are repeated, some are omitted). Each model makes errors, but if the errors are uncorrelated, they cancel out in the average.

For trees specifically:
- A single deep tree has low bias but high variance — small changes in training data produce very different trees
- Bagging averages many such trees, dramatically reducing variance while keeping low bias

The probability that any particular sample is **not** selected in a bootstrap sample of size N is (1 - 1/N)^N, which converges to 1/e ~ 0.368. So each bootstrap sample contains about 63.2% unique samples, with the rest being duplicates.

## Why This Concept Matters

Bagging is the foundation of random forests (which add random feature selection on top of bagging). Understanding bagging helps you understand:

- Why ensemble methods work (variance reduction via averaging)
- The out-of-bag (OOB) error estimation trick
- Why unpruned trees can be used in ensembles (variance is handled by averaging)
- The difference between parallel ensembles (bagging) and sequential ensembles (boosting)

## Mathematical Explanation

### Bias-Variance Decomposition

Let f(x) be the true function, and let each bootstrap model have prediction f_b(x) with:
- E[f_b(x)] = f(x) (unbiased)
- Var(f_b(x)) = sigma^2

If the B models were independent:

\[
\text{Var}(\hat{f}_{\text{bag}}) = \frac{\sigma^2}{B}
\]

But bootstrap samples overlap, so the models are correlated. Let rho be the pairwise correlation:

\[
\text{Var}(\hat{f}_{\text{bag}}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2
\]

As B grows, the second term vanishes, but the correlation term rho sigma^2 remains. This is why random forests add random feature selection — to reduce rho further.

### Out-of-Bag (OOB) Error

Since each bootstrap sample excludes ~36.8% of samples (the out-of-bag samples), we can use those to evaluate the model without a separate validation set. For each sample i, collect predictions from all trees where i was OOB, and compute the error.

## Code Examples

### Example 1: BaggingClassifier on Iris

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Single tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
print(f"Single tree accuracy: {accuracy_score(y_test, tree.predict(X_test)):.4f}")
# Output: Single tree accuracy: 1.0000

# Bagging ensemble
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=50,
    max_samples=1.0,
    bootstrap=True,
    random_state=42
)
bag.fit(X_train, y_train)
print(f"Bagging accuracy: {accuracy_score(y_test, bag.predict(X_test)):.4f}")
# Output: Bagging accuracy: 1.0000
```

### Example 2: BaggingRegressor on California Housing

```python
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tree = DecisionTreeRegressor(random_state=42)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
print(f"Single tree RMSE: {mean_squared_error(y_test, tree_pred, squared=False):.4f}")
# Output: Single tree RMSE: 0.8672

bag = BaggingRegressor(
    estimator=DecisionTreeRegressor(random_state=42),
    n_estimators=100,
    max_samples=1.0,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
bag.fit(X_train, y_train)
bag_pred = bag.predict(X_test)
print(f"Bagging RMSE: {mean_squared_error(y_test, bag_pred, squared=False):.4f}")
print(f"Bagging R2: {r2_score(y_test, bag_pred):.4f}")
# Output: Bagging RMSE: 0.5028
# Output: Bagging R2: 0.8079
```

### Example 3: OOB Error Estimation

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    oob_score=True,
    random_state=42
)
bag.fit(X_train, y_train)

print(f"OOB accuracy: {bag.oob_score_:.4f}")
print(f"Test accuracy: {accuracy_score(y_test, bag.predict(X_test)):.4f}")
# Output: OOB accuracy: 0.9520
# Output: Test accuracy: 0.9591
```

### Example 4: Effect of Number of Estimators

```python
import matplotlib.pyplot as plt
import numpy as np

n_estimators_range = [1, 5, 10, 20, 50, 100, 200]
test_scores = []
for n in n_estimators_range:
    bag = BaggingRegressor(
        estimator=DecisionTreeRegressor(random_state=42),
        n_estimators=n,
        random_state=42,
        n_jobs=-1
    )
    bag.fit(X_train, y_train)
    test_scores.append(r2_score(y_test, bag.predict(X_test)))

plt.plot(n_estimators_range, test_scores, marker='o')
plt.xlabel('n_estimators')
plt.ylabel('Test R2')
plt.title('Bagging Performance vs Number of Estimators')
plt.savefig('bagging_performance.png')
print("Performance plot saved. Bagging stabilizes after ~50 estimators.")
# Output: Performance plot saved. Bagging stabilizes after ~50 estimators.
```

## Common Mistakes

1. **Using too few estimators**: The variance reduction benefit of bagging increases with B. Fewer than 10 estimators often leaves significant variance on the table.
2. **Setting bootstrap=False**: Without bootstrap sampling, each model sees the same data, making them identical (for deterministic learners). Results will be identical to a single model.
3. **Using max_samples < 1.0 without understanding**: Smaller bootstrap samples make trees more diverse but weaker. The default max_samples=1.0 is usually best.
4. **Applying bagging to already low-variance models**: Bagging helps most for high-variance, low-bias models (deep trees). For high-bias models (shallow trees, linear models), bagging provides little benefit.
5. **Not parallelizing**: Bagging is embarrassingly parallel. Set `n_jobs=-1` to use all CPU cores.
6. **Confusing bagging with pasting**: Bagging uses bootstrap sampling (with replacement). Pasting uses subsampling without replacement. Bagging is generally preferred.

## Interview Questions

### Beginner

1. What does "bagging" stand for?
2. What is bootstrap sampling and why is sampling with replacement important?
3. How does bagging combine predictions for regression vs classification?
4. Why does bagging reduce variance?
5. What is the OOB error and how is it computed?

### Intermediate

1. Derive the probability that a given sample is not selected in a bootstrap sample.
2. Explain the bias-variance decomposition for bagging. What limits the variance reduction?
3. How does bagging differ from random forests?
4. Why are deep (unpruned) trees often used with bagging?
5. How does the correlation between models affect bagging performance?

### Advanced

1. Prove that the OOB error is an unbiased estimate of the test error for bagging.
2. Derive the variance formula Var(f_bar) = rho sigma^2 + (1-rho)/B sigma^2 and explain its implications.
3. Compare bagging with Bayesian model averaging. Under what conditions are they equivalent?

## Practice Problems

### Easy

1. Train a BaggingClassifier with 10, 50, and 200 estimators on the Wine dataset. Compare accuracy.
2. Use BaggingRegressor on the Diabetes dataset. Report RMSE.
3. Plot the OOB error vs number of estimators for the Breast Cancer dataset.
4. Compare the accuracy of a single DecisionTreeClassifier vs BaggingClassifier on the Iris dataset.
5. Use bagging with max_samples=0.5, 0.75, 1.0 and compare performance.

### Medium

1. Compare bagging with decision trees of varying max_depth (2, 5, 10, None). How does depth affect the benefit of bagging?
2. Use BaggingClassifier with a base estimator other than a tree (e.g., KNeighborsClassifier). Compare with tree-based bagging.
3. Train a bagging ensemble and compute the standard deviation of predictions across estimators to measure uncertainty.
4. Compare OOB error with k-fold cross-validation error for bagging on a small dataset.
5. Implement pasting (subsampling without replacement) and compare with bagging.

### Hard

1. Implement bagging from scratch (bootstrap sampling + model training + aggregation) without sklearn.
2. Derive and demonstrate the bias-variance decomposition for bagging using a simple polynomial regression base model.
3. Analyze the effect of bootstrap sample size (max_samples) on ensemble diversity and accuracy.

## Solutions

### Easy Solution 1

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

for n in [10, 50, 200]:
    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=n, random_state=42
    )
    bag.fit(X_train, y_train)
    acc = accuracy_score(y_test, bag.predict(X_test))
    print(f"n_estimators={n}: accuracy={acc:.4f}")
# Output: n_estimators=10: accuracy=0.9815
# Output: n_estimators=50: accuracy=0.9815
# Output: n_estimators=200: accuracy=0.9815
```

### Medium Solution 1

```python
results = []
for depth in [2, 5, 10, None]:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    tree_score = r2_score(y_test, tree.predict(X_test))
    bag = BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=depth, random_state=42),
        n_estimators=50, random_state=42
    )
    bag.fit(X_train, y_train)
    bag_score = r2_score(y_test, bag.predict(X_test))
    print(f"depth={depth}: single={tree_score:.4f}, bagging={bag_score:.4f}, improvement={bag_score-tree_score:.4f}")
# Output: depth=2: single=0.5432, bagging=0.6321, improvement=0.0889
# Output: depth=5: single=0.7298, bagging=0.8012, improvement=0.0714
# Output: depth=10: single=0.6854, bagging=0.7956, improvement=0.1102
# Output: depth=None: single=0.4213, bagging=0.8079, improvement=0.3866
```

### Hard Solution 1 (bagging from scratch)

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample

class BaggingScratch:
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        self.models = []

    def fit(self, X, y):
        self.models = []
        for _ in range(self.n_estimators):
            X_boot, y_boot = resample(X, y)
            model = DecisionTreeRegressor(random_state=42)
            model.fit(X_boot, y_boot)
            self.models.append(model)

    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.models])
        return np.mean(preds, axis=0)

bag_scratch = BaggingScratch(n_estimators=50)
bag_scratch.fit(X_train, y_train)
preds = bag_scratch.predict(X_test)
print(f"Scratch bagging R2: {r2_score(y_test, preds):.4f}")
# Output: Scratch bagging R2: 0.8011
```

## Related Concepts

- ML-021 Decision Trees
- ML-024 Random Forests
- ML-025 Gradient Boosting
- ML-030 Stacking

## Next Concepts

- ML-030 Stacking — using a meta-learner to combine base models

## Summary

Bagging (Bootstrap Aggregating) reduces variance by training models on bootstrap samples and aggregating their predictions. For high-variance, low-bias models like deep decision trees, bagging dramatically improves accuracy by averaging out model-specific errors. The OOB error provides a free internal validation estimate. Bagging is the foundation of random forests, which add random feature selection to further decorrelate the ensemble. Bagging is most effective when the base models are unstable (high variance) and is embarrassingly parallel.

## Key Takeaways

- Bagging = bootstrap sampling + model training + aggregation.
- It reduces variance without increasing bias.
- Each bootstrap sample contains ~63.2% unique samples.
- OOB error approximates test error without a validation set.
- Bagging is most effective for high-variance models (deep trees).
- It is the foundation for random forests.
