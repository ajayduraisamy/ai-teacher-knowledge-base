# Concept: Random Forests

## Concept ID

ML-024

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Tree-Based Models

## Learning Objectives

- Describe the random forest algorithm: bagging + random feature selection
- Explain how ensemble averaging reduces variance while maintaining low bias
- Tune n_estimators and max_features for optimal performance
- Interpret out-of-bag (OOB) error as an unbiased internal evaluation metric
- Extract and interpret feature importance using mean decrease in impurity
- Compare random forests with single decision trees on accuracy and stability

## Prerequisites

- ML-021 Decision Trees
- ML-022 Gini Impurity
- ML-029 Bagging

## Definition

A random forest is an ensemble learning method that constructs a large number of decision trees at training time and outputs the class majority (classification) or mean prediction (regression) of the individual trees. It improves upon bagging by introducing **random feature selection**: at each split, only a random subset of features is considered, decorrelating the trees.

Formally, for a random forest with \(B\) trees:

\[
\hat{f}(\mathbf{x}) = \begin{cases}
\text{mode}\{\hat{f}_b(\mathbf{x})\}_{b=1}^B & \text{(classification)} \\
\frac{1}{B} \sum_{b=1}^B \hat{f}_b(\mathbf{x}) & \text{(regression)}
\end{cases}
\]

Each tree \(\hat{f}_b\) is trained on a bootstrap sample of the original data (sampling \(N\) samples with replacement) and at each split, a random subset of \(m \le p\) features is considered (where \(p\) is the total number of features).

## Intuition

A single decision tree is prone to overfitting — small changes in the training data can produce very different trees. The key insight of random forests is that **averaging many noisy, uncorrelated trees reduces variance without significantly increasing bias**.

Consider a set of \(B\) identically distributed random variables with variance \(\sigma^2\) and pairwise correlation \(\rho\). The variance of the average is:

\[
\text{Var}\left(\frac{1}{B}\sum_{b=1}^B X_b\right) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2
\]

As \(B\) grows, the second term vanishes, but the first term (due to correlation) remains. Bagging alone produces trees that are correlated because they all consider the same strong features. Random forest **decorrelates** the trees by forcing them to consider different random subsets of features, reducing \(\rho\) and thus the overall variance.

## Why This Concept Matters

Random forests are one of the most widely used machine learning methods in practice. They:

- Achieve state-of-the-art accuracy on many tabular datasets with minimal hyperparameter tuning
- Provide built-in feature importance scores
- Handle high-dimensional data gracefully
- Are robust to outliers and missing values
- Scale well to large datasets with parallel implementation

They are a go-to baseline model for any classification or regression task before attempting more advanced methods like gradient boosting.

## Mathematical Explanation

### Algorithm

Given training data \(D = \{(\mathbf{x}_i, y_i)\}_{i=1}^N\) with \(p\) features:

1. For \(b = 1\) to \(B\):
   a. Draw a bootstrap sample \(D_b\) of size \(N\) from \(D\) (with replacement)
   b. Build a decision tree \(\hat{f}_b\) on \(D_b\):
      - At each node, randomly select \(m\) features from the \(p\) total features
      - Choose the best split among those \(m\) features using Gini (classification) or MSE (regression)
      - Grow the tree to maximum depth (no pruning)
2. Return the ensemble \(\{\hat{f}_b\}_{b=1}^B\)

### Out-of-Bag (OOB) Error

Because each tree is trained on a bootstrap sample, roughly \(1 - 1/e \approx 63.2\%\) of the original samples are used per tree. The remaining \(\approx 36.8\%\) are **out-of-bag** — they were not included in that tree's training sample.

For each observation \(\mathbf{x}_i\), we can aggregate predictions from all trees where \(\mathbf{x}_i\) was OOB. The OOB error is the error computed on these predictions. OOB error approximates cross-validation error without requiring a separate validation set.

### Feature Importance

**Mean Decrease in Impurity (MDI)**: For a feature \(X_j\), sum the weighted impurity decreases across all nodes where \(X_j\) was used for splitting, averaged over all trees:

\[
\text{Importance}(X_j) = \frac{1}{B} \sum_{b=1}^B \sum_{\text{nodes } n \text{ in tree } b} \mathbb{I}(j_n = j) \cdot \Delta G_n
\]

where \(\Delta G_n\) is the impurity reduction from splitting on feature \(j\) at node \(n\).

Alternative methods include permutation importance (mean decrease in accuracy when a feature's values are shuffled).

### Hyperparameters

| Parameter | Effect |
|-----------|--------|
| n_estimators | More trees → lower variance, diminishing returns. Monitor OOB error. |
| max_features | Lower → more decorrelation, but individual trees become weaker. Default: \(\sqrt{p}\) (classification), \(p/3\) (regression). |
| max_depth | If None, trees grow fully. For high-dimensional data, limiting depth saves memory. |
| min_samples_leaf | Prevents very small leaf nodes. Helps in noisy data. |
| bootstrap | Whether to use bootstrap sampling. If False, all data used for each tree (not standard RF). |
| oob_score | If True, compute OOB R² or accuracy. |
| n_jobs | Parallelization. -1 uses all cores. |

## Code Examples

### Example 1: Random Forest Classifier on Iris

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42, oob_score=True)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"OOB score: {rf.oob_score_:.4f}")
# Output: Test accuracy: 1.0000
# Output: OOB score: 0.9429
```

### Example 2: Random Forest Regression with OOB Error

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_reg = RandomForestRegressor(n_estimators=200, max_features=1/3, oob_score=True, random_state=42, n_jobs=-1)
rf_reg.fit(X_train, y_train)

y_pred = rf_reg.predict(X_test)
print(f"Test RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
print(f"Test R²: {r2_score(y_test, y_pred):.4f}")
print(f"OOB R²: {rf_reg.oob_score_:.4f}")
# Output: Test RMSE: 0.4952
# Output: Test R²: 0.8134
# Output: OOB R²: 0.8012
```

### Example 3: Feature Importance

```python
import pandas as pd
import numpy as np

X, y = load_iris(return_X_y=True)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

feature_names = load_iris().feature_names
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances,
    'std': std
}).sort_values('importance', ascending=False)

print(importance_df)
# Output:            feature  importance       std
# Output: 2  petal length (cm)    0.447311  0.126700
# Output: 3   petal width (cm)    0.413138  0.137163
# Output: 0  sepal length (cm)    0.107656  0.065745
# Output: 1   sepal width (cm)    0.031895  0.033870
```

### Example 4: Effect of n_estimators on OOB Error

```python
import matplotlib.pyplot as plt

rf = RandomForestClassifier(warm_start=True, oob_score=True, max_features='sqrt', random_state=42)
oob_errors = []

for n in range(10, 210, 10):
    rf.n_estimators = n
    rf.fit(X_train, y_train)
    oob_errors.append(1 - rf.oob_score_)

plt.plot(range(10, 210, 10), oob_errors, marker='o')
plt.xlabel('n_estimators')
plt.ylabel('OOB Error')
plt.title('OOB Error vs Number of Trees')
plt.savefig('oob_error.png')
print("Plot saved. OOB error stabilizes around 100 trees.")
# Output: Plot saved. OOB error stabilizes around 100 trees.
```

## Common Mistakes

1. **Setting n_estimators too low**: Fewer than 50 trees often leaves performance on the table. 100–500 is typical. More trees never hurt (just increase computation time and memory).
2. **Tuning max_depth for random forests**: Unlike single trees, random forests benefit from deep trees (low bias). Pruning is not needed because averaging reduces variance. Default `max_depth=None` is often best.
3. **Using default max_features without consideration**: The default \(\sqrt{p}\) works for classification but may be too low for datasets with very few features. Try \(\sqrt{p}\), \(p/3\), and all features.
4. **Not using OOB error**: OOB error is a free internal validation metric. Set `oob_score=True` to monitor performance without cross-validation.
5. **Misinterpreting feature importance**: MDI importance can be biased toward high-cardinality features. Permutation importance is more reliable but more expensive.
6. **Ignoring warm_start**: When tuning n_estimators, use `warm_start=True` to add trees incrementally instead of retraining from scratch.

## Interview Questions

### Beginner

1. What is a random forest? How does it differ from a single decision tree?
2. What is bootstrap sampling and why is it used in random forests?
3. What is the purpose of max_features in a random forest?
4. How does a random forest make a prediction for regression vs classification?
5. What is OOB error and why is it useful?

### Intermediate

1. Explain how random forests reduce variance compared to a single tree.
2. How does the correlation between trees affect random forest performance? How does random feature selection help?
3. How is feature importance calculated in RandomForestClassifier?
4. Compare random forests with bagging. What is the key difference?
5. How would you tune a random forest for a dataset with 10,000 features vs 10 features?

### Advanced

1. Derive the bias-variance decomposition for a random forest. Show how averaging reduces variance.
2. Explain the theoretical advantage of random forests over bagging in terms of tree correlation. Include the formula \(\text{Var}(\bar{f}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2\).
3. Discuss the consistency properties of random forests. Under what conditions are random forests consistent?

## Practice Problems

### Easy

1. Train a RandomForestClassifier on the Wine dataset. Report test accuracy with n_estimators=10, 50, 100, 500.
2. On the Breast Cancer dataset, compute and plot feature importance using RandomForestClassifier.
3. Compare the test accuracy of a single DecisionTreeClassifier vs RandomForestClassifier on the same data.
4. Train a RandomForestRegressor on the Diabetes dataset. Report RMSE and R².
5. Use OOB error to determine a good value of n_estimators for the Iris dataset.

### Medium

1. Use GridSearchCV to tune max_features and n_estimators on the California Housing dataset.
2. Compare models trained with bootstrap=True vs False. How does OOB error change?
3. Implement a function that computes permutation importance for a trained random forest.
4. Train random forests with varying max_features values (\sqrt(p), \log_2(p), p) and compare accuracy on the Wine dataset.
5. Create a parity plot (y_true vs y_pred) for a random forest regressor on California Housing.

### Hard

1. Implement a simplified random forest from scratch (bootstrap sampling + unpruned decision trees + random feature selection) and compare with sklearn.
2. Analyze the bias of MDI feature importance. Create a synthetic dataset where MDI ranks a noise feature highly.
3. Prove that as n_estimators → ∞, the random forest MSE converges to the variance of a single tree times the correlation between trees.

## Solutions

### Easy Solution 1

```python
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

for n in [10, 50, 100, 500]:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"n_estimators={n}: accuracy={acc:.4f}")
# Output: n_estimators=10: accuracy=0.9815
# Output: n_estimators=50: accuracy=0.9815
# Output: n_estimators=100: accuracy=0.9815
# Output: n_estimators=500: accuracy=0.9815
```

### Medium Solution 1

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2', None]
}
rf = RandomForestRegressor(random_state=42)
grid = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV R²: {grid.best_score_:.4f}")
# Output: Best params: {'max_features': 'sqrt', 'n_estimators': 200}
# Output: Best CV R²: 0.8033
```

### Hard Solution 1 (simplified RF)

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample

class SimpleRandomForest:
    def __init__(self, n_estimators=100, max_features='sqrt', max_depth=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n, p = X.shape
        m = int(np.sqrt(p)) if self.max_features == 'sqrt' else p
        for _ in range(self.n_estimators):
            X_boot, y_boot = resample(X, y)
            tree = DecisionTreeRegressor(max_features=m, max_depth=self.max_depth)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X):
        preds = np.array([t.predict(X) for t in self.trees])
        return np.mean(preds, axis=0)

sr = SimpleRandomForest(n_estimators=50)
sr.fit(X_train, y_train)
preds = sr.predict(X_test)
print(f"Simple RF Test R²: {r2_score(y_test, preds):.4f}")
# Output: Simple RF Test R²: 0.7892
```

## Related Concepts

- ML-021 Decision Trees
- ML-029 Bagging
- ML-025 Gradient Boosting
- ML-026 XGBoost
- ML-027 LightGBM
- ML-028 CatBoost

## Next Concepts

- ML-025 Gradient Boosting — sequential ensemble that corrects errors
- ML-026 XGBoost — regularized gradient boosting with optimized implementation

## Summary

Random forests combine bagging with random feature selection to build an ensemble of decorrelated decision trees. By averaging many trees, variance is dramatically reduced while bias stays low. Key hyperparameters are n_estimators (number of trees), max_features (random subset size), and whether to use bootstrap sampling. OOB error provides free internal validation. Feature importance via MDI is a built-in interpretability tool. Random forests are robust, accurate, and require minimal tuning, making them an essential baseline model.

## Key Takeaways

- Random forests = bagging + random feature subset selection.
- Averaging uncorrelated trees reduces variance without increasing bias.
- OOB error approximates cross-validation without a held-out set.
- Feature importance via MDI aggregates impurity decreases across all trees.
- max_features controls the trade-off between tree strength and correlation.
- Random forests are robust to overfitting and perform well out-of-the-box.
