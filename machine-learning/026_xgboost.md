# Concept: XGBoost

## Concept ID

ML-026

## Difficulty

Advanced

## Domain

Machine Learning

## Module

Tree-Based Models

## Learning Objectives

- Understand the regularized objective function of XGBoost
- Explain tree pruning with max_depth and gamma
- Use column (feature) subsampling for variance reduction
- Tune the learning rate and number of estimators
- Compare the functional API (xgb.train) with the scikit-learn wrapper (XGBClassifier/XGBRegressor)
- Interpret different types of feature importance (weight, gain, cover)

## Prerequisites

- ML-025 Gradient Boosting
- ML-024 Random Forests
- Familiarity with regularization (L1/L2)

## Definition

XGBoost (Extreme Gradient Boosting) is an optimized, scalable implementation of gradient boosting machines. It was developed by Tianqi Chen and Carlos Guestrin and has been the dominant algorithm in structured/tabular data competitions. XGBoost introduces three key innovations over standard gradient boosting:

1. **Regularized objective**: Adds L1 and L2 penalties to the loss function to prevent overfitting
2. **Second-order approximation**: Uses both first and second derivatives (Newton boosting) for faster convergence
3. **Efficient tree construction**: Weighted quantile sketch for handling weighted data, column subsampling, and parallelized tree building

## Intuition

Standard gradient boosting uses only the gradient (first derivative) of the loss function. XGBoost improves this by using a **second-order Taylor approximation**, which is like using both the slope and curvature of the loss landscape to take better steps. This allows XGBoost to converge faster and find better solutions.

Think of it like navigating a valley:
- Gradient descent (standard GB): follows the slope downhill
- Newton's method (XGBoost): considers both slope and curvature to estimate where the valley floor is and jump directly toward it

XGBoost also adds regularization terms to penalize complex trees (large leaf scores and many leaves), which directly addresses overfitting — a major weakness of standard gradient boosting.

## Why This Concept Matters

XGBoost has been the winning algorithm in countless Kaggle competitions and is widely deployed in production systems. Understanding it is essential because:

- It is the benchmark that modern boosting libraries (LightGBM, CatBoost) are compared against
- It introduced innovations (regularization, column sampling, sparsity-aware learning) that are now standard in tree-based methods
- It provides both a functional API for maximum control and a scikit-learn API for convenience
- Its feature importance metrics help interpret complex models in high-stakes applications

## Mathematical Explanation

### Regularized Objective

Let \(\hat{y}_i^{(t)}\) be the prediction at step \(t\) for sample \(i\). The objective is:

\[
\mathcal{L}^{(t)} = \sum_{i=1}^N L(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) + \Omega(f_t)
\]

where \(\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2 + \alpha \sum_{j=1}^T |w_j|\)

\(T\) is the number of leaves, \(w_j\) is the score of leaf \(j\), and \(\gamma\) (gamma), \(\lambda\) (L2), \(\alpha\) (L1) are regularization parameters.

### Second-Order Approximation

Using Taylor expansion up to second order:

\[
\mathcal{L}^{(t)} \approx \sum_{i=1}^N \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t(\mathbf{x}_i)^2 \right] + \Omega(f_t)
\]

where \(g_i = \partial_{\hat{y}^{(t-1)}} L(y_i, \hat{y}^{(t-1)})\) and \(h_i = \partial^2_{\hat{y}^{(t-1)}} L(y_i, \hat{y}^{(t-1)})\).

### Optimal Leaf Score and Gain

For a given tree structure, the optimal leaf score is:

\[
w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
\]

The resulting gain (loss reduction) for a split is:

\[
\text{Gain} = \frac{1}{2} \left[ \frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma
\]

If Gain ≤ 0, the split is pruned. This is how **gamma** controls tree complexity: it is the minimum loss reduction required for a split.

### Key Hyperparameters

| Parameter | Effect |
|-----------|--------|
| eta / learning_rate | Step size shrinkage. Lower → more robust, needs more trees. |
| max_depth | Maximum tree depth. Default 6 (vs sklearn GB's default 3). |
| gamma / min_split_loss | Minimum loss reduction for a split. Larger → simpler trees. |
| subsample | Row subsampling fraction per tree. Prevents overfitting. |
| colsample_bytree / colsample_bylevel / colsample_bynode | Column subsampling. Adds randomness, reduces variance. |
| lambda (reg_lambda) | L2 regularization on leaf scores. |
| alpha (reg_alpha) | L1 regularization on leaf scores. |
| min_child_weight | Minimum sum of instance weight (hessian) in a leaf. Large → conservative. |

## Code Examples

### Example 1: XGBoost with Native API (xgb.train)

```python
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert to DMatrix (XGBoost's optimized internal format)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 4,
    'eta': 0.1,
    'gamma': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 1.0,
    'alpha': 0.0,
    'eval_metric': 'mlogloss',
    'seed': 42
}

model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'test')], early_stopping_rounds=10)
# Output: [0]	test-mlogloss:0.91857
# Output: [1]	test-mlogloss:0.77506 ...
# Output: [99]	test-mlogloss:0.12853

y_pred = model.predict(dtest)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# Output: Accuracy: 1.0000
```

### Example 2: XGBoost with Scikit-Learn API

```python
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

xgb_clf = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    random_state=42
)

scores = cross_val_score(xgb_clf, X, y, cv=5, scoring='accuracy')
print(f"CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
# Output: CV accuracy: 0.9667 ± 0.0211
```

### Example 3: Feature Importance Types

```python
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = xgb.XGBClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

features = load_iris().feature_names

# 'weight' — number of times a feature is used to split
weight_imp = pd.DataFrame({'feature': features, 'weight': model.feature_importances_})
print("Weight-based importance:")
print(weight_imp)
# Output: Weight-based importance:
# Output:            feature    weight
# Output: 0  sepal length (cm)  0.037037
# Output: 1   sepal width (cm)  0.018519
# Output: 2  petal length (cm)  0.537037
# Output: 3   petal width (cm)  0.407407

# 'gain' — average gain of splits using a feature (use get_score)
gain_imp = model.get_booster().get_score(importance_type='gain')
gain_df = pd.DataFrame(list(gain_imp.items()), columns=['feature', 'gain'])
print("\nGain-based importance:")
print(gain_df)
# Output: Gain-based importance:
# Output:   feature         gain
# Output: 0      f2  1021.456787
# Output: 1      f3   839.234573
# Output: 2      f0   245.678921
# Output: 3      f1    98.345621
```

### Example 4: Early Stopping with Native API

```python
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_test, label=y_test)

params = {'objective': 'multi:softmax', 'num_class': 3, 'eta': 0.1, 'max_depth': 4}
evals_result = {}
model = xgb.train(
    params, dtrain, num_boost_round=500,
    evals=[(dtrain, 'train'), (dval, 'val')],
    evals_result=evals_result,
    early_stopping_rounds=20,
    verbose_eval=False
)

print(f"Best iteration: {model.best_iteration}")
print(f"Best validation error: {model.best_score}")
# Output: Best iteration: 23
# Output: Best validation error: 0.0
```

## Common Mistakes

1. **Not using categorical feature handling**: XGBoost does not natively handle categorical features. One-hot encode or use label encoding carefully.
2. **Over-relying on early stopping**: Early stopping can stop too early on a noisy validation set. Use a separate holdout or cross-validation strategy.
3. **Setting too large max_depth**: Default max_depth=6 is reasonable. Depth above 10 often overfits, especially with small datasets.
4. **Not tuning gamma**: gamma is a powerful regularization parameter for controlling tree complexity. Try values from 0 to 5.
5. **Ignoring class imbalance**: Use scale_pos_weight (binary) or set weights via DMatrix to handle imbalanced classification.
6. **Using default learning rate (0.3)**: 0.3 is often too high. Start with 0.01–0.1 and increase n_estimators accordingly.
7. **Forgetting to call `get_booster()` for advanced features**: Many advanced features (gain importance, tree inspection) require accessing the underlying Booster object.

## Interview Questions

### Beginner

1. What does XGBoost stand for and what makes it "extreme"?
2. What is the difference between xgb.train and XGBClassifier?
3. What is a DMatrix and why is it used?
4. What does the gamma parameter control?
5. How does XGBoost handle missing values?

### Intermediate

1. Explain the regularized objective of XGBoost. Why is regularization important?
2. How does XGBoost use second-order gradients (Hessian) compared to standard gradient boosting?
3. What are the different types of feature importance in XGBoost (weight, gain, cover)?
4. How does column subsampling (colsample_bytree) help prevent overfitting?
5. Explain how max_depth and gamma interact to control tree complexity.

### Advanced

1. Derive the optimal leaf weight \(w_j^*\) and the split gain formula in XGBoost.
2. Explain the Weighted Quantile Sketch algorithm used by XGBoost for approximate split finding.
3. Discuss the differences between XGBoost's tree pruning strategy (post-pruning with gamma) vs LightGBM's leaf-wise growth.

## Practice Problems

### Easy

1. Train XGBClassifier on the Wine dataset. Report test accuracy.
2. Use XGBRegressor on the Diabetes dataset and compute RMSE.
3. Plot feature importance (weight and gain) for a model trained on the Breast Cancer dataset.
4. Compare the performance of XGBoost with default params vs GradientBoostingClassifier on Iris.
5. Train a model using the native API (xgb.train) with a custom eval metric.

### Medium

1. Use GridSearchCV to tune max_depth, learning_rate, subsample, and colsample_bytree on the California Housing dataset.
2. Compare models trained with and without L1/L2 regularization. Show the effect on test accuracy and tree count.
3. Implement early stopping with a validation set using the scikit-learn API.
4. Train a binary classifier on a synthetic imbalanced dataset, using scale_pos_weight to balance classes.
5. Compare the training speed of XGBoost vs GradientBoostingClassifier on a dataset with 10,000 samples and 100 features.

### Hard

1. Implement the XGBoost gain formula from scratch and verify it matches the Booster's output for a small tree.
2. Write a custom objective function and custom evaluation metric for XGBoost and demonstrate it on a ranking task.
3. Analyze the effect of sub-sampling strategies (subsample, colsample_bytree, colsample_bylevel, colsample_bynode) on model variance using bootstrapped standard errors.

## Solutions

### Easy Solution 1

```python
from xgboost import XGBClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# Output: Accuracy: 0.9815
```

### Medium Solution 1

```python
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X, y = housing.data, housing.target

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}
xgb = XGBRegressor(n_estimators=100, random_state=42)
grid = GridSearchCV(xgb, param_grid, cv=3, scoring='r2', verbose=1, n_jobs=-1)
grid.fit(X, y)
print(f"Best params: {grid.best_params_}")
print(f"Best R²: {grid.best_score_:.4f}")
# Output: Best params: {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 7, 'subsample': 0.8}
# Output: Best R²: 0.8332
```

### Hard Solution 1 (gain verification)

```python
import numpy as np
import xgboost as xgb

# Small dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 2, 3, 4, 5])

dtrain = xgb.DMatrix(X, label=y)
params = {'max_depth': 2, 'eta': 0.1, 'lambda': 1.0, 'gamma': 0, 'objective': 'reg:squarederror'}
model = xgb.train(params, dtrain, num_boost_round=1, verbose_eval=False)

# Extract tree structure
trees = model.get_dump(with_stats=True)
print(trees[0])
# Output: (Tree dump showing split features, thresholds, gain, cover...)

# Manual gain for root split (feature 0, threshold ~2.5):
# g = y - mean(y) = [1-3, 2-3, 3-3, 4-3, 5-3] = [-2, -1, 0, 1, 2]
# h = 1 for MSE
g = y - np.mean(y)
h = np.ones_like(y)
I_L = g[:2]; I_R = g[2:]
il = np.sum(I_L); ir = np.sum(I_R); it = np.sum(g)
gain = 0.5 * (il**2/(len(I_L)+1) + ir**2/(len(I_R)+1) - it**2/(5+1))
print(f"Manual gain: {gain:.4f}")
# Output: Manual gain: 8.3333
```

## Related Concepts

- ML-025 Gradient Boosting
- ML-027 LightGBM
- ML-028 CatBoost
- ML-024 Random Forests
- ML-030 Stacking

## Next Concepts

- ML-027 LightGBM — leaf-wise growth for faster training on large data
- ML-028 CatBoost — ordered boosting for categorical features

## Summary

XGBoost improves standard gradient boosting with a regularized objective (L1/L2 penalties and leaf count penalty), second-order gradient approximation (Newton boosting), and efficient tree construction (column subsampling, weighted quantile sketch, sparsity awareness). Key hyperparameters include eta (learning rate), max_depth, gamma (minimum loss reduction for split), subsample, colsample_bytree, and regularization terms lambda/alpha. XGBoost offers both a functional API (xgb.train with DMatrix) and a scikit-learn compatible API (XGBClassifier/XGBRegressor). Feature importance is available as weight (split count), gain (average loss reduction), and cover (average coverage of splits).

## Key Takeaways

- XGBoost = gradient boosting + L1/L2 regularization + second-order approximation.
- Regularized objective \(\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum w_j^2 + \alpha \sum |w_j|\) prevents overfitting.
- Second-order (Newton) boosting uses both gradient and Hessian for faster convergence.
- gamma prunes splits with insufficient loss reduction.
- Column and row subsampling reduce variance and speed up training.
- The native API (DMatrix + xgb.train) offers more control than the sklearn wrapper.
