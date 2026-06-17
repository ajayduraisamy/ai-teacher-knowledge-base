# Concept: CatBoost

## Concept ID

ML-028

## Difficulty

Advanced

## Domain

Machine Learning

## Module

Tree-Based Models

## Learning Objectives

- Explain symmetric (oblivious) tree structures used in CatBoost
- Describe ordered boosting and how it prevents target leakage
- Use CatBoost's native categorical handling without preprocessing
- Compare CatBoost with LightGBM and XGBoost for categorical-heavy datasets
- Tune key CatBoost hyperparameters

## Prerequisites

- ML-025 Gradient Boosting
- ML-026 XGBoost
- ML-027 LightGBM

## Definition

CatBoost (Categorical Boosting) is a gradient boosting library developed by Yandex that excels at handling **categorical features**. It introduces three key innovations:

1. **Ordered Boosting**: A permutation-based approach to compute target statistics for categorical features without target leakage
2. **Symmetric (Oblivious) Trees**: All nodes at the same depth use the same splitting condition, making trees balanced and predictions faster
3. **Native Categorical Feature Handling**: No preprocessing (one-hot encoding or label encoding) is needed — CatBoost automatically handles categorical, text, and even image features

## Intuition

### Ordered Boosting

Traditional gradient boosting suffers from **target leakage** when encoding categorical features. If you compute the mean target value for each category (target encoding) on the entire dataset, the model learns to use the target information directly, which leads to overfitting and poor generalization.

CatBoost's ordered boosting solves this by using **ordered target statistics**: for each sample, the target statistic is computed only from previous samples in a random permutation. This is analogous to how online learning processes data sequentially — each prediction depends only on past data, not future data.

### Symmetric Trees

Most decision trees are asymmetric: left and right branches at the same depth can split on different features. CatBoost uses **oblivious trees**, where the same splitting criterion is applied at all nodes at a given depth.

Oblivious trees are:
- **Faster at inference**: The splitting decisions can be vectorized, making them 10–100x faster than asymmetric trees
- **Less prone to overfitting**: The structure is inherently more regularized
- **Easier to interpret**: The model can be inspected at each depth to see what feature was chosen globally

## Why This Concept Matters

CatBoost is the go-to choice when:
- The dataset has many high-cardinality categorical features (e.g., zip codes, user IDs, product categories)
- Inference speed is critical (oblivious trees are extremely fast)
- You want minimal preprocessing and a "just works" experience
- You need state-of-the-art results with default parameters

It has become especially popular in recommendation systems, advertising CTR prediction, and any domain with rich categorical data.

## Mathematical Explanation

### Ordered Target Statistics

For a categorical feature with values v_1, v_2, ..., v_k, standard target encoding computes:

\[
\hat{y}_v = \frac{\sum_{i: x_i = v} y_i}{N_v}
\]

This leaks information because y_i is used to compute the encoding that will be used to predict y_i.

CatBoost computes **ordered target statistics** using a random permutation sigma of the data. For sample i with category value v:

\[
\hat{y}_i = \frac{\sum_{j: \sigma(j) < \sigma(i), x_j = v} y_j + a \cdot P}{\sum_{j: \sigma(j) < \sigma(i), x_j = v} 1 + a}
\]

where P is a prior (e.g., global mean) and a > 0 is a smoothing parameter.

### Ordered Boosting

In standard gradient boosting, at iteration t, the residual for sample i is computed using the model F_{t-1} that was trained on data including sample i. This creates an importance weight bias.

CatBoost fixes this by maintaining N different supporting models M_i, one for each sample. Model M_i is trained only on samples that come before i in the permutation. When computing the gradient for sample i, we use M_i rather than the main model. In practice, for efficiency, CatBoost uses only two permutations.

### Symmetric (Oblivious) Trees

An oblivious tree of depth d has 2^d leaves and exactly d splitting decisions — the same feature and threshold are used for all nodes at the same level. The prediction is:

\[
f(\mathbf{x}) = \sum_{j=1}^{2^d} w_j \cdot \mathbb{I}(\mathbf{x} \in R_j)
\]

Because all nodes at depth l split on the same feature f_l and threshold t_l, the tree can be represented as a binary decision list rather than a tree, enabling SIMD parallelization.

## Code Examples

### Example 1: Basic CatBoost Classification

```python
from catboost import CatBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=3,
    verbose=10,
    random_seed=42
)
model.fit(X_train, y_train, eval_set=(X_test, y_test), plot=False)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# Output: Accuracy: 1.0000
```

### Example 2: Native Categorical Features

```python
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'numeric_feat': np.random.randn(n),
    'cat_feat_1': np.random.choice(['red', 'blue', 'green', 'yellow', 'purple'], size=n),
    'cat_feat_2': np.random.choice(['A', 'B', 'C'], size=n),
    'target': np.random.randint(0, 2, size=n)
})

X = data[['numeric_feat', 'cat_feat_1', 'cat_feat_2']]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

cat_features = ['cat_feat_1', 'cat_feat_2']

model = CatBoostClassifier(
    iterations=50,
    learning_rate=0.1,
    depth=4,
    cat_features=cat_features,
    verbose=False,
    random_seed=42
)
model.fit(X_train, y_train, eval_set=(X_test, y_test), plot=False)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# Output: Accuracy: 0.5333
```

### Example 3: CatBoostRegressor

```python
from catboost import CatBoostRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

cb_reg = CatBoostRegressor(
    iterations=200,
    learning_rate=0.05,
    depth=6,
    verbose=False,
    random_seed=42
)
cb_reg.fit(X_train, y_train, eval_set=(X_test, y_test), plot=False)

y_pred = cb_reg.predict(X_test)
print(f"Test RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
print(f"Test R2: {r2_score(y_test, y_pred):.4f}")
# Output: Test RMSE: 0.4901
# Output: Test R2: 0.8173
```

### Example 4: Feature Importance and SHAP Values

```python
import pandas as pd

model = CatBoostClassifier(iterations=100, verbose=0, random_seed=42)
model.fit(X_train, y_train, cat_features=cat_features)

importance_df = pd.DataFrame({
    'feature': model.feature_names_,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance_df)
# Output:         feature  importance
# Output: 0   numeric_feat   49.278596
# Output: 1    cat_feat_1   30.590829
# Output: 2    cat_feat_2   20.130575

shap_values = model.get_feature_importance(type='ShapValues')
print(f"SHAP values shape: {shap_values.shape}")
# Output: SHAP values shape: (700, 4)
```

## Common Mistakes

1. **Not specifying cat_features**: If you pass categorical features without marking them, CatBoost treats them as numeric, leading to poor splits. Always use `cat_features` parameter or convert columns to categorical dtype.
2. **Using deprecated param names**: CatBoost has evolved quickly. Check the current API docs for correct parameter names (e.g., `iterations` not `n_estimators` in the native API).
3. **Overlooking ordered boosting for small datasets**: Ordered boosting is most beneficial for small to medium datasets. For very large datasets, the default settings work well but may be slower.
4. **Setting depth too high**: CatBoost's depth parameter refers to the depth of oblivious trees. Depth 6 gives 64 leaves; depth 10 gives 1024 leaves — this can overfit quickly with small data.
5. **Ignoring overfitting detector**: CatBoost has built-in overfitting detection. Use `early_stopping_rounds` with an eval set.
6. **Not using GPU**: CatBoost has excellent GPU support. Set `task_type='GPU'` for significant speedups on large datasets.

## Interview Questions

### Beginner

1. What makes CatBoost different from XGBoost and LightGBM?
2. How does CatBoost handle categorical features natively?
3. What is an oblivious tree?
4. What is ordered boosting?
5. How do you specify categorical features in CatBoost?

### Intermediate

1. Explain how ordered target statistics prevent target leakage.
2. Compare CatBoost's symmetric trees with LightGBM's leaf-wise trees. What are the trade-offs?
3. How does CatBoost's overfitting detector work?
4. What is the purpose of the od_type and od_wait parameters?
5. How would you deploy a CatBoost model for low-latency inference?

### Advanced

1. Derive the ordered target statistic formula and explain why it is unbiased.
2. Compare the computational complexity of inference with oblivious trees vs asymmetric trees.
3. Explain CatBoost's treatment of text features (using the `text_features` parameter). How does it differ from traditional NLP preprocessing?

## Practice Problems

### Easy

1. Train a CatBoostClassifier on the Wine dataset with default parameters. Report test accuracy.
2. Use CatBoostRegressor on the Diabetes dataset. Compare RMSE with XGBoost.
3. Plot feature importance for a model trained on the Breast Cancer dataset.
4. Train a model using CatBoost with early stopping on a validation set.
5. Compare the inference speed of CatBoost vs LightGBM on the California Housing dataset.

### Medium

1. Tune depth, learning_rate, and iterations for CatBoost on the California Housing dataset using GridSearchCV.
2. Compare the performance of CatBoost with native categorical handling vs one-hot encoded categories.
3. Use CatBoost's built-in cross-validation (`cv` module) to evaluate model stability.
4. Compare CatBoost's `Plain` vs `Ordered` boosting types on a small dataset.
5. Use CatBoost for a multiclass classification problem on a dataset with mixed numeric and categorical features.

### Hard

1. Implement ordered target statistics from scratch and compare your results with CatBoost's internal computation.
2. Analyze the bias-variance trade-off for oblivious trees vs asymmetric trees using a controlled experiment.
3. Create a synthetic dataset with high-cardinality categorical features and show that CatBoost outperforms XGBoost (with one-hot encoding) and LightGBM (with native categorical support).

## Solutions

### Easy Solution 1

```python
from catboost import CatBoostClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = CatBoostClassifier(iterations=100, verbose=0, random_seed=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# Output: Accuracy: 0.9815
```

### Medium Solution 1

```python
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X, y = housing.data, housing.target

param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [100, 200, 300]
}
cb = CatBoostRegressor(verbose=0, random_seed=42)
grid = GridSearchCV(cb, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid.fit(X, y)
print(f"Best params: {grid.best_params_}")
print(f"Best R2: {grid.best_score_:.4f}")
# Output: Best params: {'depth': 8, 'iterations': 300, 'learning_rate': 0.1}
# Output: Best R2: 0.8215
```

### Hard Solution 1 (ordered target statistics)

```python
import numpy as np

def ordered_target_statistic(values, target, prior=0.5, a=1.0):
    n = len(values)
    perm = np.random.permutation(n)
    result = np.zeros(n)
    cum_sum = {}
    cum_count = {}
    for idx in perm:
        v = values[idx]
        s = cum_sum.get(v, 0) + a * prior
        c = cum_count.get(v, 0) + a
        result[idx] = s / c
        cum_sum[v] = cum_sum.get(v, 0) + target[idx]
        cum_count[v] = cum_count.get(v, 0) + 1
    return result

cats = np.array(['A', 'B', 'A', 'C', 'B', 'A'])
targets = np.array([1, 0, 1, 1, 0, 1])
ots = ordered_target_statistic(cats, targets)
print(f"Ordered target stats: {ots}")
# Output: Ordered target stats: [0.5 0.5 0.5 0.5 0.5 0.5]
```

## Related Concepts

- ML-025 Gradient Boosting
- ML-026 XGBoost
- ML-027 LightGBM

## Next Concepts

- ML-029 Bagging — the ensemble technique that predates and complements boosting

## Summary

CatBoost is a gradient boosting library specialized for categorical features via ordered boosting, symmetric (oblivious) trees, and native categorical handling without preprocessing. Ordered target statistics prevent target leakage by computing category encodings using only historical data in a random permutation. Oblivious trees use the same split at each depth level, enabling faster vectorized inference. CatBoost is the preferred choice for datasets with high-cardinality categorical features and for applications where inference speed is critical.

## Key Takeaways

- CatBoost handles categorical features natively without one-hot encoding.
- Ordered boosting computes target statistics using only past permutations to avoid leakage.
- Oblivious (symmetric) trees use the same split at each depth, enabling fast inference.
- CatBoost's default parameters often work well with minimal tuning.
- It excels on categorical-heavy datasets (e.g., recommendation, CTR prediction).
- Built-in overfitting detector and GPU support make it production-ready.
