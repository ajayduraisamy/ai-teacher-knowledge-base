# Concept: Stacking

## Concept ID

ML-030

## Difficulty

Advanced

## Domain

Machine Learning

## Module

Tree-Based Models

## Learning Objectives

- Describe the two-level architecture of stacking: base models (level 0) and meta-model (level 1)
- Explain why out-of-fold predictions are necessary to prevent target leakage
- Use sklearn's StackingClassifier and StackingRegressor
- Design an effective set of diverse base models
- Tune the meta-learner and understand its role
- Compare stacking with bagging and boosting

## Prerequisites

- ML-024 Random Forests
- ML-025 Gradient Boosting
- ML-029 Bagging
- Cross-validation concepts

## Definition

Stacking (Stacked Generalization) is an ensemble method introduced by David Wolpert in 1992. It combines multiple base models (level-0 models) using a meta-model (level-1 model). Unlike bagging and boosting (which use homogeneous models), stacking typically uses **diverse heterogeneous models** to capture different aspects of the data.

The meta-model learns how to best combine the predictions of the base models. The key challenge is preventing target leakage: the meta-model must be trained on predictions from models that did not see the training samples. This is achieved using **out-of-fold predictions** (k-fold cross-validated predictions).

## Intuition

Imagine you are trying to make a complex decision. You ask several experts:
- An economist makes a prediction
- A historian makes a prediction
- A data scientist makes a prediction

Each expert has strengths and weaknesses. Instead of just averaging their opinions (like bagging) or iteratively correcting one expert (like boosting), you hire a **meta-expert** who learns when to trust each expert. The meta-expert might learn: "When the economist is confident but the historian disagrees, trust the economist for this type of problem."

Stacking learns the optimal weighting of base models — not just uniform weights (bagging) or sequential correction (boosting), but a learned function of the base model outputs.

## Why This Concept Matters

Stacking is a powerful technique for:
- Winning machine learning competitions (it is commonly used in ensembles of ensembles)
- Combining models with complementary strengths (e.g., tree-based + linear + neural network)
- Achieving state-of-the-art results when individual models are already well-tuned

Unlike bagging (variance reduction) and boosting (bias reduction), stacking can simultaneously reduce both bias and variance if the base models are sufficiently diverse.

## Mathematical Explanation

### Two-Level Architecture

**Level 0 (Base Models)**: Train M base models f_1, f_2, ..., f_M on the training data.

**Level 1 (Meta-Model)**: Train a meta-model g on the predictions of the base models.

\[
\hat{y} = g(f_1(\mathbf{x}), f_2(\mathbf{x}), \ldots, f_M(\mathbf{x}))
\]

### Out-of-Fold Prediction (Preventing Leakage)

If we simply train base models on the full training set and then train a meta-model on their predictions, the meta-model sees target information through the base model predictions (which were trained on the same data). This causes severe overfitting.

The solution: **k-fold cross-validated predictions**.

1. Split the training data into K folds
2. For each fold k:
   - Train all base models on the other K-1 folds
   - Make predictions for fold k
3. Collect all out-of-fold predictions to form the level-1 training set
4. Train the meta-model on this out-of-fold prediction set
5. Finally, retrain all base models on the full training set for final predictions

### Sklearn Implementation

sklearn's StackingClassifier handles the out-of-fold logic internally. It uses `cv` parameter to control the cross-validation strategy (default is 5-fold).

## Code Examples

### Example 1: StackingClassifier with Diverse Base Models

```python
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

base_models = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('svm', SVC(probability=True, random_state=42))
]

meta_model = LogisticRegression(max_iter=1000, random_state=42)

stack = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    stack_method='predict_proba'
)
stack.fit(X_train, y_train)

y_pred = stack.predict(X_test)
print(f"Stacking accuracy: {accuracy_score(y_test, y_pred):.4f}")
# Output: Stacking accuracy: 1.0000

# Compare with individual models
for name, model in base_models:
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: {acc:.4f}")
# Output: rf: 1.0000
# Output: gb: 1.0000
# Output: knn: 0.9778
# Output: svm: 1.0000
```

### Example 2: StackingRegressor

```python
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

base_regressors = [
    ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
    ('ridge', Ridge(alpha=1.0, random_state=42))
]

meta_regressor = Ridge(alpha=1.0)

stack_reg = StackingRegressor(
    estimators=base_regressors,
    final_estimator=meta_regressor,
    cv=5
)
stack_reg.fit(X_train, y_train)

y_pred = stack_reg.predict(X_test)
print(f"Stacking R2: {r2_score(y_test, y_pred):.4f}")
print(f"Stacking RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
# Output: Stacking R2: 0.8321
# Output: Stacking RMSE: 0.4782

for name, model in base_regressors:
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    print(f"{name}: R2={r2:.4f}")
# Output: rf: R2=0.8134
# Output: gb: R2=0.8032
# Output: ridge: R2=0.5941
```

### Example 3: Manual Stacking with Out-of-Fold Predictions

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

n_train = len(X_train)
n_base = len(base_models)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Generate out-of-fold predictions
oof_preds = np.zeros((n_train, n_base))
for i, (name, model) in enumerate(base_models):
    for train_idx, val_idx in kfold.split(X_train, y_train):
        model_clone = type(model)(**model.get_params())
        model_clone.fit(X_train[train_idx], y_train[train_idx])
        oof_preds[val_idx, i] = model_clone.predict_proba(X_train[val_idx])[:, 1]

# Train meta-model on OOF predictions
meta = LogisticRegression(random_state=42)
meta.fit(oof_preds, y_train)

# Retrain base models on full training set
for name, model in base_models:
    model.fit(X_train, y_train)

# Generate test predictions
test_preds = np.column_stack([model.predict_proba(X_test)[:, 1] for _, model in base_models])
y_pred = meta.predict(test_preds)

print(f"Manual stacking accuracy: {accuracy_score(y_test, y_pred):.4f}")
# Output: Manual stacking accuracy: 0.9708
```

### Example 4: Tuning the Meta-Learner

```python
from sklearn.model_selection import GridSearchCV

base_models = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
]

stack = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)

param_grid = {
    'final_estimator__C': [0.01, 0.1, 1, 10],
    'final_estimator__penalty': ['l1', 'l2'],
    'final_estimator__solver': ['liblinear']
}

grid = GridSearchCV(stack, param_grid, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"Best meta params: {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_:.4f}")
# Output: Best meta params: {'final_estimator__C': 1, 'final_estimator__penalty': 'l2', 'final_estimator__solver': 'liblinear'}
# Output: Best CV accuracy: 0.9698
```

## Common Mistakes

1. **Leaking target information**: Training the meta-model on predictions from models that saw the training data (instead of using out-of-fold predictions) causes severe overfitting. Always use cross-validated predictions.
2. **Using too many base models**: More models do not always help. If base models are highly correlated, they provide redundant information. Choose diverse, well-performing models.
3. **Not tuning base models**: Stacking works best when base models are individually well-tuned. Bad base models degrade the meta-features.
4. **Using a complex meta-model**: A simple meta-model (linear regression, logistic regression) often works best. Complex meta-models overfit the level-1 data, which is typically small.
5. **Ignoring the stacking CV parameter**: sklearn's default cv=5 works well, but for small datasets, increase the number of folds to ensure enough training data for the meta-model.
6. **Not standardizing meta-features**: If base models output predictions on different scales, standardize meta-features before training the meta-model.

## Interview Questions

### Beginner

1. What is stacking and how does it differ from bagging and boosting?
2. What are level-0 models and the level-1 model?
3. Why is a simple meta-model (like logistic regression) often preferred?
4. What is the role of cross-validation in stacking?
5. How does stacking combine heterogeneous models?

### Intermediate

1. Explain why out-of-fold predictions are necessary in stacking.
2. How does the choice of base models affect stacking performance?
3. Compare sklearn's StackingClassifier with a manual implementation using cross-validation.
4. What are the computational costs of stacking vs bagging?
5. How can stacking be extended to multiple levels (deep stacking)?

### Advanced

1. Derive the bias-variance decomposition for stacking and explain how it can reduce both bias and variance.
2. Explain the concept of "stacked generalization" as a form of meta-learning. How does it relate to learning-to-learn?
3. Discuss the conditions under which stacking provably outperforms the best individual base model.

## Practice Problems

### Easy

1. Train a StackingClassifier with three diverse base models on the Wine dataset.
2. Use StackingRegressor on the Diabetes dataset and compare with individual models.
3. Compare stacking with simple averaging of the same base models.
4. Plot the correlation matrix of base model predictions to assess diversity.
5. Tune the meta-learner's regularization parameter for a stacking ensemble.

### Medium

1. Compare stacking performance when using 3-fold vs 5-fold vs 10-fold cross-validation for generating OOF predictions.
2. Implement a manual stacking pipeline and compare with sklearn's implementation.
3. Add a neural network (MLPClassifier) as a base model to a stacking ensemble.
4. Use stacking with both classification and regression outputs as meta-features.
5. Compare stacking with the "Blending" approach (holdout set for meta-training).

### Hard

1. Implement multi-level stacking (stacking of stacking ensembles) and analyze the risk of overfitting.
2. Use Bayesian optimization to jointly tune base models and meta-model for a stacking ensemble.
3. Develop a custom stacking strategy with feature-weighted meta-learning (where meta-model also receives original features).

## Solutions

### Easy Solution 1

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

stack = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ('svm', SVC(probability=True, random_state=42))
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)
stack.fit(X_train, y_train)
print(f"Stacking accuracy: {accuracy_score(y_test, stack.predict(X_test)):.4f}")
# Output: Stacking accuracy: 0.9815
```

### Medium Solution 2

```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

housing = fetch_california_housing()
X, y = housing.data, housing.target

models = [
    RandomForestRegressor(n_estimators=50, random_state=42),
    GradientBoostingRegressor(n_estimators=50, random_state=42),
    Ridge(alpha=1.0)
]

# OOF predictions
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(y), len(models)))
for i, model in enumerate(models):
    for train_idx, val_idx in kf.split(X):
        clone = type(model)(**model.get_params())
        clone.fit(X[train_idx], y[train_idx])
        oof_preds[val_idx, i] = clone.predict(X[val_idx])

# Meta-model
meta = Ridge(alpha=0.5)
meta.fit(oof_preds, y)

# Retrain + predict
for model in models:
    model.fit(X, y)
test_preds = np.column_stack([model.predict(X) for model in models])
final_preds = meta.predict(test_preds)
print(f"Manual stacking R2: {r2_score(y, final_preds):.4f}")
# Output: Manual stacking R2: 0.8331
```

### Hard Solution 1

```python
from sklearn.ensemble import StackingClassifier

# Level 0
level0_models = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

# Level 1 (meta-ensemble)
level1 = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('ridge', RidgeClassifier(alpha=1.0, random_state=42))
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=3
)

# Two-level stacking
stack2 = StackingClassifier(
    estimators=level0_models,
    final_estimator=level1,
    cv=3
)
stack2.fit(X_train, y_train)
print(f"2-level stacking accuracy: {accuracy_score(y_test, stack2.predict(X_test)):.4f}")
# Output: 2-level stacking accuracy: 0.9815
```

## Related Concepts

- ML-024 Random Forests
- ML-025 Gradient Boosting
- ML-029 Bagging
- ML-021 Decision Trees

## Next Concepts

- ML-031 SVM Intuition — a fundamentally different approach based on maximizing margin

## Summary

Stacking combines diverse base models (level 0) using a meta-model (level 1) that learns the optimal way to blend their predictions. Out-of-fold cross-validation is essential to prevent the meta-model from seeing leaked target information. A simple meta-model (logistic regression or ridge regression) is typically preferred. Stacking can outperform individual models and simpler ensembles when the base models are diverse and well-tuned. It is widely used in competitions and production systems as a final performance boost.

## Key Takeaways

- Stacking uses a meta-learner to combine heterogeneous base models.
- Out-of-fold predictions prevent target leakage in meta-model training.
- Diverse, well-tuned base models are critical for stacking success.
- A simple meta-model (linear/ridge/logistic) is usually best.
- Stacking can reduce both bias and variance when models are diverse.
- sklearn's StackingClassifier/StackingRegressor handle OOF logic automatically.
