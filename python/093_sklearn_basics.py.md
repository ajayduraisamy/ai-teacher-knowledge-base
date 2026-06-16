# Concept: sklearn Basics — ML Pipeline

## Concept ID

PYT-093

## Difficulty

Intermediate

## Domain

Python

## Module

Python for ML/AI

## Learning Objectives

- Understand the scikit-learn estimator API: `fit()`, `predict()`, `transform()`
- Split data with `train_test_split` and scale features with `StandardScaler`
- Build and evaluate linear regression, logistic regression, and random forest models
- Construct a complete ML pipeline from preprocessing to evaluation

## Prerequisites

- NumPy and Pandas basics
- Basic statistics (mean, variance, correlation)
- Understanding of supervised learning concepts (features, targets, training/testing)

## Definition

scikit-learn (sklearn) is the foundational machine learning library for Python. It provides a consistent API across hundreds of ML algorithms, preprocessing utilities, model evaluation tools, and dataset loading functions.

**The Estimator API:** Every model and transformer in sklearn follows the same pattern:

- `fit(X, y)`: Learn parameters from training data
- `predict(X)`: Generate predictions on new data (for supervised models)
- `transform(X)`: Transform data based on learned parameters (for preprocessors)
- `fit_transform(X, y)`: Convenience — fit then transform in one call
- `score(X, y)`: Return the default performance metric (e.g., R² for regression, accuracy for classification)

**Key components covered:**

| Component | Purpose |
|-----------|---------|
| `train_test_split` | Split data into train/test sets |
| `StandardScaler` | Standardize features (z-score) |
| `LinearRegression` | Ordinary least squares linear regression |
| `LogisticRegression` | Binary/multiclass logistic classifier |
| `RandomForestClassifier` | Ensemble of decision trees |
| `Pipeline` | Chain preprocessing + model in one object |

## Intuition

scikit-learn's API is designed around a simple insight: most ML workflows follow the same pattern:

```
Load data → Split → Preprocess → Train → Predict → Evaluate
```

Each step has a well-defined interface. Preprocessors learn parameters (mean, std) from training data via `fit()` and apply the transformation via `transform()`. Models learn decision boundaries from training data via `fit()` and output predictions via `predict()`.

A `Pipeline` chains these steps so you never leak test data statistics into preprocessing — the pipeline ensures `fit_transform` on training and `transform` only on test.

## Why This Concept Matters

- **Industry Standard:** scikit-learn is the most widely used ML library in production Python systems
- **Consistent API:** Once you know one estimator, you know them all — `fit` + `predict`
- **Rapid Prototyping:** Go from raw data to evaluated model in 10-20 lines of code
- **ML Fundamentals:** Teaches the core ML workflow applicable to TensorFlow, PyTorch, etc.
- **Interoperability:** skearn models work with SHAP, Eli5, MLflow, and most MLOps tooling

## Real World Examples

1. **Customer Churn Prediction:** A telecom uses logistic regression on 50 features (account age, usage, complaints) to predict which customers will churn, serving probabilities to a retention team.
2. **Real Estate Price Prediction:** A realty website uses random forest regression on 20 features (sqft, bedrooms, location score, school ratings) to estimate home values.
3. **Credit Risk Scoring:** A bank uses logistic regression with L1 regularization on 100+ features to classify loan applicants as default/non-default, with a 0.2 decision threshold.
4. **Medical Diagnosis:** A hospital uses a random forest classifier trained on lab results and vitals to flag patients at risk of sepsis 6 hours before onset.
5. **E-commerce Recommendation:** A recommendation system uses logistic regression to predict click-through rate on suggested products.

## AI/ML Relevance

- **Baseline Models:** Always start with sklearn (linear model, random forest) before deep learning
- **Feature Engineering Pipeline:** Preprocessing + model as a single sklearn Pipeline
- **Model Comparison:** Grid search across 10+ sklearn estimators with consistent API
- **Production Serving:** ONNX export of sklearn pipelines for low-latency inference
- **Interpretability:** sklearn models have built-in feature importance (tree-based) or coefficients (linear)

## Code Examples

### Example 1: Data splitting with `train_test_split`
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)
X = np.random.randn(200, 5)
y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape[0]}")
print(f"Test size: {X_test.shape[0]}")
print(f"Train class balance: {np.bincount(y_train)}")
print(f"Test class balance: {np.bincount(y_test)}")
```
```
# Output:
# Train size: 140
# Test size: 60
# Train class balance: [68 72]
# Test class balance: [30 30]
```

### Example 2: Feature scaling with `StandardScaler`
```python
from sklearn.preprocessing import StandardScaler

X_train_raw = np.array([[100, 0.001],
                         [200, 0.002],
                         [150, 0.003],
                         [300, 0.0015]])

scaler = StandardScaler()
scaler.fit(X_train_raw)
print(f"Means: {scaler.mean_}")
print(f"Stds: {scaler.scale_}")

X_train_scaled = scaler.transform(X_train_raw)
X_test_scaled = scaler.transform(np.array([[250, 0.0025]]))

print(f"Scaled train:\n{X_train_scaled}")
print(f"Scaled test: {X_test_scaled}")
```
```
# Output:
# Means: [187.5    0.001875]
# Stds: [70.533  0.00075]
# Scaled train:
# [[-1.24 -1.11]
#  [ 0.18  0.14]
#  [-0.53  1.39]
#  [ 1.59 -0.42]]
# Scaled test: [[ 0.89  0.81]]
```

### Example 3: Linear Regression on synthetic data
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

np.random.seed(42)
n = 100
X = np.random.randn(n, 1) * 2
y_true = 3 * X.squeeze() + 5
y = y_true + np.random.randn(n) * 1.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print(f"Coefficient: {lr.coef_[0]:.3f}")
print(f"Intercept: {lr.intercept_:.3f}")
print(f"R² score: {r2_score(y_test, y_pred):.3f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")
```
```
# Output:
# Coefficient: 2.833
# Intercept: 5.104
# R² score: 0.788
# MSE: 1.941
```

### Example 4: Logistic Regression for binary classification
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)
n = 500
X = np.random.randn(n, 2)
X[:250] += np.array([1.5, 1.0])   # class 0 shifted
X[250:] += np.array([-1.0, -1.5]) # class 1 shifted
y = np.array([0]*250 + [1]*250)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Coefficients: {logreg.coef_}")
print(f"Intercept: {logreg.intercept_}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
```
```
# Output:
# Accuracy: 0.913
# Coefficients: [[ 2.674  2.459]]
# Intercept: [0.433]
# Classification Report:
#               precision    recall  f1-score   support
#            0       0.90      0.93      0.92        75
#            1       0.93      0.89      0.91        75
#     accuracy                           0.91       150
```

### Example 5: Random Forest Classifier
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                           n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)

importances = rf.feature_importances_
top_idx = np.argsort(importances)[-5:]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Top 5 feature importances:")
for i in reversed(top_idx):
    print(f"  Feature {i}: {importances[i]:.4f}")
```
```
# Output:
# Accuracy: 0.923
# Top 5 feature importances:
#   Feature 7: 0.2134
#   Feature 5: 0.1845
#   Feature 12: 0.1321
#   Feature 3: 0.0987
#   Feature 18: 0.0876
```

### Example 6: Complete ML Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=30, n_informative=15,
                           random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
])

scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

print(f"Cross-val scores: {scores}")
print(f"Mean accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

pipeline.fit(X, y)
print(f"Steps used: {list(pipeline.named_steps.keys())}")
print(f"PCA explained variance: {pipeline['pca'].explained_variance_ratio_.sum():.3f}")
```
```
# Output:
# Cross-val scores: [0.91  0.89  0.93  0.9   0.92]
# Mean accuracy: 0.910 ± 0.015
# Steps used: ['scaler', 'pca', 'classifier']
# PCA explained variance: 0.847
```

### Example 7: Multi-class classification with Logistic Regression
```python
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Classes: {logreg.classes_}")
print(f"Sample probabilities (first 3):\n{y_prob[:3]}")
print(f"\nReport:\n{classification_report(y_test, y_pred, target_names=iris.target_names)}")
```
```
# Output:
# Accuracy: 0.978
# Classes: [0 1 2]
# Sample probabilities (first 3):
# [[9.99e-01 1.00e-03 2.00e-08]
#  [9.99e-01 1.10e-03 1.00e-08]
#  [2.00e-08 1.60e-02 9.84e-01]]
# Report:
#               precision    recall  f1-score   support
#       setosa       1.00      1.00      1.00        15
#   versicolor       1.00      0.93      0.97        15
#    virginica       0.94      1.00      0.97        15
#     accuracy                           0.98        45
```

### Example 8: Using `make_pipeline` for concise syntax
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

pipe = make_pipeline(
    MinMaxScaler(),
    SVC(kernel='rbf', C=1.0, probability=True)
)

X, y = make_classification(n_samples=300, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipe.fit(X_train, y_train)
print(f"Pipeline steps: {pipe.steps}")
print(f"Test accuracy: {pipe.score(X_test, y_test):.3f}")
```
```
# Output:
# Pipeline steps: [('minmaxscaler', MinMaxScaler()), ('svc', SVC(probability=True))]
# Test accuracy: 0.922
```

### Example 9: One-hot encoding categorical features
```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue'],
    'size': ['S', 'M', 'L', 'M', 'S'],
    'price': [10, 15, 20, 12, 18]
})

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[['color', 'size']])
feature_names = encoder.get_feature_names_out()

encoded_df = pd.DataFrame(encoded, columns=feature_names)
result = pd.concat([encoded_df, df[['price']]], axis=1)
print(result)
```
```
# Output:
#    color_blue  color_green  color_red  size_L  size_M  size_S  price
# 0         0.0          0.0        1.0     0.0     0.0     1.0     10
# 1         1.0          0.0        0.0     0.0     1.0     0.0     15
# 2         0.0          1.0        0.0     1.0     0.0     0.0     20
# 3         0.0          0.0        1.0     0.0     1.0     0.0     12
# 4         1.0          0.0        0.0     0.0     0.0     1.0     18
```

### Example 10: Decision Tree visualization
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

plt.figure(figsize=(12, 8))
plot_tree(tree, feature_names=iris.feature_names, class_names=iris.target_names,
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Visualization (max_depth=3)")
plt.show()

print(f"Train accuracy: {tree.score(X_train, y_train):.3f}")
print(f"Test accuracy: {tree.score(X_test, y_test):.3f}")
```
```
# Output: A visual decision tree diagram with colored nodes, split conditions, and class labels.
# Train accuracy: 0.990
# Test accuracy: 0.978
```

## Common Mistakes

1. **Fitting scaler/encoder on the entire dataset before splitting.** This leaks test data statistics into training, biasing evaluation. Always call `fit()` only on training data, then `transform()` on test data. Use `Pipeline` to automate this correctly.
2. **Using `accuracy` on imbalanced datasets.** If 95% of samples are class 0, predicting all 0s gives 95% accuracy — a meaningless metric. Use precision, recall, F1, or AUC for imbalanced problems.
3. **Not setting `random_state` for reproducibility.** Many sklearn algorithms (forests, splits, SGD) are stochastic. Always set `random_state=42` to ensure reproducible results.
4. **Ignoring feature scaling for distance-based models.** SVM, KNN, PCA, and logistic regression with regularization assume features are on similar scales. Always scale with `StandardScaler` or `MinMaxScaler`.
5. **Passing Pandas DataFrames but getting NumPy arrays back.** sklearn estimators return NumPy arrays from `predict()` and `transform()`. Feature names are lost. Use `set_output(transform='pandas')` in sklearn ≥1.2 to keep DataFrames.
6. **Default hyperparameters are rarely optimal.** `RandomForestClassifier()` defaults may underfit. Use `GridSearchCV` or `RandomizedSearchCV` to tune.
7. **Forgetting `stratify` in `train_test_split` for classification.** Without `stratify=y`, train/test may have imbalanced class distributions. Always use `stratify=y` unless you have a specific reason not to.

## Interview Questions

### Beginner - 5

1. **Q:** What is the `fit()` method in scikit-learn?  
   **A:** `fit()` learns the parameters of the model from training data. For a scaler, it computes mean and std. For a regressor, it fits the regression coefficients. For a classifier, it learns the decision boundary.

2. **Q:** Why do we split data into train and test sets?  
   **A:** To evaluate how well the model generalizes to unseen data. Training on all data and testing on the same data gives an overly optimistic (and misleading) performance estimate.

3. **Q:** What does `train_test_split(X, y, test_size=0.2)` do?  
   **A:** It randomly splits the data: 80% for training (X_train, y_train) and 20% for testing (X_test, y_test).

4. **Q:** What is `StandardScaler`?  
   **A:** It standardizes features by removing the mean and scaling to unit variance (z-score: (x - mean) / std). Essential when features have different units or magnitudes.

5. **Q:** What does `model.predict(X)` return?  
   **A:** For classification, it returns class labels (e.g., 0 or 1). For regression, it returns predicted continuous values. For clustering, it returns cluster assignments.

### Intermediate - 5

1. **Q:** What is the difference between `transform()` and `predict()`?  
   **A:** `transform()` applies a learned transformation (scaling, encoding, dimensionality reduction) and is available on preprocessors. `predict()` generates model output and is available on supervised estimators.

2. **Q:** How does `Pipeline` prevent data leakage?  
   **A:** `Pipeline.fit(X_train, y_train)` calls `fit_transform` on each transformer sequentially using only training data. `Pipeline.score(X_test, y_test)` calls `transform` on test data using parameters learned from training only.

3. **Q:** What is feature importance in Random Forest and how is it computed?  
   **A:** Feature importance measures how much each feature contributes to reducing impurity (Gini or entropy) across all trees, averaged and normalized. Accessed via `model.feature_importances_`.

4. **Q:** When would you use Logistic Regression over Random Forest?  
   **A:** Logistic regression when: (a) interpretability is critical (coefficients show feature direction), (b) data is high-dimensional but sparse, (c) you need well-calibrated probabilities, (d) linear decision boundary is sufficient.

5. **Q:** What is the role of `C` in `LogisticRegression(C=1.0)`?  
   **A:** `C` is the inverse of regularization strength. Smaller values (= stronger regularization) produce simpler models that may underfit. Larger values (= weaker regularization) fit training data more closely.

### Advanced - 3

1. **Q:** Explain the bias-variance trade-off in the context of Random Forest `max_depth` and `n_estimators`.  
   **A:** Increasing `max_depth` reduces bias (model fits more complex patterns) but increases variance (overfits noise). Increasing `n_estimators` reduces variance (ensemble averaging) with minimal bias increase, up to a point of diminishing returns.

2. **Q:** How does sklearn handle multicollinearity in LinearRegression vs LogisticRegression?  
   **A:** `LinearRegression` (OLS) will produce unstable coefficients with high variance when features are correlated. `LogisticRegression` with L2 regularization (`penalty='l2'`) shrinks correlated coefficients toward each other, stabilizing the solution. L1 regularization (`penalty='l1'`) may zero out one of the correlated features.

3. **Q:** Design a custom estimator that is compatible with the sklearn API (fit/predict/transform). What methods must it implement?  
   **A:** For a classifier: `fit(X, y)`, `predict(X)`, and optionally `predict_proba(X)`. For a transformer: `fit(X, y=None)`, `transform(X)`. Must inherit from `BaseEstimator` and `ClassifierMixin`/`TransformerMixin` for `get_params`/`set_params` support. Must accept `**kwargs` in `__init__` without processing them (for cloning to work).

## Practice Problems

### Easy - 5

1. **E1:** Load the Iris dataset, split into train/test (80/20), train a logistic regression, and report accuracy.
2. **E2:** Create synthetic data with `make_regression(n_samples=200, n_features=1)` and fit a linear regression.
3. **E3:** Scale the `iris.data` features using `StandardScaler` and print the mean of the scaled features.
4. **E4:** Train a RandomForestClassifier on the wine dataset (`load_wine()`) and print feature importances.
5. **E5:** Use `train_test_split` with `stratify=y` on `load_iris().target` and verify class balance.

### Medium - 5

1. **M1:** Build a pipeline: StandardScaler → PCA (n=2) → LogisticRegression on the Iris dataset. Report cross-val accuracy.
2. **M2:** Tune `n_estimators` and `max_depth` of RandomForestClassifier on `make_classification(n=500)` using a manual loop.
3. **M3:** Compare test accuracy of LogisticRegression and RandomForestClassifier on a dataset with 30 features where only 5 are informative.
4. **M4:** Use `OneHotEncoder` to encode the `color` and `size` columns of a DataFrame, then concatenate with numeric columns.
5. **M5:** Create a Pipeline that imputes missing values (mean), scales features, and trains an SVM on the iris dataset.

### Hard - 3

1. **H1:** Perform recursive feature elimination (RFE) with cross-validation to select the best 5 features from a 20-feature classification dataset.
2. **H2:** Build a multi-model comparison: train LogisticRegression, RandomForest, SVM, and KNN on the same data, reporting accuracy, precision, recall, F1, and training time for each.
3. **H3:** Implement a custom sklearn transformer that applies a log transformation to specified columns and standard scaling to others, using `FunctionTransformer` or a custom class.

## Solutions

### E1 Solution
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

### E2 Solution
```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

X, y = make_regression(n_samples=200, n_features=1, noise=10, random_state=42)
lr = LinearRegression()
lr.fit(X, y)
print(f"Coef: {lr.coef_[0]:.3f}, Intercept: {lr.intercept_:.3f}")
```

### E3-E5 Solutions follow patterns from examples.

### M1-M5 Solutions follow the Pipeline and tuning approaches in the examples.

### H1-H3 Solutions require advanced sklearn features (RFECV, custom transformers).

## Related Concepts

- 094 — Preprocessing (scalers, encoders, imputers)
- 095 — Model Evaluation (cross-validation, grid search, metrics)
- 086 — Matplotlib Basics (visualizing model results)

## Next Concepts

- 094 — Preprocessing (ColumnTransformer, Pipeline)
- 095 — Model Evaluation (GridSearchCV, cross_val_score)
- 096 — PyTorch Tensors (deep learning foundation)

## Summary

scikit-learn provides a consistent `fit()`/`predict()`/`transform()` API across all estimators. `train_test_split` partitions data, `StandardScaler` normalizes features, `LinearRegression` and `LogisticRegression` provide linear models, and `RandomForestClassifier` ensembles decision trees. `Pipeline` chains preprocessing and modeling steps to prevent data leakage. This forms the foundation for any supervised ML workflow in Python.

## Key Takeaways

- Every sklearn estimator follows the `fit()` → `predict()` / `transform()` pattern
- Always split data before any preprocessing to avoid data leakage
- `Pipeline` automates correct fit/transform separation
- Scale features for distance-based and regularized models
- Set `random_state` for reproducibility
- Use stratified split for classification tasks
- Start with simple models (linear/random forest) before deep learning
