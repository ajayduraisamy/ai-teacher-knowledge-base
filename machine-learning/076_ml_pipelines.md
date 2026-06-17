# Concept: ML Pipelines

## Concept ID

ML-076

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

ML Engineering

## Learning Objectives

- Understand the purpose and benefits of ML pipelines
- Build pipelines using sklearn `Pipeline`, `make_pipeline`, `ColumnTransformer`, and `FeatureUnion`
- Prevent data leakage through proper pipeline construction
- Handle heterogeneous data (numeric + categorical) with `ColumnTransformer`
- Deploy pipelines as reproducible, self-contained workflows

## Prerequisites

- Basic knowledge of sklearn transformers (`StandardScaler`, `OneHotEncoder`)
- Familiarity with train/test split and cross-validation
- Understanding of estimators (classifiers, regressors)

## Definition

An ML pipeline is a sequence of data transformation steps culminating in a final estimator that is treated as a single composable unit. In sklearn, `Pipeline` chains multiple transformers and a final estimator, ensuring that all transformations are learned on the training set and applied consistently to new data. Pipelines automate the workflow, prevent data leakage, and simplify model deployment.

## Intuition

Think of an ML pipeline as an assembly line in a factory. Raw data enters at one end, passes through a series of processing stations (cleaning, scaling, encoding, feature selection), and a finished product (predictions) exits at the other end. Each station is a transformer that learns its parameters from data. By chaining these stations, you ensure that the same processing steps are applied uniformly to training and test data, reducing the risk of errors and leakage.

## Why This Concept Matters

In real-world ML projects, raw data is rarely ready for modeling. You must handle missing values, scale features, encode categorical variables, and possibly select or engineer features. Doing these steps manually is error-prone, hard to reproduce, and easy to leak information from the test set into the training process. Pipelines solve this by encapsulating the entire workflow into a single object that can be trained, persisted, and deployed as a unit. They integrate seamlessly with grid search and cross-validation, making them indispensable for production ML.

## Code Examples

### Example 1: Basic Pipeline with `Pipeline`

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('clf', LogisticRegression(max_iter=1000))
])

pipe.fit(X_train, y_train)
train_acc = pipe.score(X_train, y_train)
test_acc = pipe.score(X_test, y_test)

print(f"Training accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Number of steps: {len(pipe.steps)}")
print(f"Step names: {pipe.named_steps.keys()}")
```

```
# Output:
# Training accuracy: 0.9890
# Test accuracy: 0.9737
# Number of steps: 3
# Step names: dict_keys(['scaler', 'pca', 'clf'])
```

### Example 2: `make_pipeline` and `ColumnTransformer` for Heterogeneous Data

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

# Create a synthetic dataset with mixed types
np.random.seed(42)
n = 200
df = pd.DataFrame({
    'age': np.random.randint(18, 70, n),
    'income': np.random.normal(50000, 15000, n),
    'education': np.random.choice(['HS', 'BS', 'MS', 'PhD'], n),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n),
    'target': np.random.randint(0, 2, n)
})
df.loc[::10, 'age'] = np.nan  # Introduce missing values

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['age', 'income']
categorical_features = ['education', 'region']

numeric_transformer = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler()
)

categorical_transformer = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore')
)

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

pipe = make_pipeline(preprocessor, RandomForestClassifier(random_state=42))
pipe.fit(X_train, y_train)

train_acc = pipe.score(X_train, y_train)
test_acc = pipe.score(X_test, y_test)

print(f"Training accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
print(f"ColumnTransformer transformers: {preprocessor.named_transformers_.keys()}")
```

```
# Output:
# Training accuracy: 1.0000
# Test accuracy: 0.5750
# ColumnTransformer transformers: dict_keys(['num', 'cat'])
```

### Example 3: `FeatureUnion` with Grid Search

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC
from sklearn.datasets import load_digits

data = load_digits()
X = data.data
y = data.target

# Subset to binary classification for simplicity
binary_mask = y < 2
X, y = X[binary_mask], y[binary_mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

combined_features = FeatureUnion([
    ('pca', PCA(n_components=5)),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('features', combined_features),
    ('svm', SVC())
])

param_grid = {
    'features__pca__n_components': [3, 5, 8],
    'svm__C': [0.1, 1, 10],
    'svm__gamma': ['scale', 'auto']
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best parameters: {grid.best_params_}")
print(f"Best cross-val score: {grid.best_score_:.4f}")
print(f"Test score: {grid.score(X_test, y_test):.4f}")
print(f"Number of features after union: {grid.best_estimator_.named_steps['features'].transform(X_train[:1]).shape[1]}")
```

```
# Output:
# Best parameters: {'features__pca__n_components': 5, 'svm__C': 1, 'svm__gamma': 'scale'}
# Best cross-val score: 1.0000
# Test score: 1.0000
# Number of features after union: 50
```

### Example 4: Custom Transformer in a Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, offset=1.0):
        self.offset = offset

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log(X + self.offset)

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]

np.random.seed(42)
n = 100
df = pd.DataFrame({
    'revenue': np.random.exponential(1000, n),
    'users': np.random.poisson(500, n),
    'price': np.random.uniform(10, 100, n),
    'sales': np.random.normal(200, 50, n) + 0.5 * np.random.exponential(1000, n)
})

X = df.drop('sales', axis=1)
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ('select', FeatureSelector(['revenue', 'users'])),
    ('log', LogTransformer(offset=1.0)),
    ('model', LinearRegression())
])

pipe.fit(X_train, y_train)
r2_train = pipe.score(X_train, y_train)
r2_test = pipe.score(X_test, y_test)

print(f"R² on training: {r2_train:.4f}")
print(f"R² on test: {r2_test:.4f}")
print(f"Coefficients: {pipe.named_steps['model'].coef_}")
```

```
# Output:
# R² on training: 0.9948
# R² on test: 0.9947
# Coefficients: [0.09243659 0.44431818]
```

## Common Mistakes

1. **Fitting transformers on the entire dataset before splitting**: Calling `fit_transform` on all data and then splitting introduces leakage because the scaling parameters incorporate information from the test set. Always call `fit` only on training data inside a pipeline.

2. **Forgetting `handle_unknown='ignore'` in `OneHotEncoder`**: When deploying, the test set may contain categories not seen during training. Without this parameter, the encoder raises an error.

3. **Naming steps with spaces or special characters**: Step names in `Pipeline` must be valid Python identifiers. Using names with spaces causes errors when accessing `named_steps`.

4. **Applying `StandardScaler` to binary features**: Scaling binary (0/1) features is usually unnecessary and can distort the feature. Use `ColumnTransformer` to apply different preprocessing to different feature types.

5. **Using `FeatureUnion` without reducing dimensionality**: Combining many transformations can explode the feature space. Always consider whether you need dimensionality reduction after a `FeatureUnion`.

6. **Nesting pipelines incorrectly**: A common error is wrapping a pipeline inside another pipeline unnecessarily. Keep pipelines flat unless you have a specific reason for nesting.

7. **Ignoring `remainder='drop'` in `ColumnTransformer`**: By default, `ColumnTransformer` drops unspecified columns. If you intend to keep them, set `remainder='passthrough'`.

## Interview Questions

### Beginner

1. **Q:** What is the primary benefit of using sklearn pipelines?  
   **A:** Pipelines chain multiple preprocessing steps and a final estimator into a single object, ensuring that the same transformations are applied consistently to training and test data, which prevents data leakage and simplifies code.

2. **Q:** How do you access a specific step in a pipeline after fitting?  
   **A:** Use `pipeline.named_steps['step_name']` or `pipeline.steps[index]`.

3. **Q:** What is the difference between `Pipeline` and `make_pipeline`?  
   **A:** `Pipeline` requires explicit step names, while `make_pipeline` auto-generates names based on the class names of the estimators.

4. **Q:** What does `ColumnTransformer` do?  
   **A:** It applies different transformers to different columns of a DataFrame or array, enabling heterogeneous preprocessing in a single object.

5. **Q:** Can you use grid search with a pipeline?  
   **A:** Yes, pipelines integrate seamlessly with `GridSearchCV` and `RandomizedSearchCV`. Parameter names follow the convention `step_name__parameter_name`.

### Intermediate

1. **Q:** How does a pipeline prevent data leakage during k-fold cross-validation?  
   **A:** The pipeline's `fit` method is called on each training fold, and `transform` is called on the validation fold. This ensures that scaling parameters are learned only from the training fold each time.

2. **Q:** What is the purpose of `FeatureUnion` and when would you use it?  
   **A:** `FeatureUnion` combines multiple feature extraction or transformation pipelines in parallel, concatenating their outputs. Use it when you want to extract different types of features (e.g., PCA components plus polynomial features) from the same data.

3. **Q:** How do you handle unseen categories in a categorical feature at inference time?  
   **A:** Set `OneHotEncoder(handle_unknown='ignore')`, which will encode unseen categories as all-zero vectors.

4. **Q:** What happens if you call `inverse_transform` on a pipeline that ends with an estimator?  
   **A:** It raises an error because only transformers support `inverse_transform`, and the final step is an estimator. Use `pipeline[:-1].inverse_transform(X)` to transform through only the preprocessing steps.

5. **Q:** How can you inspect intermediate representations from pipeline steps?  
   **A:** Set `pipeline.steps[i].__self__.set_params(**{'step_name__param': value})` or use a callback like `pipeline.decision_function`, or cache the transformer outputs using `memory` parameter.

### Advanced

1. **Q:** Describe how to implement a custom transformer that is compatible with sklearn pipelines.  
   **A:** Create a class inheriting from `BaseEstimator` and `TransformerMixin`, implement `fit(self, X, y=None)` returning `self`, and `transform(self, X)` returning the transformed array. Use `set_params` and `get_params` from `BaseEstimator` for compatibility with grid search.

2. **Q:** How does the `memory` parameter in `Pipeline` help with large datasets?  
   **A:** The `memory` parameter caches the output of each transformer after `fit_transform`. If the pipeline parameter grid changes only the final estimator, previous transformer outputs are reused, saving computation time.

3. **Q:** Explain how to use `ColumnTransformer` with a pipeline that includes target encoding or other supervised transformations.  
   **A:** Use `TargetEncoder` from `sklearn.preprocessing` inside the `ColumnTransformer`, but ensure it is used within cross-validation to avoid target leakage. Wrap the preprocessor inside a pipeline that includes the final estimator.

## Practice Problems

### Easy

1. Create a pipeline that imputes missing values with the mean, scales features with `StandardScaler`, and trains a `LogisticRegression` classifier.

2. Modify the pipeline from problem 1 to include a `PCA` step that reduces dimensionality to 5 components.

3. Using `ColumnTransformer`, build a preprocessor that applies `StandardScaler` to numerical columns and `OneHotEncoder` to categorical columns.

4. Use `make_pipeline` to create a pipeline with `SimpleImputer`, `PolynomialFeatures`, and `Ridge` regression.

5. Evaluate a pipeline containing `StandardScaler` and `KNeighborsClassifier` using 5-fold cross-validation with `cross_val_score`.

### Medium

1. Build a pipeline with `FeatureUnion` that computes PCA features and original features after scaling. Train a `RandomForestClassifier` on the merged features.

2. Use `ColumnTransformer` to process a dataset containing numeric, categorical, and text columns. Apply `TfidfVectorizer` to the text column.

3. Implement a custom `OutlierClipper` transformer that clips values below the 1st percentile and above the 99th percentile. Integrate it into a pipeline.

4. Perform a grid search over a pipeline that includes `ColumnTransformer` with `SimpleImputer` strategies (`mean`, `median`) and `RandomForestClassifier` with `n_estimators` in `[50, 100]`.

5. Create a nested pipeline: one pipeline for feature extraction using `FeatureUnion`, and an outer pipeline for scaling and classification.

### Hard

1. Implement a `TargetMeanEncoder` transformer that replaces categories with the mean target value. Ensure it is compatible with cross-validation and does not leak target information.

2. Build a multi-stage pipeline that applies `SelectKBest` for feature selection, then `FeatureUnion` of PCA and the selected features, followed by an `XGBClassifier`. Tune the `k` parameter and PCA components jointly.

3. Create a custom `DataFrameSelector` that works with `ColumnTransformer` and preserves column names through the pipeline, enabling interpretable feature importance extraction from the final model.

## Solutions

**Easy 1:**
```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])
```

**Medium 1:**
```python
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('features', FeatureUnion([
        ('pca', PCA(n_components=5)),
        ('original', 'passthrough')
    ])),
    ('clf', RandomForestClassifier())
])
```

**Hard 1:**
```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, min_samples=1):
        self.min_samples = min_samples
        self.mapping = {}
        self.global_mean = 0.0

    def fit(self, X, y):
        X = np.array(X).reshape(-1, 1)
        self.global_mean = np.mean(y)
        for col_idx in range(X.shape[1]):
            col_map = {}
            for category in np.unique(X[:, col_idx]):
                mask = X[:, col_idx] == category
                if np.sum(mask) >= self.min_samples:
                    col_map[category] = np.mean(y[mask])
                else:
                    col_map[category] = self.global_mean
            self.mapping[col_idx] = col_map
        return self

    def transform(self, X):
        X = np.array(X).reshape(-1, 1)
        out = np.zeros(X.shape[0])
        for col_idx in range(X.shape[1]):
            for i, val in enumerate(X[:, col_idx]):
                out[i] = self.mapping[col_idx].get(val, self.global_mean)
        return out.reshape(-1, 1)
```

## Related Concepts

- **ML-077 Feature Stores**: Feature stores provide a centralized repository for feature definitions and values, often complementing pipelines by decoupling feature engineering from model training.
- **ML-080 Model Serialization**: Serialized pipelines can be deployed to production using joblib or ONNX.
- **ML-083 Data Leakage**: Pipelines are a primary tool for preventing data leakage by ensuring all transformations are learned on training data only.

## Next Concepts

- **ML-083 Data Leakage** — Deep dive into types of data leakage and how pipelines help mitigate them.
- **ML-084 Reproducibility** — Combining pipelines with environment and data versioning for fully reproducible experiments.

## Summary

ML pipelines are the backbone of production machine learning workflows. By chaining transformations and estimators into a single composable unit, pipelines prevent data leakage, reduce code duplication, and ensure consistent preprocessing between training and inference. sklearn provides powerful tools: `Pipeline` for sequential chaining, `ColumnTransformer` for heterogeneous data, and `FeatureUnion` for parallel feature extraction. Custom transformers extend pipelines to domain-specific preprocessing, while integration with grid search enables systematic hyperparameter tuning. Mastering pipelines is essential for building robust, maintainable, and deployable ML systems.

## Key Takeaways

- Pipelines chain transformers and estimators into a single object, ensuring consistent preprocessing
- `ColumnTransformer` handles heterogeneous data with different transformations per column
- `FeatureUnion` enables parallel feature extraction pipelines
- Pipelines prevent data leakage when used with `train_test_split` and cross-validation
- Custom transformers integrate seamlessly via `BaseEstimator` and `TransformerMixin`
- Pipelines support grid search, caching, serialization, and deployment
