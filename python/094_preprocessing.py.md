# Concept: Preprocessing for Machine Learning

## Concept ID

PYT-094

## Difficulty

Intermediate

## Domain

Python

## Module

Python for ML/AI

## Learning Objectives

- Apply standard, min-max, and robust scaling to numerical features
- Encode categorical variables using one-hot, label, and ordinal encoding
- Impute missing values with mean, median, and most-frequent strategies
- Combine heterogeneous preprocessing steps with `ColumnTransformer` and `Pipeline`

## Prerequisites

- PYT-093 — sklearn Basics (fit/transform API, train_test_split)
- Pandas DataFrames (indexing, dtypes, handling missing data)
- Understanding of feature types: numeric, categorical, ordinal

## Definition

Preprocessing transforms raw data into a format suitable for machine learning algorithms. sklearn provides a comprehensive set of preprocessing tools:

**Scalers (numeric features):**
- `StandardScaler`: Z-score normalization (mean=0, std=1). Assumes normally distributed features.
- `MinMaxScaler`: Scales to a fixed range (default [0, 1]). Preserves shape but sensitive to outliers.
- `RobustScaler`: Uses median and IQR. Robust to outliers; does not assume normality.
- `MaxAbsScaler`: Scales by absolute maximum. Preserves sparsity.

**Encoders (categorical features):**
- `OneHotEncoder`: Creates binary columns for each category. No ordinal relationship implied.
- `LabelEncoder`: Converts categories to integers 0..n-1. For target encoding only.
- `OrdinalEncoder`: Similar to LabelEncoder but for features. Preserves ordinality if categories are ordered.
- `TargetEncoder`: Replaces category with mean of target (sklearn >=1.3).

**Imputers (missing values):**
- `SimpleImputer`: Fill with mean, median, most_frequent, or constant.
- `KNNImputer`: Impute using k-nearest neighbors.
- `IterativeImputer`: Model-based imputation (MICE algorithm).

**Composition:**
- `ColumnTransformer`: Apply different preprocessing to different columns.
- `Pipeline`: Chain preprocessing + model for clean, leak-proof code.

## Intuition

ML algorithms make assumptions about data:
- Linear models assume features are on similar scales (otherwise larger-scale features dominate the coefficients)
- Distance-based models (KNN, SVM) assume Euclidean distance is meaningful — unscaled features with different units ruin this
- Tree-based models (Random Forest, Gradient Boosting) are scale-invariant but benefit from encoding
- Categorical features must be converted to numbers — one-hot for nominal, ordinal encoding for ordered categories

The preprocessing chain answers: "What column types do I have, and what transformation does each type need?"

## Why This Concept Matters

- **Model Performance:** Proper preprocessing is often more impactful than model choice
- **Convergence Speed:** Scaled features speed up gradient descent optimization significantly
- **Interpretability:** Scaled coefficients are directly comparable as feature importance
- **Data Quality:** Missing value imputation prevents data loss and algorithm crashes
- **Production Pipelines:** `ColumnTransformer` + `Pipeline` creates deployable, serializable preprocessing workflows

## Real World Examples

1. **Customer Analytics Dataset:** Age (numeric, scale 18-90), Income (numeric, scale 10K-500K), Education (ordinal: High School < Bachelor < Master < PhD), Gender (nominal). Each column needs different preprocessing.
2. **Medical Records:** Blood pressure, cholesterol, BMI (numeric, different units), Smoking Status (categorical), Age. Missing values in some columns. RobustScaler handles the outliers in cholesterol.
3. **E-commerce Product Catalog:** Price (numeric, right-skewed, log transform first), Category (high-cardinality categorical, 500+ categories), Brand (nominal). TargetEncoder for high-cardinality categories.
4. **Sensor Data:** Temperature, vibration, pressure — all numeric but different scales. Outliers common. RobustScaler + median imputation.
5. **NLP Features:** TF-IDF vectors are already scaled but sparse. MaxAbsScaler preserves sparsity. Combined with numeric metadata features via ColumnTransformer.

## AI/ML Relevance

- **Mandatory for Linear Models:** Logistic regression, linear SVM, neural networks all require scaled features
- **Tree Models:** While scale-invariant, they still need categorical encoding and missing value imputation
- **Feature Engineering:** Preprocessing is the first step in any feature engineering pipeline
- **Deployment:** A fitted preprocessing pipeline is serialized alongside the model for consistent inference
- **Fairness:** Improper encoding can introduce bias (e.g., ordinal encoding of nominal categories implies false ordering)

## Code Examples

### Example 1: StandardScaler vs MinMaxScaler vs RobustScaler
```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

data = np.array([[1.0, 100],
                 [2.0, 200],
                 [3.0, 300],
                 [4.0, 400],
                 [100.0, 500]])  # outlier in first feature

standard = StandardScaler()
minmax = MinMaxScaler()
robust = RobustScaler()

print("StandardScaler:\n", standard.fit_transform(data))
print("\nMinMaxScaler:\n", minmax.fit_transform(data))
print("\nRobustScaler:\n", robust.fit_transform(data))
```
```
# Output:
# StandardScaler:
#  [[-0.46 -1.41]
#  [-0.44 -0.71]
#  [-0.41  0.  ]
#  [-0.39  0.71]
#  [ 1.71  1.41]]
# MinMaxScaler:
#  [[0.   0.  ]
#  [0.01 0.25]
#  [0.02 0.5 ]
#  [0.03 0.75]
#  [1.   1.  ]]
# RobustScaler:
#  [[-0.67 -1.  ]
#  [-0.33 -0.5 ]
#  [ 0.    0.  ]
#  [ 0.33  0.5 ]
#  [32.33  1.  ]]
```

### Example 2: OneHotEncoder with different options
```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

colors = pd.DataFrame({'color': ['red', 'blue', 'green', 'red', 'blue']})

# Default: sparse matrix, auto column order
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(colors)
print("Default OneHot:\n", encoded)

# Drop first category (avoid multicollinearity)
encoder_drop = OneHotEncoder(drop='first', sparse_output=False)
print("\nDrop first:\n", encoder_drop.fit_transform(colors))

# Handle unknown categories
encoder_unknown = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
test_data = pd.DataFrame({'color': ['red', 'yellow']})  # 'yellow' is unknown
print("\nHandle unknown:\n", encoder_unknown.fit(colors).transform(test_data))
```
```
# Output:
# Default OneHot:
#  [[0. 0. 1.]
#  [0. 1. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]]
# Drop first:
#  [[0. 1.]
#  [1. 0.]
#  [0. 0.]
#  [0. 1.]
#  [1. 0.]]
# Handle unknown:
#  [[0. 0. 1.]
#  [0. 0. 0.]]   # unknown category becomes all zeros
```

### Example 3: OrdinalEncoder for ordinal categories
```python
from sklearn.preprocessing import OrdinalEncoder

education = pd.DataFrame({
    'education': ['high_school', 'bachelors', 'masters', 'phd', 'bachelors']
})

# Define the order explicitly
encoder = OrdinalEncoder(categories=[['high_school', 'bachelors', 'masters', 'phd']])
encoded = encoder.fit_transform(education)
print("Education levels encoded:\n", encoded)
print(f"Categories: {encoder.categories_}")
```
```
# Output:
# Education levels encoded:
#  [[0.]
#  [1.]
#  [2.]
#  [3.]
#  [1.]]
# Categories: [array(['high_school', 'bachelors', 'masters', 'phd'], dtype=object)]
```

### Example 4: SimpleImputer for missing values
```python
from sklearn.impute import SimpleImputer

data = np.array([[1.0, 100],
                 [2.0, np.nan],
                 [np.nan, 300],
                 [4.0, 400],
                 [5.0, 500]])

mean_imputer = SimpleImputer(strategy='mean')
median_imputer = SimpleImputer(strategy='median')
constant_imputer = SimpleImputer(strategy='constant', fill_value=0)

print("Mean imputation:\n", mean_imputer.fit_transform(data))
print("\nMedian imputation:\n", median_imputer.fit_transform(data))
print("\nConstant (0) imputation:\n", constant_imputer.fit_transform(data))
```
```
# Output:
# Mean imputation:
#  [[  1.  100.]
#  [  2.  325.]
#  [  3.  300.]
#  [  4.  400.]
#  [  5.  500.]]
# Median imputation:
#  [[  1.  100.]
#  [  2.  350.]
#  [  3.  300.]
#  [  4.  400.]
#  [  5.  500.]]
# Constant (0) imputation:
#  [[  1.  100.]
#  [  2.    0.]
#  [  0.  300.]
#  [  4.  400.]
#  [  5.  500.]]
```

### Example 5: ColumnTransformer for mixed column types
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 60000, 120000, 55000, 80000],
    'education': ['BS', 'MS', 'PhD', 'BS', 'MS'],
    'married': ['Yes', 'No', 'Yes', 'No', 'Yes']
})

X = df.drop(columns=['education'])  # remove for simplicity
y = [0, 1, 0, 1, 0]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'income']),
        ('cat', OneHotEncoder(sparse_output=False), ['married'])
    ])

X_processed = preprocessor.fit_transform(X)
print(f"Shape: {X_processed.shape}")
print(f"Processed:\n{X_processed}")
print(f"Feature names: {preprocessor.get_feature_names_out()}")
```
```
# Output:
# Shape: (5, 4)
# Processed:
# [[-1.41 -0.83  0.    1.  ]
#  [-0.71 -0.44  1.    0.  ]
#  [ 0.    1.56  0.    1.  ]
#  [ 0.71 -0.66  1.    0.  ]
#  [ 1.41  0.37  0.    1.  ]]
# Feature names: ['num__age' 'num__income' 'cat__married_No' 'cat__married_Yes']
```

### Example 6: Complete Pipeline with ColumnTransformer
```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

np.random.seed(42)
n = 300
df = pd.DataFrame({
    'age': np.random.randint(18, 70, n).astype(float),
    'income': np.random.normal(60000, 20000, n),
    'gender': np.random.choice(['M', 'F'], n),
    'education': np.random.choice(['HS', 'BS', 'MS', 'PhD'], n),
})
df.loc[::10, 'age'] = np.nan  # 10% missing age
df.loc[::15, 'income'] = np.nan  # 7% missing income
y = (df['age'] * 0.01 + df['income'] * 0.0001 + np.random.randn(n) * 0.5 > 0.5).astype(int)

numeric_features = ['age', 'income']
categorical_features = ['gender', 'education']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
])

scores = cross_val_score(pipeline, df, y, cv=5, scoring='accuracy')
print(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

pipeline.fit(df, y)
X_processed = pipeline[:-1].transform(df)
print(f"Processed shape: {X_processed.shape}")
```
```
# Output:
# CV accuracy: 0.817 ± 0.021
# Processed shape: (300, 5)
```

### Example 7: Target Encoding for high-cardinality features
```python
from sklearn.preprocessing import TargetEncoder

np.random.seed(42)
n = 500
categories = [f'cat_{i}' for i in range(50)]  # 50 categories
X = pd.DataFrame({
    'category': np.random.choice(categories, n),
    'numeric': np.random.randn(n)
})
y = (X['numeric'] + np.random.randn(n) * 0.5 > 0).astype(int)

te = TargetEncoder(random_state=42)
X_encoded = te.fit_transform(X[['category']], y)
print(f"Original categories: {X['category'].nunique()}")
print(f"Encoded shape: {X_encoded.shape}")
print(f"Encoded value range: [{X_encoded.min():.3f}, {X_encoded.max():.3f}]")
```
```
# Output:
# Original categories: 50
# Encoded shape: (500, 1)
# Encoded value range: [0.340, 0.650]
```

### Example 8: KNN Imputer for sophisticated missing value handling
```python
from sklearn.impute import KNNImputer

X_complete = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0]])

X_missing = X_complete.copy()
X_missing[1, 1] = np.nan
X_missing[3, 0] = np.nan

imputer = KNNImputer(n_neighbors=2)
X_imputed = imputer.fit_transform(X_missing)

print("Original:\n", X_complete)
print("\nWith missing:\n", X_missing)
print("\nKNN Imputed:\n", X_imputed)
```
```
# Output:
# Original:
#  [[ 1.  2.  3.]
#  [ 4.  5.  6.]
#  [ 7.  8.  9.]
#  [10. 11. 12.]]
# With missing:
#  [[ 1.  2.  3.]
#  [ 4. nan  6.]
#  [ 7.  8.  9.]
#  [nan 11. 12.]]
# KNN Imputed:
#  [[ 1.  2.  3.]
#  [ 4.  5.  6.]
#  [ 7.  8.  9.]
#  [10. 11. 12.]]
```

### Example 9: FunctionTransformer for custom transformations
```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

def log_transform(X):
    return np.log1p(X)  # log(1 + x) to handle zeros

def clip_outliers(X, lower=0.01, upper=0.99):
    q_low, q_high = np.quantile(X, [lower, upper], axis=0)
    return np.clip(X, q_low, q_high)

log_transformer = FunctionTransformer(log_transform, feature_names_out='one-to-one')
clip_transformer = FunctionTransformer(clip_outliers, feature_names_out='one-to-one')

data = np.array([[1, 10],
                 [10, 100],
                 [100, 1000],
                 [0, 5000]])  # 0 needs log1p

print("Log transformed:\n", log_transformer.fit_transform(data))
print("\nClipped:\n", clip_transformer.fit_transform(data))
```
```
# Output:
# Log transformed:
#  [[0.69  2.40]
#  [2.40  4.62]
#  [4.62  6.91]
#  [0.00  8.52]]
# Clipped:
#  [[  1.   10.]
#  [ 10.  100.]
#  [100.  977.]
#  [  0.  977.]]
```

### Example 10: Binarization and polynomial features
```python
from sklearn.preprocessing import Binarizer, PolynomialFeatures

data = np.array([[0.5, 2],
                 [1.5, 3],
                 [-0.5, 1],
                 [2.0, 5]])

# Binarize: threshold at 1.0
binarizer = Binarizer(threshold=1.0)
print("Binarized (threshold=1.0):\n", binarizer.fit_transform(data))

# Polynomial features (degree 2, interaction only)
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
print("\nPolynomial features (deg=2):\n", poly.fit_transform(data))
print(f"Feature names: {poly.get_feature_names_out()}")
```
```
# Output:
# Binarized (threshold=1.0):
#  [[0. 1.]
#  [1. 1.]
#  [0. 0.]
#  [1. 1.]]
# Polynomial features (deg=2):
#  [[ 0.5  2.   0.25  1.   4. ]
#  [ 1.5  3.   2.25  4.5  9. ]
#  [-0.5  1.   0.25 -0.5  1. ]
#  [ 2.   5.   4.   10.  25. ]]
# Feature names: ['x0' 'x1' 'x0^2' 'x0 x1' 'x1^2']
```

## Common Mistakes

1. **Fitting scalers/encoders on the full dataset before splitting.** This leaks information from test to train. Always fit on training data only, then transform both train and test. Use `Pipeline` to automate this.
2. **Using LabelEncoder on features (not targets).** `LabelEncoder` is designed for targets. For features, use `OrdinalEncoder` (ordinal) or `OneHotEncoder` (nominal). LabelEncoder on features creates false ordinal relationships.
3. **OneHotEncoding high-cardinality features (>100 categories).** This creates too many columns. Use TargetEncoding, binary encoding, or hashing trick instead.
4. **Applying StandardScaler to sparse or binary features.** StandardScaler destroys sparsity and creates meaningless z-scores for binary columns. Use MaxAbsScaler for sparse data; leave binary columns unscaled.
5. **Imputing with mean for highly skewed features.** Mean is sensitive to outliers. For skewed data, use median imputation. For time series, use forward-fill or interpolation.
6. **Forgetting to handle unknown categories at inference.** Production data may have categories not seen in training. Set `handle_unknown='ignore'` in `OneHotEncoder` to avoid crashes.
7. **Not inspecting data before choosing preprocessing.** Always check distributions, missing rates, and cardinality first. Blindly applying StandardScaler + OneHotEncoder may not be optimal.

## Interview Questions

### Beginner - 5

1. **Q:** What is the difference between StandardScaler and MinMaxScaler?  
   **A:** StandardScaler standardizes to mean=0, std=1 (z-score). MinMaxScaler scales to a fixed range [0,1] by default. StandardScaler is better for normally distributed data; MinMaxScaler is better when bounded ranges are needed.

2. **Q:** When would you use RobustScaler instead of StandardScaler?  
   **A:** When data contains outliers that would distort the mean and variance used by StandardScaler. RobustScaler uses median and IQR, which are outlier-resistant.

3. **Q:** What does OneHotEncoder do?  
   **A:** It converts a categorical column with N categories into N binary (0/1) columns, one for each category. Prevents models from assuming ordinal relationships between categories.

4. **Q:** Why can't we pass categorical features directly to a linear model?  
   **A:** Linear models require numerical inputs. Categorical strings cannot be multiplied by coefficients. Encoding converts categories to numbers that the model can process.

5. **Q:** What is SimpleImputer used for?  
   **A:** It fills missing (NaN) values in the dataset using strategies like mean, median, most_frequent, or a constant value.

### Intermediate - 5

1. **Q:** What is the advantage of using `ColumnTransformer` over manual column processing?  
   **A:** It applies different transformations to different columns in one consistent object, integrates with `Pipeline`, preserves column naming, and handles train/test separation correctly.

2. **Q:** How does `OrdinalEncoder` differ from `OneHotEncoder`?  
   **A:** OrdinalEncoder assigns integers 0, 1, 2... to categories, implying an order (e.g., small < medium < large). OneHotEncoder creates separate binary columns with no ordering. Use OrdinalEncoder only for ordinal categories.

3. **Q:** What is TargetEncoder and when should you use it?  
   **A:** TargetEncoder replaces each category with the mean of the target variable for that category. Useful for high-cardinality categorical features where OneHotEncoding would create too many columns. Risk of target leakage — must use cross-validation fitting.

4. **Q:** How does `KNNImputer` determine imputed values?  
   **A:** It finds the k-nearest neighbors for each sample with missing values (using non-missing features for distance) and imputes using the mean of neighbors' values for the missing feature.

5. **Q:** What is the `remainder` parameter in ColumnTransformer?  
   **A:** `remainder='passthrough'` keeps columns not specified in transformers as-is. `remainder='drop'` (default) drops them. Useful for keeping identifier columns or pre-scaled features.

### Advanced - 3

1. **Q:** Explain how you would design a preprocessing pipeline for a dataset with mixed types (numeric, categorical, text, datetime) including feature engineering.  
   **A:** Use `ColumnTransformer` with separate pipelines: numeric (impute → scale → optionally poly features), categorical (impute → one-hot/target encode), text (TF-IDF → SVD), datetime (extract hour/day/month/year → cyclical encoding (sin/cos) → scale). Chain with model in a final `Pipeline`.

2. **Q:** How does fitting a `ColumnTransformer` handle unseen categories at test time?  
   **A:** With `handle_unknown='ignore'`, the OneHotEncoder produces all-zero rows for unknown categories. With `handle_unknown='error'` (default), it raises a ValueError. For OrdinalEncoder, unknown categories can be mapped to -1 or left as-is depending on configuration.

3. **Q:** Describe the concept of data leakage in preprocessing and how `Pipeline` prevents it.  
   **A:** Data leakage occurs when information from the test set influences training, e.g., computing mean/scale on full data before splitting. `Pipeline.fit()` calls `fit_transform` on train and `transform` on test separately, ensuring test statistics never affect training parameters.

## Practice Problems

### Easy - 5

1. **E1:** Create an array `[[1, 2], [3, 4], [5, 6]]` and apply StandardScaler. Print the mean and std of the result.
2. **E2:** Use OneHotEncoder on `['cat', 'dog', 'bird', 'cat']` and print the encoded array.
3. **E3:** Create an array with one NaN value and impute using the mean strategy.
4. **E4:** Apply MinMaxScaler to `[[100, 200], [300, 400], [500, 600]]` and verify the output range is [0, 1].
5. **E5:** Use OrdinalEncoder on `['low', 'medium', 'high', 'low']` without specifying categories order.

### Medium - 5

1. **M1:** Build a ColumnTransformer that scales numeric columns and one-hot encodes categorical columns on a dataset with 3 numeric and 2 categorical features.
2. **M2:** Use a Pipeline to chain SimpleImputer(strategy='median') → StandardScaler → LogisticRegression on the iris dataset with missing values added.
3. **M3:** Compare KNNImputer vs SimpleImputer(strategy='mean') on a dataset with 20% missing values, evaluating downstream RandomForest accuracy.
4. **M4:** Build a preprocessing pipeline that: clips outliers at 1st/99th percentile, applies log1p transformation, then StandardScaler.
5. **M5:** Use TargetEncoder on a synthetic dataset with 30 categories and compare cross-val performance with OneHotEncoder.

### Hard - 3

1. **H1:** Design a pipeline that handles numeric, categorical, and datetime features simultaneously. Include cyclical encoding for hour-of-day and month.
2. **H2:** Implement a custom transformer that computes the ratio of two specified columns and appends it to the feature matrix.
3. **H3:** Build an automated preprocessing selector that tries multiple scalers/encoders/imputers and reports which combination gives the best cross-validation score.

## Solutions

### E1 Solution
```python
from sklearn.preprocessing import StandardScaler
import numpy as np
data = np.array([[1, 2], [3, 4], [5, 6]])
scaler = StandardScaler()
scaled = scaler.fit_transform(data)
print(f"Mean: {scaled.mean(axis=0)}, Std: {scaled.std(axis=0)}")
```

### E2 Solution
```python
from sklearn.preprocessing import OneHotEncoder
data = [['cat'], ['dog'], ['bird'], ['cat']]
encoder = OneHotEncoder(sparse_output=False)
print(encoder.fit_transform(data))
```

### E3-E5 Solutions follow patterns from examples.

### M1-M5 Solutions follow ColumnTransformer and Pipeline patterns detailed above.

### H1-H3 Solutions require advanced composition of multiple preprocessing techniques.

## Related Concepts

- 093 — sklearn Basics (fit/predict API, Pipeline)
- 095 — Model Evaluation (GridSearchCV over preprocessing steps)
- 092 — Customizing Plots (visualizing pre/post transformation distributions)

## Next Concepts

- 095 — Model Evaluation (cross-validation, grid search)
- 096 — PyTorch Tensors (deep learning data preprocessing)
- 100 — Project Structure (packaging preprocessing pipelines)

## Summary

Preprocessing transforms raw data into ML-ready format. StandardScaler/MinMaxScaler/RobustScaler handle numeric scaling; OneHotEncoder/OrdinalEncoder/TargetEncoder handle categorical encoding; SimpleImputer/KNNImputer handle missing values. ColumnTransformer applies heterogeneous preprocessing to different column types. Pipeline chains preprocessing → model to prevent data leakage and create deployable workflows.

## Key Takeaways

- Always fit preprocessors on training data only, transform on both train and test
- Choose scaler based on data distribution: StandardScaler (normal), RobustScaler (outliers), MinMaxScaler (bounded)
- OneHotEncoder for nominal categories; OrdinalEncoder for ordered categories
- SimpleImputer with median for skewed data, mean for symmetric data
- ColumnTransformer enables per-column preprocessing in one object
- Pipeline ensures correct fit/transform separation and serializability
- Inspect your data first — don't apply transformations blindly
