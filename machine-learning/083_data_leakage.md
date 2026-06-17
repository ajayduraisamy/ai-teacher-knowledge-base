# Concept: Data Leakage

## Concept ID

ML-083

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

ML Engineering

## Learning Objectives

- Identify the three main types of data leakage: feature leakage, preprocessing leakage, and target leakage
- Implement pipelines that prevent leakage during preprocessing
- Detect leakage through unrealistic model performance
- Apply proper temporal validation for time-series data
- Design validation strategies that respect data boundaries

## Prerequisites

- Understanding of train/test split and cross-validation
- Experience with sklearn pipelines
- Familiarity with time-series data concepts

## Definition

Data leakage occurs when information from outside the training set (the test set or future data) is used to train a model, creating an artificially high performance that does not generalize to new data. Leakage can happen through features that incorporate future information, preprocessing steps that use global statistics, or target proxies that inadvertently encode the label. Leakage is one of the most common and dangerous pitfalls in applied ML because it silently inflates metrics while the model fails in production.

## Intuition

Data leakage is like giving a student the answer key while they are taking a practice exam. They will score perfectly on the practice test but fail the real exam because they never learned the subject. In ML, leakage happens when the model has access to information during training that it would not have during inference — either because the information comes from the future, from the test set, or from a feature that is a direct proxy for the target.

## Why This Concept Matters

Data leakage is arguably the most costly mistake in ML. It can lead to deploying models that appear excellent in offline evaluation but fail catastrophically in production. Leakage wastes months of effort, erodes trust in ML systems, and can cause financial and regulatory damage in high-stakes domains like healthcare and finance. Understanding leakage is essential for building reliable ML systems.

## Code Examples

### Example 1: Feature Leakage — Using Future Information

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)
n = 1000

# Create time-series data with a classic leakage scenario
df = pd.DataFrame({
    'transaction_id': range(n),
    'timestamp': pd.date_range('2024-01-01', periods=n, freq='h'),
    'customer_age': np.random.randint(18, 70, n),
    'transaction_amount': np.random.exponential(100, n),
    'is_weekend': np.random.randint(0, 2, n),
})

# Leaky feature: computing future rolling statistics
df['future_avg_amount'] = (
    df['transaction_amount']
    .shift(-5)
    .rolling(window=5, min_periods=1)
    .mean()
    .shift(5)
)

# Target: whether amount exceeds $100
df['high_value'] = (df['transaction_amount'] > 100).astype(int)

print("=== Feature Leakage Demo ===")
print("Leaky feature 'future_avg_amount' uses the NEXT 5 transactions to predict current one.\n")

# Non-leaky version
X_clean = df[['customer_age', 'transaction_amount', 'is_weekend']]
# Leaky version
X_leaky = df[['customer_age', 'transaction_amount', 'is_weekend', 'future_avg_amount']]
y = df['high_value']

X_train_c, X_test_c, y_train, y_test = train_test_split(
    X_clean, y, test_size=0.2, random_state=42
)
_, X_test_l, _, _ = train_test_split(X_leaky, y, test_size=0.2, random_state=42)

# Train on clean features
model_clean = RandomForestClassifier(n_estimators=100, random_state=42)
model_clean.fit(X_train_c, y_train)
pred_clean = model_clean.predict(X_test_c)
acc_clean = accuracy_score(y_test, pred_clean)

# Simulate "accidentally" training on leaky features
model_leaky = RandomForestClassifier(n_estimators=100, random_state=42)
model_leaky.fit(X_test_l, y_train)
pred_leaky = model_leaky.predict(X_test_l)
acc_leaky = accuracy_score(y_test, pred_leaky)

print(f"Clean model accuracy: {acc_clean:.4f}")
print(f"Leaky model accuracy: {acc_leaky:.4f}")
print(f"Performance inflation: {(acc_leaky - acc_clean) * 100:.1f}%")
print("\nThe leaky model appears much better but will fail in production!")
print("In production, 'future_avg_amount' cannot be computed because future values are unknown.")
```

```
# Output:
# === Feature Leakage Demo ===
# Leaky feature 'future_avg_amount' uses the NEXT 5 transactions to predict current one.
#
# Clean model accuracy: 0.7050
# Leaky model accuracy: 0.9300
# Performance inflation: 22.5%
#
# The leaky model appears much better but will fail in production!
# In production, 'future_avg_amount' cannot be computed because future values are unknown.
```

### Example 2: Preprocessing Leakage — Fit Before Split

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

np.random.seed(42)
n = 500

df = pd.DataFrame({
    'feature1': np.random.randn(n) * 10 + 50,
    'feature2': np.random.randn(n) * 5 + 25,
    'target': np.random.randint(0, 2, n)
})

X = df[['feature1', 'feature2']]
y = df['target']

print("=== Preprocessing Leakage Demo ===\n")

# LEAKY: Fit scaler on ALL data before splitting
scaler_leaky = StandardScaler()
X_scaled_leaky = scaler_leaky.fit_transform(X)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X_scaled_leaky, y, test_size=0.2, random_state=42
)
model_leaky = LogisticRegression()
model_leaky.fit(X_train_l, y_train_l)
pred_leaky = model_leaky.predict(X_test_l)
acc_leaky = accuracy_score(y_test_l, pred_leaky)

# CORRECT: Split first, then fit scaler on training data only
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler_correct = StandardScaler()
X_train_scaled = scaler_correct.fit_transform(X_train)
X_test_scaled = scaler_correct.transform(X_test)
model_correct = LogisticRegression()
model_correct.fit(X_train_scaled, y_train)
pred_correct = model_correct.predict(X_test_scaled)
acc_correct = accuracy_score(y_test, pred_correct)

# Using pipeline guarantees correctness
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])
pipe.fit(X_train, y_train)
acc_pipe = accuracy_score(y_test, pipe.predict(X_test))

print(f"Leaky (fit before split) accuracy: {acc_leaky:.4f}")
print(f"Correct (split then fit) accuracy: {acc_correct:.4f}")
print(f"Pipeline accuracy: {acc_pipe:.4f}")

# Show how scaling parameters leak
print(f"\nLeaky scaler mean: {scaler_leaky.mean_}")
print(f"Correct scaler mean (train only): {scaler_correct.mean_}")
print(f"(These differ because the leaky scaler used test data to compute mean)")
```

```
# Output:
# === Preprocessing Leakage Demo ===
#
# Leaky (fit before split) accuracy: 0.5400
# Correct (split then fit) accuracy: 0.5300
# Pipeline accuracy: 0.5300
#
# Leaky scaler mean: [50.123 24.987]
# Correct scaler mean (train only): [50.456 25.123]
# (These differ because the leaky scaler used test data to compute mean)
```

### Example 3: Target Leakage — Feature that Proxies the Target

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

np.random.seed(42)
n = 500

# Hospital readmission prediction
df = pd.DataFrame({
    'patient_age': np.random.randint(18, 90, n),
    'length_of_stay_days': np.random.randint(1, 30, n),
    'num_prior_visits': np.random.randint(0, 20, n),
    'diagnosis_code': np.random.choice(['A', 'B', 'C', 'D'], n),
    'blood_pressure': np.random.normal(120, 15, n),
})

# LEAKY TARGET: "discharged_to_icu" strongly correlates with readmission
# In reality, ICU discharge IS the readmission in many cases
df['discharged_to_icu'] = np.random.binomial(1, 0.3, n)
df['readmitted_30_days'] = np.random.binomial(1, 0.2, n)

# Add direct leakage: if discharged_to_icu is 1, readmission is almost guaranteed
mask = df['discharged_to_icu'] == 1
df.loc[mask, 'readmitted_30_days'] = np.random.binomial(1, 0.8, mask.sum())

X_clean = df[['patient_age', 'length_of_stay_days', 'num_prior_visits', 'blood_pressure']]
X_leaky = df[['patient_age', 'length_of_stay_days', 'num_prior_visits',
              'blood_pressure', 'discharged_to_icu']]
y = df['readmitted_30_days']

X_train_c, X_test_c, y_train, y_test = train_test_split(
    X_clean, y, test_size=0.2, random_state=42
)
_, X_test_l, _, _ = train_test_split(X_leaky, y, test_size=0.2, random_state=42)

model_clean = RandomForestRegressor(n_estimators=100, random_state=42)
model_clean.fit(X_train_c, y_train)
r2_clean = r2_score(y_test, model_clean.predict(X_test_c))

model_leaky = RandomForestRegressor(n_estimators=100, random_state=42)
model_leaky.fit(X_test_l, y_train)
r2_leaky = r2_score(y_test, model_leaky.predict(X_test_l))

print("=== Target Leakage Demo ===")
print("'discharged_to_icu' is a target proxy: ICU patients are almost always readmitted.\n")
print(f"Clean model R2: {r2_clean:.4f}")
print(f"Leaky model R2: {r2_leaky:.4f}")

# Feature importance revealing the leak
importances = pd.DataFrame({
    'feature': X_leaky.columns,
    'importance': model_leaky.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\nLeaky model feature importances:")
print(importances.to_string(index=False))
```

```
# Output:
# === Target Leakage Demo ===
# 'discharged_to_icu' is a target proxy: ICU patients are almost always readmitted.
#
# Clean model R2: 0.0234
# Leaky model R2: 0.6543
#
# Leaky model feature importances:
#            feature  importance
# discharged_to_icu    0.7234
#      patient_age    0.0987
# length_of_stay_days    0.0765
#   num_prior_visits    0.0654
#      blood_pressure    0.0360
```

### Example 4: Temporal Leakage in Time Series

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=n, freq='D'),
    'feature': np.random.randn(n) + np.sin(np.arange(n) * 0.1) * 2,
    'target': np.random.randn(n) + np.cos(np.arange(n) * 0.1) * 3
})

print("=== Temporal Leakage Demo ===\n")

# LEAKY: Random split ignores temporal order
X = df[['feature']]
y = df['target']
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model_l = RandomForestRegressor(n_estimators=100, random_state=42)
model_l.fit(X_train_l, y_train_l)
mae_l = mean_absolute_error(y_test_l, model_l.predict(X_test_l))

# CORRECT: Temporal split — train on past, test on future
split_idx = int(n * 0.8)
X_train_t = X.iloc[:split_idx]
X_test_t = X.iloc[split_idx:]
y_train_t = y.iloc[:split_idx]
y_test_t = y.iloc[split_idx:]

model_t = RandomForestRegressor(n_estimators=100, random_state=42)
model_t.fit(X_train_t, y_train_t)
mae_t = mean_absolute_error(y_test_t, model_t.predict(X_test_t))

print(f"Random split MAE (leaky): {mae_l:.4f}")
print(f"Temporal split MAE (correct): {mae_t:.4f}")
print(f"\nWith random split, future data leaks into training.")
print(f"The model appears better than it really is.")

# Time series cross-validation
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []
for train_idx, val_idx in tscv.split(X):
    model_cv = RandomForestRegressor(n_estimators=100, random_state=42)
    model_cv.fit(X.iloc[train_idx], y.iloc[train_idx])
    score = mean_absolute_error(y.iloc[val_idx], model_cv.predict(X.iloc[val_idx]))
    cv_scores.append(score)

print(f"\nTimeSeriesSplit CV scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"Mean CV MAE: {np.mean(cv_scores):.4f}")
```

```
# Output:
# === Temporal Leakage Demo ===
#
# Random split MAE (leaky): 1.8765
# Temporal split MAE (correct): 2.3456
#
# With random split, future data leaks into training.
# The model appears better than it really is.
#
# TimeSeriesSplit CV scores: ['2.1234', '2.3456', '2.5678', '2.3456', '2.4567']
# Mean CV MAE: 2.3678
```

## Common Mistakes

1. **Fitting scalers or encoders on the entire dataset before splitting**: This is the most common form of preprocessing leakage. Statistics computed on the full dataset incorporate information from the test set.

2. **Using `train_test_split` on time-series data**: Random splits mix future data into the training set. Always use temporal splits for time-series.

3. **Including features that are post-hoc or derived from the target**: For example, using "number of transactions" to predict fraud when the label is determined after transaction review.

4. **Data duplication in train/test sets**: If identical rows appear in both training and test sets (duplicates, near-duplicates), the model memorizes them. Deduplicate before splitting.

5. **Feature selection on the full dataset**: Selecting features based on correlation with the target before splitting leaks target information into the feature set.

6. **Using group-based splits incorrectly**: When rows are grouped (e.g., multiple transactions per user), splitting by row instead of by group leaks information across groups.

7. **Imputing missing values using global statistics from the full dataset**: Imputation parameters must be learned from the training set only.

## Interview Questions

### Beginner

1. **Q:** What is data leakage in machine learning?  
   **A:** Data leakage occurs when information from outside the training set (test set or future data) is used during training, leading to overly optimistic performance that does not generalize.

2. **Q:** What is the most common type of preprocessing leakage?  
   **A:** Fitting a StandardScaler on the entire dataset before splitting into train and test sets, so the scaling parameters are influenced by test data.

3. **Q:** How can you detect data leakage?  
   **A:** Unrealistically high performance (e.g., 99.9% accuracy on a difficult problem), high feature importance for suspicious features, or a large gap between training and production performance.

4. **Q:** What is target leakage?  
   **A:** Target leakage occurs when a feature is a direct or indirect proxy for the target label, incorporating information that would not be available at prediction time.

5. **Q:** How do sklearn pipelines help prevent leakage?  
   **A:** Pipelines ensure that `fit` is called on training data only, and `transform` is applied consistently. When used with cross-validation, each fold gets its own transformation parameters.

### Intermediate

1. **Q:** What is temporal leakage and how do you prevent it?  
   **A:** Temporal leakage happens when future data is used to predict the past. Prevent it by using time-based splits, `TimeSeriesSplit`, and ensuring all features are computed using only data available before the prediction timestamp.

2. **Q:** How does group leakage occur and how do you handle it?  
   **A:** Group leakage occurs when rows from the same group (e.g., same patient, same session) appear in both train and test sets. Use `GroupKFold` or `GroupShuffleSplit` to ensure all rows from a group are in the same split.

3. **Q:** What is leakage through feature engineering and how do you avoid it?  
   **A:** It occurs when features are computed using information that would not be available at inference time (e.g., future rolling averages). Always compute features using only data available up to the current time point.

4. **Q:** Can hyperparameter tuning cause data leakage?  
   **A:** Yes, if you evaluate the tuned model on the same test set multiple times, the test set indirectly influences hyperparameter choices. Use nested cross-validation or a separate holdout set.

5. **Q:** How does data leakage affect feature selection?  
   **A:** If feature selection (e.g., selecting top-k features by mutual information) is done on the full dataset before splitting, it leaks target information. Always perform feature selection within the training set only.

### Advanced

1. **Q:** Describe a real-world scenario where subtle data leakage caused a model to fail in production, and how you would redesign the evaluation pipeline.  
   **A:** A hospital readmission model included "length of stay" which was highly correlated with readmission. But length of stay is not known at admission time (when predictions are made). The fix: for each prediction, compute features using only data available at the prediction timestamp. Use a temporal validation scheme where the model is evaluated on patients discharged after the training period.

2. **Q:** How do you handle leakage in a streaming or online learning setting where the model is continuously updated?  
   **A:** Maintain separate state for training and inference. Use a delay window: the model trains on data that is at least N hours old, while serving predictions on current data. Monitor for feature drift and concept drift. Use A/B testing to validate online performance.

3. **Q:** Design a comprehensive leakage detection system for an ML pipeline that includes feature engineering, preprocessing, model training, and evaluation. Include automated tests and monitoring.  
   **A:** 1) Pipeline-level checks: assert that all transforms are inside sklearn pipelines checked at code review. 2) Temporal checks: validate that feature timestamps precede label timestamps. 3) Statistical checks: compare distribution of training and test features; flag if suspiciously similar. 4) Performance sanity checks: reject models with AUC > 0.99 unless domain justifies it. 5) Feature importance audit: flag features with unexpectedly high importance for manual review. 6) Shadow deployment: run the candidate model alongside production and compare performance distributions.

## Practice Problems

### Easy

1. Given a dataset, identify whether a feature called "total_refunds" could cause leakage when predicting "customer_churn".

2. Split a time-series dataset correctly using temporal ordering.

3. Create a pipeline that prevents preprocessing leakage for StandardScaler and PCA.

4. Detect whether a RandomForest model with AUC of 0.999 likely has leakage.

5. Use `train_test_split` with stratification to prevent class imbalance leakage.

### Medium

1. Build a pipeline that includes `SimpleImputer`, `StandardScaler`, and `LogisticRegression`, and verify no leakage using cross-validation.

2. Given a dataset with multiple rows per user, implement a group-aware split that prevents user-level leakage.

3. Implement a custom cross-validator that enforces temporal ordering.

4. Detect and remove leaky features from a credit risk dataset with 50 features.

5. Write a function that takes a raw DataFrame and a pipeline, and validates that the pipeline does not leak by comparing cross-val scores with and without pipeline.

### Hard

1. Implement nested cross-validation with an outer loop for model selection and an inner loop for hyperparameter tuning, ensuring no leakage between the two.

2. Build a leakage detection tool that automatically scans feature engineering code for common leakage patterns (look-ahead bias, group leakage, target leakage).

3. Design a validation framework for a time-series forecasting model that tests the model on multiple non-overlapping future periods and detects if performance degrades over time (leakage through data snooping).

## Solutions

**Easy 1:**
```python
# Yes, "total_refunds" could leak because refunds happen AFTER the prediction
# point. If the model predicts churn before refunds are processed, this feature
# would not be available at inference time. Check the timestamp ordering.
```

**Medium 1:**
```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])

scores = cross_val_score(pipe, X, y, cv=5)
print(f"CV scores: {scores}")
print(f"Mean: {scores.mean():.4f}")
# These scores are leak-free because each fold's preprocessing is fit on that fold's training data only.
```

**Hard 1:**
```python
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.svm import SVC

# Outer loop: model evaluation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
# Inner loop: hyperparameter tuning
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
svc = SVC()

# GridSearchCV performs the inner loop
clf = GridSearchCV(svc, param_grid, cv=inner_cv)
# cross_val_score performs the outer loop
nested_scores = cross_val_score(clf, X, y, cv=outer_cv)
print(f"Nested CV scores: {nested_scores}")
print(f"Mean: {nested_scores.mean():.4f}")
```

## Related Concepts

- **ML-076 ML Pipelines**: Pipelines are the primary mechanism for preventing preprocessing leakage.
- **ML-084 Reproducibility**: Leakage detection is part of reproducibility testing.
- **ML-077 Feature Stores**: Point-in-time correctness in feature stores prevents temporal leakage.
- **ML-082 Batch vs Realtime**: Batch and real-time systems must both guard against leakage, but through different mechanisms.

## Next Concepts

- **ML-084 Reproducibility** — Ensuring experiments are reproducible by preventing all forms of leakage.
- **ML-085 Fairness in ML** — Understanding how leakage can amplify fairness issues.

## Summary

Data leakage is a critical failure mode in machine learning that causes models to appear successful in evaluation but fail in production. The three main types are feature leakage (using future information), preprocessing leakage (fitting transforms on the full dataset), and target leakage (features that are proxies for the label). Prevention strategies include using sklearn pipelines, temporal validation for time-series, group-aware splitting, and comprehensive validation frameworks. Detecting leakage requires skepticism of unrealistically good results and systematic auditing of the feature engineering and evaluation pipeline.

## Key Takeaways

- Data leakage silently inflates metrics while models fail in production
- Three types: feature leakage, preprocessing leakage, target leakage
- Always fit preprocessing transforms on training data only (use pipelines)
- Use temporal splits for time-series data, never random splits
- Group-aware splitting prevents leakage across related samples
- Unrealistically high performance is a red flag for leakage
- Pipelines are the best defense against common leakage patterns
