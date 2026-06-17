# Concept: Training and Test Split

## Concept ID

ML-003

## Difficulty

BEGINNER

## Domain

Machine Learning

## Module

ML Fundamentals

## Learning Objectives

- Understand why splitting data is essential for evaluating ML models.
- Distinguish between training, validation, and test sets.
- Implement train-test splits using scikit-learn.
- Apply stratified splitting for classification problems.
- Recognize time-series specific splitting requirements.

## Prerequisites

- Basic Python programming
- Understanding of supervised learning (ML-002)

## Definition

The **train-test split** is the process of dividing a dataset into separate subsets: one for training a machine learning model and another for evaluating its performance. The training set is used to learn patterns, while the test set provides an unbiased estimate of how well the model generalizes to unseen data.

## Intuition

Imagine studying for an exam. You have a textbook with practice problems. If you only study the exact problems that appear on the exam, you will do well on that exam but fail to solve new problems. This is memorization, not learning.

Splitting data works the same way. The model should be evaluated on data it has never seen during training. If we evaluate on the same data used for training, we get an overly optimistic picture of performance — the model may simply have memorized the answers.

## Why This Concept Matters

Without proper data splitting, every evaluation metric is meaningless. Data splitting is arguably the most fundamental validation technique in ML because:

- It detects overfitting (model memorizes training data but fails on new data).
- It provides an unbiased estimate of real-world performance.
- It enables hyperparameter tuning through validation sets.
- It prevents data leakage between training and evaluation.

## Mathematical Explanation

Given a dataset $D = \{(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)\}$ with $n$ samples, we partition it into:

$$D = D_{\text{train}} \cup D_{\text{test}} \quad \text{where} \quad D_{\text{train}} \cap D_{\text{test}} = \emptyset$$

Common split ratios:
- 80% train / 20% test
- 70% train / 30% test
- 60% train / 20% validation / 20% test (three-way split)

For a dataset with $n$ samples and test proportion $p$, the test set contains $n \times p$ samples and the training set contains $n \times (1-p)$ samples.

When the dataset is large ($n > 100{,}000$), even a small test set (e.g., 10,000 samples) is sufficient for reliable evaluation.

## Code Examples

### Example 1: Basic Train-Test Split

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

print(f"Total samples: {X.shape[0]}")
print(f"Features: {X.shape[1]}")

# Split into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Output:
# Total samples: 150
# Features: 4
# Training set size: 120
# Test set size: 30
```

### Example 2: Stratified Split

```python
from sklearn.model_selection import train_test_split
import numpy as np

# Imbalanced dataset
X = np.random.randn(100, 2)
y = np.array([0] * 90 + [1] * 10)  # 90% class 0, 10% class 1

# Without stratification — test set might have no class 1 samples
X_train_bad, X_test_bad, y_train_bad, y_test_bad = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Without stratification — test class distribution: {np.bincount(y_test_bad)}")
# Output:
# Without stratification — test class distribution: [20  0]

# With stratification — preserves class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"With stratification — test class distribution: {np.bincount(y_test)}")
# Output:
# With stratification — test class distribution: [18  2]
```

### Example 3: Three-Way Split (Train/Validation/Test)

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# First split: separate test set
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: separate validation from training
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
# Output:
# Train: 90, Val: 30, Test: 30
```

### Example 4: Time-Series Split (No Future Leakage)

```python
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Simulated time series data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
X = pd.DataFrame({'feature': range(100)}, index=dates)
y = pd.Series(range(100), index=dates)

tscv = TimeSeriesSplit(n_splits=5)
for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"Fold {i+1}: Train {len(train_idx)}, Test {len(test_idx)}")
    # Always uses past to predict future

# Output:
# Fold 1: Train 20, Test 20
# Fold 2: Train 40, Test 20
# Fold 3: Train 60, Test 20
# Fold 4: Train 80, Test 20
# Fold 5: Train 80, Test 20
```

### Example 5: The Danger of No Split

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np

# This is WRONG — evaluating on training data
X, y = load_iris(return_X_y=True)

# Train on ALL data
model = DecisionTreeClassifier(random_state=42, max_depth=10)
model.fit(X, y)

# Evaluate on the SAME data (overly optimistic!)
train_preds = model.predict(X)
train_acc = accuracy_score(y, train_preds)

print(f"Training accuracy (no split): {train_acc:.3f}")
# Output:
# Training accuracy (no split): 1.000

# This is MISLEADING — the model may have just memorized!
```

## Common Mistakes

1. **Testing on training data:** Evaluating on the same data used for training gives an overly optimistic (and useless) accuracy estimate. This is the most common beginner mistake.

2. **Data leakage through splitting:** Using future data to predict the past. Always split time-series chronologically, never randomly.

3. **Forgetting to set random_state:** Without `random_state`, you get a different split each run, making results non-reproducible.

4. **Splitting before preprocessing:** Scaling, encoding, or imputing before the split causes data leakage — the test set indirectly influences preprocessing statistics.

5. **Too small test set:** With very small test sets, evaluation metrics have high variance. Ensure at least 1000 test samples for reliable estimates (or use cross-validation).

6. **Ignoring class imbalance:** Random splitting of imbalanced data can produce test sets with zero samples from minority classes. Always use `stratify` for classification.

## Interview Questions

### Beginner

1. **Why do we split data into train and test sets?**
   *Answer: To evaluate how well the model generalizes to unseen data. Evaluating on the training set overestimates performance because the model may simply memorize the data.*

2. **What is a typical train-test split ratio?**
   *Answer: Common ratios are 80/20 or 70/30 for train/test. For very large datasets (1M+ samples), even 90/10 or 99/1 works because the test set needs fewer samples for reliable statistics.*

3. **What is the difference between training, validation, and test sets?**
   *Answer: Training set fits model parameters. Validation set tunes hyperparameters and selects models. Test set gives final unbiased evaluation — it should only be used once.*

4. **What does `random_state=42` do in train_test_split?**
   *Answer: It seeds the random number generator so the split is reproducible. Same random_state always produces the same split, which is essential for debugging and comparing models.*

5. **What happens if you split time-series data randomly?**
   *Answer: Random splitting leaks future information into the training set, making evaluation unrealistically optimistic. Time-series must be split chronologically.*

### Intermediate

1. **Explain stratified splitting. When is it necessary?**
   *Answer: Stratified splitting preserves the class proportion in both training and test sets. It is necessary when dealing with imbalanced datasets; otherwise, a random split might produce a test set with very few or zero samples from the minority class.*

2. **What is data leakage in the context of train-test splitting?**
   *Answer: Data leakage occurs when information from outside the training set influences the training process. Common examples: scaling before splitting (test statistics leak into training), using future data to predict past, or including target-correlated features.*

3. **How do you determine an appropriate test set size?**
   *Answer: It depends on dataset size. For small datasets (n < 1000), use larger test proportions (20-30%) or cross-validation. For large datasets (n > 100k), even 1-2% is sufficient. The test set should be large enough to produce low-variance metric estimates.*

4. **What is the proper order: split then preprocess, or preprocess then split? Why?**
   *Answer: Split first, then preprocess. Preprocessing before splitting causes data leakage because statistics (mean, variance for scaling; min, max for normalization) are computed on the entire dataset, including the test set. The test set should be transformed using parameters learned ONLY from the training set.*

5. **How does k-fold cross-validation relate to the train-test split concept?**
   *Answer: k-fold CV is an extension that splits data into k folds, iteratively using k-1 folds for training and 1 for testing. Each sample gets tested exactly once. It is more robust than a single split because it uses all data for both training and testing at different points.*

### Advanced

1. **Derive the variance of the test accuracy estimate as a function of test set size n_test. How does this guide the choice of split ratio?**
   *Answer: For accuracy (proportion of correct predictions), the variance is approximately $p(1-p)/n_{test}$ where $p$ is true accuracy. The standard error is $\sqrt{p(1-p)/n_{test}}$. For a standard error below 1%, we need $n_{test} > p(1-p)/(0.01)^2 \approx 2500$ (worst case $p=0.5$). This guides split choice — ensure test set has enough samples for reliable estimation.*

2. **In the context of deep learning, what is the purpose of a held-out validation set vs. using only train/test splits?**
   *Answer: Deep learning involves extensive hyperparameter tuning (learning rate, architecture, regularization). Using the test set for this would bias evaluation. A validation set enables model selection without touching the test set. For very large models, the validation set may also be used for early stopping.*

3. **Describe group splitting (GroupKFold). When is it required, and what happens if you use regular splitting instead?**
   *Answer: Group splitting ensures that all samples from the same group (e.g., same patient, same session) stay in the same fold. Without it, correlated samples leak across train/test splits, leading to overly optimistic performance. For medical ML with multiple scans per patient, GroupKFold is essential to prevent patient-level leakage.*

## Practice Problems

### Easy

1. A dataset has 1000 samples. If you use a 70/30 train-test split, how many samples are in each set?

2. Write code to split `X = [[1,2],[3,4],[5,6],[7,8]]` and `y = [0,0,1,1]` with `test_size=0.25`.

3. What's the difference between `shuffle=True` and `shuffle=False` in `train_test_split`?

4. If you run `train_test_split(X, y, test_size=0.2, random_state=7)` twice, do you get the same split both times?

5. For a binary classification dataset with 95% class A and 5% class B, why is `stratify=y` important?

### Medium

1. Write code that creates a three-way split (60% train, 20% val, 20% test) for the Iris dataset.

2. Explain and demonstrate how data leakage occurs if you apply `StandardScaler` before the train-test split.

3. For a time series with 365 daily observations, how should you split to predict the final 30 days? Write the code.

4. A model achieves 99% accuracy on the training set but 75% on the test set. What is happening? How do you diagnose it?

5. Given `y = [0]*100 + [1]*900` (imbalanced), perform a stratified split with 80/20 ratio and verify the class proportions are preserved.

### Hard

1. Implement a function that performs Monte Carlo cross-validation (repeated random splits) and returns the mean and standard deviation of accuracy across 50 random splits. Compare the variance with a single 80/20 split.

2. For a dataset with patient IDs and multiple measurements per patient, implement group-aware splitting that ensures no patient appears in both train and test. Compare model performance with and without this grouping.

3. Derive the optimal test set size that minimizes the total error (bias + variance) of a model evaluation, given that training on fewer samples increases bias and testing on fewer samples increases variance.

## Solutions

### Easy Solutions

**1.** Train: $1000 \times 0.7 = 700$, Test: $1000 \times 0.3 = 300$.

**2.** 
```python
from sklearn.model_selection import train_test_split
X = [[1,2],[3,4],[5,6],[7,8]]
y = [0,0,1,1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")
```

**3.** `shuffle=True` (default) randomly shuffles data before splitting. `shuffle=False` uses the original order, which is problematic for ordered data (e.g., time series).

**4.** Yes, the same `random_state` always produces the identical split, ensuring reproducibility.

**5.** Without stratification, random splitting might produce a test set with zero class B samples, making it impossible to evaluate minority class performance.

### Medium Solutions

**1.** 
```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
```

**2.** Data leakage: fitting `StandardScaler` on all data computes mean/std from both train and test. The test indirectly influences training through scaled values, giving overly optimistic performance.

**3.** 
```python
train = df.iloc[:-30]
test = df.iloc[-30:]
```
Never shuffle time-series data.

**4.** This is classic overfitting. The model memorizes training data but fails to generalize. Diagnose with learning curves (plot train/test error vs training size) or compare cross-validation scores.

**5.** 
```python
from sklearn.model_selection import train_test_split
import numpy as np
y = [0]*100 + [1]*900
_, _, y_train, y_test = train_test_split(range(len(y)), y, test_size=0.2, stratify=y, random_state=42)
print(f"Train distribution: {np.bincount(y_train)}")
print(f"Test distribution: {np.bincount(y_test)}")
```

### Hard Solutions

**1.** 
```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
accuracies = []
for _ in range(50):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
    model = DecisionTreeClassifier().fit(X_tr, y_tr)
    accuracies.append(accuracy_score(y_te, model.predict(X_te)))
print(f"Mean: {np.mean(accuracies):.3f}, Std: {np.std(accuracies):.3f}")
```

## Related Concepts

- **Overfitting and Underfitting (ML-004):** Train-test split is essential for detecting overfitting.
- **Cross-Validation (ML-005):** Extends the single split to multiple splits for more robust evaluation.
- **Evaluation Metrics (ML-006):** Applied to test set predictions to measure performance.
- **Bias-Variance Tradeoff:** The split ratio affects the bias-variance balance of evaluation.

## Next Concepts

- **Cross-Validation (ML-005):** Moving from a single split to k-fold evaluation for more robust model assessment.
- **Overfitting and Underfitting (ML-004):** Understanding how split-based evaluation helps diagnose these conditions.
- **Evaluation Metrics (ML-006):** Once you have test predictions, you need metrics to quantify performance.

## Summary

Train-test splitting is the foundation of reliable ML evaluation. Data is partitioned into separate subsets for training and evaluation, with a validation set optionally used for hyperparameter tuning. Proper splitting prevents overfitting detection failure, ensures unbiased performance estimates, and requires care with imbalanced data (stratification) and temporal data (chronological splitting). Always split before preprocessing to avoid data leakage.

## Key Takeaways

- Always split data before training — never evaluate on the training set.
- Use 80/20 or 70/30 as default split ratios; adjust for dataset size.
- Set `random_state` for reproducible results.
- Use `stratify=y` for imbalanced classification datasets.
- Split time-series data chronologically, never randomly.
- Split BEFORE any preprocessing or scaling to prevent data leakage.
- Use a three-way split (train/val/test) when hyperparameter tuning.
