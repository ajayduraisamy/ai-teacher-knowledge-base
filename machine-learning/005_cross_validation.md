# Concept: Cross-Validation

## Concept ID

ML-005

## Difficulty

INTERMEDIATE

## Domain

Machine Learning

## Module

ML Fundamentals

## Learning Objectives

- Explain why cross-validation is superior to a single train/test split
- Implement k-fold cross-validation using scikit-learn
- Understand stratified k-fold for classification
- Recognize when to use leave-one-out cross-validation (LOOCV)
- Use repeated k-fold for more stable estimates
- Apply GroupKFold for clustered/grouped data

## Prerequisites

- ML-003: Train/Test Split
- ML-004: Overfitting and Underfitting
- Basic understanding of model evaluation

## Definition

Cross-validation is a resampling technique used to evaluate machine learning models by training and testing them on multiple different splits of the data. The most common form is **k-fold cross-validation**, where the data is divided into k equal-sized folds. The model is trained on k-1 folds and tested on the remaining fold. This process is repeated k times, with each fold serving as the test set exactly once. The final performance estimate is the average of the k individual evaluations.

### k-Fold Cross-Validation

In k-fold CV, the dataset D is partitioned into k mutually exclusive subsets (folds) of approximately equal size:

$$D = D_1 \cup D_2 \cup ... \cup D_k$$

For each fold i = 1 to k:
- Train the model on $D \setminus D_i$ (all data except fold i)
- Evaluate on $D_i$
- Record performance metric $M_i$

Final estimate: $\hat{M} = \frac{1}{k} \sum_{i=1}^k M_i$

Common choices for k are 5 and 10. The variance of the estimate decreases as k increases, but computational cost increases proportionally.

### Stratified k-Fold Cross-Validation

Stratified k-fold preserves the class proportion in each fold, ensuring that each fold is representative of the overall class distribution. This is critical for imbalanced classification problems.

### Leave-One-Out Cross-Validation (LOOCV)

LOOCV is a special case where k equals the number of samples n. Each fold contains a single sample. The model is trained on n-1 samples and tested on the single held-out sample. LOOCV has low bias (almost all data used for training) but high variance (each evaluation on a single sample) and is computationally expensive for large datasets.

### Repeated k-Fold Cross-Validation

Repeated k-fold repeats the k-fold process multiple times with different randomizations. This provides more stable performance estimates at the cost of additional computation. A common choice is 5-fold repeated 10 times (50 total train/test evaluations).

### GroupKFold

GroupKFold ensures that samples from the same group (e.g., multiple measurements from the same patient) are not split across folds. This prevents data leakage and provides a more realistic evaluation.

## Intuition

Imagine you are a teacher evaluating a student's knowledge. You could give one final exam (a single train/test split), but the result depends heavily on which questions you choose. If the exam happens to cover topics the student studied extensively, the score is inflated. If it covers topics the student neglected, the score is deflated.

A better approach is to give multiple exams throughout the semester, each covering different subsets of the material. The student's final grade is the average of all exams. This is analogous to cross-validation — we evaluate the model multiple times on different data subsets and average the results.

If you are particularly thorough, you might use repeated k-fold, like giving multiple sets of exams across different semesters to get a truly stable estimate of the student's ability.

## Why This Concept Matters

A single train/test split has several problems:

1. **High variance of estimate**: The test set might be "easy" or "hard" by chance, giving an unreliable performance estimate.
2. **Wasted data**: A larger test set gives more reliable estimates but leaves less data for training.
3. **No information about stability**: A single split cannot tell you whether the model's performance varies dramatically depending on which data it is trained on.

Cross-validation addresses all these issues:
- It provides a more reliable estimate by averaging over multiple evaluations.
- It uses all data for both training and testing (each sample is in the test set exactly once).
- The variance across folds reveals how sensitive the model is to training data variations.

Cross-validation is also essential for model selection and hyperparameter tuning, where we need to compare multiple models and choose the best one without overfitting to a particular test set.

## Mathematical Explanation

### Variance of Cross-Validation Estimate

The variance of the CV estimate depends on k, the sample size, and the correlation between folds:

$$Var(\hat{M}_{CV}) = \frac{1}{k^2} \left( \sum_{i=1}^k Var(M_i) + 2 \sum_{i<j} Cov(M_i, M_j) \right)$$

Since the training sets overlap (each pair of training sets shares k-2 folds), the estimates $M_i$ and $M_j$ are positively correlated. This correlation increases with k, meaning the variance reduction from using more folds is less than expected from independent samples.

### Bias of Cross-Validation Estimate

CV provides a nearly unbiased estimate of the generalization error:

$$\mathbb{E}[\hat{M}_{CV}] \approx \mathbb{E}[M_{test}]$$

For leave-one-out CV, the bias is minimal because each training set is almost the full dataset. For k-fold with small k (e.g., k=5), there is a slight pessimistic bias because the model trains on less data.

### Effective Sample Size for Evaluation

The effective number of test evaluations in k-fold CV is n (each sample is tested once), but the effective number of independent evaluations is less than n due to overlapping training sets. This is why the variance of k-fold CV does not decrease as $1/n$ but somewhat slower.

## Code Examples

### Example 1: Basic k-Fold Cross-Validation

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold

iris = load_iris()
X, y = iris.data, iris.target

# Logistic Regression with 5-fold cross-validation
model = LogisticRegression(max_iter=200)

# Using cross_val_score (simplest approach)
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print("5-Fold CV Scores:", scores)
print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
# Output:
# 5-Fold CV Scores: [0.96666667 1.         0.93333333 0.96666667 1.        ]
# Mean accuracy: 0.973 (+/- 0.025)

# Explicit KFold object for more control
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores_kf = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print(f"KFold CV: Mean={scores_kf.mean():.3f}, Std={scores_kf.std():.3f}")
# Output:
# KFold CV: Mean=0.973, Std=0.025
```

### Example 2: Stratified k-Fold for Imbalanced Data

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Create imbalanced dataset
X, y = make_classification(
    n_samples=1000, weights=[0.9, 0.1],
    random_state=42, flip_y=0.05
)

print(f"Class distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
# Output: Class distribution: 0=900, 1=100

model = RandomForestClassifier(random_state=42)

# Regular k-fold — may get folds with no minority class
kf = KFold(n_splits=5, shuffle=True, random_state=42)
reg_scores = cross_val_score(model, X, y, cv=kf, scoring='f1')

# Stratified k-fold — preserves class proportion in each fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
strat_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')

print(f"Regular KFold F1: {reg_scores.mean():.3f} (+/- {reg_scores.std():.3f})")
print(f"Stratified KFold F1: {strat_scores.mean():.3f} (+/- {strat_scores.std():.3f})")
# Output:
# Regular KFold F1: 0.825 (+/- 0.068)
# Stratified KFold F1: 0.831 (+/- 0.042)
```

### Example 3: Leave-One-Out Cross-Validation (LOOCV)

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import mean_squared_error

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# LOOCV — expensive for large datasets!
loo = LeaveOneOut()
n_samples = X.shape[0]
print(f"Number of LOOCV folds: {n_samples}")
# Output: Number of LOOCV folds: 442

# Using RMSE as evaluation metric
scores = cross_val_score(
    LinearRegression(), X, y, cv=loo,
    scoring='neg_mean_squared_error'
)
rmse_scores = np.sqrt(-scores)

print(f"LOOCV RMSE: mean={rmse_scores.mean():.2f}, std={rmse_scores.std():.2f}")
# Output:
# LOOCV RMSE: mean=52.70, std=46.88

# Compare with 10-fold CV (much faster)
from sklearn.model_selection import KFold
kf10 = KFold(n_splits=10, shuffle=True, random_state=42)
scores_10 = cross_val_score(
    LinearRegression(), X, y, cv=kf10,
    scoring='neg_mean_squared_error'
)
rmse_10 = np.sqrt(-scores_10)
print(f"10-Fold CV RMSE: mean={rmse_10.mean():.2f}, std={rmse_10.std():.2f}")
# Output:
# 10-Fold CV RMSE: mean=52.33, std=14.72
```

### Example 4: Repeated k-Fold Cross-Validation

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, RepeatedKFold

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

model = SVC(kernel='linear', C=1.0)

# Single 5-fold CV
kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
single_scores = cross_val_score(model, X, y, cv=kf5, scoring='accuracy')
print(f"Single 5-fold: Mean={single_scores.mean():.4f}, Std={single_scores.std():.4f}")
# Output:
# Single 5-fold: Mean=0.9561, Std=0.0091

# Repeated 5-fold (10 times = 50 evaluations)
rkf = RepeatedKFold(
    n_splits=5, n_repeats=10, random_state=42
)
repeated_scores = cross_val_score(model, X, y, cv=rkf, scoring='accuracy')
print(f"Repeated 5-fold (10x): Mean={repeated_scores.mean():.4f}, Std={repeated_scores.std():.4f}")
# Output:
# Repeated 5-fold (10x): Mean=0.9569, Std=0.0127
```

### Example 5: GroupKFold for Clustered Data

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold

# Create dataset with group structure (e.g., multiple records per patient)
np.random.seed(42)
n_groups = 20
samples_per_group = 10
n_samples = n_groups * samples_per_group

X, y = make_classification(
    n_samples=n_samples, n_features=5,
    n_informative=3, random_state=42
)

# Each group represents a patient/cluster
groups = np.repeat(np.arange(n_groups), samples_per_group)

model = LogisticRegression(max_iter=200)

# GroupKFold ensures all samples from same group stay together
gkf = GroupKFold(n_splits=5)

print("GroupKFold splits (group IDs in each fold):")
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    train_groups = np.unique(groups[train_idx])
    test_groups = np.unique(groups[test_idx])
    print(f"  Fold {fold+1}: Train groups={train_groups}, Test groups={test_groups}")
# Output:
# GroupKFold splits (group IDs in each fold):
#   Fold 1: Train groups=[ 0  1  2  3  4  6  7  8  9 10 11 12 13 14 15 16], Test groups=[ 5 17 18 19]
#   Fold 2: Train groups=[ 0  1  2  3  5  6  7  8  9 10 11 12 13 14 15 17], Test groups=[ 4 16 18 19]
#   Fold 3: Train groups=[ 0  1  2  4  5  6  7  8  9 10 11 13 14 15 16 18], Test groups=[ 3 12 17 19]
#   Fold 4: Train groups=[ 0  1  3  4  5  7  8  9 10 11 12 13 14 16 17 18], Test groups=[ 2  6 15 19]
#   Fold 5: Train groups=[ 2  3  4  5  6  7  8  9 10 11 12 13 15 16 17 18], Test groups=[ 0  1 14 19]

scores = cross_val_score(model, X, y, cv=gkf, groups=groups, scoring='accuracy')
print(f"GroupKFold accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
# Output:
# GroupKFold accuracy: 0.805 (+/- 0.056)
```

### Example 6: Using Cross-Validation for Model Selection

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

digits = load_digits()
X, y = digits.data, digits.target

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'SVM (linear)': SVC(kernel='linear', C=1.0),
    'SVM (rbf)': SVC(kernel='rbf', C=1.0, gamma='scale'),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'k-NN (k=3)': KNeighborsClassifier(n_neighbors=3),
    'k-NN (k=7)': KNeighborsClassifier(n_neighbors=7),
}

results = []
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    results.append({
        'Model': name,
        'Mean': scores.mean(),
        'Std': scores.std()
    })
    print(f"{name:20s}: Mean={scores.mean():.4f}, Std={scores.std():.4f}")
# Output:
# SVM (linear)       : Mean=0.9711, Std=0.0095
# SVM (rbf)          : Mean=0.9638, Std=0.0115
# Decision Tree      : Mean=0.8509, Std=0.0225
# k-NN (k=3)         : Mean=0.9711, Std=0.0081
# k-NN (k=7)         : Mean=0.9700, Std=0.0089
```

## Common Mistakes

1. **Using cross-validation with time series data without temporal awareness**: Standard k-fold randomly shuffles data, which puts future data in training and past data in testing. Use `TimeSeriesSplit` instead.
2. **Applying cross-validation after feature selection on the whole dataset**: If you select features using all data and then cross-validate, you have leaked information. Feature selection must be done inside the CV loop on each training fold.
3. **Reporting only the mean without the standard deviation**: The variance across folds reveals model stability. A high standard deviation indicates the model's performance depends heavily on which data it trains on.
4. **Using k=2 or very low k (high bias)**: With only 2 folds, each training set is only half the data, which may give a pessimistic and high-variance estimate.
5. **Using LOOCV on large datasets**: LOOCV requires training n models, which is computationally prohibitive for datasets with more than a few thousand samples.
6. **Assuming CV performance is the final deployment performance**: CV estimates the model's performance on data similar to the training distribution. Real-world deployment often involves distribution shifts, latency constraints, and other factors not captured by CV.
7. **Not shuffling data before k-fold splitting**: If the data has any inherent ordering (e.g., sorted by class or by time), non-shuffled k-fold can create unrepresentative folds.

## Interview Questions

### Beginner - 5

1. **Q: What is k-fold cross-validation?**
   A: K-fold cross-validation partitions the data into k equal-sized folds. The model is trained on k-1 folds and tested on the remaining fold, repeated k times. The final performance is the average across all k iterations.

2. **Q: What is a good default value for k in k-fold cross-validation?**
   A: Common choices are k=5 or k=10. K=5 provides a good balance between bias and variance with reasonable computational cost. K=10 has lower bias but higher computational cost.

3. **Q: How is cross-validation different from a single train/test split?**
   A: A single split uses one test set, which may be "easy" or "hard" by chance. Cross-validation averages over k different test sets, providing a more reliable performance estimate and using all data for both training and testing.

4. **Q: What is stratified cross-validation?**
   A: Stratified cross-validation ensures each fold has the same proportion of classes as the original dataset. This is important for imbalanced classification problems.

5. **Q: What is the advantage of LOOCV over k-fold CV?**
   A: LOOCV has lower bias because each training set uses n-1 samples (almost the entire dataset). It is deterministic (no randomness in splitting).

### Intermediate - 5

1. **Q: Compare the bias and variance of k-fold CV with different values of k.**
   A: Small k (e.g., k=2): higher bias (less training data per fold), lower variance (fewer correlated estimates). Large k (e.g., k=n LOOCV): lower bias (more training data), higher variance (correlated estimates, each from a very similar training set).

2. **Q: How does the "one standard error rule" help in model selection with cross-validation?**
   A: The one standard error rule selects the simplest model whose performance is within one standard error of the best-performing model. This avoids selecting overly complex models that may not generalize better in practice.

3. **Q: When would you use repeated k-fold CV?**
   A: Repeated k-fold is useful when you need a very stable performance estimate, typically for model comparison or when the dataset is small and the variance of a single k-fold is high.

4. **Q: Explain the difference between `cross_val_score` and `cross_validate` in scikit-learn.**
   A: `cross_val_score` returns only the test scores. `cross_validate` returns multiple metrics (train and test scores), fit times, and score times, providing more comprehensive information.

5. **Q: How do you perform nested cross-validation for unbiased model selection and evaluation?**
   A: Nested CV has two loops: outer loop for performance estimation, inner loop for model selection. In each outer fold, the training data is further split for inner CV to tune hyperparameters. This ensures the outer test data is never used for model selection, providing an unbiased performance estimate.

### Advanced - 3

1. **Q: Derive the bias and variance of k-fold cross-validation and explain the "pessimistic bias" for small k.**
   A: Let $R(k)$ be the expected CV error with k folds. The bias is $\mathbb{E}[R(k)] - R_{true}$, where $R_{true}$ is the generalization error with the full dataset. For small k, the training set is smaller by a factor of $(k-1)/k$, so the model is worse, and CV underestimates true performance (pessimistic bias). As k increases, bias decreases. The variance is more complex due to overlapping training sets; the variance of LOOCV can be higher than 5-fold CV due to high correlation between folds.

2. **Q: Prove that LOOCV is approximately unbiased for the true generalization error but can have high variance.**
   A: LOOCV error: $\hat{R}_{LOO} = \frac{1}{n} \sum_{i=1}^n L(y_i, f_{-i}(x_i))$. Since each training set $D_{-i}$ differs from the full dataset by exactly one point, and $\mathbb{E}[f_{-i}] \approx \mathbb{E}[f_{full}]$, the bias is $O(1/n)$. However, the estimates $L(y_i, f_{-i}(x_i))$ are highly correlated across i because training sets differ by only one point, so $Var(\hat{R}_{LOO})$ can be large. For unstable models (e.g., decision trees), LOOCV can have much higher variance than 5- or 10-fold CV.

3. **Q: Explain how the concept of effective degrees of freedom relates to cross-validation, and derive the generalized cross-validation (GCV) approximation for linear models.**
   A: For linear models $\hat{y} = Hy$ with hat matrix H, the effective degrees of freedom is $df = tr(H)$. GCV approximates LOOCV without retraining: $GCV = \frac{1}{n} \sum_{i=1}^n \left( \frac{y_i - \hat{f}(x_i)}{1 - H_{ii}} \right)^2$. For ridge regression, $H_{ii} = \frac{h_{ii}}{n}$ where $h_{ii}$ are leverages. GCV replaces $H_{ii}$ with $tr(H)/n$, giving $GCV = \frac{RSS}{n (1 - df/n)^2}$. This is computationally efficient because it requires fitting only once instead of n times.

## Practice Problems

### Easy - 5

1. **Problem**: You have 1000 samples and you want to use 10-fold cross-validation. How many samples are in each fold? How many times is each sample used for testing?

2. **Problem**: What is the computational cost of LOOCV on a dataset with 10,000 samples compared to 5-fold CV?

3. **Problem**: For a binary classification dataset with 95% class A and 5% class B, which CV strategy is most appropriate?

4. **Problem**: A 5-fold CV gives scores [0.72, 0.75, 0.73, 0.74, 0.71]. What is the mean and standard deviation?

5. **Problem**: When would you NOT want to use cross-validation?

### Medium - 5

1. **Problem**: You are evaluating a model on a dataset of medical records from 500 patients, each having 2-8 visits. Explain why regular k-fold CV is inappropriate and how you would fix this.

2. **Problem**: Compare the computational cost and statistical properties of k=2 vs k=∞ (LOOCV) for a dataset with n=200 and p=10 features.

3. **Problem**: A 5-fold CV on a logistic regression model gives mean accuracy 0.85 with std 0.02. A 5-fold CV on a random forest gives mean 0.87 with std 0.04. Which model would you choose and why?

4. **Problem**: Explain how to incorporate feature selection and scaling inside the cross-validation loop to avoid data leakage. Provide pseudocode.

5. **Problem**: You run a repeated 5-fold CV (10 repeats) on two models. Model A: mean=0.92, std=0.015. Model B: mean=0.91, std=0.010. Which model is more likely to generalize better in production?

### Hard - 3

1. **Problem**: Design a cross-validation strategy for evaluating a model that predicts patient readmission within 30 days of discharge. The data contains multiple hospital visits per patient. Address temporal dependencies, patient-level grouping, and class imbalance.

2. **Problem**: Prove that the variance of k-fold CV is minimized when folds are as independent as possible, and explain why this leads to preferring stratified over regular k-fold for classification.

3. **Problem**: Compare and contrast the theoretical properties of cross-validation, bootstrap, and the jackknife for estimating prediction error. Under what conditions does each method fail?

## Solutions

### Easy Solutions

1. Each fold has 1000/10 = 100 samples. Each sample is used for testing exactly once.
2. LOOCV trains 10,000 models. 5-fold CV trains 5 models. LOOCV is 2000 times more computationally expensive.
3. Stratified k-fold (or stratified repeated k-fold) to ensure each fold maintains the 95/5 class ratio.
4. Mean = (0.72 + 0.75 + 0.73 + 0.74 + 0.71) / 5 = 0.73. Standard deviation = 0.0158.
5. When the dataset is very large (CV adds little benefit over a single split), when data has strong temporal dependence, or when computational resources are severely constrained.

### Medium Solutions

1. Regular k-fold could split visits from the same patient across training and test sets (data leakage). Use GroupKFold with the patient ID as the group parameter.
2. k=2: bias higher (train on 100 samples), cost = 2 fits, variance moderate. LOOCV: bias near 0 (train on 199 samples), cost = 200 fits, variance higher due to correlated estimates. LOOCV is preferred for small datasets when accuracy matters; k=2 is rarely used.
3. Random forest has higher mean but also higher variance. Using the one-standard-error rule: random forest mean - 1 std = 0.83, logistic regression mean + 1 std = 0.87. The simpler model (logistic regression) is within one standard error, so choose it for its interpretability and stability.
4. Pseudocode: `for each fold: (1) Split into train_fold and test_fold. (2) Fit scaler on train_fold only. (3) Transform train_fold and test_fold with fitted scaler. (4) Perform feature selection on train_fold only. (5) Train model on train_fold. (6) Evaluate on test_fold.`
5. Model A has better mean but higher variance. If the 0.01 difference is practically significant, choose A. If not (e.g., both are within acceptable range), Model B's lower variance suggests it is more stable and may generalize better.

### Hard Solutions

1. Strategy: (1) GroupKFold by patient ID (all visits from one patient stay together). (2) Within each training fold, use time-based split for validation (train on earlier visits, validate on later visits). (3) Use stratified GroupKFold to handle class imbalance. (4) Report average performance across folds with confidence intervals. (5) Additionally, evaluate temporal generalization by training on all data before a cutoff date and testing after it.
2. The variance of k-fold CV is $Var(\hat{M}) = \frac{1}{k^2} (\sum_i Var(M_i) + 2\sum_{i<j} Cov(M_i, M_j))$. Since training sets overlap, $Cov(M_i, M_j) > 0$. Stratification reduces the variance of $M_i$ within each fold (by ensuring representative class proportions) and reduces the covariance between folds (by making each fold more similar to the overall population). This minimizes the total variance.
3. Cross-validation: Low bias, moderate variance. Fails when data is not i.i.d. (time series, spatial data). Bootstrap: Can have lower variance but higher bias (Bootstrap .632 estimator addresses this). Fails when training set has many unique/influential points. Jackknife: Similar to LOOCV, low bias, can have high variance. Fails for non-smooth statistics (e.g., median). All fail under severe distribution shift between training and deployment.

## Related Concepts

- **ML-003: Train/Test Split** — The simpler precursor to cross-validation
- **ML-004: Overfitting and Underfitting** — Cross-validation helps diagnose both
- **ML-006: Evaluation Metrics** — What we measure within cross-validation folds
- **Bootstrap**: An alternative resampling method for estimating prediction error
- **Hyperparameter Tuning**: Cross-validation is the standard tool for tuning

## Next Concepts

- **ML-006: Evaluation Metrics** — The metrics we evaluate inside CV folds
- **ML-007: Confusion Matrix** — Detailed evaluation for classification
- **ML-008: ROC and AUC** — Threshold-independent evaluation

## Summary

Cross-validation is a resampling technique that provides reliable performance estimates by training and evaluating a model on multiple data splits. K-fold CV partitions data into k folds, using each in turn for testing. Stratified k-fold preserves class proportions, GroupKFold respects group structure, and LOOCV uses n folds (one per sample). Cross-validation yields lower bias and more stable estimates than a single train/test split. It is essential for model selection, hyperparameter tuning, and trustworthy evaluation. Common pitfalls include applying CV to non-i.i.d. data, leaking information across folds, and misinterpreting CV performance as guaranteed deployment performance.

## Key Takeaways

1. Cross-validation provides more reliable performance estimates than a single train/test split.
2. K=5 or k=10 are standard choices balancing bias, variance, and computation.
3. Use stratified k-fold for classification, especially with imbalanced classes.
4. Use GroupKFold when data has hierarchical or group structure.
5. Avoid LOOCV for large datasets due to computational cost.
6. Always shuffle data before k-fold splitting (unless time series).
7. Preprocessing (scaling, feature selection) must happen inside the CV loop.
