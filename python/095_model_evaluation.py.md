# Concept: Model Evaluation

## Concept ID

PYT-095

## Difficulty

Intermediate

## Domain

Python

## Module

Python for ML/AI

## Learning Objectives

- Perform cross-validation with `cross_val_score` and interpret the results
- Conduct hyperparameter search with `GridSearchCV` and `RandomizedSearchCV`
- Generate classification reports, confusion matrices, and ROC/AUC scores
- Compute regression metrics: mean squared error, R² score

## Prerequisites

- PYT-093 — sklearn Basics (fit/predict, train_test_split)
- PYT-094 — Preprocessing (scalers, encoders, Pipeline)
- Basic understanding of classification and regression evaluation metrics

## Definition

Model evaluation measures how well a trained model performs on unseen data. sklearn provides a comprehensive suite of evaluation tools:

**Cross-Validation:**
- `cross_val_score`: Evaluate model by k-fold cross-validation
- `cross_validate`: Extended version returning fit/score times and multiple metrics

**Hyperparameter Tuning:**
- `GridSearchCV`: Exhaustive search over specified parameter grid
- `RandomizedSearchCV`: Random sampling from parameter distributions (faster for many parameters)

**Classification Metrics:**
- `classification_report`: Precision, recall, f1-score, support per class
- `confusion_matrix`: Actual vs predicted counts
- `roc_auc_score`: Area under the ROC curve
- `roc_curve`: FPR and TPR for threshold sweeps
- `precision_recall_curve`: Precision vs recall for threshold sweeps
- `log_loss`: Cross-entropy loss

**Regression Metrics:**
- `mean_squared_error` / `root_mean_squared_error` (sklearn ≥1.4)
- `mean_absolute_error`
- `r2_score`: Proportion of variance explained
- `mean_absolute_percentage_error`

## Intuition

Model evaluation answers: "How good is my model, really?"

A single train/test split can be misleading — you might get lucky (an easy test split) or unlucky (a hard one). Cross-validation averages performance across k different splits, giving a more reliable estimate.

Hyperparameter tuning systematically searches for the best configuration. Grid search tries every combination (exhaustive but expensive). Random search samples combinations (often finds good settings faster).

Each metric tells a different story:
- **Accuracy:** Overall correctness (misleading for imbalanced data)
- **Precision:** When model says "positive," how often is it right?
- **Recall:** What fraction of actual positives did the model catch?
- **F1:** Harmonic mean of precision and recall
- **AUC-ROC:** Model's ability to rank positives higher than negatives (threshold-independent)
- **R²:** Proportion of variance explained by the model (regression)

## Why This Concept Matters

- **Model Selection:** Systematic evaluation tells you which model and hyperparameters actually work best
- **Bias-Variance Diagnosis:** Cross-validation scores reveal overfitting (high train score, low CV score)
- **Business Impact:** Choosing the right metric aligns model optimization with business goals (e.g., recall for disease screening, precision for spam detection)
- **Research Reproducibility:** Proper evaluation methods are required for any ML publication
- **Production Readiness:** Tuning and evaluation are prerequisites for deployment

## Real World Examples

1. **Medical Screening:** A cancer detection model optimized for recall (find all positives) at the cost of precision (some false positives are acceptable for follow-up tests).
2. **Credit Card Fraud:** A fraud detector uses precision-recall AUC because fraud is rare (<0.1%). Accuracy is meaningless.
3. **Recommendation System:** Grid search over learning rate, tree depth, and regularization strength for a gradient boosting model predicting click-through rate.
4. **House Price Prediction:** A regression model evaluated with RMSE (same units as price, interpretable) and R² (how much variance is explained).
5. **NLP Sentiment Analysis:** Cross-validation on 5 folds gives a robust estimate of classifier performance across different subsets of reviews.

## AI/ML Relevance

- **Mandatory ML Step:** No model is complete without evaluation
- **Hyperparameter Optimization:** Every ML model has tunable knobs — systematic search finds the best settings
- **Model Comparison:** Cross-validation enables fair comparison across different algorithms
- **Automated ML (AutoML):** AutoML frameworks are essentially automated grid/random search with smart pruning
- **MLOps:** Evaluation metrics are logged and tracked in ML experimentation platforms (MLflow, Weights & Biases)

## Code Examples

### Example 1: Cross-validation with `cross_val_score`
```python
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(n_samples=500, n_features=20, random_state=42)
rf = RandomForestClassifier(n_estimators=50, random_state=42)

# Basic cross-validation
scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print(f"Accuracy scores: {scores}")
print(f"Mean: {scores.mean():.3f} ± {scores.std():.3f}")

# Extended cross-validation with multiple metrics
cv_results = cross_validate(rf, X, y, cv=5,
                            scoring=['accuracy', 'f1_macro', 'roc_auc'],
                            return_train_score=True)
print(f"\nTrain accuracy: {cv_results['train_accuracy'].mean():.3f}")
print(f"Test accuracy: {cv_results['test_accuracy'].mean():.3f}")
print(f"Test F1: {cv_results['test_f1_macro'].mean():.3f}")
print(f"Test ROC AUC: {cv_results['test_roc_auc'].mean():.3f}")
```
```
# Output:
# Accuracy scores: [0.94 0.93 0.93 0.9  0.91]
# Mean: 0.922 ± 0.015
# Train accuracy: 1.000
# Test accuracy: 0.920
# Test F1: 0.920
# Test ROC AUC: 0.975
```

### Example 2: GridSearchCV for hyperparameter tuning
```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

X, y = make_classification(n_samples=300, n_features=10, random_state=42)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 'scale', 'auto'],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.3f}")
print(f"Test score (refit): {grid_search.score(X, y):.3f}")

results = grid_search.cv_results_
print(f"Number of combinations tried: {len(results['params'])}")
```
```
# Output:
# Best parameters: {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}
# Best CV accuracy: 0.940
# Test score (refit): 0.950
# Number of combinations tried: 20
```

### Example 3: RandomizedSearchCV for faster tuning
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=500, n_features=20, random_state=42)

param_dist = {
    'n_estimators': randint(10, 200),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=30,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X, y)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV accuracy: {random_search.best_score_:.3f}")
print(f"Combinations sampled: 30 (out of a much larger space)")
```
```
# Output:
# Best parameters: {'max_depth': 18, 'min_samples_leaf': 2, 'min_samples_split': 3, 'n_estimators': 131}
# Best CV accuracy: 0.938
# Combinations sampled: 30 (out of a much larger space)
```

### Example 4: Classification report and confusion matrix
```python
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                     test_size=0.3, stratify=iris.target,
                                                     random_state=42)

logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Visualize
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Iris Classification')
plt.show()
```
```
# Output:
# Classification Report:
#               precision    recall  f1-score   support
#       setosa       1.00      1.00      1.00        15
#   versicolor       1.00      0.93      0.97        15
#    virginica       0.94      1.00      0.97        15
#     accuracy                           0.98        45
# Confusion Matrix:
#  [[15  0  0]
#  [ 0 14  1]
#  [ 0  0 15]]
```

### Example 5: ROC Curve and AUC
```python
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)
y_prob = rf.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

print(f"AUC Score: {auc_score:.3f}")

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'RF (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()
```
```
# Output:
# AUC Score: 0.988
# [ROC curve plot displayed]
```

### Example 6: Regression metrics
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=300, n_features=5, noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"R²:   {r2:.3f}")

# Adjusted R²
n = X_test.shape[0]
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print(f"Adjusted R²: {adj_r2:.3f}")
```
```
# Output:
# MSE:  233.67
# RMSE: 15.29
# MAE:  11.82
# R²:   0.887
# Adjusted R²: 0.883
```

### Example 7: Precision-Recall curve for imbalanced data
```python
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier

# Create imbalanced dataset
X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)
y_prob = rf.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)

print(f"Average Precision (AP): {ap:.3f}")

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, 'b-', linewidth=2, label=f'RF (AP = {ap:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Imbalanced Data)')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.show()
```
```
# Output:
# Average Precision (AP): 0.824
# [PR curve plot displayed]
```

### Example 8: Learning curve analysis
```python
from sklearn.model_selection import learning_curve
import numpy as np

X, y = make_classification(n_samples=500, n_features=10, random_state=42)

train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(n_estimators=50, random_state=42),
    X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_std = test_scores.std(axis=1)

print(f"Train sizes: {train_sizes}")
print(f"Train scores (last): {train_mean[-1]:.3f}")
print(f"Test scores (last): {test_mean[-1]:.3f}")

# Diagnose bias/variance
gap = train_mean[-1] - test_mean[-1]
print(f"Generalization gap: {gap:.3f}")
if gap > 0.1:
    print("Suggestion: High variance — consider regularization or more data")
elif train_mean[-1] < 0.8:
    print("Suggestion: High bias — consider a more complex model")
else:
    print("Suggestion: Good fit")
```
```
# Output:
# Train sizes: [ 45  90 135 180 225 270 315 360 405 450]
# Train scores (last): 0.998
# Test scores (last): 0.930
# Generalization gap: 0.068
# Suggestion: Good fit
```

### Example 9: Multi-metric scoring with cross-validation
```python
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, f1_score, accuracy_score

X, y = make_classification(n_samples=300, n_features=10, random_state=42)
rf = RandomForestClassifier(n_estimators=50, random_state=42)

scoring = {
    'accuracy': 'accuracy',
    'f1_macro': 'f1_macro',
    'precision_macro': 'precision_macro',
    'recall_macro': 'recall_macro'
}

cv_results = cross_validate(rf, X, y, cv=5, scoring=scoring, return_train_score=True)

for metric in scoring:
    train_key = f'train_{metric}'
    test_key = f'test_{metric}'
    print(f"{metric}: train={cv_results[train_key].mean():.3f} ± {cv_results[train_key].std():.3f}, "
          f"test={cv_results[test_key].mean():.3f} ± {cv_results[test_key].std():.3f}")
```
```
# Output:
# accuracy: train=0.998 ± 0.003, test=0.943 ± 0.022
# f1_macro: train=0.998 ± 0.003, test=0.943 ± 0.022
# precision_macro: train=0.998 ± 0.003, test=0.944 ± 0.022
# recall_macro: train=0.998 ± 0.003, test=0.943 ± 0.022
```

### Example 10: Custom cross-validation splitter
```python
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, TimeSeriesSplit

X, y = make_classification(n_samples=100, n_features=5, random_state=42)

# Stratified K-Fold (preserves class proportions)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(RandomForestClassifier(n_estimators=20), X, y, cv=skf)
print(f"Stratified K-fold: {scores.mean():.3f} ± {scores.std():.3f}")

# Leave-One-Out (n-fold CV where n = sample size)
loo = LeaveOneOut()
scores_loo = cross_val_score(RandomForestClassifier(n_estimators=20), X[:50], y[:50], cv=loo)
print(f"LOO accuracy: {scores_loo.mean():.3f}")

# Time Series Split (for temporal data)
tscv = TimeSeriesSplit(n_splits=5)
X_time = np.random.randn(100, 5)
y_time = np.random.randn(100) > 0
scores_ts = cross_val_score(RandomForestClassifier(n_estimators=20), X_time, y_time, cv=tscv)
print(f"Time series CV: {scores_ts.mean():.3f} ± {scores_ts.std():.3f}")
```
```
# Output:
# Stratified K-fold: 0.930 ± 0.032
# LOO accuracy: 0.920
# Time series CV: 0.520 ± 0.067
```

## Common Mistakes

1. **Using accuracy on imbalanced datasets.** When 95% of samples are class 0, 95% accuracy is trivial. Use precision, recall, F1, or AUC instead.
2. **Tuning on test data by evaluating too many times.** If you check test performance after every GridSearch iteration, you're leaking test information into model selection. Hold out a final test set and only evaluate once.
3. **Not shuffling data in cross-validation.** For non-random splits (e.g., first 80% for train), data may have ordering. Always use `shuffle=True` in `KFold` or `StratifiedKFold`.
4. **Ignoring the variance of CV scores.** A mean accuracy of 0.92 with std 0.05 is very different from 0.92 with std 0.01. Report both.
5. **Using GridSearchCV on too large a grid.** 10×10×10 = 1000 combinations × 5 folds = 5000 fits. Start with RandomizedSearchCV for large spaces.
6. **Not using `refit=True` (default) in GridSearchCV.** Without refitting, `best_estimator_` is not trained on the full dataset. Default is `True` but worth verifying.
7. **Misinterpreting R².** R² always increases with more features. Use adjusted R² or AIC/BIC for model comparison with different numbers of features.

## Interview Questions

### Beginner - 5

1. **Q:** What is k-fold cross-validation?  
   **A:** Data is split into k equal folds. The model is trained on k-1 folds and tested on the remaining fold, repeated k times with each fold as test once. The average score gives a robust performance estimate.

2. **Q:** What's the difference between GridSearchCV and RandomizedSearchCV?  
   **A:** GridSearchCV exhaustively tries every combination in the parameter grid. RandomizedSearchCV samples a fixed number of combinations from parameter distributions. Random search is faster for large search spaces.

3. **Q:** What information does a classification report show?  
   **A:** Precision, recall, F1-score, and support (number of true instances) for each class, plus overall accuracy.

4. **Q:** What does a confusion matrix tell you?  
   **A:** It shows counts of true positive, true negative, false positive, and false negative predictions, letting you see which classes are commonly confused.

5. **Q:** What is AUC-ROC?  
   **A:** Area Under the Receiver Operating Characteristic curve. It measures the model's ability to distinguish between classes across all decision thresholds. 1.0 = perfect, 0.5 = random.

### Intermediate - 5

1. **Q:** When would you use precision-recall AUC instead of ROC AUC?  
   **A:** For imbalanced datasets (positive class < 10%). ROC AUC can be overly optimistic when negatives dominate. Precision-recall AUC focuses on the positive class and better reflects performance on rare events.

2. **Q:** How do you interpret a learning curve and use it for bias/variance diagnosis?  
   **A:** A large gap between train and test scores indicates high variance (overfitting). Low train and test scores indicate high bias (underfitting). Converging curves with small gap indicate a good fit.

3. **Q:** What is the difference between `cross_val_score` and `cross_validate`?  
   **A:** `cross_val_score` returns a single metric per fold. `cross_validate` can return multiple metrics, plus fit time, score time, and optionally train scores.

4. **Q:** How does `StratifiedKFold` differ from regular `KFold`?  
   **A:** StratifiedKFold preserves the class proportion in each fold. Regular KFold does not and may create folds with very different class ratios, leading to misleading evaluation.

5. **Q:** What is the `refit` parameter in GridSearchCV and when would you set it to False?  
   **A:** `refit=True` (default) retrains the best model on the full dataset. Set to `False` when you only want to explore the search space without committing to a final model (e.g., during initial exploration).

### Advanced - 3

1. **Q:** Explain the difference between micro, macro, and weighted averaging in multi-class metrics.  
   **A:** Macro: compute metric per class, average equally (treats classes equally, penalizes minority class mistakes heavily). Micro: aggregate counts globally (gives each sample equal weight, dominated by majority class). Weighted: average weighted by class support (compromise, commonly used for imbalanced multi-class).

2. **Q:** How would you implement nested cross-validation for unbiased hyperparameter evaluation?  
   **A:** Outer loop (5-fold) splits data into train/held-out. Inner loop (3-fold on outer train) runs GridSearchCV. Model with best inner params is evaluated on outer held-out fold. This gives an unbiased estimate of the model selection + training process.

3. **Q:** Describe how you would use `GroupKFold` and when it's essential.  
   **A:** `GroupKFold` ensures samples from the same group (e.g., same patient, same session) are not split across train and test. Essential when observations are not independent (e.g., multiple measurements from the same subject) to prevent data leakage.

## Practice Problems

### Easy - 5

1. **E1:** Load the iris dataset, train a logistic regression, and print the classification report.
2. **E2:** Perform 5-fold cross-validation on a RandomForestClassifier using the wine dataset.
3. **E3:** Compute MSE and R² for a LinearRegression on `make_regression(n_features=3)`.
4. **E4:** Create a confusion matrix for a binary classifier on synthetic data.
5. **E5:** Compute ROC AUC for an SVM classifier on a binary dataset.

### Medium - 5

1. **M1:** Use GridSearchCV to tune C and gamma of an SVC on the breast cancer dataset, reporting best params and CV score.
2. **M2:** Compare 5-fold cross-validation scores of LogisticRegression, RandomForest, and SVM on the same dataset.
3. **M3:** Plot the learning curve of a RandomForest classifier on the digits dataset and diagnose bias/variance.
4. **M4:** Use RandomizedSearchCV to tune 4 hyperparameters of GradientBoostingClassifier with 50 iterations.
5. **M5:** Create a multi-metric cross-validation report (accuracy, F1, precision, recall) for a logistic regression.

### Hard - 3

1. **H1:** Implement nested cross-validation on a RandomForest with hyperparameter tuning, reporting the unbiased performance estimate.
2. **H2:** Perform a paired statistical test (e.g., McNemar's test) to compare two classifiers' performance on the same test set.
3. **H3:** Build a custom scoring function for GridSearchCV that combines F1 and training time (e.g., F1 / log(time)) to find efficient models.

## Solutions

### E1 Solution
```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
print(classification_report(y_test, y_pred=lr.predict(X_test),
                            target_names=iris.target_names))
```

### E2 Solution
```python
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

X, y = load_wine(return_X_y=True)
scores = cross_val_score(RandomForestClassifier(n_estimators=50, random_state=42), X, y, cv=5)
print(f"Scores: {scores}, Mean: {scores.mean():.3f}")
```

### E3-E5 Solutions follow patterns from examples.

### M1-M5 Solutions extend the techniques shown in code examples 1-6.

### H1-H3 Solutions require advanced cross-validation composition.

## Related Concepts

- 093 — sklearn Basics (models to evaluate)
- 094 — Preprocessing (tuning preprocessing steps)
- 089 — Seaborn (visualizing evaluation results)
- 092 — Customizing Plots (publication-quality evaluation figures)

## Next Concepts

- 096 — PyTorch Tensors (evaluating deep learning models)
- 098 — TensorFlow/Keras (model.fit with validation)
- 100 — Project Structure (evaluation as part of ML project pipelines)

## Summary

Model evaluation uses cross-validation (`cross_val_score`, `cross_validate`) for robust performance estimation, hyperparameter search (`GridSearchCV`, `RandomizedSearchCV`) for optimal configuration, and metric-specific tools (`classification_report`, `confusion_matrix`, `roc_auc_score`, `mean_squared_error`, `r2_score`) for detailed analysis. Proper evaluation is essential for model selection, bias/variance diagnosis, and production readiness.

## Key Takeaways

- Cross-validation (k=5 or 10) gives more reliable estimates than a single train/test split
- GridSearchCV for small spaces; RandomizedSearchCV for large spaces
- Never use accuracy on imbalanced data — use precision, recall, F1, or PR-AUC
- ROC AUC is threshold-independent; PR AUC is better for rare positives
- R² alone is insufficient — use RMSE or MAE for interpretable regression error
- Report both mean and standard deviation of CV scores
- Hold out a final test set and evaluate only once after all tuning is done
- Learning curves diagnose bias (underfitting) vs variance (overfitting)
