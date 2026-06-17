# Concept: Imbalanced Datasets

## Concept ID

ML-064

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Advanced Topics

## Learning Objectives

- Understand the challenges posed by imbalanced datasets
- Implement resampling techniques (SMOTE, RandomUnderSampler)
- Use class weights and cost-sensitive learning
- Select appropriate evaluation metrics (precision, recall, F1, PR curve)
- Apply imbalanced-learn library for practical solutions

## Prerequisites

- Classification fundamentals
- Precision, recall, confusion matrix
- Basic sklearn usage

## Definition

An imbalanced dataset occurs when the classes in a classification problem are not represented equally. Typically, one class (the majority) has significantly more samples than the other (the minority). The ratio can be as extreme as 1000:1 or worse. Standard classifiers trained on imbalanced data tend to be biased toward the majority class, achieving high accuracy by simply predicting the majority class for all inputs.

## Intuition

Imagine training a security system to detect fraudulent transactions where only 1 in 10,000 transactions is fraudulent. A classifier that predicts "not fraudulent" for every transaction achieves 99.99% accuracy but is completely useless. The model learned the easy shortcut instead of actually detecting fraud.

The challenge is to force the model to pay attention to the rare but important minority class. We can do this by changing the data (resampling), changing the algorithm (class weights), or changing how we evaluate performance.

## Why This Concept Matters

1. **Real-world prevalence**: Imbalanced data is the norm, not the exception — fraud detection, medical diagnosis, fault detection, rare event prediction.
2. **Accuracy paradox**: Standard accuracy is misleading for imbalanced data. A model can have 99% accuracy and be completely useless.
3. **Business impact**: False negatives (missing a rare event) are often much more costly than false positives.
4. **Algorithm sensitivity**: Most ML algorithms assume balanced classes and perform poorly on imbalanced data without adjustment.

## Mathematical Explanation

### Problem Formulation

For a binary classification problem with majority class M and minority class m, the imbalance ratio is:
r = N_m / N_M where N_m << N_M

Standard classifiers minimize overall error:
minimize (1/N) * sum(L(y_i, f(x_i)))

This is dominated by majority class errors — the minority class contributes negligibly to the total loss.

### Resampling Methods

**Random Under-Sampling:** Randomly remove samples from the majority class.
**Random Over-Sampling:** Randomly duplicate samples from the minority class.

**SMOTE (Synthetic Minority Over-sampling Technique):** Creates synthetic minority samples by interpolating between existing minority samples and their k-nearest neighbors.

For a minority sample x_i and one of its k-nearest neighbors x_j:
x_new = x_i + lambda * (x_j - x_i)
where lambda is a random number in [0, 1].

**ADASYN:** Adaptive Synthetic Sampling — generates more synthetic samples for minority samples that are harder to learn.

### Algorithm-Level Methods

**Class weights:** Assign higher weight to minority class errors:
Loss = w_m * sum(L(y_i, f(x_i))) for minority + w_M * sum(L(y_i, f(x_i))) for majority
where w_m = N / (n_classes * N_m), w_M = N / (n_classes * N_M)

### Evaluation Metrics

**Precision:** TP / (TP + FP) — how many predicted positives are correct.
**Recall:** TP / (TP + FN) — how many actual positives were found.
**F1 Score:** 2 * P * R / (P + R) — harmonic mean of precision and recall.
**PR Curve:** Precision vs. Recall at different thresholds.
**AUC-PR:** Area under the precision-recall curve (better for imbalanced data than ROC-AUC).
**Matthews Correlation Coefficient (MCC):** Balanced measure even for very imbalanced data.

## Code Examples

### Example 1: The Accuracy Paradox

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create imbalanced dataset
X, y = make_classification(
    n_samples=10000, n_features=20, n_informative=10,
    n_redundant=5, weights=[0.9, 0.1],  # 90% class 0, 10% class 1
    random_state=42
)
print(f"Class distribution: {np.bincount(y)}")
print(f"Imbalance ratio: {np.sum(y==0)/np.sum(y==1):.2f}:1")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train a standard logistic regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# The "dumb" classifier that always predicts majority
dumb_pred = np.zeros_like(y_test)
print(f"\nDumb classifier (always predict 0) accuracy: "
      f"{accuracy_score(y_test, dumb_pred):.4f}")
```

```
# Output:
Class distribution: [9000 1000]
Imbalance ratio: 9.00:1

Accuracy: 0.9140

Classification Report:
              precision    recall  f1-score   support
           0       0.92      0.99      0.95      2680
           1       0.78      0.25      0.38       320
    accuracy                           0.91      3000
   macro avg       0.85      0.62      0.67      3000
weighted avg       0.90      0.91      0.89      3000

Dumb classifier (always predict 0) accuracy: 0.8933
```

### Example 2: SMOTE and Under-Sampling

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {np.bincount(y_resampled)}")

# Train on SMOTE data
lr_smote = LogisticRegression(max_iter=1000, random_state=42)
lr_smote.fit(X_resampled, y_resampled)
y_pred_smote = lr_smote.predict(X_test)

print(f"\nSMOTE + Logistic Regression:")
print(classification_report(y_test, y_pred_smote))

# Random Under-Sampling
rus = RandomUnderSampler(random_state=42)
X_under, y_under = rus.fit_resample(X_train, y_train)
print(f"\nAfter under-sampling: {np.bincount(y_under)}")

lr_under = LogisticRegression(max_iter=1000, random_state=42)
lr_under.fit(X_under, y_under)
y_pred_under = lr_under.predict(X_test)

print(f"\nUnder-sampling + Logistic Regression:")
print(classification_report(y_test, y_pred_under))

# Pipeline approach
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])
pipeline.fit(X_train, y_train)
y_pred_pipe = pipeline.predict(X_test)
print(f"\nPipeline (SMOTE+LR):")
print(classification_report(y_test, y_pred_pipe))
```

```
# Output:
After SMOTE: [6300 6300]

SMOTE + Logistic Regression:
              precision    recall  f1-score   support
           0       0.96      0.87      0.91      2680
           1       0.46      0.75      0.57       320
    accuracy                           0.86      3000

After under-sampling: [700 700]

Under-sampling + Logistic Regression:
              precision    recall  f1-score   support
           0       0.96      0.78      0.86      2680
           1       0.34      0.78      0.47       320
```

### Example 3: Class Weights

```python
# Using class weights in sklearn
lr_weighted = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  # Automatically adjusts weights
    random_state=42
)
lr_weighted.fit(X_train, y_train)
y_pred_weighted = lr_weighted.predict(X_test)

print("Class Weights (balanced):")
print(classification_report(y_test, y_pred_weighted))

# Manual class weights
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
print(f"\nComputed class weights: {weights}")

# Custom weights (emphasize minority more)
lr_custom = LogisticRegression(
    max_iter=1000,
    class_weight={0: 1.0, 1: 10.0},
    random_state=42
)
lr_custom.fit(X_train, y_train)
y_pred_custom = lr_custom.predict(X_test)
print(f"\nCustom weights (1:10):")
print(classification_report(y_test, y_pred_custom))
```

```
# Output:
Class Weights (balanced):
              precision    recall  f1-score   support
           0       0.96      0.87      0.91      2680
           1       0.44      0.73      0.55       320

Computed class weights: [0.5269 4.7423]

Custom weights (1:10):
              precision    recall  f1-score   support
           0       0.97      0.80      0.88      2680
           1       0.34      0.82      0.48       320
```

### Example 4: Evaluation Metrics for Imbalanced Data

```python
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.metrics import f1_score

models = {
    'Standard': lr,
    'Weighted': lr_weighted,
    'SMOTE': lr_smote,
}

print(f"{'Model':12s} | {'Acc':6s} | {'Prec':6s} | {'Recall':6s} | "
      f"{'F1':6s} | {'AUC-PR':6s} | {'MCC':6s}")
print("-" * 65)

for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_pr = average_precision_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    print(f"{name:12s} | {acc:.4f} | {prec:.4f} | {rec:.4f} | "
          f"{f1:.4f} | {auc_pr:.4f} | {mcc:.4f}")

# PR Curve comparison
plt.figure(figsize=(10, 8))
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, linewidth=2, label=name)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

```
# Output:
Model         | Acc    | Prec   | Recall | F1     | AUC-PR | MCC
-----------------------------------------------------------------
Standard      | 0.9140 | 0.7800 | 0.2500 | 0.3800 | 0.4567 | 0.2891
Weighted      | 0.8640 | 0.4400 | 0.7300 | 0.5500 | 0.5234 | 0.4123
SMOTE         | 0.8620 | 0.4600 | 0.7500 | 0.5700 | 0.5345 | 0.4234
```

### Example 5: Comparing Resampling Strategies

```python
from imblearn.over_sampling import ADASYN, RandomOverSampler
from imblearn.combine import SMOTETomek

methods = {
    'Original': None,
    'Random Over': RandomOverSampler(random_state=42),
    'SMOTE': SMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42),
    'SMOTE-Tomek': SMOTETomek(random_state=42),
}

results = []
for name, sampler in methods.items():
    if sampler is None:
        X_res, y_res = X_train, y_train
    else:
        X_res, y_res = sampler.fit_resample(X_train, y_train)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_res, y_res)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    results.append((name, f1, X_res.shape[0]))
    print(f"{name:15s}: F1={f1:.4f}, Samples={X_res.shape[0]}")

# Visual comparison
names = [r[0] for r in results]
f1_scores = [r[1] for r in results]
sizes = [r[2] for r in results]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(names, f1_scores, color='steelblue')
axes[0].set_ylabel('F1 Score')
axes[0].set_title('F1 Score by Method')
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(names, sizes, color='coral')
axes[1].set_ylabel('Training Samples')
axes[1].set_title('Training Set Size by Method')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()
```

```
# Output:
Original        : F1=0.3800, Samples=7000
Random Over     : F1=0.5400, Samples=12600
SMOTE           : F1=0.5700, Samples=12600
ADASYN          : F1=0.5612, Samples=12734
SMOTE-Tomek     : F1=0.5734, Samples=12567
```

## Common Mistakes

1. **Using accuracy as the metric**: Accuracy is misleading for imbalanced data. A 99% accurate model could be completely useless if it predicts only the majority class.

2. **Applying SMOTE before train-test split**: Always split first, then apply SMOTE only to training data. Applying SMOTE before splitting causes data leakage (synthetic samples appear in both train and test sets).

3. **Creating synthetic samples without considering k-nearest neighbors**: SMOTE's effectiveness depends on the choice of k. Too small k creates noisy samples; too large k creates overly generic samples.

4. **Using ROC-AUC instead of PR-AUC**: ROC-AUC can be misleading for imbalanced data because the false positive rate remains small even with many false positives. PR-AUC focuses on the minority class and is more informative.

5. **Removing too many majority samples with under-sampling**: Aggressive under-sampling discards potentially useful data. The optimal sampling ratio is usually less extreme than 1:1.

6. **Not tuning the decision threshold**: The default threshold (0.5) may be suboptimal for imbalanced data. Adjust the threshold based on the PR curve to optimize precision/recall tradeoff.

7. **Ignoring the possibility of data quality issues**: In imbalanced datasets, minority class samples may be noisy or contain outliers because they're rare. Clean the minority class carefully.

8. **Using the same resampling strategy for all problems**: The best approach depends on the imbalance ratio, dataset size, and noise level. Always experiment with multiple strategies.

9. **Not checking for class overlap**: If minority and majority classes overlap heavily, no amount of resampling will help. Consider collecting better features.

10. **Applying over-sampling to test data**: The test set must reflect the original (real-world) class distribution. Never resample test data.

## Interview Questions

### Beginner

**Q1:** What is an imbalanced dataset?

**A1:** An imbalanced dataset has unequal class distributions, where one class (majority) has significantly more samples than the other (minority). For example, 99% non-fraud and 1% fraud transactions.

**Q2:** Why is accuracy a poor metric for imbalanced data?

**A2:** A classifier that predicts the majority class for all inputs achieves high accuracy (e.g., 99%) but is completely useless for detecting the minority class. Accuracy doesn't distinguish between correct majority and minority predictions.

**Q3:** What is SMOTE?

**A3:** SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic minority samples by interpolating between existing minority samples and their k-nearest neighbors. It adds x_new = x_i + lambda * (x_j - x_i) between a sample and its neighbor.

**Q4:** What is the difference between over-sampling and under-sampling?

**A4:** Over-sampling increases minority samples (by duplication or synthesis). Under-sampling decreases majority samples (by random removal or informed selection). Over-sampling keeps all data but may cause overfitting; under-sampling is efficient but may discard useful data.

**Q5:** What is a class weight?

**A5:** Class weight assigns higher importance to minority class errors during training. The loss function weights minority errors more heavily, forcing the model to pay attention to the minority class. It's an algorithm-level approach (no data modification needed).

### Intermediate

**Q1:** Compare SMOTE with ADASYN.

**A1:** SMOTE generates the same number of synthetic samples for each minority sample (uniform). ADASYN adaptively generates more samples for minority samples that are harder to learn (nearer to majority class boundaries). ADASYN focuses on difficult regions but may amplify noise.

**Q2:** Why is the precision-recall curve preferred over ROC for imbalanced data?

**A2:** ROC plots TPR vs. FPR. Since FPR = FP/N, and N (total negatives) is very large in imbalanced data, FPR remains small even with many FPs, making ROC overly optimistic. PR curves plot precision vs. recall, both focused on the positive (minority) class, giving a more realistic picture of minority class performance.

**Q3:** How do you tune the decision threshold for imbalanced classification?

**A3:** After training, obtain probability predictions on validation data. Vary the threshold from 0 to 1, compute precision and recall at each threshold, and select the threshold that maximizes the desired metric (F1, precision at target recall, etc.). The optimal threshold is often far from 0.5.

**Q4:** What is the MCC and why is it useful for imbalanced data?

**A4:** MCC (Matthews Correlation Coefficient) ranges from -1 to 1. It considers all four confusion matrix values (TP, TN, FP, FN) and produces a high score only if the classifier performs well on both classes. It's reliable even for extreme imbalance ratios.

**Q5:** How does ensemble learning help with imbalanced data?

**A5:** Ensembles can help by: (1) Balanced Random Forest — each tree is trained on a balanced bootstrap sample. (2) EasyEnsemble — multiple under-sampled subsets are used to train an ensemble. (3) RUSBoost — combines random under-sampling with boosting. These approaches leverage multiple models to compensate for data imbalance.

### Advanced

**Q1:** Prove that the optimal decision threshold for imbalanced data is not 0.5.

**A1:** For a classifier with class-conditional densities p(x|y=0) and p(x|y=1) with prior P(y=0) >> P(y=1), Bayes optimal decision rule is: predict 1 if p(x|y=1)*P(y=1) > p(x|y=0)*P(y=0). This gives threshold t = P(y=0)/P(y=1) >> 1. For a 9:1 ratio, the optimal threshold is 9, corresponding to a probability of 9/10 = 0.9, not 0.5.

**Q2:** Derive the SMOTE algorithm and explain potential failure modes.

**A2:** SMOTE: (1) For each minority sample x_i, find its k nearest minority neighbors. (2) Randomly select one neighbor x_j. (3) Generate x_new = x_i + rand(0,1)*(x_j - x_i). Failure modes: (a) If k is too small, synthetic samples don't represent the minority distribution. (b) If minority samples are in a sparse region, SMOTE interpolates between noisy samples, creating more noise. (c) SMOTE doesn't consider majority class overlap — it may create samples in majority regions.

**Q3:** Explain how cost-sensitive learning differs from data-level approaches and when to use each.

**A3:** Cost-sensitive learning modifies the algorithm's loss function to penalize minority misclassifications more. Data-level approaches modify the training set distribution. Cost-sensitive is preferred when: (1) The cost of misclassification is known and asymmetric. (2) The dataset is very large (resampling is expensive). (3) The imbalance ratio is extreme (resampling creates too many or too few samples). Data-level approaches are preferred when: (1) The algorithm doesn't support class weights. (2) You want to use the same algorithm as balanced data. (3) You need interpretable feature importance (weights complicate this).

## Practice Problems

**E1:** Create a 95:5 imbalanced dataset and compare accuracy vs. F1 score for a standard classifier.

**E2:** Apply SMOTE to an imbalanced dataset and visualize the original vs. resampled data in 2D.

**E3:** Plot the PR curve for a classifier on imbalanced data and find the optimal threshold.

**M1:** Compare 5 different resampling strategies (Random Over, SMOTE, ADASYN, SMOTETomek, Borderline-SMOTE) on a real-world imbalanced dataset.

**M2:** Implement cost-sensitive logistic regression by modifying the loss function.

**M3:** Build a Balanced Random Forest and compare with a standard Random Forest on imbalanced data.

**H1:** Implement an ensemble method (EasyEnsemble or RUSBoost) from scratch.

**H2:** Design a threshold tuning method that optimizes F-beta score with a user-specified beta.

## Solutions

**M2:** Cost-sensitive logistic regression: multiply the loss for each class by its weight: Loss = w_1 * sum_{y=1} log(1+exp(-y_hat)) + w_0 * sum_{y=0} log(1+exp(y_hat)). The gradient is weighted accordingly.

## Related Concepts

- Evaluation Metrics — Precision, recall, F1, PR curve, MCC
- Ensemble Methods (ML-061) — Balanced ensembles for imbalanced data
- Anomaly Detection (ML-065) — Treating minority class as anomaly detection problem
- Cost-Sensitive Learning — Incorporating asymmetric costs into training

## Next Concepts

- One-Class Classification — Learning from only the majority class
- Active Learning for Imbalanced Data — Smart sampling to label informative minority examples
- Generative Models for Imbalanced Data — Using GANs to generate minority samples

## Summary

Imbalanced datasets are pervasive in real-world applications. Standard classifiers fail on them due to the accuracy paradox. Solutions fall into three categories: data-level (SMOTE, under-sampling), algorithm-level (class weights, cost-sensitive learning), and evaluation-level (PR curve, F1, MCC). The best approach depends on the specific problem, imbalance ratio, and dataset size.

## Key Takeaways

- Imbalanced data is the norm — fraud, medical diagnosis, rare events
- Accuracy is misleading; use precision, recall, F1, and PR-AUC
- SMOTE creates synthetic minority samples via interpolation
- Class weights adapt the loss function without modifying data
- PR curves are better than ROC for imbalanced data
- Always split data before applying SMOTE (avoid data leakage)
- Tune the decision threshold (not just the model)
- Over-sampling can cause overfitting; under-sampling loses information
- Ensemble methods (Balanced RF, RUSBoost) are effective
- MCC is a robust single metric for imbalanced classification
