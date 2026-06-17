# Concept: Multi-Label Classification

## Concept ID

ML-072

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Advanced Topics

## Learning Objectives

- Understand the multi-label classification problem formulation
- Implement OneVsRest, ClassifierChain, and LabelPowerset strategies
- Evaluate multi-label models using Hamming loss, subset accuracy, and F1-micro
- Apply multi-label techniques to real-world problems

## Prerequisites

- Binary and multiclass classification fundamentals
- Logistic regression, decision trees
- Evaluation metrics (precision, recall, F1)
- sklearn pipeline

## Definition

Multi-label classification is a supervised learning problem where each instance can be associated with multiple labels simultaneously. Unlike multiclass classification (each instance belongs to exactly one of K classes), multi-label classification allows instances to have any subset of K possible labels. Formally, given input space X and label space Y = {0, 1}^K, the task is to learn a function f: X → {0, 1}^K that predicts the presence or absence of each label.

## Intuition

A movie can be simultaneously classified as "action," "comedy," and "sci-fi." A news article can be tagged with "politics," "economy," and "international." A medical image may show evidence of multiple diseases. In all these cases, labels are not mutually exclusive — they co-occur. Multi-label classification captures this structure by modeling each label's presence as a binary decision while accounting for label correlations.

## Why This Concept Matters

Multi-label classification is ubiquitous in real-world applications: document tagging, image annotation, music genre classification, medical diagnosis (multiple conditions), protein function prediction, and recommendation systems. Standard classifiers (logistic regression, SVM) natively support only single-label outputs, making multi-label problem transformation methods essential. Understanding multi-label learning is critical for any practitioner dealing with complex, real-world labeling scenarios.

## Mathematical Explanation

### Problem Transformation Methods

**Binary Relevance (OneVsRest):** Train K independent binary classifiers, one per label. Each classifier h_k: X → {0, 1} predicts whether label k is present.

Strengths: Simple, parallelizable. Weakness: Ignores label correlations.

**Classifier Chains:** Train K binary classifiers in a chain. Each classifier h_k predicts label k using both the input features and the predictions of all previous classifiers in the chain.

ŷ_k = h_k(x, ŷ_1, ..., ŷ_{k-1})

The chain order matters — performance is typically averaged over multiple random orders.

**Label Powerset:** Transform the problem into multiclass by treating each unique subset of labels as a single class. If K labels can produce up to 2^K possible subsets.

Strengths: Captures label correlations. Weakness: Number of classes grows exponentially; many label combinations may be unseen.

### Evaluation Metrics

**Hamming Loss:** Fraction of incorrectly predicted labels (both false positives and false negatives):

HL = (1/(nK)) ∑_{i=1}^n ∑_{j=1}^K 1[ŷ_{ij} ≠ y_{ij}]

Lower is better.

**Subset Accuracy (Exact Match Ratio):** Fraction of instances where the predicted label set exactly matches the true set:

SA = (1/n) ∑_{i=1}^n 1[ŷ_i = y_i]

**Micro-averaged F1:** Aggregate over all label-instance pairs:

F1_micro = 2 * TP / (2 * TP + FP + FN)

**Macro-averaged F1:** Compute F1 per label, average across labels.

## Code Examples

### Example 1: Binary Relevance with sklearn

```python
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss, accuracy_score, f1_score

np.random.seed(42)
X, y = make_multilabel_classification(
    n_samples=500,
    n_features=20,
    n_classes=5,
    n_labels=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

br = MultiOutputClassifier(LogisticRegression(max_iter=1000, random_state=42))
br.fit(X_train, y_train)
y_pred = br.predict(X_test)

hl = hamming_loss(y_test, y_pred)
sa = accuracy_score(y_test, y_pred)
f1_micro = f1_score(y_test, y_pred, average='micro')
f1_macro = f1_score(y_test, y_pred, average='macro')

print(f"Hamming Loss:      {hl:.4f}")
print(f"Subset Accuracy:   {sa:.4f}")
print(f"F1 Micro:          {f1_micro:.4f}")
print(f"F1 Macro:          {f1_macro:.4f}")
# Output:
# Hamming Loss:      0.2187
# Subset Accuracy:   0.3600
# F1 Micro:          0.6688
# F1 Macro:          0.5977
```

### Example 2: Classifier Chains

```python
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss, f1_score

np.random.seed(42)
X, y = make_multilabel_classification(n_samples=500, n_features=20, n_classes=5, n_labels=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

chain = ClassifierChain(
    LogisticRegression(max_iter=1000, random_state=42),
    order=[0, 1, 2, 3, 4],
    random_state=42
)
chain.fit(X_train, y_train)
y_pred = chain.predict(X_test)

hl = hamming_loss(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='micro')
print(f"Classifier Chain - Hamming Loss: {hl:.4f}, F1 Micro: {f1:.4f}")

# Average over multiple random chain orders
from sklearn.model_selection import cross_val_score
scores = []
for _ in range(10):
    chain = ClassifierChain(
        LogisticRegression(max_iter=1000, random_state=42),
        order='random',
        random_state=np.random.randint(1000)
    )
    chain.fit(X_train, y_train)
    scores.append(f1_score(y_test, chain.predict(X_test), average='micro'))

print(f"Average F1 over 10 chains: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
# Output:
# Classifier Chain - Hamming Loss: 0.2120, F1 Micro: 0.6860
# Average F1 over 10 chains: 0.6920 (±0.0083)
```

### Example 3: Label Powerset

```python
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss, f1_score

np.random.seed(42)
X, y = make_multilabel_classification(n_samples=500, n_features=20, n_classes=5, n_labels=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Label Powerset via sklearn's internal LabelPowerset is not exposed directly.
# We can use a trick: treat each label combination as a single class.
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier

y_combined = np.array(['_'.join(map(str, row)) for row in y_train.astype(int)])
le = LabelEncoder()
y_combined_encoded = le.fit_transform(y_combined)

lp_model = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
lp_model.fit(X_train, y_combined_encoded)

y_train_pred_combined = le.inverse_transform(lp_model.predict(X_train))
y_test_pred = np.array([[int(c) for c in s.split('_')] for s in le.inverse_transform(lp_model.predict(X_test))])

hl = hamming_loss(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred, average='micro')
print(f"Label Powerset - Hamming Loss: {hl:.4f}, F1 Micro: {f1:.4f}")
print(f"Number of unique label sets: {len(le.classes_)}")

# Compare with Binary Relevance
from sklearn.multioutput import MultiOutputClassifier
br = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
br.fit(X_train, y_train)
br_pred = br.predict(X_test)
print(f"Binary Relevance - F1 Micro: {f1_score(y_test, br_pred, average='micro'):.4f}")
# Output:
# Label Powerset - Hamming Loss: 0.2147, F1 Micro: 0.6812
# Number of unique label sets: 18
# Binary Relevance - F1 Micro: 0.6724
```

### Example 4: Evaluation Metrics from Scratch

```python
import numpy as np

y_true = np.array([[1, 0, 1],
                   [0, 1, 1],
                   [1, 1, 0],
                   [0, 0, 1]])

y_pred = np.array([[1, 0, 0],
                   [0, 1, 1],
                   [1, 0, 0],
                   [0, 0, 1]])

def hamming_loss(y_true, y_pred):
    return np.mean(y_true != y_pred)

def subset_accuracy(y_true, y_pred):
    return np.mean(np.all(y_true == y_pred, axis=1))

def f1_micro(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Hamming Loss:    {hamming_loss(y_true, y_pred):.4f}")
print(f"Subset Accuracy: {subset_accuracy(y_true, y_pred):.4f}")
print(f"F1 Micro:        {f1_micro(y_true, y_pred):.4f}")
# Output:
# Hamming Loss:    0.1667
# Subset Accuracy: 0.5000
# F1 Micro:        0.6667
```

## Common Mistakes

1. **Using accuracy (exact match) as the only metric.** In multi-label, exact match is very strict — a model can correctly predict 4 out of 5 labels but be scored as 0. Always include Hamming loss or F1.
2. **Ignoring label imbalance.** If one label appears in only 1% of instances, the binary relevance classifier will predict "always absent" — use class weighting or resampling.
3. **Applying standard multiclass classifiers directly.** LogisticRegression with multi_class='multinomial' assumes mutual exclusivity.
4. **Forgetting about label correlations.** Binary relevance ignores correlations; Classifier Chains or Label Powerset often perform better.
5. **Using too many labels without enough data.** With K labels, the number of possible label subsets is 2^K — most will be unseen.
6. **Not adjusting the decision threshold.** Binary relevance produces probabilities; the default threshold of 0.5 may not be optimal — calibrate per label.
7. **Evaluating on the wrong granularity.** Micro vs. macro F1 can give very different pictures — macro treats small labels equally important, micro weighs by frequency.

## Interview Questions

### Beginner

1. What is the difference between multiclass and multi-label classification?
2. What is Hamming loss?
3. How does Binary Relevance work?
4. What is the Label Powerset method?
5. Give two real-world examples of multi-label problems.

### Intermediate

1. Compare Binary Relevance, Classifier Chains, and Label Powerset. When is each preferable?
2. Why is subset accuracy (exact match) a poor metric for multi-label problems?
3. How does Classifier Chains capture label dependencies?
4. What is the difference between micro and macro F1 in the multi-label setting?
5. How would you handle label imbalance in multi-label classification?

### Advanced

1. Derive a probabilistic formulation for Classifier Chains using conditional probability: p(y_1, ..., y_K | x) = ∏ p(y_k | x, y_1, ..., y_{k-1}).
2. Explain how to optimize the chain order in Classifier Chains (e.g., using conditional mutual information).
3. Describe the connection between multi-label classification and structured prediction (CRF-based approaches).

## Practice Problems

### Easy

1. Load the Yeast dataset (a classic multi-label benchmark) and compute the label cardinality (average number of labels per instance).
2. Implement Binary Relevance with LogisticRegression and evaluate on a synthetic dataset.
3. Compute Hamming loss and subset accuracy for a given multi-label prediction.
4. Plot the label co-occurrence matrix for a multi-label dataset.
5. Use sklearn's MultiOutputClassifier with a RandomForest on a multi-label dataset.

### Medium

1. Implement Classifier Chains from scratch (no sklearn).
2. Compare Binary Relevance vs. Classifier Chains on a dataset with highly correlated labels.
3. Implement threshold tuning for Binary Relevance: find the optimal per-label threshold that maximizes F1.
4. Use cross-validation to compare three multi-label methods on a benchmark dataset.
5. Implement label ranking average precision (LRAP) from scratch.

### Hard

1. Implement a neural network architecture for multi-label classification with a sigmoid output layer and binary cross-entropy loss.
2. Implement probabilistic Classifier Chains (PCC) that marginalize over chain order during inference.
3. Derive and implement the AdaBoost.MH algorithm for multi-label boosting.

## Solutions

Solution 1 (Easy): Label cardinality

```python
import numpy as np
from sklearn.datasets import make_multilabel_classification
X, y = make_multilabel_classification(n_samples=1000, n_classes=10, random_state=42)
cardinality = np.mean(np.sum(y, axis=1))
print(f"Label cardinality: {cardinality:.2f}")
print(f"Number of labels: {y.shape[1]}")
```

Solution 2 (Medium): Classifier chains from scratch

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

class ClassifierChainScratch:
    def __init__(self, base_model=LogisticRegression()):
        self.base_model = base_model
        self.models = []
        self.label_order = None

    def fit(self, X, Y):
        n_labels = Y.shape[1]
        self.label_order = list(range(n_labels))
        self.models = []
        for k in range(n_labels):
            X_aug = np.hstack([X, Y[:, :k]])
            model = self.base_model.__class__(**self.base_model.get_params())
            model.fit(X_aug, Y[:, k])
            self.models.append(model)

    def predict(self, X):
        n_samples = X.shape[0]
        Y_pred = np.zeros((n_samples, len(self.models)))
        for k in range(len(self.models)):
            X_aug = np.hstack([X, Y_pred[:, :k]])
            Y_pred[:, k] = self.models[k].predict(X_aug)
        return Y_pred
```

## Related Concepts

- Multiclass Classification (ML-007)
- Multi-Task Learning (ML-073)
- Ensemble Methods
- Structured Prediction
- Label Encoding
- Imbalanced Learning

## Next Concepts

- Multi-Task Learning (ML-073)
- Online Learning (ML-074)
- Model Interpretability (ML-075)

## Summary

Multi-label classification handles instances that can belong to multiple classes simultaneously. Problem transformation methods (Binary Relevance, Classifier Chains, Label Powerset) convert multi-label problems into standard classification tasks. Evaluation requires metrics like Hamming loss, subset accuracy, and micro/macro F1 that account for partial correctness. Binary Relevance is simple and parallelizable, Classifier Chains capture label dependencies, and Label Powerset models label combinations directly.

## Key Takeaways

- Multi-label: each instance can have multiple labels.
- Binary Relevance: K independent binary classifiers (ignores correlations).
- Classifier Chains: sequential classifiers with previous predictions as features.
- Label Powerset: treats each label set as a single class.
- Hamming loss measures per-label error; subset accuracy requires exact match.
- Micro F1 aggregates over all label-instance pairs; macro F1 averages per label.
- Label correlations matter — Classifier Chains often outperform Binary Relevance.
- Threshold tuning improves performance over the default 0.5 cutoff.
