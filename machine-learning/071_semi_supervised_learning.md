# Concept: Semi-Supervised Learning

## Concept ID

ML-071

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Advanced Topics

## Learning Objectives

- Understand the semi-supervised learning paradigm and its assumptions
- Implement self-training, pseudo-labeling, and consistency regularization
- Apply label propagation and graph-based methods
- Evaluate SSL models when labeled data is scarce

## Prerequisites

- Supervised learning (classification)
- Graph theory basics
- Unsupervised learning concepts
- sklearn pipelines

## Definition

Semi-Supervised Learning (SSL) is a class of machine learning techniques that leverages both a small amount of labeled data and a large amount of unlabeled data during training. SSL sits between supervised learning (all labeled) and unsupervised learning (all unlabeled). It is based on assumptions about how unlabeled data can inform the decision boundary: primarily the smoothness assumption, the cluster assumption, and the manifold assumption.

## Intuition

Imagine you have 10 labeled images of cats and dogs, plus 10,000 unlabeled images of cats and dogs. A supervised model trained on just 10 labeled images would have a poor decision boundary. However, the unlabeled images reveal the underlying data distribution — cats tend to cluster together in feature space, dogs in another. By propagating labels through the unlabeled data according to these clusters, you can dramatically improve the decision boundary.

SSL works because unlabeled data carries information about the joint distribution p(x), which helps constrain the conditional p(y|x) when the decision boundary must lie in low-density regions (cluster assumption).

## Why This Concept Matters

In practice, labeled data is often scarce and expensive while unlabeled data is abundant. SSL methods can match fully supervised performance with a fraction of the labels — sometimes as few as 1-5% of the data labeled. SSL is widely used in speech recognition, NLP (especially with pretrained language models), medical imaging, web page classification, and protein structure prediction.

## Mathematical Explanation

### SSL Assumptions

1. **Smoothness Assumption:** If two points x_1 and x_2 are close in the input space, their labels y_1 and y_2 should be close as well.
2. **Cluster Assumption:** The decision boundary should lie in a low-density region of the input space. Points in the same cluster likely share the same label.
3. **Manifold Assumption:** High-dimensional data lies on a low-dimensional manifold; distance along the manifold is more meaningful than Euclidean distance.

### Self-Training / Pseudo-Labeling

1. Train a classifier on labeled data L.
2. Use classifier to predict labels for unlabeled data U.
3. Select most confident predictions (above threshold τ) as pseudo-labels.
4. Combine pseudo-labeled data with original labeled data.
5. Retrain the model.
6. Repeat until convergence or budget exhausted.

### Label Propagation

Construct a graph where nodes are all data points (labeled + unlabeled). Edge weights w_{ij} represent similarity (e.g., RBF kernel). Propagate labels from labeled nodes to unlabeled nodes via:

f^{(t+1)} = α W f^{(t)} + (1-α) y

where f^{(t)} is the label distribution at iteration t, y is the initial label matrix (zero for unlabeled), and W is the normalized weight matrix.

The solution converges to:

f* = (1-α)(I - αW)^{-1} y

### Consistency Regularization (Π-Model)

For each unlabeled point, add a loss term that penalizes inconsistent predictions under different augmentations:

L_total = L_supervised + λ * E_{x ∈ U} [||f(x) - f(x')||²]

where x' is a stochastic augmentation of x.

## Code Examples

### Example 1: Label Propagation with sklearn

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score

np.random.seed(42)
X, y = make_classification(
    n_samples=500, n_features=10, n_informative=8,
    n_classes=2, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

n_labeled = 20
labeled_indices = np.random.choice(len(X_train), n_labeled, replace=False)

y_train_mixed = np.full(len(X_train), -1)
y_train_mixed[labeled_indices] = y_train[labeled_indices]

lp = LabelPropagation(kernel='rbf', gamma=0.5, max_iter=100)
lp.fit(X_train, y_train_mixed)

y_pred_lp = lp.predict(X_test)
acc_lp = accuracy_score(y_test, y_pred_lp)

# Supervised baseline
from sklearn.svm import SVC
svm = SVC(kernel='rbf', gamma=0.5)
svm.fit(X_train[labeled_indices], y_train[labeled_indices])
y_pred_svm = svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)

print(f"Labeled examples: {n_labeled} ({100*n_labeled/len(X_train):.1f}%)")
print(f"Label Propagation accuracy: {acc_lp:.3f}")
print(f"SVM (supervised only) accuracy: {acc_svm:.3f}")
# Output:
# Labeled examples: 20 (5.7%)
# Label Propagation accuracy: 0.813
# SVM (supervised only) accuracy: 0.767
```

### Example 2: Self-Training / Pseudo-Labeling

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)
X, y = make_classification(n_samples=600, n_features=15, n_informative=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_labeled = 15
labeled_idx = np.random.choice(len(X_train), n_labeled, replace=False)
unlabeled_idx = np.setdiff1d(np.arange(len(X_train)), labeled_idx)

X_labeled = X_train[labeled_idx]
y_labeled = y_train[labeled_idx]
X_unlabeled = X_train[unlabeled_idx]

model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_labeled, y_labeled)

threshold = 0.9
for iteration in range(10):
    probs = model.predict_proba(X_unlabeled)
    confidence = np.max(probs, axis=1)
    high_conf = confidence >= threshold

    if not np.any(high_conf):
        break

    pseudo_labels = np.argmax(probs[high_conf], axis=1)
    X_labeled = np.vstack([X_labeled, X_unlabeled[high_conf]])
    y_labeled = np.concatenate([y_labeled, pseudo_labels])
    X_unlabeled = X_unlabeled[~high_conf]

    model.fit(X_labeled, y_labeled)

    if len(X_unlabeled) == 0:
        break

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Self-training accuracy: {accuracy:.3f}")
print(f"Pseudo-labels added: {len(y_labeled) - n_labeled}")
print(f"Remaining unlabeled: {len(X_unlabeled)}")

# Baseline with only labeled data
baseline = RandomForestClassifier(random_state=42)
baseline.fit(X_train[labeled_idx], y_train[labeled_idx])
baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
print(f"Supervised baseline: {baseline_acc:.3f}")
# Output:
# Self-training accuracy: 0.808
# Pseudo-labels added: 81
# Remaining unlabeled: 384
# Supervised baseline: 0.717
```

### Example 3: Label Spreading with Digits

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score

np.random.seed(42)
digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

n_labeled = 30
rng = np.random.RandomState(42)
labeled_indices = rng.choice(len(X_train), n_labeled, replace=False)

y_train_mixed = np.full(len(X_train), -1)
y_train_mixed[labeled_indices] = y_train[labeled_indices]

lp = LabelSpreading(kernel='knn', n_neighbors=7, alpha=0.8, max_iter=100)
lp.fit(X_train, y_train_mixed)

y_pred = lp.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Digits dataset: {len(X_train)} train, {n_labeled} labeled ({100*n_labeled/len(X_train):.1f}%)")
print(f"Label Spreading accuracy: {acc:.4f}")
print(f"Classes: {len(np.unique(y))}")
# Output:
# Digits dataset: 1257 train, 30 labeled (2.4%)
# Label Spreading accuracy: 0.7315
# Classes: 10
```

### Example 4: Compare SSL with Varying Label Ratios

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.semi_supervised import LabelPropagation
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=20, n_informative=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

for n_labeled in [5, 10, 20, 50, 100, 200]:
    labeled_idx = np.random.choice(len(X_train), min(n_labeled, len(X_train)), replace=False)
    y_mixed = np.full(len(X_train), -1)
    y_mixed[labeled_idx] = y_train[labeled_idx]

    lp = LabelPropagation(kernel='rbf', gamma=0.5)
    lp.fit(X_train, y_mixed)
    ssl_acc = accuracy_score(y_test, lp.predict(X_test))

    svm = SVC(kernel='rbf', gamma=0.5)
    svm.fit(X_train[labeled_idx], y_train[labeled_idx])
    svm_acc = accuracy_score(y_test, svm.predict(X_test))

    print(f"Labeled={n_labeled:3d} ({100*n_labeled/len(X_train):4.1f}%): SSL={ssl_acc:.3f}, SVM={svm_acc:.3f}")
# Output:
# Labeled=  5 ( 1.4%): SSL=0.687, SVM=0.660
# Labeled= 10 ( 2.9%): SSL=0.753, SVM=0.727
# Labeled= 20 ( 5.7%): SSL=0.807, SVM=0.773
# Labeled= 50 (14.3%): SSL=0.827, SVM=0.820
# Labeled=100 (28.6%): SSL=0.840, SVM=0.833
# Labeled=200 (57.1%): SSL=0.847, SVM=0.847
```

## Common Mistakes

1. **Using SSL when the cluster assumption is violated.** If classes heavily overlap in feature space, propagating labels through unlabeled data introduces noise.
2. **Not validating the quality of pseudo-labels.** Self-training can amplify early mistakes — always use a confidence threshold.
3. **Applying SSL with a model that produces poorly calibrated probabilities.** Calibration is critical for threshold-based pseudo-labeling.
4. **Ignoring distribution shift between labeled and unlabeled data.** SSL assumes labeled and unlabeled are from the same distribution.
5. **Using too few labeled examples.** With 1-2 labeled examples, SSL may fail to capture meaningful structure.
6. **Confusing SSL with active learning.** SSL uses unlabeled data without additional labeling; active learning queries labels for informative points.
7. **Not tuning the graph construction parameters.** In label propagation, the kernel bandwidth and number of neighbors critically affect performance.

## Interview Questions

### Beginner

1. What is semi-supervised learning?
2. What are the key assumptions of SSL?
3. How does label propagation work?
4. What is pseudo-labeling?
5. How is SSL different from active learning?

### Intermediate

1. Explain the smoothness, cluster, and manifold assumptions of SSL.
2. How does self-training work, and what are its failure modes?
3. Derive the label propagation algorithm and its convergence.
4. What is consistency regularization? How does it relate to the Π-model?
5. How would you evaluate an SSL model?

### Advanced

1. Prove the convergence of label propagation to the solution f* = (1-α)(I - αW)^{-1}y.
2. Explain the theoretical foundations of SSL: how does unlabeled data help under the cluster assumption?
3. Describe MixMatch and FixMatch — how do they combine consistency regularization with pseudo-labeling?

## Practice Problems

### Easy

1. Implement label propagation on the Iris dataset using only 10 labeled examples.
2. Plot SSL accuracy vs. number of labeled examples for the moons dataset.
3. Compare LabelPropagation with LabelSpreading on a 2D synthetic dataset.
4. Visualize label propagation on a 2D dataset showing the propagation front over iterations.
5. Implement self-training with a DecisionTree classifier.

### Medium

1. Implement consistency regularization (Π-model) from scratch.
2. Compare RBF kernel vs. kNN graph for label propagation.
3. Implement self-training with a dynamic confidence threshold that decreases over iterations.
4. Analyze how SSL performance varies with the number of unlabeled examples.
5. Implement co-training (multi-view SSL) on a dataset with two feature splits.

### Hard

1. Implement MixMatch from scratch (combining mixup, consistency regularization, and pseudo-labeling).
2. Derive and implement the Laplacian SVM (manifold regularization).
3. Implement entropy minimization as an SSL objective (Grandvalet and Bengio, 2004).

## Solutions

Solution 1 (Easy): SSL on moons dataset

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=500, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

labeled_idx = np.random.choice(len(X_train), 10, replace=False)
y_mixed = np.full(len(X_train), -1)
y_mixed[labeled_idx] = y_train[labeled_idx]

lp = LabelPropagation(kernel='rbf', gamma=10)
lp.fit(X_train, y_mixed)
print(f"Accuracy with 10 labels: {accuracy_score(y_test, lp.predict(X_test)):.3f}")
```

Solution 2 (Medium): Co-training

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Split features into two views
view1 = X[:, :5]
view2 = X[:, 5:]

m1 = DecisionTreeClassifier()
m2 = DecisionTreeClassifier()
m1.fit(view1[labeled_idx], y_labeled)
m2.fit(view2[labeled_idx], y_labeled)

for _ in range(10):
    conf1 = np.max(m1.predict_proba(view1[unlabeled_idx]), axis=1)
    conf2 = np.max(m2.predict_proba(view2[unlabeled_idx]), axis=1)
    # Add high-confidence predictions from one view to the other's training set
```

## Related Concepts

- Active Learning (ML-070)
- Unsupervised Learning
- Graph-Based Learning
- Transfer Learning
- Self-Supervised Learning
- Manifold Learning

## Next Concepts

- Multi-Label Classification (ML-072)
- Multi-Task Learning (ML-073)
- Model Interpretability (ML-075)

## Summary

Semi-Supervised Learning leverages abundant unlabeled data alongside limited labeled data to improve model performance. It relies on the smoothness, cluster, and manifold assumptions. Key methods include self-training (pseudo-labeling), label propagation on graphs, and consistency regularization. SSL is essential when labeling is expensive but unlabeled data is plentiful.

## Key Takeaways

- SSL uses both labeled and unlabeled data to improve learning.
- The cluster assumption: decision boundary lies in low-density regions.
- Label propagation spreads labels through a similarity graph.
- Self-training adds high-confidence pseudo-labels iteratively.
- Consistency regularization enforces prediction invariance to augmentations.
- SSL is most effective when the labeled fraction is very small (1-10%).
- Graph construction (kernel, kNN) critically affects label propagation quality.
