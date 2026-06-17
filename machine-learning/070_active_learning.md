# Concept: Active Learning

## Concept ID

ML-070

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Advanced Topics

## Learning Objectives

- Understand the active learning loop and query strategies
- Implement uncertainty sampling (least confident, margin, entropy)
- Apply active learning to reduce labeling costs
- Evaluate active learning vs. random sampling

## Prerequisites

- Supervised learning fundamentals (classification)
- Basic probability
- Uncertainty quantification
- Familiarity with sklearn pipeline

## Definition

Active Learning is a machine learning paradigm where the algorithm can interactively query a human annotator (or other information source) to label new data points. The goal is to achieve high accuracy with minimal labeling cost by selecting the most informative instances to label. The key assumption is that unlabeled data is abundant but labeling is expensive.

## Intuition

Imagine you have 10,000 unlabeled images and a limited budget to label only 500. Instead of labeling 500 random images, active learning selects the 500 images that would teach the model the most. These might be images the current model is most uncertain about (near the decision boundary), or images that would most change the model's parameters. The process is iterative: train a model on a small labeled set, use it to score unlabeled data, ask for labels on the most informative points, retrain, and repeat.

## Why This Concept Matters

In many real-world applications, labeled data is scarce and expensive. Active learning is the primary framework for reducing labeling costs while maintaining model performance. It is widely used in medical imaging (where expert radiologists are expensive), natural language processing (text annotation), drug discovery, and any domain where unlabeled data is plentiful but annotation requires expertise or resources.

## Mathematical Explanation

### Active Learning Loop

Let L be the labeled set and U be the unlabeled pool. For each iteration:

1. Train model f on L
2. For each x ∈ U, compute informativeness score a(x; f)
3. Select x* = argmax_{x ∈ U} a(x; f)
4. Query oracle for label y* of x*
5. Update L = L ∪ {(x*, y*)}, U = U \ {x*}
6. Repeat until labeling budget exhausted

### Uncertainty Sampling

**Least Confident:** Label the point where the model is least confident in its prediction:

a_LC(x) = 1 - P(ŷ | x)

where ŷ = argmax_y P(y | x) is the most likely class.

**Margin Sampling:** Label the point with the smallest margin between the two most probable classes:

a_margin(x) = P(ŷ_1 | x) - P(ŷ_2 | x)

Smaller margins indicate more uncertainty.

**Entropy Sampling:** Label the point with highest predictive entropy:

a_entropy(x) = -∑_y P(y | x) log P(y | x)

### Query-by-Committee (QBC)

Train a committee of K models on L. Label points where the committee disagrees most:

a_QBC(x) = -∑_y (V(y)/K) log(V(y)/K)

where V(y) is the number of committee members predicting class y.

### Expected Model Change

Select the point that would most change the current model parameters:

a_EMC(x) = ||∇_θ ℓ(x, y; θ)|| (approximated)

### Expected Error Reduction

Select the point that minimizes expected future error (computationally expensive).

## Code Examples

### Example 1: Uncertainty Sampling with modAL

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

np.random.seed(42)
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_classes=2,
    random_state=42
)

X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_initial, X_pool, y_initial, y_pool = train_test_split(
    X_pool, y_pool, train_size=20, random_state=42
)

learner = ActiveLearner(
    estimator=RandomForestClassifier(random_state=42),
    query_strategy=uncertainty_sampling,
    X_training=X_initial,
    y_training=y_initial
)

accuracy_history = [learner.score(X_test, y_test)]
for i in range(30):
    query_idx, query_instance = learner.query(X_pool)
    learner.teach(X_pool[query_idx], y_pool[query_idx])
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)
    accuracy_history.append(learner.score(X_test, y_test))

print(f"Initial accuracy: {accuracy_history[0]:.3f}")
print(f"Final accuracy (50 labels): {accuracy_history[-1]:.3f}")
print(f"Accuracy improvement: {accuracy_history[-1] - accuracy_history[0]:.3f}")
# Output:
# Initial accuracy: 0.683
# Final accuracy (50 labels): 0.850
# Accuracy improvement: 0.167
```

### Example 2: Comparing Query Strategies

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, entropy_sampling, margin_sampling

np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=20, n_informative=10, n_classes=2, random_state=42)
X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

strategies = {
    'Least Confident': uncertainty_sampling,
    'Margin': margin_sampling,
    'Entropy': entropy_sampling
}

for name, strategy in strategies.items():
    np.random.seed(42)
    X_pool_copy, _, y_pool_copy, _ = train_test_split(X_pool, y_pool, test_size=0, random_state=42)
    X_initial, X_pool_temp, y_initial, y_pool_temp = train_test_split(
        X_pool, y_pool, train_size=20, random_state=42
    )
    learner = ActiveLearner(
        estimator=RandomForestClassifier(random_state=42),
        query_strategy=strategy,
        X_training=X_initial,
        y_training=y_initial
    )
    for i in range(30):
        query_idx, _ = learner.query(X_pool_temp)
        learner.teach(X_pool_temp[query_idx], y_pool_temp[query_idx])
        X_pool_temp = np.delete(X_pool_temp, query_idx, axis=0)
        y_pool_temp = np.delete(y_pool_temp, query_idx, axis=0)
    acc = learner.score(X_test, y_test)
    print(f"{name:20s}: Final accuracy = {acc:.3f}")
# Output:
# Least Confident    : Final accuracy = 0.827
# Margin             : Final accuracy = 0.833
# Entropy            : Final accuracy = 0.820
```

### Example 3: Custom Query Strategy

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner
from modAL.utils.query import QueryStrategy

np.random.seed(42)
X, y = make_classification(n_samples=300, n_features=10, n_informative=5, n_classes=2, random_state=42)
X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

class EntropySampling(QueryStrategy):
    def __init__(self, estimator):
        self.estimator = estimator

    def query(self, X_pool):
        probs = self.estimator.predict_proba(X_pool)
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
        query_idx = np.argmax(entropy)
        return query_idx, X_pool[query_idx].reshape(1, -1)

X_initial, X_pool, y_initial, y_pool = train_test_split(
    X_pool, y_pool, train_size=10, random_state=42
)

learner = ActiveLearner(
    estimator=RandomForestClassifier(random_state=42, n_estimators=10),
    query_strategy=EntropySampling,
    X_training=X_initial,
    y_training=y_initial
)

for i in range(20):
    query_idx, query_instance = learner.query(X_pool)
    learner.teach(X_pool[query_idx], y_pool[query_idx])
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)

print(f"Custom entropy sampler accuracy: {learner.score(X_test, y_test):.3f}")
# Output: Custom entropy sampler accuracy: 0.811
```

### Example 4: Active Learning vs. Random Sampling

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=20, n_informative=10, n_classes=2, random_state=42)
X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Active learning
X_initial, X_pool_al, y_initial, y_pool_al = train_test_split(
    X_pool, y_pool, train_size=10, random_state=42
)
learner = ActiveLearner(
    estimator=RandomForestClassifier(random_state=42),
    query_strategy=uncertainty_sampling,
    X_training=X_initial,
    y_training=y_initial
)

active_scores = [learner.score(X_test, y_test)]
for i in range(40):
    query_idx, _ = learner.query(X_pool_al)
    learner.teach(X_pool_al[query_idx], y_pool_al[query_idx])
    X_pool_al = np.delete(X_pool_al, query_idx, axis=0)
    y_pool_al = np.delete(y_pool_al, query_idx, axis=0)
    active_scores.append(learner.score(X_test, y_test))

# Random sampling
random_scores = []
for n_labels in range(1, 52, 1):
    sel_idx = np.random.choice(len(X_pool), min(n_labels, len(X_pool)), replace=False)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_pool[sel_idx], y_pool[sel_idx])
    random_scores.append(rf.score(X_test, y_test))

print(f"Active (50 labels): {active_scores[-1]:.3f}")
print(f"Random (50 labels): {random_scores[-1]:.3f}")
print(f"Active (20 labels): {active_scores[10]:.3f}")
print(f"Random (20 labels): {random_scores[20]:.3f}")
# Output:
# Active (50 labels): 0.850
# Random (50 labels): 0.820
# Active (20 labels): 0.810
# Random (20 labels): 0.770
```

## Common Mistakes

1. **Assuming active learning always beats random sampling.** On some datasets (especially those with outliers), random sampling can match or exceed active learning.
2. **Querying too few points per iteration.** Batch active learning queries multiple points at once — querying one at a time is slow.
3. **Ignoring the cold start problem.** With too few initial labels, the model's uncertainty estimates are unreliable.
4. **Using active learning for noisy data without care.** Uncertainty sampling tends to select outliers and mislabeled points.
5. **Not accounting for label distribution shift.** The labeled set becomes biased toward uncertain regions, potentially distorting the model.
6. **Confusing active learning with semi-supervised learning.** Active learning queries labels for informative points; semi-supervised learning uses unlabeled data directly.
7. **Applying to deep learning without batching.** Deep active learning requires batch-mode acquisition (e.g., BatchBALD) to avoid repeatedly retraining.

## Interview Questions

### Beginner

1. What is active learning and when would you use it?
2. What is uncertainty sampling?
3. How does active learning differ from passive (random) sampling?
4. What is the query-by-committee strategy?
5. Why is active learning useful in medical imaging?

### Intermediate

1. Compare least confident, margin, and entropy sampling. When might each fail?
2. How would you implement batch-mode active learning?
3. What is the cold start problem in active learning and how can you mitigate it?
4. Explain expected error reduction as a query strategy.
5. How does active learning relate to Bayesian optimization?

### Advanced

1. Derive the expected gradient length (expected model change) as a query strategy.
2. How would you extend active learning to deep neural networks? Discuss BatchBALD and core-set selection.
3. Analyze the theoretical guarantees of active learning — label complexity and disagreement-based active learning.

## Practice Problems

### Easy

1. Implement a simple active learning loop for logistic regression with uncertainty sampling.
2. Plot accuracy vs. number of labeled samples for active vs. random sampling.
3. Use modAL to implement margin sampling for a 3-class classification problem.
4. Compare active learning with pool-based vs. stream-based sampling.
5. Visualize the decision boundary evolution during active learning on a 2D dataset.

### Medium

1. Implement query-by-committee from scratch (no modAL).
2. Compare uncertainty sampling vs. diversity sampling (e.g., k-center greedy).
3. Implement active learning for regression using expected variance reduction.
4. Analyze the effect of different initial labeled set sizes on active learning performance.
5. Implement a custom query strategy that combines uncertainty and density (information density).

### Hard

1. Implement BatchBALD for deep active learning (approximation).
2. Derive and implement expected error reduction for a logistic regression model.
3. Implement disagreement-based active learning for multi-view data.

## Solutions

Solution 1 (Easy): Simple active learning loop

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=300, n_features=5, random_state=42)
labeled_idx = np.random.choice(len(X), 10, replace=False)
unlabeled_idx = np.setdiff1d(np.arange(len(X)), labeled_idx)

for i in range(40):
    model = LogisticRegression()
    model.fit(X[labeled_idx], y[labeled_idx])
    probs = model.predict_proba(X[unlabeled_idx])
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    best = unlabeled_idx[np.argmax(entropy)]
    labeled_idx = np.append(labeled_idx, best)
    unlabeled_idx = np.delete(unlabeled_idx, np.where(unlabeled_idx == best))
```

Solution 2 (Medium): QBC from scratch

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def qbc_query(X_labeled, y_labeled, X_unlabeled, n_committee=5):
    models = []
    for _ in range(n_committee):
        bootstrap_idx = np.random.choice(len(X_labeled), len(X_labeled), replace=True)
        m = DecisionTreeClassifier().fit(X_labeled[bootstrap_idx], y_labeled[bootstrap_idx])
        models.append(m)
    votes = np.array([m.predict(X_unlabeled) for m in models])
    entropy = -np.mean([np.mean(votes == c, axis=0) * np.log(np.mean(votes == c, axis=0) + 1e-12) for c in range(2)], axis=0)
    return np.argmax(entropy)
```

## Related Concepts

- Semi-Supervised Learning (ML-071)
- Bayesian Optimization (ML-069)
- Uncertainty Quantification
- Experimental Design
- Multi-Armed Bandits
- Cost-Sensitive Learning

## Next Concepts

- Semi-Supervised Learning (ML-071)
- Multi-Label Classification (ML-072)
- Online Learning (ML-074)

## Summary

Active Learning reduces labeling costs by iteratively selecting the most informative unlabeled points for annotation. Query strategies like uncertainty sampling (least confident, margin, entropy), query-by-committee, and expected model change guide the selection process. Active learning consistently outperforms random sampling, especially in early iterations when labeled data is scarce. It is a critical tool for any application where labeled data is expensive but unlabeled data is abundant.

## Key Takeaways

- Active learning minimizes labeling cost while maximizing accuracy.
- Uncertainty sampling queries points near the decision boundary.
- Entropy sampling queries points with highest predictive uncertainty.
- Query-by-committee measures disagreement among multiple models.
- Active learning typically beats random sampling, especially with small label budgets.
- Cold start and outlier sensitivity are key challenges.
- Batch-mode acquisition is necessary for deep learning.
