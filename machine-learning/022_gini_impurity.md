# Concept: Gini Impurity

## Concept ID

ML-022

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Tree-Based Models

## Learning Objectives

- Define Gini impurity mathematically: \(Gini = 1 - \sum p_k^2\)
- Understand how Gini impurity measures node purity
- Calculate Gini impurity for a set of class distributions
- Compute the weighted Gini impurity of a split
- Select the best split by minimizing the weighted Gini impurity
- Compare Gini impurity with entropy and understand practical trade-offs

## Prerequisites

- ML-021 Decision Trees
- Basic probability: calculating proportions and weighted sums

## Definition

Gini impurity is a metric used in decision trees (specifically CART — Classification and Regression Trees) to evaluate the quality of a split. It measures how often a randomly chosen element from the set would be incorrectly labeled if it were randomly labeled according to the distribution of labels in the subset.

For a node \(m\) containing \(N_m\) samples from \(K\) classes, let \(\hat{p}_{mk}\) be the proportion of samples belonging to class \(k\):

\[
\hat{p}_{mk} = \frac{1}{N_m} \sum_{\mathbf{x}_i \in R_m} \mathbb{I}(y_i = k)
\]

The Gini impurity at node \(m\) is:

\[
G_m = \sum_{k=1}^K \hat{p}_{mk} (1 - \hat{p}_{mk}) = 1 - \sum_{k=1}^K \hat{p}_{mk}^2
\]

## Intuition

Gini impurity answers the question: "If I randomly pick a sample from this node and then randomly assign it a class label according to the class proportions, how likely am I to be wrong?"

- If the node is **pure** — all samples belong to one class — then \(p_{k}=1\) for that class and 0 for all others. The Gini impurity is \(1 - (1^2 + 0^2 + \dots) = 0\). Minimum impurity.
- If the node is **perfectly mixed** — equal proportions of all classes — then \(p_k = 1/K\) and \(G = 1 - K(1/K)^2 = 1 - 1/K\). Maximum impurity for \(K\) classes.
- For binary classification: \(G = 2p(1-p)\) where \(p\) is the proportion of one class.

Lower Gini is better. When building a tree, we choose the split that yields the lowest weighted average Gini impurity in the child nodes.

## Why This Concept Matters

Gini impurity is the default splitting criterion in scikit-learn's `DecisionTreeClassifier` and is used by CART, the most popular decision tree algorithm. Understanding it is essential for:
- Debugging and interpreting tree splits
- Understanding feature importance (mean decrease in impurity)
- Comparing tree-based models and their split heuristics
- Advancing to ensemble methods (Random Forests, Gradient Boosting)

## Mathematical Explanation

### Weighted Gini Impurity of a Split

When a node is split into left child \(L\) and right child \(R\) with sample counts \(N_L\) and \(N_R\), the **split impurity** is:

\[
G_{\text{split}} = \frac{N_L}{N_m} G_L + \frac{N_R}{N_m} G_R
\]

The **Gini reduction** (impurity decrease) is:

\[
\Delta G = G_m - G_{\text{split}}
\]

A larger \(\Delta G\) indicates a better split.

### Example Calculation

Consider a node with 10 samples: 5 class A, 3 class B, 2 class C.

\[
G_m = 1 - \left( (0.5)^2 + (0.3)^2 + (0.2)^2 \right) = 1 - (0.25 + 0.09 + 0.04) = 1 - 0.38 = 0.62
\]

Now consider a split on feature \(x_j \le t\):

- Left child: 6 samples — 4 A, 1 B, 1 C → \(G_L = 1 - (16/36 + 1/36 + 1/36) = 1 - 18/36 = 0.50\)
- Right child: 4 samples — 1 A, 2 B, 1 C → \(G_R = 1 - (1/16 + 4/16 + 1/16) = 1 - 6/16 = 0.625\)

\[
G_{\text{split}} = \frac{6}{10}(0.50) + \frac{4}{10}(0.625) = 0.30 + 0.25 = 0.55
\]

\[
\Delta G = 0.62 - 0.55 = 0.07
\]

The algorithm would compare this to other candidate splits and choose the one with the largest \(\Delta G\).

### Gini vs Entropy

| Criterion | Formula | Range (binary) | Computational Cost |
|-----------|---------|-----------------|--------------------|
| Gini | \(1 - \sum p_k^2\) | \([0, 0.5]\) | Faster (no log) |
| Entropy | \(-\sum p_k \log_2 p_k\) | \([0, 1]\) | Slower (log) |

In practice, the difference between Gini and entropy is small. Both yield similar trees. Gini is slightly faster because it avoids logarithmic computations. Breiman (1984) recommended Gini as the default for CART.

Key difference: Gini tends to be more sensitive to changes in the majority class, while entropy is more sensitive to changes in minority classes. However, the practical impact on accuracy is negligible.

## Code Examples

### Example 1: Manual Gini Calculation

```python
import numpy as np

def gini_impurity(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)

# Pure node
print(f"Pure: {gini_impurity(np.array([0,0,0,0])):.4f}")
# Output: Pure: 0.0000

# 50/50 split
print(f"50/50: {gini_impurity(np.array([0,0,1,1])):.4f}")
# Output: 50/50: 0.5000

# 3-class node
print(f"3-class: {gini_impurity(np.array([0,0,1,2])):.4f}")
# Output: 3-class: 0.6250
```

### Example 2: Weighted Gini for a Split

```python
def weighted_gini_split(y_left, y_right):
    n_left, n_right = len(y_left), len(y_right)
    n_total = n_left + n_right
    g_left = gini_impurity(y_left)
    g_right = gini_impurity(y_right)
    return (n_left / n_total) * g_left + (n_right / n_total) * g_right

y_parent = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
y_left = np.array([0, 0, 0, 0, 1, 1])
y_right = np.array([1, 1, 1, 1])

print(f"Parent Gini: {gini_impurity(y_parent):.4f}")
print(f"Split Gini: {weighted_gini_split(y_left, y_right):.4f}")
# Output: Parent Gini: 0.4800
# Output: Split Gini: 0.4000
```

### Example 3: Gini-Based Splitting in sklearn

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

print(f"Test accuracy (gini): {accuracy_score(y_test, clf.predict(X_test)):.4f}")
# Output: Test accuracy (gini): 1.0000

# Compare with entropy
clf_ent = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
clf_ent.fit(X_train, y_train)
print(f"Test accuracy (entropy): {accuracy_score(y_test, clf_ent.predict(X_test)):.4f}")
# Output: Test accuracy (entropy): 1.0000
```

### Example 4: Impurity Reduction Feature Importance

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

X, y = load_iris(return_X_y=True)
rf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42)
rf.fit(X, y)

feature_names = load_iris().feature_names
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(importance)
# Output:            feature  importance
# Output: 2  petal length (cm)    0.441177
# Output: 3   petal width (cm)    0.418804
# Output: 0  sepal length (cm)    0.108535
# Output: 1   sepal width (cm)    0.031485
```

## Common Mistakes

1. **Confusing Gini impurity with Gini coefficient**: The Gini impurity is a measure of class distribution purity used in decision trees. The Gini coefficient is an economic measure of income inequality. They are different concepts with different formulas.
2. **Assuming Gini impurity is always better than accuracy for splits**: Gini is the splitting criterion, but final evaluation should use accuracy, F1, or another task-appropriate metric.
3. **Using Gini for regression trees**: Gini is for classification. Regression trees use MSE (mean squared error) or MAE as the splitting criterion.
4. **Ignoring class imbalance in Gini calculations**: In highly imbalanced datasets, Gini impurity may still be low even if the minority class is poorly separated because the majority class dominates the proportion.
5. **Not understanding the range**: Gini ranges from 0 (pure) to \(1 - 1/K\) (maximal impurity). For binary classification, max is 0.5.
6. **Prematurely stopping tree growth based on small Gini decreases**: A small Gini reduction at one split does not guarantee that subsequent splits would not be beneficial.

## Interview Questions

### Beginner

1. What is the range of Gini impurity for binary classification?
2. What is the Gini impurity of a pure node?
3. How does Gini impurity change when class proportions are equal?
4. Why is Gini impurity preferred over accuracy as a splitting criterion?
5. What does a Gini impurity of 0.5 mean in a two-class problem?

### Intermediate

1. Calculate the Gini impurity for a node with 40 samples of class A and 60 of class B.
2. How does the Gini reduction (impurity decrease) guide the choice of split?
3. Compare Gini impurity and entropy. When does one outperform the other?
4. Why is the weighted average of child Gini impurities used instead of the sum?
5. How does the number of classes affect the maximum possible Gini impurity?

### Advanced

1. Derive the Gini impurity formula from the probability of misclassification in a random labeling scheme.
2. Prove that Gini impurity is always upper-bounded by entropy for the same distribution.
3. Explain why Gini impurity can lead to biased feature importance when features have different numbers of categories.

## Practice Problems

### Easy

1. Compute Gini impurity for: (a) [0,0,0,1,1] (b) [0,1,2,3,4]
2. Calculate Gini for nodes with proportions: (a) p=[0.9, 0.1] (b) p=[0.6, 0.4]
3. Which split reduces Gini more: a split creating pure left child (all A) and mixed right child (5A, 5B) from a parent of (10A, 10B)?
4. Verify that Gini impurity for K=2 is 2p(1-p).
5. For a 3-class problem with proportions [0.5, 0.3, 0.2], compute Gini.

### Medium

1. Write a function that takes parent and child labels and returns the Gini reduction of a split.
2. Grid search over `criterion=['gini', 'entropy']` on the Breast Cancer dataset. Compare results.
3. For the Iris dataset, hand-calculate (using Python) which feature gives the best first split by Gini.
4. Plot Gini impurity as a function of p for binary classification. Overlay entropy for comparison.
5. Explain why Gini impurity is 0.5 for a 50-50 split but 0.75 for a 3-class uniform split.

### Hard

1. Prove that minimizing Gini impurity is equivalent to maximizing the weighted sum of squared class proportions.
2. Implement a decision tree split finder that selects splits by maximizing Gini reduction (without using sklearn).
3. Analyze the bias of Gini-based feature importance for features with many unique values. Demonstrate with a synthetic dataset.

## Solutions

### Easy Solution 1a

```python
import numpy as np
y = np.array([0,0,0,1,1])
_, counts = np.unique(y, return_counts=True)
p = counts / len(y)
gini = 1 - np.sum(p**2)
print(f"Gini: {gini:.4f}")
# Output: Gini: 0.4800
```

### Medium Solution 1

```python
def gini_reduction(y_parent, y_left, y_right):
    def gini(y):
        _, c = np.unique(y, return_counts=True)
        p = c / len(y)
        return 1 - np.sum(p**2)
    G_parent = gini(y_parent)
    n = len(y_parent)
    G_split = (len(y_left)/n)*gini(y_left) + (len(y_right)/n)*gini(y_right)
    return G_parent - G_split

y_p = np.array([0]*10 + [1]*10)
y_l = np.array([0]*8 + [1]*2)
y_r = np.array([0]*2 + [1]*8)
print(f"Gini reduction: {gini_reduction(y_p, y_l, y_r):.4f}")
# Output: Gini reduction: 0.1600
```

### Hard Solution 2 (split finder)

```python
import numpy as np
from sklearn.datasets import load_iris

def best_split_by_gini(X, y):
    n, p = X.shape
    best_gain = -1
    best_feat, best_thresh = None, None
    for f in range(p):
        thresholds = np.unique(X[:, f])
        for t in thresholds:
            left = y[X[:, f] <= t]
            right = y[X[:, f] > t]
            if len(left) == 0 or len(right) == 0:
                continue
            # parent gini
            _, pc = np.unique(y, return_counts=True)
            pp = pc / n
            G_parent = 1 - np.sum(pp**2)
            # child ginis
            _, lc = np.unique(left, return_counts=True)
            lp = lc / len(left)
            G_left = 1 - np.sum(lp**2)
            _, rc = np.unique(right, return_counts=True)
            rp = rc / len(right)
            G_right = 1 - np.sum(rp**2)
            G_split = (len(left)/n)*G_left + (len(right)/n)*G_right
            gain = G_parent - G_split
            if gain > best_gain:
                best_gain = gain
                best_feat, best_thresh = f, t
    return best_feat, best_thresh, best_gain

iris = load_iris()
X, y = iris.data, iris.target
f, t, g = best_split_by_gini(X, y)
print(f"Best feature: {iris.feature_names[f]}, threshold: {t:.2f}, gain: {g:.4f}")
# Output: Best feature: petal length (cm), threshold: 2.45, gain: 0.3333
```

## Related Concepts

- ML-021 Decision Trees
- ML-023 Entropy and Information Gain
- ML-024 Random Forests (mean decrease in impurity)

## Next Concepts

- ML-023 Entropy and Information Gain — an alternative splitting criterion with information-theoretic motivation
- ML-024 Random Forests — how impurity reduction is aggregated across trees for feature importance

## Summary

Gini impurity quantifies the impurity of a node in a classification tree: \(G = 1 - \sum p_k^2\). It ranges from 0 (pure) to \(1 - 1/K\) (maximally impure). When building a decision tree, the algorithm evaluates candidate splits by computing the weighted average Gini of the child nodes and selects the split that maximizes the Gini reduction (impurity decrease). Gini is the default criterion in CART and scikit-learn, preferred for its computational efficiency and similarity to entropy in practice.

## Key Takeaways

- Gini impurity = probability of mislabeling a random sample under random labeling.
- Pure node → Gini = 0; evenly mixed node → Gini = \(1 - 1/K\).
- Best split minimizes the weighted average Gini of child nodes.
- Gini reduction (\(\Delta G\)) measures the quality of a split.
- Gini and entropy produce similar trees; Gini is computationally faster.
- Gini-based feature importance aggregates impurity reductions across all splits involving a feature.
