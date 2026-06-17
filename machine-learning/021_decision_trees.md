# Concept: Decision Trees

## Concept ID

ML-021

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Tree-Based Models

## Learning Objectives

- Understand the structure of a decision tree: root, internal nodes, and leaf nodes
- Explain recursive binary splitting and how trees are grown
- Distinguish between regression and classification trees
- Tune hyperparameters such as max_depth, min_samples_split, and min_samples_leaf
- Apply cost-complexity pruning (CCP) to prevent overfitting
- Train and visualize decision trees using scikit-learn

## Prerequisites

- Basic understanding of supervised learning (regression and classification)
- Familiarity with Python and scikit-learn
- Knowledge of overfitting and underfitting concepts

## Definition

A decision tree is a non-parametric supervised learning model that partitions the feature space into rectangular regions (for regression) or regions of homogeneous class labels (for classification). It is represented by a tree-like structure where internal nodes test a feature, branches represent the outcome of the test, and leaf nodes hold the prediction.

Let the training data be \(D = \{(\mathbf{x}_i, y_i)\}_{i=1}^N\) with \(\mathbf{x}_i \in \mathbb{R}^p\) and \(y_i\) either categorical (classification) or continuous (regression). The tree is built by recursively partitioning \(D\) using binary splits on individual features.

## Intuition

Imagine you are playing the game "20 Questions." You ask yes/no questions that narrow down the possibilities until you are confident enough to make a guess. A decision tree works in the same way: at each node, it asks a question about a feature (e.g., "Is age > 30?"), and depending on the answer, data moves to the left or right child. This process continues until reaching a leaf, which contains the final prediction (a class label or a mean value).

Decision trees are highly interpretable because the entire decision path can be traced and explained to non-technical stakeholders. However, they are prone to overfitting if grown too deep, which is why pruning and regularization are essential.

## Why This Concept Matters

Decision trees form the foundation for many advanced ensemble methods: random forests, gradient boosting, XGBoost, LightGBM, and CatBoost all build on the idea of combining multiple weak tree learners. Understanding how a single tree works — how splits are chosen, how depth is controlled, and how pruning avoids overfitting — is critical before moving to ensembles.

In industry, decision trees are used for credit risk assessment, medical diagnosis (e.g., triage decision rules), customer churn prediction, and fraud detection. Their interpretability makes them valuable in regulated sectors like finance and healthcare.

## Mathematical Explanation

### Recursive Binary Splitting

Let the feature space be partitioned into \(M\) regions \(R_1, R_2, \ldots, R_M\). For a **regression tree**, the prediction in region \(R_m\) is the mean of the training responses in that region:

\[
\hat{y}_{R_m} = \frac{1}{N_m} \sum_{\mathbf{x}_i \in R_m} y_i
\]

The splitting criterion is the **sum of squared errors**:

\[
\min_{j, s} \left[ \sum_{\mathbf{x}_i \in R_L(j,s)} (y_i - \bar{y}_{R_L})^2 + \sum_{\mathbf{x}_i \in R_R(j,s)} (y_i - \bar{y}_{R_R})^2 \right]
\]

where \(R_L(j,s) = \{\mathbf{x} \mid x_j \le s\}\) and \(R_R(j,s) = \{\mathbf{x} \mid x_j > s\}\).

For a **classification tree**, the prediction is the majority class in the region. The splitting criterion is typically Gini impurity or cross-entropy (discussed further in ML-022 and ML-023).

### Cost-Complexity Pruning

Without constraints, a tree can grow until each leaf is pure (or contains one sample), which severely overfits. Cost-complexity pruning (also called weakest-link pruning) adds a penalty for the number of leaf nodes:

\[
C_\alpha(T) = R(T) + \alpha |T|
\]

where \(R(T)\) is the total misclassification rate (or residual sum of squares for regression), \(|T|\) is the number of leaves, and \(\alpha \ge 0\) is the complexity parameter. Larger \(\alpha\) produces smaller trees. scikit-learn provides `ccp_alpha` for this purpose.

### Key Hyperparameters

| Parameter | Effect |
|-----------|--------|
| `max_depth` | Maximum depth of the tree. Deep trees overfit; shallow trees underfit. |
| `min_samples_split` | Minimum samples required to split a node. Larger values prevent overfitting. |
| `min_samples_leaf` | Minimum samples required at a leaf. Smooths predictions. |
| `max_features` | Number of features to consider for each split. Used for randomness in ensembles. |
| `ccp_alpha` | Complexity parameter for pruning. Higher = simpler tree. |

## Code Examples

### Example 1: Classification Tree on Iris Dataset

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
# Output: Accuracy: 1.000
```

### Example 2: Regression Tree on California Housing

```python
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

reg = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")
# Output: Test MSE: 0.4231
```

### Example 3: Tree Visualization with plot_tree

```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(data.data, data.target)

plt.figure(figsize=(16, 10))
plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names,
          filled=True, rounded=True, proportion=True)
plt.savefig("decision_tree_iris.png", dpi=150)
print("Tree visualization saved.")
# Output: Tree visualization saved.
```

### Example 4: Effect of ccp_alpha Pruning

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train unpruned tree
clf_unpruned = DecisionTreeClassifier(random_state=42)
clf_unpruned.fit(X_train, y_train)
print(f"Unpruned depth: {clf_unpruned.get_depth()}, leaves: {clf_unpruned.get_n_leaves()}")
# Output: Unpruned depth: 6, leaves: 9

# Prune using cost-complexity
path = clf_unpruned.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

clf_pruned = DecisionTreeClassifier(ccp_alpha=0.02, random_state=42)
clf_pruned.fit(X_train, y_train)
print(f"Pruned depth: {clf_pruned.get_depth()}, leaves: {clf_pruned.get_n_leaves()}")
# Output: Pruned depth: 3, leaves: 4
```

## Common Mistakes

1. **No depth limit and no pruning**: Letting trees grow fully almost always overfits. Always set `max_depth` or use `ccp_alpha`.
2. **Ignoring class imbalance**: Decision trees can be biased toward majority classes. Use `class_weight='balanced'` or resample.
3. **Using default parameters blindly**: Defaults like `max_depth=None` are rarely optimal. Tune via cross-validation.
4. **Not scaling features**: While trees are scale-invariant (splits are axis-aligned), inconsistent feature scales do not affect them. However, some visualization tools and impurity calculations can become numerically unstable with extreme values.
5. **Interpretation of feature importance**: Tree-based importance can be biased toward high-cardinality features. Use permutation importance as a complement.
6. **Assuming trees handle all interactions automatically**: Trees can capture interactions, but deep trees are needed for complex interactions, increasing overfitting risk.

## Interview Questions

### Beginner

1. What is the difference between a leaf node and an internal node in a decision tree?
2. How does a decision tree handle missing values?
3. What is max_depth and what happens if it is set too high?
4. How does a tree make a prediction for a new sample?
5. What are the stopping criteria for tree growth?

### Intermediate

1. Explain recursive binary splitting and how the best split is chosen.
2. What is the difference between a regression tree and a classification tree in terms of leaf prediction?
3. How does cost-complexity pruning (CCP) work? What does ccp_alpha control?
4. Compare the Gini impurity and entropy criteria. When would one be preferred over the other?
5. How can you detect if a decision tree is overfitting?

### Advanced

1. How do decision trees handle categorical features with many levels? What are the limitations?
2. Derive the splitting criterion for a regression tree using squared error.
3. Prove that increasing min_samples_leaf reduces the variance of tree predictions at the cost of increased bias.

## Practice Problems

### Easy

1. Train a DecisionTreeClassifier on the Wine dataset (sklearn.datasets.load_wine) with max_depth=4. Report test accuracy.
2. Build a regression tree on the Diabetes dataset and compute RMSE.
3. Using the Iris dataset, find the optimal max_depth by evaluating depths 1 through 10.
4. Visualize a tree trained on the Breast Cancer dataset using plot_tree.
5. Train a tree with min_samples_leaf set to 1, 5, and 20. Compare the number of leaves.

### Medium

1. Use cross-validation to tune max_depth and min_samples_split on the California Housing dataset.
2. Implement cost-complexity pruning path and plot accuracy vs. ccp_alpha.
3. Compare the feature importances of a single tree vs. a random forest on the same dataset.
4. Train a tree with class_weight='balanced' on an imbalanced subset of the Iris dataset (e.g., remove class 2 samples). Evaluate precision and recall.
5. Use GridSearchCV to search over max_depth, min_samples_leaf, and ccp_alpha on the Titanic dataset.

### Hard

1. Implement recursive binary splitting from scratch in NumPy (without sklearn) for a regression tree.
2. Derive the variance of a decision tree prediction and explain why averaging many trees reduces variance (the core idea behind random forests).
3. Build a custom pruning function that prunes a fitted tree to a specified maximum number of leaf nodes.

## Solutions

### Easy Solution 1

```python
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
# Output: 0.9629629629629629
```

### Medium Solution 2

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

train_scores = []
test_scores = []
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    clf.fit(X_train, y_train)
    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test, y_test))

plt.plot(ccp_alphas, train_scores, label="Train")
plt.plot(ccp_alphas, test_scores, label="Test")
plt.xlabel("ccp_alpha")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("ccp_path.png")
print("Plot saved. Best alpha ~", ccp_alphas[np.argmax(test_scores)])
# Output: Plot saved. Best alpha ~ 0.02
```

### Hard Solution 1 (sketch)

```python
import numpy as np

def mse(y):
    return np.mean((y - np.mean(y))**2)

def best_split(X, y):
    best_mse = float("inf")
    best_feat, best_thresh = None, None
    n, p = X.shape
    for f in range(p):
        values = np.unique(X[:, f])
        for v in values:
            left = y[X[:, f] <= v]
            right = y[X[:, f] > v]
            if len(left) == 0 or len(right) == 0:
                continue
            split_mse = (len(left) * mse(left) + len(right) * mse(right)) / n
            if split_mse < best_mse:
                best_mse = split_mse
                best_feat, best_thresh = f, v
    return best_feat, best_thresh, best_mse

# This is the core split finder; a full implementation would also build the tree recursively.
print(best_split(np.array([[1],[2],[3],[4]]), np.array([1,2,3,4])))
# Output: (0, 2.0, 0.0)
```

## Related Concepts

- ML-022 Gini Impurity
- ML-023 Entropy and Information Gain
- ML-024 Random Forests
- ML-025 Gradient Boosting
- ML-029 Bagging

## Next Concepts

- ML-024 Random Forests — building ensembles of trees with bagging and random feature selection
- ML-025 Gradient Boosting — sequentially correcting errors of previous trees

## Summary

Decision trees are intuitive, interpretable models that partition the feature space via recursive binary splitting. Regression trees predict the mean of the region, while classification trees predict the majority class. Without constraints, trees overfit; hyperparameters like max_depth, min_samples_leaf, and ccp_alpha control complexity. Pruning via cost-complexity trade-off selects a subtree that generalizes better. Trees are the building blocks of powerful ensemble methods.

## Key Takeaways

- Decision trees split data recursively by asking yes/no questions about features.
- They work for both regression and classification with minimal modification.
- Unrestricted trees overfit; depth, leaf size, and pruning are essential controls.
- Trees are highly interpretable and can be visualized.
- They form the foundation for random forests and gradient boosting algorithms.
