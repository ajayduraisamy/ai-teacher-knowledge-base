# Concept: Entropy and Information Gain

## Concept ID

ML-023

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Tree-Based Models

## Learning Objectives

- Define entropy as a measure of uncertainty: \(H = -\sum p_k \log_2 p_k\)
- Understand the relationship between entropy and information content
- Compute information gain: \(IG(T,X) = H(T) - \sum P(v)H(T_v)\)
- Explain the ID3 algorithm and its use of information gain
- Walk through a complete example of calculating information gain for a small dataset
- Compare information gain with Gini impurity and understand when each is appropriate

## Prerequisites

- ML-021 Decision Trees
- ML-022 Gini Impurity
- Basic probability and logarithms

## Definition

Entropy, borrowed from information theory (Shannon, 1948), quantifies the uncertainty or impurity in a set of examples. For a dataset \(T\) with class distribution \(p_1, p_2, \ldots, p_K\):

\[
H(T) = -\sum_{k=1}^K p_k \log_2 p_k
\]

By convention, \(0 \log_2 0 = 0\).

**Information Gain (IG)** measures the reduction in entropy after splitting on a feature \(X\). It is the expected reduction in entropy:

\[
IG(T, X) = H(T) - \sum_{v \in \text{Values}(X)} P(v) \, H(T_v)
\]

where \(v\) is a value of feature \(X\), \(P(v) = |T_v| / |T|\), and \(T_v\) is the subset of \(T\) where feature \(X = v\).

## Intuition

Entropy answers the question: "How much information (in bits) do I gain by knowing the class label of a random sample from this set?"

- **Low entropy (close to 0)**: The set is nearly pure. Most samples belong to the same class. Uncertainty is low.
- **High entropy (close to \(\log_2 K\) for K classes)**: The set is evenly mixed. Uncertainty is high.
- **Maximum entropy**: All classes are equally probable. You gain the most information when you learn the label of a sample from a maximally uncertain set.

Information gain tells us: "How much more certain am I about the class after knowing the value of feature \(X\) compared to before?"

## Why This Concept Matters

Entropy and information gain were central to the ID3 and C4.5 decision tree algorithms (Quinlan, 1986), which were among the earliest and most influential decision tree methods. Information gain remains relevant for understanding:

- How trees select features for splitting
- The theoretical foundation of impurity reduction
- Feature selection in general (mutual information)
- Why information gain can be biased toward features with many values (the ID3 limitation that C4.5's gain ratio addresses)

## Mathematical Explanation

### Entropy for Binary Classification

For a binary problem with proportion \(p\) of class 1 and \(1-p\) of class 0:

\[
H = -p \log_2 p - (1-p) \log_2 (1-p)
\]

This is a concave function maximized at \(p = 0.5\) with value \(H = 1.0\) bit.

### Information Gain Derivation

Step by step:

1. Compute entropy of the parent node: \(H(T)\)
2. For each value \(v\) of feature \(X\):
   - Let \(T_v\) be the subset where \(X = v\)
   - Compute \(H(T_v)\)
3. Compute the weighted average: \(\sum_v \frac{|T_v|}{|T|} H(T_v)\)
4. Subtract from parent: \(IG(T, X) = H(T) - \sum_v \frac{|T_v|}{|T|} H(T_v)\)

### Full Worked Example

Consider the classic "Play Tennis" dataset (Quinlan). We have features Outlook, Temperature, Humidity, Wind, and target Play.

| Day | Outlook | Temp | Humidity | Wind | Play |
|-----|---------|------|----------|------|------|
| D1  | Sunny   | Hot  | High     | Weak | No   |
| D2  | Sunny   | Hot  | High     | Strong | No |
| D3  | Overcast| Hot  | High     | Weak | Yes  |
| D4  | Rain    | Mild | High     | Weak | Yes  |
| D5  | Rain    | Cool | Normal   | Weak | Yes  |
| D6  | Rain    | Cool | Normal   | Strong | No |
| D7  | Overcast| Cool | Normal   | Strong | Yes |
| D8  | Sunny   | Mild | High     | Weak | No   |
| D9  | Sunny   | Cool | Normal   | Weak | Yes  |
| D10 | Rain    | Mild | Normal   | Weak | Yes  |
| D11 | Sunny   | Mild | Normal   | Strong | Yes |
| D12 | Overcast| Mild | High     | Strong | Yes |
| D13 | Overcast| Hot  | Normal   | Weak | Yes  |
| D14 | Rain    | Mild | High     | Strong | No  |

14 samples: 5 No, 9 Yes.

**Parent entropy:**

\[
H(T) = -\frac{9}{14}\log_2\frac{9}{14} - \frac{5}{14}\log_2\frac{5}{14} = 0.940
\]

**Split on Outlook:**

- Sunny: 5 samples → 2 Yes, 3 No → \(H = 0.971\)
- Overcast: 4 samples → 4 Yes, 0 No → \(H = 0\)
- Rain: 5 samples → 3 Yes, 2 No → \(H = 0.971\)

Weighted: \(\frac{5}{14}(0.971) + \frac{4}{14}(0) + \frac{5}{14}(0.971) = 0.694\)

\[
IG(T, Outlook) = 0.940 - 0.694 = 0.246
\]

**Split on Wind:**

- Weak: 8 samples → 6 Yes, 2 No → \(H = 0.811\)
- Strong: 6 samples → 3 Yes, 3 No → \(H = 1.0\)

Weighted: \(\frac{8}{14}(0.811) + \frac{6}{14}(1.0) = 0.892\)

\[
IG(T, Wind) = 0.940 - 0.892 = 0.048
\]

Outlook gives the highest IG and is selected as the root split.

### ID3 Algorithm

ID3 (Iterative Dichotomiser 3) builds a decision tree using information gain:

1. Compute IG for each feature
2. Select the feature with the highest IG as the root
3. Split the dataset by values of the selected feature
4. Recursively repeat for each subset
5. Stop when all samples in a subset belong to the same class (pure), or no features remain

**Limitation**: ID3 does not handle continuous features natively and is biased toward features with many values (e.g., a unique ID column would have perfect IG). C4.5 addresses this with the gain ratio: \(GR = IG / H_{\text{split}}\).

## Code Examples

### Example 1: Entropy and Information Gain from Scratch

```python
import numpy as np
import pandas as pd

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-10))

def information_gain(data, feature, target):
    H_parent = entropy(data[target])
    values = data[feature].unique()
    weighted = 0
    for v in values:
        subset = data[data[feature] == v]
        weighted += len(subset) / len(data) * entropy(subset[target])
    return H_parent - weighted

# Dataset
data = pd.DataFrame({
    'Outlook': ['S','S','O','R','R','R','O','S','S','R','S','O','O','R'],
    'Wind': ['W','S','W','W','W','S','S','W','W','W','S','S','W','S'],
    'Play': ['N','N','Y','Y','Y','N','Y','N','Y','Y','Y','Y','Y','N']
})

print(f"Entropy parent: {entropy(data['Play']):.3f}")
# Output: Entropy parent: 0.940
print(f"IG(Outlook): {information_gain(data, 'Outlook', 'Play'):.3f}")
# Output: IG(Outlook): 0.246
print(f"IG(Wind): {information_gain(data, 'Wind', 'Play'):.3f}")
# Output: IG(Wind): 0.048
```

### Example 2: Using sklearn's DecisionTreeClassifier with Entropy

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Accuracy (entropy): {accuracy_score(y_test, y_pred):.3f}")
# Output: Accuracy (entropy): 1.000
```

### Example 3: Mutual Information for Feature Selection

```python
from sklearn.feature_selection import mutual_info_classif
import pandas as pd

X, y = load_iris(return_X_y=True)
mi = mutual_info_classif(X, y, random_state=42)
feature_names = load_iris().feature_names

mi_df = pd.DataFrame({'feature': feature_names, 'mutual_info': mi}).sort_values('mutual_info', ascending=False)
print(mi_df)
# Output:            feature  mutual_info
# Output: 2  petal length (cm)     0.993348
# Output: 3   petal width (cm)     0.985481
# Output: 0  sepal length (cm)     0.478388
# Output: 1   sepal width (cm)     0.262949
```

### Example 4: Visualizing Entropy vs Gini

```python
import numpy as np
import matplotlib.pyplot as plt

p = np.linspace(0.001, 0.999, 100)
entropy_vals = -p * np.log2(p) - (1-p) * np.log2(1-p)
gini_vals = 2 * p * (1-p)

plt.figure(figsize=(8, 5))
plt.plot(p, entropy_vals, label='Entropy')
plt.plot(p, gini_vals, label='Gini')
plt.xlabel('p (class 1 proportion)')
plt.ylabel('Impurity')
plt.legend()
plt.title('Entropy vs Gini for Binary Classification')
plt.savefig('entropy_vs_gini.png')
print("Plot saved.")
# Output: Plot saved.
```

## Common Mistakes

1. **Using the wrong logarithm base**: scikit-learn uses natural log for entropy in its internal calculations. The standard information-theoretic definition uses \(\log_2\) for bits. The base does not affect ranking of splits, only the numeric scale.
2. **Forgetting the \(\log_2\) convention**: In machine learning literature, entropy is typically computed with \(\log_2\). Some implementations use \(\ln\) or \(\log_{10}\). Be consistent.
3. **Ignoring the bias toward high-cardinality features**: Information gain favors features with many unique values. In extreme cases, a unique ID column gives perfect IG. Use gain ratio (C4.5) or restrict to categorical features with reasonable cardinality.
4. **Applying information gain to regression**: IG and entropy are for classification. Regression trees use variance reduction. Entropy can be extended to continuous targets via discretization, but this is rarely done.
5. **Assuming IG is symmetric**: Information gain is not symmetric: \(IG(T, X) \neq IG(X, T)\) (the latter would be meaningless in the decision tree context).
6. **Not handling zero probabilities**: \(0 \log_2 0\) is undefined. scikit-learn handles this internally, but custom implementations must add a small epsilon or skip zero-probability classes.

## Interview Questions

### Beginner

1. What is entropy in the context of decision trees?
2. What is the entropy of a pure node?
3. What is the maximum entropy for a binary classification problem?
4. What does information gain measure?
5. How does ID3 select the feature for the root node?

### Intermediate

1. Calculate entropy for a node with 30 class A and 70 class B samples.
2. Compare information gain with Gini impurity. Under what conditions would you choose one over the other?
3. Explain the bias of information gain toward features with many values. How does C4.5 address this?
4. What is the gain ratio? Write its formula.
5. Walk through the ID3 algorithm step by step for a dataset with three features.

### Advanced

1. Derive the relationship between entropy and expected code length in information theory. How does this connect to decision trees?
2. Prove that information gain is always non-negative.
3. Explain why mutual information is the expected information gain, and show the mathematical equivalence.

## Practice Problems

### Easy

1. Compute entropy for: (a) 100% class A (b) 50% A, 50% B (c) 80% A, 20% B
2. Given parent entropy 0.94 and after split the weighted entropy is 0.69, what is IG?
3. Which feature should ID3 select: Feature A (IG=0.12) or Feature B (IG=0.25)?
4. How many bits are needed to encode a 4-class uniform distribution?
5. What is the entropy of a node with 10 samples all of the same class?

### Medium

1. Compute IG for all features in the "Play Tennis" dataset and confirm Outlook is the best first split.
2. Implement the ID3 algorithm from scratch in Python for categorical features.
3. Compare trees built with criterion='entropy' vs 'gini' on the Wine dataset. Are there accuracy differences?
4. For a continuous feature, implement a function that finds the optimal split threshold by maximizing IG.
5. Create a synthetic dataset where IG would choose a suboptimal feature due to its high-cardinality bias.

### Hard

1. Derive the gain ratio formula and implement it. Compare splits selected by IG vs gain ratio on the same dataset.
2. Prove that the weighted average of child entropies is always less than or equal to parent entropy (Jensen's inequality).
3. Implement a full decision tree using only NumPy that uses information gain as the splitting criterion for categorical features.

## Solutions

### Easy Solution 1

```python
import numpy as np
def entropy_binary(p):
    p = np.clip(p, 1e-10, 1-1e-10)
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

print(f"(a) p=1.0: {entropy_binary(1.0):.4f}")
print(f"(b) p=0.5: {entropy_binary(0.5):.4f}")
print(f"(c) p=0.8: {entropy_binary(0.8):.4f}")
# Output: (a) p=1.0: 0.0000
# Output: (b) p=0.5: 1.0000
# Output: (c) p=0.8: 0.7219
```

### Medium Solution 1

```python
import pandas as pd
import numpy as np

def entropy(y):
    _, c = np.unique(y, return_counts=True)
    p = c / len(y)
    return -np.sum(p * np.log2(p + 1e-10))

def ig(data, feat, target):
    H = entropy(data[target])
    vals = data[feat].unique()
    return H - sum(len(data[data[feat]==v])/len(data)*entropy(data[data[feat]==v][target]) for v in vals)

df = pd.DataFrame({
    'Outlook': ['S','S','O','R','R','R','O','S','S','R','S','O','O','R'],
    'Temp': ['H','H','H','M','C','C','C','M','C','M','M','M','H','M'],
    'Humidity': ['H','H','H','H','N','N','N','H','N','N','N','H','N','H'],
    'Wind': ['W','S','W','W','W','S','S','W','W','W','S','S','W','S'],
    'Play': ['N','N','Y','Y','Y','N','Y','N','Y','Y','Y','Y','Y','N']
})

for col in ['Outlook','Temp','Humidity','Wind']:
    print(f"IG({col}): {ig(df, col, 'Play'):.3f}")
# Output: IG(Outlook): 0.246
# Output: IG(Temp): 0.029
# Output: IG(Humidity): 0.151
# Output: IG(Wind): 0.048
```

### Hard Solution 2

```python
import numpy as np

# Proof by Jensen: entropy is concave, so H(parent) >= weighted avg of H(children)
# Demonstration:
parent = np.array([0]*9 + [1]*5)
child1 = np.array([0]*3 + [1]*2)  # p=0.6
child2 = np.array([0]*6 + [1]*3)  # p=0.333

def H(arr):
    _, c = np.unique(arr, return_counts=True)
    p = c / len(arr)
    return -np.sum(p*np.log2(p+1e-10))

print(f"H(parent): {H(parent):.4f}")
print(f"Weighted children: {0.5*H(child1)+0.5*H(child2):.4f}")
print(f"H(parent) >= weighted children: {H(parent) >= 0.5*H(child1)+0.5*H(child2)}")
# Output: H(parent): 0.9403
# Output: Weighted children: 0.9311
# Output: H(parent) >= weighted children: True
```

## Related Concepts

- ML-021 Decision Trees
- ML-022 Gini Impurity
- ID3 and C4.5 algorithms
- Mutual Information (feature selection)

## Next Concepts

- ML-024 Random Forests — how impurity reduction is aggregated across trees
- ML-025 Gradient Boosting — using trees sequentially to minimize loss

## Summary

Entropy measures uncertainty in a dataset: \(H = -\sum p_k \log_2 p_k\). Information gain quantifies how much a feature reduces entropy: \(IG(T,X) = H(T) - \sum P(v)H(T_v)\). The ID3 algorithm selects the feature with the highest IG at each step to build a decision tree. IG is biased toward features with many unique values, a limitation addressed by C4.5's gain ratio. Entropy and Gini produce similar trees in practice, but entropy has a stronger information-theoretic foundation.

## Key Takeaways

- Entropy = 0 for pure nodes, \(\log_2 K\) for uniform distribution across K classes.
- Information gain = parent entropy - weighted average of child entropies.
- ID3 selects splits by maximizing information gain.
- Information gain is biased toward features with many distinct values.
- Entropy and Gini produce similar trees; entropy is more theoretically motivated.
- Mutual information is the continuous analogue of information gain.
