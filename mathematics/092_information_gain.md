# Concept: Information Gain

## Concept ID

MATH-092

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Information Theory

## Learning Objectives

- Define information gain as the reduction in entropy after splitting on an attribute
- Compute information gain for categorical features with small datasets by hand
- Understand how information gain is used as a splitting criterion in decision trees (ID3, C4.5, CART)
- Recognise the bias of information gain toward features with many values
- Apply information gain for feature importance ranking in ensemble methods

## Prerequisites

- Entropy (MATH-088)
- Mutual Information (MATH-091)
- Probability (MATH-065)
- Conditional Probability (MATH-068)
- Basic dataset manipulation and counting

## Definition

**Information gain** (IG) measures the reduction in entropy of a target variable $T$ after splitting a dataset on an attribute $a$. It quantifies how much information the attribute provides about the target.

For a dataset $S$ with target variable $T$ and an attribute $a$ that partitions $S$ into subsets $\{S_1, S_2, \dots, S_v\}$ (one for each distinct value of $a$), the information gain is:

$$
IG(S, a) = H(T) - \sum_{v \in \text{Values}(a)} \frac{|S_v|}{|S|} H(T_v)
$$

where:
- $H(T)$ is the entropy of the target variable in the full dataset
- $S_v$ is the subset of $S$ where attribute $a$ takes value $v$
- $|S_v|/|S|$ is the weight of each subset (proportion of examples)
- $H(T_v)$ is the entropy of the target within subset $S_v$

Information gain is identical to mutual information between the target $T$ and the attribute $a$:

$$
IG(S, a) = I(T; A)
$$

where $A$ is the random variable representing attribute $a$. This connection shows that information gain measures the reduction in uncertainty about $T$ after observing $a$.

## Intuition

Information gain answers: "How much does knowing the value of feature $a$ reduce my uncertainty about the target $T$?" If a feature perfectly separates the classes, the information gain equals $H(T)$ (all uncertainty is removed). If a feature provides no information about the target, the gain is zero.

Think of a decision tree as a game of 20 Questions. At each node, you choose the question (feature) that, on average, leaves you most certain about the answer (target class). Information gain quantifies how good each question is.

If you have a dataset where the target is "Play Tennis" (Yes/No) and the features are "Outlook," "Temperature," etc., information gain tells you which feature to split on first. The feature with the highest IG becomes the root of the tree.

The term $\sum_v (|S_v|/|S|) H(T_v)$ is the weighted average of child entropies, representing the remaining uncertainty after splitting. Information gain is the amount of uncertainty removed.

## Why This Concept Matters

Information gain is the core splitting criterion in several foundational machine learning algorithms:

- **Decision trees (ID3, C4.5):** These algorithms use information gain (or gain ratio, a normalised version for C4.5) to select the best feature at each node. ID3, developed by Ross Quinlan, was one of the earliest decision tree induction algorithms.
- **Feature importance ranking:** Information gain provides a natural ranking of feature importance. Features with high IG are more discriminative and more valuable for prediction.
- **Ensemble methods:** Random forests and gradient boosting machines (XGBoost, LightGBM) use information gain-based criteria for split finding. XGBoost uses an approximation of information gain for fast split search.
- **Explainable AI:** Decision trees built with information gain are interpretable: the top splits reveal the most important factors for prediction, providing natural explanations.
- **Feature selection:** IG-based feature selection is fast, interpretable, and works well for categorical data.

## Historical Background

Information gain was introduced as the splitting criterion in the **ID3** (Iterative Dichotomiser 3) algorithm by **Ross Quinlan** in 1979. Quinlan's ID3 was one of the first practical decision tree induction algorithms and became highly influential in machine learning.

ID3 used information gain directly, but Quinlan observed a bias: information gain favours attributes with many values (e.g., a unique ID feature would have maximum IG by creating one leaf per example). To address this, Quinlan developed **C4.5** (1993), which uses the **gain ratio**:

$$
\text{GainRatio}(S, a) = \frac{IG(S, a)}{\text{SplitInfo}(S, a)}
$$

where $\text{SplitInfo}(S, a) = -\sum_v \frac{|S_v|}{|S|} \log_2 \frac{|S_v|}{|S|}$ is the entropy of the split itself, penalising features with many values.

The **CART** (Classification and Regression Trees) algorithm, developed by Breiman, Friedman, Olshen, and Stone (1984), uses the **Gini impurity** instead of entropy. Gini impurity is $G(T) = 1 - \sum_c P(c)^2$, and the Gini gain is analogous to information gain. Gini impurity is closely related to entropy (both measure node impurity) and often produces similar trees.

Modern gradient boosting libraries like XGBoost and LightGBM use information gain-based criteria for split finding, with refinements for efficiency (histogram-based splitting) and regularisation.

## Real World Examples

1. **Medical diagnosis:** A decision tree predicts whether a patient has diabetes. The root split is on "Blood Sugar Level" because it has the highest information gain. Other features like "Age" and "BMI" may appear deeper in the tree. Information gain reveals which tests are most diagnostic.

2. **Credit scoring:** A bank builds a decision tree to predict loan default. Information gain identifies "Credit History" as the most informative feature, followed by "Debt-to-Income Ratio" and "Employment Status." The resulting tree is both predictive and explainable for regulatory compliance.

3. **Customer churn:** A telecom company builds a random forest to predict customer churn. Feature importance (average information gain across all trees) shows that "Contract Length" and "Customer Service Calls" are the top predictors. This guides retention strategies.

4. **Spam detection:** A decision tree classifier for spam uses information gain to select the most discriminative words. "Free" might have high IG, appearing near the root. "Dear" might have lower IG, appearing deeper or not at all.

5. **Fraud detection:** In a gradient boosting model for credit card fraud, XGBoost's split finding uses a gain-based criterion. The model learns that "Transaction Amount" and "Time Since Last Transaction" are high-gain features for detecting fraud.

## AI/ML Relevance

**Decision tree induction (ID3):** At each node, evaluate every feature, compute IG, and split on the feature with highest IG. Recurse on each child. Stop when all instances in a node belong to the same class (entropy = 0) or when no features remain.

**Gain ratio (C4.5):** Corrects for the bias of IG toward features with many values:
$$
\text{GR}(S, a) = \frac{IG(S, a)}{\text{SplitInfo}(S, a)}
$$
where $\text{SplitInfo}(S, a) = -\sum_v p_v \log_2 p_v$, with $p_v = |S_v|/|S|$.

**XGBoost split finding:** XGBoost uses a regularised gain function for split finding:
$$
\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma
$$
where $G$ and $H$ are the sum of gradients and Hessians in left/right branches, $\lambda$ is the L2 regularisation, and $\gamma$ is the complexity cost. This is a second-order approximation of information gain.

**Feature importance:** In tree-based ensembles, feature importance can be measured as:
- **Gain importance:** Total reduction in IG (or Gini impurity) contributed by splits on a feature
- **Weight importance:** Number of times a feature is used for splitting
- **Cover importance:** Average number of examples affected by splits on a feature

**Information gain ratio for feature selection:** The gain ratio is used in feature selection pipelines to rank features while mitigating the bias toward high-cardinality features.

**Bias-variance trade-off:** Information gain tends to create deep, specific splits that can overfit. Pruning techniques (cost-complexity pruning, reduced-error pruning) are essential to prevent overfitting.

## Mathematical Explanation

For a dataset $S$ with target classes $c \in C$, the entropy of the target is:

$$
H(T) = -\sum_{c \in C} P(c) \log_2 P(c)
$$

where $P(c)$ is the proportion of examples in class $c$.

When splitting on attribute $a$ with values $v \in V$, the dataset is partitioned into subsets $S_v$. The weighted remaining entropy is:

$$
H(T|a) = \sum_{v \in V} \frac{|S_v|}{|S|} H(T_v)
$$

Information gain is the difference:

$$
IG(S, a) = H(T) - H(T|a)
$$

**Equivalence to mutual information:** If we define random variables $T$ (target) and $A$ (attribute), then:
- $H(T)$ is the marginal entropy of the target
- $H(T|A) = \sum_v P(a=v) H(T|A=v)$ is the conditional entropy
- $IG = H(T) - H(T|A) = I(T; A)$, the mutual information

**Gini impurity alternative:** CART uses Gini impurity instead of entropy:
$$
G(T) = 1 - \sum_{c \in C} P(c)^2
$$

Gini gain is $\Delta G = G(T) - \sum_v \frac{|S_v|}{|S|} G(T_v)$. Gini impurity is similar to entropy but computationally cheaper (no logarithms).

**Normalised information gain:** Information gain can be normalised by the entropy of the target to get a value in $[0, 1]$:

$$
\text{Normalised IG} = \frac{IG(S, a)}{H(T)}
$$

This represents the fraction of target entropy explained by the attribute.

### Detailed Hand Calculation for the Tennis Dataset

**Problem:** Consider the classic "Play Tennis" dataset:

| Day | Outlook | Temperature | Humidity | Wind | PlayTennis |
|-----|---------|-------------|----------|------|------------|
| D1  | Sunny   | Hot         | High     | Weak | No         |
| D2  | Sunny   | Hot         | High     | Strong | No       |
| D3  | Overcast| Hot         | High     | Weak | Yes        |
| D4  | Rain    | Mild        | High     | Weak | Yes        |
| D5  | Rain    | Cool        | Normal   | Weak | Yes        |
| D6  | Rain    | Cool        | Normal   | Strong | No       |
| D7  | Overcast| Cool        | Normal   | Strong | Yes       |
| D8  | Sunny   | Mild        | High     | Weak | No         |
| D9  | Sunny   | Cool        | Normal   | Weak | Yes        |
| D10 | Rain    | Mild        | Normal   | Weak | Yes        |
| D11 | Sunny   | Mild        | Normal   | Strong | Yes       |
| D12 | Overcast| Mild        | High     | Strong | Yes       |
| D13 | Overcast| Hot         | Normal   | Weak | Yes        |
| D14 | Rain    | Mild        | High     | Strong | No        |

Compute the information gain for each feature to determine which should be the root split.

**Step 1: Entropy of the target $H(T)$**

Total: 14 examples. Yes: 9, No: 5.
$$P(\text{Yes}) = 9/14, \quad P(\text{No}) = 5/14$$

$$
H(T) = -\frac{9}{14}\log_2\frac{9}{14} - \frac{5}{14}\log_2\frac{5}{14}
$$
$$
H(T) = -0.643(-0.637) - 0.357(-1.486) = 0.410 + 0.530 = 0.940 \text{ bits}
$$

**Step 2: Information gain for Outlook**

Outlook has 3 values: Sunny (5 days), Overcast (4 days), Rain (5 days).

*Sunny subset (D1, D2, D8, D9, D11):* Yes: 2 (D9, D11), No: 3 (D1, D2, D8).
$$P(\text{Yes}) = 2/5 = 0.4, \quad P(\text{No}) = 3/5 = 0.6$$
$$H(T_{\text{Sunny}}) = -0.4\log_2 0.4 - 0.6\log_2 0.6 = -0.4(-1.322) - 0.6(-0.737) = 0.529 + 0.442 = 0.971 \text{ bits}$$

*Overcast subset (D3, D7, D12, D13):* Yes: 4, No: 0.
$$H(T_{\text{Overcast}}) = -1\log_2 1 - 0\log_2 0 = 0 \text{ bits}$$

*Rain subset (D4, D5, D6, D10, D14):* Yes: 3 (D4, D5, D10), No: 2 (D6, D14).
$$P(\text{Yes}) = 3/5 = 0.6, \quad P(\text{No}) = 2/5 = 0.4$$
$$H(T_{\text{Rain}}) = -0.6\log_2 0.6 - 0.4\log_2 0.4 = -0.6(-0.737) - 0.4(-1.322) = 0.442 + 0.529 = 0.971 \text{ bits}$$

Weighted remaining entropy:
$$H(T|\text{Outlook}) = \frac{5}{14}(0.971) + \frac{4}{14}(0) + \frac{5}{14}(0.971) = 0.347 + 0 + 0.347 = 0.694 \text{ bits}$$

Information gain:
$$IG(T, \text{Outlook}) = 0.940 - 0.694 = 0.246 \text{ bits}$$

**Step 3: Information gain for Temperature**

Temperature has 3 values: Hot (4 days), Mild (6 days), Cool (4 days).

*Hot subset (D1, D2, D3, D13):* Yes: 2 (D3, D13), No: 2 (D1, D2).
$$H(T_{\text{Hot}}) = -0.5\log_2 0.5 - 0.5\log_2 0.5 = 1 \text{ bit}$$

*Mild subset (D4, D8, D10, D11, D12, D14):* Yes: 4 (D4, D10, D11, D12), No: 2 (D8, D14).
$$P(\text{Yes}) = 4/6, \quad P(\text{No}) = 2/6$$
$$H(T_{\text{Mild}}) = -\frac{4}{6}\log_2\frac{4}{6} - \frac{2}{6}\log_2\frac{2}{6} = -0.667(-0.585) - 0.333(-1.585) = 0.390 + 0.528 = 0.918 \text{ bits}$$

*Cool subset (D5, D6, D7, D9):* Yes: 3 (D5, D7, D9), No: 1 (D6).
$$P(\text{Yes}) = 3/4, \quad P(\text{No}) = 1/4$$
$$H(T_{\text{Cool}}) = -0.75\log_2 0.75 - 0.25\log_2 0.25 = -0.75(-0.415) - 0.25(-2) = 0.311 + 0.5 = 0.811 \text{ bits}$$

Weighted remaining entropy:
$$H(T|\text{Temperature}) = \frac{4}{14}(1) + \frac{6}{14}(0.918) + \frac{4}{14}(0.811) = 0.286 + 0.393 + 0.232 = 0.911 \text{ bits}$$

Information gain:
$$IG(T, \text{Temperature}) = 0.940 - 0.911 = 0.029 \text{ bits}$$

**Step 4: Information gain for Humidity**

Humidity has 2 values: High (7 days), Normal (7 days).

*High subset (D1, D2, D3, D4, D8, D12, D14):* Yes: 3 (D3, D4, D12), No: 4 (D1, D2, D8, D14).
$$P(\text{Yes}) = 3/7, \quad P(\text{No}) = 4/7$$
$$H(T_{\text{High}}) = -\frac{3}{7}\log_2\frac{3}{7} - \frac{4}{7}\log_2\frac{4}{7} = -0.429(-1.222) - 0.571(-0.808) = 0.524 + 0.461 = 0.985 \text{ bits}$$

*Normal subset (D5, D6, D7, D9, D10, D11, D13):* Yes: 6 (D5, D7, D9, D10, D11, D13), No: 1 (D6).
$$P(\text{Yes}) = 6/7, \quad P(\text{No}) = 1/7$$
$$H(T_{\text{Normal}}) = -\frac{6}{7}\log_2\frac{6}{7} - \frac{1}{7}\log_2\frac{1}{7} = -0.857(-0.222) - 0.143(-2.807) = 0.190 + 0.401 = 0.591 \text{ bits}$$

Weighted remaining entropy:
$$H(T|\text{Humidity}) = \frac{7}{14}(0.985) + \frac{7}{14}(0.591) = 0.493 + 0.296 = 0.789 \text{ bits}$$

Information gain:
$$IG(T, \text{Humidity}) = 0.940 - 0.789 = 0.151 \text{ bits}$$

**Step 5: Information gain for Wind**

Wind has 2 values: Weak (8 days), Strong (6 days).

*Weak subset (D1, D3, D4, D5, D8, D9, D10, D13):* Yes: 6 (D3, D4, D5, D9, D10, D13), No: 2 (D1, D8).
$$P(\text{Yes}) = 6/8 = 0.75, \quad P(\text{No}) = 2/8 = 0.25$$
$$H(T_{\text{Weak}}) = -0.75\log_2 0.75 - 0.25\log_2 0.25 = -0.75(-0.415) - 0.25(-2) = 0.311 + 0.5 = 0.811 \text{ bits}$$

*Strong subset (D2, D6, D7, D11, D12, D14):* Yes: 3 (D7, D11, D12), No: 3 (D2, D6, D14).
$$H(T_{\text{Strong}}) = -0.5\log_2 0.5 - 0.5\log_2 0.5 = 1 \text{ bit}$$

Weighted remaining entropy:
$$H(T|\text{Wind}) = \frac{8}{14}(0.811) + \frac{6}{14}(1) = 0.463 + 0.429 = 0.892 \text{ bits}$$

Information gain:
$$IG(T, \text{Wind}) = 0.940 - 0.892 = 0.048 \text{ bits}$$

**Conclusion:** The information gains are:
- Outlook: 0.246 bits
- Humidity: 0.151 bits
- Wind: 0.048 bits
- Temperature: 0.029 bits

**Outlook** has the highest information gain and should be the root split.

## Formula(s)

**Information gain:**
$$
IG(S, a) = H(T) - \sum_{v \in \text{Values}(a)} \frac{|S_v|}{|S|} H(T_v)
$$

**Entropy of target:**
$$
H(T) = -\sum_{c \in C} P(c) \log_2 P(c)
$$

**Gain ratio (C4.5):**
$$
\text{GainRatio}(S, a) = \frac{IG(S, a)}{\text{SplitInfo}(S, a)}
$$

**Split information:**
$$
\text{SplitInfo}(S, a) = -\sum_{v \in \text{Values}(a)} \frac{|S_v|}{|S|} \log_2 \frac{|S_v|}{|S|}
$$

**Gini impurity:**
$$
G(T) = 1 - \sum_{c \in C} P(c)^2
$$

**Gini gain:**
$$
\Delta G(S, a) = G(T) - \sum_{v} \frac{|S_v|}{|S|} G(T_v)
$$

**XGBoost gain (regularised):**
$$
\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma
$$

**Normalised information gain:**
$$
\text{Normalised IG} = \frac{IG(S, a)}{H(T)}
$$

## Properties

- **Non-negativity:** $IG(S, a) \geq 0$, with $IG(S, a) = 0$ iff $T$ is independent of $a$ (i.e., $H(T|a) = H(T)$).
- **Upper bound:** $IG(S, a) \leq H(T)$, with equality iff $a$ perfectly determines $T$ (all child nodes are pure).
- **Symmetry in target:** $IG(S, a) = I(T; A) = I(A; T)$, but $IG$ is not symmetric between different features.
- **Bias toward many-valued features:** $IG$ tends to be higher for features with many distinct values, since the conditional entropy can be made smaller with finer partitions. SplitInfo in the gain ratio corrects for this.
- **Additivity:** For independent features, $IG$ values roughly add, but features with redundant information have overlapping $IG$ that cannot be summed.
- **Relation to mutual information:** $IG(S, a) = I(T; A)$, the mutual information between target and attribute.
- **Data processing inequality:** If $T \to A \to B$ (target influences $A$, then $B$), then $IG(T, A) \geq IG(T, B)$.
- **Invariance to class permutation:** $IG$ depends only on the distribution of classes across attribute values, not on the specific class labels.

## Step-by-Step Worked Examples

### Example 1: The Full Tennis Dataset (above)

The detailed computation for Outlook, Temperature, Humidity, and Wind is shown in the Mathematical Explanation section. The winner is Outlook with IG = 0.246 bits.

### Example 2: Binary Classification with Two Features

**Problem:** A small dataset has 6 examples with binary target $Y \in \{0,1\}$ and two binary features $A$ and $B$:

| Example | A | B | Y |
|---------|---|---|---|
| 1       | 0 | 0 | 0 |
| 2       | 0 | 1 | 0 |
| 3       | 0 | 1 | 1 |
| 4       | 1 | 0 | 1 |
| 5       | 1 | 0 | 1 |
| 6       | 1 | 1 | 1 |

Compute the information gain for $A$ and $B$ and determine which is the better split.

**Solution:**

Step 1: $H(Y)$.
Total: 6. $Y=0$: 2 (examples 1, 2). $Y=1$: 4 (examples 3, 4, 5, 6).
$$P(0) = 2/6 = 1/3, \quad P(1) = 4/6 = 2/3$$
$$H(Y) = -\frac{1}{3}\log_2\frac{1}{3} - \frac{2}{3}\log_2\frac{2}{3} = -0.333(-1.585) - 0.667(-0.585) = 0.528 + 0.390 = 0.918 \text{ bits}$$

Step 2: $IG(Y, A)$.
$A=0$: examples 1, 2, 3. $Y=0$: 2 (1, 2), $Y=1$: 1 (3).
$$H(Y|A=0) = -\frac{2}{3}\log_2\frac{2}{3} - \frac{1}{3}\log_2\frac{1}{3} = -0.667(-0.585) - 0.333(-1.585) = 0.390 + 0.528 = 0.918 \text{ bits}$$

$A=1$: examples 4, 5, 6. $Y=0$: 0, $Y=1$: 3.
$$H(Y|A=1) = -0\log_2 0 - 1\log_2 1 = 0 \text{ bits}$$

Weighted remaining entropy:
$$H(Y|A) = \frac{3}{6}(0.918) + \frac{3}{6}(0) = 0.459 \text{ bits}$$

$$IG(Y, A) = 0.918 - 0.459 = 0.459 \text{ bits}$$

Step 3: $IG(Y, B)$.
$B=0$: examples 1, 4, 5. $Y=0$: 1 (1), $Y=1$: 2 (4, 5).
$$H(Y|B=0) = -\frac{1}{3}\log_2\frac{1}{3} - \frac{2}{3}\log_2\frac{2}{3} = 0.918 \text{ bits}$$

$B=1$: examples 2, 3, 6. $Y=0$: 1 (2), $Y=1$: 2 (3, 6).
$$H(Y|B=1) = 0.918 \text{ bits}$$

Weighted remaining entropy:
$$H(Y|B) = \frac{3}{6}(0.918) + \frac{3}{6}(0.918) = 0.918 \text{ bits}$$

$$IG(Y, B) = 0.918 - 0.918 = 0 \text{ bits}$$

**Answer:** $IG(Y, A) = 0.459$ bits, $IG(Y, B) = 0$ bits. Feature $A$ is the better split.

Interpretation: $A$ perfectly separates the data into a pure $Y=1$ subset when $A=1$, while $B$ provides no information about $Y$ at all (the class distribution is the same for both values of $B$).

### Example 3: Information Gain with Continuous Features

**Problem:** A dataset has a continuous feature "Score" and binary target "Pass":

| Score | Pass |
|-------|------|
| 85    | Yes  |
| 92    | Yes  |
| 67    | No   |
| 73    | No   |
| 90    | Yes  |
| 55    | No   |
| 78    | Yes  |
| 60    | No   |

Find the best split threshold for Score using information gain.

**Solution:**

Step 1: Sort by Score. [(55, No), (60, No), (67, No), (73, No), (78, Yes), (85, Yes), (90, Yes), (92, Yes)].

Step 2: $H(T)$.
Total: 8. Pass=Yes: 4, Pass=No: 4.
$$H(T) = -0.5\log_2 0.5 - 0.5\log_2 0.5 = 1 \text{ bit}$$

Step 3: Evaluate candidate thresholds (midpoints between consecutive points with different classes).

Candidate thresholds:
- Between 73 (No) and 78 (Yes): threshold = 75.5
- Between 67 (No) and 78 (Yes): threshold = 72.5
- Between 60 (No) and 78 (Yes): threshold = 69

Let's evaluate threshold 75.5 (midpoint of 73 and 78).

Left branch (Score $\leq$ 75.5): examples (55, No), (60, No), (67, No), (73, No). All No.
$$H(T_{\text{left}}) = 0 \text{ bits}$$

Right branch (Score $>$ 75.5): (78, Yes), (85, Yes), (90, Yes), (92, Yes). All Yes.
$$H(T_{\text{right}}) = 0 \text{ bits}$$

Weighted remaining entropy:
$$H(T|\text{Score} \leq 75.5) = \frac{4}{8}(0) + \frac{4}{8}(0) = 0$$

$$IG = 1 - 0 = 1 \text{ bit}$$

This threshold perfectly separates the classes. Let's verify other thresholds to ensure 75.5 is best.

Threshold 72.5 (between 67 and 78):
Left (Score $\leq$ 72.5): (55, No), (60, No), (67, No). All No. $H=0$.
Right: (73, No), (78, Yes), (85, Yes), (90, Yes), (92, Yes). No: 1, Yes: 4.
$$H = -\frac{1}{5}\log_2\frac{1}{5} - \frac{4}{5}\log_2\frac{4}{5} = -0.2(-2.322) - 0.8(-0.322) = 0.464 + 0.258 = 0.722 \text{ bits}$$

Weighted: $\frac{3}{8}(0) + \frac{5}{8}(0.722) = 0.451$ bits. $IG = 1 - 0.451 = 0.549$ bits.

Threshold 69 (between 60 and 67):
Left: (55, No), (60, No). All No. $H=0$.
Right: (67, No), (73, No), (78, Yes), (85, Yes), (90, Yes), (92, Yes). No: 2, Yes: 4.
$$H = -\frac{2}{6}\log_2\frac{2}{6} - \frac{4}{6}\log_2\frac{4}{6} = 0.918 \text{ bits}$$

Weighted: $\frac{2}{8}(0) + \frac{6}{8}(0.918) = 0.689$ bits. $IG = 1 - 0.689 = 0.311$ bits.

**Answer:** The best threshold is 75.5, giving $IG = 1$ bit (perfect split). The Score feature with threshold 75.5 perfectly separates passes from failures.

## Visual Interpretation

Information gain can be visualised as the reduction in the "impurity" of a dataset after splitting. A bar chart showing entropy before and after the split illustrates the gain. The total height of the bars represents $H(T)$; after splitting, the weighted average of the child entropies (shorter bars) shows the remaining uncertainty.

A decision tree visualisation shows the recursive splitting process. At each node, the feature with highest IG is selected. The tree grows until all leaf nodes are pure (IG = 0) or no features remain. Each split is annotated with the IG value, showing the contribution of each decision.

For binary classification, a histogram of class proportions in each child node after splitting on a feature shows why IG is high or low: pure nodes (all one colour) give high IG; nodes with the same class mixture as the parent give zero IG.

## Common Mistakes

1. **Using information gain for features with many unique values:** Information gain is biased toward features with many values (e.g., a unique ID column). The gain ratio (C4.5) or other regularisation techniques should be used instead. A unique ID feature will have $IG = H(T)$ because each subset has 1 example with entropy 0, but this is meaningless for generalisation.

2. **Ignoring the cost of splitting:** Information gain does not consider the cost of measuring a feature. In some applications, certain features are expensive to obtain (e.g., medical tests). Cost-sensitive splitting criteria incorporate feature cost alongside information gain.

3. **Continuous feature threshold selection:** When handling continuous features, evaluating all possible thresholds by sorting and scanning is efficient, but some implementations use approximations (e.g., percentile-based candidates) that can miss optimal splits.

4. **Assuming IG works identically for regression:** Information gain as defined above applies to classification. For regression trees, variance reduction (or standard deviation reduction) is used instead of entropy reduction. The Mean Squared Error (MSE) gain is analogous to IG for regression.

5. **Misinterpreting IG as predictive power on unseen data:** High IG on training data does not guarantee good performance on test data. Features with high IG can still lead to overfitting if the split is too specific. Cross-validation or pruning is essential.

6. **Forgetting to handle missing values:** Decision tree algorithms handle missing values differently. C4.5 sends examples with missing values to all children with fractional weights. XGBoost learns a default direction. Simply ignoring missing values can bias IG calculations.

7. **Correlated features reduce per-feature IG:** If two features are highly correlated, the second feature to be split on will appear to have lower IG than it actually has, because the first split already captured that information. This is why feature importance in trees can be misleading for correlated features.

## Interview Questions

### Beginner - 5

1. **Q:** What is information gain in the context of decision trees?
   **A:** Information gain measures the reduction in entropy of the target variable after splitting on a feature. The feature with the highest gain is chosen for the split.

2. **Q:** What is the range of information gain?
   **A:** $0 \leq IG \leq H(T)$ (the entropy of the target). Zero means the feature provides no information; $H(T)$ means the feature perfectly separates the classes.

3. **Q:** Which feature would a decision tree choose as the root: one with IG = 0.3 or IG = 0.1?
   **A:** The feature with IG = 0.3, because it provides more information about the target.

4. **Q:** What is the relationship between information gain and mutual information?
   **A:** Information gain $IG(S, a) = I(T; A)$, the mutual information between the target $T$ and the attribute $A$.

5. **Q:** Why is information gain biased toward features with many values?
   **A:** Features with more values create finer partitions, which tend to have lower entropy (purer subsets) by chance, even if the feature is not genuinely informative.

### Intermediate - 5

1. **Q:** What is the gain ratio and how does it address the bias of information gain?
   **A:** Gain ratio divides information gain by split information: $\text{GR} = IG / \text{SplitInfo}$, where $\text{SplitInfo} = -\sum_v p_v \log p_v$. SplitInfo penalises features with many values, normalising the gain. This is used in C4.5.

2. **Q:** How does information gain change if two features are perfectly correlated?
   **A:** The first feature split will capture all the information. The second correlated feature will have information gain near zero because the target's entropy has already been reduced to near zero in the child nodes. This is why tree-based feature importance can be misleading for correlated features.

3. **Q:** How do you compute information gain for a continuous feature?
   **A:** Sort the data by the feature value. Evaluate candidate thresholds at midpoints between consecutive values. Compute IG for each threshold (treating it as a binary split) and select the threshold with maximum IG. Efficient implementations use a single scan.

4. **Q:** Compare information gain with Gini impurity for decision tree splitting.
   **A:** Both measure node impurity. Entropy ($-\sum p\log p$) and Gini ($1-\sum p^2$) are similar — both are maximised at uniform distributions and minimised at pure nodes. Gini is slightly faster (no logarithms) and sometimes produces different splits, but the resulting trees are often similar. Entropy tends to produce more balanced trees; Gini tends to isolate the most frequent class.

5. **Q:** How is information gain used in XGBoost?
   **A:** XGBoost uses a regularised gain function based on second-order gradient statistics: $\text{Gain} = \frac{1}{2}[G_L^2/(H_L+\lambda) + G_R^2/(H_R+\lambda) - (G_L+G_R)^2/(H_L+H_R+\lambda)] - \gamma$, where $G$ and $H$ are sums of gradients and Hessians. This approximates the information gain in a gradient boosting context with L2 regularisation ($\lambda$) and tree complexity penalty ($\gamma$).

### Advanced - 3

1. **Q:** Derive the relationship between information gain and mutual information, and explain how this connects to the data processing inequality.
   **A:** $IG(S, a) = H(T) - H(T|a) = I(T; A)$. The data processing inequality states that for any function $f$, $I(T; A) \geq I(T; f(A))$. This means that post-processing a feature (e.g., binning, transforming) cannot increase its information gain. However, pre-processing that constructs better features from raw data (e.g., feature crosses, nonlinear transformations) can increase $I(T; f(\text{raw data}))$ if $f$ extracts meaningful patterns.

2. **Q:** Analyse the bias of information gain as a splitting criterion. Derive the gain ratio and discuss its limitations.
   **A:** IG is maximised by features that create many, small, pure partitions. This is addressed by the gain ratio: $\text{GR} = IG / \text{SplitInfo}$. However, gain ratio can overcorrect: a feature with very low split information (almost all examples in one branch) can get an artificially high gain ratio because the denominator is small. C4.5 handles this by (a) only considering features with above-average IG, then (b) selecting the one with highest gain ratio among them. An alternative is the distance-based splitting criterion used in several modern implementations.

3. **Q:** How does XGBoost's gain formula for split finding relate to the classical information gain? Derive the XGBoost gain from the second-order Taylor expansion of the loss function.
   **A:** XGBoost minimises $\mathcal{L} = \sum_i l(y_i, \hat{y}_i) + \sum_j \Omega(f_j)$ where each $f_j$ is a tree and $\Omega$ is a regularisation term. For a given tree structure, the loss at a leaf is approximated by a second-order Taylor expansion: $\mathcal{L} \approx \sum_i [g_i f(x_i) + \frac{1}{2}h_i f(x_i)^2] + \frac{1}{2}\lambda \sum_j w_j^2$, where $g_i$ and $h_i$ are gradients and Hessians. For a split producing left ($L$) and right ($R$) branches, the optimal leaf weights are $w_L^* = -G_L/(H_L+\lambda)$, $w_R^* = -G_R/(H_R+\lambda)$. The gain (reduction in loss) is: $\text{Gain} = \frac{1}{2}[G_L^2/(H_L+\lambda) + G_R^2/(H_R+\lambda) - (G_L+G_R)^2/(H_L+H_R+\lambda)] - \gamma$. When using logistic loss (binary classification), $g_i = \hat{y}_i - y_i$ and $h_i = \hat{y}_i(1-\hat{y}_i)$, and this gain approximates the information gain with L2 regularisation on leaf weights and a complexity penalty $\gamma$ on splits.

## Practice Problems

### Easy - 5

1. Compute $IG$ if $H(T) = 1$ bit and the weighted average child entropy is 0.4 bits.

2. If a feature perfectly splits the data into pure subsets, what is $IG$?

3. True or false: Information gain is always positive.

4. In a dataset with equal class distribution ($P = [0.5, 0.5]$), what is $H(T)$?

5. What is the minimum possible value of information gain?

### Medium - 5

1. A dataset has 10 examples, 5 Yes and 5 No. A binary feature splits them: branch 1 has 4 Yes, 1 No; branch 2 has 1 Yes, 4 No. Compute $IG$.

2. Given the following small dataset:

| Feature X | Target Y |
|-----------|----------|
| A         | 0        |
| A         | 0        |
| B         | 1        |
| B         | 1        |
| A         | 1        |
| B         | 0        |

Compute $IG(Y, X)$.

3. For the dataset in problem 2 above, compute the gain ratio.

4. A continuous feature has sorted values $[1, 2, 3, 4, 5]$ with targets $[0, 0, 1, 1, 1]$. Find the best split threshold.

5. Explain why a feature with 6 distinct values (one per example) in a 6-example dataset might have high IG but be useless for generalisation.

### Hard - 3

1. Derive the closed-form expression for information gain of a binary feature $A$ with values $\{0, 1\}$ in terms of the counts $n_{ij}$ where $i$ is the class and $j$ is the attribute value.

2. Prove that information gain $IG(S, a)$ is equivalent to mutual information $I(T; A)$ between the target $T$ and attribute $A$.

3. Compare and contrast information gain, gain ratio, and the XGBoost gain formula. Under what circumstances would each be preferred?

## Solutions

**Easy:**

1. $IG = 1.0 - 0.4 = 0.6$ bits.

2. $IG = H(T)$ (the total entropy of the target). All uncertainty is removed by the split.

3. False. $IG \geq 0$ (non-negative, not always positive). $IG = 0$ when the feature provides no information about the target.

4. $H(T) = -0.5\log_2 0.5 - 0.5\log_2 0.5 = 1$ bit.

5. 0 (when the feature provides no information about the target).

**Medium:**

1. $H(T) = 1$ bit (equal classes).
Branch 1: $4$ Yes, $1$ No. $H_1 = -0.8\log_2 0.8 - 0.2\log_2 0.2 = 0.722$ bits.
Branch 2: $1$ Yes, $4$ No. $H_2 = 0.722$ bits.
$H(T|X) = 0.5(0.722) + 0.5(0.722) = 0.722$ bits.
$IG = 1 - 0.722 = 0.278$ bits.

2. 6 examples. $Y=0$: 3 (examples 1, 2, 6), $Y=1$: 3 (examples 3, 4, 5). $H(T) = 1$ bit.
$X=A$: examples 1, 2, 5. $Y=0$: 2, $Y=1$: 1. $H(T|A) = -2/3\log(2/3) - 1/3\log(1/3) = 0.918$ bits.
$X=B$: examples 3, 4, 6. $Y=0$: 1, $Y=1$: 2. $H(T|B) = 0.918$ bits.
$H(T|X) = 3/6(0.918) + 3/6(0.918) = 0.918$ bits.
$IG = 1 - 0.918 = 0.082$ bits.

3. SplitInfo = $-[3/6\log(3/6) + 3/6\log(3/6)] = 1$ bit. GainRatio = $0.082/1 = 0.082$.

4. Sort: $(1,0), (2,0), (3,1), (4,1), (5,1)$. Candidate thresholds: 2.5 (between 2 and 3).
Threshold 2.5: left $(1,0),(2,0)$ = pure (0). Right $(3,1),(4,1),(5,1)$ = pure (0). $IG = H(T) - 0 = 1$ bit.
Threshold 1.5: left $(1,0)$ pure, right $(2,0),(3,1),(4,1),(5,1)$ with mixed classes. Lower IG.
Best threshold: 2.5 with $IG = 1$ bit.

5. A unique ID feature creates one subset per example. Each subset has entropy 0 (pure), so $H(T|ID) = 0$. Therefore $IG = H(T)$, which is maximal. However, this split would not generalise at all — it simply memorises the training data. This is the classic illustration of information gain's bias toward high-cardinality features.

**Hard:**

1. For binary target $T \in \{0,1\}$ and binary attribute $A \in \{0,1\}$:
Let $n = |S|$. Let $n_{00}$ = examples where $T=0, A=0$; $n_{01}$ = $T=0, A=1$; $n_{10}$ = $T=1, A=0$; $n_{11}$ = $T=1, A=1$.
Marginals: $n_{0\cdot} = n_{00}+n_{01}$, $n_{1\cdot} = n_{10}+n_{11}$, $n_{\cdot 0} = n_{00}+n_{10}$, $n_{\cdot 1} = n_{01}+n_{11}$.
$H(T) = -\frac{n_{0\cdot}}{n}\log\frac{n_{0\cdot}}{n} - \frac{n_{1\cdot}}{n}\log\frac{n_{1\cdot}}{n}$.
$H(T|A=0) = -\frac{n_{00}}{n_{\cdot 0}}\log\frac{n_{00}}{n_{\cdot 0}} - \frac{n_{10}}{n_{\cdot 0}}\log\frac{n_{10}}{n_{\cdot 0}}$.
$H(T|A=1) = -\frac{n_{01}}{n_{\cdot 1}}\log\frac{n_{01}}{n_{\cdot 1}} - \frac{n_{11}}{n_{\cdot 1}}\log\frac{n_{11}}{n_{\cdot 1}}$.
$IG = H(T) - \frac{n_{\cdot 0}}{n}H(T|A=0) - \frac{n_{\cdot 1}}{n}H(T|A=1)$.

2. $IG(S,a) = H(T) - H(T|A) = H(T) - [H(T,A) - H(A)] = H(T) + H(A) - H(T,A)$. Since $H(T,A) = H(T) + H(A) - I(T;A)$, substituting gives $IG(S,a) = H(T) + H(A) - [H(T) + H(A) - I(T;A)] = I(T;A)$. This completes the proof, confirming information gain is exactly mutual information.

3. **Information gain** (ID3): Simple, interpretable, but biased toward features with many values. Best for datasets with few categorical features of similar cardinality, or as a baseline.

**Gain ratio** (C4.5): Normalises by split information to penalise many-valued features. Best when features have varying cardinalities (e.g., some binary, some 10-valued). Can overcorrect for features with very low split information. C4.5 mitigates this by only considering features with above-average IG, then picking the best gain ratio among them.

**XGBoost gain**: Uses second-order gradient statistics with L1/L2 regularisation and complexity penalty. Best for gradient boosting settings where handling continuous features, missing values, and overfitting are critical. The regularisation terms $\lambda$ and $\gamma$ provide built-in pruning. XGBoost gain generalises information gain: for classification with shallow trees and no regularisation, it approximates the classical gain.

## Related Concepts

- Entropy (MATH-088) — the uncertainty measure that information gain reduces
- Mutual Information (MATH-091) — $IG(S,a) = I(T;A)$, the mutual information between target and attribute
- Cross Entropy (MATH-089) — related to the loss minimised by classification trees
- KL Divergence (MATH-090) — information gain is a type of KL divergence between distributions before and after splitting
- Decision Tree — the algorithm that uses information gain as a splitting criterion
- Gini Impurity — alternative splitting criterion used by CART
- Gain Ratio — normalised version used by C4.5

## Next Concepts

- Random Forest — ensemble of decision trees using information gain
- Gradient Boosting (XGBoost, LightGBM) — regularised gain-based split finding
- Feature Importance — ranking features by their total information gain across an ensemble
- Pruning — reducing overfitting in IG-based trees
- Chi-Square Split — statistical significance-based alternative to IG

## Summary

Information gain $IG(S,a) = H(T) - \sum_v (|S_v|/|S|) H(T_v)$ measures the reduction in entropy of the target $T$ after splitting dataset $S$ on attribute $a$. It is the primary splitting criterion in decision tree algorithms (ID3, C4.5, CART when using entropy). Higher IG indicates a more informative feature. Information gain is identical to mutual information $I(T;A)$. A detailed hand computation on the classic Play Tennis dataset shows Outlook (IG = 0.246 bits) as the best root split, followed by Humidity (0.151 bits), Wind (0.048 bits), and Temperature (0.029 bits). Information gain is biased toward features with many distinct values, which C4.5's gain ratio corrects using split information. In modern gradient boosting (XGBoost), a regularised second-order approximation of gain is used for split finding. Information gain also provides a foundation for feature importance ranking in ensemble methods.

## Key Takeaways

- $IG(S,a) = H(T) - \sum_v (|S_v|/|S|) H(T_v)$: reduction in entropy after splitting.
- $0 \leq IG \leq H(T)$; higher IG = more informative feature.
- $IG(S,a) = I(T; A)$: information gain = mutual information.
- ID3, C4.5, and CART use IG (or gain ratio) as the splitting criterion.
- Bias toward many-valued features: gain ratio corrects this.
- For continuous features: sort, evaluate thresholds at midpoints, pick max IG.
- XGBoost uses a regularised gain: $\frac{1}{2}[G_L^2/(H_L+\lambda) + G_R^2/(H_R+\lambda) - (G_L+G_R)^2/(H_L+H_R+\lambda)] - \gamma$.
- Gain importance: sum of IG reductions across all splits on a feature.
- Hand calculation on Play Tennis: Outlook (0.246) > Humidity (0.151) > Wind (0.048) > Temperature (0.029).
- Pruning is essential to prevent overfitting from IG-based growth.
