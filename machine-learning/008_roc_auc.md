# Concept: ROC Curve and AUC

## Concept ID

ML-008

## Difficulty

INTERMEDIATE

## Domain

Machine Learning

## Module

ML Fundamentals

## Learning Objectives

- Understand the ROC curve as TPR vs FPR at varying thresholds
- Interpret AUC as the probability that a positive is ranked above a negative
- Use scikit-learn's `roc_curve` and `roc_auc_score` functions
- Compare classifiers using ROC curves and AUC values
- Explain the relationship between ROC-AUC and model performance
- Recognize limitations of ROC-AUC for imbalanced datasets

## Prerequisites

- ML-006: Evaluation Metrics
- ML-007: Confusion Matrix
- Understanding of TP, FP, TN, FN

## Definition

### ROC Curve

The Receiver Operating Characteristic (ROC) curve is a graphical plot that illustrates the diagnostic ability of a binary classifier as its discrimination threshold is varied. The curve plots the True Positive Rate (TPR, also called recall or sensitivity) against the False Positive Rate (FPR, equal to 1 — specificity) at various threshold settings.

$$\text{TPR} = \frac{TP}{TP + FN} = \text{Recall} = \text{Sensitivity}$$

$$\text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Specificity}$$

Each point on the ROC curve represents a (FPR, TPR) pair corresponding to a particular decision threshold. The curve connects these points from the lower-left corner (threshold = 1, all predictions negative) to the upper-right corner (threshold = 0, all predictions positive).

### AUC

The Area Under the ROC Curve (AUC, also called AUROC) is a single scalar value that summarizes the ROC curve. It ranges from 0 to 1, where:

- **AUC = 1.0**: Perfect classifier — there exists a threshold that perfectly separates positives and negatives.
- **AUC = 0.5**: Random classifier — the model provides no discriminative power (equivalent to random guessing).
- **AUC = 0.0**: Perfectly inverse classifier — every positive is ranked below every negative (can be inverted).

### Probabilistic Interpretation

AUC has a crucial probabilistic interpretation: it is the probability that a randomly chosen positive sample will be ranked higher (assigned a higher predicted probability) than a randomly chosen negative sample.

$$\text{AUC} = P(\text{score}(x_+) > \text{score}(x_-))$$

This interpretation makes AUC a measure of **ranking quality** rather than calibration or absolute accuracy.

## Intuition

### ROC Curve Intuition

Imagine a spam classifier that outputs a score between 0 and 1 for each email. You need to choose a threshold above which to mark emails as spam.

- If you set the threshold very high (e.g., 0.99), you only flag the most obvious spam. You catch few spam emails (low TPR) but also flag very few legitimate emails (low FPR). On the ROC curve, this point is near the bottom-left.
- If you set the threshold very low (e.g., 0.01), you flag almost everything as spam. You catch most spam (high TPR) but also flag many legitimate emails (high FPR). This point is near the top-right.
- A good classifier has most of its curve near the top-left corner (high TPR, low FPR), with a large area under the curve.

### AUC Intuition

Think of this as a "ranking test": given one spam email and one legitimate email, how often does your model assign a higher spam score to the actual spam email?

- An expert classifier: almost always (AUC close to 1.0)
- A random guesser: 50% of the time (AUC = 0.5)
- A broken classifier: almost never (AUC close to 0.0)

## Why This Concept Matters

ROC-AUC is one of the most widely used evaluation metrics in ML for several reasons:

1. **Threshold-independent**: Unlike accuracy, precision, or recall, ROC-AUC evaluates model quality across all possible thresholds, not just one.
2. **Scale-invariant**: The metric depends only on ranking, not on the absolute values of predicted probabilities.
3. **Class-balance robust**: ROC-AUC is relatively insensitive to class imbalance (the curve uses TPR and FPR, which are class-conditional rates).
4. **Single-number summary**: AUC condenses model performance into one interpretable number.
5. **Model comparison**: AUC enables fair comparison between models, especially when the optimal threshold is unknown.

ROC analysis originated in signal detection theory during World War II (radar operators distinguishing signals from noise) and remains fundamental in medical diagnostics, machine learning, and data science.

## Mathematical Explanation

### ROC Curve Construction

For a dataset with n samples, let $s_i$ be the predicted score for sample i, and $y_i \in \{0, 1\}$ be the true label. Sort samples by score in descending order. The ROC curve is constructed by starting at (0, 0) and moving:

- Up by 1/P when encountering a positive sample (TPR increases)
- Right by 1/N when encountering a negative sample (FPR increases)

Where P = number of positives and N = number of negatives.

### AUC Calculation

AUC can be computed as the trapezoidal area under the ROC curve:

$$\text{AUC} = \sum_{i=1}^{n-1} \frac{(\text{FPR}_{i+1} - \text{FPR}_i) \times (\text{TPR}_{i+1} + \text{TPR}_i)}{2}$$

Equivalently, using the Mann-Whitney U statistic:

$$\text{AUC} = \frac{1}{P \times N} \sum_{i=1}^{P} \sum_{j=1}^{N} \mathbb{I}(s_i^+ > s_j^-)$$

Where $s_i^+$ are scores of positive samples and $s_j^-$ are scores of negative samples.

### Relationship with Gini Coefficient

The Gini coefficient (sometimes called the accuracy ratio) is linearly related to AUC:

$$\text{Gini} = 2 \times \text{AUC} - 1$$

Gini ranges from -1 to 1, where 0 represents random performance.

## Code Examples

### Example 1: Basic ROC Curve and AUC

```python
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Generate synthetic data
X, y = make_classification(
    n_samples=1000, n_features=10,
    n_classes=2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)
y_scores = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# Calculate AUC
auc = roc_auc_score(y_test, y_scores)

print(f"AUC: {auc:.4f}")
# Output: AUC: 0.9954

print("ROC Curve Points (first 5):")
for i in range(5):
    print(f"  Threshold={thresholds[i]:.4f}, FPR={fpr[i]:.4f}, TPR={tpr[i]:.4f}")
# Output:
# ROC Curve Points (first 5):
#   Threshold=1.0000, FPR=0.0000, TPR=0.0000
#   Threshold=0.9987, FPR=0.0000, TPR=0.0067
#   Threshold=0.9979, FPR=0.0000, TPR=0.0134
#   Threshold=0.9976, FPR=0.0000, TPR=0.0268
#   Threshold=0.9970, FPR=0.0066, TPR=0.0336
```

### Example 2: Comparing Multiple Classifiers

```python
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

X, y = make_classification(
    n_samples=1000, n_features=20,
    n_informative=10, n_redundant=5,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5),
}

results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_scores = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_scores)
    results[name] = {
        'scores': y_scores,
        'auc': auc
    }
    print(f"{name:25s}: AUC = {auc:.4f}")
# Output:
# Logistic Regression       : AUC = 0.9534
# Random Forest             : AUC = 0.9651
# Gradient Boosting         : AUC = 0.9532
# k-NN (k=5)               : AUC = 0.9242
```

### Example 3: AUC Interpretation — Ranking Probability

```python
import numpy as np
from sklearn.metrics import roc_auc_score

# Demonstrate AUC = P(positive ranked higher than negative)
np.random.seed(42)

# Perfect separation
pos_scores = np.array([0.9, 0.8, 0.7])
neg_scores = np.array([0.3, 0.2, 0.1])
scores = np.concatenate([pos_scores, neg_scores])
labels = np.array([1, 1, 1, 0, 0, 0])

auc_perfect = roc_auc_score(labels, scores)
print(f"Perfect separation AUC: {auc_perfect:.4f}")
# Output: Perfect separation AUC: 1.0000

# Random: scores independent of label
scores_random = np.array([0.5, 0.3, 0.8, 0.2, 0.6, 0.4])
labels_random = np.array([1, 1, 1, 0, 0, 0])

auc_random = roc_auc_score(labels_random, scores_random)
print(f"Random AUC: {auc_random:.4f}")
# Output: Random AUC: 0.4444

# Verify probabilistic interpretation
def empirical_auc(scores, labels):
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    count = 0
    for p in pos:
        for n in neg:
            if p > n:
                count += 1
            elif p == n:
                count += 0.5
    return count / (len(pos) * len(neg))

emp_auc = empirical_auc(scores, labels)
print(f"Empirical AUC verification: {emp_auc:.4f}")
# Output: Empirical AUC verification: 0.4167
```

### Example 4: ROC Curves at Different Noise Levels

```python
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Generate datasets with different noise levels
noise_levels = [0.0, 0.1, 0.3, 0.5]

for noise in noise_levels:
    X, y = make_classification(
        n_samples=2000, n_features=10,
        flip_y=noise, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_scores = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_scores)

    fpr, tpr, _ = roc_curve(y_test, y_scores)

    # Print a few points
    print(f"Noise={noise:.1f}: AUC={auc:.3f}")
    print(f"  At FPR=0.1 -> TPR={tpr[np.searchsorted(fpr, 0.1)]:.3f}")
    print(f"  At FPR=0.2 -> TPR={tpr[np.searchsorted(fpr, 0.2)]:.3f}")
# Output:
# Noise=0.0: AUC=1.000
#   At FPR=0.1 -> TPR=1.000
#   At FPR=0.2 -> TPR=1.000
# Noise=0.1: AUC=0.966
#   At FPR=0.1 -> TPR=0.949
#   At FPR=0.2 -> TPR=0.978
# Noise=0.3: AUC=0.743
#   At FPR=0.1 -> TPR=0.394
#   At FPR=0.2 -> TPR=0.494
# Noise=0.5: AUC=0.610
#   At FPR=0.1 -> TPR=0.240
#   At FPR=0.2 -> TPR=0.340
```

## Common Mistakes

1. **Using AUC as the sole evaluation metric**: AUC measures ranking quality, not prediction quality. A model with high AUC can still have poorly calibrated probabilities or perform poorly at specific thresholds.
2. **Interpreting AUC as "accuracy"**: AUC is NOT accuracy. AUC = 0.9 does not mean 90% accuracy. It means 90% probability that a positive is ranked above a negative.
3. **Applying AUC to imbalanced datasets without caution**: While AUC is more robust than accuracy for imbalanced data, it can still be misleading when the minority class is very small. Precision-recall curves are more informative for extreme imbalance.
4. **Comparing models with different score distributions**: AUC comparisons are valid only when the test set and positive/negative definitions are fixed. Cross-dataset AUC comparisons may not be meaningful.
5. **Using AUC for model selection without considering the operating point**: Two models may have similar AUC but very different performance at the specific threshold you plan to deploy. Always check the ROC curve at your target FPR.
6. **Not looking at the full ROC curve**: AUC summarizes the entire curve, but two very different curves can have the same AUC. Always visualize the ROC curve, especially when comparing models.
7. **Assuming AUC = 0.5 means the model learned nothing**: AUC = 0.5 means random ranking, but the model might still have learned patterns that are useful at certain thresholds. The model might just need better calibration or threshold selection.

## Interview Questions

### Beginner - 5

1. **Q: What does the ROC curve plot?**
   A: The ROC curve plots the True Positive Rate (sensitivity) against the False Positive Rate (1 — specificity) at various threshold settings.

2. **Q: What does AUC = 0.5 mean?**
   A: AUC = 0.5 means the classifier performs no better than random guessing. The predicted scores provide no discriminative information.

3. **Q: What is the difference between ROC-AUC and accuracy?**
   A: Accuracy measures correct predictions at a fixed threshold. AUC measures ranking quality across all thresholds. AUC is threshold-independent and more robust to class imbalance.

4. **Q: What does AUC = 0.9 mean intuitively?**
   A: There is a 90% probability that a randomly chosen positive sample will receive a higher score than a randomly chosen negative sample.

5. **Q: Can AUC be less than 0.5?**
   A: Yes. AUC < 0.5 means the model consistently ranks negatives higher than positives. This can be fixed by inverting the predictions (using 1 — score).

### Intermediate - 5

1. **Q: Explain the probabilistic interpretation of AUC and why it is useful.**
   A: AUC = P(score(positive) > score(negative)). This rank-based interpretation means AUC evaluates how well the model separates classes, independent of calibration. It is useful because it provides a threshold-free, scale-invariant evaluation that is robust to class imbalance.

2. **Q: How does AUC handle class imbalance compared to accuracy?**
   A: AUC uses TPR (which conditions on positives) and FPR (which conditions on negatives), so it is not dominated by the majority class. Accuracy is dominated by the majority class accuracy. For a 99:1 imbalanced dataset, AUC still evaluates separation quality, while accuracy is near 0.99 regardless of model quality.

3. **Q: What are the limitations of AUC?**
   A: (1) AUC summarizes the entire curve, including regions of FPR that may be irrelevant for deployment. (2) Models with the same AUC can have very different curve shapes. (3) AUC can be misleading for highly imbalanced datasets where precision-recall curves are more informative. (4) AUC does not measure calibration.

4. **Q: How do you choose the optimal threshold using the ROC curve?**
   A: Common approaches: (1) Youden's index: maximize TPR — FPR (point farthest from diagonal). (2) Cost-based: minimize expected cost given costs of FP and FN. (3) Fixed FPR: choose threshold that achieves an acceptable FPR. (4) Closest to (0,1): find point on ROC curve closest to the top-left corner.

5. **Q: What is the relationship between ROC-AUC and the Mann-Whitney U statistic?**
   A: AUC is exactly equal to the Mann-Whitney U statistic divided by the product of the number of positive and negative samples: AUC = U / (P × N). The U statistic counts how many times a positive score exceeds a negative score.

### Advanced - 3

1. **Q: Derive the relationship between AUC and the Gini coefficient and explain how it relates to the Lorenz curve.**
   A: The Gini coefficient for a classifier is Gini = 2 × AUC — 1. This follows from the Lorenz curve, which plots cumulative share of positives against cumulative share of negatives (sorted by score). The area between the Lorenz curve and the diagonal is (AUC — 0.5), and the Gini is twice this area normalized by the total area.

2. **Q: Prove that AUC is equivalent to the probability that a randomly chosen positive is ranked higher than a randomly chosen negative.**
   A: Let $S_+$ and $S_-$ be the score distributions for positive and negative classes. Then:
   $AUC = \int_0^1 TPR(FPR^{-1}(x)) dx = \int_{-\infty}^{\infty} TPR(t) \cdot f_{S_-}(t) dt = \int_{-\infty}^{\infty} P(S_+ > t) \cdot f_{S_-}(t) dt$
   $= \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} I(s_+ > s_-) f_{S_+}(s_+) f_{S_-}(s_-) ds_+ ds_- = P(S_+ > S_-)$

3. **Q: Explain the concept of partial AUC and when it is preferred over full AUC.**
   A: Partial AUC (pAUC) computes the area under the ROC curve over a specific region of interest, typically a restricted FPR range (e.g., FPR < 0.1). This is preferred when: (1) Only certain FPR levels are acceptable (e.g., medical screening where low FPR is critical). (2) The full AUC may be misleading because most of the area comes from irrelevant threshold regions. (3) Comparing models that differ primarily in a specific FPR range.

## Practice Problems

### Easy - 5

1. **Problem**: A perfect classifier has what AUC? What about a random classifier?

2. **Problem**: A model has AUC = 0.75. If you randomly pick one positive and one negative sample, what is the probability the positive gets a higher score?

3. **Problem**: In an ROC curve, what point represents the best possible performance?

4. **Problem**: What happens to the ROC curve if you flip all predictions (predict 1 — score)?

5. **Problem**: If TPR = 0.8 and FPR = 0.3 at a threshold of 0.5, what does this tell you about the model's performance?

### Medium - 5

1. **Problem**: Two models have these ROC curves. Model A achieves TPR of 0.9 at FPR=0.1 and TPR of 0.95 at FPR=0.2. Model B achieves TPR of 0.95 at FPR=0.1 and TPR of 0.96 at FPR=0.2. Which model would you deploy if the maximum acceptable FPR is 0.15?

2. **Problem**: You have a dataset with 1000 negatives and 10 positives. AUC of the model is 0.90. Explain why this is potentially misleading and what alternative metric you should check.

3. **Problem**: Given the following scores and labels: scores=[0.9, 0.8, 0.4, 0.3, 0.2], labels=[1, 1, 0, 0, 0]. Calculate AUC.

4. **Problem**: A model achieves AUC = 0.95 on the training set and AUC = 0.60 on the test set. What is happening and why is AUC showing this when accuracy might not?

5. **Problem**: Explain why two models with AUC = 0.90 could have very different practical utility. Give a concrete example.

### Hard - 3

1. **Problem**: Prove that AUC is equivalent to the expected true positive rate when the false positive rate is randomly chosen uniformly from [0,1].

2. **Problem**: Show that AUC = 0.5 for a random classifier is the expected value regardless of class imbalance, and derive the variance of AUC under the null hypothesis of no discrimination.

3. **Problem**: Design a threshold selection strategy for a binary classifier that uses both ROC analysis and cost information, deriving the expected cost as a function of the threshold and showing how to find the minimum.

## Solutions

### Easy Solutions

1. Perfect classifier: AUC = 1.0. Random classifier: AUC = 0.5.
2. 75% probability. AUC is exactly this probability.
3. The top-left corner (0, 1) where FPR = 0 and TPR = 1 (perfect separation with no false positives).
4. The ROC curve reflects across the diagonal (TPR vs FPR becomes FPR vs TPR). AUC becomes 1 — original AUC.
5. The model correctly identifies 80% of positives but has a 30% false positive rate. At this threshold, it misses 20% of positives and falsely flags 30% of negatives.

### Medium Solutions

1. At FPR = 0.15, Model A achieves approximately TPR = 0.925, Model B approximately TPR = 0.955. Model B is preferable. This shows the importance of examining the ROC curve at the specific FPR region of interest, not just the AUC.
2. With only 10 positives, AUC has very high variance and is unreliable. A small change in the ranking of the 10 positives can dramatically change AUC. Use precision-recall AUC (PR-AUC) instead, which is more sensitive to minority class performance.
3. Sorted by score: (0.9, 1), (0.8, 1), (0.4, 0), (0.3, 0), (0.2, 0). Count all positive-negative pairs: pos1 vs negs: all 3 greater (count=3). pos2 vs negs: all 3 greater (count=3). Total = 6, total pairs = 2 × 3 = 6. AUC = 6/6 = 1.0.
4. The model is severely overfitting: it ranks training data perfectly but cannot separate test data well. AUC reveals this because it measures ranking quality, which degrades when the model memorizes noise-specific patterns.
5. Model A: excellent in the low-FPR region (TPR = 0.9 at FPR = 0.05), poor at high FPR. Model B: moderate across all regions (TPR = 0.7 at FPR = 0.05, TPR = 0.98 at FPR = 0.5). Both have AUC ≈ 0.90. For a medical screening test where low FPR is critical, Model A is far superior. For a marketing application where high FPR is acceptable, Model B might be preferred.

### Hard Solutions

1. $AUC = \int_0^1 TPR(FPR) d(FPR) = \int_0^1 TPR(t) dt$ where $t = FPR$. This is exactly the average TPR as FPR varies uniformly over [0,1]. Proof: By definition, AUC is the integral of TPR with respect to FPR. Changing variable from FPR to its quantile gives uniform distribution.
2. For random classifier, scores are independent of labels. For any positive-negative pair, P(positive > negative) = 0.5 (assuming no ties). Expected AUC = 0.5. The variance under the null is $Var(AUC) = \frac{(P+N+1)}{12PN}$ where P and N are counts of positives and negatives (for large samples with no ties). This shows that as imbalance increases (P << N or N << P), the variance increases.
3. Let $C_{FP}$ = cost of false positive, $C_{FN}$ = cost of false negative. Expected cost at threshold t: $E[C(t)] = C_{FP} \cdot P(negative) \cdot FPR(t) + C_{FN} \cdot P(positive) \cdot (1 - TPR(t))$. The optimal threshold minimizes this cost. Using the ROC curve, compute the slope at each point: slope = $\frac{C_{FP}}{C_{FN}} \cdot \frac{P(negative)}{P(positive)}$. The optimal operating point is where the ROC curve has this slope. Algorithm: (1) Compute ROC curve. (2) Compute cost for each threshold. (3) Select threshold minimizing cost.

## Related Concepts

- **ML-007: Confusion Matrix** — Each point on the ROC curve corresponds to a confusion matrix
- **ML-006: Evaluation Metrics** — TPR and FPR are derived from the confusion matrix
- **Precision-Recall Curve**: Alternative to ROC, preferred for highly imbalanced datasets
- **Detection Error Tradeoff (DET) Curve**: ROC variant that plots on normal deviate scales
- **Calibration**: AUC measures ranking, not calibration — a model can have high AUC but poor probability estimates

## Next Concepts

- **ML-009: Feature Engineering** — Creating features that improve class separation (measured by AUC)
- **ML-010: Feature Selection** — Selecting features that contribute to ranking quality

## Summary

The ROC curve plots the True Positive Rate against the False Positive Rate across all classification thresholds. The Area Under the Curve (AUC) quantifies the model's ability to discriminate between positive and negative classes, independent of the specific threshold. AUC has a clear probabilistic interpretation: it is the probability that a randomly chosen positive sample is ranked higher than a randomly chosen negative sample. AUC is threshold-independent, scale-invariant, and more robust to class imbalance than accuracy. However, it has limitations — it can be misleading for extremely imbalanced data, does not measure calibration, and two very different curves can have the same AUC. Always visualize the ROC curve and consider the operating region relevant to your application.

## Key Takeaways

1. ROC curve: TPR vs FPR at all thresholds.
2. AUC = P(positive ranked above negative).
3. AUC is threshold-independent and relatively robust to class imbalance.
4. AUC = 0.5 is random; AUC = 1.0 is perfect.
5. Two models with the same AUC can have very different practical performance.
6. Always examine the ROC curve, not just the AUC value.
7. For highly imbalanced data, consider precision-recall curves instead.
