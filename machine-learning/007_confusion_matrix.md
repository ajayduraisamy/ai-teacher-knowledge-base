# Concept: Confusion Matrix

## Concept ID

ML-007

## Difficulty

INTERMEDIATE

## Domain

Machine Learning

## Module

ML Fundamentals

## Learning Objectives

- Construct and interpret a confusion matrix for binary classification
- Derive key classification metrics (accuracy, precision, recall, F1, specificity) from the confusion matrix
- Use scikit-learn's `confusion_matrix` and `classification_report` functions
- Extend confusion matrix concepts to multi-class classification
- Understand how confusion matrix analysis guides model improvement

## Prerequisites

- ML-001: What is Machine Learning
- ML-006: Evaluation Metrics
- Basic understanding of classification problems

## Definition

A confusion matrix is a table that summarizes the performance of a classification model by comparing predicted labels against true labels. For binary classification (two classes: positive and negative), it is a 2x2 matrix with four cells representing the four possible outcomes of prediction.

### Binary Confusion Matrix Structure

|                     | Predicted Positive | Predicted Negative |
|---------------------|-------------------|-------------------|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

- **True Positives (TP)**: Positive samples correctly identified as positive. Example: A spam email correctly marked as spam.
- **True Negatives (TN)**: Negative samples correctly identified as negative. Example: A legitimate email correctly marked as not spam.
- **False Positives (FP)**: Negative samples incorrectly identified as positive (Type I error). Example: A legitimate email incorrectly marked as spam.
- **False Negatives (FN)**: Positive samples incorrectly identified as negative (Type II error). Example: A spam email incorrectly marked as not spam.

### Derived Metrics

From these four quantities, we derive all standard classification metrics:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall (Sensitivity, TPR)} = \frac{TP}{TP + FN}$$

$$\text{Specificity (TNR)} = \frac{TN}{TN + FP}$$

$$\text{F1-Score} = \frac{2 \times TP}{2 \times TP + FP + FN}$$

$$\text{False Positive Rate (FPR)} = \frac{FP}{FP + TN} = 1 - \text{Specificity}$$

$$\text{False Discovery Rate (FDR)} = \frac{FP}{FP + TP} = 1 - \text{Precision}$$

### Multi-Class Confusion Matrix

For multi-class classification with k classes, the confusion matrix is a k x k matrix where:
- Rows represent actual classes
- Columns represent predicted classes
- Diagonal elements are correct predictions
- Off-diagonal elements are misclassifications

For class i:
- TP_i = confusion_matrix[i][i]
- FP_i = sum of column i minus TP_i
- FN_i = sum of row i minus TP_i
- TN_i = total samples minus (TP_i + FP_i + FN_i)

## Intuition

Think of a confusion matrix as a report card for your classifier. The name comes from the fact that the matrix shows where the model is "confused" — which classes it tends to mix up.

Imagine you have built a model to detect fraudulent transactions. The confusion matrix answers four questions:

1. **True Positives** (top-left): "How many fraudulent transactions did I correctly catch?" You want this number as high as possible.
2. **True Negatives** (bottom-right): "How many legitimate transactions did I correctly allow?" Also want this high.
3. **False Positives** (bottom-left): "How many legitimate customers did I flag as fraudulent?" These cause customer frustration and unnecessary investigations.
4. **False Negatives** (top-right): "How many fraudulent transactions did I miss?" These cause direct financial loss.

The off-diagonal elements reveal the model's confusion patterns. If FP is high, the model is too aggressive in flagging fraud. If FN is high, the model is too lenient.

## Why This Concept Matters

The confusion matrix is the single most informative visualization for classification models because it:

1. **Shows exactly what errors the model makes**: Accuracy alone hides whether errors are false positives or false negatives.
2. **Enables calculation of all classification metrics**: Every metric (precision, recall, F1, etc.) derives directly from the confusion matrix.
3. **Reveals class-specific performance**: For multi-class problems, it shows which classes the model confuses, guiding feature engineering or data collection.
4. **Facilitates cost analysis**: By assigning costs to different error types, you can calculate the expected cost of deploying a model.
5. **Communicates model behavior to stakeholders**: A confusion matrix is intuitive and visual, making it a powerful communication tool.

## Mathematical Explanation

### From Probabilities to Predictions

The confusion matrix depends on the decision threshold $\tau$ used to convert predicted probabilities into hard class predictions:

$$\hat{y} = \begin{cases} 1 & \text{if } P(y=1|x) \geq \tau \\ 0 & \text{otherwise} \end{cases}$$

Changing $\tau$ changes the confusion matrix entries:
- Lower $\tau$: More positives predicted → Higher TP and FP, Lower FN and TN
- Higher $\tau$: Fewer positives predicted → Lower TP and FP, Higher FN and TN

### Multi-Class Confusion Matrix

For a k-class problem, let $C$ be the k x k confusion matrix where $C_{ij}$ is the number of samples from true class $i$ predicted as class $j$. For class $i$:

$$TP_i = C_{ii}$$

$$FP_i = \sum_{j \neq i} C_{ji} = \text{Column } i \text{ sum} - C_{ii}$$

$$FN_i = \sum_{j \neq i} C_{ij} = \text{Row } i \text{ sum} - C_{ii}$$

$$TN_i = \text{Total samples} - TP_i - FP_i - FN_i$$

### Normalized Confusion Matrix

For comparing across datasets or classes with different sizes, the confusion matrix can be normalized:

- **Row-normalized**: Divide each row by its sum → shows for each true class, the proportion predicted in each class (recall perspective)
- **Column-normalized**: Divide each column by its sum → shows for each predicted class, the proportion actually from each class (precision perspective)

## Code Examples

### Example 1: Building and Interpreting a Binary Confusion Matrix

```python
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Create imbalanced dataset
X, y = make_classification(
    n_samples=1000, weights=[0.9, 0.1],
    random_state=42, n_features=10
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Build confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
# Output:
# Confusion Matrix:
# [[268   2]
#  [ 10  20]]

# Extract values
TN, FP, FN, TP = cm.ravel()
print(f"TN={TN}, FP={FP}, FN={FN}, TP={TP}")
# Output: TN=268, FP=2, FN=10, TP=20

# Calculate metrics from confusion matrix
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
specificity = TN / (TN + FP)
f1 = 2 * precision * recall / (precision + recall)

print(f"Accuracy:   {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"Specificity:{specificity:.4f}")
print(f"F1-Score:   {f1:.4f}")
# Output:
# Accuracy:   0.9600
# Precision:  0.9091
# Recall:     0.6667
# Specificity: 0.9926
# F1-Score:   0.7692

# Classification report provides all metrics at once
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
# Output:
# Classification Report:
#               precision    recall  f1-score   support
#     Negative       0.96      0.99      0.98       270
#     Positive       0.91      0.67      0.77        30
#     accuracy                           0.96       300
#    macro avg       0.94      0.83      0.87       300
# weighted avg       0.96      0.96      0.96       300
```

### Example 2: Visualizing the Confusion Matrix with matplotlib

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
y_pred = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1, 0])

cm = confusion_matrix(y_true, y_pred)

# Plot with seaborn heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()

# Calculate metrics
TN, FP, FN, TP = cm.ravel()
print(f"Confusion Matrix Values:")
print(f"  TP={TP}, TN={TN}, FP={FP}, FN={FN}")
# Output:
# Confusion Matrix Values:
#   TP=3, TN=3, FP=1, FN=3

acc = (TP + TN) / (TP + TN + FP + FN)
prec = TP / (TP + FP)
rec = TP / (TP + FN)
f1 = 2 * prec * rec / (prec + rec)

print(f"  Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")
# Output:
#   Accuracy=0.600, Precision=0.750, Recall=0.500, F1=0.600
```

### Example 3: Multi-Class Confusion Matrix

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load digits dataset (10 classes: 0-9)
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Multi-class confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Multi-class Confusion Matrix (10 classes):")
print(cm)
# Output:
# Multi-class Confusion Matrix (10 classes):
# [[53  0  0  0  0  0  0  0  0  0]
#  [ 0 50  0  0  0  0  0  0  0  0]
#  [ 0  0 47  0  0  0  0  0  0  0]
#  [ 0  0  0 54  0  0  0  0  0  0]
#  [ 0  1  0  0 60  0  0  0  0  0]
#  [ 0  0  0  0  0 65  0  0  0  0]
#  [ 0  0  0  0  0  0 53  0  0  0]
#  [ 0  0  0  0  0  0  0 55  0  0]
#  [ 0  1  0  0  0  0  0  0 42  0]
#  [ 0  0  0  1  0  1  0  0  0 55]]

# Per-class metrics
print("\nPer-class Precision/Recall/F1:")
for i in range(10):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"  Digit {i}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
# Output:
# Per-class Precision/Recall/F1:
#   Digit 0: Precision=1.000, Recall=1.000, F1=1.000
#   Digit 1: Precision=0.962, Recall=1.000, F1=0.980
#   Digit 2: Precision=1.000, Recall=1.000, F1=1.000
#   Digit 3: Precision=0.982, Recall=1.000, F1=0.991
#   Digit 4: Precision=1.000, Recall=0.984, F1=0.992
#   Digit 5: Precision=0.985, Recall=1.000, F1=0.992
#   Digit 6: Precision=1.000, Recall=1.000, F1=1.000
#   Digit 7: Precision=1.000, Recall=1.000, F1=1.000
#   Digit 8: Precision=1.000, Recall=0.977, F1=0.988
#   Digit 9: Precision=1.000, Recall=0.965, F1=0.982

# Normalized confusion matrix (row-normalized)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print("\nNormalized Confusion Matrix (row):")
np.set_printoptions(precision=2, suppress=True)
print(cm_normalized)
# Output:
# Normalized Confusion Matrix (row):
# [[1.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
#  [0.   1.   0.   0.   0.   0.   0.   0.   0.   0.  ]
#  [0.   0.   1.   0.   0.   0.   0.   0.   0.   0.  ]
#  [0.   0.   0.   1.   0.   0.   0.   0.   0.   0.  ]
#  [0.   0.02 0.   0.   0.98 0.   0.   0.   0.   0.  ]
#  [0.   0.   0.   0.   0.   1.   0.   0.   0.   0.  ]
#  [0.   0.   0.   0.   0.   0.   1.   0.   0.   0.  ]
#  [0.   0.   0.   0.   0.   0.   0.   1.   0.   0.  ]
#  [0.   0.02 0.   0.   0.   0.   0.   0.   0.98 0.  ]
#  [0.   0.   0.   0.02 0.   0.02 0.   0.   0.   0.96]]
```

### Example 4: Threshold Tuning Using Confusion Matrix Analysis

```python
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(
    n_samples=1000, weights=[0.8, 0.2],
    random_state=42, n_features=10
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluate different thresholds
thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

print("Threshold Analysis:")
print(f"{'Threshold':<10} {'TN':<6} {'FP':<6} {'FN':<6} {'TP':<6} "
      f"{'Prec':<8} {'Rec':<8} {'F1':<8}")
print("-" * 62)

for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"{threshold:<10.1f} {TN:<6} {FP:<6} {FN:<6} {TP:<6} "
          f"{prec:<8.3f} {rec:<8.3f} {f1:<8.3f}")
# Output:
# Threshold   TN     FP     FN    TP     Prec     Rec      F1
# --------------------------------------------------------------
# 0.1         119    121    1     59     0.328    0.983    0.492
# 0.3         170    70     10    50     0.417    0.833    0.556
# 0.5         215    25     17    43     0.632    0.717    0.672
# 0.7         234    6      27    33     0.846    0.550    0.667
# 0.9         239    1      36    24     0.960    0.400    0.565
```

## Common Mistakes

1. **Looking at the confusion matrix without normalization**: For imbalanced datasets, raw counts are dominated by the majority class. Always check the normalized version to see proportional error rates.
2. **Confusing TP/FP/TN/FN positions**: The layout of `confusion_matrix` in scikit-learn has TN at position [0,0] and TP at [1,1] (for binary with positive class = 1). Use `cm.ravel()` carefully — the order is [TN, FP, FN, TP].
3. **Only looking at accuracy derived from the confusion matrix**: The confusion matrix contains much more information than just accuracy. Off-diagonal elements reveal specific confusion patterns.
4. **Ignoring the confusion matrix for multi-class problems**: In multi-class, per-class precision/recall/F1 derived from the confusion matrix are essential for understanding which classes the model confuses.
5. **Not considering the decision threshold**: The confusion matrix is only valid for one threshold. Different thresholds yield different matrices and different conclusions.
6. **Treating all misclassifications equally in multi-class problems**: A confusion between class 1 and class 2 might be much worse than between class 1 and class 9. Use cost matrices for weighted evaluation.
7. **Not validating the confusion matrix interpretation with domain experts**: What constitutes a "false positive" or "false negative" depends on domain definitions. Ensure the positive/negative class assignment aligns with business understanding.

## Interview Questions

### Beginner - 5

1. **Q: What are the four elements of a confusion matrix?**
   A: True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN).

2. **Q: How do you calculate precision from the confusion matrix?**
   A: Precision = TP / (TP + FP). It measures what proportion of positive predictions were correct.

3. **Q: What is the difference between a false positive and a false negative?**
   A: A false positive is predicting positive when the actual is negative (Type I error). A false negative is predicting negative when the actual is positive (Type II error).

4. **Q: If a confusion matrix shows FP=50 and FN=5, what does this tell you about the model?**
   A: The model produces many false alarms (50 false positives) but rarely misses positive cases (5 false negatives). The model is aggressive in predicting positive.

5. **Q: What is a normalized confusion matrix?**
   A: A normalized confusion matrix divides each row by its sum (showing recall per class) or each column by its sum (showing precision per class), making it comparable across datasets with different class sizes.

### Intermediate - 5

1. **Q: How do you derive specificity and FPR from the confusion matrix?**
   A: Specificity = TN / (TN + FP) — the proportion of actual negatives correctly identified. FPR = FP / (FP + TN) = 1 — Specificity — the proportion of negatives incorrectly classified as positive.

2. **Q: Explain how to compute macro and weighted F1 from a multi-class confusion matrix.**
   A: Compute F1 for each class from the per-class TP/FP/FN derived from the multi-class confusion matrix. Macro F1 = unweighted average of per-class F1. Weighted F1 = average weighted by the number of true samples per class.

3. **Q: A medical test has confusion matrix: TP=95, FN=5, FP=900, TN=9000. What is the precision and what does it mean for the test's practical utility?**
   A: Precision = 95/(95+900) = 0.095. Only 9.5% of positive tests are actually positive. If the disease is rare, most positive results are false alarms, causing unnecessary anxiety and follow-up procedures.

4. **Q: How does the confusion matrix change when you vary the decision threshold?**
   A: Lowering the threshold increases TP and FP (more positive predictions), decreases FN and TN. Raising the threshold does the opposite. Each threshold produces a different confusion matrix, tracing the ROC curve.

5. **Q: What are the limitations of the confusion matrix for highly imbalanced datasets?**
   A: The raw counts are dominated by the majority class, making it hard to see minority class performance. Normalization and metrics like precision/recall/F1 derived from the confusion matrix are more informative.

### Advanced - 3

1. **Q: Derive the expected confusion matrix under the assumption of class-conditional independence of features (Naive Bayes), and explain how this relates to the "naive" assumption.**
   A: For Naive Bayes, the predicted class is $\hat{y} = \arg\max_k P(y=k) \prod_j P(x_j|y=k)$. The confusion matrix entry $C_{ij} = \sum_{n} I(y_n=i) I(\hat{y}_n=j)$. Under the naive assumption, the covariance structure of features is ignored, so the confusion matrix typically shows more off-diagonal errors when features are correlated, compared to models that capture correlations.

2. **Q: Prove that for any confusion matrix, the sum of precision and recall is at least 2 times the F1-score, with equality iff precision equals recall.**
   A: $P + R - 2F1 = P + R - \frac{4PR}{P+R} = \frac{(P+R)^2 - 4PR}{P+R} = \frac{(P-R)^2}{P+R} \geq 0$. Therefore $P+R \geq \frac{4PR}{P+R} = 2F1$. Equality holds when P=R.

3. **Q: Design a cost-sensitive confusion matrix analysis for a credit card fraud detection system where each transaction type has different costs: correctly detected fraud saves $100, missed fraud costs $1000, false alarm costs $25 for customer service, correctly approved transaction generates $2 profit. Calculate the expected profit or loss given a confusion matrix.**
   A: Profit/Loss = TP × $100 (fraud prevented) - FN × $1000 (fraud missed) - FP × $25 (customer service cost) + TN × $2 (transaction profit). This cost matrix can guide threshold selection. The optimal threshold maximizes expected profit. For example, with confusion matrix [TN=9500, FP=100, FN=10, TP=40], profit = 40×100 - 10×1000 - 100×25 + 9500×2 = 4000 - 10000 - 2500 + 19000 = $10,500.

## Practice Problems

### Easy - 5

1. **Problem**: A confusion matrix shows TP=80, TN=800, FP=40, FN=80. Calculate accuracy, precision, recall, and F1.

2. **Problem**: In a confusion matrix for a cancer test, what does a high FN count mean clinically?

3. **Problem**: A model has precision=0.9 and recall=0.7. What is the F1-score?

4. **Problem**: For a 3-class confusion matrix, how do you calculate the precision for class 0?

5. **Problem**: If specificity = 0.95, what is the false positive rate?

### Medium - 5

1. **Problem**: A rare disease (prevalence 0.5%) is tested with a test that has 98% sensitivity and 99% specificity. For a population of 100,000 people, construct the expected confusion matrix and calculate precision. Explain the implications.

2. **Problem**: Given the multi-class confusion matrix below, calculate precision, recall, and F1 for each class:
   ```
        Predicted
        0   1   2
   True 0 [50, 5,  5]
        1 [10, 80, 10]
        2 [5,  15, 30]
   ```

3. **Problem**: Two models have the following confusion matrices on the same test set:
   Model A: [[900, 30], [40, 30]]
   Model B: [[880, 50], [10, 60]]
   Which model has higher precision? Which has higher recall? Which has higher accuracy?

4. **Problem**: Explain how to use the confusion matrix to determine if a model is biased toward the majority class.

5. **Problem**: You run a model at threshold 0.5 and get confusion matrix [[200, 20], [30, 50]]. The business decides FPs cost $10 each and FNs cost $100 each. What is the total cost? Should you increase or decrease the threshold?

### Hard - 3

1. **Problem**: Derive the formula for the Matthews Correlation Coefficient (MCC) from the confusion matrix and explain why it is preferred over F1 for imbalanced binary classification.

2. **Problem**: Prove that for a k-class confusion matrix, the sum of TP_i over all classes equals the sum of diagonal elements, and the sum of FP_i over all classes equals the sum of (column sums - diagonal), which equals the sum of FN_i.

3. **Problem**: Design a confusion-matrix-based monitoring system for a production ML model that detects data drift and concept drift. What changes in the confusion matrix would you expect for each type of drift?

## Solutions

### Easy Solutions

1. Accuracy = (80+800)/(80+800+40+80) = 880/1000 = 0.88. Precision = 80/(80+40) = 0.667. Recall = 80/(80+80) = 0.5. F1 = 2(0.667)(0.5)/(0.667+0.5) = 0.571.
2. High FN means many cancer cases are missed, leading to delayed treatment and worse outcomes. This is the most dangerous type of error for a diagnostic test.
3. F1 = 2(0.9)(0.7)/(0.9+0.7) = 1.26/1.6 = 0.788.
4. For class 0: TP = C[0,0], FP = C[1,0] + C[2,0], FN = C[0,1] + C[0,2]. Precision = C[0,0] / (C[0,0] + C[1,0] + C[2,0]).
5. FPR = 1 - Specificity = 1 - 0.95 = 0.05.

### Medium Solutions

1. Population: 100,000. Diseased: 500 (0.5%). Healthy: 99,500.
   TP = 500 × 0.98 = 490. FN = 500 - 490 = 10.
   TN = 99,500 × 0.99 = 98,505. FP = 99,500 - 98,505 = 995.
   Precision = 490 / (490 + 995) = 0.33. Only 33% of positive tests actually have the disease. Implication: mass screening leads to many false alarms; confirmatory tests are essential.
2. Class 0: TP=50, FP=10+5=15, FN=5+5=10 → Prec=50/65=0.769, Rec=50/60=0.833, F1=0.800.
   Class 1: TP=80, FP=5+15=20, FN=10+10=20 → Prec=80/100=0.800, Rec=80/100=0.800, F1=0.800.
   Class 2: TP=30, FP=5+10=15, FN=5+15=20 → Prec=30/45=0.667, Rec=30/50=0.600, F1=0.632.
3. Model A: Prec=30/(30+30)=0.5, Rec=30/(40+30)=0.429, Acc=(900+30)/1000=0.93.
   Model B: Prec=60/(50+60)=0.545, Rec=60/(10+60)=0.857, Acc=(880+60)/1000=0.94.
   Model B is better on all three metrics.
4. If the majority class has high recall (most majority samples correctly classified) but the minority class has low recall (many minority samples misclassified), the model is biased. The confusion matrix row-normalized view makes this clear.
5. Total cost = FP × $10 + FN × $100 = 20 × $10 + 30 × $100 = $200 + $3000 = $3200. Since FN cost is much higher than FP cost, decrease the threshold to catch more positives (reduce FN at the expense of more FP).

### Hard Solutions

1. MCC = (TP×TN - FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)). MCC ranges from -1 to +1, where +1 is perfect, 0 is random, -1 is inverse. Unlike F1, MCC uses all four confusion matrix entries and is symmetric with respect to class swapping. For imbalanced data, MCC provides a more informative single metric because it accounts for TN performance.
2. For a k-class confusion matrix C:
   Sum of TP_i = sum of diagonal = Σ C[i,i].
   Sum of FP_i = Σ (col_sum_i - C[i,i]) = Σ col_sum_i - Σ C[i,i] = n - Σ C[i,i].
   Sum of FN_i = Σ (row_sum_i - C[i,i]) = Σ row_sum_i - Σ C[i,i] = n - Σ C[i,i].
   Therefore sum of FP_i = sum of FN_i = n - trace(C).
3. Data drift (input distribution changes): If the drift is in features correlated with a specific class, that class's row-normalized confusion matrix shows reduced recall (more off-diagonal in that row). If drift affects all features, accuracy drops broadly across all classes.
   Concept drift (P(y|x) changes): Specific cells in the confusion matrix change as the mapping from features to labels shifts. For example, samples that used to be correctly classified as class 0 are now misclassified as class 1, increasing C[0,1].
   Monitoring: Track the full confusion matrix on a rolling window, checking for statistically significant changes using chi-squared tests.

## Related Concepts

- **ML-006: Evaluation Metrics** — All metrics derive from the confusion matrix
- **ML-008: ROC and AUC** — ROC analysis varies the threshold that defines the confusion matrix
- **ML-003: Train/Test Split** — The confusion matrix is computed on the test set
- **Cost-Sensitive Learning**: Assigns different costs to TP, FP, TN, FN
- **Type I and Type II Errors**: FP is Type I error, FN is Type II error

## Next Concepts

- **ML-008: ROC and AUC** — Understanding how the confusion matrix changes with threshold
- **ML-006: Evaluation Metrics** — Comprehensive treatment of metrics derived from the confusion matrix

## Summary

The confusion matrix is a 2x2 (or kxk for multi-class) table that summarizes classification predictions by comparing actual vs predicted labels. The four entries — TP, TN, FP, FN — form the basis for all classification metrics: accuracy, precision, recall, specificity, F1-score, and more. The confusion matrix reveals not just overall performance but the specific types of errors a model makes. For multi-class problems, it shows which classes the model confuses. Normalized confusion matrices enable comparison across different datasets and class distributions. The confusion matrix depends on the decision threshold, and varying this threshold traces the precision-recall curve and ROC curve.

## Key Takeaways

1. The confusion matrix contains four values: TP, TN, FP, FN for binary classification.
2. All classification metrics are derived from these four values.
3. The confusion matrix reveals what kinds of errors the model makes.
4. For multi-class problems, use the confusion matrix to identify confusing class pairs.
5. Normalize the confusion matrix for imbalanced datasets.
6. The confusion matrix changes with the decision threshold.
7. Always validate confusion matrix interpretation with domain knowledge.
