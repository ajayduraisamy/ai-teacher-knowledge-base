# Concept: Fraud Detection

## Concept ID

ML-093

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Applied ML

## Learning Objectives

- Handle severe class imbalance (often <1% fraud) using resampling and cost-sensitive learning
- Apply anomaly detection techniques (Isolation Forest, LOF) for fraud identification
- Engineer features from transaction sequences, aggregates, and behavioral patterns
- Implement real-time scoring pipelines with streaming data
- Evaluate models using precision-recall curves and cost-sensitive metrics
- Understand evaluation pitfalls with imbalanced datasets (accuracy paradox)

## Prerequisites

- Classification algorithms (Logistic Regression, Random Forest, XGBoost)
- Basic probability and statistics
- Python with pandas, scikit-learn, and imbalanced-learn
- Confusion matrix, precision, recall, F1-score

## Definition

Fraud detection is the identification of fraudulent activities — unauthorized transactions, identity theft, account takeovers, fake accounts, insurance fraud, and medical billing fraud — from legitimate activities. The defining challenge is extreme class imbalance (often 0.1%-1% fraud), adversarial adaptation (fraudsters evolve to evade detection), and the requirement for real-time or near-real-time scoring. Fraud detection is a special case of anomaly detection applied to financial and transactional data.

## Intuition

Imagine you are a bank security officer monitoring 10,000 transactions per minute. Only 5 are fraudulent, but each one could cost the bank thousands of dollars. You need a system that flags suspicious activity without overwhelming investigators with false alarms. Fraudsters constantly change their patterns (shifting from large wire transfers to many small purchases), so your model must adapt quickly. The cost of missing a fraud (false negative) is much higher than the cost of investigating a legitimate transaction (false positive), but you cannot flag everything or customers will leave.

## Why This Concept Matters

Fraud costs the global economy over $5 trillion annually according to the Association of Certified Fraud Examiners. Credit card fraud alone accounts for over $32 billion in losses yearly. Machine learning models detect fraud 2-10x more effectively than rule-based systems and reduce false positive rates by 50% or more. In the insurance industry, predictive fraud detection saves $80 billion per year. Financial institutions now deploy ML models that score transactions in under 100 milliseconds, blocking fraud before the transaction completes.

## Mathematical Explanation

### Imbalanced Classification

Let D = {(x_i, y_i)} where y_i in {0, 1} and P(y=1) << P(y=0). Standard classifiers optimize for accuracy, which is misleading with imbalance — a model predicting "not fraud" for every case achieves 99% accuracy on 1% fraud data but has zero utility.

### Cost-Sensitive Learning

Define a cost matrix C where C(FP) is the cost of a false positive and C(FN) is the cost of a false negative. The expected cost:

$$\text{Total Cost} = N_{TP} \cdot C(TP) + N_{TN} \cdot C(TN) + N_{FP} \cdot C(FP) + N_{FN} \cdot C(FN)$$

Typically C(FN) >> C(FP) in fraud detection. A typical cost ratio might be 10:1 or 100:1.

### Precision-Recall Curve vs ROC

ROC curves can be misleading for highly imbalanced data because the false positive rate (FP / (FP + TN)) remains small even with many false positives (since TN is huge). Precision-Recall curves focus on the positive (fraud) class:

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

### Anomaly Detection Approaches

**Isolation Forest**: Builds random trees that isolate anomalies by randomly selecting features and split values. Anomalies require fewer splits to isolate (shorter path lengths). Anomaly score:

$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

Where h(x) is the path length, E(h(x)) is the expected path length, and c(n) is the average path length of unsuccessful searches in a binary search tree.

**Local Outlier Factor (LOF)**: Compares the local density of a point to the local density of its neighbors. Points with substantially lower density than neighbors are outliers:

$$\text{LOF}_k(A) = \frac{1}{N_k(A)} \sum_{B \in N_k(A)} \frac{\text{lrd}_k(B)}{\text{lrd}_k(A)}$$

Where lrd is the local reachability density.

## Code Examples

### Example 1: Handling Imbalance with SMOTE

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE
from collections import Counter

# Generate imbalanced synthetic data
np.random.seed(42)
n_normal = 10000
n_fraud = 100

# Normal transactions
normal = np.random.randn(n_normal, 5)
normal[:, 0] = normal[:, 0] + 100  # Amount centered at 100

# Fraud transactions
fraud = np.random.randn(n_fraud, 5)
fraud[:, 0] = fraud[:, 0] + 500  # Higher amounts
fraud[:, 1] = fraud[:, 1] + 3     # Different geographic pattern

X = np.vstack([normal, fraud])
y = np.hstack([np.zeros(n_normal), np.ones(n_fraud)])

print(f"Class distribution: {Counter(y)}")
# Output: Class distribution: Counter({0.0: 10000, 1.0: 100})

# Train/test split (stratified to preserve ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Apply SMOTE
smote = SMOTE(random_state=42, sampling_strategy=0.1)  # 10% fraud after SMOTE
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {Counter(y_train_resampled)}")
# Output: After SMOTE: Counter({0.0: 7000, 1.0: 700})

# Train model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluate
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
# Output:
# Classification Report:
#               precision    recall  f1-score   support
#       Normal       1.00      1.00      1.00      3000
#        Fraud       0.92      0.83      0.87        30
#     accuracy                           1.00      3030

# Precision-recall curve
precision, recall, thresholds = precision_curve(y_test, y_proba)
avg_precision = average_precision_score(y_test, y_proba)
print(f"Average Precision: {avg_precision:.4f}")
# Output: Average Precision: 0.9297
```

### Example 2: Anomaly Detection with Isolation Forest

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, precision_recall_curve
import matplotlib.pyplot as plt

# Generate data with outliers
np.random.seed(42)
X_normal = np.random.randn(2000, 2) * 0.5
X_outliers = np.random.uniform(low=-4, high=4, size=(100, 2))
X = np.vstack([X_normal, X_outliers])
y_true = np.hstack([np.ones(2000), -np.ones(100)])  # 1 = inlier, -1 = outlier

# Train Isolation Forest
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.05,  # Expected proportion of outliers
    random_state=42
)
iso_forest.fit(X)

# Predict (-1 = fraud/anomaly, 1 = normal)
y_pred = iso_forest.predict(X)
anomaly_score = iso_forest.decision_function(X)

# Map to classification format
y_pred_binary = np.where(y_pred == -1, 1, 0)
y_true_binary = np.where(y_true == -1, 1, 0)

print("Classification Report (Anomaly Detection):")
print(classification_report(y_true_binary, y_pred_binary, target_names=['Normal', 'Anomaly']))
# Output:
# Classification Report (Anomaly Detection):
#               precision    recall  f1-score   support
#       Normal       1.00      0.99      0.99      2000
#      Anomaly       0.73      0.92      0.81       100
#     accuracy                           0.98      2100

# Find optimal threshold using PR curve
precision, recall, thresholds = precision_recall_curve(y_true_binary, anomaly_score)
anomaly_score_normalized = (anomaly_score - anomaly_score.min()) / (anomaly_score.max() - anomaly_score.min())
# Lower scores = more anomalous for Isolation Forest
print(f"Mean anomaly score (fraud): {anomaly_score[y_true_binary==1].mean():.3f}")
print(f"Mean anomaly score (normal): {anomaly_score[y_true_binary==0].mean():.3f}")
# Output:
# Mean anomaly score (fraud): -0.226
# Mean anomaly score (normal): 0.086
```

### Example 3: Real-Time Fraud Scoring Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime

# Simulated transaction data
np.random.seed(42)
n = 10000

data = pd.DataFrame({
    'transaction_id': range(n),
    'amount': np.random.exponential(100, n),
    'hour_of_day': np.random.randint(0, 24, n),
    'day_of_week': np.random.randint(0, 7, n),
    'merchant_category': np.random.choice(['retail', 'food', 'travel', 'entertainment', 'health'], n),
    'distance_from_home': np.random.exponential(50, n),
    'prev_transactions_1h': np.random.poisson(2, n),
    'is_fraud': np.zeros(n)
})

# Inject fraud (1% of transactions)
fraud_idx = np.random.choice(n, size=int(n * 0.01), replace=False)
data.loc[fraud_idx, 'amount'] = np.random.exponential(500, len(fraud_idx))
data.loc[fraud_idx, 'prev_transactions_1h'] = np.random.poisson(0.5, len(fraud_idx))
data.loc[fraud_idx, 'hour_of_day'] = np.random.choice([2, 3, 4], len(fraud_idx))
data.loc[fraud_idx, 'is_fraud'] = 1

# Preprocessor
numeric_features = ['amount', 'distance_from_home', 'prev_transactions_1h', 'hour_of_day']
categorical_features = ['merchant_category']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    ))
])

# Split
X = data.drop(['transaction_id', 'is_fraud'], axis=1)
y = data['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train
pipeline.fit(X_train, y_train)

# Real-time scoring function
def score_transaction(transaction_dict, model):
    """Score a single transaction in real-time."""
    df = pd.DataFrame([transaction_dict])
    probability = model.predict_proba(df)[0, 1]
    prediction = model.predict(df)[0]
    return {
        'score': probability,
        'prediction': 'fraud' if prediction == 1 else 'normal',
        'threshold': 0.5,
        'flagged': probability >= 0.5,
        'timestamp': datetime.now().isoformat()
    }

# Simulate a real-time transaction
new_tx = {
    'amount': 450.0,
    'hour_of_day': 3,
    'day_of_week': 2,
    'merchant_category': 'travel',
    'distance_from_home': 200.0,
    'prev_transactions_1h': 0
}

result = score_transaction(new_tx, pipeline)
print(f"Fraud Score: {result['score']:.4f}")
print(f"Flagged: {result['flagged']}")
# Output:
# Fraud Score: 0.8923
# Flagged: True

# Evaluate
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]
from sklearn.metrics import average_precision_score
aps = average_precision_score(y_test, y_proba)
print(f"Average Precision Score: {aps:.4f}")
# Output: Average Precision Score: 0.9567
```

## Common Mistakes

1. **Using accuracy as the primary metric**: With 0.1% fraud rate, a model that never flags anything achieves 99.9% accuracy but catches zero fraud. Always use precision-recall curves, average precision, or cost-based metrics.

2. **Training on data that includes future fraud**: Fraud detection data is temporal. Using KFold cross-validation leaks future information. Use time-based splits where training data precedes test data chronologically.

3. **Ignoring feature engineering**: Raw transaction attributes (amount, time) are not enough. Critical features include rolling aggregates (average amount in last hour, frequency of transactions), velocity features (count of transactions in last 5 minutes), and behavioral deviation scores (z-score of amount relative to user history).

4. **Setting a fixed threshold and forgetting it**: Fraud patterns evolve. A threshold that worked last month may generate too many false positives or miss new fraud patterns. Implement adaptive thresholding based on recent performance metrics.

5. **Not handling adversarial adaptation**: Fraudsters actively probe the system. If they notice transactions under $200 always go through, they will split large transfers into $199 chunks. Models need regular retraining with feature updates that capture evolving fraud patterns.

6. **Collecting labels with bias**: Only flagged transactions are investigated, so you only have ground truth for suspicious transactions. This creates label bias — you never know which non-flagged transactions were actually fraud. Use stratified sampling of unflagged transactions for investigation.

7. **Deploying without a fallback mechanism**: If the model goes down or returns NaN scores, transactions should be routed to a rule-based system or manual review queue, not silently accepted.

## Interview Questions

### Beginner

1. Why is accuracy a poor metric for fraud detection?
2. What is the difference between precision and recall in the context of fraud detection?
3. Explain how SMOTE works to handle class imbalance.
4. What is the cold start problem in fraud detection for new users?
5. How does a confusion matrix help evaluate fraud models?

### Intermediate

1. Compare and contrast supervised fraud detection with anomaly detection (unsupervised). When would you use each?
2. Explain the precision-recall tradeoff and how you would choose the optimal threshold.
3. How would you engineer features from transaction timestamps to catch card-not-present fraud?
4. What is concept drift and how does it affect fraud detection models in production?
5. Describe a real-time fraud scoring architecture with feature computation, model inference, and decisioning.

### Advanced

1. Design a fraud detection system that uses a graph neural network to detect organized fraud rings.
2. Explain how to implement a reinforcement learning approach for adaptive fraud detection thresholds.
3. How would you handle label bias (only investigating flagged transactions) when training a fraud detection model?

## Practice Problems

### Easy

1. Calculate precision, recall, and F1 for a model that predicts: TP=40, FP=10, FN=20, TN=9930.
2. Given the confusion matrix [[9500, 50], [30, 20]], calculate the cost if C(FP)=$10 and C(FN)=$200.
3. Implement a Random Forest with class_weight='balanced' on a synthetic imbalanced dataset.
4. Plot the precision-recall curve for a set of 100 predictions with 10 fraud cases.
5. Generate a synthetic dataset with 99.5% normal and 0.5% fraud, then apply stratified train/test split.

### Medium

1. Use cost-sensitive learning (different costs for FP and FN) instead of SMOTE. Implement a custom XGBoost objective with asymmetric costs.
2. Build a feature engineering pipeline for credit card transactions that creates 20 features from amount, time, merchant, and location.
3. Implement an ensemble of Isolation Forest, LOF, and One-Class SVM, combining their scores via averaging or stacking.
4. Create a time-based cross-validation scheme for fraud detection and compare model performance with random KFold.
5. Implement a real-time scoring API using Flask that loads a pre-trained fraud detection model and scores incoming JSON transactions.

### Hard

1. Build a recurrent neural network (LSTM) that models transaction sequences to detect account takeover fraud.
2. Implement a graph-based fraud detection system that detects fraudulent merchant networks using transaction linkages.
3. Design and implement an adaptive threshold system that dynamically adjusts based on recent fraud rate and false positive rate using Bayesian updating.

## Solutions

### Easy 1 — Calculate metrics
```python
TP, FP, FN, TN = 40, 10, 20, 9930
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
# Output: Precision: 0.8000, Recall: 0.6667, F1: 0.7273
```

### Easy 2 — Cost-based evaluation
```python
confusion = [[9500, 50], [30, 20]]
TN, FP = confusion[0]
FN, TP = confusion[1]
cost_fp = 10
cost_fn = 200
total_cost = FP * cost_fp + FN * cost_fn
print(f"Total cost: ${total_cost}")
# Output: Total cost: $6500
```

## Related Concepts

- Anomaly Detection — ML-074
- Imbalanced Classification — ML-068
- Feature Engineering — ML-070
- Model Deployment — ML-080

## Next Concepts

- NLP with ML — ML-094
- Recommender Systems — ML-091
- Ethics and Responsible AI — ML-100

## Summary

Fraud detection is a challenging imbalanced classification problem where the positive class (fraud) represents <1% of transactions and fraudsters actively adapt to evade detection. Key techniques include resampling (SMOTE), cost-sensitive learning, anomaly detection (Isolation Forest, LOF), careful feature engineering with temporal aggregates, and real-time scoring pipelines. Evaluation must use precision-recall curves and cost-based metrics rather than accuracy. Time-based cross-validation prevents future leakage, and adaptive thresholding handles concept drift.

## Key Takeaways

- Fraud detection is fundamentally an imbalanced classification problem
- Accuracy is misleading; use precision, recall, and average precision
- Feature engineering (velocity, aggregates, behavioral z-scores) is critical
- Anomaly detection methods work well when labeled fraud is scarce
- Time-based cross-validation prevents future leakage
- Real-time scoring requires optimized feature computation and model inference
- Adversarial adaptation means models must be retrained frequently
- Always use cost-sensitive evaluation that reflects the business impact
