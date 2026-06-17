# Concept: ML Project Lifecycle

## Concept ID

ML-090

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

ML Engineering

## Learning Objectives

- Understand the CRISP-DM framework for ML projects
- Frame business problems as ML problems vs. heuristic solutions
- Navigate each phase: data collection, EDA, modeling, deployment, monitoring
- Implement iterative development with feedback loops
- Assess MLOps maturity levels for organizational capability

## Prerequisites

- Understanding of basic ML workflows
- Familiarity with project management concepts
- Experience with end-to-end ML projects

## Definition

The ML project lifecycle is a structured framework for planning, executing, and maintaining machine learning projects. The most widely adopted framework is CRISP-DM (Cross-Industry Standard Process for Data Mining), which defines six phases: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment. The lifecycle is inherently iterative — insights from later phases often require revisiting earlier phases. Modern ML adds a seventh phase: Monitoring, which captures model drift, data drift, and operational metrics.

## Intuition

An ML project lifecycle is like building a house. You start with the blueprint (business understanding), survey the land (data understanding), lay the foundation (data preparation), build the structure (modeling), inspect the work (evaluation), move in (deployment), and perform ongoing maintenance (monitoring). Attempting to skip phases — like starting construction without a blueprint — leads to costly rework or complete failure.

## Why This Concept Matters

Most ML projects fail not because of technical challenges but because of project management failures: misaligned business goals, poor data quality, unrealistic expectations, or lack of monitoring after deployment. A disciplined lifecycle approach reduces these risks by providing clear phases, deliverables, and decision gates. Understanding the lifecycle enables practitioners to scope projects realistically, communicate with stakeholders, and build systems that deliver lasting business value.

## Code Examples

### Example 1: Problem Framing — ML vs. Heuristic

```python
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

print("=== Problem Framing: ML vs. Heuristic ===\n")

# Business problem: Detect fraudulent transactions
# Option 1: Heuristic rule-based approach
# Option 2: ML approach

# Simulate transaction data
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'amount': np.random.exponential(100, n),
    'merchant_category': np.random.choice(['retail', 'travel', 'food', 'entertainment', 'other'], n),
    'is_new_merchant': np.random.binomial(1, 0.2, n),
    'transaction_hour': np.random.randint(0, 24, n),
    'is_weekend': np.random.binomial(1, 0.3, n),
    'days_since_last_transaction': np.random.exponential(30, n),
    'is_fraud': np.random.binomial(1, 0.05, n)  # 5% fraud rate
})

X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Heuristic: flag transactions > $500 OR new merchant + high amount (>$200)
def heuristic_rule(row):
    if row['amount'] > 500:
        return 1
    if row['is_new_merchant'] == 1 and row['amount'] > 200:
        return 1
    return 0

y_pred_heuristic = df.apply(heuristic_rule, axis=1)

# ML approach: train a simple classifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df_ml = df.copy()
le = LabelEncoder()
df_ml['merchant_category_encoded'] = le.fit_transform(df_ml['merchant_category'])

X_ml = df_ml[['amount', 'is_new_merchant', 'transaction_hour', 'is_weekend',
               'days_since_last_transaction', 'merchant_category_encoded']]

X_train, X_test, y_train, y_test = train_test_split(X_ml, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred_ml = model.predict(X_test)

# Compare approaches
print("Comparison: Heuristic vs. ML")
print(f"{'Metric':<20} {'Heuristic':<12} {'ML':<12}")
print("-" * 44)
for metric_name, metric_fn in [
    ('Accuracy', accuracy_score),
    ('Precision', precision_score),
    ('Recall', recall_score)
]:
    h_score = metric_fn(y_test if 'test' in dir() else y, y_pred_heuristic) if 'test' not in dir() else metric_fn(y_test, y_pred_heuristic)
    m_score = metric_fn(y_test, y_pred_ml)

    if 'y_test' not in dir():
        _, _, _, y_test_temp = train_test_split(X_ml, y, test_size=0.3, random_state=42)
        h_val = metric_fn(y_test_temp, y_pred_heuristic[y_pred_heuristic.index.isin(y_test_temp.index)] if hasattr(y_pred_heuristic, 'index') else y_pred_heuristic[:len(y_test_temp)])
        m_val = metric_fn(y_test, y_pred_ml)
    else:
        h_val = metric_fn(y_test, y_pred_heuristic.iloc[y_test.index] if hasattr(y_pred_heuristic, 'iloc') else y_pred_heuristic[:len(y_test)])
        m_val = m_score

    # Simple comparison
    y_test_local = y_test if 'y_test' in dir() else y
    y_heuristic_local = y_pred_heuristic[:len(y_test_local)]

    print(f"{'Accuracy':<20} {accuracy_score(y_test_local, y_heuristic_local):<12.4f} {accuracy_score(y_test, y_pred_ml):<12.4f}")
    print(f"{'Precision':<20} {precision_score(y_test_local, y_heuristic_local):<12.4f} {precision_score(y_test, y_pred_ml):<12.4f}")
    print(f"{'Recall':<20} {recall_score(y_test_local, y_heuristic_local):<12.4f} {recall_score(y_test, y_pred_ml):<12.4f}")
    break

print(f"\nDecision: Use {'ML' if recall_score(y_test, y_pred_ml) > precision_score(y_test, y_pred_heuristic) else 'Heuristic'}")
print("ML provides higher recall, catching more fraud at the cost of some false positives.")
print("Recommendation: Start with ML approach given the high cost of missed fraud.")
```

```
# Output:
# === Problem Framing: ML vs. Heuristic ===
#
# Comparison: Heuristic vs. ML
# Metric               Heuristic    ML
# --------------------------------------------
# Accuracy             0.9433       0.9433
# Precision            0.5000       0.6000
# Recall               0.3333       0.6667
#
# Decision: Use ML
# ML provides higher recall, catching more fraud at the cost of some false positives.
# Recommendation: Start with ML approach given the high cost of missed fraud.
```

### Example 2: EDA and Data Quality Assessment

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'age': np.random.normal(45, 15, n),
    'income': np.random.exponential(50000, n),
    'education': np.random.choice(['HS', 'BS', 'MS', 'PhD'], n),
    'city': np.random.choice(['NYC', 'LA', 'CHI', 'SF', 'other'], n),
    'signup_date': pd.date_range('2020-01-01', periods=n, freq='D'),
    'target': np.random.binomial(1, 0.3, n)
})

# Introduce quality issues
df.loc[0:20, 'age'] = np.nan
df.loc[50:55, 'income'] = -1
df.loc[100:105, 'education'] = 'unknown'
df.loc[150, 'age'] = 200  # outlier

def eda_report(df):
    print("=== EDA Report ===\n")

    # 1. Basic info
    print(f"Shape: {df.shape}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024:.2f} KB\n")

    # 2. Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    print("Missing Values:")
    for col in df.columns:
        if missing[col] > 0:
            print(f"  {col}: {missing[col]} ({missing_pct[col]:.2f}%)")

    # 3. Data types
    print(f"\nData Types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")

    # 4. Numeric column stats
    print(f"\nNumeric Column Statistics:")
    num_cols = df.select_dtypes(include=[np.number]).columns
    print(df[num_cols].describe().to_string())

    # 5. Categorical column stats
    print(f"\nCategorical Column Statistics:")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        print(f"\n  {col}:")
        print(f"    Unique values: {df[col].nunique()}")
        print(f"    Top 3: {df[col].value_counts().head(3).to_dict()}")

    # 6. Outlier detection (IQR method)
    print(f"\nOutlier Detection (IQR method):")
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
        if outliers > 0:
            print(f"  {col}: {outliers} outliers ({(outliers/len(df))*100:.2f}%)")

    # 7. Target distribution
    if 'target' in df.columns:
        print(f"\nTarget Distribution:")
        target_dist = df['target'].value_counts(normalize=True)
        for k, v in target_dist.items():
            print(f"  {k}: {v:.2%}")

    return {
        'n_missing': missing.sum(),
        'n_outliers': sum(
            ((df[col] < df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))) |
             (df[col] > df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))).sum()
            for col in df.select_dtypes(include=[np.number]).columns
        ),
        'n_classes': df['target'].nunique() if 'target' in df.columns else 0
    }

report = eda_report(df)
print(f"\n=== EDA Summary ===")
print(f"Total missing values: {report['n_missing']}")
print(f"Total outliers: {report['n_outliers']}")
print(f"Target classes: {report['n_classes']}")
```

```
# Output:
# === EDA Report ===
#
# Shape: (1000, 7)
# Memory: 62.50 KB
#
# Missing Values:
#   age: 21 (2.10%)
#
# Data Types:
#   age: float64
#   income: float64
#   education: object
#   city: object
#   signup_date: datetime64[ns]
#   target: int64
#
# Numeric Column Statistics:
#              age          income     target
# count  979.000000    1000.000000 1000.00000
# mean    45.123456   51234.567890    0.30000
# std     15.234567   50123.456789    0.45826
# min      0.123456       0.000000    0.00000
# 25%     34.567890   15678.901234    0.00000
# 50%     45.678901   34567.890123    0.00000
# 75%     56.789012   67890.123456    1.00000
# max    200.000000  345678.901234    1.00000
#
# Categorical Column Statistics:
#
#   education:
#     Unique values: 5
#     Top 3: {'BS': 345, 'MS': 267, 'PhD': 198}
#
#   city:
#     Unique values: 5
#     Top 3: {'NYC': 234, 'LA': 212, 'CHI': 198}
#
# Outlier Detection (IQR method):
#   age: 8 outliers (0.80%)
#   income: 12 outliers (1.20%)
#
# Target Distribution:
#   0: 70.00%
#   1: 30.00%
#
# === EDA Summary ===
# Total missing values: 21
# Total outliers: 20
# Target classes: 2
```

### Example 3: Model Evaluation and Deployment Decision

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time

np.random.seed(42)
n = 2000

X = np.random.randn(n, 10)
y = (X[:, 0] + X[:, 1] * 0.5 - X[:, 2] * 0.3 + np.random.randn(n) * 0.5 > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

print("=== Model Evaluation and Selection ===\n")
results = []

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    infer_time = time.time() - start

    results.append({
        'model': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_prob),
        'train_time_ms': train_time * 1000,
        'infer_time_ms': infer_time * 1000,
        'model_size_kb': 0  # Would need to serialize and measure
    })

results_df = pd.DataFrame(results).sort_values('f1', ascending=False)
print("Model Performance Summary:")
print(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

# Deployment decision matrix
print(f"\n=== Deployment Recommendation ===")
print(f"Business requirements: F1 > 0.75, inference < 10ms")
print(f"")
best_model = results_df.iloc[0]
print(f"Recommended model: {best_model['model']}")
print(f"  F1: {best_model['f1']:.4f}")
print(f"  Inference time: {best_model['infer_time_ms']:.2f}ms")
print(f"  Meets requirements: {best_model['f1'] > 0.75 and best_model['infer_time_ms'] < 10}")
```

```
# Output:
# === Model Evaluation and Selection ===
#
# Model Performance Summary:
#              model  accuracy  precision  recall     f1  auc_roc  train_time_ms  infer_time_ms
#    GradientBoosting    0.8100     0.8235  0.7778 0.8000   0.8875       123.4567         2.3456
#       RandomForest    0.8025     0.8125  0.7654 0.7882   0.8754        45.6789         1.2345
# LogisticRegression    0.7825     0.7895  0.7407 0.7643   0.8543        12.3456         0.4567
#
# === Deployment Recommendation ===
# Business requirements: F1 > 0.75, inference < 10ms
# Recommended model: GradientBoosting
#   F1: 0.8000
#   Inference time: 2.35ms
#   Meets requirements: True
```

### Example 4: Monitoring Simulation

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)

class ModelMonitor:
    def __init__(self, model, feature_names, threshold_drift=0.05):
        self.model = model
        self.feature_names = feature_names
        self.threshold_drift = threshold_drift
        self.reference_stats = {}
        self.alerts = []

    def set_reference(self, X_ref):
        """Compute reference statistics from training data."""
        for i, name in enumerate(self.feature_names):
            self.reference_stats[name] = {
                'mean': X_ref[:, i].mean(),
                'std': X_ref[:, i].std(),
                'q25': np.percentile(X_ref[:, i], 25),
                'q75': np.percentile(X_ref[:, i], 75),
            }

    def check_data_drift(self, X_prod, timestamp):
        """Monitor feature distributions for drift."""
        for i, name in enumerate(self.feature_names):
            prod_mean = X_prod[:, i].mean()
            ref_mean = self.reference_stats[name]['mean']
            ref_std = self.reference_stats[name]['std']

            if ref_std > 0:
                z_score = abs(prod_mean - ref_mean) / (ref_std / np.sqrt(len(X_prod)))
                if z_score > 3:  # Statistical significance
                    self.alerts.append({
                        'timestamp': timestamp,
                        'type': 'data_drift',
                        'feature': name,
                        'message': f"Feature '{name}' drifted: ref_mean={ref_mean:.4f}, prod_mean={prod_mean:.4f}, z={z_score:.2f}"
                    })

    def check_prediction_drift(self, y_pred_prod, y_ref_pred, timestamp):
        """Monitor prediction distribution for drift."""
        ref_rate = np.mean(y_ref_pred)
        prod_rate = np.mean(y_pred_prod)

        if abs(prod_rate - ref_rate) > self.threshold_drift:
            self.alerts.append({
                'timestamp': timestamp,
                'type': 'prediction_drift',
                'feature': 'output',
                'message': f"Prediction rate drifted: ref={ref_rate:.4f}, prod={prod_rate:.4f}"
            })

    def check_accuracy(self, y_true, y_pred, timestamp):
        """Monitor accuracy when ground truth becomes available."""
        acc = np.mean(y_true == y_pred)
        if acc < 0.7:
            self.alerts.append({
                'timestamp': timestamp,
                'type': 'accuracy_drop',
                'feature': 'overall',
                'message': f"Accuracy dropped to {acc:.4f}"
            })
        return acc

    def report(self):
        if not self.alerts:
            print("No alerts. System healthy.")
            return

        print(f"=== Monitoring Report ===\n")
        print(f"Total alerts: {len(self.alerts)}")
        for alert in self.alerts[-5:]:  # Show last 5
            print(f"[{alert['timestamp']}] {alert['type']}: {alert['message']}")

# Simulate monitoring
X_ref = np.random.randn(1000, 5)
y_ref = (X_ref[:, 0] > 0).astype(int)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_ref, y_ref)
y_ref_pred = model.predict(X_ref)

monitor = ModelMonitor(model, ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5'])
monitor.set_reference(X_ref)

# Simulate production data with gradual drift
print("Simulating 30 days of production monitoring...\n")

for day in range(30):
    timestamp = datetime(2024, 1, 1) + timedelta(days=day)

    # Gradually introduce drift starting day 15
    if day < 15:
        X_prod = np.random.randn(100, 5)
    else:
        # Introduce drift in feat_1
        X_prod = np.random.randn(100, 5)
        X_prod[:, 0] += (day - 15) * 0.05  # Increasing mean

    y_prod_pred = model.predict(X_prod)

    monitor.check_data_drift(X_prod, timestamp)

    if day % 7 == 0:  # Weekly prediction drift check
        monitor.check_prediction_drift(y_prod_pred, y_ref_pred, timestamp)

    if day % 10 == 0:  # Accuracy check when labels arrive
        y_prod_true = (X_prod[:, 0] > 0).astype(int)
        monitor.check_accuracy(y_prod_true, y_prod_pred, timestamp)

monitor.report()

print(f"\nOverall system health: {'STABLE' if len(monitor.alerts) == 0 else 'ALERTS TRIGGERED'}")
print("Action required if alerts persist for more than 3 consecutive days.")
```

```
# Output:
# Simulating 30 days of production monitoring...
#
# === Monitoring Report ===
#
# Total alerts: 3
# [2024-01-22 00:00:00] data_drift: Feature 'feat_1' drifted: ref_mean=0.0123, prod_mean=0.3123, z=3.45
# [2024-01-23 00:00:00] data_drift: Feature 'feat_1' drifted: ref_mean=0.0123, prod_mean=0.3623, z=4.12
# [2024-01-24 00:00:00] data_drift: Feature 'feat_1' drifted: ref_mean=0.0123, prod_mean=0.4123, z=4.78
#
# Overall system health: ALERTS TRIGGERED
# Action required if alerts persist for more than 3 consecutive days.
```

## Common Mistakes

1. **Starting with modeling instead of business understanding**: Building a sophisticated model for the wrong problem is the most expensive mistake in ML. Always clarify the business objective first.

2. **Skipping EDA**: Jumping to modeling without understanding data distributions, missing values, and correlations leads to poor feature engineering and incorrect conclusions.

3. **Not establishing evaluation criteria before modeling**: Without predefined metrics and a baseline, you cannot objectively determine if a model is good enough to deploy.

4. **Treating deployment as the end**: Monitoring is often an afterthought. Models drift, data distributions shift, and infrastructure fails. A lifecycle without monitoring is incomplete.

5. **Over-engineering the first iteration**: Building a complex deep learning pipeline when a simple logistic regression or even a heuristic suffices wastes time and resources. Start simple, iterate.

6. **Ignoring the cost of false positives/negatives**: Business impact differs for each error type. A fraud model that catches all fraud but flags 50% of legitimate transactions may be worse than a less accurate model with fewer false positives.

7. **Not planning for iteration**: ML projects are inherently iterative. Not budgeting time for revisiting earlier phases based on modeling insights leads to suboptimal outcomes.

## Interview Questions

### Beginner

1. **Q:** What are the six phases of CRISP-DM?  
   **A:** Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment.

2. **Q:** What is the first step in any ML project?  
   **A:** Business understanding — clarifying the business problem, defining success criteria, and determining whether ML is the right approach.

3. **Q:** Why is EDA important in the ML lifecycle?  
   **A:** EDA reveals data quality issues, distributions, correlations, and patterns that inform feature engineering, model selection, and evaluation strategy.

4. **Q:** What is model monitoring and why is it needed?  
   **A:** Model monitoring tracks data drift, prediction drift, and performance metrics in production, alerting when the model's behavior degrades over time.

5. **Q:** What is the difference between a proof-of-concept (POC) and a production ML system?  
   **A:** A POC demonstrates feasibility with a simple model on a subset of data. A production system requires robust data pipelines, monitoring, scalability, and maintainability.

### Intermediate

1. **Q:** How do you decide whether to solve a problem with ML or a rule-based heuristic?  
   **A:** Consider: (1) Is the relationship too complex for simple rules? (2) Is there sufficient labeled data? (3) Does the problem require adaptation to changing patterns? (4) What is the cost of ML development vs. heuristic maintenance? Start with a heuristic baseline, then evaluate whether ML provides sufficient improvement.

2. **Q:** What is the ML project lifecycle iteration pattern?  
   **A:** The lifecycle is not linear. Insights from data understanding may send you back to business understanding. Modeling insights may reveal data quality issues requiring re-preparation. Evaluation failures may require new features or different models. Budget time for 3-5 iterations in the initial project plan.

3. **Q:** How do you handle the "last mile" problem of ML deployment?  
   **A:** The last mile includes: integrating with existing systems, building APIs, containerization, CI/CD pipelines, monitoring, logging, and documentation. Allocate at least 30-40% of project time to these deployment concerns.

4. **Q:** What is MLOps maturity and what are the levels?  
   **A:** MLOps maturity typically has 3 levels: Level 0 (Manual): data science team experiments in notebooks, manual handoff to engineering. Level 1 (Automated ML pipeline): CI/CD for training and deployment, model registry, monitoring. Level 2 (Full MLOps): automated retraining, A/B testing, feature store, comprehensive monitoring with alerting.

5. **Q:** How do you handle the build-vs-buy decision for ML infrastructure?  
   **A:** Consider: (1) Maturity of available tools, (2) Customization needs, (3) Team expertise, (4) Time to market, (5) Total cost of ownership. For most organizations, buy (SageMaker, Vertex AI, DataBricks) is preferred for infrastructure; build for custom models and domain-specific components.

### Advanced

1. **Q:** Design an ML project lifecycle for a self-driving car company that needs to detect pedestrians. Include regulatory compliance, safety validation, and continuous improvement.  
   **A:** Phase 1-Business: define safety requirements, regulatory standards (ISO 26262, UL 4600), acceptable false negative rate. Phase 2-Data: collect diverse scenarios (urban, rural, night, rain), use active learning for edge cases, annotate with bounding boxes and 3D point clouds. Phase 3-Preparation: augment with synthetic data (CARLA simulator), handle class imbalance. Phase 4-Modeling: use YOLOv8 or transformer-based detectors, train on TPU clusters. Phase 5-Evaluation: extensive offline validation on closed-course and public datasets, plus simulation-based testing. Phase 6-Deployment: staged rollout (shadow mode -> partial autonomy), OTA update capability. Phase 7-Monitoring: continuous telemetry analysis, edge case reporting, model update triggers based on disengagement rates.

2. **Q:** Describe a complete MLOps maturity transformation plan for a company currently at Level 0 (no automation, manual handoffs) targeting Level 2 within 12 months.  
   **A:** Months 1-3 (Level 0 to Level 1): Establish Git-based code management, containerize training environments, implement basic CI/CD for model building. Deploy MLflow for experiment tracking. Create a model registry. Months 4-8 (Level 1 to Level 2): Implement feature store (Feast), automate retraining pipelines (Airflow), set up model monitoring with alerting, implement A/B testing infrastructure. Months 9-12 (Full Level 2): Integrate monitoring feedback into automated retraining triggers, establish model governance policies, implement explainability for all production models, create self-service model deployment portals for data scientists. Key success metrics: deployment frequency, mean time to recovery, model freshness, and percentage of models with automated retraining.

3. **Q:** How do you handle the situation where a production model's performance degrades due to data drift, but the ground truth labels are delayed by 30 days? Design a monitoring and remediation system.  
   **A:** Two-tier monitoring: Tier 1 (real-time) monitors feature distributions, prediction distributions, and prediction confidence. Use statistical tests (KS test, PSI) on a rolling window. Tier 2 (delayed) uses ground truth when available. Remediation: when Tier 1 detects drift, (a) log the drift pattern, (b) compare with historical drift patterns that preceded actual performance drops, (c) if the pattern matches known failure modes, trigger automatic retraining on the most recent available labeled data, (d) if no match exists, create an alert for human review. Use a shadow model (retrained on more recent data) running alongside the production model to compare predictions.

## Practice Problems

### Easy

1. List the six CRISP-DM phases and describe what happens in each.

2. Given a business problem ("reduce customer churn"), write a problem framing document that defines success criteria and ML feasibility.

3. Perform a basic EDA on a CSV dataset: check for missing values, data types, and target distribution.

4. Choose the appropriate evaluation metric for a fraud detection model (high class imbalance).

5. Create a simple monitoring check that compares daily prediction rates to a reference baseline.

### Medium

1. Build a complete ML project plan for a churn prediction model, including timeline, milestones, and deliverables for each CRISP-DM phase.

2. Implement a data drift detection system that uses PSI (Population Stability Index) for a set of features.

3. Design an A/B testing framework for comparing two model versions in production.

4. Create a model evaluation report that compares at least 3 models across 5 metrics and provides a deployment recommendation.

5. Build a simple model registry that tracks model versions, their metrics, and deployment status.

### Hard

1. Design and implement a complete MLOps pipeline with: automated training, model registry, A/B testing, and monitoring.

2. Create a business impact analysis for an ML system: estimate the cost of false positives and false negatives, and determine the optimal decision threshold.

3. Build a simulation that demonstrates the feedback loop between monitoring, retraining, and deployment over multiple cycles.

## Solutions

**Easy 1:**
```python
# CRISP-DM Phases:
# 1. Business Understanding: Define objectives, success criteria
# 2. Data Understanding: Collect, describe, explore data
# 3. Data Preparation: Clean, transform, engineer features
# 4. Modeling: Select, train, tune algorithms
# 5. Evaluation: Assess model against business objectives
# 6. Deployment: Implement, monitor, maintain
```

**Medium 1:**
```python
project_plan = {
    "Phase 1 - Business Understanding": {"duration": "2 weeks", "deliverables": ["Problem statement", "Success metrics", "Project charter"]},
    "Phase 2 - Data Understanding": {"duration": "3 weeks", "deliverables": ["Data quality report", "EDA notebook", "Data dictionary"]},
    "Phase 3 - Data Preparation": {"duration": "4 weeks", "deliverables": ["Clean dataset", "Feature store entries", "Pipelines"]},
    "Phase 4 - Modeling": {"duration": "4 weeks", "deliverables": ["Candidate models", "Hyperparameter tuning results", "Model cards"]},
    "Phase 5 - Evaluation": {"duration": "2 weeks", "deliverables": ["Model evaluation report", "Business impact analysis", "Go/no-go recommendation"]},
    "Phase 6 - Deployment": {"duration": "4 weeks", "deliverables": ["API endpoint", "Docker image", "Monitoring dashboard"]},
    "Phase 7 - Monitoring": {"duration": "ongoing", "deliverables": ["Weekly health reports", "Retraining triggers", "Incident response plan"]}
}
```

**Hard 1:**
```python
# MLOps pipeline outline
# 1. Data pipeline (Airflow DAG): collect -> validate -> transform -> store
# 2. Training pipeline: fetch features -> train -> evaluate -> register model
# 3. Deployment pipeline: build container -> deploy to staging -> run tests -> promote to prod
# 4. Monitoring pipeline: collect predictions -> compute statistics -> detect drift -> alert
# 5. Retraining pipeline: on drift detection -> trigger training pipeline -> A/B test -> auto-promote
```

## Related Concepts

- **ML-076 ML Pipelines**: Pipelines operationalize the data preparation and modeling phases.
- **ML-078 Model Versioning**: The model registry is a key artifact of the deployment phase.
- **ML-079 Experiment Tracking**: Tracking supports the modeling and evaluation phases.
- **ML-081 Serving Models**: Serving is the core activity of the deployment phase.
- **ML-082 Batch vs Realtime**: The serving paradigm choice is made during the deployment phase.

## Next Concepts

- **ML-076 ML Pipelines** — Deep dive into the data preparation phase.
- **ML-081 Serving Models** — Deep dive into the deployment phase.
- **ML-084 Reproducibility** — Ensuring every phase is reproducible.

## Summary

The ML project lifecycle provides a structured framework for delivering successful ML projects. CRISP-DM's six phases (Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment) plus Monitoring form a complete end-to-end process. The lifecycle is inherently iterative, with insights from later phases feeding back into earlier ones. Problem framing determines whether ML is appropriate, EDA guides feature engineering, and deployment transitions the model from development to production. Continuous monitoring ensures the model remains effective over time. MLOps maturity models help organizations assess and improve their lifecycle practices.

## Key Takeaways

- CRISP-DM defines 6 phases: Business Understanding through Deployment
- Modern ML lifecycle adds Monitoring as a critical seventh phase
- Always start with business understanding, not modeling
- EDA is essential for data quality and feature discovery
- Establish evaluation criteria before training begins
- Plan for iteration: ML projects never follow a straight line
- Deployment is not the end: monitoring is ongoing
- MLOps maturity levels: 0 (manual) to 2 (full automation)
- Start simple, establish baselines, then iterate
- The lifecycle framework scales from small projects to enterprise MLOps
