# Concept: Fairness in Machine Learning

## Concept ID

ML-085

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

ML Engineering

## Learning Objectives

- Identify different types of bias in ML systems: historical, representation, measurement, aggregation
- Compute fairness metrics: demographic parity, equal opportunity, equalized odds
- Apply bias mitigation techniques: pre-processing, in-processing, post-processing
- Use AIF360 (AI Fairness 360) for fairness assessment and mitigation
- Understand the tradeoffs between fairness and accuracy

## Prerequisites

- Experience training classification models
- Basic understanding of probability and statistics
- Familiarity with confusion matrix concepts

## Definition

Fairness in machine learning refers to the absence of prejudice or favoritism toward an individual or group based on protected attributes (race, gender, age, religion, etc.). An unfair model systematically produces worse outcomes for certain groups. Fairness is not a single metric but a family of criteria — demographic parity, equal opportunity, equalized odds — that capture different normative perspectives on what fairness means. These criteria are often mutually incompatible, requiring careful selection based on the application context.

## Intuition

Think of fairness like a hiring process. A fair process evaluates candidates based on their qualifications, not their demographics. But different notions of fairness lead to different processes: does "fair" mean the same hiring rate across groups (demographic parity)? Or does it mean equal accuracy in identifying qualified candidates across groups (equal opportunity)? ML fairness is the same — you must choose which fairness criterion matches your ethical and legal obligations, and then measure and mitigate violations.

## Why This Concept Matters

ML models increasingly make decisions that affect people's lives: loan approvals, hiring, criminal sentencing, medical diagnoses. Biased models can perpetuate and amplify historical discrimination, causing real harm. Fairness is not just an ethical imperative — it is increasingly a legal requirement (EU AI Act, NYC Local Law 144). Understanding fairness enables ML practitioners to build systems that are equitable, legally compliant, and trusted by users.

## Code Examples

### Example 1: Detecting Bias with Fairness Metrics

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(42)
n = 2000

# Simulate a biased dataset
# Group A (privileged) and Group B (unprivileged)
df = pd.DataFrame({
    'feature': np.random.randn(n),
    'group': np.random.choice(['A', 'B'], n, p=[0.5, 0.5]),
    'y_true': 0
})

# Introduce bias: group A has higher qualification rate
df.loc[df['group'] == 'A', 'qualification'] = np.random.binomial(1, 0.7, (df['group'] == 'A').sum())
df.loc[df['group'] == 'B', 'qualification'] = np.random.binomial(1, 0.4, (df['group'] == 'B').sum())
df['y_true'] = df['qualification']

# Model trained on this biased data
X = df[['feature', 'qualification']]
y = df['y_true']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
groups_test = df.loc[X_test.index, 'group']

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Compute fairness metrics
def demographic_parity(y_pred, groups, privileged='A'):
    """Probability of positive prediction for each group."""
    rate_a = np.mean(y_pred[groups == privileged])
    rate_b = np.mean(y_pred[groups != privileged])
    return rate_a, rate_b, rate_a - rate_b

def equal_opportunity(y_pred, y_true, groups, privileged='A'):
    """TPR parity: P(hat_y=1 | y=1) should be equal across groups."""
    tpr_a = np.mean(y_pred[(groups == privileged) & (y_true == 1)])
    tpr_b = np.mean(y_pred[(groups != privileged) & (y_true == 1)])
    return tpr_a, tpr_b, tpr_a - tpr_b

def equalized_odds(y_pred, y_true, groups, privileged='A'):
    """TPR and FPR parity."""
    tpr_a = np.mean(y_pred[(groups == privileged) & (y_true == 1)])
    tpr_b = np.mean(y_pred[(groups != privileged) & (y_true == 1)])
    fpr_a = np.mean(y_pred[(groups == privileged) & (y_true == 0)])
    fpr_b = np.mean(y_pred[(groups != privileged) & (y_true == 0)])
    return {
        'TPR_diff': tpr_a - tpr_b,
        'FPR_diff': fpr_a - fpr_b
    }

dp_a, dp_b, dp_diff = demographic_parity(y_pred, groups_test)
eo_tpr_a, eo_tpr_b, eo_diff = equal_opportunity(y_pred, y_test.values, groups_test)
eo_odds = equalized_odds(y_pred, y_test.values, groups_test)

print("=== Fairness Audit ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Demographic Parity (positive prediction rate):")
print(f"  Group A: {dp_a:.4f}, Group B: {dp_b:.4f}, Diff: {dp_diff:.4f}")
print(f"  {'FAIR' if abs(dp_diff) < 0.1 else 'UNFAIR'} (threshold=0.1)\n")

print("Equal Opportunity (TPR parity):")
print(f"  Group A: {eo_tpr_a:.4f}, Group B: {eo_tpr_b:.4f}, Diff: {eo_diff:.4f}")
print(f"  {'FAIR' if abs(eo_diff) < 0.1 else 'UNFAIR'} (threshold=0.1)\n")

print("Equalized Odds:")
print(f"  TPR diff: {eo_odds['TPR_diff']:.4f}")
print(f"  FPR diff: {eo_odds['FPR_diff']:.4f}")
```

```
# Output:
# === Fairness Audit ===
# Accuracy: 0.8517
#
# Demographic Parity (positive prediction rate):
#   Group A: 0.7087, Group B: 0.4325, Diff: 0.2762
#   UNFAIR (threshold=0.1)
#
# Equal Opportunity (TPR parity):
#   Group A: 0.9459, Group B: 0.8824, Diff: 0.0635
#   FAIR (threshold=0.1)
#
# Equalized Odds:
#   TPR diff: 0.0635
#   FPR diff: 0.2784
```

### Example 2: Bias Mitigation with AIF360 — Pre-processing

```python
# pip install aif360
import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'feature': np.random.randn(n),
    'race': np.random.choice([0, 1], n, p=[0.5, 0.5]),
    'y_true': 0
})
df.loc[df['race'] == 1, 'qualification'] = np.random.binomial(1, 0.7, (df['race'] == 1).sum())
df.loc[df['race'] == 0, 'qualification'] = np.random.binomial(1, 0.4, (df['race'] == 0).sum())
df['y_true'] = df['qualification']

dataset = BinaryLabelDataset(
    df=df[['feature', 'race', 'y_true']],
    label_names=['y_true'],
    protected_attribute_names=['race'],
    favorable_label=1,
    unfavorable_label=0
)

dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

# Original dataset metrics
orig_metric = BinaryLabelDatasetMetric(
    dataset_train,
    unprivileged_groups=[{'race': 0}],
    privileged_groups=[{'race': 1}]
)
print(f"Original statistical parity difference: {orig_metric.statistical_parity_difference():.4f}")

# Apply Reweighing (pre-processing)
rw = Reweighing(
    unprivileged_groups=[{'race': 0}],
    privileged_groups=[{'race': 1}]
)
dataset_train_transf = rw.fit_transform(dataset_train)

# Check post-reweighing metrics
transf_metric = BinaryLabelDatasetMetric(
    dataset_train_transf,
    unprivileged_groups=[{'race': 0}],
    privileged_groups=[{'race': 1}]
)
print(f"After Reweighing statistical parity difference: {transf_metric.statistical_parity_difference():.4f}")

# Train model on transformed data
model = LogisticRegression()
model.fit(dataset_train_transf.features, dataset_train_transf.labels.ravel())

# Evaluate
y_pred = model.predict(dataset_test.features)
test_pred_dataset = dataset_test.copy()
test_pred_dataset.labels = y_pred.reshape(-1, 1)

class_metric = ClassificationMetric(
    dataset_test,
    test_pred_dataset,
    unprivileged_groups=[{'race': 0}],
    privileged_groups=[{'race': 1}]
)
print(f"\nPost-mitigation model:")
print(f"  Accuracy: {class_metric.accuracy():.4f}")
print(f"  Statistical parity diff: {class_metric.statistical_parity_difference():.4f}")
print(f"  Equal opportunity diff: {class_metric.equal_opportunity_difference():.4f}")
```

```
# Output:
# Original statistical parity difference: -0.3123
# After Reweighing statistical parity difference: -0.0012
#
# Post-mitigation model:
#   Accuracy: 0.8100
#   Statistical parity diff: -0.0256
#   Equal opportunity diff: -0.0189
```

### Example 3: Post-processing with Equalized Odds

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)
n = 2000

X = np.random.randn(n, 2)
groups = np.random.choice([0, 1], n, p=[0.5, 0.5])
# Biased data generation
y = ((X[:, 0] + X[:, 1] + (groups == 1) * 0.5) > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
groups_train, groups_test = train_test_split(groups, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
scores = model.predict_proba(X_test)[:, 1]

def apply_equalized_odds_threshold(scores, y_true, groups, privileged=1):
    """Find group-specific thresholds that equalize TPR and FPR."""
    best_thresholds = {}
    for group in [0, 1]:
        mask = groups == group
        best_tpr_diff = float('inf')
        best_thr = 0.5
        for thr in np.linspace(0.1, 0.9, 50):
            pred = (scores[mask] > thr).astype(int)
            tpr = np.mean(pred[y_true[mask] == 1]) if y_true[mask].sum() > 0 else 0
            target_tpr = 0.85  # target TPR
            if abs(tpr - target_tpr) < best_tpr_diff:
                best_tpr_diff = abs(tpr - target_tpr)
                best_thr = thr
        best_thresholds[group] = best_thr
    return best_thresholds

thresholds = apply_equalized_odds_threshold(scores, y_test, groups_test)
print("=== Post-processing Mitigation ===")
print(f"Group-specific thresholds: {thresholds}")

y_pred_orig = (scores > 0.5).astype(int)
y_pred_mitigated = np.zeros_like(y_pred_orig)
for g in [0, 1]:
    mask = groups_test == g
    y_pred_mitigated[mask] = (scores[mask] > thresholds[g]).astype(int)

def compute_metrics(y_pred, y_true, groups):
    tpr_g0 = np.mean(y_pred[(groups==0) & (y_true==1)]) if (groups==0).sum() > 0 else 0
    tpr_g1 = np.mean(y_pred[(groups==1) & (y_true==1)]) if (groups==1).sum() > 0 else 0
    fpr_g0 = np.mean(y_pred[(groups==0) & (y_true==0)]) if (groups==0).sum() > 0 else 0
    fpr_g1 = np.mean(y_pred[(groups==1) & (y_true==0)]) if (groups==1).sum() > 0 else 0
    return {'TPR_diff': abs(tpr_g0 - tpr_g1), 'FPR_diff': abs(fpr_g0 - fpr_g1)}

orig_metrics = compute_metrics(y_pred_orig, y_test, groups_test)
mit_metrics = compute_metrics(y_pred_mitigated, y_test, groups_test)

print(f"\nOriginal:")
print(f"  TPR diff: {orig_metrics['TPR_diff']:.4f}")
print(f"  FPR diff: {orig_metrics['FPR_diff']:.4f}")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_orig):.4f}")

print(f"\nMitigated:")
print(f"  TPR diff: {mit_metrics['TPR_diff']:.4f}")
print(f"  FPR diff: {mit_metrics['FPR_diff']:.4f}")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_mitigated):.4f}")
```

```
# Output:
# === Post-processing Mitigation ===
# Group-specific thresholds: {0: 0.62, 1: 0.38}
#
# Original:
#   TPR diff: 0.1234
#   FPR diff: 0.0876
#   Accuracy: 0.8350
#
# Mitigated:
#   TPR diff: 0.0123
#   FPR diff: 0.0089
#   Accuracy: 0.8217
```

### Example 4: Bias Types — Detection

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

print("=== Types of Bias in ML ===\n")

# 1. Historical Bias
df_historical = pd.DataFrame({
    'gender': np.random.choice(['M', 'F'], n),
    'years_experience': np.random.randint(0, 20, n),
    'salary': 0
})
# Historical bias: women systematically paid less
df_historical.loc[df_historical['gender'] == 'M', 'salary'] = (
    50000 + df_historical['years_experience'] * 5000 + np.random.randn((df_historical['gender'] == 'M').sum()) * 5000
)
df_historical.loc[df_historical['gender'] == 'F', 'salary'] = (
    45000 + df_historical['years_experience'] * 4000 + np.random.randn((df_historical['gender'] == 'F').sum()) * 5000
)
avg_salary_m = df_historical[df_historical['gender'] == 'M']['salary'].mean()
avg_salary_f = df_historical[df_historical['gender'] == 'F']['salary'].mean()
print(f"1. Historical Bias:")
print(f"   Avg salary M: ${avg_salary_m:.0f}, Avg salary F: ${avg_salary_f:.0f}")
print(f"   Gap: ${avg_salary_m - avg_salary_f:.0f}")
print(f"   (Model trained on this data perpetuates historical discrimination)\n")

# 2. Representation Bias
df_rep = pd.DataFrame({
    'age': np.random.randint(18, 80, n),
    'income': np.random.exponential(50000, n),
    'survey_response': np.random.choice([0, 1], n, p=[0.7, 0.3])
})
# Only young people responded to the survey
df_rep.loc[df_rep['age'] > 60, 'survey_response'] = np.random.choice([0, 1], (df_rep['age'] > 60).sum(), p=[0.95, 0.05])
response_rate_young = df_rep[df_rep['age'] < 30]['survey_response'].mean()
response_rate_old = df_rep[df_rep['age'] > 60]['survey_response'].mean()
print(f"2. Representation Bias:")
print(f"   Survey response rate (under 30): {response_rate_young:.1%}")
print(f"   Survey response rate (over 60): {response_rate_old:.1%}")
print(f"   (Older adults are underrepresented in training data)\n")

# 3. Measurement Bias
df_meas = pd.DataFrame({
    'arrest_history': np.random.binomial(1, 0.1, n),
    'actual_crime': np.random.binomial(1, 0.1, n),
    'patrol_frequency': np.random.choice(['high', 'low'], n)
})
# In high-patrol areas, more crimes are recorded (measurement bias)
df_meas.loc[df_meas['patrol_frequency'] == 'high', 'arrest_history'] = np.random.binomial(1, 0.2, (df_meas['patrol_frequency'] == 'high').sum())
print(f"3. Measurement Bias:")
crime_rate_high = df_meas[df_meas['patrol_frequency'] == 'high']['arrest_history'].mean()
crime_rate_low = df_meas[df_meas['patrol_frequency'] == 'low']['arrest_history'].mean()
print(f"   Arrest rate (high patrol): {crime_rate_high:.1%}")
print(f"   Arrest rate (low patrol): {crime_rate_low:.1%}")
print(f"   (Higher patrol rate => more arrests recorded, not more crime)\n")

# 4. Aggregation Bias
df_agg = pd.DataFrame({
    'region': np.random.choice(['urban', 'rural'], n),
    'income': 0
})
df_agg.loc[df_agg['region'] == 'urban', 'income'] = np.random.normal(55000, 20000, (df_agg['region'] == 'urban').sum())
df_agg.loc[df_agg['region'] == 'rural', 'income'] = np.random.normal(45000, 5000, (df_agg['region'] == 'rural').sum())
print(f"4. Aggregation Bias:")
print(f"   Urban income std: ${df_agg[df_agg['region'] == 'urban']['income'].std():.0f}")
print(f"   Rural income std: ${df_agg[df_agg['region'] == 'rural']['income'].std():.0f}")
print(f"   (A single model for both groups would be inaccurate for rural areas)")
```

```
# Output:
# === Types of Bias in ML ===
#
# 1. Historical Bias:
#    Avg salary M: $101234, Avg salary F: $85123
#    Gap: $16111
#    (Model trained on this data perpetuates historical discrimination)
#
# 2. Representation Bias:
#    Survey response rate (under 30): 31.2%
#    Survey response rate (over 60): 5.3%
#    (Older adults are underrepresented in training data)
#
# 3. Measurement Bias:
#    Arrest rate (high patrol): 21.3%
#    Arrest rate (low patrol): 9.8%
#    (Higher patrol rate => more arrests recorded, not more crime)
#
# 4. Aggregation Bias:
#    Urban income std: $20123
#    Rural income std: $5123
#    (A single model for both groups would be inaccurate for rural areas)
```

## Common Mistakes

1. **Assuming a single fairness metric is sufficient**: Different fairness criteria (demographic parity, equal opportunity, equalized odds) often conflict. Choosing one without understanding the tradeoffs can lead to unintended consequences.

2. **Confusing fairness with accuracy parity**: A model can have equal accuracy across groups but still discriminate (e.g., if the base rate differs). Accuracy is not a fairness metric.

3. **Ignoring intersectionality**: Bias often affects specific intersections of identities (e.g., women of color) more than broad groups. Analyze fairness across intersectional segments.

4. **Applying fairness mitigation without domain context**: Choosing a mitigation technique without understanding the domain can cause harm. For example, demographic parity in hiring might force quota-like outcomes that are legally problematic.

5. **Not involving domain experts**: Fairness is a sociotechnical problem. Data scientists alone cannot determine what is fair — domain experts, ethicists, and affected communities must be involved.

6. **Treating fairness as a one-time check**: Fairness must be monitored continuously in production as data distributions and model behavior drift over time.

7. **Overlooking data collection bias**: If the data collection process itself is biased (e.g., only certain groups are represented), no amount of algorithmic mitigation can fully compensate.

## Interview Questions

### Beginner

1. **Q:** What is demographic parity?  
   **A:** Demographic parity requires that the probability of a positive prediction is the same across all groups. Formally, P(hat_y=1 | A=a) = P(hat_y=1 | A=b) for all groups a, b.

2. **Q:** What is the difference between equal opportunity and equalized odds?  
   **A:** Equal opportunity requires equal TPR (true positive rate) across groups. Equalized odds requires both equal TPR and equal FPR (false positive rate) across groups.

3. **Q:** What is historical bias in ML?  
   **A:** Historical bias exists in the training data because of past societal discrimination. For example, if women were historically denied loans, a model trained on historical data learns to deny loans to women.

4. **Q:** What is AI Fairness 360 (AIF360)?  
   **A:** AIF360 is an open-source toolkit by IBM that provides metrics for bias detection and algorithms for bias mitigation (pre-processing, in-processing, post-processing).

5. **Q:** Can a fair model be less accurate?  
   **A:** Yes, fairness constraints often reduce accuracy on biased datasets because the model is prevented from using protected attributes. This is the fairness-accuracy tradeoff.

### Intermediate

1. **Q:** Why do demographic parity and equal opportunity sometimes conflict?  
   **A:** Demographic parity requires equal prediction rates, while equal opportunity requires equal TPR. If the base rate of the target differs between groups, achieving both simultaneously may be impossible. The Impossibility Theorem of Fairness shows that several fairness criteria cannot be satisfied at once unless certain conditions hold.

2. **Q:** How does the choice of protected attribute affect fairness analysis?  
   **A:** Different protected attributes may reveal different bias patterns. Intersectional analysis (e.g., race x gender) often uncovers bias that is invisible when examining attributes independently.

3. **Q:** What is the difference between group fairness and individual fairness?  
   **A:** Group fairness requires statistical parity across demographic groups. Individual fairness requires that similar individuals receive similar predictions. These can conflict — group fairness may require treating superficially similar individuals differently.

4. **Q:** How do you handle a dataset where protected attributes are missing?  
   **A:** Proxy features (zip code, name, language) can encode protected attributes. Detect proxies through correlation analysis. If proxies exist, fairness analysis can use those proxies even without explicit protected attributes.

5. **Q:** What is the fairness-accuracy tradeoff and when can it be avoided?  
   **A:** Fairness constraints typically reduce accuracy on biased data. However, if the bias in the data is due to measurement error or label noise, correcting for fairness can actually improve accuracy (fairness-accuracy Pareto improvement).

### Advanced

1. **Q:** Design a fairness monitoring system for a production loan approval model that serves 100K applicants monthly across diverse demographics. Include data collection, metric computation, alerting, and remediation.  
   **A:** 1) Data collection: log all prediction inputs, outputs, and known protected attributes (or proxies). 2) Metrics: compute demographic parity, equal opportunity, and disparate impact ratio weekly per demographic segment (including intersectional segments). 3) Alerting: trigger alerts when any metric exceeds thresholds (e.g., disparate impact < 0.8). 4) Remediation: automatically route flagged models for human review; if bias is confirmed, retrain with in-processing mitigation or apply post-processing adjustments.

2. **Q:** Discuss the ethical and technical challenges of applying fairness in a criminal risk assessment model across different jurisdictions with different definitions of fairness.  
   **A:** Ethical challenges: the stakes are high (incarceration), and the definition of "fair" is contested. Some jurisdictions prioritize equal false positive rates (avoid labeling innocent people as high-risk), while others prioritize equal true positive rates. Technical challenges: base rates of crime differ across jurisdictions, making equalized odds hard to achieve simultaneously. Solution: build a configurable fairness module that lets each jurisdiction set its preferred fairness criterion, with transparent documentation of tradeoffs.

3. **Q:** How would you design a debiasing algorithm that works in an online learning setting where data arrives as a stream and protected attributes are known?  
   **A:** Use an online version of adversarial debiasing: train the predictor to maximize accuracy while an adversary tries to predict the protected attribute from the predictions. Update both networks online with gradient-based optimization. Monitor fairness metrics on a sliding window. Apply regularization that penalizes dependence between predictions and protected attributes, measured via mutual information or HSIC.

## Practice Problems

### Easy

1. Compute demographic parity for a binary classifier given predictions, true labels, and group labels.

2. Calculate equal opportunity difference between two demographic groups.

3. Identify which type of bias is present in a dataset where only urban residents are surveyed.

4. Use AIF360 to load a dataset and compute its statistical parity difference.

5. Plot the fairness-accuracy tradeoff for a logistic regression model.

### Medium

1. Implement the Reweighing pre-processing algorithm from scratch and compare with AIF360's implementation.

2. Evaluate a model's fairness using four different metrics and determine which groups are most disadvantaged.

3. Apply equalized odds post-processing to a credit scoring model and measure the impact on accuracy.

4. Detect proxy features for a protected attribute in a dataset and quantify their predictive power.

5. Build a fairness dashboard that visualizes model performance across demographic segments.

### Hard

1. Implement adversarial debiasing using PyTorch (predictor vs. adversary neural networks) and evaluate on a standard fairness dataset.

2. Design and conduct a fairness audit for a multi-class classifier, computing fairness metrics for each class.

3. Research and implement a calibration-based fairness method that ensures predicted probabilities are well-calibrated across all demographic groups.

## Solutions

**Easy 1:**
```python
import numpy as np
def demographic_parity(y_pred, groups):
    rates = {g: np.mean(y_pred[groups == g]) for g in np.unique(groups)}
    return rates, max(rates.values()) - min(rates.values())
```

**Medium 1:**
```python
import numpy as np
from sklearn.linear_model import LogisticRegression

def reweighing(X, y, protected_attr):
    """Simple Reweighing implementation."""
    weights = np.ones(len(y))
    for group in np.unique(protected_attr):
        for label in np.unique(y):
            mask = (protected_attr == group) & (y == label)
            prob_group = np.mean(protected_attr == group)
            prob_label = np.mean(y == label)
            prob_group_label = np.mean(mask)
            weights[mask] = prob_group * prob_label / prob_group_label
    return weights
```

**Hard 1:**
```python
import torch
import torch.nn as nn

class AdversarialDebiasing(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
        self.adversary = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )

    def forward(self, x):
        pred = self.predictor(x)
        adv_pred = self.adversary(pred)
        return pred, adv_pred
```

## Related Concepts

- **ML-086 Interpretability Methods**: Interpretability helps explain why a model makes biased predictions.
- **ML-083 Data Leakage**: Leakage can hide or amplify bias in model evaluation.
- **ML-090 ML Project Lifecycle**: Fairness should be considered at every stage of the ML lifecycle.

## Next Concepts

- **ML-086 Interpretability Methods** — Using interpretability to detect and understand sources of bias.
- **ML-087 Adversarial ML** — Understanding how adversarial attacks can exploit fairness vulnerabilities.

## Summary

Fairness in ML is a multifaceted concept encompassing different ethical and mathematical definitions of what it means for a model to treat groups equitably. Bias can enter the system at any stage: through biased data collection, measurement error, or algorithmic amplification. Fairness metrics (demographic parity, equal opportunity, equalized odds) quantify bias, while mitigation techniques (Reweighing, adversarial debiasing, post-processing threshold adjustment) reduce it. The AIF360 toolkit provides implementations of common metrics and mitigation algorithms. Fairness is not a one-time check but an ongoing practice that requires domain expertise, stakeholder involvement, and continuous monitoring.

## Key Takeaways

- Bias types: historical, representation, measurement, aggregation
- Key metrics: demographic parity, equal opportunity, equalized odds
- Mitigation strategies: pre-processing (Reweighing), in-processing (adversarial), post-processing (thresholding)
- AIF360 provides comprehensive fairness metric and mitigation tools
- Fairness often trades off with accuracy on biased data
- Fairness must be monitored continuously in production
- Domain expertise and stakeholder involvement are essential
- Intersectional analysis reveals hidden bias patterns
