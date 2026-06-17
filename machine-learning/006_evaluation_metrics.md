# Concept: Evaluation Metrics

## Concept ID

ML-006

## Difficulty

INTERMEDIATE

## Domain

Machine Learning

## Module

ML Fundamentals

## Learning Objectives

- Select appropriate evaluation metrics for classification and regression problems
- Understand the mathematical formulas for accuracy, precision, recall, F1-score, and specificity
- Understand MSE, RMSE, MAE, R-squared, and MAPE for regression
- Explain why accuracy is misleading for imbalanced classification
- Choose evaluation metrics aligned with business objectives

## Prerequisites

- ML-001: What is Machine Learning
- ML-003: Train/Test Split
- Basic understanding of classification and regression

## Definition

Evaluation metrics are quantitative measures used to assess the performance of machine learning models. They provide a standardized way to compare different models and determine whether a model meets the required performance threshold for deployment. Metrics are fundamentally different for classification (predicting categories) and regression (predicting continuous values).

### Classification Metrics

**Accuracy**: The proportion of correct predictions among total predictions.

$$\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} = \frac{TP + TN}{TP + TN + FP + FN}$$

Accuracy is intuitive but can be misleading for imbalanced datasets.

**Precision (Positive Predictive Value)**: The proportion of positive identifications that were actually correct.

$$\text{Precision} = \frac{TP}{TP + FP}$$

High precision means few false positives. Important when false positives are costly (e.g., spam classification — marking legitimate email as spam).

**Recall (Sensitivity, True Positive Rate)**: The proportion of actual positives that were correctly identified.

$$\text{Recall} = \frac{TP}{TP + FN}$$

High recall means few false negatives. Important when false negatives are costly (e.g., cancer detection — missing a positive case).

**F1-Score**: The harmonic mean of precision and recall, providing a single metric that balances both.

$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

F1 ranges from 0 to 1, where 1 is perfect precision and recall.

**Specificity (True Negative Rate)**: The proportion of actual negatives that were correctly identified.

$$\text{Specificity} = \frac{TN}{TN + FP}$$

### Regression Metrics

**Mean Squared Error (MSE)**: The average of squared differences between predicted and actual values.

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

MSE penalizes large errors more heavily due to squaring. It is in squared units of the target variable.

**Root Mean Squared Error (RMSE)**: The square root of MSE, in the same units as the target variable.

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Mean Absolute Error (MAE)**: The average of absolute differences between predicted and actual values.

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

MAE is less sensitive to outliers than MSE/RMSE.

**R-squared (Coefficient of Determination)**: The proportion of variance in the target variable explained by the model.

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

R² ranges from -∞ to 1, where 1 indicates perfect prediction and 0 indicates the model performs no better than predicting the mean.

**Mean Absolute Percentage Error (MAPE)**: The average of absolute percentage errors.

$$MAPE = \frac{1}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100\%$$

## Intuition

### Classification Metrics Intuition

Imagine you are building a spam filter. Your goal is to classify emails as "spam" (positive) or "not spam" (negative).

- **Accuracy** tells you: "Out of all emails, what fraction did I classify correctly?" This seems useful, but if only 1% of emails are spam, a filter that marks everything as "not spam" achieves 99% accuracy — while being completely useless.

- **Precision** answers: "When I say an email is spam, how often am I right?" High precision means you rarely bother users with spam that is not actually spam (false positives are costly in user trust).

- **Recall** answers: "Out of all actual spam emails, how many did I catch?" High recall means you rarely let spam slip through (false negatives are costly for security).

- **F1-score** combines both, useful when you want a balance.

If you are building a cancer detection system, you might prioritize recall over precision: missing a cancer case (false negative) is far worse than a false alarm (false positive) that triggers additional testing.

### Regression Metrics Intuition

Imagine you are predicting house prices.

- **MAE** tells you: "On average, my prediction is off by $X." If MAE is $20,000, your typical prediction error is $20,000.

- **RMSE** penalizes large errors more. If most predictions are off by $10,000 but some are off by $100,000, RMSE will be much higher than MAE because the large errors are squared before averaging.

- **R²** tells you: "My model explains X% of the variance in house prices." If R² = 0.85, your model explains 85% of the price variation.

- **MAPE** tells you: "On average, my prediction is off by X%." If MAPE = 12%, your typical prediction error is 12% of the actual price.

## Why This Concept Matters

Choosing the wrong evaluation metric leads to building models that optimize for the wrong objective, potentially causing real-world harm or missed business opportunities. For example:

- A fraud detection model optimized for accuracy might never detect fraud (since most transactions are legitimate).
- A regression model optimized for MAE might produce many moderate errors, while the business cares more about preventing very large errors (use RMSE).
- A model optimized for precision will flag fewer items but with higher confidence, while optimized for recall will flag more items but with more false alarms.

Understanding evaluation metrics enables:
- Alignment between model performance and business goals
- Fair comparison between candidate models
- Clear communication of model capabilities to stakeholders
- Identification of specific failure modes (too many false positives? too many false negatives?)

## Mathematical Explanation

### The Confusion Matrix Foundation

All classification metrics derive from four fundamental quantities:

- **True Positive (TP)**: Positive sample correctly predicted as positive
- **True Negative (TN)**: Negative sample correctly predicted as negative
- **False Positive (FP)**: Negative sample incorrectly predicted as positive (Type I error)
- **False Negative (FN)**: Positive sample incorrectly predicted as negative (Type II error)

From these, we derive a hierarchy of metrics:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

$$\text{Specificity} = \frac{TN}{TN + FP}$$

$$F1 = \frac{2TP}{2TP + FP + FN}$$

### Relationship Between Metrics

- Precision and recall are inversely related for a fixed model — you can typically increase one at the expense of the other by adjusting the decision threshold.
- F1 is the harmonic mean (not arithmetic mean) because precision and recall are ratios. The harmonic mean penalizes extreme imbalances: if precision = 1.0 but recall = 0.0, F1 = 0.
- Accuracy = (TP + TN) / n is sensitive to class imbalance because the majority class dominates.

### Regression Metric Properties

**MSE** = $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$
- Differentiable everywhere (useful for optimization)
- Heavily penalizes outliers
- In squared units (can be hard to interpret)

**RMSE** = $\sqrt{MSE}$
- In same units as target
- Still penalizes outliers heavily
- More interpretable than MSE

**MAE** = $\frac{1}{n}\sum |y_i - \hat{y}_i|$
- Robust to outliers
- Linear penalty for all errors
- Not differentiable at 0

**R²** = $1 - \frac{SS_{res}}{SS_{tot}}$
- Scale-invariant (always between 0 and 1 for good models)
- Can be negative (model worse than predicting mean)
- Does not indicate whether predictions are biased

## Code Examples

### Example 1: Classification Metrics with Imbalanced Data

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

# Create imbalanced dataset (95% class 0, 5% class 1)
X, y = make_classification(
    n_samples=10000, weights=[0.95, 0.05],
    random_state=42, n_features=10
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Metrics with default threshold (0.5)
print("Metrics with default threshold (0.5):")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"  F1-Score:  {f1_score(y_test, y_pred):.4f}")
# Output:
# Metrics with default threshold (0.5):
#   Accuracy:  0.9907
#   Precision: 0.8585
#   Recall:    0.6437
#   F1-Score:  0.7356

# Metrics with adjusted threshold (0.3) — favor recall
threshold = 0.3
y_pred_adj = (y_proba >= threshold).astype(int)
print(f"\nMetrics with adjusted threshold ({threshold}):")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_adj):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_adj):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_adj):.4f}")
print(f"  F1-Score:  {f1_score(y_test, y_pred_adj):.4f}")
# Output:
# Metrics with adjusted threshold (0.3):
#   Accuracy:  0.9810
#   Precision: 0.5927
#   Recall:    0.8595
#   F1-Score:  0.7014

# Full classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))
# Output:
# Classification Report:
#               precision    recall  f1-score   support
#       Class 0       0.99      1.00      0.99      2850
#       Class 1       0.86      0.64      0.74       150
#     accuracy                           0.99      3000
#    macro avg       0.93      0.82      0.87      3000
# weighted avg       0.99      0.99      0.99      3000
```

### Example 2: Regression Metrics Comparison

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge (alpha=1)': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

print("Regression Model Comparison:")
print(f"{'Model':<20} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12} {'MAPE':<12}")
print("-" * 68)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"{name:<20} {mse:<12.1f} {rmse:<12.1f} {mae:<12.1f} {r2:<12.3f} {mape:<12.3f}")
# Output:
# Regression Model Comparison:
# Model                MSE          RMSE         MAE          R²           MAPE
# --------------------------------------------------------------------
# Linear Regression    2900.2       53.9         43.4         0.452        0.394
# Ridge (alpha=1)      2900.3       53.9         43.4         0.452        0.394
# Random Forest        2925.1       54.1         45.0         0.447        0.423
```

### Example 3: Multi-Class Classification Metrics

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Per-class metrics (macro averaging)
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred
)

print("Per-Class Metrics (Iris):")
for i, target_name in enumerate(iris.target_names):
    print(f"  {target_name:10s}: Precision={precision[i]:.3f}, "
          f"Recall={recall[i]:.3f}, F1={f1[i]:.3f}, Support={support[i]}")
# Output:
# Per-Class Metrics (Iris):
#   setosa    : Precision=1.000, Recall=1.000, F1=1.000, Support=19
#   versicolor: Precision=1.000, Recall=1.000, F1=1.000, Support=13
#   virginica : Precision=1.000, Recall=1.000, F1=1.000, Support=13

# Averaging strategies
from sklearn.metrics import f1_score

print(f"\nMacro F1: {f1_score(y_test, y_pred, average='macro'):.3f}")
print(f"Micro F1: {f1_score(y_test, y_pred, average='micro'):.3f}")
print(f"Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.3f}")
# Output:
# Macro F1: 1.000
# Micro F1: 1.000
# Weighted F1: 1.000
```

### Example 4: Why Accuracy Fails for Imbalanced Data

```python
import numpy as np
from sklearn.metrics import accuracy_score

# Dummy classifier that always predicts majority class
def dummy_classifier(y_true):
    majority_class = 1 if np.sum(y_true == 1) > np.sum(y_true == 0) else 0
    return np.full_like(y_true, majority_class)

# Scenario: 99% negative, 1% positive
y_true = np.array([0] * 990 + [1] * 10)
y_pred_dummy = dummy_classifier(y_true)

# "Good" classifier that catches some positives
y_pred_good = np.array([0] * 980 + [1] * 10 + [0] * 3 + [1] * 7)

print("Dummy classifier (always predicts 0):")
print(f"  Accuracy: {accuracy_score(y_true, y_pred_dummy):.4f}")
# Output: Dummy classifier (always predicts 0):
#   Accuracy: 0.9900

print("\nGood classifier (catches 70% of positives):")
print(f"  Accuracy: {accuracy_score(y_true, y_pred_good):.4f}")
# Output:
# Good classifier (catches 70% of positives):
#   Accuracy: 0.9900

# Both have same accuracy! Accuracy does not distinguish them.
# Need precision, recall, F1 to see the difference:
from sklearn.metrics import precision_score, recall_score, f1_score

print(f"  Precision: {precision_score(y_true, y_pred_good):.4f}")
print(f"  Recall: {recall_score(y_true, y_pred_good):.4f}")
print(f"  F1: {f1_score(y_true, y_pred_good):.4f}")
# Output:
#   Precision: 0.4118
#   Recall: 0.7000
#   F1: 0.5185
```

## Common Mistakes

1. **Using accuracy for imbalanced datasets**: A 99% accurate model on 99:1 data could be completely useless (always predict majority class). Use precision, recall, F1, or AUC-ROC instead.
2. **Choosing metrics that do not align with business goals**: Optimizing for recall when false positives are costly (e.g., sending innocent people to jail) would be disastrous. Match metrics to real-world costs.
3. **Comparing models using different metrics**: One model might have better accuracy but worse F1. State clearly which metric matters for the use case.
4. **Reporting only one metric**: A single metric never tells the full story. Always report multiple metrics (at minimum: precision, recall, F1 for classification; RMSE and R² for regression).
5. **Using MAE when large errors are unacceptable**: MAE treats all errors equally. If the business cannot tolerate any large errors, use MSE/RMSE.
6. **Misinterpreting R²**: R² = 0.9 does not mean the model is "90% accurate." It means the model explains 90% of the variance. The model could still have systematic bias.
7. **Not considering cost-sensitive evaluation**: Different errors have different costs. A false negative in medical diagnosis might cost a life; a false positive costs a follow-up test. Weight metrics accordingly.

## Interview Questions

### Beginner - 5

1. **Q: What is the difference between accuracy and precision?**
   A: Accuracy measures overall correctness (correct predictions / total predictions). Precision measures how many positive predictions were actually correct (TP / (TP + FP)). A model can have high accuracy but low precision if the dataset is imbalanced.

2. **Q: What is the F1-score and when should you use it?**
   A: F1 is the harmonic mean of precision and recall. Use it when you need a single metric that balances both precision and recall, especially for imbalanced datasets.

3. **Q: What is RMSE and why is it commonly used?**
   A: RMSE is the square root of the mean squared error. It is in the same units as the target variable (interpretable) and penalizes large errors more than small ones.

4. **Q: What does R² = 0.75 mean?**
   A: The model explains 75% of the variance in the target variable. The remaining 25% is unexplained variance (noise or missing features).

5. **Q: When is MAE preferred over RMSE?**
   A: MAE is preferred when you do not want to disproportionately penalize outliers and want a metric in the same units as the target that gives equal weight to all errors.

### Intermediate - 5

1. **Q: Explain the precision-recall tradeoff and how to choose the optimal threshold.**
   A: Precision and recall are inversely related — as threshold decreases, recall increases (catch more positives) but precision decreases (more false positives). The optimal threshold depends on business costs. Use a precision-recall curve to visualize the tradeoff and choose based on the minimum acceptable precision or recall.

2. **Q: Why might a regression model have high R² but high MAE?**
   A: R² measures explained variance, not absolute error magnitude. A model could explain 90% of variance (high R²) but still have large absolute errors if the target has high variance. For example, predicting house prices with range $100K-$1M: R² = 0.9 is possible with MAE of $50K.

3. **Q: What is the difference between micro and macro averaging in multi-class metrics?**
   A: Macro averaging computes the metric independently for each class and averages them (all classes equally weighted). Micro averaging aggregates contributions of all classes to compute the overall metric (each sample equally weighted). Macro is sensitive to minority class performance; micro is dominated by majority class.

4. **Q: How do you handle multiple metrics with conflicting results (e.g., Model A has better precision, Model B has better recall)?**
   A: (1) Define a primary metric based on business requirements. (2) Use a composite metric like F1 or F-beta (where beta weights recall). (3) Use cost-sensitive evaluation where each error type has an associated cost. (4) Present both metrics and let stakeholders decide based on acceptable tradeoffs.

5. **Q: What is MAPE and what are its limitations?**
   A: MAPE is the mean absolute percentage error. Limitations: (1) Undefined when actual values are zero. (2) Asymmetric — penalizes over-prediction more than under-prediction. (3) Sensitive to scale — a $10 error on a $10 item (100%) vs a $10 error on a $1000 item (1%).

### Advanced - 3

1. **Q: Derive the relationship between precision and recall for a fixed classifier and prove that F1 is their harmonic mean.**
   A: For a fixed classifier, as the decision threshold varies, precision and recall trace a curve. They are inversely related because: increasing recall requires classifying more samples as positive, which adds more false positives (reducing precision). The F1-score is the harmonic mean: $F1 = \frac{2}{\frac{1}{P} + \frac{1}{R}} = \frac{2PR}{P+R}$. The harmonic mean is appropriate because precision and recall are ratios — the harmonic mean penalizes imbalance more than the arithmetic mean.

2. **Q: Explain the concept of proper scoring rules and why accuracy is not a proper scoring rule.**
   A: A scoring rule is proper if it is maximized by predicting the true probabilities. Accuracy is not a proper scoring rule because it depends on a hard decision threshold and does not consider prediction confidence. Proper scoring rules (log loss, Brier score) evaluate probabilistic predictions and cannot be gamed. Log loss = $-\frac{1}{n}\sum[y_i\log(p_i) + (1-y_i)\log(1-p_i)]$ is strictly proper — it is minimized only when predicted probabilities match true probabilities.

3. **Q: Prove that for regression, MSE can be decomposed into bias² + variance + irreducible error, and explain how different evaluation metrics capture different aspects of predictive performance.**
   A: $\mathbb{E}[(y - \hat{f})^2] = (\mathbb{E}[\hat{f}] - f)^2 + \mathbb{E}[(\hat{f} - \mathbb{E}[\hat{f}])^2] + \sigma^2_\epsilon$. MSE captures all three components. MAE does not have this clean decomposition due to non-differentiability. R² normalizes MSE by the variance of y: $R^2 = 1 - \frac{MSE}{Var(y)}$, so it measures relative improvement over the mean predictor. Each metric captures different aspects: MSE penalizes variance most, MAE captures absolute accuracy, R² captures explained variance ratio.

## Practice Problems

### Easy - 5

1. **Problem**: A classifier predicts 100 samples: 40 TP, 30 TN, 10 FP, 20 FN. Calculate accuracy, precision, recall, F1, and specificity.

2. **Problem**: You have a regression model with the following predictions: actual = [100, 200, 300, 400, 500], predicted = [110, 190, 310, 390, 510]. Calculate MSE, MAE, and RMSE.

3. **Problem**: For a spam detection system, which error is worse: FP (marking legitimate email as spam) or FN (letting spam through)? Which metric would you optimize?

4. **Problem**: A model achieves RMSE = 15 and MAE = 10. What does the difference between these values tell you about the error distribution?

5. **Problem**: What is the F1-score if precision = 0.8 and recall = 0.5?

### Medium - 5

1. **Problem**: You have a fraud detection model. The cost of a false negative (missed fraud) is $1000. The cost of a false positive (false alarm) is $50. Derive a cost-sensitive metric and explain how to choose the optimal threshold.

2. **Problem**: Compare and contrast macro F1, micro F1, and weighted F1 for a 3-class problem with classes A (1000 samples), B (100 samples), C (10 samples).

3. **Problem**: A model predicts house prices with R² = 0.92 on training and R² = 0.88 on test. MAPE is 8%. What else would you check before deploying? List at least three additional diagnostics.

4. **Problem**: Design an evaluation strategy for a medical diagnostic test that screens for a rare disease (prevalence 0.1%). Consider the costs of false positives (unnecessary follow-up procedures) and false negatives (missed diagnosis).

5. **Problem**: Explain why the F1-score might not be appropriate when precision and recall have different business importance, and propose an alternative.

### Hard - 3

1. **Problem**: Derive the F-beta score and explain how it generalizes the F1-score. For a cancer detection system where missing a case is 5x worse than a false alarm, what beta should you use?

2. **Problem**: Prove that for any classifier, the arithmetic mean of precision and recall is always ≥ the F1-score, with equality only when precision = recall.

3. **Problem**: Design a comprehensive evaluation framework for a self-driving car's pedestrian detection system. Discuss metrics, cost functions, threshold selection, and validation strategy.

## Solutions

### Easy Solutions

1. Accuracy = (40+30)/(40+30+10+20) = 70/100 = 0.70. Precision = 40/(40+10) = 0.80. Recall = 40/(40+20) = 0.667. F1 = 2(0.80)(0.667)/(0.80+0.667) = 0.727. Specificity = 30/(30+10) = 0.75.
2. MSE = [(100-110)² + (200-190)² + (300-310)² + (400-390)² + (500-510)²] / 5 = [100 + 100 + 100 + 100 + 100] / 5 = 100. RMSE = 10. MAE = [|10|+|10|+|10|+|10|+|10|] / 5 = 10.
3. Both are bad, but usually FN is worse (spam reaching inbox is security risk). Optimize for recall (catch rate) with a minimum precision threshold.
4. RMSE > MAE indicates there are some large errors (outliers) because RMSE penalizes large errors more. If errors were uniform, RMSE ≈ MAE.
5. F1 = 2(0.8)(0.5)/(0.8+0.5) = 0.8/1.3 = 0.615.

### Medium Solutions

1. Cost per sample = FN_cost × FN_rate + FP_cost × FP_rate. Total cost = sum over all samples. The optimal threshold minimizes total cost. For this case, FN cost is 20x FP cost, so we want high recall even at the expense of precision. Threshold should be set lower than 0.5.
2. Macro F1: average of per-class F1 (each class equal weight). With small support for class C, poor performance on C drags macro F1 down severely. Micro F1: aggregates TP/FP/FN across all classes (equal sample weight). Dominated by class A (1000 samples). Weighted F1: average of per-class F1 weighted by support. Balances between macro and micro.
3. (1) Check residual plots for heteroscedasticity or non-linearity. (2) Check for systematic bias (mean residual near zero?). (3) Check performance on specific segments (expensive vs cheap houses). (4) Verify no data leakage. (5) Test on out-of-time data.
4. With 0.1% prevalence, accuracy is useless. Use recall as primary metric (cannot miss cases). Set precision threshold where cost of follow-ups is acceptable. Use F-beta with beta > 1 to weight recall higher. Confusion matrix is essential. Also compute number-needed-to-screen (how many false alarms per detected case).
5. F-beta score generalizes F1: $F_\beta = (1+\beta^2) \frac{PR}{\beta^2 P + R}$. Beta > 1 weights recall higher; beta < 1 weights precision higher. The F1 score assumes equal importance of precision and recall. If the business prioritizes recall 5:1, use F-5 score.

### Hard Solutions

1. $F_\beta = (1+\beta^2) \frac{PR}{\beta^2 P + R}$ where $\beta$ indicates how many times recall is more important than precision. For cancer detection, recall is 5x more important than precision, so $\beta = 5$. $F_5 = 26 \frac{PR}{25P + R}$. With beta=5, achieving high F-beta requires high recall even if precision is lower.
2. By the AM-GM inequality: $\frac{P+R}{2} \geq \sqrt{PR}$, so $\frac{P+R}{2} \geq \frac{2PR}{P+R}$ (since $\sqrt{PR} \geq \frac{2PR}{P+R}$ by rearrangement). Therefore arithmetic mean ≥ harmonic mean = F1. Equality holds only when P = R, where AM = P = R = F1.
3. Framework: (1) Primary metric: Recall (missing a pedestrian is catastrophic), target > 99.9%. Secondary: Precision (false alarms cause unnecessary braking), target > 99%. (2) Cost function: Missed pedestrian = $10^6$ (life value), false alarm = $10^2$ (traffic disruption). (3) Threshold selection: choose threshold where $10^6 \times FN + 10^2 \times FP$ is minimized. (4) Validation: temporal CV (train on old data, test on new), geographic CV (different cities), adverse condition testing (night, rain, fog). (5) Additional metrics: detection distance, response time, IoU (bounding box overlap). (6) Safety: evaluate on corner cases (children, bicycles, occluded pedestrians).

## Related Concepts

- **ML-007: Confusion Matrix** — The foundation for all classification metrics
- **ML-008: ROC and AUC** — Threshold-independent evaluation
- **ML-004: Overfitting and Underfitting** — Overfit models show poor evaluation metrics
- **Loss Functions**: The training objectives that metrics evaluate
- **Cost-Sensitive Learning**: Incorporating different error costs into model training

## Next Concepts

- **ML-007: Confusion Matrix** — Detailed breakdown of classification performance
- **ML-008: ROC and AUC** — Threshold-independent evaluation metrics

## Summary

Evaluation metrics quantify model performance and must be chosen carefully based on the problem type and business context. For classification, key metrics include accuracy, precision, recall, F1-score, and specificity. For regression, key metrics include MSE, RMSE, MAE, R-squared, and MAPE. No single metric tells the complete story — always report multiple metrics. Accuracy is misleading for imbalanced datasets; use precision, recall, and F1 instead. The choice between RMSE and MAE depends on whether large errors must be penalized more heavily. Evaluation metrics should align with the real-world costs of different types of errors.

## Key Takeaways

1. Classification metrics (accuracy, precision, recall, F1) capture different aspects of performance.
2. Accuracy is misleading for imbalanced datasets — use precision, recall, and F1.
3. Regression metrics (MSE, RMSE, MAE, R²) have different properties and sensitivities.
4. R² measures explained variance, not "accuracy."
5. Always report multiple metrics for a complete picture.
6. Choose metrics aligned with business goals and error costs.
7. Threshold adjustment can trade off precision and recall for a given model.
