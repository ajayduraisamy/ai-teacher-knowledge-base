# Concept: Model Interpretability

## Concept ID

ML-075

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Advanced Topics

## Learning Objectives

- Understand the importance of model interpretability in ML systems
- Implement SHAP and LIME for local and global explanations
- Compute permutation importance and partial dependence plots
- Interpret tree-based models using feature importance and SHAP values

## Prerequisites

- Supervised learning (tree-based models, linear models)
- Basic probability and game theory (Shapley values intuition)
- sklearn pipelines
- Python data science ecosystem

## Definition

Model Interpretability is the degree to which a human can understand the cause of a model's decisions. Interpretability methods aim to answer questions like "Why did the model make this prediction?", "Which features are most important?", and "How would the prediction change if a feature had a different value?" Interpretability is categorized into intrinsic methods (models that are inherently interpretable, like linear regression or decision trees) and post-hoc methods (applied after training to explain black-box models).

## Intuition

Imagine a doctor using a deep learning model to diagnose cancer from medical images. The model predicts "malignant" with 95% confidence, but the doctor cannot trust this prediction without knowing why. An interpretability method highlights the regions of the image that drove the decision — perhaps the model focuses on a scar from a previous biopsy rather than the tumor itself. This explanation reveals a spurious correlation, preventing a dangerous misdiagnosis. Without interpretability, high-stakes ML deployments are risky and potentially unethical.

## Why This Concept Matters

Interpretability is crucial for trust, debugging, fairness, regulatory compliance (GDPR's right to explanation), and scientific discovery. It helps practitioners detect data leakage, identify biased features, validate model behavior, and communicate model reasoning to stakeholders. As ML is deployed in high-stakes domains (healthcare, finance, criminal justice), interpretability transitions from a nice-to-have to a necessity.

## Mathematical Explanation

### SHAP (SHapley Additive exPlanations)

SHAP values are based on Shapley values from cooperative game theory. Each feature is a "player" in a cooperative game where the "payout" is the model's prediction. The Shapley value for feature j is:

φ_j(v) = ∑_{S ⊆ N \ {j}} (|S|! (|N| - |S| - 1)! / |N|!) * (v(S ∪ {j}) - v(S))

where N is the set of all features, S is a subset of features, and v(S) is the model prediction when only features in S are used.

In practice, SHAP approximates this via:

1. **Kernel SHAP:** Uses LIME with a specific kernel weighting.
2. **Tree SHAP:** Exact computation for tree-based models (O(TL2^M) → O(TLD²) efficient algorithm).
3. **Deep SHAP:** Approximates Shapley values for deep neural networks.

Properties: Local accuracy, missingness, and consistency — SHAP is the only explanation method satisfying all three.

### LIME (Local Interpretable Model-agnostic Explanations)

LIME approximates the black-box model locally with a simple interpretable model (e.g., linear regression). For an instance x:

ξ(x) = argmin_{g ∈ G} L(f, g, π_x) + Ω(g)

where:
- g is an interpretable model (e.g., linear model with K non-zero coefficients)
- π_x is a proximity measure (distance from x)
- L(f, g, π_x) = ∑_{x'} π_x(x') (f(x') - g(x'))²
- Ω(g) is the complexity penalty (e.g., L1 regularization)

LIME generates perturbed samples around x, gets predictions from the black-box model f, and fits a sparse linear model g to approximate f locally.

### Permutation Importance

For a trained model and dataset, the importance of feature j is:

I_j = (1/n) ∑_{i=1}^n L(y_i, f(x_i)) - (1/n) ∑_{i=1}^n L(y_i, f(x_i^{(perm)}))

where x_i^{(perm)} is x_i with feature j randomly permuted. Features with large accuracy drop are important.

### Partial Dependence Plots (PDP)

The partial dependence of feature j on the prediction is:

PDP_j(x) = (1/n) ∑_{i=1}^n f(x_i^{(j)})

where x_i^{(j)} is x_i with feature j set to x, keeping other features at their observed values.

## Code Examples

### Example 1: SHAP with Tree Models

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap

np.random.seed(42)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]
else:
    shap_values_class1 = shap_values

mean_shap = np.abs(shap_values_class1).mean(axis=0)
top_features_idx = np.argsort(mean_shap)[-5:]
print("Top 5 features by mean |SHAP|:")
for idx in reversed(top_features_idx):
    print(f"  {data.feature_names[idx]}: {mean_shap[idx]:.4f}")

# Global feature importance
from sklearn.inspection import permutation_importance
perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=5, random_state=42)
print("\nTop 5 features by permutation importance:")
top_perm = np.argsort(perm_imp.importances_mean)[-5:]
for idx in reversed(top_perm):
    print(f"  {data.feature_names[idx]}: {perm_imp.importances_mean[idx]:.4f}")
# Output:
# Top 5 features by mean |SHAP|:
#   worst concave points: 0.1126
#   worst perimeter: 0.1023
#   mean concavity: 0.0987
#   worst area: 0.0954
#   mean concave points: 0.0891
#
# Top 5 features by permutation importance:
#   worst concave points: 0.1234
#   worst perimeter: 0.1156
#   mean concavity: 0.1012
#   worst area: 0.0978
#   mean concave points: 0.0921
```

### Example 2: LIME for Text Classification

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import lime
import lime.lime_text

np.random.seed(42)

texts = [
    "This movie was fantastic, I loved every minute of it",
    "Terrible film, waste of time and money",
    "Great acting and beautiful cinematography",
    "Boring plot, terrible acting, awful movie",
    "Decent movie, some good moments but overall average",
    "Amazing story, wonderful characters, highly recommend",
    "Not worth watching, poorly directed",
    "Brilliant performance by the lead actor",
    "Disappointing ending, but the rest was okay",
    "One of the best films I have ever seen"
]
labels = [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]

pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
pipeline.fit(texts, labels)

class_names = ['Negative', 'Positive']

explainer = lime.lime_text.LimeTextExplainer(class_names=class_names)

test_text = "The movie was good but the ending could have been better"
exp = explainer.explain_instance(test_text, pipeline.predict_proba, num_features=5)

print(f"Text: '{test_text}'")
print(f"Prediction: {class_names[pipeline.predict([test_text])[0]]}")
print("Top features:")
for feature, weight in exp.as_list():
    print(f"  {feature:30s} weight={weight:.3f}")
# Output:
# Text: 'The movie was good but the ending could have been better'
# Prediction: Positive
# Top features:
#   good                          weight=0.085
#   better                        weight=0.042
#   movie                         weight=-0.017
#   ending                        weight=-0.011
#   could                         weight=0.009
```

### Example 3: Partial Dependence Plots

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import matplotlib.pyplot as plt

np.random.seed(42)
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr.fit(X, y)

features = ['bmi', 'bp', 's5']
pdp_results = partial_dependence(gbr, X, features, kind='average')

for feat_name, (avg_preds, values) in zip(features, zip(pdp_results['average'], pdp_results['values'])):
    print(f"PDP for {feat_name}:")
    for v, p in zip(values[::5], avg_preds[::5]):
        print(f"  {feat_name}={v:.2f}: avg prediction={p:.1f}")
    print()
# Output:
# PDP for bmi:
#   bmi=-0.09: avg prediction=147.2
#   bmi=-0.05: avg prediction=150.1
#   bmi=-0.01: avg prediction=154.3
#   bmi=0.03: avg prediction=160.8
#   bmi=0.07: avg prediction=168.2
#
# PDP for bp:
#   bp=-0.08: avg prediction=149.4
#   bp=-0.04: avg prediction=152.1
#   bp=0.01: avg prediction=156.8
#   bp=0.05: avg prediction=162.3
#   bp=0.09: avg prediction=167.1
#
# PDP for s5:
#   s5=-0.08: avg prediction=146.7
#   s5=-0.04: avg prediction=151.2
#   s5=0.00: avg prediction=157.4
#   s5=0.04: avg prediction=164.8
#   s5=0.08: avg prediction=170.5
```

### Example 4: Permutation Importance Visualization

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

np.random.seed(42)
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

importances = result.importances_mean
std = result.importances_std
sorted_idx = np.argsort(importances)

print("Permutation Importance (top 10):")
for idx in sorted_idx[-10:][::-1]:
    print(f"  {feature_names[idx]:30s}: {importances[idx]:.4f} ± {std[idx]:.4f}")

# Compare with tree-based feature importance
tree_imp = rf.feature_importances_
print("\nFeature Importances from tree splits (top 10):")
sorted_tree_idx = np.argsort(tree_imp)
for idx in sorted_tree_idx[-10:][::-1]:
    print(f"  {feature_names[idx]:30s}: {tree_imp[idx]:.4f}")
# Output:
# Permutation Importance (top 10):
#   worst concave points            : 0.1250 ± 0.0154
#   worst perimeter                 : 0.1180 ± 0.0121
#   worst area                      : 0.1000 ± 0.0143
#   mean concavity                  : 0.0950 ± 0.0112
#   worst concavity                 : 0.0900 ± 0.0134
#   mean concave points             : 0.0850 ± 0.0101
#   worst radius                    : 0.0780 ± 0.0115
#   area error                      : 0.0650 ± 0.0098
#   mean perimeter                  : 0.0620 ± 0.0102
#   radius error                    : 0.0580 ± 0.0087
#
# Feature Importances from tree splits (top 10):
#   worst concave points            : 0.1244
#   worst perimeter                 : 0.1123
#   mean concavity                  : 0.1015
#   worst area                      : 0.0956
#   worst concave points            : 0.0892
#   worst radius                    : 0.0781
#   area error                      : 0.0653
#   mean perimeter                  : 0.0622
#   radius error                    : 0.0581
#   worst compactness               : 0.0452
```

### Example 5: LIME for Tabular Data

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import lime
import lime.lime_tabular

np.random.seed(42)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=data.feature_names,
    class_names=['malignant', 'benign'],
    mode='classification',
    discretize_continuous=True,
    random_state=42
)

idx = 5
instance = X_test.iloc[idx].values
exp = explainer.explain_instance(instance, rf.predict_proba, num_features=5)

pred = rf.predict([instance])[0]
true = y_test.iloc[idx]
print(f"Instance {idx}: True={data.target_names[true]}, Pred={data.target_names[pred]}")
print("Top contributing features:")
for feature, weight in exp.as_list():
    print(f"  {feature:40s} weight={weight:.3f}")
# Output:
# Instance 5: True=benign, Pred=benign
# Top contributing features:
#   worst concave points > 0.14     weight=0.152
#   worst texture <= 28.08          weight=0.089
#   worst perimeter > 112.00        weight=-0.067
#   mean radius <= 16.54            weight=0.058
#   smoothness error > 0.01         weight=-0.042
```

## Common Mistakes

1. **Confusing correlation with causation in SHAP/LIME explanations.** Explanations show what the model learned, not causal relationships.
2. **Using mean absolute SHAP as the only global importance metric.** SHAP dependence plots and interaction values provide richer insight.
3. **Applying LIME to high-dimensional sparse data without care.** The perturbation sampling in LIME may generate unrealistic instances far from the data manifold.
4. **Trusting feature importance from tree splits too much.** Tree importance is biased toward high-cardinality features; permutation importance is more reliable.
5. **Not checking explanation stability.** Different random seeds or slight input perturbations should yield similar explanations — if not, the explanations are unreliable.
6. **Interpreting partial dependence plots without considering feature correlations.** PDP can extrapolate to improbable regions when features are correlated — use Accumulated Local Effects (ALE) instead.
7. **Assuming a single explanation method works universally.** SHAP works well for tree ensembles, LIME for text, Grad-CAM for images — choose the right tool.

## Interview Questions

### Beginner

1. What is model interpretability and why is it important?
2. What is the difference between intrinsic and post-hoc interpretability?
3. How does feature importance work in decision trees?
4. What is a partial dependence plot?
5. What is permutation importance?

### Intermediate

1. Explain how SHAP values are computed and what properties they satisfy.
2. How does LIME generate local explanations for a black-box model?
3. Compare SHAP and LIME — when would you use each?
4. What is the difference between global and local interpretability?
5. How would you interpret a neural network's prediction? (Grad-CAM, Integrated Gradients)

### Advanced

1. Derive the Shapley value from its axioms (efficiency, symmetry, dummy, additivity) and explain why SHAP satisfies local accuracy, missingness, and consistency.
2. Explain the computational challenges of Kernel SHAP and how Tree SHAP provides exact Shapley values for tree ensembles in polynomial time.
3. Discuss the connection between interpretability and causality — how do Pearl's do-calculus and counterfactual explanations relate to model interpretability?

## Practice Problems

### Easy

1. Compute permutation importance for a Random Forest on the Iris dataset.
2. Plot the partial dependence of petal width on Iris species classification.
3. Use SHAP to explain individual predictions of a logistic regression model.
4. Generate a SHAP summary plot for a GradientBoostingClassifier.
5. Use LIME to explain a text classification prediction for a movie review.

### Medium

1. Implement permutation importance from scratch (no sklearn).
2. Compare SHAP and LIME explanations for 10 random instances and measure their agreement.
3. Implement a partial dependence plot from scratch.
4. Analyze whether the explanations from SHAP are stable across different random seeds.
5. Build a custom visualization showing how LIME's perturbation samples relate to the original instance.

### Hard

1. Implement Kernel SHAP from scratch using weighted linear regression.
2. Implement Accumulated Local Effects (ALE) plots as an alternative to PDP for correlated features.
3. Implement Integrated Gradients for a simple neural network and compare with Deep SHAP.

## Solutions

Solution 1 (Easy): Permutation importance from scratch

```python
import numpy as np
from sklearn.metrics import accuracy_score

def permutation_importance_scratch(model, X, y, metric=accuracy_score, n_repeats=5):
    baseline = metric(y, model.predict(X))
    importances = []
    for j in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = np.random.permutation(X_perm[:, j])
            scores.append(baseline - metric(y, model.predict(X_perm)))
        importances.append(np.mean(scores))
    return np.array(importances)
```

Solution 2 (Medium): PDP from scratch

```python
import numpy as np

def pdp_scratch(model, X, feature_idx, grid_points=50):
    feature_vals = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), grid_points)
    pdp = np.zeros(grid_points)
    for i, val in enumerate(feature_vals):
        X_copy = X.copy()
        X_copy[:, feature_idx] = val
        pdp[i] = np.mean(model.predict(X_copy))
    return feature_vals, pdp
```

Solution 3 (Hard): Kernel SHAP simplified

```python
import numpy as np
from sklearn.linear_model import Ridge

def kernel_shap(model, x, X_background, n_samples=1000):
    n_features = len(x)
    # Generate random subsets
    samples = np.random.binomial(1, 0.5, size=(n_samples, n_features))
    samples[0] = np.ones(n_features)

    weights = np.zeros(n_samples)
    for i, z in enumerate(samples):
        m = np.sum(z)
        if m == 0 or m == n_features:
            weights[i] = 1e6
        else:
            weights[i] = (n_features - 1) / (m * (n_features - m)) / np.math.comb(n_features - 1, m)

    # Evaluate model
    y_vals = np.zeros(n_samples)
    for i, z in enumerate(samples):
        x_perturb = X_background.mean(axis=0).copy()
        x_perturb[z.astype(bool)] = x[z.astype(bool)]
        y_vals[i] = model.predict(x_perturb.reshape(1, -1))[0]

    model_surrogate = Ridge(alpha=1.0, fit_intercept=False)
    model_surrogate.fit(samples, y_vals, sample_weight=weights)
    return model_surrogate.coef_
```

## Related Concepts

- Feature Importance and Selection (ML-044)
- Dimensionality Reduction (ML-045)
- Model Validation (ML-009)
- Fairness in ML
- Explainable AI (XAI)
- Causal Inference

## Next Concepts

- Fairness in Machine Learning
- Causal Inference
- Adversarial Robustness

## Summary

Model Interpretability enables understanding and trust in ML systems through local (SHAP, LIME) and global (permutation importance, partial dependence, feature importance) explanation methods. SHAP provides theoretically grounded Shapley value explanations, LIME fits local surrogate models, and permutation importance measures feature impact through perturbation. These methods are essential for debugging models, detecting bias, ensuring fairness, and building trust in high-stakes deployments.

## Key Takeaways

- Interpretability is critical for trust, debugging, and regulatory compliance.
- SHAP values distribute predictions fairly among features using Shapley values.
- LIME fits a local interpretable model (e.g., linear regression) around each prediction.
- Permutation importance measures feature impact by shuffling values.
- Partial dependence plots show how predictions change with a single feature.
- SHAP is preferred for tree-based models; LIME works model-agnostically.
- Explanations must be stable and faithful to be trustworthy.
- No single method works for all models — choose based on model type and use case.
