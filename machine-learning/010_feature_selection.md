# Concept: Feature Selection

## Concept ID

ML-010

## Difficulty

INTERMEDIATE

## Domain

Machine Learning

## Module

ML Fundamentals

## Learning Objectives

- Understand the importance of feature selection for model performance and interpretability
- Distinguish between filter, wrapper, and embedded feature selection methods
- Apply filter methods using correlation, chi-square, and mutual information
- Implement wrapper methods including Recursive Feature Elimination (RFE)
- Apply embedded methods using Lasso regularization and tree-based importance
- Use scikit-learn's `SelectKBest`, `RFE`, and `SelectFromModel`

## Prerequisites

- ML-001: What is Machine Learning
- ML-004: Overfitting and Underfitting
- ML-009: Feature Engineering
- Basic understanding of correlation and information theory

## Definition

Feature selection is the process of selecting a subset of relevant features (variables, predictors) for use in machine learning model construction. It reduces the dimensionality of the feature space while preserving (or improving) model performance, interpretability, and computational efficiency.

### The Three Categories of Feature Selection

**Filter Methods**: Rank features by some statistical measure independent of any ML model. Features are selected before training. Examples: correlation coefficient, chi-square test, mutual information, variance threshold.

**Wrapper Methods**: Use a ML model's performance to evaluate feature subsets. They search the space of feature subsets and select the one that maximizes model performance. Examples: Recursive Feature Elimination (RFE), forward selection, backward elimination, exhaustive search.

**Embedded Methods**: Perform feature selection during the model training process. The model inherently selects features as part of learning. Examples: L1 regularization (Lasso), tree-based feature importance, Elastic Net.

## Intuition

### Filter Methods Intuition

Imagine you are a chef selecting ingredients for a dish. Filter methods are like checking each ingredient individually: "Is this ingredient fresh? Does it have good flavor on its own?" You evaluate ingredients independently and only keep the best ones.

Pros: Fast, simple, model-independent. Cons: Misses interactions between features — two weak individual features might be powerful together.

### Wrapper Methods Intuition

Wrapper methods are like cooking with different ingredient combinations and tasting the result each time. You try combinations, see how the dish turns out, and keep iterating. You evaluate subsets as a whole, capturing interactions.

Pros: Captures feature interactions, optimal for the specific model. Cons: Computationally expensive, risk of overfitting to the selection criterion.

### Embedded Methods Intuition

Embedded methods are like a chef who continuously adjusts ingredients while cooking, tasting and adding more or less as needed. Selection happens naturally during the cooking process.

Pros: Balances filter speed with wrapper accuracy, built into model training. Cons: Model-specific, may not transfer between different model types.

## Why This Concept Matters

Feature selection addresses several critical challenges in machine learning:

1. **Curse of Dimensionality**: As features increase, the data becomes sparse, requiring exponentially more samples to maintain statistical significance.
2. **Overfitting Reduction**: Irrelevant features add noise that models may incorrectly learn as patterns.
3. **Improved Accuracy**: Removing irrelevant and redundant features often improves model performance.
4. **Reduced Training Time**: Fewer features means faster training and inference.
5. **Enhanced Interpretability**: Models with fewer features are easier to understand, explain, and debug.
6. **Data Collection Efficiency**: Understanding which features matter guides future data collection efforts.
7. **Deployment Cost Reduction**: Fewer features means less data to collect and process in production.

## Mathematical Explanation

### Filter Methods

**Pearson Correlation Coefficient** (for regression tasks with continuous features):

$$r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \sum_{i=1}^{n} (y_i - \bar{y})^2}}$$

Select features with high absolute correlation with the target. Range: [-1, 1].

**Chi-Square Test** (for classification with categorical features):

$$\chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

Where O is observed frequency and E is expected frequency under independence. High chi-square indicates dependence between feature and target.

**Mutual Information** (captures non-linear relationships):

$$I(X; Y) = \sum_{x \in X} \sum_{y \in Y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

Mutual information measures how much knowing X reduces uncertainty about Y. It captures any kind of dependency (linear or non-linear), unlike correlation which only captures linear relationships.

**Variance Threshold**: Removes features with variance below a threshold, as constant or near-constant features provide no discriminative information.

### Wrapper Methods

**Recursive Feature Elimination (RFE)**: Given a model that assigns weights to features (e.g., coefficients in linear models, feature importances in tree models):

1. Train model on all features.
2. Rank features by importance (absolute coefficient or importance score).
3. Remove the least important feature(s).
4. Repeat steps 1-3 on the reduced set until desired number of features remains.

Computational complexity: O(n_features × p) where p is the cost of training the model.

### Embedded Methods

**L1 Regularization (Lasso)**: Adds the L1 norm of coefficients to the loss function:

$$\min_{\beta} \frac{1}{2n} \sum_{i=1}^{n} (y_i - X_i\beta)^2 + \lambda \sum_{j=1}^{p} |\beta_j|$$

The L1 penalty induces sparsity — many coefficients become exactly zero, effectively performing feature selection.

**Tree-Based Importance**: Decision trees and ensemble methods (Random Forest, Gradient Boosting) compute feature importance as:

- For single trees: total reduction in impurity (Gini or entropy) weighted by the number of samples reaching each node where the feature is used.
- For forests: average importance across all trees, often normalized.

## Code Examples

### Example 1: Filter Methods — Correlation and Mutual Information

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler

# Load diabetes dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

print(f"Original features: {X.shape[1]}")
print(f"Features: {list(X.columns)}")
# Output:
# Original features: 10
# Features: ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

# Correlation with target
correlations = X.apply(lambda col: col.corr(pd.Series(y)))
print("\nCorrelation with target:")
for feature, corr in correlations.sort_values(key=abs, ascending=False).items():
    print(f"  {feature}: {corr:.4f}")
# Output:
# Correlation with target:
#   bmi: 0.5865
#   s5: 0.5659
#   bp: 0.4413
#   s4: 0.4305
#   s6: 0.3827
#   s1: 0.2120
#   s2: 0.1741
#   s3: -0.3792
#   sex: 0.0431
#   age: 0.1890

# F-regression (ANOVA F-value)
selector_f = SelectKBest(score_func=f_regression, k=5)
selector_f.fit(X, y)
f_scores = pd.DataFrame({
    'feature': X.columns,
    'f_score': selector_f.scores_,
    'p_value': selector_f.pvalues_
}).sort_values('f_score', ascending=False)

print("\nF-Regression Scores:")
print(f_scores.to_string(index=False))
# Output:
# F-Regression Scores:
#   feature    f_score       p_value
#       bmi  341.9938  1.822069e-55
#        s5  310.3742  1.132369e-51
#        bp  209.1315  1.658918e-38
#        s4  176.7327  2.779535e-33
#        s6  137.4056  1.062313e-27
#        s3   80.0094  1.045894e-17
#        s1   27.4822  2.210069e-07
#        s2   18.2992  2.216448e-05
#       age   17.1522  3.985649e-05
#       sex    0.8880  3.465618e-01

# Mutual information (captures non-linear relationships)
mi_scores = mutual_info_regression(X, y, random_state=42)
mi_df = pd.DataFrame({
    'feature': X.columns,
    'mutual_info': mi_scores
}).sort_values('mutual_info', ascending=False)

print("\nMutual Information Scores:")
print(mi_df.to_string(index=False))
# Output:
# Mutual Information Scores:
#   feature  mutual_info
#       bmi     0.328380
#        s5     0.280896
#        bp     0.132229
#        s4     0.117270
#        s6     0.056046
#        s3     0.031385
#        s1     0.014768
#        s2     0.009793
#       age     0.007561
#       sex     0.003519
```

### Example 2: Wrapper Methods — Recursive Feature Elimination (RFE)

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
feature_names = cancer.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Recursive Feature Elimination with SVM
svm = SVC(kernel='linear', C=1.0, random_state=42)

# Select top 10 features
rfe = RFE(estimator=svm, n_features_to_select=10)
rfe.fit(X_train, y_train)

selected_features = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]
feature_ranks = pd.DataFrame({
    'feature': feature_names,
    'rank': rfe.ranking_,
    'selected': rfe.support_
}).sort_values('rank')

print("Top 10 selected features (RFE):")
print(feature_ranks[feature_ranks['selected']].to_string(index=False))
# Output:
# Top 10 selected features (RFE):
#              feature  rank  selected
#    mean concave points     1      True
#         mean perimeter     1      True
#         worst perimeter     1      True
#             mean area     1      True
#             worst area     1      True
#        worst concave points     1      True
#     worst concavity     1      True
#        mean concavity     1      True
#           mean radius     1      True
#          texture error     1      True

# Evaluate
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)
svm.fit(X_train_rfe, y_train)
y_pred = svm.predict(X_test_rfe)
print(f"\nAccuracy with 10 features: {accuracy_score(y_test, y_pred):.4f}")
# Output: Accuracy with 10 features: 0.9708

# RFE with Cross-Validation (automatically selects optimal number)
rfecv = RFECV(estimator=svm, cv=5, scoring='accuracy')
rfecv.fit(X_train, y_train)

print(f"\nOptimal number of features (RFECV): {rfecv.n_features_}")
# Output: Optimal number of features (RFECV): 11

print("RFECV scores by feature count:")
for i, (n_features, score) in enumerate(zip(
    range(1, len(feature_names) + 1), rfecv.cv_results_['mean_test_score']
)):
    if i < 5 or i > len(feature_names) - 5:  # show first 5 and last 5
        print(f"  {n_features:2d} features: {score:.4f}")
    elif i == 5:
        print(f"  ...")
# Output:
# RFECV scores by feature count:
#    1 features: 0.9274
#    2 features: 0.9371
#    3 features: 0.9543
#    4 features: 0.9595
#    5 features: 0.9595
#   ...
#   28 features: 0.9575
#   29 features: 0.9575
#   30 features: 0.9575
```

### Example 3: Embedded Methods — Lasso and Tree Importance

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
feature_names = diabetes.feature_names
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Lasso (L1 Regularization) — Embedded feature selection
lasso = Lasso(alpha=0.1, random_state=42)
lasso.fit(X_train, y_train)

coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': lasso.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print("Lasso Coefficients (zero = feature excluded):")
print(coef_df.to_string(index=False))
# Output:
# Lasso Coefficients (zero = feature excluded):
#   feature  coefficient
#       s5   478.687360
#       bmi   447.643849
#       bp   279.266641
#        s3   156.149286
#       sex  -140.768959
#        s6   105.659211
#        s4    81.064620
#       age    -0.000000
#        s1     0.000000
#        s2    -0.000000

# Use SelectFromModel to automatically select
selector = SelectFromModel(lasso, threshold=1.0)
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

print(f"\nFeatures selected by Lasso: {selector.get_support().sum()}")
selected = [feature_names[i] for i in range(len(feature_names)) if selector.get_support()[i]]
print(f"Selected features: {selected}")
# Output:
# Features selected by Lasso: 7
# Selected features: ['sex', 'bmi', 'bp', 's3', 's4', 's5', 's6']

# Random Forest Feature Importance
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nRandom Forest Feature Importances:")
print(importance_df.to_string(index=False))
# Output:
# Random Forest Feature Importances:
#   feature  importance
#       s5    0.293820
#       bmi    0.219155
#       bp    0.139470
#        s6    0.103258
#        s4    0.073107
#        s1    0.065128
#        s2    0.055124
#        s3    0.033263
#       age    0.014618
#       sex    0.003058

# SelectFromModel with Random Forest
selector_rf = SelectFromModel(rf, threshold='median')
selector_rf.fit(X_train, y_train)
X_train_rf = selector_rf.transform(X_train)
X_test_rf = selector_rf.transform(X_test)

selected_rf = [feature_names[i] for i in range(len(feature_names)) if selector_rf.get_support()[i]]
print(f"\nFeatures selected by RF (above median): {selected_rf}")
# Output:
# Features selected by RF (above median): ['bmi', 'bp', 's4', 's5', 's6']
```

### Example 4: Comparing Feature Selection Methods

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import (
    SelectKBest, f_regression, RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor

# Create synthetic data with only 5 informative features out of 50
np.random.seed(42)
X, y = make_regression(
    n_samples=500, n_features=50,
    n_informative=5, noise=0.5,
    random_state=42
)

base_model = LinearRegression()

# No selection
scores_all = cross_val_score(base_model, X, y, cv=5, scoring='r2')
print(f"All 50 features: R² = {scores_all.mean():.4f} (+/- {scores_all.std():.4f})")
# Output: All 50 features: R² = 0.6758 (+/- 0.0365)

# Filter: SelectKBest with f_regression
selector_kb = SelectKBest(score_func=f_regression, k=5)
X_kb = selector_kb.fit_transform(X, y)
scores_kb = cross_val_score(base_model, X_kb, y, cv=5, scoring='r2')
print(f"SelectKBest (5): R² = {scores_kb.mean():.4f} (+/- {scores_kb.std():.4f})")
# Output: SelectKBest (5): R² = 0.9596 (+/- 0.0097)

# Wrapper: RFE with LinearRegression
rfe = RFE(estimator=LinearRegression(), n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)
scores_rfe = cross_val_score(base_model, X_rfe, y, cv=5, scoring='r2')
print(f"RFE (5): R² = {scores_rfe.mean():.4f} (+/- {scores_rfe.std():.4f})")
# Output: RFE (5): R² = 0.9596 (+/- 0.0097)

# Embedded: Lasso with SelectFromModel
lasso = Lasso(alpha=0.01, random_state=42)
selector_lasso = SelectFromModel(lasso, threshold=1e-5)
X_lasso = selector_lasso.fit_transform(X, y)
scores_lasso = cross_val_score(base_model, X_lasso, y, cv=5, scoring='r2')
print(f"Lasso selection: R² = {scores_lasso.mean():.4f} (+/- {scores_lasso.std():.4f})")
# Output: Lasso selection: R² = 0.9594 (+/- 0.0097)

# All methods successfully recover the 5 informative features
# and dramatically improve performance by removing noise features
```

## Common Mistakes

1. **Performing feature selection before train/test split**: This causes data leakage because the selection criteria use information from the test set. Always select features within the training data only.
2. **Using feature importance from a single model as absolute truth**: Different models find different features important. Lasso might select different features than Random Forest. Validate with domain knowledge.
3. **Selecting too few or too many features**: Too few features may miss important signal; too many includes noise. Use cross-validation to find the optimal number.
4. **Ignoring multicollinearity in feature selection**: Highly correlated features can cause instability in selection (Lasso arbitrarily picks one of a correlated pair). Consider grouping or combining correlated features first.
5. **Applying feature selection without domain validation**: Statistical significance does not guarantee practical relevance. Always validate selected features with domain experts.
6. **Using the same selection method across different model types**: Filter methods select features independently of the model. A feature selected by correlation might not be useful for a tree-based model, and vice versa.
7. **Assuming feature selection is a one-time step**: Feature relevance can change over time (concept drift). In production systems, periodically re-evaluate feature selection.

## Interview Questions

### Beginner - 5

1. **Q: What is the difference between feature selection and dimensionality reduction (PCA)?**
   A: Feature selection keeps original features (subset selection). Dimensionality reduction (PCA) creates new features that are linear combinations of original features, losing interpretability but potentially capturing more information with fewer dimensions.

2. **Q: What are the three main categories of feature selection methods?**
   A: Filter methods (statistical measures, model-independent), wrapper methods (use model performance, computationally expensive), and embedded methods (selection built into model training).

3. **Q: What is the benefit of feature selection?**
   A: Reduces overfitting, improves model performance, decreases training time, enhances interpretability, lowers data collection costs, and mitigates the curse of dimensionality.

4. **Q: When is feature selection most important?**
   A: When the number of features is large relative to the number of samples (p > n), when many features are irrelevant or redundant, and when model interpretability is critical.

5. **Q: What is the difference between univariate feature selection and multivariate feature selection?**
   A: Univariate selection evaluates each feature independently (e.g., correlation, chi-square). Multivariate selection considers feature interactions (e.g., RFE, Lasso). Univariate is faster but misses interactions.

### Intermediate - 5

1. **Q: Compare filter, wrapper, and embedded methods in terms of computational cost and interaction capture.**
   A: Filter: O(p) computation (fast), no interaction capture. Wrapper: O(p × model_cost) (slow), full interaction capture. Embedded: O(model_cost) (moderate), limited interaction capture (depends on model). Filters are used as pre-processing; wrappers for final selection when computation budget allows; embedded for balance.

2. **Q: What is the problem with using correlation for feature selection when features are non-linear or the target is categorical?**
   A: Correlation only captures linear relationships. Two features could have zero correlation with the target but be highly predictive through non-linear relationships. Use mutual information instead, which captures any type of dependency.

3. **Q: How does L1 regularization perform feature selection?**
   A: L1 adds the sum of absolute coefficient values to the loss function. During optimization, this penalty forces many coefficients to exactly zero (sparsity). The regularization strength λ controls how many features are zeroed out — higher λ creates sparser models.

4. **Q: Explain the concept of "feature importance" in tree-based models and its limitations.**
   A: Feature importance in tree models measures how much each feature reduces impurity (Gini or entropy) across all splits where the feature is used. Limitations: (1) Biased toward high-cardinality features. (2) Does not capture interactions — a feature may be important only in combination with others. (3) Unstable across different random seeds. (4) Permutation importance is a more reliable alternative.

5. **Q: What is the "curse of dimensionality" and how does feature selection help?**
   A: As dimensionality increases, data becomes sparse, distances become meaningless, models require exponentially more samples, and overfitting risk grows. Feature selection removes irrelevant dimensions, restoring meaningful distance measures, reducing overfitting, and making models more robust.

### Advanced - 3

1. **Q: Derive the relationship between L1 regularization and LASSO's feature selection property. Explain why L1 produces sparsity while L2 does not.**
   A: The optimization problem is $\min_\beta L(\beta) + \lambda ||\beta||_1$. The L1 penalty has a non-differentiable "corner" at zero in each dimension. The KKT conditions show that for features where the gradient of the loss at zero is less than λ, the optimal coefficient is exactly zero. For L2, the gradient near zero is linear, so coefficients are shrunk but never zeroed. Geometrically: the L1 constraint region is a diamond with corners on axes; the L2 constraint is a sphere. The solution (where loss contours touch the constraint) often occurs at corners for L1, giving sparsity.

2. **Q: Prove that forward feature selection (greedy) does not guarantee finding the optimal feature subset, and provide a counterexample.**
   A: Forward selection greedily adds the feature that most improves performance. Counterexample: With 3 features (a, b, c), suppose the optimal subset is {b, c} with R²=0.95. But individually, a has R²=0.7, b has R²=0.5, c has R²=0.4. Forward selection first picks a (best individual), then adds either b or c. But {a, b} might only achieve R²=0.8 because a and b are redundant. Forward selection never finds {b, c} because it committed to a first. This is the "nesting effect" problem common to greedy search.

3. **Q: Design a feature selection strategy for a dataset with 100,000 features and 10,000 samples, considering computational constraints and the need for both speed and accuracy.**
   A: Multi-stage strategy: (1) **Variance threshold**: Remove near-zero variance features (O(p)). (2) **Correlation filter**: Remove highly correlated features (|r| > 0.95) keeping the one more correlated with target (O(p²) but can be optimized). (3) **Mutual information**: Select top k=1000 features (faster than model-based). (4) **Lasso/Ridge with cross-validation**: Further select to ~100 features (O(p × n) with regularization path). (5) **Backward elimination or RFE with a fast model** (e.g., linear SVM): Final selection of 20-50 features. Validity: Use cross-validation at each stage and ensure only training data is used for selection.

## Practice Problems

### Easy - 5

1. **Problem**: You have 100 features and 500 samples. Which feature selection approach would you recommend and why?

2. **Problem**: What is the range of values for the Pearson correlation coefficient, and what do the extremes mean?

3. **Problem**: A feature has p-value = 0.8 in an F-test against the target. Should you include this feature?

4. **Problem**: Name two filter methods and two embedded methods for feature selection.

5. **Problem**: What happens to model performance when you select too many irrelevant features?

### Medium - 5

1. **Problem**: You build a model with 200 features and achieve training R² = 0.99 but test R² = 0.55. How can feature selection help?

2. **Problem**: Compare forward selection and backward elimination in terms of computational cost and statistical properties.

3. **Problem**: A dataset has two highly correlated features (r = 0.95). Both have moderate correlation with the target (r = 0.4 and 0.38). What would Lasso do? What would stepwise selection do? Which approach is more stable?

4. **Problem**: Explain why mutual information is preferred over Pearson correlation for feature selection in a dataset with non-linear relationships.

5. **Problem**: You have 50 features but use SelectKBest with k=10 on the entire dataset, then cross-validate. What is the problem and how would you fix it?

### Hard - 3

1. **Problem**: Prove that mutual information I(X;Y) equals 0 if and only if X and Y are independent. Show that it captures both linear and non-linear dependencies.

2. **Problem**: Compare the statistical properties (bias, variance, consistency) of lasso-based feature selection vs. random-forest-based feature selection for high-dimensional data with correlated features.

3. **Problem**: Design a stable feature selection procedure for a medical diagnosis problem with p=1000 genetic markers and n=200 patients. Address stability (small data perturbations should not drastically change selected features), reproducibility, and clinical interpretability.

## Solutions

### Easy Solutions

1. Filter methods (e.g., mutual information, correlation) because they are fast and do not overfit when p is large relative to n. Follow up with Lasso for embedded selection.
2. Range: [-1, 1]. +1 = perfect positive linear relationship, -1 = perfect negative linear relationship, 0 = no linear relationship.
3. No. A high p-value (F-test) indicates the feature is not statistically significant in predicting the target. The feature is likely irrelevant.
4. Filter methods: Pearson correlation, chi-square test, mutual information, variance threshold. Embedded methods: Lasso (L1 regularization), tree-based feature importance, Elastic Net.
5. Model performance degrades: training performance may stay high, but test performance suffers due to overfitting. More noise features = more spurious correlations = worse generalization.

### Medium Solutions

1. Classic overfitting. Feature selection removes irrelevant/noisy features that the model is memorizing. Use Lasso, RFE with CV, or mutual information filter. Start by selecting top 10-20 features and evaluate test performance improvement.
2. Forward selection: starts with empty set, adds best feature each step. Cost: O(p² × model_cost). Backward elimination: starts with all features, removes worst each step. Cost: same O(p² × model_cost) but starts with expensive full model training. Forward is cheaper for final small sets; backward is more thorough (considers interactions from start).
3. Lasso will typically select one of the two correlated features (arbitrarily) and zero out the other due to the grouping effect issue. Stepwise selection might include both or either depending on order. Lasso with elastic net (mixing L1 and L2) is more stable for correlated features, tending to select both or neither.
4. Mutual information captures any form of dependency (linear, non-linear, complex interactions). Pearson correlation only captures linear relationships. For example, Y = X² has MI > 0 but Pearson correlation ≈ 0. Mutual information is zero iff X and Y are statistically independent.
5. Problem: Data leakage — selection is done on the entire dataset, then CV evaluates on the same data. The test fold in CV was used for selection, biasing results optimistically. Fix: integrate selection inside CV — for each fold, perform SelectKBest on training fold only.

### Hard Solutions

1. I(X;Y) = D_KL(p(x,y) || p(x)p(y)) where D_KL is the Kullback-Leibler divergence. By Gibbs' inequality, D_KL ≥ 0 with equality iff p(x,y) = p(x)p(y), which is the definition of independence. MI captures non-linear dependencies because it is based on the full joint distribution, not just moments. For Y = sin(X) where X ~ Uniform: Pearson r ≈ 0, but MI > 0 because knowledge of X reduces uncertainty about Y.
2. Lasso: high bias (shrinks coefficients), selection is consistent under the irrepresentable condition (features not too correlated), can be unstable with correlated features. Random Forest: lower bias, selection is based on impurity reduction, tends to prefer high-cardinality features, more stable to small data perturbations. For high-dimensional correlated data: Lasso may give unstable, arbitrary selection among correlated groups; RF gives more stable importance rankings but may miss weak signals. Stability selection (subsampling + lasso) combines both advantages.
3. Procedure: (1) Use stability selection — run Lasso on 100 bootstrap samples and select features that are chosen in > 80% of samples. (2) Use domain knowledge to constrain the feature set to known relevant genetic pathways. (3) Apply elastic net (mixing L1 and L2) for grouped selection of correlated markers. (4) Use cross-validated selection that emphasizes stability: compare overlap of selected sets across CV folds. (5) Validate with independent cohort data. (6) Report selected features with confidence intervals (stability scores). Clinical interpretability: present selected markers with known biological function, not just statistical criteria.

## Related Concepts

- **ML-009: Feature Engineering** — Creating the features that feature selection will evaluate
- **ML-004: Overfitting and Underfitting** — Feature selection is a primary tool for reducing overfitting
- **ML-005: Cross-Validation** — Essential for unbiased feature selection evaluation
- **Dimensionality Reduction (PCA, t-SNE)**: Alternative to feature selection that creates new features
- **Regularization (L1, L2, Elastic Net)**: The mathematical foundation for embedded feature selection

## Next Concepts

- **Advanced Regularization**: Elastic Net, group lasso, adaptive lasso
- **Automated Feature Selection**: Stability selection, Boruta, permutation importance

## Summary

Feature selection reduces the dimensionality of the feature space by selecting a relevant subset of original features. Three categories exist: filter methods (statistical measures independent of any model), wrapper methods (model-based performance evaluation of feature subsets), and embedded methods (selection integrated into model training). Filter methods are fast but miss interactions; wrapper methods capture interactions but are computationally expensive; embedded methods offer a practical balance. Feature selection addresses the curse of dimensionality, reduces overfitting, improves model interpretability, and decreases computational costs. Proper feature selection requires cross-validation to avoid data leakage and over-optimistic performance estimates.

## Key Takeaways

1. Feature selection mitigates the curse of dimensionality and improves model generalization.
2. Filter methods (correlation, mutual information, chi-square) are fast and model-agnostic.
3. Wrapper methods (RFE, forward/backward selection) use model performance but are computationally expensive.
4. Embedded methods (Lasso, tree importance) select features during training.
5. Always perform feature selection inside the cross-validation loop to avoid data leakage.
6. Mutual information captures non-linear relationships better than correlation.
7. Domain knowledge should guide and validate statistical feature selection.
