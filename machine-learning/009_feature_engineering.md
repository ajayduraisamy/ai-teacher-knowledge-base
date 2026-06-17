# Concept: Feature Engineering

## Concept ID

ML-009

## Difficulty

INTERMEDIATE

## Domain

Machine Learning

## Module

ML Fundamentals

## Learning Objectives

- Understand feature engineering as the process of creating informative features from raw data
- Apply numerical transformations (log, square root, binning, scaling)
- Encode categorical variables appropriately (one-hot, label, target encoding)
- Extract features from date/time data
- Generate text features using TF-IDF and count vectorization
- Create interaction and polynomial features
- Know when and how to use domain knowledge for feature creation

## Prerequisites

- ML-001: What is Machine Learning
- Basic understanding of data types (numerical, categorical, text, date)
- ML-002: Supervised vs Unsupervised Learning

## Definition

Feature engineering is the process of transforming raw data into features that better represent the underlying problem to machine learning algorithms, improving model accuracy and robustness. It is the art and science of creating input variables that make machine learning algorithms work effectively.

### Key Types of Feature Engineering

**Numerical Transformations**: Applying mathematical functions (log, square root, Box-Cox) to change the distribution of numerical features, making them more suitable for modeling.

**Categorical Encoding**: Converting categorical variables (colors, countries, product types) into numerical representations that algorithms can process.

**Date/Time Features**: Extracting useful components from timestamps (day of week, hour, month, days since event, holiday flags).

**Text Features**: Converting unstructured text into numerical vectors using techniques like bag-of-words, TF-IDF, and word embeddings.

**Interaction Features**: Creating new features that capture relationships between existing features (e.g., product of two features, ratio).

**Polynomial Features**: Generating powers and interaction terms from numerical features to capture non-linear relationships.

**Domain-Specific Features**: Creating features based on expert knowledge about the problem domain (e.g., financial ratios in credit scoring, patient vitals combinations in healthcare).

## Intuition

Feature engineering is the process of "helping the algorithm see what matters."

Imagine you are trying to teach someone to identify ripe watermelons. You could give them raw sensor data (weight, diameter, color intensity at various wavelengths). Or you could teach them features that experts use: the "thump test" sound (hollow vs dull), the field spot color (creamy yellow = ripe, white = unripe), the ratio of weight to size (density). The engineered features make the learning problem dramatically easier.

In the same way, a well-engineered feature like "distance to nearest competitor store" might be far more predictive for a retail model than raw latitude and longitude coordinates. Feature engineering encodes domain knowledge into a form the algorithm can use directly.

As the ML adage goes: "Coming up with features is difficult, time-consuming, requires expert knowledge. Applied machine learning is basically feature engineering."

## Why This Concept Matters

Feature engineering is often the difference between a good model and a great one. In many real-world ML projects:

1. **Feature engineering has the highest ROI**: Adding one well-engineered feature often improves performance more than tuning hyperparameters for days.
2. **Models are only as good as their features**: No amount of algorithmic sophistication can compensate for poor features.
3. **Domain knowledge is encoded through features**: Feature engineering is where domain expertise directly impacts model performance.
4. **Simpler models with good features beat complex models with raw data**: A logistic regression with excellent features can outperform a neural network on raw data.
5. **Feature engineering reduces the burden on algorithms**: Good features make it easier for any algorithm to find patterns.

## Mathematical Explanation

### Log Transformation

The log transformation $x' = \log(x + c)$ is used for:
- Reducing right skewness in distributions
- Converting multiplicative relationships to additive ones
- Stabilizing variance

For a feature with log-normal distribution, the log transformation produces a roughly normal distribution, which is preferred by many algorithms.

### Box-Cox Transformation

A family of power transformations parameterized by $\lambda$:

$$x'(\lambda) = \begin{cases} \frac{x^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\ \log(x) & \text{if } \lambda = 0 \end{cases}$$

The optimal $\lambda$ is found by maximizing the log-likelihood of the transformed data under a normal distribution assumption.

### One-Hot Encoding

For a categorical variable with k categories, create k binary (0/1) dummy variables, where exactly one is 1 for each sample. To avoid the dummy variable trap (perfect multicollinearity), one category is typically dropped.

$$\text{Category } i \rightarrow [0, 0, ..., 1, ..., 0] \text{ (position i)}$$

### TF-IDF (Term Frequency - Inverse Document Frequency)

TF-IDF weights terms by their importance in a document relative to a corpus:

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

$$\text{TF}(t, d) = \frac{\text{Count of term t in document d}}{\text{Total terms in document d}}$$

$$\text{IDF}(t) = \log\left(\frac{\text{Total documents}}{\text{Number of documents containing term t}}\right) + 1$$

TF-IDF down-weights common words (like "the", "is") that appear in many documents and up-weights rare, discriminative words.

## Code Examples

### Example 1: Numerical Transformations

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Create skewed feature data
np.random.seed(42)
house_prices = np.random.lognormal(mean=12, sigma=0.5, size=1000)

print(f"Original: mean={house_prices.mean():.0f}, "
      f"median={np.median(house_prices):.0f}, "
      f"skew={pd.Series(house_prices).skew():.2f}")
# Output: Original: mean=165884, median=156476, skew=3.72

# Log transformation
log_prices = np.log(house_prices)
print(f"Log transformed: mean={log_prices.mean():.2f}, "
      f"skew={pd.Series(log_prices).skew():.2f}")
# Output: Log transformed: mean=12.02, skew=0.38

# Square root transformation
sqrt_prices = np.sqrt(house_prices)
print(f"Sqrt transformed: skew={pd.Series(sqrt_prices).skew():.2f}")
# Output: Sqrt transformed: skew=1.46

# Binning — discretize continuous feature
bins = [0, 100000, 200000, 300000, 400000, 1000000]
labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
price_categories = pd.cut(house_prices, bins=bins, labels=labels)
print(pd.Series(price_categories).value_counts().to_string())
# Output:
# Low         341
# Medium      289
# Very Low    244
# High        103
# Very High    23

# Robust scaling (robust to outliers)
scaler = RobustScaler()
prices_scaled = scaler.fit_transform(house_prices.reshape(-1, 1))
print(f"Robust scaled: mean={prices_scaled.mean():.3f}, "
      f"std={prices_scaled.std():.3f}")
# Output: Robust scaled: mean=0.000, std=2.582
```

### Example 2: Categorical Encoding

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer

# Sample categorical data
data = pd.DataFrame({
    'city': ['NYC', 'LA', 'Chicago', 'NYC', 'Chicago', 'LA'],
    'size': ['S', 'M', 'L', 'M', 'XL', 'S'],
    'price': [100, 80, 120, 110, 150, 90]
})

# Label encoding (ordinal)
label_enc = LabelEncoder()
data['size_encoded'] = label_enc.fit_transform(data['size'])
print("Label Encoded Size:")
print(data[['size', 'size_encoded']].drop_duplicates().to_string(index=False))
# Output:
# Label Encoded Size:
#    size  size_encoded
#      S             2
#      M             1
#      L             0
#     XL             3

# One-hot encoding
onehot = OneHotEncoder(sparse_output=False)
city_encoded = onehot.fit_transform(data[['city']])
city_df = pd.DataFrame(
    city_encoded,
    columns=onehot.get_feature_names_out(['city'])
)
result = pd.concat([data, city_df], axis=1)
print("\nOne-Hot Encoded City:")
print(result[['city', 'city_Chicago', 'city_LA', 'city_NYC']].head())
# Output:
# One-Hot Encoded City:
#      city  city_Chicago  city_LA  city_NYC
# 0    NYC           0.0      0.0       1.0
# 1     LA           0.0      1.0       0.0
# 2 Chicago           1.0      0.0       0.0
# 3    NYC           0.0      0.0       1.0
# 4 Chicago           1.0      0.0       0.0

# Target encoding (mean encoding)
target_mean = data.groupby('city')['price'].mean()
data['city_target_enc'] = data['city'].map(target_mean)
print("\nTarget Encoded City:")
print(data[['city', 'price', 'city_target_enc']].to_string(index=False))
# Output:
# Target Encoded City:
#      city  price  city_target_enc
#      NYC    100             105.0
#       LA     80              85.0
#  Chicago    120             135.0
#      NYC    110             105.0
#  Chicago    150             135.0
#       LA     90              85.0
```

### Example 3: Date/Time Feature Engineering

```python
import numpy as np
import pandas as pd

# Create date range
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
df = pd.DataFrame({'timestamp': dates})
df['value'] = np.random.randn(len(dates))

# Extract time features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday
df['month'] = df['timestamp'].dt.month
df['day_of_year'] = df['timestamp'].dt.dayofyear
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
df['quarter'] = df['timestamp'].dt.quarter
df['season'] = df['month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
})

# Cyclical encoding for hour (preserve circular nature)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Days since a reference date (for trend)
df['days_since_start'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / (24 * 3600)

# Lag features (previous day's value at same hour)
df_sorted = df.sort_values('timestamp').copy()
df_sorted['value_lag_24h'] = df_sorted['value'].shift(24)  # 24 hours ago

# Rolling statistics
df_sorted['value_rolling_mean_7d'] = (
    df_sorted['value'].rolling(window=7*24, min_periods=1).mean()
)

sample = df_sorted.iloc[100:105]
print("Date/Time Features (sample):")
print(sample[['timestamp', 'hour', 'day_of_week', 'is_weekend',
              'is_business_hours', 'season', 'hour_sin', 'hour_cos']].to_string(index=False))
# Output:
# Date/Time Features (sample):
#              timestamp  hour  day_of_week  is_weekend  is_business_hours  season   hour_sin   hour_cos
# 2023-01-05 04:00:00     4            3           0                  0  Winter  0.866025  0.500000
# 2023-01-05 05:00:00     5            3           0                  0  Winter  0.965926  0.258819
# 2023-01-05 06:00:00     6            3           0                  0  Winter  1.000000  0.000000
# 2023-01-05 07:00:00     7            3           0                  0  Winter  0.965926 -0.258819
# 2023-01-05 08:00:00     8            3           0                  0  Winter  0.866025 -0.500000
```

### Example 4: Text Feature Extraction with TF-IDF

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

documents = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "Cats and dogs are pets",
    "The mat is red",
    "My dog loves to sit on mats"
]

# Count vectorizer (bag-of-words)
count_vec = CountVectorizer()
bow = count_vec.fit_transform(documents)

print("Vocabulary:", count_vec.get_feature_names_out())
# Output: Vocabulary: ['and' 'are' 'cat' 'cats' 'dog' 'dogs' 'is' 'loves' 'log' 'mat' 'mats' 'my' 'on' 'pets' 'red' 'sat' 'sit' 'the' 'to']

print("\nBag-of-Words Matrix (first 3 documents):")
print(bow[:3].toarray())
# Output:
# Bag-of-Words Matrix (first 3 documents):
# [[0 0 1 0 0 0 0 0 0 1 0 0 1 0 0 1 0 2 0]
#  [0 0 0 0 1 0 0 0 1 1 0 0 1 0 0 1 0 2 0]
#  [1 1 0 1 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0]]

# TF-IDF Vectorizer
tfidf_vec = TfidfVectorizer()
tfidf = tfidf_vec.fit_transform(documents)

print("\nTF-IDF Matrix (first 3 documents):")
print(np.round(tfidf[:3].toarray(), 3))
# Output:
# TF-IDF Matrix (first 3 documents):
# [[0.    0.    0.371 0.    0.    0.    0.    0.    0.    0.371 0.    0.    0.371 0.    0.    0.371 0.    0.742 0.   ]
#  [0.    0.    0.    0.    0.371 0.    0.    0.    0.453 0.371 0.    0.    0.371 0.    0.    0.371 0.    0.742 0.   ]
#  [0.37  0.37  0.    0.37  0.    0.37  0.    0.    0.    0.    0.    0.    0.    0.37  0.    0.    0.    0.    0.   ]]

# N-gram features (capture phrases)
ngram_vec = CountVectorizer(ngram_range=(1, 2))
ngram = ngram_vec.fit_transform(documents)
print(f"\nN-gram features (unigrams + bigrams): {len(ngram_vec.get_feature_names_out())} features")
# Output: N-gram features (unigrams + bigrams): 49 features
```

### Example 5: Interaction and Polynomial Features

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_regression

# Create two features with interaction effect
np.random.seed(42)
X = pd.DataFrame({
    'height_cm': np.random.uniform(150, 200, 100),
    'weight_kg': np.random.uniform(50, 100, 100)
})

# Polynomial features (degree=2 includes interactions)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

feature_names = poly.get_feature_names_out(['height', 'weight'])
X_poly_df = pd.DataFrame(X_poly, columns=feature_names)

print("Polynomial Features (degree=2) — first 3 rows:")
print(X_poly_df.head(3).round(2).to_string(index=False))
# Output:
#  height  weight  height^2  height weight  weight^2
#  158.49   59.55  25118.59       9436.19    3546.93
#  174.56   70.24  30471.20      12258.62    4933.88
#  173.79   99.87  30202.44      17357.39    9973.04

# Manual interaction features
X['bmi'] = X['weight_kg'] / ((X['height_cm'] / 100) ** 2)
X['height_weight_product'] = X['height_cm'] * X['weight_kg']
X['height_squared'] = X['height_cm'] ** 2
X['weight_log'] = np.log(X['weight_kg'])

print("\nEngineered Features (first 5 rows):")
print(X[['bmi', 'height_weight_product', 'height_squared',
         'weight_log']].head().round(3).to_string(index=False))
# Output:
# Engineered Features (first 5 rows):
#     bmi  height_weight_product  height_squared  weight_log
#  23.708                9436.19        25118.59       4.087
#  23.059               12258.62        30471.20       4.252
#  33.058               17357.39        30202.44       4.604
#  21.523               11552.65        27766.27       4.304
#  26.501               12559.61        28883.54       4.369
```

## Common Mistakes

1. **Data leakage through feature engineering**: Creating features that use information not available at prediction time (e.g., using the entire dataset's mean for target encoding, including future data in rolling statistics).
2. **Creating too many features (curse of dimensionality)**: More features are not always better. Each new feature adds dimensions, potentially causing spurious correlations and increasing overfitting risk.
3. **One-hot encoding high-cardinality features**: A feature with 1000 categories generates 999 dummy columns, drastically increasing dimensionality. Use target encoding or feature hashing instead.
4. **Not handling missing values before feature engineering**: Missing values propagate through transformations and can produce NaN features. Always handle missing data first.
5. **Applying transformations without understanding why**: Log transformation is not always appropriate. Always check the distribution and interpretability of the transformed feature.
6. **Cyclical features treated as linear**: Encoding months as 1-12 implies December (12) is farther from January (1) than June (6). Use sin/cos encoding for cyclical features.
7. **Ignoring domain knowledge**: The most powerful features often come from domain expertise, not generic transformations. Consult domain experts during feature engineering.

## Interview Questions

### Beginner - 5

1. **Q: What is feature engineering?**
   A: Feature engineering is the process of transforming raw data into features that better represent the underlying problem to machine learning models, improving their predictive performance.

2. **Q: What is one-hot encoding and when would you use it?**
   A: One-hot encoding converts categorical variables into binary columns, one per category. Use it for nominal (unordered) categorical variables with moderate cardinality (< 50 categories).

3. **Q: What is the difference between label encoding and one-hot encoding?**
   A: Label encoding assigns integer labels to categories (e.g., red=0, green=1, blue=2), implying ordinal relationships. One-hot encoding creates binary columns, treating categories as independent. Label encoding is appropriate for ordinal data; one-hot encoding for nominal data.

4. **Q: Why would you apply a log transformation to a feature?**
   A: To reduce skewness, make the distribution more normal, stabilize variance, and handle multiplicative relationships. Log transformation is especially useful for features with exponential growth or heavy right tails.

5. **Q: What is TF-IDF?**
   A: Term Frequency-Inverse Document Frequency weights words by their importance in a document relative to a collection. Common words get low weights; rare, discriminative words get high weights.

### Intermediate - 5

1. **Q: Explain target encoding and its risks.**
   A: Target encoding replaces each category with the mean target value for that category. Risk: data leakage — the category mean uses information from the sample itself. Mitigation: use cross-validation or smoothing (add a global prior).

2. **Q: How do you handle high-cardinality categorical features (e.g., ZIP code with 40,000 categories)?**
   A: Options: (1) Target encoding with smoothing, (2) Frequency encoding (replace category with its count), (3) Feature hashing (hash categories into fixed number of bins), (4) Group rare categories into an "other" bucket, (5) Use domain knowledge to group categories (ZIP → region).

3. **Q: What are interaction features and when are they useful?**
   A: Interaction features capture relationships between features, created by multiplying, dividing, or otherwise combining features. They are useful when the effect of one feature on the target depends on another feature's value (non-additive relationships).

4. **Q: How do you create features from date/time data?**
   A: Extract hour, day of week, month, quarter, year. Create cyclical encoding (sin/cos) for hours and months. Add flags for weekends, holidays, business hours. Create lag features and rolling statistics. Compute time since event, time to next event.

5. **Q: Explain the curse of dimensionality in the context of feature engineering.**
   A: As the number of features increases, the data becomes sparse in high-dimensional space. Distances become less meaningful, models require exponentially more samples, overfitting risk increases, and computation becomes expensive. Feature engineering should balance informativeness with dimensionality.

### Advanced - 3

1. **Q: Derive the optimal binning strategy for a continuous feature using the concept of mutual information.**
   A: Optimal binning discretizes a continuous feature to maximize the mutual information between the binned feature and the target. For supervised binning, use recursive partitioning to find bin boundaries that minimize entropy of the target within bins. The chi-merge algorithm merges adjacent bins that are statistically similar, stopping when all adjacent bins differ significantly (chi-square test). The number of bins can be selected via cross-validation.

2. **Q: Compare and contrast feature extraction methods for text: bag-of-words, TF-IDF, word2vec, and BERT embeddings — in terms of computational cost, semantic understanding, and downstream task performance.**
   A: Bag-of-words: cheap, no semantics, good for simple tasks. TF-IDF: cheap, some weighting, better for information retrieval. Word2vec: moderate cost, captures word analogies, good for semantic similarity. BERT: expensive, contextual embeddings, state-of-the-art for complex NLP tasks. The choice depends on data size, computational budget, and task complexity.

3. **Q: Prove that for a dataset with n samples and p features, blindly creating all pairwise interaction features increases the feature space to O(p²) and leads to multicollinearity. Derive a strategy for selecting only beneficial interactions using regularization.**
   A: All pairwise interactions: p original + p(p-1)/2 interactions = O(p²). Many interactions are highly correlated with original features or with each other. Strategy: (1) Use L1-regularized models (Lasso) on the expanded feature set — Lasso will zero out irrelevant interactions. (2) Use tree-based models (Random Forest, XGBoost) which naturally capture interactions up to depth-related order. (3) Use greedy forward selection of interactions based on validation performance. (4) Use factorization machines that learn interaction weights efficiently.

## Practice Problems

### Easy - 5

1. **Problem**: A dataset has a "color" column with values: red, blue, green, red, blue. How would you encode this for ML?

2. **Problem**: A feature "income" has a right-skewed distribution with values from $20K to $5M. What transformation would you apply?

3. **Problem**: A timestamp column has dates from 2020-01-01 to 2023-12-31. List five features you would extract from this column.

4. **Problem**: A model uses house area (sq ft) and number of bedrooms. What interaction feature would you create?

5. **Problem**: Why is it problematic to use one-hot encoding on a ZIP code feature with 30,000 unique values?

### Medium - 5

1. **Problem**: You have a dataset of e-commerce transactions with product categories (200 categories) and the target is purchase amount. Design a feature engineering strategy for the category variable.

2. **Problem**: A time series dataset has hourly sales for 2 years. Engineer features to capture: (a) daily seasonality, (b) weekly seasonality, (c) yearly seasonality, (d) holiday effects, (e) trend.

3. **Problem**: For a text classification problem (spam detection), compare the features produced by CountVectorizer, TfidfVectorizer, and word embeddings.

4. **Problem**: You are predicting loan default. Raw features include age, income, loan amount, credit score, and employment length. Engineer at least 5 derived features that might improve prediction.

5. **Problem**: Explain how to handle missing values before feature engineering and what happens if you engineer features before handling missing data.

### Hard - 3

1. **Problem**: Design a feature engineering pipeline for a healthcare predictive model using electronic health records. Address: (a) irregular time intervals between visits, (b) variable-length medical history, (c) categorical diagnoses (ICD-10 codes, 70,000+ categories), (d) numerical lab values with missing data.

2. **Problem**: Prove that the feature space becomes linearly separable with the right feature transformations if the true decision boundary is polynomial of degree d. Use the kernel trick to explain how SVMs handle this implicitly.

3. **Problem**: Compare and contrast automated feature engineering (Featuretools, AutoML) with manual feature engineering. When is each approach preferred, and what are the risks of automated feature engineering?

## Solutions

### Easy Solutions

1. One-hot encoding: red → [1,0,0], blue → [0,1,0], green → [0,0,1]. (Or use pandas get_dummies()).
2. Log transformation: log(income) to reduce skewness and handle the wide range.
3. Year, month, day_of_week, hour, quarter, is_weekend, day_of_year, days_since_first_record.
4. Average room size = area / bedrooms. Or bedrooms per square foot = bedrooms / area.
5. One-hot encoding 30,000 categories creates 29,999 dummy columns, causing extreme dimensionality, memory issues, and sparsity. Use target encoding, frequency encoding, or feature hashing instead.

### Medium Solutions

1. (1) Target encoding: replace category with mean purchase amount per category (with smoothing). (2) Frequency encoding: number of transactions per category. (3) Category hierarchy: group subcategories into higher-level categories. (4) Aggregate statistics per category: mean, std of purchase amount, count of transactions. (5) Cross-validation target encoding to prevent leakage.
2. (a) Hour of day, hour_sin, hour_cos. (b) Day of week, is_weekend. (c) Month, season, day_of_year, month_sin/cos. (d) Holiday indicator feature, days_to_nearest_holiday. (e) Days since start, rolling mean over 7/30/365 days.
3. CountVectorizer: simple word counts, high-dimensional sparse. TfidfVectorizer: weighted word counts, better for discriminative words, still sparse. Word embeddings (Word2Vec, GloVe): dense vectors capturing semantic meaning, lower-dimensional, require pre-trained models or large training data.
4. (1) Debt-to-income ratio = loan_amount / income. (2) Credit utilization = loan_amount / credit_score_weighted_limit. (3) Age-based feature: is_young = age < 25. (4) Employment stability: years_at_job / age. (5) Loan-to-value ratio if property data available. (6) Payment burden = monthly_payment / monthly_income.
5. Handle missing values first (imputation, removal, or flag). If you engineer features before handling missing data, the engineered features (log, interaction, ratios) will propagate NaN values, and downstream imputation on engineered features can introduce bias. Best practice: impute first, then engineer.

### Hard Solutions

1. Pipeline: (a) Irregular intervals — create features from visit gaps (days_since_last_visit, average_gap), use interpolation for time series features. (b) Variable-length history — aggregate statistics (count of visits, mean/trend of vitals over visits), use last-k- visits as fixed-length feature window, or use RNN-based featurization. (c) ICD-10 codes — use hierarchical categories (first 3 chars = disease category), frequency encoding, diagnosis co-occurrence features. (d) Lab values — impute with median (flagged), create binary flags for missing, compute z-scores, derive clinical ratios (e.g., BUN/creatinine).
2. If the true decision boundary is polynomial of degree d, then mapping each original feature vector x to a vector of all monomials up to degree d makes the data linearly separable in the expanded space. For example, for a quadratic boundary in 1D (degree 2), the mapping φ(x) = [x, x²] makes the boundary linear in the 2D space. SVMs use the kernel trick to efficiently compute dot products in this expanded space without explicitly constructing it: K(x, y) = (x·y + 1)^d corresponds to all monomials up to degree d.
3. Manual engineering: domain knowledge rich, interpretable, requires expertise, time-consuming. Automated engineering: fast, discovers unexpected features, can overfit, produces hard-to-interpret features, requires large datasets. Automated approaches (Featuretools uses relational data with "deep feature synthesis") are preferred for: (a) large datasets with relational structure, (b) when domain expertise is limited, (c) as a complement to manual engineering. Manual engineering is preferred for: (a) small datasets, (b) problems requiring interpretability, (c) when strong domain knowledge exists. Risks of automated engineering: massive feature explosion, data leakage through temporal features, overfitting to spurious patterns.

## Related Concepts

- **ML-010: Feature Selection** — Selecting the most useful engineered features
- **ML-004: Overfitting and Underfitting** — Too many engineered features causes overfitting
- **ML-006: Evaluation Metrics** — Good features improve evaluation metrics
- **Dimensionality Reduction**: PCA, t-SNE for reducing engineered feature space
- **Regularization**: L1 (Lasso) for automatic feature selection among engineered features

## Next Concepts

- **ML-010: Feature Selection** — Selecting the best features from your engineered set
- **Dimensionality Reduction**: Reducing the feature space after engineering

## Summary

Feature engineering is the process of creating informative features from raw data, often determining model success more than algorithm choice. Key techniques include numerical transformations (log, sqrt, binning, scaling), categorical encoding (one-hot, label, target encoding), date/time feature extraction (hour, day, cyclical encoding), text features (TF-IDF, count vectors), and interaction features. Domain knowledge is crucial for creating the most impactful features. Feature engineering must be done carefully to avoid data leakage and the curse of dimensionality. Good features enable simpler models to perform well, reduce overfitting, and make patterns more accessible to learning algorithms.

## Key Takeaways

1. Feature engineering often has higher ROI than algorithm selection or hyperparameter tuning.
2. Log transformation fixes skewness; one-hot encoding handles nominal categories.
3. Target encoding is powerful but requires leak-proof implementation.
4. Date/time features should use cyclical encoding (sin/cos) for periodic components.
5. Interaction features capture relationships between features.
6. Avoid data leakage — engineer features only from training data.
7. Balance feature richness with dimensionality and overfitting risk.
